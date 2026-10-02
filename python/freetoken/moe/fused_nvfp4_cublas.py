"""Tensor-core NVFP4 MoE prefill for SM75 (Phase E, expert branch).

The production Triton ``_prefill_nvfp4_moe_kernel`` dequantizes NVFP4 inline in the
GEMM K-loop, but on SM75 Triton emits **zero** tensor-core MMA (the Phase E2 sweep
proved this is a hard codegen limitation, not a tuning gap), so it runs at ~0.76
TFLOPS on CUDA-core FMA -- 136 ms/layer at M=2048.

This module replaces the grouped NVFP4 GEMM with a tensor-core path:

  1. A **fused single-launch** Triton dequant (all experts in a group, one launch)
     -> FP16 weights, with the per-row **global scale folded in** so the GEMM reads
     plain FP16 weights and never materializes an fp32 temporary. (A Python loop over
     256 experts took 150 ms; one launch takes ~15 ms -- the difference that makes the
     whole path viable.)
  2. **cuBLAS FP16** ``torch.bmm`` (tensor cores) for gemm1 and gemm2.
  3. A **vectorized, sync-free** scatter (``argsort`` + ``index_add_``) for the
     grouped-MoE routing.

The work is **grouped** to bound VRAM: the full-layer FP16 expansion (1.6 GiB) does
not fit on the 8 GB card (0.84 GiB free after load), and the dense E1 temps + base
activations consume most of that headroom, so a group's ``gate_up`` dequant (gu =
group*4 MiB) must stay small -> **group=8** (gu=32 MiB). Only ``group`` experts' FP16
weights are live at once, ``gate_up`` is freed before ``down`` is allocated, and the
activation buffers are allocated **per group** (not once for all 256 experts), so the
peak E3 transient is ~gu + one bmm output (<= ~96 MiB at M=2048, group=8).

Numerics match the production kernel to fp16 rounding (rel-L2 ~8e-4, cosine 1.0) --
a faithful drop-in, not a re-quantization.

Gate (see :func:`fused_experts_nvfp4` in ``fused_nvfp4.py``):
``FREETOKEN_E3_CUBLAS=1`` + SM75-or-lower + prefill + a plain ``*_and_mul``
activation + router weight applied on the gemm2 output (the model default).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.layers import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@triton.jit
def _dequant_nvfp4_group_kernel(
    packed_ptr,        # [G, N, K // 2] uint8 (a group slice, contiguous)
    scale_ptr,         # [G, N, K // 16] uint8 (e4m3 bits)
    global_ptr,        # [G, N] fp16 (per-row global scale)
    out_ptr,           # [G, N, K] fp16
    lut_ptr,           # [16] fp32 e2m1 values
    N, K,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
):
    """Dequantize one group of NVFP4 experts to FP16 in a single launch.

    ``e = program_id(0)`` is the *local* expert index within the group slice; the
    caller passes ``packed[e0:e1]`` so the base pointer is already offset. Each
    program handles a [BLOCK_KB bytes, BLOCK_N] tile: 2*BLOCK_KB k-values x BLOCK_N
    output rows. The e4m3 block scale is decoded inline (the uint16-bitcast form of
    ``e4m3_u8_to_f32``) and the per-row **global scale is folded in** (weight * global)
    before the FP16 store, so the GEMM reads plain FP16 weights and no fp32 temporary
    is ever materialized.
    """
    e = tl.program_id(0)
    nb = tl.program_id(1)
    kb = tl.program_id(2)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = kb * BLOCK_KB + tl.arange(0, BLOCK_KB)
    K_BYTES = K // 2
    bm = offs_kb < K_BYTES
    nm = offs_n < N
    p = packed_ptr + e * N * K_BYTES + offs_n[None, :] * K_BYTES + offs_kb[:, None]
    b = tl.load(p, mask=bm[:, None] * nm[None, :], other=0).to(tl.int32)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    s = scale_ptr + e * N * (K // 16) + offs_n[None, :] * (K // 16) + (offs_kb // 8)[:, None]
    sraw = tl.load(s, mask=bm[:, None] * nm[None, :], other=0)
    h = ((sraw & 0x80).to(tl.uint16) << 8) | ((sraw & 0x7F).to(tl.uint16) << 7)
    scale = h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    g = tl.load(global_ptr + e * N + offs_n, mask=nm, other=0).to(tl.float32)
    blo = tl.load(lut_ptr + lo) * scale * g[None, :]
    bhi = tl.load(lut_ptr + hi) * scale * g[None, :]
    ob = out_ptr + e * N * K + offs_n[None, :] * K
    tl.store(ob + (2 * offs_kb)[:, None], blo.to(tl.float16), mask=bm[:, None] * nm[None, :])
    tl.store(ob + (2 * offs_kb + 1)[:, None], bhi.to(tl.float16), mask=bm[:, None] * nm[None, :])


def _e2m1_lut(device_index: int) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, dtype=torch.float32,
                        device=torch.device("cuda", device_index))


def _dequant_group(packed, scale, global_scale, lut, N, K, e0, e1):
    """Dequantize experts [e0, e1) to a contiguous FP16 [G, N, K] tensor with the
    per-row global scale folded in (weight * global)."""
    G = e1 - e0
    out = torch.empty(G, N, K, dtype=torch.float16, device=packed.device)
    _dequant_nvfp4_group_kernel[(G, triton.cdiv(N, 64), triton.cdiv(K // 2, 64))](
        packed[e0:e1], scale[e0:e1], global_scale[e0:e1], out, lut, N, K,
        BLOCK_N=64, BLOCK_KB=64, num_warps=4)
    return out


def _build_scatter(topk_ids, num_experts, top_k):
    """Vectorized, sync-free grouped-MoE routing structure (computed once/layer).

    Returns, indexed by sorted position p (routes grouped by expert):
      sorted_expert[p] : expert of the p-th sorted route
      token_sorted[p]  : its token id
      rank_sorted[p]   : its offset within its expert's block (0-based)
    plus ``order`` (sorted -> original route) and ``expert`` (per original route).
    """
    R = topk_ids.numel()
    dev = topk_ids.device
    expert = topk_ids.flatten()
    order = torch.argsort(expert, stable=True)
    token = torch.arange(R, device=dev) // top_k
    sorted_expert = expert[order]
    counts = torch.bincount(expert, minlength=num_experts)
    start_pos = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=dev),
        torch.cumsum(counts, 0)[:-1],
    ])
    start_at_p = start_pos[sorted_expert]
    rank_sorted = torch.arange(R, device=dev) - start_at_p
    token_sorted = token[order]
    return expert, order, token_sorted, sorted_expert, rank_sorted


def fused_experts_nvfp4_cublas(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    group: int = 8,
) -> torch.Tensor:
    """Tensor-core NVFP4 MoE prefill (SM75). Drop-in for ``fused_experts_nvfp4``.

    ``gate_up_packed`` is [E, 2I, H//2] uint8, ``down_packed`` is [E, H, I//2] uint8;
    scales are the e4m3-bit tensors (viewed as uint8 on SM75); globals are [E, N] fp16.
    ``topk_ids`` index expert rows in [0, num_experts). The router weight is applied
    to the gemm2 output (the model default; ``apply_router_weight_on_input`` must be
    False for the caller to route here). The per-row global scale is folded into the
    dequantized weights (the production kernel applies it to the gemm output -- the
    same value, no fp32 temporary).
    """
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    lut = _e2m1_lut(dev.index)
    act = _ACT[activation]

    # e4m3 scales as the dequant kernel expects (uint8 view on SM75)
    gu_scale = e4m3_kernel_view(gate_up_scale)
    dn_scale = e4m3_kernel_view(down_scale)

    expert, order, token_sorted, sorted_expert, rank_sorted = _build_scatter(
        topk_ids, num_experts, top_k)
    gw_sorted = topk_weights.reshape(-1)[order]
    counts = torch.bincount(expert, minlength=num_experts)
    start_pos = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=dev),
        torch.cumsum(counts, 0),
    ])  # [E+1]; start_pos[e] = first sorted position of expert e

    # Precompute each group's route indices once (sorted positions are contiguous
    # per expert). Doing this up front keeps the timed dequant/bmm loop sync-light.
    groups = []
    for e0 in range(0, num_experts, group):
        e1 = min(e0 + group, num_experts)
        p0 = int(start_pos[e0])
        p1 = int(start_pos[e1])
        if p0 == p1:
            groups.append((e0, e1, None, None))
            continue
        sel = torch.arange(p0, p1, device=dev)
        rows = sorted_expert[sel] - e0
        cols = rank_sorted[sel]
        Mp_g = int(counts[e0:e1].max().item())
        groups.append((e0, e1, (sel, rows, cols), Mp_g))

    out = torch.zeros(M, H, dtype=dt, device=dev)
    for e0, e1, gidx, Mp_g in groups:
        if gidx is None:
            continue
        sel, rows, cols = gidx
        G = e1 - e0
        # Per-group activation gather + scatter to a [G, Mp_g, H] padded buffer
        # (allocated per group, not once for all E experts -> ~6 MiB, not ~90 MiB).
        A_in_sel = hidden_states[token_sorted[sel]]           # [n_sel, H]
        A_pad = torch.zeros(G, Mp_g, H, dtype=dt, device=dev)
        A_pad.view(-1, H)[rows * Mp_g + cols] = A_in_sel
        del A_in_sel
        # gemm1 on gate_up (global scale folded into the weights), FREE gu + A_pad
        # before allocating down. Peak = gu, never gu + dn.
        gu = _dequant_group(gate_up_packed, gu_scale, gate_up_global, lut, two_i, H, e0, e1)
        ic1 = torch.bmm(A_pad, gu.transpose(1, 2))            # [G, Mp_g, 2I] fp16
        del gu, A_pad
        ic2 = act(ic1.reshape(-1, two_i)).reshape(G, Mp_g, inter)
        del ic1
        dn = _dequant_group(down_packed, dn_scale, down_global, lut, H, inter, e0, e1)
        ic3 = torch.bmm(ic2, dn.transpose(1, 2))              # [G, Mp_g, H] fp16
        del dn
        del ic2
        vals = ic3[rows, cols] * gw_sorted[sel][:, None]
        del ic3
        out.index_add_(0, token_sorted[sel], vals.half())
    return out


__all__ = ["fused_experts_nvfp4_cublas"]
