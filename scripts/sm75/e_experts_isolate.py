"""Phase E (NVFP4 expert branch) — isolation benchmark.

Measures, for ONE MoE layer at prefill M=2048 / top_k=8 / E=256 / H=2048 / I=512:
  1. the current Triton inline-dequant grouped GEMM (fused_experts_nvfp4),
  2. a bounded per-expert NVFP4->FP16 dequant + cuBLAS FP16 candidate,
  3. correctness of both vs a float32 reference.

The candidate dequantizes ONE expert's weights to FP16 at a time (~6 MB transient,
fits the 0.84 GiB free VRAM), runs the two cuBLAS GEMMs for that expert's tokens,
and discards the FP16 weights before the next expert. This is the plan's
"second choice" (bounded dequant + FP16 GEMM), applied per-expert because the
full-layer FP16 expansion (1.6 GiB) does not fit on the 8 GB card.

Run:  .venv/bin/python scripts/sm75/e_experts_isolate.py
"""
from __future__ import annotations

import time

import torch

torch.manual_seed(0)
dev = torch.device("cuda")

# ---- model shapes (Qwen3.6-35B-A3B-NVFP4) ---------------------------------
H = 2048          # hidden
I = 512           # moe_intermediate_size
E = 256           # num_experts
TOP_K = 8         # num_experts_per_tok
M = 2048          # prefill tokens
DT = torch.float16

_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32, device=dev,
)


def _e4m3_safe_bytes(shape):
    """Valid e4m3 bytes in a small positive-normal range (~1.0 .. ~1.375), avoiding
    NaN codes (0x7F/0xFF) and denormals. 0x38 (=1.0) .. 0x3B (=1.375).
    Kept small so fp16 GEMM intermediates (summed over K=2048) stay well under
    the fp16 max (65504) -- realistic for a trained model's block scales."""
    lo, hi = 0x38, 0x3B
    return torch.randint(lo, hi + 1, shape, dtype=torch.uint8, device=dev)


def make_nvfp4_banks():
    """Fabricate realistic native NVFP4 banks for one layer (valid e4m3 scales)."""
    # gate_up: N=2I out, K=H
    gu_packed = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    gu_scale = _e4m3_safe_bytes((E, 2 * I, H // 16))
    gu_global = (torch.rand(E, 2 * I, device=dev) * 0.5 + 0.5).half()
    # down: N=H out, K=I
    dn_packed = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    dn_scale = _e4m3_safe_bytes((E, H, I // 16))
    dn_global = (torch.rand(E, H, device=dev) * 0.5 + 0.5).half()
    return gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global


def _decode_e4m3_u8(v: torch.Tensor) -> torch.Tensor:
    """Decode e4m3 bytes (uint8) -> fp32, exact bit trick (matches e4m3_compat)."""
    vi = v.to(torch.int32)
    h = ((vi & 0x80) << 8) | ((vi & 0x7F) << 7)
    return h.to(torch.int16).view(torch.float16).float() * 256.0


def _dequant_fp16(packed: torch.Tensor, scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Dequant one expert's [N, K] weight to FP16 (global NOT folded in)."""
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    w = _LUT[codes]  # [N, K] fp32 (E2M1)
    s = _decode_e4m3_u8(scale)  # [N, K//16] fp32
    s = s.repeat_interleave(16, dim=1)  # [N, K]
    return (w * s).half()  # [N, K] fp16


def _dequant_fp32(packed: torch.Tensor, scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Dequant one expert's [N, K] weight to FP32 (global NOT folded in)."""
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    w = _LUT[codes]
    s = _decode_e4m3_u8(scale).repeat_interleave(16, dim=1)
    return w * s  # [N, K] fp32


def _route_indices():
    routes = torch.arange(M * TOP_K, device=dev)
    tok = routes // TOP_K
    rk = routes % TOP_K
    return tok, rk


def ref_expert_out(hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
                   topk_ids, topk_weights):
    """Float32 reference (ground truth): dequant fp32, GEMM fp32, silu, sum-reduce."""
    tok, rk = _route_indices()
    expert = topk_ids[tok, rk]
    act = hidden[tok].float()            # [M*topk, H]
    gw = topk_weights[tok, rk].float()   # [M*topk]
    ic3 = torch.zeros(M * TOP_K, H, dtype=torch.float32, device=dev)
    for e in range(E):
        sel = (expert == e).nonzero().squeeze(-1)
        if sel.numel() == 0:
            continue
        a_e = act[sel]
        gu_e = _dequant_fp32(gu_packed[e], gu_scale[e], 2 * I, H) * gu_global[e].float()[:, None]
        ic1 = a_e @ gu_e.T               # [m_e, 2I]
        g, u = ic1[:, :I], ic1[:, I:]
        ic2 = torch.nn.functional.silu(g) * u
        dn_e = _dequant_fp32(dn_packed[e], dn_scale[e], H, I) * dn_global[e].float()[:, None]
        ic3_e = ic2 @ dn_e.T             # [m_e, H]
        ic3[sel] = ic3_e * gw[sel, None]
    out = torch.zeros(M, H, dtype=torch.float32, device=dev)
    out.index_add_(0, tok, ic3)
    return out


def candidate_expert_out(hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
                         topk_ids, topk_weights):
    """Bounded per-expert NVFP4->FP16 dequant + cuBLAS FP16 candidate.

    Matches the current kernel's numerics: global applied to gemm1 output BEFORE
    activation (fp32), routed weight applied after the down GEMM (fp32).
    """
    from freetoken.kernel.triton.activation import silu_and_mul

    tok, rk = _route_indices()
    expert = topk_ids[tok, rk]
    act = hidden[tok].contiguous()           # [M*topk, H] fp16
    gw = topk_weights[tok, rk].float()       # [M*topk] fp32
    ic3 = torch.empty(M * TOP_K, H, dtype=DT, device=dev)
    for e in range(E):
        sel = (expert == e).nonzero().squeeze(-1)
        if sel.numel() == 0:
            continue
        a_e = act[sel].contiguous()          # [m_e, H]
        gu_e = _dequant_fp16(gu_packed[e], gu_scale[e], 2 * I, H)   # [2I, H] fp16
        ic1 = a_e @ gu_e.T                   # [m_e, 2I] fp16 (cuBLAS tensor core)
        ic1 = ic1.float() * gu_global[e].float()[None, :]  # global BEFORE act (fp32, per-n)
        ic2 = silu_and_mul(ic1.half())       # [m_e, I] fp16
        dn_e = _dequant_fp16(dn_packed[e], dn_scale[e], H, I)       # [H, I] fp16
        ic3_e = ic2 @ dn_e.T                 # [m_e, H] fp16
        ic3_e = ic3_e.float() * dn_global[e].float()[None, :] * gw[sel, None]
        ic3.index_copy_(0, sel, ic3_e.half())
    out = torch.zeros(M, H, dtype=DT, device=dev)
    out.index_add_(0, tok, ic3)
    return out


def _dequant_group_fp16(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1):
    """Dequant a group of experts [e0, e1) to FP16 (global NOT folded)."""
    gu = torch.empty(e1 - e0, 2 * I, H, dtype=DT, device=dev)
    dn = torch.empty(e1 - e0, H, I, dtype=DT, device=dev)
    for e in range(e0, e1):
        gu[e - e0] = _dequant_fp16(gu_packed[e], gu_scale[e], 2 * I, H)
        dn[e - e0] = _dequant_fp16(dn_packed[e], dn_scale[e], H, I)
    return gu, dn


def candidate_bmm_expert_out(hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale,
                             dn_global, topk_ids, topk_weights, group=32):
    """Grouped-dequant + torch.bmm (tensor-core) candidate.

    Dequant `group` experts to FP16, scatter their tokens into a per-expert batch,
    run ONE bmm over the group (gate_up GEMM), act, ONE bmm (down GEMM), write back.
    Bounded transient: `group` experts x 1.5 MiB ~= 48 MB (group=32). No 256-launch
    Python loop -- one bmm per group per GEMM.
    """
    from freetoken.kernel.triton.activation import silu_and_mul

    routes = torch.arange(M * TOP_K, device=dev)
    tok = routes // TOP_K
    rk = routes % TOP_K
    expert = topk_ids[tok, rk]  # [M*topk]
    act = hidden[tok].contiguous()  # [M*topk, H] fp16
    gw = topk_weights[tok, rk].float()  # [M*topk]

    # per-expert token lists (routes)
    expert_routes = [ (expert == e).nonzero().squeeze(-1) for e in range(E) ]
    ic3 = torch.empty(M * TOP_K, H, dtype=DT, device=dev)

    for e0 in range(0, E, group):
        e1 = min(e0 + group, E)
        gu, dn = _dequant_group_fp16(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1)
        # build a uniform-M batch for this group: M_g = max tokens among group experts
        cnts = [expert_routes[e].numel() for e in range(e0, e1)]
        M_g = max(cnts) if any(cnts) else 0
        if M_g == 0:
            continue
        # pad each expert's token list to M_g with route id = -1 (masked)
        batch_a = torch.zeros(M_g, H, dtype=DT, device=dev)
        batch_gw = torch.zeros(M_g, dtype=torch.float32, device=dev)
        batch_tok = torch.zeros(M_g, dtype=torch.int64, device=dev)  # route index
        batch_valid = torch.zeros(M_g, dtype=torch.bool, device=dev)
        # we need a per-expert batch; bmm needs [G, M_g, K]. Build [G, M_g, H].
        A = torch.zeros(e1 - e0, M_g, H, dtype=DT, device=dev)
        GWb = torch.zeros(e1 - e0, M_g, dtype=torch.float32, device=dev)
        for j, e in enumerate(range(e0, e1)):
            r = expert_routes[e]
            if r.numel() == 0:
                continue
            A[j, :r.numel()] = act[r]
            GWb[j, :r.numel()] = gw[r]
        # gate_up GEMM: [G, M_g, H] @ [G, H, 2I] -> [G, M_g, 2I]
        ic1 = torch.bmm(A, gu.transpose(1, 2))  # gu [G,2I,H] -> [G,H,2I]
        # global before act (fp32), per output-n (last dim)
        ic1 = ic1.float() * gu_global[e0:e1].float()[:, None, :]
        ic2 = silu_and_mul(ic1.half().reshape(-1, 2 * I)).reshape(e1 - e0, M_g, I)
        # down GEMM: [G, M_g, I] @ [G, I, H] -> [G, M_g, H]
        ic3_g = torch.bmm(ic2, dn.transpose(1, 2))  # dn [G,H,I] -> [G,I,H]
        ic3_g = ic3_g.float() * dn_global[e0:e1].float()[:, None, :]
        # write back per expert
        for j, e in enumerate(range(e0, e1)):
            r = expert_routes[e]
            if r.numel() == 0:
                continue
            ic3[r] = (ic3_g[j, :r.numel()] * GWb[j, :r.numel(), None]).half()
    out = torch.zeros(M, H, dtype=DT, device=dev)
    out.index_add_(0, tok, ic3)
    return out


def timeit(fn, warmup=3, reps=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm() / (b.norm() + 1e-12)


def main():
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    print(f"shapes: M={M} top_k={TOP_K} E={E} H={H} I={I}  dt={DT}")
    gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global = make_nvfp4_banks()
    hidden = (torch.randn(M, H, device=dev) * 0.5).half()
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=dev)
    topk_w = torch.softmax(torch.randn(M, TOP_K, device=dev), dim=-1)

    cnt = torch.bincount(topk_ids.flatten().long(), minlength=E)
    print(f"tokens/expert: mean={cnt.float().mean():.1f} min={cnt.min().item()} "
          f"max={cnt.max().item()} empty={int((cnt == 0).sum())}")

    print("\n[1] current Triton inline-dequant (fused_experts_nvfp4) ...")
    t_cur = timeit(lambda: fused_experts_nvfp4(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_w, topk_ids, E, "silu", False))
    out_cur = fused_experts_nvfp4(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_w, topk_ids, E, "silu", False)
    print(f"    {t_cur:.1f} ms/layer  ->  {40 * t_cur / 1000:.2f} s for 40 layers")

    print("\n[2] candidate per-expert dequant + cuBLAS FP16 (Python loop) ...")
    t_cand = timeit(lambda: candidate_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w))
    out_cand = candidate_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w)
    print(f"    {t_cand:.1f} ms/layer  ->  {40 * t_cand / 1000:.2f} s for 40 layers")

    print("\n[3] candidate grouped-dequant + torch.bmm (tensor core, group=32) ...")
    t_bmm = timeit(lambda: candidate_bmm_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w, group=32))
    out_bmm = candidate_bmm_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w, group=32)
    print(f"    {t_bmm:.1f} ms/layer  ->  {40 * t_bmm / 1000:.2f} s for 40 layers")

    print("\n[4] float32 reference (ground truth) ...")
    out_ref = ref_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w)

    print("\n=== correctness (rel-L2 vs fp32 ref) ===")
    print(f"  current   vs ref : {rel_l2(out_cur, out_ref):.6f}")
    print(f"  per-expert vs ref: {rel_l2(out_cand, out_ref):.6f}")
    print(f"  bmm       vs ref : {rel_l2(out_bmm, out_ref):.6f}")
    print(f"  bmm       vs cur : {rel_l2(out_bmm, out_cur):.6f}")

    print("\n=== summary ===")
    print(f"  current (triton) : {t_cur:8.1f} ms/layer")
    print(f"  per-expert loop  : {t_cand:8.1f} ms/layer")
    print(f"  bmm group=32     : {t_bmm:8.1f} ms/layer   ({t_cur / t_bmm:.1f}x vs current)" if t_bmm > 0 else "")
    print(f"  prefill-2048 (40 layers): current {40*t_cur/1000:.2f} s -> bmm {40*t_bmm/1000:.2f} s")
    print(f"  dequant transient per expert: ~6 MB (bounded, fits 0.84 GiB free)")


if __name__ == "__main__":
    main()
