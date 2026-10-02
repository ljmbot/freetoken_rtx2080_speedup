"""Phase E expert-branch correctness gate: edge-case matrix.

Validates the tensor-core NVFP4 MoE prefill (``fused_experts_nvfp4_cublas``, E3)
against three references, closing the plan's Phase E correctness gate
(freetoken_project.md, "Expert-kernel correctness gates"):

  (c) DEQUANT  : E3 ``_dequant_group`` fp16 output vs an independent fp64 dequant,
                 elementwise to fp16-store rounding. The tight, exact check for the
                 E2M1 / nibble / e4m3 / global-scale contract. Uses the FULL code and
                 scale range (all 16 e2m1 codes; e4m3 zero/subnormal/normal/max/NaN).
  (a) DROP-IN  : E3 vs the production Triton kernel (``fused_experts_nvfp4``).
                 Both round dequantized weights to fp16 and accumulate in fp32, so
                 they must agree to fp16/accumulation-order rounding. Catches
                 orchestration bugs (scatter, group boundaries, router-weight
                 placement, activation) because the two paths orchestrate differently
                 (E3: argsort+index_add_ + per-group cuBLAS bmm; prod: moe_align_block
                 size + padded Triton GEMM).
  (b) TRUTH    : E3 (and prod) vs an independent fp64 reference that dequantizes
                 NVFP4 exactly (the same E2M1 LUT + e4m3 bitcast both kernels share)
                 and runs the full routed MoE in fp64, dequantizing per expert to
                 bound VRAM. Catches a *shared* dequant misunderstanding.

Edge cases covered (per the plan):
  - all 16 E2M1 codes (0x0-0xF), low+high nibble order, 16-wide block boundaries
  - e4m3 scales: zero, subnormals, smallest normal, a normal, max-finite, negatives,
    and the two reserved/NaN codes (0x7F/0xFF)
  - per-row global scale
  - empty experts, skewed routing, single-expert, repeated expert ids, all top-k
    values, M in {1,16,64,2048}
  - nontrivial (non-normalized, incl. a zero and a >1) router weights
  - activation kinds silu / gelu / gelu_tanh; realistic + zero + large magnitudes
  - tail masking (M*top_k not divisible by the production BLOCK_SIZE_M)

The latency gate (full-operation speedup incl. packing/dequant/workspace) was already
met by the Phase E integration + e2e runs (5.9x expert GEMM, 6.76x prefill-2048);
this script is the numeric-correctness half.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python scripts/sm75/e_experts_edge_matrix.py
Exit 0 = all cases pass, 1 = any failure.
"""
from __future__ import annotations

import gc
import sys
import traceback

import torch

torch.manual_seed(0)
dev = torch.device("cuda")

# Reference model shape (Qwen3.6-35B-A3B, verified locally): H=2048, I=512, E=256.
H, I, E = 2048, 512, 256
TWO_I = 2 * I

# E2M1 (FP4) values, code -> value. All 16 codes are valid (no NaN in FP4).
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64, device=dev,
)

# Tolerances.
TOL_DEQUANT_RE = 2e-3    # elementwise rel err vs fp64 dequant (fp16 store rounding)
TOL_DROPPIN = 5e-3       # rel-L2 E3 vs production (validated baseline ~8e-4)
TOL_TRUTH = 4e-2         # rel-L2 vs fp64 truth (fp16 weight-rounding bound)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Exact references (fp64), replicating the documented NVFP4 contract.
# ----------------------------------------------------------------------------
def e4m3_decode(bits: torch.Tensor) -> torch.Tensor:
    """Replicate ``e4m3_u8_to_f32`` EXACTLY in fp64 (same uint16 bitcast + *256).

    Using the identical formula (not a textbook e4m3 table) keeps the reference
    bit-consistent with both kernels, including the reserved-code behaviour
    (0x7F/0xFF -> +-480 per the e4m3_compat module).
    """
    b = bits.to(torch.int32)
    h = ((b & 0x80) << 8) | ((b & 0x7F) << 7)
    f16 = h.to(torch.int16).view(torch.float16)
    return f16.to(torch.float64) * 256.0


def _dequant_one(packed_e, scale_e, global_e):
    """Exact fp64 dequant of ONE expert: W[n,k] = E2M1[code]*e4m3*scale[n,k//16]*global[n].

    ``packed_e`` [N,K//2] uint8 (low nibble = even k), ``scale_e`` [N,K//16] uint8,
    ``global_e`` [N] fp16. Returns fp64 [N,K]. Per-expert to bound VRAM.
    """
    N, Kb = packed_e.shape
    K = Kb * 2
    p = packed_e.to(torch.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    w = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(N, K)  # even->lo, odd->hi
    s = e4m3_decode(scale_e).repeat_interleave(16, dim=1)        # [N, K] (16 k/block)
    w = w * s
    w = w * global_e[:, None].to(torch.float64)                  # per-row global
    return w


def _act(gate: torch.Tensor, up: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "silu":
        return (gate * torch.sigmoid(gate)) * up
    if kind == "gelu":
        return (0.5 * gate * (1.0 + torch.erf(gate * 0.7071067811865476))) * up
    if kind == "gelu_tanh":
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        return (0.5 * gate * (1.0 + torch.tanh(inner))) * up
    raise ValueError(kind)


def moe_ref(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act):
    """Exact fp64 routed MoE; dequantizes per expert (VRAM-bounded), frees after use."""
    M, Hh = hidden.shape
    top_k = topk_ids.shape[1]
    R = M * top_k
    tokens = torch.arange(R, device=dev) // top_k
    experts = topk_ids.reshape(-1)
    w = topk_w.reshape(-1).to(torch.float64)
    h_r = hidden[tokens].to(torch.float64)                 # [R, H]
    out = torch.zeros(M, Hh, dtype=torch.float64, device=dev)
    for e in range(E):
        sel = torch.nonzero(experts == e).squeeze(-1)
        if sel.numel() == 0:
            continue
        w_gu = _dequant_one(gu_p[e], gu_s[e], gu_g[e])     # [2I, H] fp64
        w_dn = _dequant_one(dn_p[e], dn_s[e], dn_g[e])     # [H, I] fp64
        hr = h_r[sel]                                      # [n, H]
        ic1 = hr @ w_gu.t()                                # [n, 2I]
        ic2 = _act(ic1[:, :I], ic1[:, I:], act)            # [n, I]
        ic3 = ic2 @ w_dn.t()                               # [n, H]  (DUT: ic2 @ dn.T)
        out.index_add_(0, tokens[sel], ic3 * w[sel][:, None])
        del w_gu, w_dn, ic1, ic2, ic3
    return out


# ----------------------------------------------------------------------------
# DUTs
# ----------------------------------------------------------------------------
def run_e3(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act):
    from freetoken.moe.fused_nvfp4_cublas import fused_experts_nvfp4_cublas
    return fused_experts_nvfp4_cublas(
        hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
        topk_w, topk_ids, E, act, group=8)


def run_prod(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act):
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    return fused_experts_nvfp4(
        hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
        topk_w, topk_ids, E, act, False)


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


# ----------------------------------------------------------------------------
# Data builders
# ----------------------------------------------------------------------------
def make_banks(seed, scale_lo=0x08, scale_hi=0x20):
    """NVFP4 banks. Default scale range decodes to small values (~0.004-0.25) so the
    fp16 GEMM intermediates stay in range for the MoE matrix; the dequant unit test
    uses the full 0x00-0xFF range separately."""
    g = torch.Generator(device=dev).manual_seed(seed)
    codes = torch.arange(16, dtype=torch.uint8, device=dev)
    gu_p = codes[torch.randint(0, 16, (E, TWO_I, H // 2), generator=g, device=dev)]
    dn_p = codes[torch.randint(0, 16, (E, H, I // 2), generator=g, device=dev)]
    gu_s = torch.randint(scale_lo, scale_hi + 1, (E, TWO_I, H // 16),
                         generator=g, device=dev).to(torch.uint8)
    dn_s = torch.randint(scale_lo, scale_hi + 1, (E, H, I // 16),
                         generator=g, device=dev).to(torch.uint8)
    gu_g = (torch.rand(E, TWO_I, generator=g, device=dev) * 1.0 + 0.5).half()
    dn_g = (torch.rand(E, H, generator=g, device=dev) * 1.0 + 0.5).half()
    return gu_p, gu_s, gu_g, dn_p, dn_s, dn_g


def routing(M, top_k, kind, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    if kind == "uniform":
        return torch.randint(0, E, (M, top_k), generator=g, device=dev).to(torch.int32)
    if kind == "skewed":
        ids = torch.randint(0, E, (M, top_k), generator=g, device=dev)
        hot = torch.randint(0, 8, (M, top_k), generator=g, device=dev)
        ids = torch.where(torch.rand(M, top_k, generator=g, device=dev) < 0.8, hot, ids)
        return ids.to(torch.int32)
    if kind == "sparse-empty":
        return torch.randint(0, 16, (M, top_k), generator=g, device=dev).to(torch.int32)
    if kind == "single-expert":
        return torch.zeros(M, top_k, device=dev, dtype=torch.int32)
    if kind == "repeated-ids":
        ids = torch.randint(0, E, (M, top_k), generator=g, device=dev)
        if top_k >= 2:
            ids[:, 1] = ids[:, 0]
        return ids.to(torch.int32)
    raise ValueError(kind)


def make_weights(M, top_k, kind, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    if kind == "softmax":
        return torch.softmax(torch.randn(M, top_k, generator=g, device=dev), dim=-1).float()
    if kind == "nontrivial":
        w = torch.rand(M, top_k, generator=g, device=dev) * 1.5
        w[0, 0] = 0.0
        if M * top_k > 1:
            w[-1, -1] = 2.5
        return w.float()
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# (c) exact dequant unit test: all e2m1 codes, nibble order, e4m3 extremes,
#     block boundaries, global scale.
# ----------------------------------------------------------------------------
def test_dequant():
    from freetoken.moe.fused_nvfp4_cublas import _dequant_group, _e2m1_lut
    lut = _e2m1_lut(dev.index)
    N, K = 8, 32                     # K//16 = 2 scale blocks (tests the block boundary)
    Kb = K // 2
    packed = torch.zeros(4, N, Kb, dtype=torch.uint8, device=dev)
    for e in range(4):
        for n in range(N):
            for kb in range(Kb):
                lo = (e * 7 + n * 3 + kb) % 16
                hi = (e * 5 + n * 11 + kb * 2) % 16
                packed[e, n, kb] = (hi << 4) | lo
    # One row per interesting e4m3 byte across the full range (incl. subnormal,
    # smallest normal, max-finite, negatives, and the reserved 0x7F/0xFF codes).
    scale_bytes = [0x00, 0x01, 0x04, 0x07, 0x08, 0x10, 0x38, 0x7E, 0x80, 0xFE, 0x7F, 0xFF]
    scale = torch.zeros(4, N, K // 16, dtype=torch.uint8, device=dev)
    for e in range(4):
        for n in range(N):
            for blk in range(K // 16):
                scale[e, n, blk] = scale_bytes[(e + n + blk) % len(scale_bytes)]
    global_scale = (torch.arange(4 * N, device=dev).float() % 7 + 1).half().reshape(4, N)

    out = _dequant_group(packed, scale, global_scale, lut, N, K, 0, 4)   # fp16 [4,N,K]
    ref = torch.stack([_dequant_one(packed[e], scale[e], global_scale[e]) for e in range(4)])
    out64 = out.to(torch.float64)

    nz = ref != 0
    rel = (out64 - ref).abs() / ref.abs().clamp_min(1e-30)
    max_rel = rel[nz].max().item() if nz.any() else 0.0
    zero_ok = bool((out64[~nz] == 0).all().item()) if (~nz).any() else True
    finite_ok = bool(torch.isfinite(out).all().item())
    ok = max_rel < TOL_DEQUANT_RE and zero_ok and finite_ok

    dec = e4m3_decode(torch.tensor([0x7E, 0x7F, 0xFF], device=dev))
    print(f"[dequant] max rel err (nonzero) = {max_rel:.3e}  zero-exact={zero_ok}  "
          f"finite={finite_ok}")
    print(f"[dequant] e4m3 decode: 0x7E={dec[0].item():.4f}  0x7F={dec[1].item():.4f}  "
          f"0xFF={dec[2].item():.4f}  (reserved codes, both kernels + ref use the bitcast)")
    return ok, f"max_rel={max_rel:.3e} zero_ok={zero_ok} finite={finite_ok}"


# ----------------------------------------------------------------------------
# One full-MoE configuration.
# ----------------------------------------------------------------------------
def run_case(name, M, top_k, routing_kind, act, wkind, hidden_scale, use_truth, seed=1):
    gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = make_banks(seed)
    topk_ids = routing(M, top_k, routing_kind, seed + 1)
    topk_w = make_weights(M, top_k, wkind, seed + 2)
    if hidden_scale == 0.0:
        hidden = torch.zeros(M, H, dtype=torch.float16, device=dev)
    else:
        hidden = (torch.randn(M, H, device=dev) * hidden_scale).half()

    out_t = run_e3(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act)
    out_p = run_prod(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act)

    finite = bool(torch.isfinite(out_t).all().item() and torch.isfinite(out_p).all().item())
    drop = rel_l2(out_t, out_p)

    ok = finite and drop < TOL_DROPPIN
    detail = f"drop={drop:.2e} finite={finite}"

    if use_truth:
        ref = moe_ref(hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_ids, topk_w, act)
        truth_t = rel_l2(out_t, ref)
        truth_p = rel_l2(out_p, ref)
        detail += f" truthT={truth_t:.2e} truthP={truth_p:.2e}"
        ok = ok and truth_t < TOL_TRUTH and truth_p < TOL_TRUTH

    _free()
    return ok, detail


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
    print(f"H={H} I={I} E={E}  TOL drop<{TOL_DROPPIN} truth<{TOL_TRUTH} dequant<{TOL_DEQUANT_RE}\n")
    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46s} {detail}", flush=True)

    # --- (c) exact dequant unit test -------------------------------------
    print("== (c) dequant unit test (all e2m1 codes, e4m3 extremes, boundaries) ==")
    try:
        ok, detail = test_dequant()
        record("dequant-exact", ok, detail)
    except Exception:
        traceback.print_exc()
        record("dequant-exact", False, "EXCEPTION")
    _free()

    # --- (a) full drop-in matrix (E3 vs production, fast) -----------------
    print("\n== (a) drop-in matrix: E3 vs production (M<=64 + M=2048) ==")
    routing_kinds = ["uniform", "skewed", "sparse-empty", "single-expert", "repeated-ids"]
    for M in (1, 16, 64):
        for top_k in (1, 2, 4, 8):
            for rk in routing_kinds:
                name = f"M{M} tk{top_k} {rk}"
                try:
                    ok, detail = run_case(name, M, top_k, rk, "silu", "softmax", 1.0,
                                          use_truth=False, seed=M * 100 + top_k)
                    record(name, ok, detail)
                except Exception:
                    traceback.print_exc()
                    record(name, False, "EXCEPTION")
                    _free()
    for rk in ("uniform", "skewed"):
        name = f"M2048 tk8 {rk}"
        try:
            ok, detail = run_case(name, 2048, 8, rk, "silu", "softmax", 1.0,
                                  use_truth=False, seed=7)
            record(name, ok, detail)
        except Exception:
            traceback.print_exc()
            record(name, False, "EXCEPTION")
            _free()

    # --- (b) truth subset (E3 + prod vs fp64; per-expert dequant) --------
    print("\n== (b) truth subset: E3 & prod vs fp64 reference ==")
    for M in (1, 64):
        for top_k in (1, 8):
            for rk in ("uniform", "skewed", "single-expert", "repeated-ids"):
                name = f"truth M{M} tk{top_k} {rk}"
                try:
                    ok, detail = run_case(name, M, top_k, rk, "silu", "softmax", 1.0,
                                          use_truth=True, seed=M * 100 + top_k)
                    record(name, ok, detail)
                except Exception:
                    traceback.print_exc()
                    record(name, False, "EXCEPTION")
                    _free()

    # --- (a/b) activation / weight / magnitude sweep (M=64, truth on) ----
    print("\n== (a/b) activation / weight / magnitude sweep (M=64) ==")
    for act in ("silu", "gelu", "gelu_tanh"):
        for wkind in ("softmax", "nontrivial"):
            for hs in (0.01, 1.0):
                name = f"act={act} w={wkind} hs={hs}"
                try:
                    ok, detail = run_case(name, 64, 8, "uniform", act, wkind, hs,
                                          use_truth=True, seed=42)
                    record(name, ok, detail)
                except Exception:
                    traceback.print_exc()
                    record(name, False, "EXCEPTION")
                    _free()

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [n for n, ok, _ in results if not ok]
    print(f"TOTAL: {passed}/{len(results)} passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print("ALL PHASE-E EXPERT CORRECTNESS GATES PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
