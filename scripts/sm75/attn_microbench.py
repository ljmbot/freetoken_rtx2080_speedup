"""Idea-5 microbench: sm75 config sweep for the full-attention prefill kernel.

Targets `_extend_attention_split_kernel` (the extend/prefill path used by the
10 full-attention layers; 42.3% of device at n8192 post-GDN-tune).

Faithful production geometry:
  T = 8192 (n8192), num_q_heads = 32, num_kv_heads = 16 (GQA GROUP=2),
  head_dim = 128, fresh prefill (prefix_len = 0), fp16.

On sm75 (64 KiB opt-in smem) `_select_extend_tile` returns (32, 16) with
num_warps=8, num_stages=1 hardcoded. We sweep BLOCK_M/BLOCK_N/num_warps/
num_stages within the smem budget, time each, and gate a correctness check
against torch SDPA (GQA causal) at a small T.

Usage:
  python scripts/sm75/attn_microbench.py            # sweep + correctness
  python scripts/sm75/attn_microbench.py --quick    # smaller T, fewer configs
"""
import argparse
import time

import torch
import triton

from freetoken.kernel.triton.attention import _extend_attention_split_kernel

DEV = "cuda"
DTYPE = torch.float16
SM_SCALE = None  # set per head_dim


def build_inputs(t: int, num_qh: int, num_kvh: int, hd: int, prefix: int):
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(t, num_qh, hd, generator=g, device=DEV, dtype=DTYPE) * 0.3
    k_ext = torch.randn(t, num_kvh, hd, generator=g, device=DEV, dtype=DTYPE) * 0.3
    v_ext = torch.randn(t, num_kvh, hd, generator=g, device=DEV, dtype=DTYPE) * 0.3
    # paged cache for the prefix (empty when prefix == 0)
    cache_slots = max(prefix, 1)
    k_cache = torch.randn(cache_slots, num_kvh, hd, generator=g, device=DEV, dtype=DTYPE) * 0.3
    v_cache = torch.randn(cache_slots, num_kvh, hd, generator=g, device=DEV, dtype=DTYPE) * 0.3
    qo_indptr = torch.tensor([0, t], device=DEV, dtype=torch.int32)
    kv_indptr = torch.tensor([0, prefix], device=DEV, dtype=torch.int32)
    kv_indices = torch.arange(prefix, device=DEV, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=DEV, dtype=torch.int32)
    return dict(
        q=q, k_ext=k_ext, v_ext=v_ext, k_cache=k_cache, v_cache=v_cache,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        prefix_lens=prefix_lens,
    )


def launch(cfg, inp, t, num_qh, num_kvh, hd, prefix):
    block_m, block_n, num_warps, num_stages = cfg
    block_d = triton.next_power_of_2(hd)
    o = torch.empty_like(inp["q"])
    grid = (1, num_qh, triton.cdiv(t, block_m))
    _extend_attention_split_kernel[grid](
        inp["q"], inp["k_ext"], inp["v_ext"], inp["k_cache"], inp["v_cache"], o,
        inp["qo_indptr"], inp["kv_indptr"], inp["kv_indices"], inp["prefix_lens"],
        1.0 / (hd ** 0.5), inp["q"],
        inp["q"].stride(0), inp["q"].stride(1),
        inp["k_ext"].stride(0), inp["k_ext"].stride(1),
        inp["v_ext"].stride(0), inp["v_ext"].stride(1),
        inp["k_cache"].stride(0), inp["k_cache"].stride(1),
        inp["v_cache"].stride(0), inp["v_cache"].stride(1),
        o.stride(0), o.stride(1),
        GROUP=num_qh // num_kvh, D=hd, BLOCK_D=block_d, BLOCK_DV=block_d,
        BLOCK_M=block_m, BLOCK_N=block_n,
        SLIDING_WINDOW=0, HAS_SINKS=False,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def torch_ref(inp, t, num_qh, num_kvh, hd, prefix):
    """GQA causal reference via SDPA, head-by-head to bound memory."""
    q = inp["q"].float()          # [t, qh, d]
    if prefix > 0:
        k = torch.cat([inp["k_cache"][:prefix], inp["k_ext"]], dim=0).float()
        v = torch.cat([inp["v_cache"][:prefix], inp["v_ext"]], dim=0).float()
    else:
        k = inp["k_ext"].float()
        v = inp["v_ext"].float()
    group = num_qh // num_kvh
    out = torch.empty_like(q)
    # causal mask: row i attends to cols <= prefix + i
    kv_len = prefix + t
    mask = torch.zeros(t, kv_len, device=DEV, dtype=torch.bool)
    for i in range(t):
        mask[i, : prefix + i + 1] = True
    for h in range(num_qh):
        kvh = h // group
        qh = q[:, h, :]                       # [t, d]
        kh = k[:, kvh, :]                     # [kv_len, d]
        vh = v[:, kvh, :]                     # [kv_len, d]
        scores = (qh @ kh.T) * (1.0 / (hd ** 0.5))   # [t, kv_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
        out[:, h, :] = p @ vh
    return out.to(DTYPE)


def time_cfg(cfg, inp, t, num_qh, num_kvh, hd, prefix, reps=5):
    # warmup (also compiles the config)
    launch(cfg, inp, t, num_qh, num_kvh, hd, prefix)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        launch(cfg, inp, t, num_qh, num_kvh, hd, prefix)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--t", type=int, default=None, help="query tokens (default 8192, or 2048 quick)")
    ap.add_argument("--prefix", type=int, default=0, help="cached prefix length (paged path)")
    ap.add_argument("--configs", type=str, default=None,
                    help="comma list of BM,BN,warps,stages (default: full sweep)")
    ap.add_argument("--qh", type=int, default=32, help="num query heads (model: 16)")
    ap.add_argument("--kvh", type=int, default=16, help="num kv heads (model: 2)")
    ap.add_argument("--hd", type=int, default=128, help="head_dim (model: 256)")
    args = ap.parse_args()

    num_qh, num_kvh, hd = args.qh, args.kvh, args.hd
    t = args.t if args.t else (2048 if args.quick else 8192)
    prefix = args.prefix

    print(f"sm75 attention extend-kernel sweep: T={t} qh={num_qh} kvh={num_kvh} "
          f"hd={hd} prefix={prefix} reps={args.reps}")
    inp = build_inputs(t, num_qh, num_kvh, hd, prefix)

    # reference (small T only to bound the [t,t] scores memory)
    ref_t = min(t, 1024)
    print(f"correctness ref at T={ref_t} ...")
    inp_ref = build_inputs(ref_t, num_qh, num_kvh, hd, prefix)
    ref = torch_ref(inp_ref, ref_t, num_qh, num_kvh, hd, prefix)

    # candidate configs: (BLOCK_M, BLOCK_N, num_warps, num_stages)
    if args.configs:
        cands = [tuple(int(x) for x in c.split(",")) for c in args.configs.split(";")]
    elif args.quick:
        cands = [(32, 16, 8, 1), (64, 32, 8, 1), (64, 32, 4, 1), (128, 32, 8, 1)]
    else:
        cands = [
            (32, 16, 8, 1),   # current
            (32, 32, 8, 1),
            (32, 16, 4, 1),
            (64, 16, 8, 1),
            (64, 32, 8, 1),
            (64, 32, 4, 1),
            (64, 32, 8, 2),
            (64, 64, 8, 1),
            (128, 16, 8, 1),
            (128, 32, 8, 1),
            (128, 32, 4, 1),
            (128, 64, 8, 1),
            (128, 64, 8, 2),
            (32, 64, 8, 1),
            (64, 128, 8, 1),
        ]

    results = []
    for cfg in cands:
        try:
            # correctness at small T first (cheap, catches smem/compile issues)
            o_small = launch(cfg, inp_ref, ref_t, num_qh, num_kvh, hd, prefix)
            rel = (o_small.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)
            ok = bool(rel < 1e-2)
            ms = time_cfg(cfg, inp, t, num_qh, num_kvh, hd, prefix, args.reps)
            results.append((cfg, ms, rel, ok))
            print(f"  BM={cfg[0]:3d} BN={cfg[1]:3d} w={cfg[2]:2d} s={cfg[3]}  "
                  f"{ms:8.1f} ms   relL2={rel:.2e}  {'OK' if ok else 'MISMATCH'}")
        except Exception as e:
            print(f"  BM={cfg[0]:3d} BN={cfg[1]:3d} w={cfg[2]:2d} s={cfg[3]}  ERROR: {str(e)[:80]}")

    if results:
        results.sort(key=lambda r: r[1])
        cur = next((r for r in results if r[0] == (32, 16, 8, 1)), None)
        best = results[0]
        print("\n=== RANKING (fastest first) ===")
        for cfg, ms, rel, ok in results[:8]:
            tag = ""
            if cur and cfg == (32, 16, 8, 1):
                tag = "  <-- current"
            print(f"  BM={cfg[0]:3d} BN={cfg[1]:3d} w={cfg[2]:2d} s={cfg[3]}  {ms:8.1f} ms{tag}")
        if cur:
            print(f"\ncurrent (32,16,8,1): {cur[1]:.1f} ms")
            print(f"best:                  {best[1]:.1f} ms  "
                  f"({cur[1]/best[1]:.2f}x)  cfg={best[0]}  relL2={best[2]:.2e}")


if __name__ == "__main__":
    main()
