"""Decompose the NVFP4 expert GEMM cost: pure tensor-core GEMM ceiling vs the
grouped-path overhead (dequant-to-HBM + scatter/gather + bmm inefficiency).

Establishes whether tensor cores CAN beat the current 0.76 TFLOPS Triton kernel,
and where the bmm candidate's 280 ms goes.
"""
import time
import torch

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, E, TOP_K, M = 2048, 512, 256, 8, 2048
DT = torch.float16
ROUTES = M * TOP_K  # 16384


def timeit(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    print(f"ROUTES={ROUTES} H={H} I={I} E={E} TOP_K={TOP_K}\n")

    # ---- (A) pure cuBLAS FP16 dense GEMM ceiling (total expert FLOPs) ----
    A_gu = torch.randn(ROUTES, H, dtype=DT, device=dev)
    W_gu = torch.randn(H, 2 * I, dtype=DT, device=dev)
    A_dn = torch.randn(ROUTES, I, dtype=DT, device=dev)
    W_dn = torch.randn(I, H, dtype=DT, device=dev)

    flops_gu = 2 * ROUTES * H * 2 * I
    flops_dn = 2 * ROUTES * I * H
    flops_tot = flops_gu + flops_dn

    t_gu = timeit(lambda: A_gu @ W_gu)
    t_dn = timeit(lambda: A_dn @ W_dn)
    print(f"[A] pure cuBLAS FP16 dense GEMM ceiling (total expert FLOPs):")
    print(f"    gate_up : {t_gu:7.2f} ms  ({flops_gu/t_gu/1e9:6.1f} GFLOPS)")
    print(f"    down    : {t_dn:7.2f} ms  ({flops_dn/t_dn/1e9:6.1f} GFLOPS)")
    print(f"    total   : {t_gu+t_dn:7.2f} ms  ({flops_tot/(t_gu+t_dn)/1e9:6.1f} GFLOPS)  <- ceiling")

    # ---- (B) bmm grouped GEMM (pre-dequanted FP16 weights, no dequant cost) ----
    # per-expert FP16 weights: [E, 2I, H] and [E, H, I]
    Wg = torch.randn(E, 2 * I, H, dtype=DT, device=dev)
    Wd = torch.randn(E, H, I, dtype=DT, device=dev)
    # activations scattered per expert (M_e ~ 64 each)
    topk_ids = torch.randint(0, E, (ROUTES,), dtype=torch.int32, device=dev)
    A_sc = torch.randn(ROUTES, H, dtype=DT, device=dev)
    A_sc2 = torch.randn(ROUTES, I, dtype=DT, device=dev)

    # build per-expert batches (M_g = 64 uniform for a clean measure)
    Me = ROUTES // E  # 64
    Ag = A_sc.reshape(E, Me, H)          # [E, 64, H]
    A2g = A_sc2.reshape(E, Me, I)        # [E, 64, I]
    t_bmm_gu = timeit(lambda: torch.bmm(Ag, Wg.transpose(1, 2)))  # [E,64,2I]
    t_bmm_dn = timeit(lambda: torch.bmm(A2g, Wd.transpose(1, 2)))  # [E,64,H]
    print(f"\n[B] bmm grouped GEMM (pre-dequanted FP16, uniform M_e={Me}):")
    print(f"    gate_up : {t_bmm_gu:7.2f} ms  ({flops_gu/t_bmm_gu/1e9:6.1f} GFLOPS)")
    print(f"    down    : {t_bmm_dn:7.2f} ms  ({flops_dn/t_bmm_dn/1e9:6.1f} GFLOPS)")
    print(f"    total   : {t_bmm_gu+t_bmm_dn:7.2f} ms  ({flops_tot/(t_bmm_gu+t_bmm_dn)/1e9:6.1f} GFLOPS)")
    print(f"    (vs dense ceiling {t_gu+t_dn:.2f} ms: bmm is {(t_bmm_gu+t_bmm_dn)/(t_gu+t_dn):.2f}x)")

    # ---- (C) dequant cost: NVFP4 -> FP16 for all 256 experts ----
    packed = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    scale = torch.randint(0x38, 0x3C, (E, 2 * I, H // 16), dtype=torch.uint8, device=dev)
    LUT = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
                       dtype=torch.float32, device=dev)

    def dequant_one(packed_e, scale_e, N, K):
        lo = (packed_e & 0xF).long()
        hi = ((packed_e >> 4) & 0xF).long()
        codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
        w = LUT[codes]
        vi = scale_e.to(torch.int32)
        h = ((vi & 0x80) << 8) | ((vi & 0x7F) << 7)
        s = h.to(torch.int16).view(torch.float16).float() * 256.0
        s = s.repeat_interleave(16, dim=1)
        return (w * s).half()

    def dequant_all():
        out = torch.empty(E, 2 * I, H, dtype=DT, device=dev)
        for e in range(E):
            out[e] = dequant_one(packed[e], scale[e], 2 * I, H)
        return out
    t_deq = timeit(dequant_all, warmup=2, reps=5)
    wbytes = E * 2 * I * H * 0.5  # NVFP4 bytes read
    print(f"\n[C] dequant NVFP4->FP16 (all {E} experts, Python loop):")
    print(f"    {t_deq:7.2f} ms  (reads {wbytes/1e6:.0f} MiB NVFP4 -> writes {E*2*I*H*2/1e6:.0f} MiB FP16)")

    # ---- (D) reference: current Triton kernel ----
    print(f"\n[D] current Triton inline-dequant kernel (from isolate): ~136 ms/layer (0.76 TFLOPS)")

    print(f"\n=== decomposition ===")
    print(f"  dense GEMM ceiling : {t_gu+t_dn:6.2f} ms   <- best case (tensor cores, no overhead)")
    print(f"  bmm grouped GEMM   : {t_bmm_gu+t_bmm_dn:6.2f} ms   <- tensor cores + grouped layout")
    print(f"  dequant (all)      : {t_deq:6.2f} ms   <- overhead if dequant-to-HBM")
    print(f"  current Triton     : 136.0 ms   <- baseline (0 tensor cores, fused)")
    print(f"\n  => if bmm GEMM + dequant < 136 ms, tensor cores win; else current kernel wins.")
    print(f"     bmm+dequant = {t_bmm_gu+t_bmm_dn+t_deq:.1f} ms  vs  current 136 ms")


if __name__ == "__main__":
    main()
