"""Direct A/B: the REAL freetoken W8A16 _gemm (e4m3 emu on SM75) vs cuBLAS FP16,
on the exact dense-projection shapes from the profile. This isolates the kernel,
not the model, so the speedup is a clean property of the GEMM path.
"""
import time, statistics
import torch
from freetoken.kernel.triton.fp8_pertensor_linear import _gemm, FP8
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native

dev = "cuda"
print("device:", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("e4m3_native (SM89+):", e4m3_native(), "  -> SM75 uses the emu (uint8) path")

def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

def gflops(M,N,K,ms): return 2*M*N*K/1e9/(ms/1000)/1e12

M, K = 2048, 2048
counts = {2048:40, 9216:10, 12288:30}   # out_proj / full-attn qkv / GDN in_proj

print(f"\n{'N':>6} | {'W8A16 emu':>12} {'TFLOPS':>7} | {'cuBLAS fp16':>12} {'TFLOPS':>7} | {'speedup':>8}")
tot_emo = tot_cublas = 0.0
for N in (2048, 9216, 12288):
    c = counts[N]
    # W8A16 emu: a fp16, weight fp8 (viewed uint8), scale fp32
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    w_fp8 = (torch.randn(N, K, device=dev) * 0.05).to(FP8)
    scale = torch.ones(N, device=dev, dtype=torch.float32)
    w_view = e4m3_kernel_view(w_fp8)
    t_emo = bench(lambda: _gemm(a, w_view, scale, torch.float16))
    # cuBLAS fp16: same FLOPs, weight dequantized to fp16
    w16 = (w_fp8.float() * scale[:, None]).half()
    t_cb = bench(lambda: a @ w16.t())
    g_emo, g_cb = gflops(M,N,K,t_emo), gflops(M,N,K,t_cb)
    print(f"{N:6d} | {t_emo:9.1f} ms {g_emo:7.3f} | {t_cb:9.1f} ms {g_cb:7.2f} | {t_emo/t_cb:7.1f}x")
    tot_emo += t_emo*c; tot_cublas += t_cb*c
    del a, w_fp8, w_view, w16, scale
    torch.cuda.empty_cache()

tot_gf = sum(2*M*N*K/1e9*c for N,c in counts.items())
print(f"\nFull prefill (4.55 TFLOP) aggregate:")
print(f"  W8A16 emu (profiled): 27936 ms  (trace)")
print(f"  W8A16 emu (this run): {tot_emo:8.1f} ms  -> {tot_gf/1e3/(tot_emo/1000):.3f} TFLOPS")
print(f"  cuBLAS fp16:          {tot_cublas:8.1f} ms  -> {tot_gf/1e3/(tot_cublas/1000):.2f} TFLOPS")
print(f"  Speedup emu->cuBLAS:  {tot_emo/tot_cublas:.1f}x")
print(f"\n  => prefill-2048 if dense GEMM at cuBLAS-fp16: ~{(36637-27936)+tot_cublas:.0f} ms (from 36637 ms)")
