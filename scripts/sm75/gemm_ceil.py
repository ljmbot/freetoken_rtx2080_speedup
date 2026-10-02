"""Phase B microbenchmark: how fast can the dense-projection GEMM go on this 2080?

The profiled prefill spends 27.9 s in _gemm_kernel (W8A16 Triton, e4m3 software-decoded
on SM75). Question: what's the realistic ceiling for the same FLOPs?
  (a) cuBLAS FP16 x FP16 (weights dequantized to fp16 at load)  -- the obvious Phase E target
  (b) cuBLAS FP32 (sanity)
  (c) the actual _gemm_kernel (W8A16 emu) for reference
"""
import time, statistics, sys
import torch

dev = "cuda"
print("device:", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

def gflops(M,N,K,ms): return 2*M*N*K/1e9/(ms/1000)/1e12

M, K = 2048, 2048
a16 = torch.randn(M,K,device=dev,dtype=torch.float16)
a32 = a16.float()

print(f"\n{'N':>6} | {'cuBLAS fp16':>14} {'TFLOPS':>7} | {'cuBLAS fp32':>14} {'TFLOPS':>7}")
tot16 = tot32 = 0.0
for N in (2048, 9216, 12288):
    w16 = torch.randn(N,K,device=dev,dtype=torch.float16)
    w32 = w16.float()
    t16 = bench(lambda: a16 @ w16.t())
    t32 = bench(lambda: a32 @ w32.t())
    g16, g32 = gflops(M,N,K,t16), gflops(M,N,K,t32)
    print(f"{N:6d} | {t16:10.1f} ms {g16:7.2f} | {t32:10.1f} ms {g32:7.2f}")
    del w16, w32
    torch.cuda.empty_cache()

# aggregate over the 3 shapes with the right counts (40x2048, 10x9216, 30x12288)
counts = {2048:40, 9216:10, 12288:30}
def agg(dtype):
    a = a16 if dtype==torch.float16 else a32
    tot=0
    for N,c in counts.items():
        w = torch.randn(N,K,device=dev,dtype=dtype)
        t = bench(lambda: a @ w.t(), iters=10)
        tot += t*c
        del w; torch.cuda.empty_cache()
    return tot
t16 = agg(torch.float16); t32 = agg(torch.float16)  # placeholder
t16 = agg(torch.float16)
t32 = agg(torch.float32)
tot_gf = sum(2*M*N*K/1e9*c for N,c in counts.items())
print(f"\nAggregate over full prefill (4.55 TFLOP):")
print(f"  cuBLAS fp16: {t16:8.1f} ms  -> {tot_gf/1e3/(t16/1000):6.2f} TFLOPS")
print(f"  cuBLAS fp32: {t32:8.1f} ms  -> {tot_gf/1e3/(t32/1000):6.2f} TFLOPS")
print(f"  profiled _gemm_kernel (W8A16 emu): 27936 ms (from trace)")
print(f"\nSpeedup of cuBLAS fp16 over profiled GEMM: {27936/t16:.1f}x")
print(f"Estimated prefill-2048 if GEMM at cuBLAS-fp16 speed: {27936-27936+t16 + (36637-27936):.0f} ms")
