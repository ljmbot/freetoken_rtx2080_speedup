"""RIGOROUS re-measurement. The first pass showed cuBLAS fp16 at 33 TFLOPS which
exceeds the RTX 2080 hardware peak (~10 TFLOPS fp16 tensor) -- impossible, so either
the cuBLAS number or the Triton number (or both) is wrong. Re-measure with:
  - consumed outputs (no dead-code elimination)
  - correctness check (Triton vs cuBLAS) to prove the Triton kernel really computes
  - a calibration GEMM to establish the achievable peak on THIS card
"""
import time, statistics
import torch, triton, triton.language as tl
from freetoken.kernel.triton.fp8_pertensor_linear import _gemm, FP8
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view

dev = "cuda"
def bench(fn, iters=10, warmup=3, sink=None):
    for _ in range(warmup):
        r = fn()
        if sink is not None: sink.copy_(r)   # consume
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn()
        if sink is not None: sink.copy_(r)
        torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

def gfl(M,N,K,ms): return 2*M*N*K/1e9/(ms/1000)/1e12

# ---- calibration: what peak can THIS card actually hit? ----
for (M,N,K) in [(4096,4096,4096),(8192,8192,8192)]:
    a = torch.randn(M,K,device=dev,dtype=torch.float16)
    b = torch.randn(K,N,device=dev,dtype=torch.float16)
    sink = torch.empty(M,N,device=dev,dtype=torch.float16)
    t = bench(lambda: a@b, sink=sink)
    print(f"calib cuBLAS fp16 {M}x{N}x{K}: {t:8.1f} ms  {gfl(M,N,K,t):6.2f} TFLOPS")
    del a,b,sink; torch.cuda.empty_cache()

print()
# ---- the real shapes, consumed + correctness ----
@triton.jit
def _gemm_fp16(a_ptr, w_ptr, c_ptr, M, N, K, sam, sak, swn, swk, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m*BM + tl.arange(0,BM); on = pid_n*BN + tl.arange(0,BN); ok = tl.arange(0,BK)
    mm = om < M; nm = on < N
    ap = a_ptr + om[:,None]*sam + ok[None,:]*sak
    wp = w_ptr + on[:,None]*swn + ok[None,:]*swk
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(tl.cdiv(K,BK)):
        kr = K - k*BK
        a = tl.load(ap, mask=mm[:,None] & (ok[None,:]<kr), other=0.0)
        w = tl.load(wp, mask=nm[:,None] & (ok[None,:]<kr), other=0.0)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        ap += BK*sak; wp += BK*swk
    cp = c_ptr + om[:,None]*scm + on[None,:]*scn
    tl.store(cp, acc.to(tl.float16), mask=mm[:,None]&nm[None,:])

def gemm_fp16(a, w16):
    M,K = a.shape; N = w16.shape[0]
    out = torch.empty((M,N), dtype=torch.float16, device=a.device)
    BM = 64 if M>=64 else 32; BN,BK = 128,64
    _gemm_fp16[(triton.cdiv(M,BM), triton.cdiv(N,BN))](
        a, w16, out, M,N,K, a.stride(0),a.stride(1), w16.stride(0),w16.stride(1),
        out.stride(0),out.stride(1), BM=BM,BN=BN,BK=BK, num_warps=8 if M>=64 else 4, num_stages=3)
    return out

M,K = 2048,2048
N = 12288
c = torch.randn(M,K,device=dev,dtype=torch.float16)
w_fp8 = (torch.randn(N,K,device=dev)*0.05).to(FP8)
scale = torch.ones(N,device=dev,dtype=torch.float32)
w_view = e4m3_kernel_view(w_fp8)
w16 = (w_fp8.float()*scale[:,None]).half()
sink = torch.empty(M,N,device=dev,dtype=torch.float16)

# correctness: Triton fp16 vs cuBLAS
ref = c @ w16.t()
tr = gemm_fp16(c, w16)
err = (tr - ref).abs().max().item()
rel = ((tr-ref).abs()/(ref.abs()+1e-3)).max().item()
print(f"correctness Triton-fp16 vs cuBLAS: max_abs_err={err:.4f} max_rel={rel:.4f} "
      f"({'OK' if rel<0.05 else 'WRONG'})")

tA = bench(lambda: _gemm(c, w_view, scale, torch.float16), sink=sink)
tB = bench(lambda: gemm_fp16(c, w16), sink=sink)
tC = bench(lambda: c @ w16.t(), sink=sink)
print(f"\nM={M} N={N} K={K} (GDN in_proj), consumed outputs:")
print(f"  (A) Triton W8A16 emu:    {tA:8.1f} ms  {gfl(M,N,K,tA):7.3f} TFLOPS")
print(f"  (B) Triton fp16 GEMM:    {tB:8.1f} ms  {gfl(M,N,K,tB):7.2f} TFLOPS")
print(f"  (C) cuBLAS fp16:         {tC:8.1f} ms  {gfl(M,N,K,tC):7.2f} TFLOPS")
print(f"  A/C = {tA/tC:.1f}x   B/C = {tB/tC:.1f}x")
