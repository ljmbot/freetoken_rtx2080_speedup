"""Isolate WHY the W8A16 emu GEMM is 171x slow on SM75:
  (A) current _gemm: fp16 act x fp8 weight, e4m3 decode INSIDE the kernel
  (B) Triton fp16 GEMM: fp16 act x fp16 weight (pre-decoded), NO e4m3 in kernel
  (C) cuBLAS fp16
If (B)~(C) >> (A): the e4m3 in-kernel decode is the culprit -> fix = decode weights
once at load (or a better decode), keep FP8 storage only if memory forces it.
If (B) is also slow: the Triton GEMM tiling on SM75 is the culprit.
"""
import time, statistics
import torch, triton, triton.language as tl
from freetoken.kernel.triton.fp8_pertensor_linear import _gemm, FP8
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_u8_to_f32

dev = "cuda"
def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
def gfl(M,N,K,ms): return 2*M*N*K/1e9/(ms/1000)/1e12

# A plain fp16 GEMM in Triton (same tiling style as _gemm) to test the dot path
@triton.jit
def _gemm_fp16(a_ptr, w_ptr, c_ptr, M, N, K,
               sam, sak, swn, swk, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m*BM + tl.arange(0,BM); on = pid_n*BN + tl.arange(0,BN)
    ok = tl.arange(0,BK)
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

def gemm_fp16(a, w16, out_dtype):
    M,K = a.shape; N = w16.shape[0]
    out = torch.empty((M,N), dtype=out_dtype, device=a.device)
    BM = 64 if M>=64 else 32; BN,BK = 128,64
    _gemm_fp16[(triton.cdiv(M,BM), triton.cdiv(N,BN))](
        a, w16, out, M,N,K, a.stride(0),a.stride(1), w16.stride(0),w16.stride(1),
        out.stride(0),out.stride(1), BM=BM,BN=BN,BK=BK,
        num_warps=8 if M>=64 else 4, num_stages=3)
    return out

M,K = 2048,2048
N = 12288   # the dominant GDN in_proj shape
c = torch.randn(M,K,device=dev,dtype=torch.float16)
w_fp8 = (torch.randn(N,K,device=dev)*0.05).to(FP8)
scale = torch.ones(N,device=dev,dtype=torch.float32)
w_view = e4m3_kernel_view(w_fp8)
w16 = (w_fp8.float()*scale[:,None]).half()

tA = bench(lambda: _gemm(c, w_view, scale, torch.float16))
tB = bench(lambda: gemm_fp16(c, w16, torch.float16))
tC = bench(lambda: c @ w16.t())
print(f"N={N} (GDN in_proj) M={M} K={K}")
print(f"  (A) W8A16 emu (e4m3 in-kernel): {tA:8.1f} ms  {gfl(M,N,K,tA):7.3f} TFLOPS")
print(f"  (B) Triton fp16 GEMM (pre-decoded): {tB:8.1f} ms  {gfl(M,N,K,tB):7.2f} TFLOPS")
print(f"  (C) cuBLAS fp16:                    {tC:8.1f} ms  {gfl(M,N,K,tC):7.2f} TFLOPS")
print(f"  A vs C: {tA/tC:.1f}x   B vs C: {tB/tC:.2f}x")
print(f"\n  => if B~C: culprit is the e4m3 in-kernel decode (fix: decode at load or better decode).")
print(f"     if B<<C: culprit is Triton GEMM tiling on SM75 (fix: retune tiles / use cuBLAS).")
