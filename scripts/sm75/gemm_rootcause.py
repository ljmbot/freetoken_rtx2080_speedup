"""Root-cause the 160x Triton-vs-cuBLAS gap on SM75.
  1. Recompute real TFLOPS (fix the 1e12 bug).
  2. Dump the Triton GEMM PTX and count HMMA (tensor-core) vs FFMA (CUDA-core)
     instructions -> is tl.dot lowering to tensor cores on Turing?
  3. Try several Triton GEMM configs (block sizes / warps / stages) to see if ANY
     approaches cuBLAS -> is it a tuning issue or a codegen issue?
"""
import time, statistics, re
import torch, triton, triton.language as tl

dev = "cuda"
def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
def tfl(M,N,K,ms): return 2*M*N*K/1e12/(ms/1000)   # TFLOPS

M,K,N = 2048,2048,12288
a = torch.randn(M,K,device=dev,dtype=torch.float16)
w16 = torch.randn(N,K,device=dev,dtype=torch.float16)
ref = a @ w16.t()

@triton.jit
def g(a_ptr,w_ptr,c_ptr,M,N,K,sam,sak,swn,swk,scm,scn,
      BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    om=pm*BM+tl.arange(0,BM);on=pn*BN+tl.arange(0,BN);ok=tl.arange(0,BK)
    mm=om<M;nm=on<N
    ap=a_ptr+om[:,None]*sam+ok[None,:]*sak
    wp=w_ptr+on[:,None]*swn+ok[None,:]*swk
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k in range(tl.cdiv(K,BK)):
        kr=K-k*BK
        aa=tl.load(ap,mask=mm[:,None]&(ok[None,:]<kr),other=0.0)
        ww=tl.load(wp,mask=nm[:,None]&(ok[None,:]<kr),other=0.0)
        acc+=tl.dot(aa,tl.trans(ww),out_dtype=tl.float32)
        ap+=BK*sak;wp+=BK*swk
    cp=c_ptr+om[:,None]*scm+on[None,:]*scn
    tl.store(cp,acc.to(tl.float16),mask=mm[:,None]&nm[None,:])

def run(BM,BN,BK,nw,ns):
    out=torch.empty(M,N,device=dev,dtype=torch.float16)
    kern=g[(triton.cdiv(M,BM),triton.cdiv(N,BN))](
        a,w16,out,M,N,K,a.stride(0),a.stride(1),w16.stride(0),w16.stride(1),
        out.stride(0),out.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
    return out,kern

# correctness of one config
out,kern = run(64,128,64,8,3)
rel=((out-ref).abs()/(ref.abs()+1e-3)).max().item()
print(f"config(64,128,64,w8,s3) correctness max_rel={rel:.4f} ({'OK' if rel<0.05 else 'WRONG'})")

# PTX analysis: HMMA vs FFMA
ptx = kern.asm.get("ptx","")
hmma = len(re.findall(r"\bmma\.\w+|hmma\.\w+", ptx))
ffma = len(re.findall(r"\bffma\.\w+", ptx))
fma  = len(re.findall(r"\bfma\.\w+", ptx))
print(f"\nPTX instruction counts (the tl.dot loop):")
print(f"  mma/hmma (tensor-core): {hmma}")
print(f"  ffma  (fp32 CUDA-core): {ffma}")
print(f"  fma   (fp16 CUDA-core): {fma}")
print(f"  => {'tensor cores ARE used' if hmma>0 else 'NO tensor-core mma found -> tl.dot is doing CUDA-core FMA'}")

# sweep configs
t_cublas = bench(lambda: a @ w16.t())
print(f"\ncuBLAS fp16 baseline: {t_cublas:7.1f} ms  {tfl(M,N,K,t_cublas):6.2f} TFLOPS")
print(f"{'BM':>4} {'BN':>5} {'BK':>4} {'w':>2} {'s':>2} | {'ms':>9} {'TFLOPS':>7} {'x vs cuBLAS':>11}")
best=(1e18,None)
def makef(BM,BN,BK,nw,ns):
    def f():
        o=torch.empty(M,N,device=dev,dtype=torch.float16)
        g[(triton.cdiv(M,BM),triton.cdiv(N,BN))](
            a,w16,o,M,N,K,a.stride(0),a.stride(1),w16.stride(0),w16.stride(1),
            o.stride(0),o.stride(1),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
        return o
    return f
for BM in (32,64,128):
    for BN in (32,64,128,256):
        for BK in (32,64,128):
            for nw in (4,8):
                for ns in (2,3,4):
                    try:
                        t = bench(makef(BM,BN,BK,nw,ns), iters=3, warmup=1)
                    except Exception:
                        continue
                    if t<best[0]: best=(t,(BM,BN,BK,nw,ns))
                    if (BM,BN,BK) in ((64,128,64),(64,128,128),(128,128,64),(128,128,128),(128,256,64),(64,256,64)):
                        print(f"{BM:4d} {BN:5d} {BK:4d} {nw:2d} {ns:2d} | {t:9.1f} {tfl(M,N,K,t):7.2f} {t/t_cublas:10.1f}x")
print(f"\nbest triton config: {best[1]}  {best[0]:.1f} ms  {tfl(M,N,K,best[0]):.2f} TFLOPS  ({best[0]/t_cublas:.1f}x vs cuBLAS)")
