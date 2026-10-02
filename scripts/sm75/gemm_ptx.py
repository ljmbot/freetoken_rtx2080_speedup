"""PTX analysis only (fast): does tl.dot lower to tensor-core mma on SM75?"""
import re
import torch, triton, triton.language as tl

dev="cuda"
M,K,N=2048,2048,12288
a=torch.randn(M,K,device=dev,dtype=torch.float16)
w=torch.randn(N,K,device=dev,dtype=torch.float16)

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

out=torch.empty(M,N,device=dev,dtype=torch.float16)
g[(triton.cdiv(M,64),triton.cdiv(N,128))](
    a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),
    out.stride(0),out.stride(1),BM=64,BN=128,BK=64,num_warps=8,num_stages=3)

# grab the compiled kernel's PTX
import triton.runtime.jit as j
kern = g
# The .asm lives on the compiled kernel object; pull from cache
ptx = None
for key, val in list(kern.device_caches.items()) if hasattr(kern,'device_caches') else []:
    pass
# Simpler: use the returned kernel handle
# Re-launch capturing the handle
out2=torch.empty(M,N,device=dev,dtype=torch.float16)
h = g[(triton.cdiv(M,64),triton.cdiv(N,128))](
    a,w,out2,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),
    out2.stride(0),out2.stride(1),BM=64,BN=128,BK=64,num_warps=8,num_stages=3)
try:
    ptx = h.asm.get("ptx","")
except Exception as e:
    print("no asm handle:", e)
    ptx=""
if ptx:
    hmma=len(re.findall(r"\bmma\.[\w.]+|hmma\.[\w.]+",ptx))
    ffma=len(re.findall(r"\bffma\.[\w.]+",ptx))
    fma =len(re.findall(r"\bfma\.[\w.]+",ptx))
    print(f"PTX: mma/hmma={hmma}  ffma(fp32)={ffma}  fma(fp16)={fma}")
    print("=> tensor cores:", "YES" if hmma>0 else "NO (CUDA-core FMA)")
    # show a sample of mma lines
    for line in ptx.splitlines():
        if re.search(r"\bmma\.[\w.]+|hmma\.[\w.]+",line):
            print("   ", line.strip()[:100]); break
else:
    print("no ptx captured")
