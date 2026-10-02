"""Make-or-break test: can a SINGLE fused Triton dequant kernel dequantize a group
of experts in ~5 ms (vs the 150 ms Python loop)? If yes, the tensor-core path
(dequant + bmm) is viable and beats the current 136 ms fused kernel.

Also measures the scatter cost (building per-expert activation batches from the
moe_align_block_size sorted structure) so we can estimate the full path.
"""
import time
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, E, TOP_K, M = 2048, 512, 256, 8, 2048
DT = torch.float16
ROUTES = M * TOP_K
LUT = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
                   dtype=torch.float32, device=dev)


@triton.jit
def _dequant_kernel(
    packed_ptr,   # [E, N, K//2] uint8
    scale_ptr,    # [E, N, K//16] fp32 (block scale, decoded)
    out_ptr,      # [E, N, K] fp16
    lut_ptr,      # [16] fp32
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr,
):
    e = tl.program_id(0)
    nb = tl.program_id(1)
    kb = tl.program_id(2)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = kb * BLOCK_KB + tl.arange(0, BLOCK_KB)
    K_BYTES = K // 2
    byte_mask = offs_kb < K_BYTES
    n_mask = offs_n < N

    p_ptrs = packed_ptr + e * N * K_BYTES + offs_n[None, :] * K_BYTES + offs_kb[:, None]
    b = tl.load(p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0).to(tl.int32)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    sblk = offs_kb // 8
    s_ptrs = scale_ptr + e * N * (K // 16) + offs_n[None, :] * (K // 16) + sblk[:, None]
    scale = tl.load(s_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0.0)
    b_lo = tl.load(lut_ptr + lo) * scale
    b_hi = tl.load(lut_ptr + hi) * scale

    # write two fp16 halves: even k = lo, odd k = hi
    out_base = out_ptr + e * N * K + offs_n[None, :] * K
    tl.store(out_base + (2 * offs_kb)[:, None], b_lo.to(tl.float16),
             mask=byte_mask[:, None] & n_mask[None, :])
    tl.store(out_base + (2 * offs_kb + 1)[:, None], b_hi.to(tl.float16),
             mask=byte_mask[:, None] & n_mask[None, :])


def fused_dequant(Eg, N, K, BLOCK_N=64, BLOCK_KB=64, num_warps=4, num_stages=3):
    out = torch.empty(Eg, N, K, dtype=DT, device=dev)
    packed = torch.randint(0, 256, (Eg, N, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.rand(Eg, N, K // 16, dtype=torch.float32, device=dev) * 0.5 + 0.5
    grid = (Eg, triton.cdiv(N, BLOCK_N), triton.cdiv(K // 2, BLOCK_KB))
    _dequant_kernel[grid](packed, scale, out, LUT, N, K,
                          BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB,
                          num_warps=num_warps, num_stages=num_stages)
    return out, packed, scale


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
    print(f"H={H} I={I} E={E} TOP_K={TOP_K} M={M}\n")
    for Eg in (32, 64, 128, 256):
        buf_mb = Eg * 2 * I * H * 2 / 1e6
        # gate_up shape: N=2I=1024, K=H=2048
        t_gu = timeit(lambda: fused_dequant(Eg, 2 * I, H), warmup=2, reps=6)
        # down shape: N=H=2048, K=I=512
        t_dn = timeit(lambda: fused_dequant(Eg, H, I), warmup=2, reps=6)
        tot = t_gu + t_dn
        print(f"E={Eg:3d}  buf={buf_mb:6.0f} MiB  dequant: gate_up {t_gu:6.2f} ms + "
              f"down {t_dn:6.2f} ms = {tot:6.2f} ms")

    print("\n=== estimate full tensor-core path (per layer) ===")
    # best-case: dequant (fused) + bmm GEMM (4.81 ms measured) + scatter/gather
    for Eg in (128, 256):
        # dequant for Eg experts (both gemms)
        t_gu = timeit(lambda: fused_dequant(Eg, 2 * I, H), warmup=2, reps=6)
        t_dn = timeit(lambda: fused_dequant(Eg, H, I), warmup=2, reps=6)
        ndq = (256 // Eg) * (t_gu + t_dn)
        print(f"  group={Eg}: dequant {ndq:6.2f} ms + bmm ~4.8 ms + scatter ~? ms")
    print("  vs current Triton: 136 ms")


if __name__ == "__main__":
    main()
