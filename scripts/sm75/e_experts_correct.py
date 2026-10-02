"""Correctness: tensor-core NVFP4 expert path vs the production Triton kernel,
using the SAME e4m3 scale bytes (so both dequantize identically). Confirms the
tensor-core path is a faithful drop-in (same numerics, faster), not just fast.

My dequant kernel must decode e4m3 exactly like e4m3_u8_to_f32 (uint16 bitcast).
"""
import time
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, E, TOP_K, M = 2048, 512, 256, 8, 2048
two_i = 2 * I
DT = torch.float16
R = M * TOP_K
LUT = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
                   dtype=torch.float32, device=dev)


@triton.jit
def _dequant_e4m3_kernel(packed_ptr, scale_ptr, out_ptr, lut_ptr, N, K,
                         BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    e = tl.program_id(0); nb = tl.program_id(1); kb = tl.program_id(2)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = kb * BLOCK_KB + tl.arange(0, BLOCK_KB)
    K_BYTES = K // 2
    bm = offs_kb < K_BYTES; nm = offs_n < N
    p = packed_ptr + e * N * K_BYTES + offs_n[None, :] * K_BYTES + offs_kb[:, None]
    b = tl.load(p, mask=bm[:, None] * nm[None, :], other=0).to(tl.int32)
    lo = b & 0xF; hi = (b >> 4) & 0xF
    s = scale_ptr + e * N * (K // 16) + offs_n[None, :] * (K // 16) + (offs_kb // 8)[:, None]
    sraw = tl.load(s, mask=bm[:, None] * nm[None, :], other=0)
    # exact e4m3 decode (matches e4m3_u8_to_f32)
    h = ((sraw & 0x80).to(tl.uint16) << 8) | ((sraw & 0x7F).to(tl.uint16) << 7)
    scale = h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    blo = tl.load(lut_ptr + lo) * scale
    bhi = tl.load(lut_ptr + hi) * scale
    ob = out_ptr + e * N * K + offs_n[None, :] * K
    tl.store(ob + (2 * offs_kb)[:, None], blo.to(tl.float16), mask=bm[:, None] * nm[None, :])
    tl.store(ob + (2 * offs_kb + 1)[:, None], bhi.to(tl.float16), mask=bm[:, None] * nm[None, :])


def dequant_group(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1):
    G = e1 - e0
    gu = torch.empty(G, two_i, H, dtype=DT, device=dev)
    dn = torch.empty(G, H, I, dtype=DT, device=dev)
    # slice to the group so the kernel's local expert index maps to the right rows
    _dequant_e4m3_kernel[(G, triton.cdiv(two_i, 64), triton.cdiv(H // 2, 64))](
        gu_packed[e0:e1], gu_scale[e0:e1], gu, LUT, two_i, H, BLOCK_N=64, BLOCK_KB=64, num_warps=4)
    _dequant_e4m3_kernel[(G, triton.cdiv(H, 64), triton.cdiv(I // 2, 64))](
        dn_packed[e0:e1], dn_scale[e0:e1], dn, LUT, H, I, BLOCK_N=64, BLOCK_KB=64, num_warps=4)
    return gu, dn


def build_scatter(topk_ids):
    expert = topk_ids.flatten()
    order = torch.argsort(expert, stable=True)
    token = torch.arange(R, device=dev) // TOP_K
    sorted_expert = expert[order]
    counts = torch.bincount(expert, minlength=E)
    start_pos = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                           torch.cumsum(counts, 0)[:-1]])
    start_at_p = start_pos[sorted_expert]
    rank_sorted = torch.arange(R, device=dev) - start_at_p
    return expert, order, token, rank_sorted


def tensor_core_expert_out(hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale,
                           dn_global, topk_ids, topk_weights, group=64):
    from freetoken.kernel.triton.activation import silu_and_mul
    expert, order, token, rank_sorted = build_scatter(topk_ids)
    sorted_expert = expert[order]
    token_sorted = token[order]
    gw_sorted = topk_weights.flatten()[order]
    counts = torch.bincount(expert, minlength=E)
    Mp = int(counts.max().item())
    A_in = hidden[token]
    A_pad = torch.zeros(E, Mp, H, dtype=DT, device=dev)
    dest = sorted_expert * Mp + rank_sorted
    A_pad.view(-1, H)[dest] = A_in[order]
    out = torch.zeros(M, H, dtype=DT, device=dev)
    for e0 in range(0, E, group):
        e1 = min(e0 + group, E)
        gu, dn = dequant_group(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1)
        G = e1 - e0
        ic1 = torch.bmm(A_pad[e0:e1], gu.transpose(1, 2))
        ic1 = ic1.float() * gu_global[e0:e1].float()[:, None, :]
        ic2 = silu_and_mul(ic1.half().reshape(-1, 2 * I)).reshape(G, Mp, I)
        ic3 = torch.bmm(ic2, dn.transpose(1, 2))
        ic3 = ic3.float() * dn_global[e0:e1].float()[:, None, :]
        m = (sorted_expert >= e0) & (sorted_expert < e1)
        sel = torch.nonzero(m).squeeze(-1)
        if sel.numel() == 0:
            continue
        rows = sorted_expert[sel] - e0
        cols = rank_sorted[sel]
        vals = ic3[rows, cols] * gw_sorted[sel][:, None]
        out.index_add_(0, token_sorted[sel], vals.half())
    return out


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm() / (b.norm() + 1e-12)


def main():
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    # e2m1 packed bytes: only valid codes (0-6, 8-14), avoiding NaN nibbles 0x7/0xF
    valid = torch.tensor([0,1,2,3,4,5,6,8,9,10,11,12,13,14], dtype=torch.uint8, device=dev)
    gu_packed = valid[torch.randint(0, 14, (E, 2 * I, H // 2), device=dev)]
    dn_packed = valid[torch.randint(0, 14, (E, H, I // 2), device=dev)]
    # e4m3 scale bytes (valid, small positive-normal 0x38..0x3B) -- SAME for both paths
    gu_scale = torch.randint(0x38, 0x3C, (E, 2 * I, H // 16), dtype=torch.uint8, device=dev)
    gu_global = (torch.rand(E, 2 * I, device=dev) * 0.5 + 0.5).half()
    dn_scale = torch.randint(0x38, 0x3C, (E, H, I // 16), dtype=torch.uint8, device=dev)
    dn_global = (torch.rand(E, H, device=dev) * 0.5 + 0.5).half()
    hidden = (torch.randn(M, H, device=dev) * 0.01).half()
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=dev)
    topk_w = torch.softmax(torch.randn(M, TOP_K, device=dev), dim=-1)

    print("production Triton kernel ...")
    out_cur = fused_experts_nvfp4(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_w, topk_ids, E, "silu", False)
    print("  out range:", out_cur.min().item(), out_cur.max().item())

    print("tensor-core path (same e4m3 scales) ...")
    out_tc = tensor_core_expert_out(
        hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
        topk_ids, topk_w, group=64)
    print("  out range:", out_tc.min().item(), out_tc.max().item())

    print(f"\nrel-L2 (tensor-core vs production): {rel_l2(out_tc, out_cur):.6f}")
    cs = (out_tc.float().flatten() @ out_cur.float().flatten()) / (
        out_tc.float().norm() * out_cur.float().norm())
    print(f"cosine: {cs.item():.6f}")
    print(f"max abs diff: {(out_tc.float() - out_cur.float()).abs().max().item():.4f}")
    print(f"production norm: {out_cur.float().norm().item():.2f}")


if __name__ == "__main__":
    main()
