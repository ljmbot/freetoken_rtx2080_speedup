"""Make-or-break end-to-end test: full tensor-core NVFP4 expert path vs the current
Triton fused kernel.

Path: fused single-launch dequant (per group) + vectorized sync-free scatter +
torch.bmm (tensor-core) + silu + scatter-back + sum-reduce. Grouped to bound VRAM
(group=64 -> 0.49 GiB, fits the 0.84 GiB free).

If this beats the current 136 ms/layer, the tensor-core branch is a real win (~6x).
"""
import time
import torch
import triton
import triton.language as tl

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, E, TOP_K, M = 2048, 512, 256, 8, 2048
DT = torch.float16
R = M * TOP_K
LUT = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
                   dtype=torch.float32, device=dev)


@triton.jit
def _dequant_kernel(packed_ptr, scale_ptr, out_ptr, lut_ptr, N, K,
                    BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
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
    out_base = out_ptr + e * N * K + offs_n[None, :] * K
    tl.store(out_base + (2 * offs_kb)[:, None], b_lo.to(tl.float16),
             mask=byte_mask[:, None] & n_mask[None, :])
    tl.store(out_base + (2 * offs_kb + 1)[:, None], b_hi.to(tl.float16),
             mask=byte_mask[:, None] & n_mask[None, :])


def dequant_group(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1):
    """Fused single-launch dequant of experts [e0, e1) -> (gu [G,2I,H], dn [G,H,I]) fp16."""
    G = e1 - e0
    gu = torch.empty(G, 2 * I, H, dtype=DT, device=dev)
    dn = torch.empty(G, H, I, dtype=DT, device=dev)
    _dequant_kernel[(G, triton.cdiv(2 * I, 64), triton.cdiv(H // 2, 64))](
        gu_packed, gu_scale, gu, LUT, 2 * I, H, BLOCK_N=64, BLOCK_KB=64, num_warps=4)
    _dequant_kernel[(G, triton.cdiv(H, 64), triton.cdiv(I // 2, 64))](
        dn_packed, dn_scale, dn, LUT, H, I, BLOCK_N=64, BLOCK_KB=64, num_warps=4)
    return gu, dn


def build_scatter(topk_ids):
    """Vectorized sync-free scatter structure (computed once per layer).

    Returns, for each route r in ORIGINAL order:
      expert[r]  : its expert id
      order      : argsort of expert (routes grouped by expert)
      token[r]   : route -> token id
      rank_sorted: for each SORTED position p, its offset within its expert's block
    """
    expert = topk_ids.flatten()                       # [R]
    order = torch.argsort(expert, stable=True)        # [R]
    token = torch.arange(R, device=dev) // TOP_K      # [R]
    sorted_expert = expert[order]                     # [R]
    counts = torch.bincount(expert, minlength=E)      # [E] tokens per expert
    start_pos = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                           torch.cumsum(counts, 0)[:-1]])  # [E] sorted start pos of each expert
    start_at_p = start_pos[sorted_expert]             # [R] start pos of the block at sorted p
    rank_sorted = torch.arange(R, device=dev) - start_at_p  # [R] offset in block
    return expert, order, token, rank_sorted


def tensor_core_expert_out(hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale,
                           dn_global, topk_ids, topk_weights, group=64):
    from freetoken.kernel.triton.activation import silu_and_mul
    expert, order, token, rank_sorted = build_scatter(topk_ids)
    sorted_expert = expert[order]
    token_sorted = token[order]
    gw_sorted = topk_weights.flatten()[order]      # [R]
    # max tokens per expert (for padding)
    counts = torch.bincount(expert, minlength=E)
    Mp = int(counts.max().item())                  # one host sync (max), acceptable
    A_in = hidden[token]                           # [R, H] (route -> token row)
    # scatter-forward: fill A_pad at dest[p] (sorted position p) with route order[p]'s act
    A_pad = torch.zeros(E, Mp, H, dtype=DT, device=dev)
    dest = sorted_expert * Mp + rank_sorted
    A_pad.view(-1, H)[dest] = A_in[order]
    torch.cuda.synchronize(); print("  [scatter-fwd ok]", flush=True)
    out = torch.zeros(M, H, dtype=DT, device=dev)
    for e0 in range(0, E, group):
        e1 = min(e0 + group, E)
        gu, dn = dequant_group(gu_packed, gu_scale, dn_packed, dn_scale, e0, e1)
        torch.cuda.synchronize(); print(f"  [dequant g{e0}-{e1} ok]", flush=True)
        G = e1 - e0
        A_g = A_pad[e0:e1]                          # [G, Mp, H]
        ic1 = torch.bmm(A_g, gu.transpose(1, 2))    # [G, Mp, 2I] tensor core
        torch.cuda.synchronize(); print(f"  [bmm1 g{e0} ok]", flush=True)
        ic1 = ic1.float() * gu_global[e0:e1].float()[:, None, :]
        ic2 = silu_and_mul(ic1.half().reshape(-1, 2 * I)).reshape(G, Mp, I)
        torch.cuda.synchronize(); print(f"  [act g{e0} ok]", flush=True)
        ic3 = torch.bmm(ic2, dn.transpose(1, 2))    # [G, Mp, H] tensor core
        torch.cuda.synchronize(); print(f"  [bmm2 g{e0} ok]", flush=True)
        ic3 = ic3.float() * dn_global[e0:e1].float()[:, None, :]
        # scatter-back: gather this group's routes, weight, index_add into out
        m = (sorted_expert >= e0) & (sorted_expert < e1)
        sel = torch.nonzero(m).squeeze(-1)
        if sel.numel() == 0:
            continue
        rows = sorted_expert[sel] - e0
        cols = rank_sorted[sel]
        vals = ic3[rows, cols] * gw_sorted[sel][:, None]
        out.index_add_(0, token_sorted[sel], vals.half())
        torch.cuda.synchronize(); print(f"  [scatter-back g{e0} ok]", flush=True)
    return out


def timeit(fn, warmup=3, reps=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm() / (b.norm() + 1e-12)


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    gu_packed = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    gu_scale = torch.rand(E, 2 * I, H // 16, dtype=torch.float32, device=dev) * 0.5 + 0.5
    gu_global = (torch.rand(E, 2 * I, device=dev) * 0.5 + 0.5).half()
    dn_packed = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    dn_scale = torch.rand(E, H, I // 16, dtype=torch.float32, device=dev) * 0.5 + 0.5
    dn_global = (torch.rand(E, H, device=dev) * 0.5 + 0.5).half()
    hidden = (torch.randn(M, H, device=dev) * 0.5).half()
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=dev)
    topk_w = torch.softmax(torch.randn(M, TOP_K, device=dev), dim=-1)

    t_cur = 136.0  # measured separately
    out_cur = None
    if mode in ("both", "cur"):
        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
        print("[1] current Triton fused kernel ...")
        t_cur = timeit(lambda: fused_experts_nvfp4(
            hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
            topk_w, topk_ids, E, "silu", False))
        out_cur = fused_experts_nvfp4(
            hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
            topk_w, topk_ids, E, "silu", False)
        print(f"    {t_cur:.1f} ms/layer")

    for group in (32, 64):
        print(f"\n[2] tensor-core path (group={group}) ...")
        t_tc = timeit(lambda: tensor_core_expert_out(
            hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
            topk_ids, topk_w, group=group))
        out_tc = tensor_core_expert_out(
            hidden, gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global,
            topk_ids, topk_w, group=group)
        print(f"    {t_tc:.1f} ms/layer   ({t_cur/t_tc:.1f}x vs current)")
        if out_cur is not None:
            print(f"    rel-L2 vs current: {rel_l2(out_tc, out_cur):.6f}")
        print(f"    prefill-2048 (40 layers): {40*t_cur/1000:.2f} s -> {40*t_tc/1000:.2f} s")

    print(f"\n=== verdict ===")
    print(f"  current Triton fused : {t_cur:.1f} ms/layer")
    print(f"  (tensor-core numbers above; group=64 is the memory-safe choice)")


if __name__ == "__main__":
    main()
