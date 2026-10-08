"""Sub-byte V KV cache (k8v4 / k8v5) attention tests.

K stays int8 (Q8_0); V is quantized to int4/int5 and packed into a uint8 slab
(head_dim//2 / 5*head_dim//8 bytes per head) with per-(token, head) fp32 scales.
The Triton kernels dequantize V in-kernel; these tests compare against a
reference run on the *dequantized* cache, isolating packing/dequant correctness
from quantization noise.
"""

from __future__ import annotations

import pytest
import torch


def _quantize(x: torch.Tensor, bits: int):
    """Mirror of MHAKVCache._quantize: per-(token, head) symmetric scale."""
    qmax = (1 << (bits - 1)) - 1
    x = x.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / qmax).clamp_min(1e-8)
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float32)


def _pack_v(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Mirror of MHAKVCache._pack_v (kept independent so the test catches drift)."""
    if bits == 8:
        return q.to(torch.uint8)
    if bits == 4:
        lo = q[..., 0::2] & 0xF
        hi = q[..., 1::2] & 0xF
        return (lo | (hi << 4)).to(torch.uint8)
    assert bits == 5
    u = (q & 0x1F).to(torch.int64).reshape(*q.shape[:-1], -1, 8)
    i = torch.arange(8, dtype=torch.int64, device=q.device)
    acc = (u << (5 * i)).sum(dim=-1, keepdim=True)
    b = torch.arange(5, dtype=torch.int64, device=q.device)
    return ((acc >> (8 * b)) & 0xFF).to(torch.uint8).reshape(*q.shape[:-1], -1)


def _unpack_v(vb: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of _pack_v: uint8 slab -> int8 values (last dim D)."""
    b = vb.to(torch.int64)
    if bits == 4:
        lo = (b & 0xF)
        hi = (b >> 4)
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        return torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], -1).to(torch.int8)
    assert bits == 5
    d = b.shape[-1] * 8 // 5
    fields = []
    for j in range(d):
        byte = b[..., (5 * j) // 8]
        r = (5 * j) % 8
        nxt = b[..., (5 * j) // 8 + 1] if (5 * j) // 8 + 1 < b.shape[-1] else torch.zeros_like(byte)
        f = (byte >> r) | ((nxt & ((1 << r) - 1)) << (8 - r)) if r else byte
        f = f & 0x1F
        fields.append(torch.where(f >= 16, f - 32, f))
    return torch.stack(fields, dim=-1).to(torch.int8)


def _make_q8v_caches(kf: torch.Tensor, vf: torch.Tensor, v_bits: int, dtype: torch.dtype):
    """Quantize K to int8 and V to int4/int5-packed; return (k_int8, v_packed,
    k_scale, v_scale, k_deq, v_deq) with dequantized fp copies for the reference."""
    kq, ks = _quantize(kf, 8)
    vq, vs = _quantize(vf, v_bits)
    vb = _pack_v(vq, v_bits)
    k_deq = (kq.to(torch.float32) * ks[..., None]).to(dtype)
    v_deq = (_unpack_v(vb, v_bits).to(torch.float32) * vs[..., None]).to(dtype)
    return kq, vb, ks, vs, k_deq, v_deq


def test_pack_v_roundtrip():
    """CPU-only: _pack_v/_unpack_v are exact inverses for int4 and int5."""
    torch.manual_seed(3)
    for bits in (4, 5):
        qmax = (1 << (bits - 1)) - 1
        q = torch.randint(-qmax, qmax + 1, (3, 2, 128), dtype=torch.int8)
        vb = _pack_v(q, bits)
        assert vb.shape == (3, 2, 128 * bits // 8)
        assert vb.dtype == torch.uint8
        assert torch.equal(_unpack_v(vb, bits), q), f"int{bits} round-trip"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("v_bits", [4, 5])
def test_paged_attention_q8v_matches_reference(v_bits: int):
    from freetoken.kernel.triton.attention import paged_attention
    from tests.kernels.test_triton_attention import _reference_paged_attention

    torch.manual_seed(4)
    device = torch.device("cuda")
    head_dim = 128
    num_q_heads = 2
    num_kv_heads = 1
    q = torch.randn(4, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(8, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(8, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    kq, vb, ks, vs, k_deq, v_deq = _make_q8v_caches(k_cache, v_cache, v_bits, torch.bfloat16)
    indptr = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([2, 3, 4, 2], dtype=torch.int64, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        kq,
        vb,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        block_n=4,
        k_scale=ks,
        v_scale=vs,
    )
    expected = _reference_paged_attention(
        q,
        k_deq,
        v_deq,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("v_bits", [4, 5])
def test_decode_attention_q8v_matches_reference(v_bits: int):
    from freetoken.kernel.triton.attention import decode_paged_attention
    from tests.kernels.test_triton_attention import _reference_paged_attention

    torch.manual_seed(5)
    device = torch.device("cuda")
    head_dim = 128
    batch = 2
    num_q_heads = 16
    num_kv_heads = 2
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    kq, vb, ks, vs, k_deq, v_deq = _make_q8v_caches(k_cache, v_cache, v_bits, torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q,
        kq,
        vb,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_scale=ks,
        v_scale=vs,
    )
    expected = _reference_paged_attention(
        q,
        k_deq,
        v_deq,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("v_bits", [4, 5])
@pytest.mark.parametrize("use_split_inputs", [False, True])
def test_extend_attention_q8v_matches_reference(v_bits: int, use_split_inputs: bool):
    from freetoken.kernel.triton.attention import extend_paged_attention
    from tests.kernels.test_triton_attention import _reference_paged_attention

    torch.manual_seed(6)
    device = torch.device("cuda")
    head_dim = 128
    num_q_heads = 4
    num_kv_heads = 1
    cached_lens = [3, 2]
    extend_lens = [2, 3]
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q = sum(extend_lens)
    total_kv = sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    kq, vb, ks, vs, k_deq, v_deq = _make_q8v_caches(k_cache, v_cache, v_bits, torch.bfloat16)
    # The extend chunk is stored fp16 (v_extend path); the prefix reads hit the
    # packed slab. Mirror the pool: prefix slots are quantized, extend slots are
    # the raw fp16 values the kernel reads from v_extend.
    qo_indptr = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    k_extend = torch.empty(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_extend = torch.empty(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    offset = 0
    kv_offset = 0
    for req_idx, (cached_len, extend_len) in enumerate(zip(cached_lens, extend_lens)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(
            cached_len, cached_len + extend_len, dtype=torch.int64, device=device
        )
        k_extend[offset : offset + extend_len] = k_cache[
            kv_offset + cached_len : kv_offset + cached_len + extend_len
        ]
        v_extend[offset : offset + extend_len] = v_cache[
            kv_offset + cached_len : kv_offset + cached_len + extend_len
        ]
        # Reference: the split kernel reads the extend chunk from v_extend (raw
        # fp16); the non-split kernel reads it from the cache (quantized). Mirror
        # each: raw override only for the split path.
        if use_split_inputs:
            k_deq[kv_offset + cached_len : kv_offset + cached_len + extend_len] = k_extend[
                offset : offset + extend_len
            ]
            v_deq[kv_offset + cached_len : kv_offset + cached_len + extend_len] = v_extend[
                offset : offset + extend_len
            ]
        offset += extend_len
        kv_offset += cached_len + extend_len
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q,
        kq,
        vb,
        qo_indptr,
        kv_indptr,
        indices,
        prefix_lens,
        max(extend_lens),
        sm_scale,
        k_extend=k_extend if use_split_inputs else None,
        v_extend=v_extend if use_split_inputs else None,
        k_scale=ks,
        v_scale=vs,
    )
    expected = _reference_paged_attention(
        q,
        k_deq,
        v_deq,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
