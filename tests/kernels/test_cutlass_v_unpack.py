"""Cutlass sub-byte V unpack round-trip: mha_pool._pack_v -> cutlass._unpack_v.

The CUTLASS dequant path reads the packed uint8 V slab; _unpack_v must invert
_pack_v exactly for int4 and int5 (all field values, all bit-boundary cases).
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.attention.cutlass import CutlassAttentionBackend
from freetoken.kvcache.mha_pool import MHAKVCache  # noqa: E402  (_pack_v owner)

_unpack = CutlassAttentionBackend._unpack_v


@pytest.mark.parametrize("bits", [4, 5])
def test_unpack_roundtrip_all_field_values(bits):
    qmax = (1 << (bits - 1)) - 1
    # Sweep every legal quantized value, incl. both sign extremes, in a
    # [slots, kv_heads, head_dim] tile (head_dim=256 -> full boundary coverage).
    slots, kvh, D = 3, 2, 256
    vals = torch.arange(-qmax, qmax + 1, dtype=torch.int8)
    n = vals.numel()
    reps = (slots * kvh * D + n - 1) // n
    q = vals.repeat(reps)[: slots * kvh * D].reshape(slots, kvh, D).contiguous()
    packed = MHAKVCache._pack_v(q, bits)
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == D * bits // 8
    out = _unpack(packed, D)
    assert out.shape == (slots, kvh, D)
    assert torch.equal(out, q.to(torch.int32))


@pytest.mark.parametrize("bits", [4, 5])
def test_unpack_roundtrip_random(bits):
    torch.manual_seed(7)
    qmax = (1 << (bits - 1)) - 1
    q = torch.randint(-qmax, qmax + 1, (5, 2, 256), dtype=torch.int8)
    packed = MHAKVCache._pack_v(q, bits)
    assert torch.equal(_unpack(packed, 256), q.to(torch.int32))


def test_dequant_kv_packed_matches_reference():
    """_dequant_kv on a packed slab == quantized values * scale (fp32 math)."""
    torch.manual_seed(11)
    bits, slots, kvh, D = 5, 4, 2, 256
    qmax = (1 << (bits - 1)) - 1
    q = torch.randint(-qmax, qmax + 1, (slots, kvh, D), dtype=torch.int8)
    scale = torch.rand(slots, kvh, dtype=torch.float32) + 0.5
    packed = MHAKVCache._pack_v(q, bits)
    # K slab: plain int8 (v unpack is the code under test).
    k = torch.randint(-127, 128, (slots, kvh, D), dtype=torch.int8)
    k_scale = torch.rand(slots, kvh, dtype=torch.float32) + 0.5
    backend = object.__new__(CutlassAttentionBackend)  # no model needed
    k_out, v_out = backend._dequant_kv(
        k, packed, k_scale, scale,
        torch.arange(slots), torch.float16,
    )
    v_ref = (q.to(torch.float32) * scale.unsqueeze(-1)).to(torch.float16)
    k_ref = (k.to(torch.float32) * k_scale.unsqueeze(-1)).to(torch.float16)
    torch.testing.assert_close(v_out, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(k_out, k_ref, atol=0, rtol=0)


def test_dequant_kv_int8_path_unchanged():
    torch.manual_seed(13)
    slots, kvh, D = 4, 2, 256
    k = torch.randint(-127, 128, (slots, kvh, D), dtype=torch.int8)
    v = torch.randint(-127, 128, (slots, kvh, D), dtype=torch.int8)
    k_scale = torch.rand(slots, kvh, dtype=torch.float32) + 0.5
    v_scale = torch.rand(slots, kvh, dtype=torch.float32) + 0.5
    backend = object.__new__(CutlassAttentionBackend)
    k_out, v_out = backend._dequant_kv(
        k, v, k_scale, v_scale, torch.arange(slots), torch.float16
    )
    v_ref = (v.to(torch.float32) * v_scale.unsqueeze(-1)).to(torch.float16)
    torch.testing.assert_close(v_out, v_ref, atol=0, rtol=0)
