from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        q8: bool = False,
        v_bits: int = 8,
    ) -> None:
        # Q8_0 KV cache: store the paged K/V as int8 values + a per-(token, head)
        # fp32 scale (llama.cpp's Q8_0) instead of fp16. Halves KV VRAM. The value
        # buffer is int8 and a separate scale buffer holds the fp32 scales; readers
        # dequantize (value * scale) before attention, so compute stays fp16.
        # v_bits < 8 (4/5) packs the V slab sub-byte (uint8) on top of q8:
        # int4 = 2 signed nibbles/byte, int5 = 8 fields per 5-byte group.
        if v_bits not in (4, 5, 8):
            raise ValueError(f"kv v_bits must be 4, 5, or 8, got {v_bits}")
        if v_bits < 8 and not q8:
            raise ValueError("sub-byte V (v_bits < 8) requires the q8 scale machinery")
        self._q8 = bool(q8)
        self._v_bits = int(v_bits)
        self._fp16_dtype = dtype
        if self._q8:
            self._value_dtype = torch.int8
            self._scale_dtype = torch.float32
        else:
            self._value_dtype = dtype
        self._v_dtype = torch.uint8 if self._v_bits < 8 else self._value_dtype
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._v_dim = head_dim * self._v_bits // 8
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads, head_dim, device)

    def _alloc(
        self,
        num_storage_layers: int,
        num_pages: int,
        page_size: int,
        local_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        # K and V slabs are allocated separately: a packed V slab has a narrower
        # last dim (head_dim * v_bits / 8) and uint8 dtype, so a single [2, ...]
        # tensor with a uniform dtype no longer fits.
        self._kv_buffer = torch.empty(
            (num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=self._value_dtype,
        )
        self._v_buffer = torch.empty(
            (num_storage_layers, num_pages, page_size, local_kv_heads, self._v_dim),
            device=device,
            dtype=self._v_dtype,
        )
        self._scale_buffer = (
            torch.empty(
                (2, num_storage_layers, num_pages, page_size, local_kv_heads),
                device=device,
                dtype=self._scale_dtype,
            )
            if self._q8
            else None
        )
        self._k_buffer = self._kv_buffer
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self._v_storage_shape = (num_pages * page_size, local_kv_heads, self._v_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads, head_dim, device)

    @classmethod
    def kv_cost(cls, config, kv_q8: bool = False, kv_v_bits: int = 8, **kwargs) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        if kv_q8:
            # Q8_0: int8 value (1 B) + a per-(token, head) fp32 scale (4 B) replace the
            # fp16 value (2 B). Per token: value = fp16_bytes/2, plus the scale overhead
            # (2 slabs x 4 B x kv_heads x layers, amortized over head_dim). Net ~51% of
            # fp16 (the scale premium is 4 B per head over head_dim=256 dims, ~2%).
            per_token = per_token // 2 + cls._q8_scale_overhead(config)
            if kv_v_bits < 8:
                # Sub-byte V: the V slab shrinks from head_dim B to
                # head_dim*v_bits/8 B per (token, head).
                per_token -= cls._v_packing_savings(config, kv_v_bits)
        return per_token * config.page_size, 0, config.page_size, 0

    @classmethod
    def _v_packing_savings(cls, config, v_bits: int) -> int:
        # Bytes saved per token on the V slab: (head_dim - packed_dim) per
        # (layer, kv head), summed over non-SWA groups.
        from freetoken.utils import div_even

        total = 0
        for spec in config.model_config.kv_cache_group_specs():
            # MLA groups store one slab (V aliases K): no V slab to pack.
            if spec.is_swa or spec.mla:
                continue
            kvh = div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
            total += kvh * spec.num_layers * (spec.head_dim - spec.head_dim * v_bits // 8)
        return total

    @classmethod
    def _q8_scale_overhead(cls, config) -> int:
        # 2 (K+V slabs) x fp32 scale (4 B) x local kv heads, summed over non-SWA layers.
        from freetoken.utils import div_even

        total = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            total += (
                2
                * 4
                * div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
                * spec.num_layers
            )
        return total

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[1]) * int(buf.shape[2])
        kv_bytes = int(buf.numel() * buf.element_size())
        v_buf = self._v_buffer
        kv_bytes += int(v_buf.numel() * v_buf.element_size())
        if self._q8 and self._scale_buffer is not None:
            kv_bytes += int(self._scale_buffer.numel() * self._scale_buffer.element_size())
        return kv_bytes // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scale(self, index: int) -> torch.Tensor:
        """Per-(token, head) fp32 K scale, ``[pages, page_size, kv_heads]`` (Q8 only)."""
        assert self._q8 and self._scale_buffer is not None
        return self._scale_buffer[0][self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor:
        """Per-(token, head) fp32 V scale, ``[pages, page_size, kv_heads]`` (Q8 only)."""
        assert self._q8 and self._scale_buffer is not None
        return self._scale_buffer[1][self._dense(index)]

    @property
    def q8(self) -> bool:
        return self._q8

    @property
    def v_bits(self) -> int:
        """V-slab quantization width (8 = int8, 4/5 = packed sub-byte)."""
        return self._v_bits

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype attention math runs in (fp16/bf16) -- the dequant target."""
        return self._fp16_dtype

    def _quantize(self, x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform symmetric quantization + per-(token, head) fp32 scale.

        ``x`` is ``[tokens, kv_heads, head_dim]``. The scale is ``max|x| / qmax`` per
        (token, head); the value is ``round(x / scale)`` clamped to ``[-qmax, qmax]``.
        A zero row (all-zero key) gets scale 1.0 to avoid divide-by-zero (its value
        is 0 anyway). bits=8 -> int8 (qmax 127); bits=4 -> int4 (qmax 7);
        bits=5 -> int5 (qmax 15). The int4/int5 values stay in an int8 tensor and
        are packed to bytes by ``_pack_v``.
        """
        qmax = (1 << (bits - 1)) - 1
        x = x.to(torch.float32)
        amax = x.abs().amax(dim=-1, keepdim=True)  # [tokens, kv_heads, 1]
        scale = (amax / qmax).clamp_min(1e-8)
        q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
        return q, scale.squeeze(-1).to(torch.float32)

    @staticmethod
    def _pack_v(q: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack int4/int5 values (int8 tensor, last dim D) into uint8 bytes.

        int4: two signed nibbles per byte -- byte = (lo & 0xF) | (hi << 4),
        where lo/hi are the two's-complement nibbles of q[..., 2i]/q[..., 2i+1].
        int5: 8 fields per 5-byte group. The 8 fields form a 40-bit value
        (field i at bits [5i, 5i+5), field 0 in the LOW bits) emitted as 5
        bytes little-endian: byte b = (acc >> 8b) & 0xFF. Field i therefore
        reads as (b[5i//8] >> (5i%8)) | (b[5i//8 + 1] << (8 - 5i%8)) masked
        to 5 bits -- the byte-group scheme from the plan, linear (no wrap).
        """
        if bits == 8:
            return q.to(torch.uint8)
        if bits == 4:
            lo = q[..., 0::2] & 0xF
            hi = q[..., 1::2] & 0xF
            return (lo | (hi << 4)).to(torch.uint8)
        # int5: 8 fields -> 40 bits -> 5 bytes, little-endian.
        assert bits == 5
        u = (q & 0x1F).to(torch.int64).reshape(*q.shape[:-1], -1, 8)  # 5-bit fields
        i = torch.arange(8, dtype=torch.int64, device=q.device)
        acc = (u << (5 * i)).sum(dim=-1, keepdim=True)  # [..., D//8, 1]: 40-bit group
        b = torch.arange(5, dtype=torch.int64, device=q.device)
        out = ((acc >> (8 * b)) & 0xFF).to(torch.uint8)  # [..., D//8, 5]
        return out.reshape(*q.shape[:-1], -1)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        if not self._q8:
            store_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
            return
        # Q8: quantize this chunk to int8 + per-(token, head) scale, then store the
        # value (int8) and the scale (fp32) into their respective slabs. With
        # v_bits < 8 the V values are quantized to int4/int5 and packed into the
        # uint8 V slab (the store kernel copies the narrower V rows).
        assert self._scale_buffer is not None
        assert self._kv_buffer is not None
        # k/v arrive 2D [num_tokens, kv_heads*head_dim] (the model flattens them).
        # Quantize per-(token, head): reshape to [tokens, kvh, D], then flatten the
        # int8 values back to 2D for the byte-copy store kernel (which expects 2D rows).
        D = self._kv_buffer.shape[-1]
        kvh = k.shape[-1] // D
        kf = k.view(-1, kvh, D)
        vf = v.view(-1, kvh, D)
        kq, ks = self._quantize(kf)
        vq, vs = self._quantize(vf, self._v_bits)
        if self._v_bits < 8:
            vq = self._pack_v(vq, self._v_bits)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._v_storage_shape),
            indices=out_loc,
            k=kq.reshape(-1, kvh * D),
            v=vq.reshape(-1, kvh * self._v_dim),
        )
        # Scale slab is [pages, page_size, kv_heads]; scatter the per-token scale rows
        # (out_loc is the global token-slot index) with an index write. index_copy_
        # needs int64 indices (the store kernel above wanted int32, so cast here).
        scale_heads = self._scale_buffer[0][dense].shape[-1]
        loc64 = out_loc.to(torch.int64)
        self._scale_buffer[0][dense].reshape(-1, scale_heads).index_copy_(0, loc64, ks)
        self._scale_buffer[1][dense].reshape(-1, scale_heads).index_copy_(0, loc64, vs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
