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
    ) -> None:
        # Q8_0 KV cache: store the paged K/V as int8 values + a per-(token, head)
        # fp32 scale (llama.cpp's Q8_0) instead of fp16. Halves KV VRAM. The value
        # buffer is int8 and a separate scale buffer holds the fp32 scales; readers
        # dequantize (value * scale) before attention, so compute stays fp16.
        self._q8 = bool(q8)
        self._fp16_dtype = dtype
        if self._q8:
            self._value_dtype = torch.int8
            self._scale_dtype = torch.float32
        else:
            self._value_dtype = dtype
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
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=self._value_dtype,
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
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=self._value_dtype,
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
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config, kv_q8: bool = False, **kwargs) -> tuple[int, int, int, int]:
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
        return per_token * config.page_size, 0, config.page_size, 0

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
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        kv_bytes = int(buf.numel() * buf.element_size())
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
    def compute_dtype(self) -> torch.dtype:
        """The dtype attention math runs in (fp16/bf16) -- the dequant target."""
        return self._fp16_dtype

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Q8_0: int8 value + per-(token, head) fp32 scale from an fp16/bf16 tensor.

        ``x`` is ``[tokens, kv_heads, head_dim]``. The scale is ``max|x| / 127`` per
        (token, head); the value is ``round(x / scale)`` clamped to int8. A zero row
        (all-zero key) gets scale 1.0 to avoid divide-by-zero (its value is 0 anyway).
        """
        x = x.to(torch.float32)
        amax = x.abs().amax(dim=-1, keepdim=True)  # [tokens, kv_heads, 1]
        scale = (amax / 127.0).clamp_min(1e-8)
        q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return q, scale.squeeze(-1).to(torch.float32)

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
        # value (int8) and the scale (fp32) into their respective slabs.
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
        vq, vs = self._quantize(vf)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=kq.reshape(-1, kvh * D),
            v=vq.reshape(-1, kvh * D),
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
