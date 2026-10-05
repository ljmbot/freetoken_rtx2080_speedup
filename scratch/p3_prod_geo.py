"""Phase 3 verification @ production geometry (L=40, E=256, S=640).
Confirms: production prefill (materialize_layer, --disable-moe-prefill-overlap)
warms ONLY the last layer's decode LRU map; the other 39 layers are cold.
"""
import torch
from freetoken.moe.offload_cache import OffloadMoeCache

L, E, S = 40, 256, 640
dev = torch.device("cuda")
cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=dev,
                        cache_policy="lru", prefill_overlap=False, quant_format="bf16")
src = {b: [torch.randn(E, 2, 2, dtype=torch.bfloat16, pin_memory=True) for _ in range(L)]
       for b in ("gate_up", "down")}
cache.set_bank_sources(src)

for lid in range(L):
    cache.materialize_layer(lid)

mapped = int((cache.slot_for_id >= 0).sum().item())
total = L * E
warm_layers = [lid for lid in range(L) if int((cache.slot_for_id[lid] >= 0).sum().item()) > 0]
print(f"Production geometry L={L} E={E} S={S}")
print(f"After full prefill: {mapped}/{total} (layer,expert) pairs mapped for decode")
print(f"  = {mapped/total*100:.1f}% of the decode address space")
print(f"  Warm layers: {warm_layers}  (count={len(warm_layers)} of {L})")
print(f"  Cold layers: {L - len(warm_layers)} of {L}")
