#!/usr/bin/env python3
"""CLEAN boundary confirmation. FRESH unique-base prompts (no prefix sharing),
each sent twice. Under cap -> 2nd should HIT; at/over cap -> 2nd MISS.
max_prefill_length = 2048 (full tokens)."""
import httpx, time, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4', trust_remote_code=False)
url = "http://127.0.0.1:1920/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-NVFP4"

def content_ids(tag, n):
    bw = []; i = 0
    while len(bw) < n + 64:
        bw.append(f"{tag}{i:06d}"); i += 1
    text = " ".join(bw)
    ids = tok.encode(text, add_special_tokens=False)[:n]
    if len(ids) < n:
        pad = tok.encode(" pad", add_special_tokens=False)
        while len(ids) < n:
            ids.extend(pad)
        ids = ids[:n]
    return ids

def send(msgs):
    payload = {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0.0,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); ttft = None; cached = 0
    with httpx.stream("POST", url, json=payload, timeout=600) as r:
        for line in r.iter_lines():
            line = (line or "").strip()
            if line.startswith("data:"):
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:
                    continue
                if o.get("usage"):
                    cached = (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                ch = o.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
    return (ttft or 0) * 1000, cached

# unique base per length; full ~= content + 10
for tag, n in (("UNDER15", 1500), ("UNDER19", 1900)):  # full ~1510, ~1910 (both < 2048)
    ids = content_ids(tag, n)
    msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
    t1, c1 = send(msg)
    t2, c2 = send(msg)
    verdict = "HIT " if c2 > 0 else "MISS"
    print(f"{verdict} {tag} content={n} (full~{n+10}): 1st ttft={t1:.0f}ms cached={c1} | 2nd ttft={t2:.0f}ms cached={c2}", flush=True)
