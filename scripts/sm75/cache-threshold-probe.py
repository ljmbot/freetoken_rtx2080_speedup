#!/usr/bin/env python3
"""Pin the hybrid-radix cache hit threshold: send each length's prompt twice,
read cached_tokens on the 2nd. Writes one line per (length, send) to stdout."""
import httpx, time, json, sys
url = "http://127.0.0.1:1920/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-NVFP4"

def send(msgs, max_tokens=1):
    t0 = time.perf_counter(); ttft = None; cached = 0; ptok = None
    with httpx.stream("POST", url, json={"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            line = (line or "").strip()
            if line.startswith("data:"):
                d = line[5:].strip()
                if d == "[DONE]": break
                try: o = json.loads(d)
                except: continue
                if o.get("usage"):
                    ptok = o["usage"].get("prompt_tokens")
                    cached = (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                ch = o.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
    return (ttft or 0) * 1000, cached, ptok

for nwords in (1000, 2000, 3000, 4000, 6000, 8000):
    words = " ".join(f"w{i:06d}" for i in range(nwords))
    msgs = [{"role": "user", "content": words}]
    t1, c1, p1 = send(msgs)
    t2, c2, p2 = send(msgs)
    speedup = f"{t1/t2:.0f}x" if t2 > 1 else "-"
    line = f"words={nwords} ptok={p2} | 1st ttft={t1:.0f}ms cached={c1} | 2nd ttft={t2:.0f}ms cached={c2} -> {speedup}"
    print(line, flush=True)
