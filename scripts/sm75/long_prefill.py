#!/usr/bin/env python
"""Precise long-prefill A/B helper: sends a fixed N-content-token prompt (max_tokens=1)
and reports TTFT (prefill proxy). Deterministic prompt so triton vs cutlass compare
like-for-like. Usage: long_prefill.py <port> <n_content_tokens> <label>"""
import json, sys, time, urllib.request
from transformers import AutoTokenizer

port, n, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tok = AutoTokenizer.from_pretrained("/home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4", trust_remote_code=False)
words = [f"tok{i:06d}" for i in range(n + 64)]
text = " ".join(words)
ids = tok.encode(text, add_special_tokens=False)[:n]
while len(ids) < n:
    ids += tok.encode(" pad", add_special_tokens=False)
ids = ids[:n]
full = tok.apply_chat_template([{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}],
                               tokenize=True, add_generation_prompt=True,
                               chat_template_kwargs={"enable_thinking": False})
if hasattr(full, "input_ids"):
    full = full.input_ids
if isinstance(full, (list, tuple)) and full and isinstance(full[0], (list, tuple)):
    full = full[0]
full = [int(t) for t in full]
payload = {"model": "Qwen3.6-35B-A3B-NVFP4",
           "messages": [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}],
           "temperature": 0.0, "top_p": 1.0, "max_tokens": 1, "stream": False,
           "seed": 1234, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=1200) as r:
    obj = json.load(r)
dt = (time.perf_counter() - t0) * 1000.0
usage = obj.get("usage", {})
print(f"[{label}] port={port} full_tokens={len(full)} prompt_tokens={usage.get('prompt_tokens')} "
      f"TTFT={dt:.0f} ms  ({len(full)/ (dt/1000.0):.0f} tok/s prefill)")
