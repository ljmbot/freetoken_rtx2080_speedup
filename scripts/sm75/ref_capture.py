#!/usr/bin/env python
"""Capture a deterministic (greedy) completion from the running server.
Usage: ref_capture.py <port> <outfile>
"""
import json, sys, urllib.request

port, outfile = sys.argv[1], sys.argv[2]
prompt = ("Explain in two sentences why the sky is blue, then give the "
          "capital of France. Be concise.")
payload = {
    "model": "Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 96,
    "stream": False,
    "seed": 1234,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    obj = json.load(r)
text = obj["choices"][0]["message"]["content"]
usage = obj.get("usage", {})
with open(outfile, "w") as f:
    f.write(text)
print(f"[port {port}] prompt_tokens={usage.get('prompt_tokens')} "
      f"completion_tokens={usage.get('completion_tokens')}")
print("--- output ---")
print(text)
