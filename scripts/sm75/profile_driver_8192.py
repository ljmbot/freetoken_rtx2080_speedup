#!/usr/bin/env python3
"""Capture steady-state op tables for a single 8192-token chunk (and 4096).

Server: serve-profile-post.sh MAX_PREFILL=8192, FREETOKEN_PROFILE=1 AFTER=4 COUNT=3.
Internal warmup = forwards 1,2 (80/128). Plan:
  f3 warm-2048 (skip)
  f4 warm-8192 (skip)
  f5 t8192-a   (CAPTURED)
  f6 t8192-b   (CAPTURED)
  f7 t4096     (CAPTURED)
"""
import json, sys, time
import httpx
from transformers import AutoTokenizer
sys.path.insert(0, "/home/lmeek/src/freetoken-sm75-native-prefill/scripts/sm75")
from profile_driver_post import build_content_token_ids, stream_chat, wait_ready

MODEL = "/home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4"
HOST, PORT = "127.0.0.1", 1920

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
base = f"http://{HOST}:{PORT}"
url = base + "/v1/chat/completions"
if not wait_ready(base + "/health"):
    print("!! not ready", file=sys.stderr); sys.exit(1)
print("server ready", file=sys.stderr)

client = httpx.Client()
plan = [
    ("warm-2048", 2048, "PW1", "pw1"),
    ("warm-8192", 8192, "PW2", "pw2"),
    ("t8192-a", 8192, "PT1", "pt1"),
    ("t8192-b", 8192, "PT2", "pt2"),
    ("t4096", 4096, "PT3", "pt3"),
]
out = []
for name, n, salt, wp in plan:
    ids = build_content_token_ids(tok, n, tok.encode(salt, add_special_tokens=False)[-1], word_prefix=wp)
    msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
    print(f"== {name} n={n} ==", file=sys.stderr, flush=True)
    try:
        ttft, total, n_out, usage = stream_chat(client, url, MODEL, msg, 1)
        row = {"name": name, "n": n, "ok": True,
               "ttft_ms": ttft * 1000 if ttft else None,
               "tok_per_s": (usage or {}).get("prompt_tokens") / ttft if (ttft and usage) else None}
    except Exception as e:
        row = {"name": name, "n": n, "ok": False, "error": repr(e)}
    out.append(row)
    print(json.dumps(row), file=sys.stderr, flush=True)
client.close()
with open("/home/lmeek/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/profile-8192/driver.json", "w") as f:
    json.dump(out, f, indent=2)
print("done", file=sys.stderr)
