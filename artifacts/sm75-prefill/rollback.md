# Rollback — FreeToken SM75 native-prefill candidate

## What "rollback" means here

The candidate is an **independent installation** (`~/src/freetoken-sm75-native-prefill`)
running its own server on **loopback:1920**. The **original service**
(`~/src/freetoken-sm75`, bound to **0.0.0.0:8888**) was never modified and is the
default. Rollback = **stop the test server and keep using the original endpoint**.
No code rollback, dependency reinstall, or system CUDA change is required.

## Stop the candidate test server

```bash
# find and stop the candidate ft serve (loopback:1920)
pkill -f 'ft serve'                 # or: kill $(pgrep -f 'ft serve')

# confirm the port is free
ss -ltnp | grep ':1920' || echo '1920 free'
```

If the original service was paused (with permission) to give the GPU to the
uncontended benchmark, **resume it with its unchanged launch configuration**:

```bash
# original service (unchanged) — the user's normal launch
bash ~/src/freetoken-sm75/qwen.sh        # binds 0.0.0.0:8888
```

## Why no code/dependency rollback is needed

- The candidate lives in its own git clone, its own `.venv`, and its own cache dirs
  (`~/src/freetoken-sm75-native-prefill/.cache/candidate`). The original tree and
  venv are untouched.
- The two Phase E code paths are **env-gated** (`FREETOKEN_E1_CUBLAS`,
  `FREETOKEN_E3_CUBLAS`). Even if the candidate code were ever promoted, removing
  the env vars (or not setting them) returns it to the baseline Triton path — the
  new kernels are never compiled/dispatched unless the gate is on and the GPU is
  SM75-or-lower.
- The model weights are shared **read-only** from `~/Models`; nothing is written.

## If you want the candidate's code but not its new paths

Launch the candidate with the gates off (baseline behaviour on the new tree):

```bash
FREETOKEN_E1_CUBLAS=0 FREETOKEN_E3_CUBLAS=0 bash \
  ~/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/launch-candidate.sh
```

## Verify the original is serving

```bash
curl -s http://127.0.0.1:8888/health
```

## Provenance of the original (for reference)

- Original tree: `~/src/freetoken-sm75` @ `35668da` ("fix(cuda): support Turing GPUs").
- Original launch: `~/src/freetoken-sm75/qwen.sh` (0.0.0.0:8888), env from
  `~/freetoken-sm75-env.sh`. Captured verbatim in `launch-baseline.txt`.
