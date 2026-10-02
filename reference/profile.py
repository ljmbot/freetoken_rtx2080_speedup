import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Invoke exactly one representative prefill here
    run_one_prefill()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=40,
))
