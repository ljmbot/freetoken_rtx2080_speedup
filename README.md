<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## sm75 / 8 GB improvements (this fork)

This is a fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken),
cloned at its `sm75` branch (commit `35668da`, "fix(cuda): support Turing GPUs"
from [PR #24](https://github.com/FlashML-org/FreeToken/pull/24) by Lukas Rakauskas),
plus the sm75 work below on top. It runs
**RedHat (Neural Magic)
[Ornith-1.0-35B-NVFP4](https://huggingface.co/RedHatAI/Ornith-1.0-35B-NVFP4)**
(35B Qwen3.5-MoE, W4A4 NVFP4) on **8 GB Turing GPUs (RTX 2080/2080 Ti, sm75)**
with a **64K context window**. On an RTX 2080 we measured ~55 tok/s prefill,
~55 tok/s decode, and 7.4 GB peak VRAM. [launch_ornith.md](launch_ornith.md)
has the full command, every `--switch`/env var, and tuning notes.

Why Ornith is the model to run here: on the hard agent task "write JavaScript
that computes pi to 100 decimal places", Qwen3.6-35B-A3B-Q4 loops without
converging, while Ornith converges on both its variants (AEON and plain) —
verified by executing the generated code against true pi. If you care about
reliable multi-step math and coding on a Turing card, this is the pairing that
works.

What changed:

- **CUTLASS prefill attention** (`--attention-backend cutlass`): prefill uses
  PyTorch's CUTLASS memory-efficient SDPA, 10-13x faster than the Triton extend
  kernel at production geometry. If a long-context expansion won't fit in free
  VRAM, a built-in guard falls back to the Triton kernel.
- **Tensor-core prefill GEMMs on sm75** (`FREETOKEN_E1_CUBLAS=1`,
  `FREETOKEN_E3_CUBLAS=1`, `FREETOKEN_E3_GROUP=16`): dense and NVFP4-expert
  prefill GEMMs go through cuBLAS fp16 tensor cores, with grouped dequant sized
  to fit 8 GB. Does nothing on sm89+.
- **sm75-tuned GDN recurrence** (`FREETOKEN_GDN_SM75_TUNE=1`): the gated delta
  net recurrence is ~4.5x faster on sm75, with bit-identical output.
- **Q8 KV cache** (`--kv-q8`): int8 paged KV with per-head scales, dequantized
  in-kernel. KV memory is halved, which is what makes 64K fit on 8 GB.
- **Ornith model support**: loader, weight mapping, and model files for the
  compressed-tensors NVFP4 format (Qwen3.5-MoE). Set
  `FREETOKEN_LOAD_VISION=0` to skip the vision tower for text-only serving.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

The sm75 work in this fork is independent of [FreeToken-Kai](https://github.com/yuuki-net/FreeToken-Kai),
an unofficial fork that also targets Turing cards; it's worth a look if you want a tuned-settings
console on Ampere or Turing.

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
