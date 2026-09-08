# MUSA empty runtime with SGLang 0.5.18

This setup upgrades the Moore Threads empty image used for Qwen3.6 while
retaining its MUSA 4.3.5, torch/torch_musa 2.9.0, Triton 3.2.0, FlagGems
5.3.0rc2, MATE/flash_attn_3 0.2.1 and FlagCX 0.13.0 stack. The optional
MUSA `sglang-kernel` 0.4.2 wheel is already present in this image and remains
necessary for the vendor MoE implementation. This is the document's hybrid
empty configuration, not a claim that every model runs without vendor kernels.

## Installation

Use `docker/mthreads/empty-0.5.18.containerfile`, or upgrade an isolated
container from its `BASE_IMAGE` manually:

```bash
export PATH=/root/.virtualenvs/sglang-0.5.6/bin:$PATH
export SGLANG_BUILD_RUST_EXTS=none
git clone --depth 1 --branch v0.5.18 https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_other.toml pyproject.toml
printf 'torch==2.9.0\ntorch_musa==2.9.0\ntriton==3.2.0\n' > /tmp/musa-constraints.txt
python -m pip install -c /tmp/musa-constraints.txt '.[srt_empty]'
python -m pip install --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0
cd /path/to/sglang-plugin-FL
python -m pip install --no-deps .
```

SGLang 0.5.18 includes `srt_empty`; do not apply the 0.5.11 empty patches.
It imports xgrammar and compressed-tensors during server startup even for
unquantized models, so these are installed explicitly without their PyTorch
dependency chain. Rust extensions are disabled for this text/image setup.

The image also contains an older system-level `sglang_fl`. A regular install
into the inference virtual environment takes precedence. For editable
development, explicitly put the current plugin checkout on `PYTHONPATH` and
check `sglang_fl.__file__` before starting the server.

## Serving

Select free physical GPU IDs with `MUSA_VISIBLE_DEVICES` before launch.

```bash
export FLAGCX_PATH=/sgl-workspace/FlagCX
export TORCH_COMPILE_DISABLE=1
export SGLANG_MUSA_FP32_TP_ALLREDUCE=1
export SGLANG_FL_PER_OP='silu_and_mul=flagos;mrotary_embedding=flagos;topk=reference;gemma_rms_norm=reference;fused_moe=vendor;chunk_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule_packed_decode=vendor'

python -m sglang.launch_server \
  --model-path /models/Qwen3.6-27B --tp-size 4 \
  --host 127.0.0.1 --port 30000 \
  --attention-backend fa3 --sampling-backend pytorch \
  --page-size 1 --disable-radix-cache --trust-remote-code \
  --context-length 4096 --chunked-prefill-size 2048 \
  --max-running-requests 8 --mem-fraction-static 0.65 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full --cuda-graph-max-bs 4
```

The first request may take several minutes while MATE compiles attention
kernels. Keep installations and server/test commands in named `tmux`
sessions on remote machines.

Use `/models/Qwen3.6-35B-A3B` to run the MoE model with the same settings.
For eager decode, replace the last line with
`--cuda-graph-backend-decode disabled`.

## Compatibility changes

- Unbridged fused operators fall back to `forward_musa`; FL registrations
  retain priority over the native fallback.
- Recurrent FLA preserves the old state-indexing arguments on kernels that
  accept them. New kernels omit unset arguments and reject unsupported
  nonempty indexing requests instead of silently changing their semantics.
- Triton 3.2 can resolve the two CUDA PDL names in disabled constexpr branches
  of upstream GDN normalization/convolution kernels. A live PDL call still
  fails compilation with an explicit MUSA error. Existing symbols are preserved.
- The vision FA3 entry point binds to the installed MUSA varlen implementation.

Run the targeted regression checks in the inference environment:

```bash
python -m pytest -q \
  tests/unit_tests/platform/test_musa_sglang_compat.py \
  tests/unit_tests/platform/test_musa_gated_layernorm.py \
  tests/unit_tests/platform/test_musa_triton_compat.py \
  tests/unit_tests/dispatch/test_base_fused_op_registration.py \
  tests/unit_tests/distributed/test_communicator_hooks.py
```

## Validation (2026-09-08)

Validated on four MTT S5000 80 GB GPUs using the base image above and the
unmodified upstream SGLang v0.5.18 source. FlagGems ATen replacement and FL
fused-op dispatch were enabled throughout model validation.

| Case | Result |
| --- | --- |
| Qwen3-0.6B, TP1, eager | Sequential + four concurrent requests; 16 output tokens each |
| Qwen3.6-27B, TP4, eager | Text, image, four concurrent chat requests correct |
| Qwen3.6-27B, TP4, decode graph | Captured batch sizes 1/2/4 on all ranks; text/image/concurrent chat correct; sequential + four concurrent 64-token decode requests passed with graph replay |
| Qwen3.6-35B-A3B, TP4, decode graph | Same graph, chat and 64-token decode checks passed using vendor MoE |
| FlagCX, four ranks | Active communicator asserted; FP32/BF16 all-reduce matched expected sums on every rank |
| Regression suite above | 37 passed, including 16 GPU normalization cases and rejection of live PDL |
| Packaging | Plugin wheel built successfully |

Chat checks asked for France's capital and the color of a red image, and
verified `Paris` / `red` in the actual responses, with HTTP 200 and positive
completion-token counts. The 64-token checks forced continued decoding to
exercise graph replay beyond the short factual answers.

Cross-node TP/PP was not tested because the second host's GPUs were occupied.
Prefill graphs, speculative decoding, audio and performance benchmarking are
outside this validation. The Dockerfile mirrors the manual setup; a complete
Docker image build was not run.
