# MUSA empty runtime with SGLang 0.5.18

This setup upgrades the Moore Threads empty image used for Qwen3.6 while
retaining its MUSA 4.3.5, torch/torch_musa 2.9.0,
MATE/flash_attn_3 0.2.1 and FlagCX 0.13.0 stack. Upgrade the compiler to
FlagTree `0.6.2a3+mthreads3.6` (Triton 3.6) and FlagGems to master commit
`01433e8304d78ef6c16ce76fe68e51e16c0b4d66`, the latest snapshot selected
on 2026-09-08. The snapshot package version below is a local build label,
not an upstream release number. The optional
MUSA `sglang-kernel` 0.4.2 wheel is already present in this image and remains
necessary for the vendor MoE implementation. This is the document's hybrid
empty configuration, not a claim that every model runs without vendor kernels.

## Installation

Use `docker/mthreads/empty-0.5.18.containerfile`, or upgrade an isolated
container from its `BASE_IMAGE` manually:

```bash
export PATH=/root/.virtualenvs/sglang-0.5.6/bin:$PATH
export SGLANG_BUILD_RUST_EXTS=none
python -m pip uninstall -y triton flagtree
python -m pip install --no-deps 'flagtree==0.6.2a3+mthreads3.6' \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
mkdir -p /opt/FlagGems
curl -fL https://codeload.github.com/flagos-ai/FlagGems/tar.gz/01433e8304d78ef6c16ce76fe68e51e16c0b4d66 \
  -o /tmp/flaggems.tar.gz
tar -xzf /tmp/flaggems.tar.gz --strip-components=1 -C /opt/FlagGems
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.4.0.dev20260908+g01433e830 \
  python -m pip install --no-deps -e /opt/FlagGems
python -m pip install 'packaging>=26.0' PyYAML==6.0.1 sqlalchemy==2.0.48
export PYTHONPATH=/opt/FlagGems/src:${PYTHONPATH:-}
git clone --depth 1 --branch v0.5.18 https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_other.toml pyproject.toml
printf 'torch==2.9.0\ntorch_musa==2.9.0\n' > /tmp/musa-constraints.txt
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

This FlagGems snapshot needs a source installation: its regular wheel omits
the `fused/DSA` namespace directory. Keep `/opt/FlagGems/src` first on
`PYTHONPATH`, because the base image also has an older system FlagGems that
can shadow an editable install. Check `flag_gems.__file__` and
`triton.__version__` in the actual inference Python before testing.

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
- MUSA's default FlagGems blacklist includes `broadcast_tensors`. The tested
  master snapshot mishandles zero-length dimensions, causing top-p sampling
  to crash when the top-k mask selects no tokens.
- Native `slice` handles the boolean buffers filled by SGLang 0.5.18's eager
  runner and preserves their view aliasing. The tested FlagGems snapshot
  rejects boolean slicing, which the original offline example exposed.
  If overriding `SGLANG_FL_FLAGOS_BLACKLIST`, include `broadcast_tensors`
  and `slice` alongside your other required exclusions.

Run the targeted regression checks in the inference environment:

```bash
python -m pytest -q \
  tests/unit_tests/platform/test_musa_sglang_compat.py \
  tests/unit_tests/platform/test_musa_gated_layernorm.py \
  tests/unit_tests/platform/test_musa_triton_compat.py \
  tests/unit_tests/platform/test_musa_sampling_mask.py \
  tests/unit_tests/dispatch/test_base_fused_op_registration.py \
  tests/unit_tests/distributed/test_communicator_hooks.py
```

## Additional model matrix

`tests/manual/musa_model_matrix.py` launches each model serially and checks
factual/arithmetic chat answers, sequential and four-concurrent 64-token
greedy decoding, four-concurrent top-p sampling, and decode graph capture
and replay. Qwen3.6 cases additionally check a red image. It records answers,
commands, versions, module locations and failures in JSON, with one server
log per model. These are serving smoke checks, not a model-quality benchmark.

With the serving environment above and a directory containing the model
subdirectories named in `CASES`, run inside `tmux`:

```bash
python tests/manual/musa_model_matrix.py \
  --model-root /models --output-dir /work/model-matrix \
  --large-model-tp 2 \
  --models phi4 gemma3 cohere qwen36_dense qwen36_moe
```

Phi-4-mini-instruct exercises `Phi3ForCausalLM`, rnj-1-instruct exercises
`Gemma3ForCausalLM`, and aya-23-8B exercises `CohereForCausalLM`. These are
architectures supported by upstream v0.5.18, beyond the Qwen examples;
this does not imply each architecture was first introduced in that release.

## Updated-stack validation (2026-09-08)

The matrix uses FlagTree `0.6.2a3+mthreads3.6` / Triton `3.6.0`, FlagGems
`01433e8304d78ef6c16ce76fe68e51e16c0b4d66` and unmodified SGLang `0.5.18`.
The actual imported FlagGems source location was checked. Runs used the
plugin's MUSA blacklist, including the new `broadcast_tensors` exclusion,
with FlagGems ATen replacement and FL fused-op dispatch enabled.

| Model | TP | Chat, 64-token decode, four-way greedy/sampling, graph replay |
| --- | --- | --- |
| Phi-4-mini-instruct | 1 | Passed; `Paris`, `221` |
| rnj-1-instruct (Gemma3) | 1 | Passed; `Paris`, `221` |
| aya-23-8B (Cohere) | 1 | Passed; `Paris`, `221` |
| Qwen3.6-27B | 2 | Passed; also identified the image as `red` |
| Qwen3.6-35B-A3B | 2 | Passed with vendor MoE; also identified the image as `red` |

All 40 regression tests listed above passed in this updated environment.
Two-rank FlagCX FP32/BF16 all-reduce also passed, with the communicator
explicitly checked to be active. The vendor MoE library reused its existing
tuning configuration because a Triton 3.6-specific configuration was absent;
no performance claim is made. The plugin wheel built successfully.

The additional runs used two available S5000 GPUs on one host. Other jobs
occupied the remaining devices; an eight-rank scheduler was still present
on the second host. Cross-node TP/PP has not been validated.

## Initial baseline validation (2026-09-08)

Before the FlagTree/FlagGems upgrade, validated on four MTT S5000 80 GB GPUs
with Triton 3.2.0 / FlagGems 5.3.0rc2 from the base image and the
unmodified upstream SGLang v0.5.18 source. FlagGems ATen replacement and FL
fused-op dispatch were enabled throughout model validation.

| Case | Result |
| --- | --- |
| Qwen3-0.6B, TP1, eager | Sequential + four concurrent requests; 16 output tokens each |
| Qwen3.6-27B, TP4, eager | Text, image, four concurrent chat requests correct |
| Qwen3.6-27B, TP4, decode graph | Captured batch sizes 1/2/4 on all ranks; text/image/concurrent chat correct; sequential + four concurrent 64-token decode requests passed with graph replay |
| Qwen3.6-35B-A3B, TP4, decode graph | Same graph, chat and 64-token decode checks passed using vendor MoE |
| FlagCX, four ranks | Active communicator asserted; FP32/BF16 all-reduce matched expected sums on every rank |
| Regression suite before adding the sampling-mask cases | 37 passed, including 16 GPU normalization cases and rejection of live PDL |
| Packaging | Plugin wheel built successfully |

Chat checks asked for France's capital and the color of a red image, and
verified `Paris` / `red` in the actual responses, with HTTP 200 and positive
completion-token counts. The 64-token checks forced continued decoding to
exercise graph replay beyond the short factual answers.

Cross-node TP/PP was not tested because the second host's GPUs were occupied.
Prefill graphs, speculative decoding, audio and performance benchmarking are
outside this validation. The Dockerfile mirrors the manual setup; a complete
Docker image build was not run.
