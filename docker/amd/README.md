# AMD Radeon (ROCm)

Verified on **AMD Radeon AI PRO R9700** (gfx1201, 32 GB), ROCm 7.2.4, PyTorch 2.10, inside the
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` container.

| Model | Parallelism | Status |
|-------|-------------|--------|
| Qwen3.5-9B | TP=1, TP=2 (FlagCX) | Verified |
| Qwen3.6-27B | TP=2 (FlagCX) | Verified (`--mem-fraction-static 0.92`) |
| Qwen3.6-35B-A3B-FP8 | TP=2 (FlagCX) | Verified |
| Qwen3.6-35B-A3B (bf16) | TP=4 | Not verified yet (67 GB of weights do not fit in 2 × 32 GB) |

## Setup

```bash
git clone https://github.com/flagos-ai/sglang-plugin-FL
bash sglang-plugin-FL/docker/amd/setup_env.sh /path/to/workdir
. /path/to/workdir/venv/bin/activate

# FlagCX for tensor-parallel communication
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX && make USE_AMD=1 && export FLAGCX_PATH="$PWD"
```

`setup_env.sh` creates a venv on top of the container's Python, so the container's ROCm `torch`
and `triton` are reused and pinned rather than replaced. It then installs:

- **SGLang 0.5.11 with `--no-deps`**, plus its pure-Python runtime dependencies. SGLang's default
  dependency set would pull in the CUDA PyTorch stack. `apache-tvm-ffi` is required: SGLang
  JIT-compiles some HIP kernels through it, and these work on gfx1201.
- **Import-only stubs for `aiter` and `sgl_kernel`** ([fl_stub.py](./fl_stub.py)). SGLang imports
  both at module level on ROCm, but neither supports gfx1201: `aiter` rejects the architecture on
  import, and `sgl_kernel`'s ROCm build targets gfx942/gfx950. Any call into a stub raises. The
  kernels SGLang would otherwise take from them are served by FlagGems, by SGLang's own Triton
  kernels, or by this plugin's `vendor/amd` backend.
- **FlagGems v5.3.0** with one fix: its `_amd` backend needs `triton_extra_name="hip"`. Without it,
  FlagGems selects CUDA libdevice, and ROCm Triton rejects its `__nv_*` calls (for example in `pow`).
- **This plugin** (editable).

## Run

```bash
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero
export FLAGCX_PATH=/path/to/FlagCX

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --tp 2 --port 30000 \
    --attention-backend triton \
    --disable-piecewise-cuda-graph \
    --disable-custom-all-reduce
```

`--disable-custom-all-reduce` skips SGLang's default AMD custom all-reduce, which needs `aiter`.
Without the flag, SGLang logs a warning and falls back on its own. Collectives go through FlagCX
either way.

## What the `vendor/amd` backend provides

- `fused_moe`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`,
  `fused_recurrent_gated_delta_rule_packed_decode`: these reuse SGLang's own Triton kernels, the
  same ones the CUDA backend uses.
- `patch.py`: in SGLang's Triton fused-MoE runner, `moe_align_block_size`, `silu_and_mul` and
  `gelu_and_mul` come from `sgl_kernel`. On AMD architectures that `sgl_kernel` does not support,
  the patch routes them to FlagGems. gfx942/gfx950 keep using `sgl_kernel`.

## Known limitations

- `gemma_rms_norm` runs on the PyTorch reference backend.
- The FP8 block GEMM uses SGLang's default Triton config. No tuned config for this GPU exists yet,
  so performance is not optimized.
- With FlagCX >= 0.14, the FlagCX communicator needs the wrapper-path fix from
  flagos-ai/sglang-plugin-FL#156. Without it, collectives silently fall back to
  `torch.distributed`.
