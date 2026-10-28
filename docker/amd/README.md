# AMD Radeon (ROCm)

Verified with FlagTree as the Triton compiler on **AMD Radeon AI PRO R9700** (gfx1201, 32 GB)
and **AMD Radeon Pro W7900** (gfx1100, 48 GB). The software stack is ROCm 7.2.4 and PyTorch 2.10,
inside the `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` container.

| GPU | Model | Parallelism | Notes |
|-----|-------|-------------|-------|
| R9700 | Qwen3.5-9B | TP=1, TP=2 (FlagCX) | |
| R9700 | Qwen3.6-27B | TP=2 (FlagCX) | `--mem-fraction-static 0.92` |
| R9700 | Qwen3.6-35B-A3B-FP8 | TP=2 (FlagCX) | needs the FlagTree fix below |
| R9700 | Qwen3.6-35B-A3B | TP=4 (FlagCX) | |
| W7900 | Qwen3.5-9B | TP=1 | |
| W7900 | Qwen3.6-35B-A3B-FP8 | TP=1 | `--mem-fraction-static 0.9`; ~3.5 h kernel compile on first start |

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
is reused and pinned rather than replaced. It then installs:

- **[FlagTree](https://github.com/flagos-ai/FlagTree)** (FlagOS's Triton, AMD backend) as the
  `triton` package. By default this is the published `flagtree===0.7.0+amd3.6` wheel. Set
  `FLAGTREE_SRC=/path/to/FlagTree` to build it from a source checkout instead, or point
  `FLAGTREE_SRC` at an already-built `flagtree-*.whl`. If the container ships no ROCm Triton,
  the build that torch's metadata requires is installed behind the venv, so pip can resolve
  torch-dependent packages. FlagTree still provides `import triton`.
- **SGLang 0.5.11 with `--no-deps`**, plus its pure-Python runtime dependencies. SGLang's default
  dependency set would pull in the CUDA PyTorch stack. `apache-tvm-ffi` is required: SGLang
  JIT-compiles some HIP kernels through it, and these work on gfx1201 and gfx1100.
- **Import-only stubs for `aiter` and `sgl_kernel`** ([fl_stub.py](./fl_stub.py)). SGLang imports
  both at module level on ROCm, but neither supports these GPUs: `aiter` rejects gfx1201 on
  import, and `sgl_kernel`'s ROCm build targets only gfx942/gfx950. Any call into a stub raises. The
  kernels SGLang would otherwise take from them are served by FlagGems, by SGLang's own Triton
  kernels, or by this plugin's `vendor/amd` backend.
- **FlagGems v5.3.0** with one fix (flagos-ai/FlagGems#7908, also part of #6530): its `_amd`
  backend needs `triton_extra_name="hip"`. Without it, libdevice math ops such as `pow`, `tanh`
  and `erf` resolve to CUDA libdevice, and the AMD backend rejects their `__nv_*` calls. SGLang
  hits this at startup through RoPE's `base ** (arange / dim)`.
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

- **FP8 models need a FlagTree with flagos-ai/FlagTree#1456.** FlagTree 0.7.0's
  `TritonAMDGPUCanonicalizePointers` pass asserts that both arms of a pointer `arith.select` carry
  the same attributes. SGLang's FP8 block GEMM (`_w8a8_block_fp8_matmul`) breaks that assumption
  at small batch sizes, and the compiler aborts during graph capture. Until the fix is released,
  build FlagTree from source with it applied (`FLAGTREE_SRC`). bf16 models are not affected.
- **FP8 on gfx1100 (W7900) compiles very slowly the first time.** RDNA3 has no FP8 hardware, so
  FP8 kernels convert in software. On the resulting code, LLVM's AMDGPU VOPD pairing
  (`VOPDPairingMutation`) takes about 25 minutes per FP8 GEMM/MoE kernel variant. ROCm Triton
  behaves the same way. The first Qwen3.6-35B-A3B-FP8 start took about 3.5 h, almost all of it
  in graph capture. Compiled kernels are cached in `~/.triton/cache`. For that first start, pass
  `--watchdog-timeout` with a large value and `--skip-server-warmup`, so SGLang does not abort
  while kernels compile.
- `gemma_rms_norm` runs on the PyTorch reference backend.
- The FP8 block GEMM uses SGLang's default Triton config. No tuned config for this GPU exists yet,
  so performance is not optimized.
- With FlagCX >= 0.14, the FlagCX communicator needs the wrapper-path fix from
  flagos-ai/sglang-plugin-FL#156. Without it, collectives silently fall back to
  `torch.distributed`.
