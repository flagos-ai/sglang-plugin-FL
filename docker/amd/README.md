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
| R9700 x2 nodes | Qwen3.6-35B-A3B | TP=2 x PP=2 (FlagCX) | 2 GPUs per node, see "Multi-node" |
| R9700 x2 nodes | Qwen3.6-27B | TP=2 x PP=2 (FlagCX) | 2 GPUs per node, see "Multi-node" |
| W7900 | Qwen3.5-9B | TP=1 | |
| W7900 | Qwen3.6-35B-A3B-FP8 | TP=1 | `--mem-fraction-static 0.9`; ~3.5 h kernel compile on first start |

## Setup

```bash
git clone https://github.com/flagos-ai/sglang-plugin-FL
bash sglang-plugin-FL/docker/amd/setup_env.sh /path/to/workdir
. /path/to/workdir/venv/bin/activate

# FlagCX v0.13.0 for tensor-parallel communication (see "Known limitations" for other versions)
git clone --branch v0.13.0 --depth 1 https://github.com/flagos-ai/FlagCX.git
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
  both at module level on ROCm, but both mainly target AMD Instinct GPUs and neither is installed
  here. Any call into a stub raises. The kernels SGLang would otherwise take from them are served
  by FlagGems, by SGLang's own Triton kernels, or by this plugin's `vendor/amd` backend.
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
    --disable-piecewise-cuda-graph \
    --disable-custom-all-reduce
```

`--disable-custom-all-reduce` skips SGLang's default AMD custom all-reduce, which needs `aiter`.
Without the flag, SGLang logs a warning and falls back on its own. Collectives go through FlagCX
either way.

The attention backend defaults to SGLang's built-in `triton` backend on AMD (Triton kernels
compiled by FlagTree); pass `--attention-backend` to override it. SGLang's AMD-specific `aiter`
attention backend targets Instinct GPUs and is not used.

## Vendor-specific operator libraries

None are required on Radeon GPUs. AMD's operator libraries, `aiter` and the ROCm build of
`sgl_kernel`, mainly target AMD Instinct GPUs. `sgl_kernel`'s `setup_rocm.py` only supports gfx942
(MI300/MI325) and gfx950 (MI350). Some `aiter` operators do run on Radeon GPUs, such as the Triton
kernels behind FlashAttention-2's Triton backend (see below), but Instinct remains its primary
platform: the PyPI release we tried (amd-aiter 0.1.7) refuses gfx1201 at import, while newer
commits import on both Radeon GPUs. On Radeon, the operators SGLang would take from `aiter` and
`sgl_kernel` are served by FlagGems, by SGLang's own Triton kernels, or by this plugin's
`vendor/amd` backend. The stubs described above exist because SGLang imports both packages
unconditionally on ROCm and neither is installed here.

## Attention kernels and FlashAttention

SGLang's `triton` attention backend runs FlashAttention-style tiled attention written in Triton:
online softmax over KV blocks, and flash-decoding (split-KV) for decode. It is not the
FlashAttention library. In Qwen3.6, one layer in four is full attention and uses this backend;
the others are Gated DeltaNet layers, which use the separate Triton kernels from `vendor/amd`.

FlashAttention-2 also has a Triton backend for AMD (CDNA and RDNA GPUs), built on aiter's Triton
kernels (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, see the FlashAttention README). In a
standalone test with FlagTree it ran correctly on both Radeon GPUs (bf16, head_dim 256, GQA;
causal prefill, varlen and KV-cache decode). For single-sequence causal prefill with Qwen3.6
shapes, it was about as fast as SGLang's Triton kernel on R9700 and 2x to 3.6x faster on W7900
for 512 to 8192 tokens. SGLang has no FlashAttention-2 backend today (its `fa3` backend calls
the FA3 kernels in `sgl_kernel`), so using it needs a new attention backend. That is not part of
this PR.

## Running the repository examples

The Qwen3.6 examples run unchanged. Each one answers two text prompts and four image questions and
validates the answers:

```bash
cd sglang-plugin-FL/examples
TP_SIZE=4 python qwen3_6_27b_offline_inference.py
TP_SIZE=4 python qwen3_6_35b_a3b_offline_inference.py
```

Verified on 4x R9700 with FlagCX v0.13.0: both scripts finish with `All validations passed.`, and
SGLang's tensor-parallel collectives go through FlagCX. Qwen3.6-27B needs `TP_SIZE=4` here because
its bf16 weights do not fit on one 32 GB card.

## Multi-node (TP x PP)

The multi-node examples run unchanged. Each node runs the same script with a different `--role`; start
the master first. Verified with two R9700 machines (2 GPUs each, plain Ethernet, no InfiniBand), TP=2
x PP=2, FlagCX v0.13.0:

```bash
# both nodes
export SGLANG_FL_DIST_BACKEND=flagcx FLAGCX_PATH=/path/to/FlagCX
export SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export GLOO_SOCKET_IFNAME=<nic> NCCL_SOCKET_IFNAME=<nic>      # the NIC that carries the node IP
export HIP_VISIBLE_DEVICES=<two free GPUs>

# node 0
python qwen3_6_35b_a3b_multinode.py --role master --master-addr <node0 ip> --tp 2 --pp 2
# node 1
python qwen3_6_35b_a3b_multinode.py --role worker --master-addr <node0 ip> --tp 2 --pp 2
```

`qwen3_6_27b_multinode.py` works the same way. Both scripts finish with `ALL TESTS PASSED` (text,
longer generation, 4 and 32 concurrent text requests, image questions, 8 concurrent image requests).
SGLang's TP all-reduce / all-gather and the PP send / recv go through FlagCX: the counts on the two
nodes match (for 27B, 354 sends on one node against 354 receives on the other), and nothing falls back
to `torch.distributed`. The network is plain Ethernet, so this run checks function, not throughput.

## What the `vendor/amd` backend provides

- `fused_moe`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`,
  `fused_recurrent_gated_delta_rule_packed_decode`: these reuse SGLang's own Triton kernels, the
  same ones the CUDA backend uses.
- `patch.py`: in SGLang's Triton fused-MoE runner, `moe_align_block_size`, `silu_and_mul` and
  `gelu_and_mul` come from `sgl_kernel`. On AMD architectures that `sgl_kernel` does not support,
  the patch routes them to FlagGems. gfx942/gfx950 keep using `sgl_kernel`.

## Known limitations

- **FP8 models need a FlagTree with flagos-ai/FlagTree#1456 (merged, not in a release yet).** FlagTree 0.7.0's
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
- **FlagCX version.** Use FlagCX v0.13.0, as in the setup above: the plugin finds its Python wrapper
  at `plugin/interservice/flagcx_wrapper.py`, and TP collectives go through FlagCX. FlagCX >= 0.14
  moved the wrapper to `src/flagcx/api.py`; with it, the plugin needs the wrapper-path fix from
  flagos-ai/sglang-plugin-FL#156, otherwise it only logs a warning and silently falls back to
  `torch.distributed`. The v0.13.0 wrapper has no bool or FP8 dtype, so bool/FP8 tensors in the
  byte-moving collectives need flagos-ai/sglang-plugin-FL#161; plain TP does not send them. v0.13.0's
  HIP adaptor also lacks `eventElapsedTime` (added in flagos-ai/FlagCX#679), so the FlagCX tuner
  (`FLAGCX_USE_TUNER=1`) cannot time collectives on AMD with it.
