# flashinfer-shim

On-disk `flashinfer` import-face stub for CUDA-alias vendors that have no
flashinfer build (iluvatar/corex today).

## Why this exists

Some vendors' torch is CUDA-alias: `torch.version.cuda` is set, so sglang's
`is_cuda()` is True even though the platform is not NVIDIA. Three module-level
imports sit under a bare `if is_cuda` with no availability check, and the first
is fatal on every serve:

```
ServerArgs -> model_config -> quantization -> fp8 -> fp8_utils
    from flashinfer import bmm_fp8 as _raw_bmm_fp8_batched
ModuleNotFoundError: No module named 'flashinfer'
```

## Why a package on disk instead of a plugin-side `sys.modules` seed

sglang runs its scheduler in a *spawned* worker. That worker imports
`sglang.srt.managers.scheduler` — and through it the quantization package — at
module-import time, before any plugin is loaded. A stub registered from the
plugin is therefore never in place early enough; the name has to resolve from
disk. The same reasoning already made `sgl_kernel_npu` an on-disk stub for
ascend (see `addon/sgl-kernel-shim`).

## Scope

Stubbed symbols are an explicit allowlist; every other name raises
AttributeError, so `hasattr(...)`-style capability probes in sglang keep
answering honestly rather than being fooled into a CUDA path. Every stub *call*
raises RuntimeError — nothing silently computes a wrong result.

`is_flashinfer_available()` still has to be False for this platform (it selects
the sampling and attention backends, and it gates on
`SGLANG_IS_FLASHINFER_AVAILABLE` before touching `find_spec`), so images that
install this wheel also set that env var.

## Install scope

Publish to the vendor index of the backends that need it and list it in their
`deps_app` — a stub flashinfer must never shadow a real one.

## Build

```bash
bash build.sh [outdir]
```
