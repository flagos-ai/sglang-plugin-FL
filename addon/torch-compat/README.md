# torch-compat-shim

On-disk compat layer for running sglang 0.5.18 on a torch < 2.8 build
(iluvatar/corex 4.4.0 today, whose SDK pins torch 2.7.1).

## Why this exists

sglang 0.5.18 imports two torch surfaces that torch < 2.8 does not have, on the
plain Qwen3 startup path:

| gap | where | torch 2.7 has |
|---|---|---|
| `torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`, `_cuda_endAllocateToPool` | `pynccl_allocator` | the same two operations as `_cuda_beginAllocateToPool` / `_cuda_endAllocateCurrentStreamToPool` |
| `torch.distributed._symmetric_memory` (import fails: no `_C._distributed_c10d._SymmetricMemory`) | `logits_processor` → `triton_symm_mem_ag` | nothing — it is an NVLink multicast all-gather |

The first is a rename, not a different operation: `pynccl_allocator` itself
chooses between the two spellings by version (`after_2_8_0`). The second is an
NVLink feature this platform never runs.

## Why nothing is called

Symmetric memory is off through sglang's own gate —
`pynccl_allocator.is_symmetric_memory_enabled()` returns
`get_exec().comm.enable_symm_mem`, False without real symmetric memory. So the
aliased entry points and the stubbed names are imported and never reached. The
stub keeps the flashinfer-shim discipline: stubbed symbols are an explicit
allowlist, every other name raises `AttributeError` (capability probes stay
honest) and every stub *call* raises `RuntimeError` — nothing silently computes
a wrong result.

## Why `sitecustomize` instead of a plugin patch

sglang runs its scheduler in a *spawned* worker, and that worker imports
`sglang.srt.managers.scheduler` — taking in both surfaces — before any sglang
plugin is loaded. A plugin-side patch is never in place early enough; the fix
must be on disk. Same ordering that made `addon/flashinfer-shim` an installed
package.

`site` imports `sitecustomize` at interpreter startup, so the patch is in place
in the worker too. The hook only wraps the import of the `sglang` package, so
the cost is paid by interpreters that import sglang and by no others. Both
patches are no-ops on torch >= 2.8.

## Install scope

Publish to the vendor index of the backends that need it and list it in their
`deps_app`. This shim exists only for platforms whose SDK pins torch < 2.8 —
installing it where torch is newer is harmless but pointless.

## Build

```bash
bash build.sh [outdir]
```
