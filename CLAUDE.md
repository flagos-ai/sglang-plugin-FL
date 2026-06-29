# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also `AGENTS.md` for full setup, testing, and configuration details. This file focuses on architectural knowledge that requires reading multiple files to internalize.

## Plugin loading lifecycle (`sglang_fl/__init__.py`)

`load_plugin()` is the single entry point called by SGLang via setuptools entry_points. It runs these steps **in order**:

1. **Build config** — `_build_config()` merges env vars > YAML > platform defaults > code defaults
2. **Apply patches** — monkey-patches to sglang core. `per_rank_log`, `device_support`, `platform_stubs`, `model_runner` apply always; `dist_init`, `fused_moe` are gated on `torch.txda.is_available()`.
3. **FlagGems ATen** — `_setup_flaggems(config)` calls `flag_gems.enable()` with whitelist/blacklist filtering
4. **Dispatch init** — `_init_dispatch(config)` creates `SelectionPolicy`, sets it globally, forces `OpManager.ensure_initialized()` which registers all backends
5. **Dispatch AROUND hook** — intercepts `MultiPlatformOp.dispatch_forward()` with bridge routing (gated by `SGLANG_FL_OOT_ENABLED`)
6. **FLA + Rotary patches** — `fla_patch.py` replaces module-level FLA functions; `rotary_patch.py` fixes MUSA RotaryEmbedding stomp
7. **Communicator hooks** — AROUND hooks on `GroupCoordinator` for all_reduce/all_gather/etc.
8. **Banner** — prints activation summary (rank 0 only)

`activate_platform()` runs separately (earlier) to provide `PlatformFL` for device identity.

## Config priority chain (two-stage merging)

**Stage 1** (`sglang_fl/dispatch/config/utils.py:get_effective_config()`):
```
SGLANG_FL_CONFIG YAML > platform auto-detect YAML (nvidia.yaml/ascend.yaml/etc.) > empty dict
```

**Stage 2** (`sglang_fl/__init__.py:_build_config()`):
```
SGLANG_FL_* env vars > Stage 1 result > code defaults
```

Important: env vars win over YAML for **every** field — not just `prefer`. The dispatch policy layer (`policy.py:_policy_from_env()`) applies the same chain independently.

## Dispatch architecture — three-layer resolution

```
AROUND hook on dispatch_forward()
  → bridge function translates SGLang params → standardized op signature
    → call_op("op_name", obj, ...)
      → OpManager.resolve() or call() with fallback
        → SelectionPolicy.get_default_order() → ["flagos", "vendor", "reference"]
          → OpRegistry snapshot filtered by availability + vendor allow/deny
            → match_token() resolves each order token to concrete OpImpl
```

**Cache key**: `(op_name, policy_fingerprint, epoch)` — epoch bumps on policy change or after fork.

**Fallback mode** (default: `SGLANG_FL_STRICT=1` means fallback **enabled** — counterintuitive naming): `OpManager.call()` tries each candidate in order, tracks failures in `_failed_impls`, and skips known-bad impls on subsequent calls.

## Two Backend ABCs (not one)

- `sglang_fl/dispatch/backends/Backend` — legacy ABC used by vendor backends (`vendor/{cuda,ascend,musa,txda}/`). Methods take `(self, obj, ...)`.
- `sglang_fl/dispatch/ops/FLBackendBase` — newer ABC with `@abstractmethod` signatures documented in `ops.py`. The flagos and reference backends subclass this.

They serve the same purpose but aren't unified yet. Vendor backends register via `register_ops.py` → `OpImpl(fn=backend.method, ...)`, not by subclassing `FLBackendBase`.

## Bridge layer responsibilities

Each bridge in `sglang_fl/dispatch/bridge/` translates SGLang-specific calling conventions to dispatch-standard signatures:

| Bridge | SGLang-specific handling |
|--------|--------------------------|
| `silu_and_mul_bridge` | Identity (1:1 mapping) |
| `rms_norm_bridge` | Merges `post_residual_addition` into `residual` before dispatch |
| `gemma_rms_norm_bridge` | Like rms_norm but handles Gemma-specific weight/variance_epsilon access |
| `rotary_embedding_bridge` | Falls through to `forward_native` for `fused_set_kv_buffer_arg`; extracts cos/sin from `self.cos_sin_cache`; reshapes query/key from `[B, NH]` to `[B, N, H]`; splits rotary/pass-through dimensions when `rotary_dim < head_size` |
| `mrotary_embedding_bridge` | Handles multi-section rotary (mrope) with section-aware position offsets |
| `topk_bridge` | Identity (1:1 mapping) — passes `num_token_non_padded` and `expert_location_dispatch_info` as kwargs |
| `fused_moe_bridge` | Identity (1:1 mapping) — passes `layer` and `dispatch_output` |
| `fla_chunk_bridge` / `fla_fused_recurrent_bridge` / `fla_packed_decode_bridge` | FLA-specific parameter translation; originals saved in `fla_patch._originals` for backends that need them |

## FLA patching — module-level function replacement

Unlike other ops (which go through `MultiPlatformOp.dispatch_forward`), FLA ops are **module-level functions** in `sglang.srt.layers.attention.fla.*`. `fla_patch.py` replaces them at import time with bridge functions. Original functions are preserved in `fla_patch._originals` dict so backends can call the unpatched versions without recursion.

The patch also replaces imports in `gdn_triton.py` (the actual call site for some FLA functions) to prevent cached module references from bypassing the bridge.

## TXDA (TsingMicro) vendor backend

The `vendor/txda/` backend (`TxdaBackend`) is a full vendor backend for TsingMicro hardware. It implements **all 10 ops**: silu_and_mul, rms_norm, gemma_rms_norm, rotary_embedding, mrotary_embedding, topk, fused_moe, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_packed_decode. Detection: `hasattr(torch, "txda") and torch.txda.is_available()`. Vendor name registered as `"tsingmicro"`.

Unlike other platform adaptations that go through FlagGems, TXDA relies heavily on the **patches subsystem** (above) to make SGLang core compatible — `torch_txda` transparently maps CUDA ops to TXDA hardware, so the patches primarily fix device detection, distributed init, and MoE import guards that check `is_cuda()`/`is_hip()`/etc. and would otherwise fail.

## Platform detection — two independent paths

1. **`activate_platform()`** — uses `flag_gems.runtime.backend.device.DeviceDetector`; fallback checks `torch.txda`, `torch.npu`, `torch.musa`, `torch.cuda`.
2. **`config/utils.py:get_platform_name()`** — independent implementation checking same torch attributes in different order: txda → npu → musa → cuda → env override.

These can diverge — the config path has `SGLANG_FL_PLATFORM` override while the platform activation path does not.

## Vendor backend auto-discovery

`builtin_ops.py:_register_vendor_backends()` scans `sglang_fl/dispatch/backends/vendor/<name>/register_ops.py` (skipping `__*`, `template`). Each vendor's `register_builtins(registry)` function adds `OpImpl` entries with `_is_available` closures bound to the backend's `is_available()` method. No central registry of vendors exists — just the filesystem.

Current vendors: `cuda` (sgl_kernel), `ascend` (torch_npu), `musa` (torch_musa), `txda` (torch_txda / TsingMicro).

## Dispatch config YAMLs

Platform YAMLs live in `sglang_fl/dispatch/config/` (not `sglang_fl/config/`). Current platforms: `nvidia.yaml`, `ascend.yaml`, `musa.yaml`, `tsingmicro.yaml`. The README references `sglang_fl/config/sample.yaml` but no sample is shipped — use env vars or write your own YAML. The config loader in `utils.py` defines `_CONFIG_DIR = Path(__file__).parent` (i.e., `dispatch/config/`).

## `SGLANG_FL_STRICT` naming trap

`SGLANG_FL_STRICT=1` (default) → **fallback ENABLED** (tries next candidate on failure).
`SGLANG_FL_STRICT=0` → **direct resolve only, no fallback** (errors immediately if preferred backend unavailable).

This is the inverse of what the name suggests. The `SelectionPolicy.strict` field means "enable fallback", not "strict error mode".

## `--disable-piecewise-cuda-graph` is mandatory

FlagGems Triton kernels contain `logging.Logger` calls incompatible with `torch.compile` (used by SGLang's piecewise CUDA graph). Always pass this flag when launching. Regular CUDA graph capture (`CudaGraphRunner`) works fine. `PlatformFL.support_piecewise_cuda_graph()` returns `True` only for `device_type == "cuda"`, and `get_compile_backend()` always returns `"eager"` regardless.

## CommunicatorFL — three-tier backend selection

`CommunicatorFL.__init__()` resolves:
1. FlagCX (if `_dist_backend == "flagcx"`), creating `FlagCXCommunicator`
2. Falls back to `torch.distributed` (nccl/hccl/etc.) if FlagCX init fails
3. `disabled` flag set to `True` if `world_size <= 1`

The communicator is created **per GroupCoordinator** via AROUND hook on `__init__`, not once globally.

## TXDA patches subsystem

`patches/__init__.py:apply_all_txda_patches()` is called unconditionally in `load_plugin()`. Six patch modules are applied, all idempotent:

| Patch module | Gating | What it patches |
|---|---|---|
| `per_rank_log` | Always (if `SGLANG_FL_LOG_DIR` set) | Redirects scheduler subprocess stdout/stderr to `{dir}/rank_TP{n}.log` by monkey-patching `configure_scheduler_process` |
| `device_support` | Always | 5 patches: (1) tolerant `_register_fake` for missing torchvision ops, (2) safe `get_device_properties` when CUDA unavailable, (3) `get_device()` txda recognition, (4) inject `is_txda()` utility (txda-gated), (5) append `"txda"` to `SUPPORTED_DEVICES` (txda-gated) |
| `platform_stubs` | Always | Injects stub `_StubModule` instances into `sys.modules` for CUDA-only dependencies (`sgl_kernel`, `flashinfer`, `pynccl_allocator`) on non-CUDA platforms |
| `model_runner` | Always (gated by `SGLANG_FL_TIMER_ENABLE`) | Wraps `ModelRunner.forward()` with wall-clock timing (sync → forward → sync → print) |
| `dist_init` | `torch.txda.is_available()` | 5 sub-patches: (1) adds `"txda": "tccl"` to `_DEVICE_TO_DISTRIBUTED_BACKEND`, (2) wraps `GroupCoordinator.__init__` to set txda device (tricks `is_cuda_alike()` into True, then replaces device with `txda:{rank}`), (3) forces `backend="tccl"` in srt `init_distributed_environment`, (4) same for multimodal_gen, (5) routes `all_gather_into_tensor` through raw `_all_gather_into_tensor` path (bypasses custom-op) |
| `fused_moe` | `torch.txda.is_available()` | 4 sub-patches: (1) replaces `moe_align_block_size` with txda-safe version, (2) forces `_has_vllm_ops = False`, (3) can bypass `@register_custom_op` for `inplace_fused_experts`, (4) routes `UnquantizedFusedMoEMethod.apply` through `self.forward()` (dispatch system) |

The `per_rank_log` patch runs before dist is initialized — it intercepts `tp_rank` from `configure_scheduler_process` args before `configure_logger` creates its `StreamHandler`.

## Log suppression on non-rank-0

`_is_rank0()` checks `multiprocessing.parent_process()` — if we're a spawned subprocess (tp_rank >= 1), returns False. Non-rank-0 processes get `WARNING` log level to prevent duplicate output. This runs before dist is initialized, so it can't use `torch.distributed.get_rank()`.

## Process-fork safety

`OpManager.__init__` registers `os.register_at_fork(after_in_child=self._reset_after_fork)`. After fork: `initialized=False`, cache cleared, `policy_epoch` bumped. The next `ensure_initialized()` call in the child re-registers all backends and rebuilds the cache. This is critical for tp>1 where SGLang spawns worker processes.
