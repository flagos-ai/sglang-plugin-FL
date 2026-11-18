# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey patches to adapt sglang core for txda (TsingMicro) platform.

These patches replace direct modifications to the sglang source tree with
runtime monkey-patching, keeping all txda-specific logic inside the plugin.

Patches applied:
  cuda_alias      – torch.cuda -> txda (torch_txda.transfer_to_txda), first
  device_support  – torchvision fake-register tolerance (CPU torch build)
  dist_init       – GroupCoordinator device (txda instead of cpu),
                    all_gather_into_tensor txda routing
  per_rank_log    – Redirect scheduler subprocess stdout/stderr to
                    per-rank log files (SGLANG_FL_LOG_DIR)
  fused_moe       – moe_align_block_size for the Triton MoE runner fallback
  platform_stubs  – Inject stub modules for CUDA-only deps on non-CUDA platforms
  model_runner    – Per-iteration forward() timing (SGLANG_FL_TIMER_ENABLE)
  scheduler_pp_mixin –  PP send/recv even/odd ordering for txda (same as XPU)
  qwen3_asr_config – Default AutoConfig.register exist_ok=True to tolerate
                    transformers 5.15+ built-in qwen3_asr registration
  speculative_tree – EAGLE/MTP tree build via SGLang's Triton kernel (no CUDA
                    sgl_kernel binding); greedy verify in torch (the Triton
                    verify is wrong on txda)
  gdn_verify_conv – MTP verify: GDN causal_conv1d_update one draft token per
                    call (FlagTree cannot compile the multi-token variant)
  speculative_cache_locs – speculative KV slot bookkeeping in torch (verify
                    write locations: no txda branch upstream; req_to_token writes
                    and assign_extend_cache_locs: Triton kernels wrong on txda)
  vocab_embedding – Keep VocabParallelEmbedding on the torch path on txda
                    (SGLang 0.5.18 Triton embedding: wrong rows for out-of-shard ids)
  triton_load_watch – Disable SGLang 0.5.18 CUDA-only Triton kernel-load
                    diagnostics (crash on txda device properties)
  dummy_weights   – load_format=dummy: draw txda dummy weights on the CPU
                    (torch_txda uniform_ rejects a txda generator)
  gdn_norm        – Pin GatedDeltaNet's gated RMSNorm to
                    fla.layernorm_gated.RMSNorm (SGLANG_FL_GDN_NORM, default 1)

Note: attention kernel routing (extend and decode to the flag_gems
``_tsingmicro`` kernels) lives in ``impl/attention_backend.py``, not here --
it is a backend concern, bound per instance in ``TxdaAttnBackend.__init__``.

Usage (called automatically by load_plugin()):

    from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import apply_all_txda_patches
    apply_all_txda_patches()
"""

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as is_txda_available

_log = patch_logger("patches")


# Process-wide knobs read by the TXDA Triton backend (FlagTree) at kernel
# compile time. Set here, during plugin load, so every process -- including
# SGLang's spawned schedulers, which load the plugin again -- inherits them
# before its first compilation. Existing values are never overridden.
_TXDA_ENV_DEFAULTS = {
    # FlagTree tsingmicro precision mode (0/1/2); the validated Qwen3.6
    # configuration uses 2. See triton/backends/tsingmicro/precision.py.
    "PRECISION_MODE": "2",
}


def _apply_txda_env_defaults() -> None:
    import os

    applied = []
    for key, value in _TXDA_ENV_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(f"{key}={value}")
    if applied:
        _log.applied("txda env defaults: %s", ", ".join(applied))


def apply_all_txda_patches() -> None:
    """Apply all monkey patches required for txda platform inference.

    Idempotent: safe to call multiple times.  Non-txda platforms are no-ops
    for per-patch submodules; per_rank_log and device_support are applied
    unconditionally.
    """
    from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import cuda_alias

    # First: torch.cuda -> txda. Everything below (and SGLang) relies on it.
    cuda_alias.patch()

    from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import per_rank_log, device_support, platform_stubs, model_runner, qwen3_asr_config, kv_leak_fix

    per_rank_log.patch()
    qwen3_asr_config.patch()  # tolerate transformers 5.15+ built-in qwen3_asr (no-op with 0.5.18's pinned 5.12.1)
    device_support.patch()   # torchvision fake-register tolerance
    platform_stubs.patch()   # dummy-module injection for non-CUDA platforms
    model_runner.patch()     # gated by SGLANG_FL_TIMER_ENABLE internally
    kv_leak_fix.patch()     # hardened KV release; on by default, SGLANG_KVLEAK_FIX=0 disables

    if not is_txda_available():
        _log.skipped("txda not available — txda-specific patches skipped")
        return

    _apply_txda_env_defaults()

    from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (
        dist_init,
        dummy_weights,
        fused_moe,
        gdn_verify_conv,
        gdn_norm,
        scheduler_pp_mixin,
        speculative_cache_locs,
        triton_load_watch,
        vocab_embedding,
        speculative_tree,
    )

    dist_init.patch()
    fused_moe.patch()  # moe_align_block_size for the Triton MoE fallback
    scheduler_pp_mixin.patch()
    speculative_tree.patch()  # EAGLE/MTP tree build (Triton) + greedy verify (torch)
    gdn_verify_conv.patch()  # MTP verify: GDN conv update one draft token per call (FlagTree)
    speculative_cache_locs.patch()  # speculative KV slot bookkeeping in torch
    vocab_embedding.patch()  # 0.5.18 Triton embedding returns non-zero rows for out-of-shard ids on txda
    triton_load_watch.patch()  # 0.5.18 CUDA-only kernel-load hook crashes on txda
    dummy_weights.patch()  # load_format=dummy: torch_txda uniform_ rejects a txda generator
    gdn_norm.patch()  # pin GDN gated norm to fla.layernorm_gated.RMSNorm

    _log.applied("all txda monkey patches applied")
