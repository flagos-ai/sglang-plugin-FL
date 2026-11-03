# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model-scoped MThreads compatibility patches."""


def require_supported_sglang(version=None):
    """Reject private-hook installation outside the tested upstream release."""
    if version is None:
        from importlib.metadata import version as distribution_version

        version = distribution_version("sglang")
    if version != "0.5.18":
        raise RuntimeError(
            f"MUSA MiniMax compatibility requires SGLang 0.5.18, found {version}"
        )


def _install_atomically(targets, installers):
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for install in installers:
            install()
    except BaseException:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        raise


def install_minimax_w8a8_support():
    """Install interdependent quantization/pool routing as one transaction."""
    require_supported_sglang()
    from sglang.kernels.ops.attention.minimax_sparse.prefill import flash_with_topk_idx
    from sglang.kernels.ops.quantization import int8_kernel
    from sglang.srt.mem_cache import kv_cache_configurator

    from .int8_rounding import install_int8_rounding_patch
    from .minimax_m3 import install_minimax_m3_patches
    from .sparse_topk import install_sparse_topk_patch

    _install_atomically(
        [
            (int8_kernel, "_per_token_quant_int8"),
            (flash_with_topk_idx, "_topk_index_kernel"),
            (kv_cache_configurator.KVCacheConfigurator, "_build_oot_mha_kv_pool"),
        ],
        [
            lambda: install_int8_rounding_patch(int8_kernel),
            lambda: install_sparse_topk_patch(flash_with_topk_idx),
            lambda: install_minimax_m3_patches(kv_cache_configurator),
        ],
    )
