# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FlagOS mHC implementations backed by FlagGems."""

from __future__ import annotations

import importlib
import weakref
from typing import Optional

import torch


class _TensorIdentityWeakCache:
    """Weak cache keyed by object identity without Tensor equality checks."""

    def __init__(self):
        self._data = {}

    def _remove(self, key_id, key_ref):
        current = self._data.get(key_id)
        if current is not None and current[0] is key_ref:
            self._data.pop(key_id, None)

    def get(self, key, default=None):
        item = self._data.get(id(key))
        if item is None:
            return default
        key_ref, value = item
        if key_ref() is key:
            return value
        self._data.pop(id(key), None)
        return default

    def __setitem__(self, key, value):
        key_id = id(key)
        key_ref = weakref.ref(
            key,
            lambda ref, key_id=key_id: self._remove(key_id, ref),
        )
        self._data[key_id] = (key_ref, value)


def _patch_flaggems_mhc_tensor_cache() -> None:
    """Avoid WeakKeyDictionary Tensor equality during repeated Graph capture."""

    module = importlib.import_module("flag_gems.fused.mhc.mhc_pre")
    cache = getattr(module, "_FN_BF16_CACHE", None)
    if cache is not None and not isinstance(cache, _TensorIdentityWeakCache):
        module._FN_BF16_CACHE = _TensorIdentityWeakCache()


def mhc_pre_flagos(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the FlagGems mHC pre kernel.

    ``n_splits_pre`` is accepted for compatibility with SGLang's TileLang
    wrapper. FlagGems selects its own pre-projection strategy.
    """
    del n_splits_pre
    _patch_flaggems_mhc_tensor_cache()
    from flag_gems.fused.mhc import mhc_pre

    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def mhc_post_flagos(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Run the FlagGems mHC post kernel."""
    from flag_gems.fused.mhc import mhc_post

    return mhc_post(x, residual, post_layer_mix, comb_res_mix)
