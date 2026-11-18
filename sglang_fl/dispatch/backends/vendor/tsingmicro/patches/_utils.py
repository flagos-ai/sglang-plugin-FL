# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for the patches subsystem."""

import importlib.abc
import importlib.util
import sys

import torch


def is_txda() -> bool:
    """Check whether the txda platform is available.

    Returns True when ``torch_txda`` is loaded and reports hardware as ready.
    """
    return hasattr(torch, "txda") and torch.txda.is_available()


class _PostImportLoader(importlib.abc.Loader):
    """Forward to the original loader, then run the callbacks after the module loads."""

    def __init__(self, inner, callbacks):
        self._inner = inner
        self._callbacks = callbacks

    def create_module(self, spec):
        try:
            return self._inner.create_module(spec)
        except AttributeError:
            return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        for callback in list(self._callbacks):
            callback()


class _PostImportWrapFinder(importlib.abc.MetaPathFinder):
    """One-shot finder: wrap a target module's loader to run its callbacks after import."""

    def __init__(self, target, callback):
        self._target = target
        self._callbacks = [callback]
        self._finding = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._finding = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostImportLoader(spec.loader, self._callbacks)
        return spec


def _install_post_import_wrap(module_name, callback) -> None:
    """Apply `callback` once `module_name` has been imported by sglang itself.

    If the module is already loaded, apply immediately. Otherwise install a
    meta_path finder that fires the callback right after the module loads.
    Several patches may hook the same module; each callback runs once, in
    registration order.
    """
    if module_name in sys.modules:
        callback()
        return
    for f in sys.meta_path:
        if type(f) is _PostImportWrapFinder and f._target == module_name:
            if callback not in f._callbacks:
                f._callbacks.append(callback)
            return
    sys.meta_path.insert(0, _PostImportWrapFinder(module_name, callback))
