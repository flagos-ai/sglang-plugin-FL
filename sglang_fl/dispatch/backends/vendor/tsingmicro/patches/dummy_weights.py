# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate dummy weights on the CPU for txda parameters (``load_format=dummy``).

SGLang's ``initialize_dummy_weights`` (``sglang.srt.model_loader.weight_utils``)
seeds one generator per parameter on the parameter's device::

    generator = torch.Generator(device=param.data.device)   # txda generator: OK
    param.uniform_(low, high, generator=generator)           # fails on txda

torch_txda creates the txda generator, but ``uniform_`` on a txda tensor runs
the CPU kernel, which only accepts a CPU generator::

    RuntimeError: Expected a 'cpu' device type for generator but found 'txda'

Every dummy-weight entry point (``sglang.bench_one_batch``, ``--load-format
dummy``) fails while loading the model.

For parameters on txda, draw the same per-parameter values on the CPU with a
CPU generator seeded like upstream, then copy them to the device. Upstream's
contract is kept: values only depend on the seed, element count and dtype, so
every TP rank gets consistent weights. Parameters on other devices go through
SGLang's original function. Real checkpoints never reach this code.

Remove once torch_txda's ``uniform_`` accepts a txda generator.

``loader.py`` imports the function by name, so the patch rebinds it there (and
in ``weight_utils``) right after SGLang imports the loader.
"""

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("dummy_weights")

_LOADER_MODULE = "sglang.srt.model_loader.loader"
_patched = False


def _make_txda_aware(orig_fn):
    def initialize_dummy_weights(
        model: torch.nn.Module,
        low: float = -1e-3,
        high: float = 1e-3,
        seed: int = 1234,
    ) -> None:
        others = {}
        for name, param in model.state_dict().items():
            if not torch.is_floating_point(param):
                continue
            if param.device.type != "txda":
                others[name] = param
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            # Same dtype rule as upstream: uniform_ has no < 16-bit kernels.
            gen_dtype = torch.float16 if torch.finfo(param.dtype).bits < 16 else param.dtype
            values = torch.empty(param.shape, dtype=gen_dtype, device="cpu")
            values.uniform_(low, high, generator=generator)
            param.data.copy_(values.to(param.data.dtype))
        if others:
            # Non-txda parameters (none on a txda run): upstream behavior.
            orig_fn(_StateDictView(others), low=low, high=high, seed=seed)

    initialize_dummy_weights.__wrapped__ = orig_fn
    initialize_dummy_weights._txda_dummy_weights = True
    initialize_dummy_weights.__doc__ = orig_fn.__doc__
    return initialize_dummy_weights


class _StateDictView:
    """Minimal stand-in for a module: upstream only calls ``state_dict()``."""

    def __init__(self, tensors):
        self._tensors = tensors

    def state_dict(self):
        return self._tensors


def _apply() -> None:
    try:
        import sglang.srt.model_loader.loader as loader
        import sglang.srt.model_loader.weight_utils as weight_utils

        orig = getattr(loader, "initialize_dummy_weights", None)
        if orig is None:
            _log.skipped("loader has no initialize_dummy_weights")
            return
        if getattr(orig, "_txda_dummy_weights", False):
            return
        patched = _make_txda_aware(orig)
        loader.initialize_dummy_weights = patched
        if getattr(weight_utils, "initialize_dummy_weights", None) is orig:
            weight_utils.initialize_dummy_weights = patched
        _log.applied("initialize_dummy_weights -> CPU generator for txda parameters")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch initialize_dummy_weights: %s", exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_LOADER_MODULE, _apply)
    _patched = True
