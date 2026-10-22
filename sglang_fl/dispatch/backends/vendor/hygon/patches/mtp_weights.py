# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check actual Qwen3.5/3.6 MTP weight loading on HCU, without eager imports.

The upstream loader ignores ``load_weights``' returned parameter names. The MTP
class also skips target-only checkpoint tensors, so successful target loading is
not evidence that the speculative head was initialized. This guard checks both
that return value and the checkpoint components consumed by the real MTP loader.
No tensor is copied, inspected, or retained by the guard.
"""

from functools import wraps
from importlib._bootstrap import _ModuleLockManager
import importlib.abc
import logging
import sys
from threading import RLock

logger = logging.getLogger(__name__)
_MODULE = "sglang.srt.models.qwen3_5_mtp"
_CLASS = "Qwen3_5ForCausalLMMTP"
_MARKER = "_hcu_mtp_weight_guard"
_SHARED = frozenset({"model.embed_tokens.weight", "lm_head.weight"})
_ANCHORS = frozenset(
    {
        "fc.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "model.norm.weight",
    }
)
_finder = None
_registration_lock = RLock()


def _is_hcu():
    # Platform discovery has already imported torch. Do not import it here.
    return hasattr(sys.modules.get("torch"), "__hcu_version__")


def _normalized_mtp_key(name):
    if not name.startswith("mtp."):
        return None
    return (
        name.replace("mtp.", "model.")
        .replace("model.fc", "fc")
        .replace("model.pre_fc", "pre_fc")
        .replace(".self_attn", "")
    )


def _missing_components(required, observed, num_experts):
    missing = []
    for name in sorted(required):
        if name in observed:
            continue
        if ".qkv_proj." in name:
            parts = {
                name.replace(".qkv_proj.", f".{part}_proj.") for part in ("q", "k", "v")
            }
        elif ".gate_up_proj." in name:
            parts = {
                name.replace(".gate_up_proj.", f".{part}_proj.")
                for part in ("gate", "up")
            }
        elif name.endswith(".experts.w13_weight"):
            prefix = name.removesuffix("w13_weight")
            if prefix + "gate_up_proj" in observed:
                continue
            parts = {
                f"{prefix}{i}.{part}_proj.weight"
                for i in range(num_experts)
                for part in ("gate", "up")
            }
        elif name.endswith(".experts.w2_weight"):
            prefix = name.removesuffix("w2_weight")
            if prefix + "down_proj" in observed:
                continue
            parts = {f"{prefix}{i}.down_proj.weight" for i in range(num_experts)}
        else:
            parts = {name}
        if not parts:
            missing.append(name + " (unknown expert count)")
        else:
            missing.extend(sorted(parts - observed))
    return missing


def _guard_class(cls):
    if cls.__name__ != _CLASS or cls.__module__ != _MODULE:
        raise RuntimeError(f"HCU MTP guard found unexpected model class: {cls!r}")
    original = cls.load_weights
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def checked_load_weights(self, weights, *args, **kwargs):
        if not _is_hcu():
            return original(self, weights, *args, **kwargs)
        # Preserve the upstream streaming iterator and never retain tensors.
        source_keys = set()
        consumed = 0

        def observed_weights():
            nonlocal consumed
            for name, tensor in weights:
                consumed += 1
                normalized = _normalized_mtp_key(name)
                if normalized is not None:
                    source_keys.add(normalized)
                yield name, tensor

        loaded = original(self, observed_weights(), *args, **kwargs)
        if not isinstance(loaded, (set, frozenset)) or any(
            not isinstance(name, str) for name in loaded
        ):
            raise RuntimeError(
                "HCU MTP load_weights must return loaded parameter names"
            )
        named = {name for name, _ in self.named_parameters()}
        required = named - _SHARED
        absent_anchors = _ANCHORS - required
        if absent_anchors or not any(
            name.startswith("model.layers.") for name in required
        ):
            raise RuntimeError(
                "HCU MTP model has no complete MTP branch: "
                f"missing anchors={sorted(absent_anchors)}; required={sorted(required)}"
            )
        missing = required - loaded
        num_experts = getattr(getattr(self, "config", None), "num_experts", 0) or 0
        missing_components = _missing_components(required, source_keys, num_experts)
        report = {
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "required_params": sorted(required),
            "loaded_params": sorted(loaded),
            "shared_params": sorted(named & _SHARED),
            "mtp_source_keys": sorted(source_keys),
            "checkpoint_tensors_consumed": consumed,
            "missing_params": sorted(missing),
            "missing_components": missing_components,
        }
        self._hcu_mtp_weight_coverage = report
        if missing or missing_components:
            raise RuntimeError(
                f"HCU MTP weights incomplete for {report['class']}: "
                f"missing parameters={sorted(missing)}; "
                f"missing checkpoint components={missing_components}"
            )
        logger.info(
            "HCU MTP weights verified: class=%s required=%d loaded=%d "
            "shared=%d mtp_source_keys=%d checkpoint_tensors=%d missing=0",
            report["class"],
            len(required),
            len(loaded),
            len(named & _SHARED),
            len(source_keys),
            consumed,
        )
        return loaded

    setattr(checked_load_weights, _MARKER, True)
    cls.load_weights = checked_load_weights


def _guard_module(module):
    cls = getattr(module, _CLASS, None)
    if not isinstance(cls, type):
        raise RuntimeError(f"HCU MTP guard cannot find {_MODULE}.{_CLASS}")
    _guard_class(cls)


class _MTPImportLoader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        with _registration_lock:
            if _is_hcu():
                _guard_module(module)
            if _finder in sys.meta_path:
                sys.meta_path.remove(_finder)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _MTPImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MODULE:
            return None
        # Delegate in the existing finder order without recursing into ourselves.
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            spec = find(fullname, path, target) if find is not None else None
            if spec is not None:
                if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                    raise ImportError(
                        f"HCU MTP guard requires an executable {_MODULE} loader"
                    )
                spec.loader = _MTPImportLoader(spec.loader)
                return spec
        return None


def register_hcu_mtp_weight_guard():
    """Guard only HCU MTP loading when the upstream module naturally imports.

    Safe during general-plugin startup: imports neither torch, the registry, nor
    a model module. Non-HCU processes retain the original loading path.
    """
    global _finder
    if not _is_hcu():
        return False
    # The module lock waits for a natural import on another thread without
    # importing SGLang or the model here. Acquire it before our registration
    # lock: the import loader owns that module lock when it takes our lock.
    # Reversing the order would deadlock registration against an active import.
    with _ModuleLockManager(_MODULE), _registration_lock:
        module = sys.modules.get(_MODULE)
        if module is not None:
            spec = getattr(module, "__spec__", None)
            if getattr(spec, "_initializing", False):
                # Same-thread re-entry from our loader must defer until its
                # exec_module finishes defining the class. Another thread's
                # import cannot still be initializing after the module lock.
                if isinstance(getattr(spec, "loader", None), _MTPImportLoader):
                    return True
                raise RuntimeError(
                    "Register the HCU MTP guard before importing the model; "
                    "registration re-entered its unguarded module initialization"
                )
            _guard_module(module)
            return True
        if _finder is None:
            _finder = _MTPImportFinder()
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
        return True
