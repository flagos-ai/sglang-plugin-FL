# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, per-process provenance for successfully launched Triton kernels.

Startup/capture launches establish provenance only. Offline reports intersect
these names with measured runtime kernels; preparation calls/times never enter
statistics. Failed registered FlagGems calls discard their pending evidence.
"""

from __future__ import annotations

import contextvars
import functools
import importlib.metadata
import json
import os
import time
from pathlib import Path

_active_aten = contextvars.ContextVar("sglang_profile_aten", default=None)
_seen = set()
_provider = None
_last_phase = None


def triton_provider():
    global _provider
    if _provider is None:
        import triton

        origin = Path(triton.__file__).resolve()
        matches = []
        for name in ("triton", "flagtree"):
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            if Path(dist.locate_file("triton/__init__.py")).resolve() == origin:
                matches.append(
                    {"name": name, "version": dist.version, "module_path": str(origin)}
                )
        _provider = (
            matches[0]
            if len(matches) == 1
            else {"name": "unknown", "module_path": str(origin)}
        )
    return _provider


def _write(item):
    global _last_phase
    directory = os.environ.get("SGLANG_FL_PROFILE_EVIDENCE_DIR")
    if not directory:
        return
    import torch

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    item = {
        **item,
        "rank": rank,
        "pid": os.getpid(),
        "successful_launch": True,
        "phase": "runtime" if torch.autograd._profiler_enabled() else "preparation",
    }
    if item["phase"] != _last_phase:
        _seen.clear()
        _last_phase = item["phase"]
    encoded = json.dumps(item, sort_keys=True)
    if encoded in _seen:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"worker-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(
            json.dumps({**item, "timestamp_ns": time.time_ns()}, sort_keys=True) + "\n"
        )
    _seen.add(encoded)


def record_triton_launch(kernel, fn):
    if kernel is None or not os.environ.get("SGLANG_FL_PROFILE_EVIDENCE_DIR"):
        return
    name = getattr(kernel, "name", None)
    if not name:
        return
    callable_name = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
    item = {
        "kernel_name": name,
        "callable": callable_name,
        "provider": triton_provider(),
        "evidence_kind": "successful_triton_launch",
    }
    active = _active_aten.get()
    if active is not None:
        api, pending = active
        pending.append({**item, "aten_api": api, "component": "FlagGems"})
    else:
        component = (
            "FlagGems"
            if callable_name.startswith("flag_gems.")
            else (
                "FlagTree" if item["provider"].get("name") == "flagtree" else "triton"
            )
        )
        _write({**item, "component": component})


def instrument_flaggems_registrations(flag_gems):
    """Wrap the real registration table before enable()/only_enable()."""
    if getattr(flag_gems, "_sglang_profile_wrapped", False):
        return

    def wrap(api, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            pending = []
            token = _active_aten.set(("aten::" + api.split(".", 1)[0], pending))
            try:
                result = fn(*args, **kwargs)
            finally:
                _active_aten.reset(token)
            # This executes only on success; a failed provider cannot establish
            # replacement evidence even when the dispatch manager falls back.
            for item in pending:
                _write(item)
            return result

        return call

    config = tuple(
        (item[0], wrap(item[0], item[1]), *item[2:]) for item in flag_gems._FULL_CONFIG
    )
    replacements = {
        (old[0], id(old[1])): new for old, new in zip(flag_gems._FULL_CONFIG, config)
    }
    flag_gems.FULL_CONFIG_BY_FUNC = {
        name: [replacements.get((item[0], id(item[1])), item) for item in entries]
        for name, entries in flag_gems.FULL_CONFIG_BY_FUNC.items()
    }
    flag_gems._FULL_CONFIG = config
    flag_gems._sglang_profile_wrapped = True
