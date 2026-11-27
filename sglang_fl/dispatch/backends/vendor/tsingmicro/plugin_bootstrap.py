# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the full sglang_fl plugin in TXDA workers that SGLang starts without it.

``Engine`` / ``launch_server`` / the scheduler call SGLang's ``load_plugins()``,
which runs ``sglang_fl.load_plugin()`` (FlagGems, dispatch, all txda vendor
patches). ``sglang.bench_one_batch`` (``sglang.benchmark.one_batch``) does not:
each TP worker constructs ``ModelRunner`` directly, so only the platform entry
point is active there and the model runs without the plugin (for example the
dummy-weight patch never applies and the latency smoke fails).

Wrap ``ModelConfig.from_server_args``: every worker calls it before
distributed init and model loading. If the plugin is not active yet, run
SGLang's own ``load_plugins()`` once (plugin filters and ``apply_hooks`` are
honoured), then require that every hook registered by sglang_fl was applied.
On the Engine path the plugin is already active and the wrapper only passes
through. Same mechanism as ``vendor/iluvatar/register_platform.py``.
"""

from functools import wraps
import threading

_bootstrap_lock = threading.RLock()
_bootstrap_complete = False
_bootstrap_running = False
_bootstrap_failure = None
_bootstrap_targets = frozenset()


def _applied_plugin_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry

    owned = {
        target
        for target, hooks in HookRegistry._hooks.items()
        if any(
            source.plugin_name == "sglang_fl"
            if source is not None
            else getattr(hook, "__module__", "").split(".")[0] == "sglang_fl"
            for _, hook, source in hooks
        )
    }
    if not owned or not _bootstrap_targets <= owned:
        raise RuntimeError("TsingMicro plugin hooks are missing or were reset")
    missing = owned - HookRegistry._patched
    if missing:
        raise RuntimeError(
            "TsingMicro plugin hooks were not applied: " + ", ".join(sorted(missing))
        )
    return frozenset(owned)


def _ensure_plugin_active() -> None:
    global _bootstrap_complete, _bootstrap_running, _bootstrap_failure
    global _bootstrap_targets
    import sglang_fl

    with _bootstrap_lock:
        if _bootstrap_failure is not None:
            raise RuntimeError("TsingMicro plugin bootstrap previously failed") from (
                _bootstrap_failure
            )
        if _bootstrap_running:
            raise RuntimeError("Recursive TsingMicro model-config plugin bootstrap")
        # The plugin's active flag is set before apply_hooks; keep other
        # model-config callers blocked until the whole loader returns.
        _bootstrap_running = True
        try:
            if not _bootstrap_complete and not sglang_fl.is_plugin_active():
                from sglang.srt.plugins import load_plugins

                load_plugins()
            if not sglang_fl.is_plugin_active():
                raise RuntimeError(
                    "TsingMicro requires complete sglang_fl plugin initialization "
                    "before constructing a model config; inspect the plugin "
                    "loader errors and SGLANG_PLUGINS selection"
                )
            # apply_hooks logs and swallows individual target failures; verify
            # our own registered targets, not unrelated plugin hooks.
            _bootstrap_targets = _applied_plugin_hooks()
            _bootstrap_complete = True
        except BaseException as exc:
            # A hook failure may leave the global active flag set. Never let a
            # later caller mistake that partial initialization for success.
            _bootstrap_failure = exc
            raise
        finally:
            _bootstrap_running = False


def install_model_config_bootstrap() -> None:
    """Wrap ``ModelConfig.from_server_args``. Idempotent."""
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except ImportError as exc:
        # PlatformFL treats an ImportError as an optional missing vendor module.
        raise RuntimeError("Cannot install TsingMicro model-config bootstrap") from exc

    descriptor = ModelConfig.__dict__.get("from_server_args")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unexpected SGLang ModelConfig.from_server_args contract")
    original = descriptor.__func__
    if getattr(original, "_tsingmicro_plugin_bootstrap", False):
        return

    @wraps(original)
    def from_server_args(*args, **kwargs):
        _ensure_plugin_active()
        return original(*args, **kwargs)

    from_server_args._tsingmicro_plugin_bootstrap = True
    ModelConfig.from_server_args = staticmethod(from_server_args)
