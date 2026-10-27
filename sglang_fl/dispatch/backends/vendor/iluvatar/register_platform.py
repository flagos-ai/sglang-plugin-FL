# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Initialize low-level CoreX model runners through the normal plugin loader."""

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
        raise RuntimeError("Iluvatar plugin hooks are missing or were reset")
    missing = owned - HookRegistry._patched
    if missing:
        raise RuntimeError(
            "Iluvatar plugin hooks were not applied: " + ", ".join(sorted(missing))
        )
    return frozenset(owned)


def _ensure_plugin_active() -> None:
    global _bootstrap_complete, _bootstrap_running, _bootstrap_failure
    global _bootstrap_targets
    import sglang_fl

    with _bootstrap_lock:
        if _bootstrap_failure is not None:
            raise RuntimeError("Iluvatar plugin bootstrap previously failed") from (
                _bootstrap_failure
            )
        if _bootstrap_running:
            raise RuntimeError("Recursive Iluvatar model-config plugin bootstrap")
        # Unlike Engine, one_batch directly constructs ModelRunner in each
        # worker. Use the SGLang loader so filters and apply_hooks are honored,
        # including communication. Its active flag is set before apply_hooks;
        # keep other model-config callers blocked until the whole loader returns.
        _bootstrap_running = True
        try:
            if not _bootstrap_complete and not sglang_fl.is_plugin_active():
                from sglang.srt.plugins import load_plugins

                load_plugins()
            if not sglang_fl.is_plugin_active():
                raise RuntimeError(
                    "Iluvatar requires complete sglang_fl plugin initialization "
                    "before constructing a model config; inspect the plugin "
                    "loader errors and SGLANG_PLUGINS selection"
                )
            # apply_hooks also logs and swallows individual target failures.
            # Verify our own registered targets, not unrelated plugin hooks.
            _bootstrap_targets = _applied_plugin_hooks()
            _bootstrap_complete = True
        except BaseException as exc:
            # A hook failure may leave the global active flag set. Never allow a
            # later caller to mistake that partial initialization for success.
            _bootstrap_failure = exc
            raise
        finally:
            _bootstrap_running = False


def _install_model_config_bootstrap() -> None:
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except ImportError as exc:
        # PlatformFL treats an ImportError as an optional missing vendor module.
        raise RuntimeError("Cannot install Iluvatar model-config bootstrap") from exc

    descriptor = ModelConfig.__dict__.get("from_server_args")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unexpected SGLang ModelConfig.from_server_args contract")
    original = descriptor.__func__
    if getattr(original, "_iluvatar_plugin_bootstrap", False):
        return

    @wraps(original)
    def from_server_args(*args, **kwargs):
        _ensure_plugin_active()
        return original(*args, **kwargs)

    from_server_args._iluvatar_plugin_bootstrap = True
    ModelConfig.from_server_args = staticmethod(from_server_args)


# init_backend is called while ModelRunner is still being imported. Loading all
# plugins there would resolve Scheduler hooks back into the incomplete module.
# The first real model-config call runs after those imports, before the runner's
# distributed initialization, dtype gate and model loading. Engine's normal
# already-active path passes through without reentering the loader.
_install_model_config_bootstrap()
