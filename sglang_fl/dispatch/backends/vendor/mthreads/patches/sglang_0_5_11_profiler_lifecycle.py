# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transactional profiler cleanup for the fixed SGLang v0.5.11 target.

This module contains no MUSA behavior. It only closes v0.5.11 failure paths so
an external profiler-marker error cannot leave already-started profilers or the
legacy scheduler state active. Keep this shim isolated and version-locked.
"""

from __future__ import annotations

import logging
from functools import wraps
from importlib import metadata

logger = logging.getLogger(__name__)

_SUPPORTED_SGLANG_VERSION = "0.5.11"
_patches_applied = False


def _installed_sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        import sglang

        version = getattr(sglang, "__version__", None)
        return str(version) if version is not None else "unknown"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0].split(".post", 1)[0]


def _require_sglang_0_5_11() -> None:
    installed = _installed_sglang_version()
    if _base_version(installed) != _SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            "The MUSA profiler lifecycle compatibility patch requires SGLang "
            f"{_SUPPORTED_SGLANG_VERSION}, but found {installed}."
        )


def _reset_legacy_state(scheduler) -> None:
    scheduler.torch_profiler = None
    scheduler.profile_in_progress = False
    scheduler.profiler_start_forward_ct = None


def _wrap_legacy_profiler_lifecycle(profiler_mixin, marker_error_type: type) -> None:
    if getattr(profiler_mixin, "_musa_profiler_lifecycle_patched", False):
        return

    original_start = profiler_mixin.start_profile
    original_stop = profiler_mixin.stop_profile

    @wraps(original_start)
    def start_profile_with_rollback(self, *args, **kwargs):
        try:
            return original_start(self, *args, **kwargs)
        except marker_error_type:
            if getattr(self, "profile_in_progress", False):
                try:
                    # Let SGLang stop the profilers it successfully started.
                    original_stop(self, *args, **kwargs)
                except Exception:
                    logger.exception(
                        "Failed to fully roll back SGLang profilers after a "
                        "capture-marker start error"
                    )
            _reset_legacy_state(self)
            raise

    @wraps(original_stop)
    def stop_profile_with_cleanup(self, *args, **kwargs):
        try:
            return original_stop(self, *args, **kwargs)
        except marker_error_type:
            # The capture marker is the last stop operation in SGLang v0.5.11;
            # Torch, RPD, and memory profilers have already been stopped here.
            _reset_legacy_state(self)
            raise

    profiler_mixin.start_profile = start_profile_with_rollback
    profiler_mixin.stop_profile = stop_profile_with_cleanup
    profiler_mixin._musa_profiler_lifecycle_patched = True


def _wrap_profiler_list_lifecycle(profiler_list) -> None:
    if getattr(profiler_list, "_musa_profiler_lifecycle_patched", False):
        return

    def start_transactionally(self):
        started = []
        try:
            for inner in self.inners:
                inner.start()
                started.append(inner)
        except Exception:
            for inner in reversed(started):
                try:
                    inner.stop()
                except Exception:
                    logger.exception(
                        "Failed to stop a profiler while rolling back profiler startup"
                    )
            raise

    def stop_all(self):
        first_error = None
        first_traceback = None
        for inner in self.inners:
            try:
                inner.stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                    first_traceback = exc.__traceback__
                else:
                    logger.exception(
                        "Additional profiler stop failed while cleaning up",
                        exc_info=exc,
                    )
        if first_error is not None:
            raise first_error.with_traceback(first_traceback)

    profiler_list.start = start_transactionally
    profiler_list.stop = stop_all
    profiler_list._musa_profiler_lifecycle_patched = True


def apply_sglang_0_5_11_profiler_lifecycle_patch(
    marker_error_type: type,
) -> None:
    """Apply only the cleanup SGLang v0.5.11 is missing."""
    global _patches_applied
    if _patches_applied:
        return

    _require_sglang_0_5_11()

    from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
    from sglang.srt.utils.profile_utils import _ProfilerList

    _wrap_legacy_profiler_lifecycle(SchedulerProfilerMixin, marker_error_type)
    _wrap_profiler_list_lifecycle(_ProfilerList)
    _patches_applied = True
    logger.info("SGLang v0.5.11 profiler transactional cleanup patch applied")
