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

"""Transactional cleanup for SGLang's legacy and composite profilers.

The legacy adapter requires stage-aware start/stop methods and the scheduler's
existing profiler state. Its stop path must stop the other profilers before the
capture marker. The composite adapter requires an ordered ``inners`` collection
whose members expose start/stop. Keep these internal API assumptions isolated
from the Torch/MUSA leaf redirects.
"""

from __future__ import annotations

import logging
from functools import wraps
from inspect import signature

logger = logging.getLogger(__name__)

_patches_applied = False


def _require_profiler_contract(profiler_mixin, profiler_list) -> None:
    """Check the APIs used by both adapters before changing either class."""
    try:
        for method in (profiler_mixin.start_profile, profiler_mixin.stop_profile):
            method_signature = signature(method)
            if "stage" not in method_signature.parameters:
                raise ValueError("legacy profiler methods must expose stage")
            method_signature.bind(None)
            method_signature.bind(None, stage=None)

        signature(profiler_list.start).bind(None)
        signature(profiler_list.stop).bind(None)
        inners = [object(), object()]
        if list(profiler_list(inners).inners) != inners:
            raise ValueError("composite profiler must preserve inner order")
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "SGLang profiler cleanup requires stage-aware "
            "SchedulerProfilerMixin.start_profile/stop_profile and "
            "_ProfilerList(inners).start/stop with an ordered inners collection."
        ) from exc


def _reset_legacy_state(scheduler) -> None:
    scheduler.torch_profiler = None
    scheduler.profile_in_progress = False
    scheduler.profiler_start_forward_ct = None


def _wrap_legacy_profiler_lifecycle(profiler_mixin, marker_error_type: type) -> None:
    if getattr(profiler_mixin, "_musa_profiler_lifecycle_patched", False):
        return

    original_start = profiler_mixin.start_profile
    original_stop = profiler_mixin.stop_profile
    start_signature = signature(original_start)

    @wraps(original_start)
    def start_profile_with_rollback(self, *args, **kwargs):
        try:
            return original_start(self, *args, **kwargs)
        except marker_error_type:
            if getattr(self, "profile_in_progress", False):
                stage = start_signature.bind(self, *args, **kwargs).arguments.get(
                    "stage"
                )
                try:
                    # Let SGLang stop the profilers it successfully started.
                    original_stop(self, stage=stage)
                except marker_error_type:
                    # Other profilers have stopped before the failed marker.
                    logger.exception("Capture-marker stop failed during rollback")
                except Exception:
                    logger.exception(
                        "Failed to fully roll back SGLang profilers after a "
                        "capture-marker start error"
                    )
                    # Preserve active state when native cleanup is incomplete.
                    raise
            _reset_legacy_state(self)
            raise

    @wraps(original_stop)
    def stop_profile_with_cleanup(self, *args, **kwargs):
        try:
            return original_stop(self, *args, **kwargs)
        except marker_error_type:
            # The legacy contract places capture-marker stop after other stops;
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


def apply_profiler_lifecycle_patch(
    marker_error_type: type,
) -> None:
    """Adapt the required legacy/composite APIs without a version-string gate."""
    global _patches_applied
    if _patches_applied:
        return

    try:
        from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
        from sglang.srt.utils.profile_utils import _ProfilerList
    except ImportError as exc:
        raise RuntimeError(
            "Required SGLang profiler cleanup APIs are unavailable"
        ) from exc

    _require_profiler_contract(SchedulerProfilerMixin, _ProfilerList)
    _wrap_legacy_profiler_lifecycle(SchedulerProfilerMixin, marker_error_type)
    _wrap_profiler_list_lifecycle(_ProfilerList)
    _patches_applied = True
    logger.info("SGLang profiler transactional cleanup patch applied")
