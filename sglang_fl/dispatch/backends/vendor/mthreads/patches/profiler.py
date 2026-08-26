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

"""MUSA redirects for SGLang v0.5.11's CUDA-oriented profiler APIs.

SGLang v0.5.11 owns the profiler lifecycle and recognizes the public activity
names ``GPU`` and ``CUDA_PROFILER``. On MUSA workers this module redirects the
two CUDA-specific leaves used by that implementation:

* ``ProfilerActivity.CUDA`` becomes TorchMUSA's PrivateUse1/MUSA activity.
* ``torch.cuda.cudart()`` returns a delegating proxy whose profiler markers
  call ``musaProfilerStart`` and ``musaProfilerStop``.

No SGLang profiler implementation is copied here. The only SGLang-internal
compatibility patch is a version-locked transactional cleanup shim for failure
paths that v0.5.11 does not roll back itself.
"""

from __future__ import annotations

import ctypes
import logging
from functools import wraps
from typing import Callable

import torch

from .sglang_0_5_11_profiler_lifecycle import (
    apply_sglang_0_5_11_profiler_lifecycle_patch,
)

logger = logging.getLogger(__name__)

MUSA_SUCCESS = 0
MUSA_ERROR_NOT_SUPPORTED = 801
_MUSART_SONAME = "libmusart.so"
_REQUIRED_MUSART_SYMBOLS = (
    "musaProfilerStart",
    "musaProfilerStop",
    "musaGetLastError",
)


class MusaProfilerError(RuntimeError):
    """A MUSA profiler runtime or ABI failure."""


class MusaProfilerApi:
    """Call MUSA capture-range markers through the runtime shared library.

    The adapter is validated with MUSA Runtime 4.3.x. It intentionally loads
    the unversioned SONAME so the active MUSA installation controls which ABI
    is selected, then requires every symbol and declares each C signature used
    by this module.
    """

    def __init__(self) -> None:
        self._library = None

    def validate(self) -> None:
        """Load the runtime and verify the required profiler ABI."""
        self._load_library()

    def _load_library(self):
        if self._library is not None:
            return self._library

        try:
            library = ctypes.CDLL(_MUSART_SONAME)
        except OSError as exc:
            raise MusaProfilerError(
                f"Cannot load {_MUSART_SONAME}; install the MUSA runtime and "
                "expose its library directory to the SGLang worker process."
            ) from exc

        missing = [
            name for name in _REQUIRED_MUSART_SYMBOLS if not hasattr(library, name)
        ]
        if missing:
            raise MusaProfilerError(
                f"{_MUSART_SONAME} is missing required profiler symbols: "
                + ", ".join(missing)
            )

        try:
            for name in (
                "musaProfilerStart",
                "musaProfilerStop",
                "musaGetLastError",
            ):
                function = getattr(library, name)
                function.argtypes = []
                function.restype = ctypes.c_int

            runtime_version = getattr(library, "musaRuntimeGetVersion", None)
            if runtime_version is not None:
                runtime_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
                runtime_version.restype = ctypes.c_int

            error_string = getattr(library, "musaGetErrorString", None)
            if error_string is not None:
                error_string.argtypes = [ctypes.c_int]
                error_string.restype = ctypes.c_char_p
        except (AttributeError, TypeError, ValueError) as exc:
            raise MusaProfilerError(
                f"Cannot configure the required {_MUSART_SONAME} profiler ABI"
            ) from exc

        self._library = library
        return library

    @staticmethod
    def _runtime_version(library) -> str:
        function = getattr(library, "musaRuntimeGetVersion", None)
        if function is None:
            return "unknown"

        version = ctypes.c_int()
        try:
            result = int(function(ctypes.byref(version)))
        except (AttributeError, OSError, TypeError, ValueError):
            return "unknown"
        return str(version.value) if result == MUSA_SUCCESS else "unknown"

    @staticmethod
    def _error_text(library, result: int) -> str:
        function = getattr(library, "musaGetErrorString", None)
        if function is None:
            return "unknown"

        try:
            value = function(result)
        except (AttributeError, OSError, TypeError, ValueError):
            return "unknown"
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value) if value else "unknown"

    def _call(self, name: str) -> int:
        library = self._load_library()
        try:
            result = int(getattr(library, name)())
        except Exception as exc:
            raise MusaProfilerError(
                f"Calling {name} from {_MUSART_SONAME} failed"
            ) from exc

        if result == MUSA_SUCCESS:
            return MUSA_SUCCESS

        try:
            cleared = int(library.musaGetLastError())
        except Exception as exc:
            raise MusaProfilerError(
                f"{name} returned MUSA error {result}, and musaGetLastError "
                "failed while clearing the sticky runtime error"
            ) from exc

        runtime_version = self._runtime_version(library)
        error_text = self._error_text(library, result)
        if result == MUSA_ERROR_NOT_SUPPORTED:
            # MUSA 4.3.x returns 801 under msys even after msys has consumed the
            # marker. Treat only that observed code as success; all other MUSA
            # errors remain fatal.
            logger.warning(
                "%s returned MUSA error 801 (%s), cleared error %d, runtime "
                "version %s; continuing because msys may already have accepted "
                "the capture-range marker",
                name,
                error_text,
                cleared,
                runtime_version,
            )
            return MUSA_SUCCESS

        raise MusaProfilerError(
            f"{name} returned MUSA error {result} ({error_text}); cleared error "
            f"{cleared}; runtime version {runtime_version}"
        )

    def start(self) -> int:
        return self._call("musaProfilerStart")

    def stop(self) -> int:
        return self._call("musaProfilerStop")


class MusaCudartProxy:
    """Override only CUDA profiler markers and delegate every other symbol."""

    def __init__(
        self,
        original_cudart: Callable[[], object],
        profiler_api: MusaProfilerApi,
    ) -> None:
        self._original_cudart = original_cudart
        self._profiler_api = profiler_api

    def cudaProfilerStart(self) -> int:
        return self._profiler_api.start()

    def cudaProfilerStop(self) -> int:
        return self._profiler_api.stop()

    def __getattr__(self, name: str):
        return getattr(self._original_cudart(), name)


_MUSA_PROFILER_API = MusaProfilerApi()
_MUSA_CUDART_PROXY = None
_patches_applied = False


def _torch_musa_activity():
    # TorchMUSA 2.9 registers the MUSA alias for PrivateUse1. Prefer the
    # underlying enum so the mapping does not depend on the optional alias.
    activity = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
    if activity is None:
        activity = getattr(torch.profiler.ProfilerActivity, "MUSA", None)
    if activity is None:
        raise MusaProfilerError(
            "This PyTorch/TorchMUSA build does not expose "
            "ProfilerActivity.PrivateUse1 or ProfilerActivity.MUSA"
        )
    return activity


def _install_torch_profiler_redirects() -> None:
    global _MUSA_CUDART_PROXY

    torch.profiler.ProfilerActivity.CUDA = _torch_musa_activity()

    original_cudart = torch.cuda.cudart
    _MUSA_CUDART_PROXY = MusaCudartProxy(original_cudart, _MUSA_PROFILER_API)

    @wraps(original_cudart)
    def musa_cudart():
        return _MUSA_CUDART_PROXY

    torch.cuda.cudart = musa_cudart


def apply_musa_profiler_patches() -> None:
    """Install the version-locked SGLang v0.5.11 profiler adaptation once.

    ``libmusart.so`` remains lazy: workers that never request
    ``CUDA_PROFILER`` do not need to load the capture-range marker ABI.
    """
    global _patches_applied
    if _patches_applied:
        return

    apply_sglang_0_5_11_profiler_lifecycle_patch(MusaProfilerError)
    _install_torch_profiler_redirects()
    _patches_applied = True
    logger.info(
        "MUSA profiler redirects applied for SGLang v0.5.11: "
        "GPU->PrivateUse1 and CUDA_PROFILER->musaProfilerStart/Stop"
    )
