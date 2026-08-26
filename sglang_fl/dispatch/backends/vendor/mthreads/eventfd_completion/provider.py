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

"""Bounded MUSA stream completion using ``musaLaunchHostFunc`` and eventfd.

The accelerator Event recorded by SGLang is retained behind a proxy and is the
correctness fallback for build, enqueue, pool-exhaustion, and wait failures.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).resolve().parent / "csrc" / "musa_eventfd_completion.cpp"
_DEFAULT_CAPACITY = 64
_POOL: Optional["_MusaEventfdCompletionPool"] | bool = None
_POOL_LOCK = threading.Lock()
_ENQUEUE = None
_ENQUEUE_LOCK = threading.Lock()


def _musa_home() -> Path:
    return Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))


def _cache_root() -> Path:
    explicit = os.environ.get("SGLANG_MUSA_EVENTFD_CACHE_DIR")
    if explicit:
        return Path(explicit)
    jit_root = Path(
        os.environ.get(
            "SGLANG_MUSA_JIT_CACHE_DIR",
            Path.home() / ".cache" / "sglang_musa_jit",
        )
    )
    return jit_root / "eventfd_completion"


def _build_provider() -> Path:
    source_digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"musa_eventfd_completion_{source_digest}.so"
    lock_path = cache_root / "build.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        temporary = output.with_suffix(f".tmp.{os.getpid()}.so")
        musa_home = _musa_home()
        command = [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(_SOURCE),
            f"-I{musa_home / 'include'}",
            f"-L{musa_home / 'lib'}",
            "-lmusart",
            "-o",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return output


def _load_enqueue():
    global _ENQUEUE
    if _ENQUEUE is not None:
        return _ENQUEUE
    with _ENQUEUE_LOCK:
        if _ENQUEUE is not None:
            return _ENQUEUE
        library = ctypes.CDLL(str(_build_provider()))
        enqueue = library.enqueue_musa_eventfd_completion
        enqueue.argtypes = [ctypes.c_uint64, ctypes.c_int]
        enqueue.restype = ctypes.c_int
        # Keep the library alive until every queued host callback has run.
        enqueue._musa_eventfd_library = library
        _ENQUEUE = enqueue
        return enqueue


class _MusaEventfdCompletion:
    def __init__(self, pool: "_MusaEventfdCompletionPool", slot: int):
        self._pool = pool
        self._slot = slot
        self._lock = threading.Lock()
        self._done = False

    def wait(self) -> None:
        # Make repeated synchronize calls safe.  CPython releases the GIL
        # around eventfd_read, while the native stream callback does not need
        # the GIL to signal completion.
        with self._lock:
            if self._done:
                return
            try:
                payload = os.eventfd_read(self._pool.event_fd(self._slot))
                if payload != 1:
                    raise RuntimeError(
                        f"Invalid MUSA completion eventfd payload: {payload}"
                    )
            except Exception:
                # Do not recycle a descriptor if its callback state is
                # unknown.  The proxy will synchronize the original Event;
                # leaking one slot is safer than delivering a late signal to
                # a later result.
                self._pool.abandon()
                self._done = True
                raise
            self._done = True
            self._pool.release(self._slot)


class _MusaEventfdCompletionPool:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("MUSA eventfd pool capacity must be positive")
        if not hasattr(os, "eventfd"):
            raise RuntimeError("Linux eventfd is unavailable")
        enqueue = _load_enqueue()
        event_fds: list[int] = []
        try:
            for _ in range(capacity):
                event_fds.append(os.eventfd(0, os.EFD_CLOEXEC))
        except Exception:
            for event_fd in event_fds:
                os.close(event_fd)
            raise
        self._enqueue = enqueue
        self._event_fds = event_fds
        self._available = list(range(capacity - 1, -1, -1))
        self._lock = threading.Lock()
        self._supported = True
        self._exhaustion_logged = False

    def event_fd(self, slot: int) -> int:
        return self._event_fds[slot]

    def enqueue(self, stream: Any) -> Optional[_MusaEventfdCompletion]:
        with self._lock:
            if not self._supported or not self._available:
                if self._supported and not self._exhaustion_logged:
                    logger.warning(
                        "MUSA eventfd completion pool exhausted; using Event fallback"
                    )
                    self._exhaustion_logged = True
                return None
            slot = self._available.pop()
        try:
            status = self._enqueue(stream.musa_stream, self._event_fds[slot])
            if status != 0:
                raise RuntimeError(f"musaLaunchHostFunc returned {status}")
        except Exception:
            with self._lock:
                self._supported = False
                self._available.append(slot)
            logger.exception(
                "Failed to enqueue MUSA eventfd completion; using Event fallback"
            )
            return None
        return _MusaEventfdCompletion(self, slot)

    def release(self, slot: int) -> None:
        with self._lock:
            self._available.append(slot)
            self._exhaustion_logged = False

    def abandon(self) -> None:
        with self._lock:
            self._supported = False


def _pool_capacity() -> int:
    return int(
        os.environ.get(
            "SGLANG_MUSA_EVENTFD_COMPLETION_POOL_SIZE", str(_DEFAULT_CAPACITY)
        )
    )


def _get_pool(device: torch.device) -> Optional[_MusaEventfdCompletionPool]:
    global _POOL
    if device.type != "musa" or _POOL is False:
        return None
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                try:
                    _POOL = _MusaEventfdCompletionPool(_pool_capacity())
                except Exception:
                    _POOL = False
                    logger.exception(
                        "Failed to initialize MUSA eventfd completion; "
                        "using Event fallback"
                    )
    return _POOL if isinstance(_POOL, _MusaEventfdCompletionPool) else None


def try_enqueue_musa_completion(device: torch.device):
    pool = _get_pool(device)
    if pool is None:
        return None
    stream = torch.get_device_module(device).current_stream(device)
    return pool.enqueue(stream)


class MusaCompletionEventProxy:
    """Event-compatible synchronization proxy with an exact Event fallback."""

    def __init__(self, completion: _MusaEventfdCompletion, fallback_event: Any):
        self._completion = completion
        self._fallback_event = fallback_event
        self._lock = threading.Lock()
        self._done = False

    def synchronize(self) -> None:
        with self._lock:
            if self._done:
                return
            try:
                self._completion.wait()
            except Exception:
                logger.exception(
                    "MUSA eventfd completion wait failed; using Event fallback"
                )
                self._fallback_event.synchronize()
            self._done = True

    def __getattr__(self, name: str):
        return getattr(self._fallback_event, name)


def reset_for_test() -> None:
    """Reset lazy globals; only unit tests should call this helper."""
    global _POOL, _ENQUEUE
    _POOL = None
    _ENQUEUE = None


__all__ = ["MusaCompletionEventProxy", "try_enqueue_musa_completion"]
