# Core operator dispatch manager.

from __future__ import annotations

import atexit
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple

from .registry import OpRegistry
from .policy import SelectionPolicy, get_policy
from .types import OpImpl, BackendImplKind, match_token
from .logger_manager import get_logger

logger = get_logger()

# Debug printing control
_DISPATCH_DEBUG = os.getenv("SGLANG_FL_DISPATCH_DEBUG", "0") == "1"

# Dispatch timing instrumentation (enable with SGLANG_FL_DISPATCH_TIMING=1)
_DISPATCH_TIMING = os.getenv("SGLANG_FL_DISPATCH_TIMING", "0") == "1"

# Disable call() cache for A/B testing (simulates pre-fix behavior)
_DISABLE_CALL_CACHE = os.getenv("SGLANG_FL_DISABLE_CALL_CACHE", "0") == "1"


@dataclass
class _DispatchTimingStats:
    """Accumulates dispatch call timing statistics."""

    total_calls: int = 0
    cache_hits: int = 0
    total_resolve_ns: int = 0
    per_op_calls: Dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_hit(self, op_name: str) -> None:
        with self.lock:
            self.total_calls += 1
            self.cache_hits += 1
            self.per_op_calls[op_name] = self.per_op_calls.get(op_name, 0) + 1
            if self.total_calls % 5000 == 0:
                self._dump()

    def record_miss(self, op_name: str, elapsed_ns: int) -> None:
        with self.lock:
            self.total_calls += 1
            self.total_resolve_ns += elapsed_ns
            self.per_op_calls[op_name] = self.per_op_calls.get(op_name, 0) + 1
            if self.total_calls % 5000 == 0:
                self._dump()

    def _dump(self) -> None:
        """Periodically write stats to file for subprocess visibility."""
        try:
            path = f"/tmp/sglang_fl_dispatch_timing_{os.getpid()}.log"
            with open(path, "w") as f:
                f.write(self.summary() + "\n")
        except Exception:
            pass

    def summary(self) -> str:
        misses = self.total_calls - self.cache_hits
        avg_resolve_us = (self.total_resolve_ns / misses / 1000) if misses > 0 else 0
        total_resolve_ms = self.total_resolve_ns / 1e6
        lines = [
            "=" * 60,
            "SGLang-FL Dispatch Timing Report",
            "=" * 60,
            f"  Total call() invocations : {self.total_calls}",
            f"  Cache hits               : {self.cache_hits}",
            f"  Cache misses (resolved)  : {misses}",
            f"  Total resolve time       : {total_resolve_ms:.3f} ms",
            f"  Avg resolve time / miss  : {avg_resolve_us:.1f} μs",
            f"  Unique ops dispatched    : {len(self.per_op_calls)}",
            "=" * 60,
        ]
        return "\n".join(lines)


_timing_stats: Optional[_DispatchTimingStats] = None
_timing_log_path: Optional[str] = None

if _DISPATCH_TIMING:
    _timing_stats = _DispatchTimingStats()
    _timing_log_path = os.environ.get(
        "SGLANG_FL_DISPATCH_TIMING_LOG",
        f"/tmp/sglang_fl_dispatch_timing_{os.getpid()}.log",
    )

    def _print_timing_report():
        if _timing_stats and _timing_stats.total_calls > 0:
            report = _timing_stats.summary()
            logger.info("\n" + report)
            import sys

            print(report, file=sys.stderr)
            # Write to file for subprocess visibility
            if _timing_log_path:
                try:
                    with open(_timing_log_path, "w") as f:
                        f.write(report + "\n")
                except Exception:
                    pass

    atexit.register(_print_timing_report)

    # Also dump on SIGUSR1 for subprocess visibility
    import signal

    def _sigusr1_handler(signum, frame):
        _print_timing_report()

    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
    except (OSError, ValueError):
        pass


@dataclass
class _OpManagerState:
    """Internal state for OpManager."""

    init_pid: int = -1
    initialized: bool = False
    policy_epoch: int = 0


class OpManager:
    """
    Main manager for operator dispatching and selection.

    Responsibilities:
    - Lazy initialization and plugin discovery
    - Multi-process safety (PID detection + at_fork)
    - Policy-based operator selection
    - Dispatch caching with invalidation
    - Ordered fallback when strict mode is enabled
    """

    def __init__(self, registry: Optional[OpRegistry] = None) -> None:
        self._lock = threading.RLock()
        self._registry = registry or OpRegistry()
        self._state = _OpManagerState()
        self._dispatch_cache: Dict[Tuple[str, str, int], Callable] = {}
        self._called_ops: Dict[str, str] = {}
        self._failed_impls: Dict[str, Set[str]] = {}

        try:
            os.register_at_fork(after_in_child=self._reset_after_fork)
        except AttributeError:
            pass

    @property
    def registry(self) -> OpRegistry:
        return self._registry

    def _reset_after_fork(self) -> None:
        with self._lock:
            self._state.initialized = False
            self._state.init_pid = -1
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()
            self._called_ops.clear()
            self._failed_impls.clear()

    def bump_policy_epoch(self) -> None:
        with self._lock:
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()
            self._failed_impls.clear()

    def ensure_initialized(self) -> None:
        """
        Ensure the manager is initialized in the current process.
        Registers built-in operator implementations on first call.
        """
        with self._lock:
            pid = os.getpid()
            if self._state.initialized and self._state.init_pid == pid:
                return

            self._state.initialized = True
            self._state.init_pid = pid

            # Register built-in operators
            from . import builtin_ops

            builtin_ops.register_builtins(self._registry)

            self._state.policy_epoch += 1
            self._dispatch_cache.clear()

            snap = self._registry.snapshot()
            total_ops = len(snap.impls_by_op)
            total_impls = sum(len(impls) for impls in snap.impls_by_op.values())
            logger.info(
                f"OpManager initialized: {total_ops} ops with {total_impls} implementations"
            )

            if _DISPATCH_DEBUG:
                self._print_registered_operators()

    def _print_registered_operators(self) -> None:
        snap = self._registry.snapshot()
        logger.info("\n" + "=" * 70)
        logger.info("SGLang-FL Dispatch: Registered Operators")
        logger.info("=" * 70)
        for op_name, impls in sorted(snap.impls_by_op.items()):
            logger.info(f"\n[Operator: {op_name}]")
            sorted_impls = sorted(
                impls, key=lambda x: (x.priority, x.impl_id), reverse=True
            )
            for impl in sorted_impls:
                available = "Y" if impl.is_available() else "N"
                vendor_info = f", vendor={impl.vendor}" if impl.vendor else ""
                logger.info(
                    f"  [{available}] {impl.impl_id} "
                    f"(kind={impl.kind.value}, priority={impl.priority}{vendor_info})"
                )
        logger.info("\n" + "=" * 70 + "\n")

    def _matches_vendor_filters(self, impl: OpImpl, policy: SelectionPolicy) -> bool:
        if impl.kind != BackendImplKind.VENDOR:
            return True
        if impl.vendor is None:
            return False
        if impl.vendor in policy.deny_vendors:
            return False
        if policy.allow_vendors is not None and impl.vendor not in policy.allow_vendors:
            return False
        return True

    def resolve(self, op_name: str) -> Callable:
        """
        Resolve the best implementation for an operator.

        Selection: cache → filter by policy → check availability → sort by order → cache.
        """
        self.ensure_initialized()

        policy = get_policy()
        policy_fp = policy.fingerprint()
        epoch = self._state.policy_epoch

        cache_key = (op_name, policy_fp, epoch)
        cached = self._dispatch_cache.get(cache_key)
        if cached is not None:
            return cached

        snap = self._registry.snapshot()
        candidates = list(snap.impls_by_op.get(op_name, []))

        # Filter by vendor policy
        candidates = [c for c in candidates if self._matches_vendor_filters(c, policy)]

        # Filter by availability
        available = []
        for c in candidates:
            try:
                if c.is_available():
                    available.append(c)
            except Exception:
                continue
        candidates = available

        if not candidates:
            raise RuntimeError(
                f"No available implementation for op='{op_name}'. "
                f"Registered: {[impl.impl_id for impl in snap.impls_by_op.get(op_name, [])]}"
            )

        # Get selection order
        order = policy.per_op_order_dict.get(op_name) or policy.get_default_order()

        # Select best
        chosen: Optional[OpImpl] = None
        for token in order:
            matches = [c for c in candidates if match_token(c, token)]
            if matches:
                matches.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
                chosen = matches[0]
                break

        if chosen is None:
            raise RuntimeError(
                f"No implementation selected for op='{op_name}'. "
                f"Candidates: {[c.impl_id for c in candidates]}, Order: {order}"
            )

        self._dispatch_cache[cache_key] = chosen.fn

        if _DISPATCH_DEBUG:
            vendor_info = f", vendor={chosen.vendor}" if chosen.vendor else ""
            logger.info(
                f"[DISPATCH] Op '{op_name}' -> '{chosen.impl_id}' "
                f"(kind={chosen.kind.value}{vendor_info})"
            )

        return chosen.fn

    def resolve_candidates(self, op_name: str) -> list[OpImpl]:
        """Resolve all available implementations sorted by policy order."""
        self.ensure_initialized()
        policy = get_policy()

        snap = self._registry.snapshot()
        candidates = list(snap.impls_by_op.get(op_name, []))
        candidates = [c for c in candidates if self._matches_vendor_filters(c, policy)]

        available = []
        for c in candidates:
            try:
                if c.is_available():
                    available.append(c)
            except Exception:
                continue
        candidates = available

        if not candidates:
            raise RuntimeError(f"No available implementation for op='{op_name}'.")

        order = policy.per_op_order_dict.get(op_name) or policy.get_default_order()

        sorted_candidates = []
        for token in order:
            matches = [c for c in candidates if match_token(c, token)]
            if matches:
                matches.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
                sorted_candidates.extend(matches)

        seen = set()
        unique = []
        for c in sorted_candidates:
            if c.impl_id not in seen:
                seen.add(c.impl_id)
                unique.append(c)

        return unique if unique else candidates

    def call(self, op_name: str, *args, **kwargs):
        """
        Resolve and call an operator implementation with optional fallback.

        When strict=True in the policy (default), tries alternative implementations
        if primary fails. When strict=False, uses direct resolve only.
        """
        policy = get_policy()
        enable_fallback = policy.strict

        if not enable_fallback:
            fn = self.resolve(op_name)
            impl_id = self._get_impl_id_for_fn(op_name, fn)
            self._log_first_call(op_name, impl_id, mode="direct")
            return fn(*args, **kwargs)

        # Fallback mode: check cache first (same cache as resolve())
        policy_fp = policy.fingerprint()
        epoch = self._state.policy_epoch
        cache_key = (op_name, policy_fp, epoch)
        if not _DISABLE_CALL_CACHE:
            cached_fn = self._dispatch_cache.get(cache_key)
            if cached_fn is not None:
                if _timing_stats is not None:
                    _timing_stats.record_hit(op_name)
                return cached_fn(*args, **kwargs)

        # Cache miss: full resolve with fallback
        _t0 = time.perf_counter_ns() if _timing_stats is not None else 0
        candidates = self.resolve_candidates(op_name)
        failed_impl_ids = self._failed_impls.get(op_name, set())
        available_candidates = [
            impl for impl in candidates if impl.impl_id not in failed_impl_ids
        ]

        if not available_candidates:
            raise RuntimeError(
                f"All implementations for op='{op_name}' have failed previously. "
                f"Failed: {failed_impl_ids}"
            )

        last_error = None
        for idx, impl in enumerate(available_candidates):
            try:
                if idx == 0:
                    self._log_first_call(op_name, impl.impl_id, mode="fallback-enabled")
                else:
                    logger.info(
                        f"Op '{op_name}' fallback to '{impl.impl_id}' "
                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                    )

                result = impl.fn(*args, **kwargs)

                # Cache the successful impl for future calls
                self._dispatch_cache[cache_key] = impl.fn

                if _timing_stats is not None:
                    _timing_stats.record_miss(op_name, time.perf_counter_ns() - _t0)

                if idx > 0:
                    with self._lock:
                        self._called_ops[op_name] = impl.impl_id
                return result

            except Exception as e:
                last_error = e
                with self._lock:
                    if op_name not in self._failed_impls:
                        self._failed_impls[op_name] = set()
                    self._failed_impls[op_name].add(impl.impl_id)

                if idx < len(available_candidates) - 1:
                    logger.warning(
                        f"Implementation '{impl.impl_id}' failed for op '{op_name}': {e}"
                    )

        raise RuntimeError(
            f"All implementations failed for op='{op_name}'. Last error: {last_error}"
        ) from last_error

    def _get_impl_id_for_fn(self, op_name: str, fn: Callable) -> str:
        snap = self._registry.snapshot()
        for impl in snap.impls_by_op.get(op_name, []):
            if impl.fn is fn:
                return impl.impl_id
        return "unknown"

    def _log_first_call(
        self, op_name: str, impl_id: str, mode: str = "default"
    ) -> None:
        last = self._called_ops.get(op_name)
        if last != impl_id:
            with self._lock:
                if self._called_ops.get(op_name) != impl_id:
                    if last is None:
                        msg = f"Op '{op_name}' using '{impl_id}' (mode={mode})"
                    else:
                        msg = f"Op '{op_name}' switched to '{impl_id}' (mode={mode})"
                    logger.info(msg)
                    self._called_ops[op_name] = impl_id
                    # Also write to dispatch log file if configured
                    self._write_dispatch_log(op_name, impl_id)

    def _write_dispatch_log(self, op_name: str, impl_id: str) -> None:
        """Write FLA op dispatch info to SGLANG_FL_DISPATCH_LOG if configured."""
        log_path = os.environ.get("SGLANG_FL_DISPATCH_LOG", "").strip()
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(f"[OOT-DISPATCH] {op_name} → {impl_id}\n")
                    f.flush()
            except Exception:
                pass


# Global singleton
_default_manager: Optional[OpManager] = None
_manager_lock = threading.RLock()


def get_default_manager() -> OpManager:
    """Get or create the global default OpManager instance."""
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = OpManager()
    return _default_manager


def reset_default_manager() -> None:
    """Reset the global default OpManager (useful for testing)."""
    global _default_manager
    with _manager_lock:
        _default_manager = None
