# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for strict SGLang multi-node worker exit normalization."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


_HELPER_PATH = Path(__file__).parents[2] / "examples" / "_multinode_worker.py"
_SPEC = importlib.util.spec_from_file_location("_multinode_worker", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

normalize_worker_exit = _MODULE.normalize_sglang_v0518_worker_exit_code
run_sglang_worker = _MODULE.run_sglang_worker


class _FlushCountingStringIO(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def _known_clean_shutdown_lines() -> list[str]:
    return [
        "Dummy health check server started in background thread at 127.0.0.1:30000\n",
        "Scheduler or DataParallelController 101 terminated with 0\n",
        "Scheduler or DataParallelController 102 terminated with 0\n",
        "INFO: Started server process [100]\n",
        "INFO: Waiting for application startup.\n",
        "ERROR: Traceback (most recent call last):\n",
        "AttributeError: 'NoneType' object has no attribute 'server_args'\n",
        "ERROR: Application startup failed. Exiting.\n",
        "kill_process_tree called: parent_pid=100, include_parent=False, pid=100\n",
    ]


def test_normalizes_only_exact_known_v0518_shutdown():
    assert normalize_worker_exit(3, 2, _known_clean_shutdown_lines()) == 0


def test_allows_recoverable_tracebacks_before_worker_readiness():
    lines = [
        "FlagGems fallback: Traceback (most recent call last):\n",
        *_known_clean_shutdown_lines(),
    ]
    assert normalize_worker_exit(3, 2, lines) == 0


def test_normalizes_clean_shutdown_with_one_local_scheduler():
    lines = [
        line
        for line in _known_clean_shutdown_lines()
        if " 102 terminated" not in line
    ]
    assert normalize_worker_exit(3, 1, lines) == 0


def test_rejects_additional_traceback_while_schedulers_are_exiting():
    lines = _known_clean_shutdown_lines()
    lines.insert(2, "ERROR: Traceback (most recent call last):\n")
    assert normalize_worker_exit(3, 2, lines) == 3


@pytest.mark.parametrize(
    ("returncode", "expected_count", "mutate", "expected_returncode"),
    [
        (1, 2, lambda lines: lines, 1),
        (3, 0, lambda lines: lines, 3),
        (3, 2, lambda lines: lines[1:], 3),
        (
            3,
            2,
            lambda lines: [line for line in lines if " 102 terminated" not in line],
            3,
        ),
        (
            3,
            2,
            lambda lines: (
                lines[:3]
                + ["Scheduler or DataParallelController 103 terminated with 0\n"]
                + lines[3:]
            ),
            3,
        ),
        (
            3,
            2,
            lambda lines: [line.replace("102", "101") for line in lines],
            3,
        ),
        (
            3,
            2,
            lambda lines: [
                line.replace("102 terminated with 0", "102 terminated with 1")
                for line in lines
            ],
            3,
        ),
        (
            3,
            2,
            lambda lines: [
                line
                for line in lines
                if "NoneType' object has no attribute 'server_args" not in line
            ],
            3,
        ),
        (
            3,
            2,
            lambda lines: [
                line for line in lines if "Application startup failed" not in line
            ],
            3,
        ),
        (
            3,
            2,
            lambda lines: [lines[0], lines[5], *lines[1:5], *lines[6:]],
            3,
        ),
    ],
)
def test_preserves_all_other_failures(
    returncode, expected_count, mutate, expected_returncode
):
    lines = mutate(_known_clean_shutdown_lines())
    assert (
        normalize_worker_exit(returncode, expected_count, lines) == expected_returncode
    )


def test_run_worker_streams_combined_output_and_preserves_unknown_failure():
    stdout = _FlushCountingStringIO()
    code = (
        "import sys; "
        "print('worker stdout', flush=True); "
        "print('worker stderr', file=sys.stderr, flush=True); "
        "raise SystemExit(3)"
    )

    returncode = run_sglang_worker(
        [sys.executable, "-c", code], expected_scheduler_count=2, output=stdout
    )

    assert returncode == 3
    assert stdout.getvalue().splitlines() == ["worker stdout", "worker stderr"]
    assert stdout.flush_count == 2


def test_run_worker_normalizes_the_exact_known_shutdown():
    stdout = io.StringIO()
    lines = _known_clean_shutdown_lines()
    code = (
        "import sys; "
        f"lines = {lines!r}; "
        "[(sys.stdout.write(line), sys.stdout.flush()) for line in lines]; "
        "raise SystemExit(3)"
    )

    returncode = run_sglang_worker(
        [sys.executable, "-c", code], expected_scheduler_count=2, output=stdout
    )

    assert returncode == 0
    assert stdout.getvalue().startswith("".join(lines))
    assert "normalizing the known SGLang v0.5.18" in stdout.getvalue()
