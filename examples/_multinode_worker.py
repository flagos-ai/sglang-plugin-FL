# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-process helpers shared by the multi-node verification examples."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable, TextIO


_READY_MARKER = "Dummy health check server started in background thread"
_SERVER_STARTED_MARKER = "Started server process"
_WAITING_FOR_STARTUP_MARKER = "Waiting for application startup."
_TRACEBACK_MARKER = "Traceback (most recent call last):"
_NONE_TOKENIZER_ERROR = (
    "AttributeError: 'NoneType' object has no attribute 'server_args'"
)
_APPLICATION_STARTUP_FAILURE = "Application startup failed. Exiting."
_KILL_PROCESS_TREE_MARKER = "kill_process_tree called:"
_SCHEDULER_EXIT_RE = re.compile(
    r"Scheduler or DataParallelController\s+(\d+)\s+terminated with\s+(-?\d+)"
)


@dataclass
class _WorkerExitEvidence:
    """Line positions needed to recognize one exact SGLang 0.5.18 bug."""

    line_number: int = 0
    ready_lines: list[int] = field(default_factory=list)
    scheduler_exits: list[tuple[int, int, int]] = field(default_factory=list)
    server_started_lines: list[int] = field(default_factory=list)
    waiting_for_startup_lines: list[int] = field(default_factory=list)
    traceback_lines: list[int] = field(default_factory=list)
    none_tokenizer_error_lines: list[int] = field(default_factory=list)
    application_startup_failure_lines: list[int] = field(default_factory=list)
    kill_process_tree_lines: list[int] = field(default_factory=list)

    def observe(self, line: str) -> None:
        self.line_number += 1
        position = self.line_number

        if _READY_MARKER in line:
            self.ready_lines.append(position)

        match = _SCHEDULER_EXIT_RE.search(line)
        if match:
            self.scheduler_exits.append(
                (position, int(match.group(1)), int(match.group(2)))
            )

        markers = (
            (_SERVER_STARTED_MARKER, self.server_started_lines),
            (_WAITING_FOR_STARTUP_MARKER, self.waiting_for_startup_lines),
            (_TRACEBACK_MARKER, self.traceback_lines),
            (_NONE_TOKENIZER_ERROR, self.none_tokenizer_error_lines),
            (_APPLICATION_STARTUP_FAILURE, self.application_startup_failure_lines),
            (_KILL_PROCESS_TREE_MARKER, self.kill_process_tree_lines),
        )
        for marker, positions in markers:
            if marker in line:
                positions.append(position)

    def is_known_v0518_clean_shutdown(
        self, returncode: int, expected_scheduler_count: int
    ) -> bool:
        """Match only the non-zero-rank HTTP-startup artifact in v0.5.18."""
        if returncode != 3 or expected_scheduler_count <= 0:
            return False

        singleton_markers = (
            self.ready_lines,
            self.server_started_lines,
            self.waiting_for_startup_lines,
            self.none_tokenizer_error_lines,
            self.application_startup_failure_lines,
            self.kill_process_tree_lines,
        )
        if any(len(positions) != 1 for positions in singleton_markers):
            return False

        if len(self.scheduler_exits) != expected_scheduler_count:
            return False

        scheduler_pids = {pid for _, pid, _ in self.scheduler_exits}
        if len(scheduler_pids) != expected_scheduler_count:
            return False
        if any(exit_code != 0 for _, _, exit_code in self.scheduler_exits):
            return False

        scheduler_lines = [position for position, _, _ in self.scheduler_exits]
        ready_line = self.ready_lines[0]
        first_scheduler_exit = min(scheduler_lines)
        last_scheduler_exit = max(scheduler_lines)
        # FlagGems may log recoverable tracebacks while the model initializes.
        # After readiness, however, the only accepted traceback is the known
        # Uvicorn lifespan failure after every local scheduler has exited.
        post_ready_tracebacks = [
            position for position in self.traceback_lines if position > ready_line
        ]
        if len(post_ready_tracebacks) != 1:
            return False
        shutdown_traceback = post_ready_tracebacks[0]

        # first_scheduler_exit == last_scheduler_exit is valid when this node
        # owns only one scheduler, so keep that boundary non-strict.
        return (
            ready_line < first_scheduler_exit <= last_scheduler_exit
            < self.server_started_lines[0]
            < self.waiting_for_startup_lines[0]
            < shutdown_traceback
            < self.none_tokenizer_error_lines[0]
            < self.application_startup_failure_lines[0]
            < self.kill_process_tree_lines[0]
        )


def normalize_sglang_v0518_worker_exit_code(
    returncode: int, expected_scheduler_count: int, output_lines: Iterable[str]
) -> int:
    """Return 0 only for the exact known clean-worker shutdown artifact."""
    evidence = _WorkerExitEvidence()
    for line in output_lines:
        evidence.observe(line)
    if evidence.is_known_v0518_clean_shutdown(returncode, expected_scheduler_count):
        return 0
    return returncode


def run_sglang_worker(
    cmd: list[str],
    expected_scheduler_count: int,
    *,
    output: TextIO | None = None,
) -> int:
    """Run a worker, stream its combined logs, and normalize one known rc=3."""
    if output is None:
        output = sys.stdout
    evidence = _WorkerExitEvidence()
    worker_env = os.environ.copy()
    worker_env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=worker_env,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            evidence.observe(line)
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if evidence.is_known_v0518_clean_shutdown(returncode, expected_scheduler_count):
        print(
            "Worker schedulers exited cleanly; normalizing the known "
            "SGLang v0.5.18 non-zero-rank HTTP shutdown artifact from "
            "exit code 3 to 0.",
            file=output,
            flush=True,
        )
        return 0
    return returncode
