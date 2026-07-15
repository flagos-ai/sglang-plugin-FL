# Copyright (c) 2026 BAAI. All rights reserved.

"""
Shared SGLang server lifecycle helper for serving E2E tests.

Provides ``SGLangServer`` — a context manager that starts an SGLang server
process, waits for readiness, and tears it down on exit.

Usage in test fixtures::

    @pytest.fixture(scope="module")
    def sglang_server():
        with SGLangServer(model_path=MODEL_PATH, tp_size=8) as srv:
            yield srv


    def test_completion(sglang_server):
        resp = requests.post(f"{sglang_server.base_url}/v1/completions", ...)
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import pytest
import requests

# Bypass HTTP proxies for local server connections.
_NO_PROXY = {"http": None, "https": None}


def _get_free_port() -> int:
    # SGLang derives an internal gRPC port from the HTTP port. Keep the HTTP
    # port low enough so derived ports stay within the valid 1-65535 range.
    for port in range(30000, 55536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free port found in range 30000-55535")


def _append_no_proxy(env: dict[str, str]) -> dict[str, str]:
    local_no_proxy = "127.0.0.1,localhost,::1"
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        env[key] = ",".join(filter(None, [current, local_no_proxy]))
    return env


@dataclass
class SGLangServer:
    """Manages an SGLang serving process lifecycle."""

    model_path: str
    tp_size: int = 1
    extra_args: list[str] = field(default_factory=list)
    host: str = "127.0.0.1"
    api_key: str = ""
    served_model_name: str = ""
    max_retries: int = 60
    poll_interval: int = 10

    # Set after start
    port: int = 0
    base_url: str = ""
    _process: subprocess.Popen | None = None
    _log_file: tempfile._TemporaryFileWrapper | None = None

    def start(self) -> None:
        if not self.port:
            self.port = _get_free_port()
        self.base_url = f"http://{self.host}:{self.port}/v1"

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tp-size",
            str(self.tp_size),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        if self.served_model_name:
            cmd.extend(["--served-model-name", self.served_model_name])
        cmd.extend(self.extra_args)

        model_short = os.path.basename(self.model_path)
        print(f"\n[Setup] Starting SGLang ({model_short}, TP={self.tp_size})")
        print(f"[Setup] Command: {' '.join(cmd)}")

        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix=f"sglang_{model_short}_",
            suffix=".log",
            delete=False,
        )
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=_append_no_proxy(os.environ.copy()),
        )
        self._wait_ready()

    def stop(self) -> None:
        if self._process is None:
            return
        print("\n[Teardown] Shutting down SGLang service...")
        try:
            self._process.terminate()
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()
        finally:
            if self._log_file:
                self._log_file.close()
                if getattr(self, "_keep_log", False):
                    print(f"[Teardown] Log preserved: {self._log_file.name}")
                else:
                    os.unlink(self._log_file.name)
            self._process = None

    def __enter__(self) -> "SGLangServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Use /health for readiness check. It returns 200 once the HTTP server
        # is accepting connections and the engine process is ready to serve.
        health_url = f"http://{self.host}:{self.port}/health"
        print(f"[Setup] Waiting for service to be ready (polling {health_url})...")
        for i in range(self.max_retries):
            if self._process and self._process.poll() is not None:
                self._fail_with_logs(
                    "SGLang process exited unexpectedly "
                    f"(code={self._process.returncode})"
                )

            try:
                resp = requests.get(
                    health_url, headers=headers, timeout=10, proxies=_NO_PROXY
                )
                if resp.status_code == 200:
                    print(f"[Setup] SGLang service ready (port={self.port})")
                    return
                detail = ""
                with contextlib.suppress(Exception):
                    detail = f" body={resp.text[:200]}"
                print(
                    f"[Setup] Waiting ({i + 1}/{self.max_retries})"
                    f" status={resp.status_code}{detail}"
                )
            except requests.exceptions.RequestException as exc:
                print(
                    f"[Setup] Waiting ({i + 1}/{self.max_retries}) {type(exc).__name__}"
                )

            time.sleep(self.poll_interval)

        self._fail_with_logs("SGLang service startup timed out")

    def _fail_with_logs(self, message: str) -> None:
        logs = ""
        log_path = ""
        if self._log_file:
            self._log_file.flush()
            log_path = self._log_file.name
            with open(log_path, encoding="utf-8", errors="replace") as f:
                logs = f.read()
        # Keep log file for post-mortem inspection.
        self._keep_log = True
        self.stop()
        tail = logs[-16000:] if len(logs) > 16000 else logs
        extra = f"\nFull log: {log_path}" if log_path else ""
        pytest.fail(
            f"{message}.{extra}\nLogs ({len(logs)} chars, showing tail):\n{tail}"
        )

