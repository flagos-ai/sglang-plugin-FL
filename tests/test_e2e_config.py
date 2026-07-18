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

"""End-to-end validation of all sglang_fl env vars and YAML config options.

Starts SGLang server with each configuration, sends a request, verifies:
1. Server starts successfully
2. Inference returns valid output
3. Config-specific side effects are correct (dispatch log, backend selection, etc.)

Usage:
    MODEL_PATH=/path/to/model python test_e2e_config.py          # Run all tests
    MODEL_PATH=/path/to/model python test_e2e_config.py --list   # List all test cases
    MODEL_PATH=/path/to/model python test_e2e_config.py --case 3 # Run specific test case

Requirements:
    - MODEL_PATH env var pointing to a chat model
      (any small chat model works, e.g. Qwen2.5-0.5B-Instruct, Qwen3-4B)
      The model must support /v1/chat/completions and be able to answer "What is 2+3?"
    - GPU: 1x NVIDIA GPU (mem-fraction-static=0.4, ~4GB VRAM)
    - sglang_fl installed (pip install -e .)
"""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not MODEL_PATH:
    print("ERROR: MODEL_PATH environment variable is required.")
    print(
        "  Example: MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python test_e2e_config.py"
    )
    sys.exit(1)
BASE_PORT = 31100  # Starting port (increments per test to avoid conflicts)
PROMPT = "What is 2+3? Answer with just the number."
TIMEOUT_STARTUP = 120  # seconds to wait for server ready
TIMEOUT_INFERENCE = 30  # seconds to wait for inference response


# ─── Test Infrastructure ─────────────────────────────────────────────────────


@dataclass
class TestCase:
    name: str
    description: str
    env: Dict[str, str] = field(default_factory=dict)
    yaml_content: Optional[str] = (
        None  # If set, write to temp file and set SGLANG_FL_CONFIG
    )
    verify: Optional[Callable] = None  # Called with (response_text, artifacts_dict)
    expect_fail: bool = False  # If True, server startup failure is expected


def _server_cmd(port: int) -> list:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.4",
    ]


def _wait_for_server(port: int, timeout: int = TIMEOUT_STARTUP) -> bool:
    """Wait for server to be ready. Returns True if ready, False if timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def _send_request(port: int) -> Optional[str]:
    """Send inference request and return generated text."""
    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0,
                "max_tokens": 16,
            },
            timeout=TIMEOUT_INFERENCE,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"    Request failed: {e}")
    return None


def _kill_server(proc):
    """Kill server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def run_test(case: TestCase, port: int) -> bool:
    """Run a single test case. Returns True if passed."""
    print(f"\n{'=' * 70}")
    print(f"  [{case.name}] {case.description}")
    print(f"{'=' * 70}")

    # Prepare environment
    env = os.environ.copy()
    env.update(case.env)

    # Replace generic log paths with per-test paths for artifact isolation
    if "SGLANG_FL_DISPATCH_LOG" in env:
        env["SGLANG_FL_DISPATCH_LOG"] = f"/tmp/test_e2e_{case.name}_dispatch.log"
    if "SGLANG_FLAGGEMS_LOG_PATH" in env:
        env["SGLANG_FLAGGEMS_LOG_PATH"] = f"/tmp/test_e2e_{case.name}_gems.log"

    # Remove conflicting env vars from parent
    for key in list(env.keys()):
        if (
            key.startswith("SGLANG_FL_")
            and key not in case.env
            and key != "SGLANG_FL_DIST_BACKEND"
        ):
            del env[key]
        if key.startswith("SGLANG_FLAGGEMS_") and key not in case.env:
            del env[key]
        if key in ("SGLANG_FLAGGEMS_MODE", "USE_FLAGGEMS") and key not in case.env:
            del env[key]

    artifacts = {}
    yaml_path = None
    stdout_path = tempfile.mktemp(suffix=".server.log")

    # Write YAML config if needed
    if case.yaml_content:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(case.yaml_content)
        f.close()
        yaml_path = f.name
        env["SGLANG_FL_CONFIG"] = yaml_path
        print(f"    YAML: {yaml_path}")

    # Print env overrides
    for k, v in sorted(case.env.items()):
        print(f"    {k}={v}")

    # Start server — redirect stdout to temp file to avoid pipe buffer issues
    cmd = _server_cmd(port)
    stdout_file = open(stdout_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )

    passed = False
    try:
        # Wait for server
        print(f"    Waiting for server on port {port}...", end="", flush=True)
        ready = _wait_for_server(port)

        if not ready:
            if case.expect_fail:
                print(" EXPECTED FAIL (server didn't start)")
                # Collect server log for verification
                stdout_file.flush()
                with open(stdout_path) as f:
                    artifacts["server_log"] = f.read()
                # Run verify on failure case (e.g. check error message)
                if case.verify:
                    try:
                        case.verify(None, artifacts)
                        print("    VERIFY: PASS")
                        passed = True
                    except AssertionError as e:
                        print(f"    VERIFY: FAIL — {e}")
                        return False
                else:
                    passed = True
            else:
                print(" TIMEOUT")
                stdout_file.flush()
                with open(stdout_path) as f:
                    server_output = f.read()
                for line in server_output.split("\n")[-20:]:
                    if line.strip():
                        print(f"      | {line}")
            return passed

        print(" READY")

        if case.expect_fail:
            print("    UNEXPECTED: server started but expected failure")
            return False

        # Send inference request
        print(f"    Sending request: '{PROMPT}'")
        text = _send_request(port)
        if text is None:
            print("    FAIL: No response from server")
            return False

        print(f"    Response: '{text}'")
        artifacts["response"] = text

        # Check basic validity: response should contain "5"
        if "5" not in text:
            print(f"    WARN: Expected '5' in response, got: {text}")

        # Collect server log
        stdout_file.flush()
        with open(stdout_path) as f:
            artifacts["server_log"] = f.read()

        # Check config-specific artifacts (use env dict which has per-test paths)
        dispatch_log = env.get("SGLANG_FL_DISPATCH_LOG", "")
        if dispatch_log and os.path.exists(dispatch_log):
            with open(dispatch_log) as f:
                artifacts["dispatch_log"] = f.read()

        flaggems_log = env.get("SGLANG_FLAGGEMS_LOG_PATH", "")
        if flaggems_log and os.path.exists(flaggems_log):
            with open(flaggems_log) as f:
                artifacts["flaggems_log"] = f.read()

        # Run custom verification
        if case.verify:
            try:
                case.verify(text, artifacts)
                print("    VERIFY: PASS")
            except AssertionError as e:
                print(f"    VERIFY: FAIL — {e}")
                return False

        passed = True
        print("    RESULT: PASS")

    finally:
        _kill_server(proc)
        stdout_file.close()
        # Cleanup temp files
        if yaml_path and os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(stdout_path):
            os.unlink(stdout_path)
        dispatch_log = env.get("SGLANG_FL_DISPATCH_LOG", "")
        if dispatch_log and os.path.exists(dispatch_log):
            os.unlink(dispatch_log)
        flaggems_log = env.get("SGLANG_FLAGGEMS_LOG_PATH", "")
        if flaggems_log and os.path.exists(flaggems_log):
            os.unlink(flaggems_log)

    return passed


# ─── Verify Helpers ──────────────────────────────────────────────────────────


def _assert_op_using_backend(server_log: str, op_name: str, expected_backend: str):
    """Assert that server log shows op_name resolved to expected_backend."""
    # OpManager logs: Op 'rms_norm' using 'default.flagos' (mode=fallback-enabled)
    pattern = rf"Op '{op_name}' using '([^']+)'"
    match = re.search(pattern, server_log)
    assert match, f"Op '{op_name}' backend selection not found in server log"
    actual = match.group(1)
    assert expected_backend in actual, (
        f"Op '{op_name}': expected backend containing '{expected_backend}', got '{actual}'"
    )


def _assert_all_ops_using_backend(server_log: str, expected_backend: str):
    """Assert all 3 fused ops use the expected backend."""
    _assert_op_using_backend(server_log, "silu_and_mul", expected_backend)
    _assert_op_using_backend(server_log, "rms_norm", expected_backend)
    _assert_op_using_backend(server_log, "rotary_embedding", expected_backend)


# ─── Verify Functions ────────────────────────────────────────────────────────


def _verify_prefer_flagos(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    assert "RMSNorm" in log, "RMSNorm not in dispatch log"
    assert "RotaryEmbedding" in log, "RotaryEmbedding not in dispatch log"
    slog = artifacts.get("server_log", "")
    _assert_all_ops_using_backend(slog, "flagos")
    # Default strict=True → fallback-enabled mode
    assert "mode=fallback-enabled" in slog, (
        "Default strict=True should use fallback-enabled mode"
    )


def _verify_prefer_reference(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    assert "RMSNorm" in log, "RMSNorm not in dispatch log"
    assert "RotaryEmbedding" in log, "RotaryEmbedding not in dispatch log"
    slog = artifacts.get("server_log", "")
    _assert_all_ops_using_backend(slog, "reference")


def _verify_prefer_vendor(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    assert "RMSNorm" in log, "RMSNorm not in dispatch log"
    assert "RotaryEmbedding" in log, "RotaryEmbedding not in dispatch log"
    slog = artifacts.get("server_log", "")
    _assert_all_ops_using_backend(slog, "vendor")


def _verify_per_op(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    slog = artifacts.get("server_log", "")
    _assert_op_using_backend(slog, "rms_norm", "reference")
    _assert_op_using_backend(slog, "silu_and_mul", "flagos")


def _verify_blacklist(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "SiluAndMul" in log, "SiluAndMul should be in dispatch log"
    assert "RMSNorm" in log, "RMSNorm should be in dispatch log"
    assert "RotaryEmbedding" not in log, (
        "RotaryEmbedding should NOT be in dispatch log (blacklisted)"
    )


def _verify_whitelist(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "RMSNorm" in log, "RMSNorm should be in dispatch log (whitelisted)"
    assert "SiluAndMul" not in log, (
        "SiluAndMul should NOT be in dispatch log (not in whitelist)"
    )
    assert "RotaryEmbedding" not in log, (
        "RotaryEmbedding should NOT be in dispatch log (not in whitelist)"
    )


def _verify_oot_disabled(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert not log.strip(), (
        f"dispatch log should be empty when OOT disabled, got: {log[:100]}"
    )
    slog = artifacts.get("server_log", "")
    assert "Layer 2 (Fused Ops) disabled" in slog, (
        "server log should mention Layer 2 disabled"
    )


def _verify_strict_off(text, artifacts):
    """STRICT=0 disables fallback. Verify ops resolve in direct mode (no fallback chain)."""
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    assert "RMSNorm" in log, "RMSNorm not in dispatch log"
    assert "RotaryEmbedding" in log, "RotaryEmbedding not in dispatch log"
    slog = artifacts.get("server_log", "")
    # STRICT=0 → direct mode (no fallback chain)
    assert "mode=direct" in slog, (
        "STRICT=0 should use direct resolve mode, but 'mode=direct' not found in server log"
    )
    assert "mode=fallback-enabled" not in slog, (
        "STRICT=0 should NOT use fallback-enabled mode"
    )


def _verify_deny_vendors(text, artifacts):
    """DENY_VENDORS=cuda + PREFER=vendor: vendor.cuda denied, ops fall back."""
    slog = artifacts.get("server_log", "")
    # With vendor.cuda denied and prefer=vendor, ops should fall back to flagos/reference
    pattern = r"Op '\w+' using 'vendor\.cuda'"
    assert not re.search(pattern, slog), (
        "vendor.cuda should NOT be used when DENY_VENDORS=cuda"
    )
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    # Ops should still resolve (via fallback to flagos or reference)
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"


def _verify_use_flaggems_off(text, artifacts):
    # Layer 2 should still work
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty (Layer 2 should be active)"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    assert "RMSNorm" in log, "RMSNorm not in dispatch log"
    assert "RotaryEmbedding" in log, "RotaryEmbedding not in dispatch log"
    # Layer 1 (FlagGems) should be off
    gems_log = artifacts.get("flaggems_log", "")
    assert not gems_log.strip(), (
        f"FlagGems log should be empty (USE_FLAGGEMS=0), got: {gems_log[:100]}"
    )


def _verify_flagos_blacklist(text, artifacts):
    gems_log = artifacts.get("flaggems_log", "")
    assert gems_log.strip(), "FlagGems ATen log is empty"
    # Blacklisted ops should NOT appear in gems log
    lines = gems_log.strip().lower().split("\n")
    for line in lines:
        assert "aten.mul" not in line or "aten.mul_" in line, (
            f"'mul' should be blacklisted but found in gems log: {line}"
        )
    # Non-blacklisted ops should appear
    assert len(lines) > 0, "FlagGems log has no entries"


def _verify_flagos_whitelist(text, artifacts):
    gems_log = artifacts.get("flaggems_log", "")
    assert gems_log.strip(), "FlagGems ATen log is empty"
    lines = gems_log.strip().split("\n")
    # Only whitelisted ATen ops (add, addmm) should appear.
    # NOTE: Layer 2 fused ops (e.g. rms_norm, silu_and_mul) internally call FlagGems
    # and may appear as "[DEBUG] flag_gems.ops.xxx" lines — these are NOT Layer 1
    # ATen replacements, so we skip them.
    for line in lines:
        line_lower = line.strip().lower()
        if not line_lower:
            continue
        # Skip Layer 2 internal FlagGems calls (not ATen-level replacements)
        if "flag_gems.ops." in line_lower:
            continue
        has_whitelisted = "add" in line_lower or "mm" in line_lower
        assert has_whitelisted, (
            f"Only 'add'/'addmm' ATen ops should appear in gems log, got: {line}"
        )


def _verify_yaml_prefer_reference(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    slog = artifacts.get("server_log", "")
    _assert_all_ops_using_backend(slog, "reference")


def _verify_yaml_blacklist(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "SiluAndMul" in log, "SiluAndMul should be in dispatch log"
    assert "RMSNorm" in log, "RMSNorm should be in dispatch log"
    assert "RotaryEmbedding" not in log, (
        "RotaryEmbedding should NOT be in dispatch log (YAML blacklisted)"
    )


def _verify_yaml_op_backends(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    slog = artifacts.get("server_log", "")
    _assert_op_using_backend(slog, "rms_norm", "reference")


def _verify_yaml_env_override(text, artifacts):
    """YAML says reference, env says flagos. Env should win."""
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    slog = artifacts.get("server_log", "")
    _assert_all_ops_using_backend(slog, "flagos")


def _verify_log_level_debug(text, artifacts):
    slog = artifacts.get("server_log", "")
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    # DEBUG level should produce more verbose dispatch output
    # Check for typical debug-level messages from OpManager
    assert "Op '" in slog, "server log should contain Op selection info at DEBUG level"


def _verify_plugin_disabled(text, artifacts):
    log = artifacts.get("dispatch_log", "")
    assert not log.strip(), (
        f"dispatch log should be empty when plugin disabled, got: {log[:100]}"
    )


def _verify_conflict_error(text, artifacts):
    """Server starts but plugin fails to load. Check error in log + no dispatch."""
    slog = artifacts.get("server_log", "")
    assert "Cannot set both" in slog, (
        f"server log should contain conflict error, got last 300 chars: {slog[-300:]}"
    )
    # Plugin should NOT be active — dispatch log should be empty or absent
    log = artifacts.get("dispatch_log", "")
    assert not log.strip(), (
        f"dispatch log should be empty when plugin fails to load, got: {log[:100]}"
    )


def _verify_flaggems_log_once_off(text, artifacts):
    """LOG_ONCE=0 means every call is logged, not just first. Should have duplicates."""
    gems_log = artifacts.get("flaggems_log", "")
    assert gems_log.strip(), "FlagGems ATen log is empty"
    lines = [line for line in gems_log.strip().split("\n") if line.strip()]
    assert len(lines) > 0, "FlagGems ATen log has no entries"
    # With LOG_ONCE=0, we expect duplicate entries for the same op.
    # Count occurrences — at least one op should appear more than once.
    from collections import Counter

    op_counts = Counter()
    for line in lines:
        # Normalize to get op name
        line_lower = line.strip().lower()
        for op in ["add", "mul", "mm", "sub", "div"]:
            if f"aten.{op}" in line_lower:
                op_counts[op] += 1
    has_duplicates = any(c > 1 for c in op_counts.values())
    assert has_duplicates, (
        f"LOG_ONCE=0 should produce duplicate entries, but all ops appeared only once. "
        f"Counts: {dict(op_counts)}, total lines: {len(lines)}"
    )


def _verify_yaml_strict(text, artifacts):
    """YAML strict=false: direct resolve only, no fallback chain."""
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"
    slog = artifacts.get("server_log", "")
    # strict=false → direct mode
    assert "mode=direct" in slog, (
        "YAML strict=false should use direct resolve mode, but 'mode=direct' not found"
    )
    assert "mode=fallback-enabled" not in slog, (
        "YAML strict=false should NOT use fallback-enabled mode"
    )


def _verify_yaml_deny_vendors(text, artifacts):
    """YAML deny_vendors=[cuda] + prefer=vendor: vendor.cuda denied, ops fall back."""
    slog = artifacts.get("server_log", "")
    pattern = r"Op '\w+' using 'vendor\.cuda'"
    assert not re.search(pattern, slog), (
        "vendor.cuda should NOT be used when deny_vendors includes cuda"
    )
    log = artifacts.get("dispatch_log", "")
    assert "[OOT-DISPATCH]" in log, "dispatch log empty"
    assert "SiluAndMul" in log, "SiluAndMul not in dispatch log"


def _verify_yaml_flagos_blacklist(text, artifacts):
    """YAML flagos_blacklist=[mul, sub]: mul should not appear in gems log."""
    gems_log = artifacts.get("flaggems_log", "")
    assert gems_log.strip(), "FlagGems ATen log is empty"
    lines = gems_log.strip().lower().split("\n")
    for line in lines:
        assert "aten.mul" not in line or "aten.mul_" in line, (
            f"'mul' should be blacklisted but found in gems log: {line}"
        )
    assert len(lines) > 0, "FlagGems log has no entries"


# ─── Test Cases ──────────────────────────────────────────────────────────────

TESTS: List[TestCase] = [
    # ═══ Layer 2: SGLANG_FL_PREFER ═══════════════════════════════════════════
    TestCase(
        name="prefer-flagos",
        description="SGLANG_FL_PREFER=flagos → all ops use flagos backend",
        env={
            "SGLANG_FL_PREFER": "flagos",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_prefer_flagos,
    ),
    TestCase(
        name="prefer-reference",
        description="SGLANG_FL_PREFER=reference → all ops use reference backend",
        env={
            "SGLANG_FL_PREFER": "reference",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_prefer_reference,
    ),
    TestCase(
        name="prefer-vendor",
        description="SGLANG_FL_PREFER=vendor → all ops use vendor backend",
        env={
            "SGLANG_FL_PREFER": "vendor",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_prefer_vendor,
    ),
    # ═══ Layer 2: SGLANG_FL_PER_OP ═══════════════════════════════════════════
    TestCase(
        name="per-op",
        description="SGLANG_FL_PER_OP: rms_norm=reference, silu_and_mul=flagos",
        env={
            "SGLANG_FL_PER_OP": "rms_norm=reference;silu_and_mul=flagos",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_per_op,
    ),
    # ═══ Layer 2: SGLANG_FL_OOT_BLACKLIST ═════════════════════════════════════
    TestCase(
        name="blacklist",
        description="SGLANG_FL_OOT_BLACKLIST=RotaryEmbedding → RotaryEmbedding skipped",
        env={
            "SGLANG_FL_OOT_BLACKLIST": "RotaryEmbedding",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_blacklist,
    ),
    # ═══ Layer 2: SGLANG_FL_OOT_WHITELIST ═════════════════════════════════════
    TestCase(
        name="whitelist",
        description="SGLANG_FL_OOT_WHITELIST=RMSNorm → only RMSNorm dispatched",
        env={
            "SGLANG_FL_OOT_WHITELIST": "RMSNorm",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_whitelist,
    ),
    # ═══ Layer 2: SGLANG_FL_OOT_ENABLED=0 ════════════════════════════════════
    TestCase(
        name="oot-disabled",
        description="SGLANG_FL_OOT_ENABLED=0 → Layer 2 disabled, dispatch log empty",
        env={
            "SGLANG_FL_OOT_ENABLED": "0",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_oot_disabled,
    ),
    # ═══ Layer 2: SGLANG_FL_STRICT=0 ═════════════════════════════════════════
    TestCase(
        name="strict-off",
        description="SGLANG_FL_STRICT=0 → no fallback, direct resolve only",
        env={
            "SGLANG_FL_STRICT": "0",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_strict_off,
    ),
    # ═══ Layer 2: SGLANG_FL_DENY_VENDORS ═════════════════════════════════════
    TestCase(
        name="deny-vendors",
        description="SGLANG_FL_DENY_VENDORS=cuda → vendor.cuda excluded from resolve",
        env={
            "SGLANG_FL_DENY_VENDORS": "cuda",
            "SGLANG_FL_PREFER": "vendor",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_deny_vendors,
    ),
    # ═══ Layer 2: OOT_WHITELIST + OOT_BLACKLIST conflict ═══════════════════════
    TestCase(
        name="whitelist-blacklist-conflict",
        description="OOT_WHITELIST + OOT_BLACKLIST both set → plugin fails, server runs vanilla",
        env={
            "SGLANG_FL_OOT_WHITELIST": "RMSNorm",
            "SGLANG_FL_OOT_BLACKLIST": "RotaryEmbedding",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_conflict_error,
    ),
    # ═══ Layer 1: USE_FLAGGEMS=0 ═════════════════════════════════════════════
    TestCase(
        name="use-flaggems-off",
        description="USE_FLAGGEMS=0 → Layer 1 off, Layer 2 on",
        env={
            "USE_FLAGGEMS": "0",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
            "SGLANG_FLAGGEMS_RECORD": "1",
            "SGLANG_FLAGGEMS_LOG_PATH": "/tmp/test_e2e_gems.log",
        },
        verify=_verify_use_flaggems_off,
    ),
    # ═══ Layer 1: SGLANG_FL_FLAGOS_BLACKLIST ══════════════════════════════════
    TestCase(
        name="flagos-blacklist",
        description="SGLANG_FL_FLAGOS_BLACKLIST=mul,sub → excluded from ATen replacement",
        env={
            "SGLANG_FL_FLAGOS_BLACKLIST": "mul,sub",
            "SGLANG_FLAGGEMS_RECORD": "1",
            "SGLANG_FLAGGEMS_LOG_PATH": "/tmp/test_e2e_gems.log",
        },
        verify=_verify_flagos_blacklist,
    ),
    # ═══ Layer 1: SGLANG_FL_FLAGOS_WHITELIST ══════════════════════════════════
    TestCase(
        name="flagos-whitelist",
        description="SGLANG_FL_FLAGOS_WHITELIST=add,addmm → only these ATen ops replaced",
        env={
            "SGLANG_FL_FLAGOS_WHITELIST": "add,addmm",
            "SGLANG_FLAGGEMS_RECORD": "1",
            "SGLANG_FLAGGEMS_LOG_PATH": "/tmp/test_e2e_gems.log",
        },
        verify=_verify_flagos_whitelist,
    ),
    # ═══ Layer 1: FLAGOS WHITELIST + BLACKLIST conflict ═══════════════════════
    TestCase(
        name="flagos-whitelist-blacklist-conflict",
        description="FLAGOS_WHITELIST + FLAGOS_BLACKLIST both set → plugin fails, server runs vanilla",
        env={
            "SGLANG_FL_FLAGOS_WHITELIST": "add",
            "SGLANG_FL_FLAGOS_BLACKLIST": "mul",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_conflict_error,
    ),
    # ═══ YAML Config ═════════════════════════════════════════════════════════
    TestCase(
        name="yaml-prefer-reference",
        description="YAML prefer=reference → all ops use reference",
        yaml_content="prefer: reference\n",
        env={
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_prefer_reference,
    ),
    TestCase(
        name="yaml-blacklist",
        description="YAML oot_blacklist=[RotaryEmbedding] → RotaryEmbedding skipped",
        yaml_content="prefer: flagos\noot_blacklist:\n  - RotaryEmbedding\n",
        env={
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_blacklist,
    ),
    TestCase(
        name="yaml-op-backends",
        description="YAML op_backends rms_norm=[reference,flagos] → RMSNorm uses reference",
        yaml_content="prefer: flagos\nop_backends:\n  rms_norm: [reference, flagos]\n",
        env={
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_op_backends,
    ),
    TestCase(
        name="yaml-env-override",
        description="YAML=reference + env=flagos → env wins (flagos)",
        yaml_content="prefer: reference\n",
        env={
            "SGLANG_FL_PREFER": "flagos",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_env_override,
    ),
    # ═══ System / Debug ══════════════════════════════════════════════════════
    TestCase(
        name="log-level-debug",
        description="SGLANG_FL_LOG_LEVEL=DEBUG → verbose dispatch output",
        env={
            "SGLANG_FL_LOG_LEVEL": "DEBUG",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_log_level_debug,
    ),
    # ═══ Plugin disabled ═════════════════════════════════════════════════════
    TestCase(
        name="plugin-disabled",
        description="SGLANG_PLUGINS=__none__ → vanilla SGLang, dispatch log empty",
        env={
            "SGLANG_PLUGINS": "__none__",
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_plugin_disabled,
    ),
    # ═══ Layer 1: SGLANG_FLAGGEMS_LOG_ONCE=0 ════════════════════════════════
    TestCase(
        name="flaggems-log-once-off",
        description="SGLANG_FLAGGEMS_LOG_ONCE=0 → every ATen call logged (duplicates)",
        env={
            "SGLANG_FLAGGEMS_LOG_ONCE": "0",
            "SGLANG_FLAGGEMS_RECORD": "1",
            "SGLANG_FLAGGEMS_LOG_PATH": "/tmp/test_e2e_gems.log",
        },
        verify=_verify_flaggems_log_once_off,
    ),
    # ═══ YAML: strict=false ═════════════════════════════════════════════════
    TestCase(
        name="yaml-strict",
        description="YAML strict=false → direct resolve only, no fallback",
        yaml_content="prefer: flagos\nstrict: false\n",
        env={
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_strict,
    ),
    # ═══ YAML: deny_vendors ═════════════════════════════════════════════════
    TestCase(
        name="yaml-deny-vendors",
        description="YAML deny_vendors=[cuda] + prefer=vendor → vendor.cuda excluded",
        yaml_content="prefer: vendor\ndeny_vendors:\n  - cuda\n",
        env={
            "SGLANG_FL_DISPATCH_LOG": "/tmp/test_e2e_dispatch.log",
        },
        verify=_verify_yaml_deny_vendors,
    ),
    # ═══ YAML: flagos_blacklist ═════════════════════════════════════════════
    TestCase(
        name="yaml-flagos-blacklist",
        description="YAML flagos_blacklist=[mul, sub] → mul excluded from ATen replacement",
        yaml_content="prefer: flagos\nflagos_blacklist:\n  - mul\n  - sub\n",
        env={
            "SGLANG_FLAGGEMS_RECORD": "1",
            "SGLANG_FLAGGEMS_LOG_PATH": "/tmp/test_e2e_gems.log",
        },
        verify=_verify_yaml_flagos_blacklist,
    ),
]


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    if "--list" in sys.argv:
        print(f"\n{'=' * 70}")
        print(f"  sglang_fl E2E Config Validation — {len(TESTS)} test cases")
        print(f"{'=' * 70}\n")
        for i, t in enumerate(TESTS, 1):
            print(f"  [{i:2d}] {t.name:35s} — {t.description}")
        print()
        return

    # Select specific case(s)
    if "--case" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--case") + 1])
        tests_to_run = [TESTS[idx - 1]]
        start_idx = idx - 1
    else:
        tests_to_run = TESTS
        start_idx = 0

    print(f"\n{'=' * 70}")
    print("  sglang_fl E2E Config Validation")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Tests: {len(tests_to_run)}")
    print(f"{'=' * 70}")

    results = []
    for i, case in enumerate(tests_to_run):
        port = BASE_PORT + start_idx + i

        passed = run_test(case, port)
        results.append((case.name, passed))

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}\n")

    passed_count = sum(1 for _, p in results if p)
    failed_count = len(results) - passed_count

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        mark = "  " if passed else "X "
        print(f"  [{mark}] {name:35s} {status}")

    print(
        f"\n  Total: {len(results)} | Passed: {passed_count} | Failed: {failed_count}"
    )
    print(f"{'=' * 70}\n")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
