# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the launcher with a recording interpreter, without starting a service."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/minimax_m31_musa_w8a8.sh"
DIAGNOSTICS = (
    "SGLANG_FLAGGEMS_RECORD",
    "SGLANG_FL_DISPATCH_DEBUG",
    "SGLANG_FLAGGEMS_LOG_PATH",
    "SGLANG_FL_DISPATCH_LOG",
)


@pytest.mark.parametrize("mode", [None, "0", "1"])
@pytest.mark.parametrize("custom_paths", [False, True])
def test_graph_launch_defaults_and_explicit_diagnostics(tmp_path, mode, custom_paths):
    interpreter = tmp_path / "record-interpreter"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"keys = {(*DIAGNOSTICS, 'USE_FLAGGEMS')!r}\n"
        "print(json.dumps({'env': {k: os.environ[k] for k in keys "
        "if k in os.environ}, 'args': sys.argv[1:]}))\n"
    )
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in DIAGNOSTICS}
    env.update(MODEL_PATH=str(tmp_path / "model"), PYTHON_BIN=str(interpreter))
    if mode is not None:
        env.update(SGLANG_FLAGGEMS_RECORD=mode, SGLANG_FL_DISPATCH_DEBUG=mode)
    if custom_paths:
        env.update(
            SGLANG_FLAGGEMS_LOG_PATH=str(tmp_path / "gems.log"),
            SGLANG_FL_DISPATCH_LOG=str(tmp_path / "dispatch.log"),
        )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--random-seed", "42"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    actual = observed["env"]
    assert actual["USE_FLAGGEMS"] == "1"
    assert actual["SGLANG_FLAGGEMS_RECORD"] == (mode or "0")
    assert actual["SGLANG_FL_DISPATCH_DEBUG"] == (mode or "0")
    for variable, filename, default in (
        ("SGLANG_FLAGGEMS_LOG_PATH", "gems.log", "/tmp/minimax-m31-flaggems.log"),
        ("SGLANG_FL_DISPATCH_LOG", "dispatch.log", "/tmp/minimax-m31-dispatch.log"),
    ):
        if mode == "1":
            assert actual[variable] == (
                str(tmp_path / filename) if custom_paths else default
            )
        else:
            assert variable not in actual
    assert observed["args"][:2] == ["-m", "sglang.launch_server"]
    args = observed["args"]
    assert args[args.index("--cuda-graph-backend-decode") + 1] == "full"
    assert args[args.index("--cuda-graph-backend-prefill") + 1] == "disabled"
    start = args.index("--cuda-graph-bs-decode") + 1
    assert args[start : start + 4] == ["1", "2", "4", "8"]
    assert "--disable-cuda-graph" not in args
    assert observed["args"][-2:] == ["--random-seed", "42"]
    assert not (tmp_path / "gems.log").exists()
    assert not (tmp_path / "dispatch.log").exists()
