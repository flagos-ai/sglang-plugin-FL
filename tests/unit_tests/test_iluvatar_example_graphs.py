# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("disabled", [None, "0", "1"])
@pytest.mark.parametrize(
    "name,factory",
    [
        ("qwen3_6_27b_offline_inference.py", "run_engine"),
        ("qwen3_6_35b_a3b_offline_inference.py", "run_engine"),
        ("qwen3_6_27b_concurrent.py", "_make_engine"),
        ("qwen3_6_35b_a3b_concurrent.py", "_make_engine"),
    ],
)
def test_acceptance_graph_switch_reaches_engine(monkeypatch, disabled, name, factory):
    class CapturedEngine(Exception):
        pass

    captured = {}

    def engine(**kwargs):
        captured.update(kwargs)
        raise CapturedEngine

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if disabled is None:
        monkeypatch.delenv("DISABLE_CUDA_GRAPH", raising=False)
    else:
        monkeypatch.setenv("DISABLE_CUDA_GRAPH", disabled)
    script = Path(__file__).resolve().parents[2] / "examples" / name
    namespace = runpy.run_path(str(script))
    with pytest.raises(CapturedEngine):
        namespace[factory]()
    assert captured["disable_cuda_graph"] is (disabled != "0")
