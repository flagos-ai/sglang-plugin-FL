# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from tests.musa.validate_qwen36 import (
    compare_records,
    normalize_output,
    read_engine_config,
    record_run,
    run_wave,
)


def _output(token=42):
    return {
        "text": "answer",
        "meta_info": {
            "completion_tokens": 1,
            "output_token_logprobs": [[-0.5, token, None]],
            "id": "server-id",
        },
    }


def _record(label="reference"):
    return {
        "schema_version": 1,
        "label": label,
        "status": "complete",
        "protocol": {"rounds": 2, "concurrency": [1, 4], "prompts": ["p"]},
        "records": [
            {
                "round": r,
                "concurrency": c,
                "prompt_index": 0,
                **normalize_output(_output()),
            }
            for r in (0, 1)
            for c in (1, 4)
        ],
    }


def test_comparison_distinguishes_self_drift_and_cross_service_mismatch():
    reference = _record()
    candidate = deepcopy(reference)
    candidate["label"] = "candidate"
    assert not any(compare_records(reference, candidate).values())
    candidate["records"][-1]["logprobs"] = [-0.4]
    result = compare_records(reference, candidate)
    assert result["reference_self_drift"] == []
    assert result["candidate_self_drift"] == [
        {"request": (1, 4, 0), "fields": ["logprobs"]}
    ]
    assert result["between_services"] == result["candidate_self_drift"]
    reference["records"][-1]["token_ids"] = [43]
    assert compare_records(reference, candidate)["reference_self_drift"]


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "error", "empty", "partial", "protocol"]
)
def test_incomplete_or_incomparable_records_cannot_pass(corruption):
    reference, candidate = _record(), _record("candidate")
    if corruption == "missing":
        candidate["records"].pop()
    elif corruption == "duplicate":
        candidate["records"].append(candidate["records"][0])
    elif corruption == "error":
        candidate["records"][0]["error"] = "launch failed"
    elif corruption == "empty":
        candidate["records"][0]["token_ids"] = []
    elif corruption == "partial":
        candidate["status"] = "running"
    else:
        candidate["protocol"]["engine"] = {"disable_cuda_graph": True}
    with pytest.raises(ValueError):
        compare_records(reference, candidate)


def test_output_requires_complete_native_token_records():
    output = _output()
    assert normalize_output(output)["token_ids"] == [42]
    output["meta_info"]["completion_tokens"] = 2
    with pytest.raises(ValueError, match="incomplete"):
        normalize_output(output)
    output["meta_info"].pop("output_token_logprobs")
    with pytest.raises(ValueError, match="missing"):
        normalize_output(output)


def test_wave_preserves_request_order_and_failure_records():
    calls = []

    class Engine:
        async def async_generate(self, **kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0)
            if kwargs["prompt"] == "bad":
                raise RuntimeError("device failure")
            output = _output()
            output["meta_info"]["id"] = kwargs["rid"]
            return output

    records = asyncio.run(
        run_wave(Engine(), ["good", "bad"], {"max_new_tokens": 1}, 2, 4, 0, "run")
    )
    assert [row["prompt_index"] for row in records] == [0, 1]
    assert records[0]["token_ids"] == [42]
    assert records[1]["error"] == "RuntimeError: device failure"
    assert [call["rid"] for call in calls] == ["run-r2-c4-p0", "run-r2-c4-p1"]
    assert all(
        call["return_logprob"] and call["logprob_start_len"] == -1 for call in calls
    )


def test_engine_config_preserves_graph_mode_and_allows_explicit_eager(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "llm:\n  model: /model\n  tp_size: 2\n  disable_cuda_graph: false\n"
    )
    graph = read_engine_config(path)
    assert graph == {"model_path": "/model", "tp_size": 2, "disable_cuda_graph": False}
    eager = read_engine_config(path, model_path="/override", eager=True)
    assert eager["disable_cuda_graph"] and eager["model_path"] == "/override"


def test_comparison_rejects_two_copies_of_the_same_arm():
    with pytest.raises(ValueError, match="reference record"):
        compare_records(_record(), _record())


@pytest.mark.parametrize("fail", [False, True])
def test_record_run_saves_progress_and_shuts_down_on_failure(
    tmp_path, monkeypatch, fail
):
    from tests.musa import validate_qwen36

    engines = []

    class Engine:
        def __init__(self, **kwargs):
            self.loop = asyncio.new_event_loop()
            self.closed = False
            engines.append(self)

        def generate(self, **kwargs):
            return _output()

        async def async_generate(self, **kwargs):
            if fail and kwargs["prompt"] == "second":
                raise RuntimeError("device failure")
            output = _output()
            output["meta_info"]["id"] = kwargs["rid"]
            return output

        def shutdown(self):
            self.loop.close()
            self.closed = True

    module = ModuleType("sglang.srt.entrypoints.engine")
    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(validate_qwen36, "runtime_metadata", lambda: {})
    config = tmp_path / "model.yaml"
    config.write_text("llm:\n  model: /model\n  tp_size: 2\n")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["first", "second"]))
    result = tmp_path / "result.json"
    args = SimpleNamespace(
        output=str(result),
        engine_config=str(config),
        model_path=None,
        eager=False,
        prompts=str(prompts),
        max_new_tokens=1,
        concurrency=[1, 2],
        rounds=2,
        label="reference",
        image_digest="image",
        mate_revision="mate",
        plugin_commit="plugin",
    )
    if fail:
        with pytest.raises(RuntimeError, match="partial records"):
            record_run(args)
    else:
        record_run(args)
    data = json.loads(result.read_text())
    assert engines[0].closed
    assert data["records"][0]["token_ids"] == [42]
    if fail:
        assert data["status"] == "failed" and len(data["records"]) == 2
        assert data["records"][1]["error"] == "RuntimeError: device failure"
    else:
        assert data["status"] == "complete" and len(data["records"]) == 8
    with pytest.raises(ValueError, match="already exists"):
        record_run(args)
