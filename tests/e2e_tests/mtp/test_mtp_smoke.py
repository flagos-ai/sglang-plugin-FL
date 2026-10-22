# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real eager HCU MTP smoke: B4 token metadata and target logprob conformance.

The two original Qwen3.6 checkpoints are selected by the existing model YAMLs.
This task uses one speculative Engine per model; it makes no graph, performance
or short-run scheduler-average claim.
"""

import json
import math
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tests.e2e_tests.plugin_utils import (
    assert_sglang_fl_plugin_loaded_and_active,
    load_e2e_model_config,
)

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/qwen3_6_27b_mtp_inference.py"


def _logprob_entry(entry, *, allow_none=False):
    if not isinstance(entry, (tuple, list)) or len(entry) != 3:
        raise ValueError("Expected actual token value, ID and decoded text")
    value, token_id, _ = entry
    if type(token_id) is not int or token_id < 0:
        raise ValueError("Invalid actual token ID")
    if value is None and allow_none:
        return token_id
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Missing or nonfinite token logprob")
    return token_id


def _validate_result(example, spec, response, rid):
    if not isinstance(response, dict):
        raise ValueError("Expected one actual generation response")
    meta = example._generation_metadata(response, speculative=True)
    if meta.get("id") != rid:
        raise ValueError("Response request ID/order differs from the B4 request")
    valid, reason = example._validate_output(spec, response["text"])
    if not valid:
        raise ValueError(f"MTP semantic contract failed: {reason}")
    prompt_count, output_count = meta["prompt_tokens"], meta["completion_tokens"]
    if output_count > 128:
        raise ValueError("Actual completion exceeds max_new_tokens=128")
    token_ids = {}
    for prefix, count in (("input", prompt_count), ("output", output_count)):
        entries = meta.get(f"{prefix}_token_logprobs")
        if not isinstance(entries, list) or len(entries) != count:
            raise ValueError(f"{prefix} token/logprob count differs from actual tokens")
        token_ids[prefix] = [
            _logprob_entry(entry, allow_none=(prefix == "input" and i == 0))
            for i, entry in enumerate(entries)
        ]
        rows = meta.get(f"{prefix}_top_logprobs")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{prefix} top5 count differs from actual tokens")
        for i, row in enumerate(rows):
            if prefix == "input" and i == 0 and row is None:
                continue  # Upstream leaves the first prompt token unscored.
            if not isinstance(row, (list, tuple)) or len(row) != 5:
                raise ValueError("Expected five actual finite top-logprob candidates")
            ids = [_logprob_entry(entry) for entry in row]
            if len(set(ids)) != 5:
                raise ValueError("Top5 candidate IDs must be distinct")
            if prefix == "output":
                chosen = [entry[0] for entry in row if entry[1] == token_ids[prefix][i]]
                if not chosen or max(entry[0] for entry in row) - max(chosen) > 1e-6:
                    raise ValueError("Temperature-zero output did not choose top1")
                if not math.isclose(
                    entries[i][0], max(chosen), rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise ValueError(
                        "Chosen logprob differs from its actual top5 entry"
                    )
    if "output_ids" in response:
        direct = response["output_ids"]
        if (
            not isinstance(direct, list)
            or any(type(value) is not int or value < 0 for value in direct)
            or direct != token_ids["output"]
        ):
            raise ValueError(
                "Direct output IDs differ from the actual logprob token IDs"
            )
    return {
        "rid": rid,
        "semantic_reason": reason,
        **{
            name: meta[name]
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "spec_verify_ct",
                "spec_num_proposed_drafts",
                "spec_num_correct_drafts",
                "spec_accept_rate",
                "spec_accept_length",
            )
        },
    }


def _run_mtp_smoke(example, engine):
    specs = example.PROMPTS[:4]
    rids = [f"hcu-mtp-smoke-{i}" for i in range(4)]
    responses = engine.generate(
        prompt=[example._text_prompt(spec["prompt"]) for spec in specs],
        sampling_params={"temperature": 0, "max_new_tokens": 128},
        return_logprob=True,
        top_logprobs_num=5,
        logprob_start_len=0,
        stream=False,
        rid=rids,
    )
    # Preserve the complete returned objects before a gate can fail.
    print(
        "HCU MTP B4 responses: " + json.dumps(responses, ensure_ascii=False), flush=True
    )
    if not isinstance(responses, list) or len(responses) != 4:
        raise ValueError("Expected four real MTP responses")
    validated = [
        _validate_result(example, spec, response, rid)
        for spec, response, rid in zip(specs, responses, rids)
    ]
    print("HCU MTP B4 validated: " + json.dumps(validated), flush=True)
    conformance = example.run_logprob_conformance(engine)
    print("HCU MTP conformance: " + json.dumps(conformance), flush=True)
    if conformance.get("passed") is not True or conformance.get("scored_tokens") != 64:
        raise ValueError(
            "MTP failed the original 64-token target logprob conformance gate"
        )


@pytest.mark.e2e
def test_mtp_smoke(monkeypatch):
    # A misrouted or missing HCU allocation must fail, rather than become a skip.
    if (os.environ.get("FL_TEST_PLATFORM"), os.environ.get("FL_TEST_DEVICE")) != (
        "hygon",
        "bw1000",
    ):
        raise ValueError("MTP CI requires the Hygon bw1000 eager profile")
    if not os.environ.get("FL_TEST_MODEL") or not os.environ.get("FL_TEST_CASE"):
        raise ValueError("MTP CI requires an explicit model and case")
    _, _, config = load_e2e_model_config()
    monkeypatch.setenv("MODEL_PATH", config.model)
    monkeypatch.setenv("TP_SIZE", "2")
    # Reassert before loading Torch/Engine, matching the native vendor wrapper.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    example = SimpleNamespace(**runpy.run_path(str(_EXAMPLE), run_name="hcu_mtp_ci"))
    if not example._is_hcu:
        raise ValueError("MTP CI requires actual HCU Torch and available HCU devices")
    # The shared factory checks resolved model/draft paths, TP2/PP1, BF16,
    # EAGLE steps3/topk1/draft4 and both graph phases disabled after construction.
    engine = example._make_mtp_engine(True, True, True)
    try:
        assert_sglang_fl_plugin_loaded_and_active()
        _run_mtp_smoke(example, engine)
    finally:
        engine.shutdown()
