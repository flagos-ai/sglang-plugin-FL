# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from tools.operator_profiling.export_schema import (
    AUDIT_FILE,
    DETAIL_FIELDS,
    OPERATOR_FIELDS,
    export_detail,
    operator_kind,
)
from tools.operator_profiling.runtime_report import read_csv, validate_runtime_report
from .test_runtime_coverage import metadata, report
from .test_trace_report import _event, _sample_events

CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures/op_profile_contract.json").read_text()
)["files"]


def test_exact_github_headers_with_and_without_reference(tmp_path):
    eager = report(tmp_path, "eager", _sample_events(), meta=metadata(mode="eager"))
    for name, reference in [("raw", None), ("refer", eager)]:
        results = report(tmp_path, name, _sample_events(), reference=reference)
        for filename, expected in CONTRACT.items():
            rows = read_csv(results / filename)
            assert list(rows[0]) == expected["fields"]
        assert all(validate_runtime_report(results).values())
    assert OPERATOR_FIELDS == CONTRACT["operator_list.csv"]["fields"]
    assert DETAIL_FIELDS == CONTRACT["kernel_shape_dtype.csv"]["fields"]


@pytest.mark.parametrize(
    "api,kernel,kind,expected",
    [
        ("aten::add", "vectorized", "aten", "aten"),
        ("null", "triton_poi_fused_add_0", "torch_compile", "triton_compiled"),
        ("triton_custom_op", "kernel", "custom", "triton_compiled"),
        ("null", "unknown", "unattributed", "unattributed"),
        ("sgl_kernel::rmsnorm", "rms", "custom", "custom"),
        ("sglang.layers.norm.forward", "rms", "custom", "runtime_operator"),
        (
            "sglang::all_reduce",
            "all_reduce_rmsnorm",
            "custom",
            "fused_communication_compute",
        ),
        ("record_param_comms", "nccl_AllReduce", "communication", "communication"),
        ("aten::add", "add_kernel", "eager-refer", "eager-refer"),
        ("null", "moe_align_block_size_stage1", "unattributed", "custom"),
        ("flagcx::all_reduce", "copy_kernel", "custom", "custom"),
    ],
)
def test_reference_taxonomy(api, kernel, kind, expected):
    assert operator_kind(api, kernel, kind) == expected


def test_nested_dtype_and_candidate_contract_preserves_raw_values():
    candidate = {
        "operator_name": "aten::add",
        "input_shapes": [[2], []],
        "input_dtypes": ["torch.bfloat16", "Scalar[int]"],
        "metadata_status": "shape_and_dtype",
    }
    row = {
        "operator_name": "aten::add",
        "kernel_name": "kernel",
        "variant_index": 1,
        "mapping_status": "native_marker_shape_ambiguous",
        "input_shapes": "null",
        "input_dtypes": json.dumps(
            [
                "torch.bfloat16",
                "torch.int64",
                "Scalar[bool]",
                "None",
                "Object[Layer]",
                "TensorList",
            ]
        ),
        "candidate_operators": json.dumps([candidate]),
        "kernel_event_count": 2,
        "kernel_time_us": "3.001",
        "eager_candidates": json.dumps([candidate]),
        "raw_operator_name": "aten::add",
        "source_library": "flag_gems",
    }
    before = dict(row)
    public = export_detail(row)
    assert row == before
    assert list(public) == DETAIL_FIELDS
    assert public["mapping_status"] == "shape_ambiguous"
    assert json.loads(public["input_dtypes"]) == [
        "c10::BFloat16",
        "long int",
        "Scalar",
        "",
        "",
        "TensorList",
    ]
    direct = json.loads(public["candidate_operators"])[0]
    assert set(direct) == {
        "operator",
        "input_shapes",
        "input_dtypes",
        "metadata_status",
    }
    assert direct["operator"] == "aten::add"
    assert direct["input_dtypes"] == ["c10::BFloat16", "Scalar"]
    eager = json.loads(public["eager_candidates"])[0]
    assert set(eager) == {"operator_name", "input_shapes", "input_dtypes"}
    assert public["kernel_event_count"] == 2 and public["kernel_time_us"] == "3.001"


def test_eager_reads_lossless_audit_after_public_dtype_normalization(tmp_path):
    events = [
        _event(
            "cpu_op",
            "aten::add",
            1,
            shapes=[[2], []],
            dtypes=["torch.bfloat16", "Scalar[int]"],
        ),
        _event("kernel", "add_kernel", 1),
    ]
    eager = report(tmp_path, "eager", events, meta=metadata(mode="eager"))
    graph = report(tmp_path, "graph", [_event("kernel", "add_kernel")], reference=eager)
    audit = read_csv(graph / AUDIT_FILE)[0]
    public = read_csv(graph / "kernel_shape_dtype.csv")[0]
    assert json.loads(audit["eager_input_dtypes"]) == ["torch.bfloat16", "Scalar[int]"]
    assert json.loads(public["eager_input_dtypes"]) == ["c10::BFloat16", "Scalar"]
    assert audit["input_dtypes"] == public["input_dtypes"] == "null"
    summary = json.loads((graph / "summary.json").read_text())
    assert summary["eager_reference"]["detail_source"] == AUDIT_FILE
    assert all(validate_runtime_report(graph).values())
    # Corrupted provenance must fail even while the public CSV stays valid.
    path = eager / AUDIT_FILE
    path.write_text(path.read_text().replace("Scalar[int]", "Scalar[float]"))
    with pytest.raises(ValueError, match="validation failed"):
        report(
            tmp_path, "bad_reference", [_event("kernel", "add_kernel")], reference=eager
        )
