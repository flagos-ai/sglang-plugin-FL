# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.operator_profiling.coverage_rules import descriptor
from tools.operator_profiling.generate_flagos_coverage import generate_coverage
from tools.operator_profiling.runtime_report import (
    generate_runtime_report,
    read_csv,
    validate_runtime_report,
)
from .test_trace_report import _event, _operator_marker, _sample_events, _write_trace


def metadata(implementation="native", mode="graph"):
    return {
        "implementation": implementation,
        "execution_mode": mode,
        "model": {"name": "test", "config_sha256": "abc"},
        "workload_config": {"input": 16, "output": 4, "concurrency": 2},
        "engine_config": {"tp_size": 2},
        "software": {"torch": "test"},
        "hardware": ["test", "test"],
        "provider": {"name": "triton"},
    }


def report(tmp_path, name, events, *, meta=None, reference=None, evidence=None):
    directory = tmp_path / name
    directory.mkdir()
    traces = directory / "traces"
    traces.mkdir()
    _write_trace(traces / "runtime-TP-0.trace.json", events)
    _write_trace(traces / "runtime-TP-1.trace.json", _sample_events(1))
    _write_trace(traces / "graph_capture_-TP-0.trace.json", _sample_events(0))
    evidence_dir = directory / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "worker.jsonl").write_text(
        "\n".join(json.dumps(item) for item in evidence or [])
    )
    results = directory / "results"
    generate_runtime_report(
        traces,
        results,
        metadata=meta or metadata(),
        eager_reference=reference,
        evidence_dir=evidence_dir,
    )
    return results


def test_rank0_runtime_includes_communication_and_unknown(tmp_path):
    results = report(tmp_path, "run", _sample_events())
    summary = json.loads((results / "summary.json").read_text())
    assert summary["runtime_kernel_event_count"] == 5
    assert summary["runtime_kernel_time_ns"] == 72_000
    assert summary["scope"]["collected_ranks"] == [0, 1]
    assert all(summary["conservation"].values())
    rows = read_csv(results / "operator_list.csv")
    assert sum(row["operator_kind"] == "communication" for row in rows) == 2
    assert any(row["operator_name"] == "null" for row in rows)
    assert all(validate_runtime_report(results).values())


def test_exact_eager_reference_preserves_graph_observations(tmp_path):
    kernel = "void example::kernel<float>(float*)"
    eager = report(
        tmp_path,
        "eager",
        [
            _event("cpu_op", "aten::add", 1, shapes=[[2]], dtypes=["float"]),
            _event("kernel", kernel, 1),
        ],
        meta=metadata(mode="eager"),
    )
    events = [
        _event("kernel", kernel, duration=2),
        _event("kernel", "void example::kernel<half>(half*)", duration=3),
        _event("cpu_op", "aten::add", 2, shapes=[[2]], dtypes=["float"]),
        _event("kernel", kernel, 2, duration=4),
    ]
    raw = report(tmp_path, "raw", events)
    enriched = report(tmp_path, "refer", events, reference=eager)
    before = json.loads((raw / "summary.json").read_text())
    after = json.loads((enriched / "summary.json").read_text())
    assert before["per_kernel"] == after["per_kernel"]
    assert before["raw_mapping_status_counts"] == after["raw_mapping_status_counts"]
    details = read_csv(enriched / "kernel_shape_dtype.csv")
    missing = [r for r in details if r["mapping_status"] == "missing_external_id"]
    assert all(r["input_shapes"] == r["input_dtypes"] == "null" for r in missing)
    row = next(r for r in missing if r["kernel_name"] == kernel)
    assert row["operator_name"] == "aten::add"
    assert row["eager_input_shapes"] == "[[2]]"
    assert (
        next(r for r in missing if "half" in r["kernel_name"])["operator_name"]
        == "null"
    )
    times = [
        r for r in read_csv(enriched / "kernel_time.csv") if r["kernel_name"] == kernel
    ]
    assert len(times) == 1 and times[0]["kernel_call_count"] == "2"
    assert times[0]["refer"] == "eager-refer"
    inventory = [
        r
        for r in read_csv(enriched / "operator_list.csv")
        if r["kernel_name"] == kernel
    ]
    assert len(inventory) == 1 and inventory[0]["operator_kind"] == "aten"


def test_ambiguous_eager_apis_remain_one_identity(tmp_path):
    eager = report(
        tmp_path,
        "eager",
        [
            _event("cpu_op", "aten::zero_", 1),
            _event("kernel", "zero", 1),
            _event("cpu_op", "aten::zeros", 2),
            _event("kernel", "zero", 2),
        ],
        meta=metadata(mode="eager"),
    )
    graph = report(tmp_path, "graph", [_event("kernel", "zero")], reference=eager)
    row = read_csv(graph / "operator_list.csv")[0]
    assert row["operator_kind"] == "eager-refer"
    assert row["operator_name"] == "aten::zero_,aten::zeros"
    assert read_csv(graph / "kernel_time.csv")[0]["operator_name"] == "null"


@pytest.mark.parametrize(
    "field", ["implementation", "model", "provider", "engine_config", "workload_config"]
)
def test_mismatched_eager_is_rejected(tmp_path, field):
    eager = report(tmp_path, "eager", _sample_events(), meta=metadata(mode="eager"))
    changed = metadata()
    changed[field] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        report(tmp_path, "graph", _sample_events(), meta=changed, reference=eager)


def test_graph_is_not_allowed_as_eager_reference(tmp_path):
    graph = report(tmp_path, "graph", _sample_events())
    with pytest.raises(ValueError, match="eager mode"):
        report(tmp_path, "second", _sample_events(), reference=graph)


def test_tampered_reference_is_rejected(tmp_path):
    eager = report(tmp_path, "eager", _sample_events(), meta=metadata(mode="eager"))
    path = eager / "kernel_shape_dtype.csv"
    path.write_text(path.read_text().replace("mm_kernel", "changed"))
    with pytest.raises(ValueError, match="validation failed"):
        report(tmp_path, "graph", _sample_events(), reference=eager)


def test_inductor_is_never_matched_from_eager(tmp_path):
    eager = report(
        tmp_path,
        "eager",
        [
            _event("cpu_op", "aten::add", 1),
            _event("kernel", "triton_poi_fused_add_0", 1),
        ],
        meta=metadata(mode="eager"),
    )
    graph = report(
        tmp_path, "graph", [_event("kernel", "triton_poi_fused_add_0")], reference=eager
    )
    assert read_csv(graph / "kernel_shape_dtype.csv")[0]["eager_candidates"] == "null"


def test_normalization_does_not_merge_stages_or_compile_modules():
    assert (
        descriptor("x", "moe_align_block_size_stage1")[1]
        != descriptor("x", "moe_align_block_size_stage2")[1]
    )
    assert (
        descriptor("a", "triton_poi_fused_0")[1]
        != descriptor("b", "triton_poi_fused_0")[1]
    )
    assert (
        descriptor("a", "silu_kernel_rank_1")[1]
        == descriptor("b", "silu_kernel_rank_2")[1]
    )
    assert descriptor("aten::add", "first")[1] == descriptor("aten::add", "second")[1]


def evidence(kernel, *, component="FlagTree", api=None, **kwargs):
    return {
        "kernel_name": kernel,
        "component": component,
        "provider": {"name": "flagtree"},
        "rank": 0,
        "successful_launch": True,
        "phase": "runtime",
        "aten_api": api,
        **kwargs,
    }


def test_coverage_requires_runtime_provider_evidence(tmp_path):
    events = [
        _event("cpu_op", "aten::add", 1),
        _event("kernel", "native_add", 1),
        _event("kernel", "triton_poi_fused_0"),
        _event("kernel", "nccl_AllReduce"),
    ]
    baseline = report(tmp_path, "native", events)
    plugin = report(
        tmp_path,
        "plugin",
        [
            _event("kernel", "gems_add"),
            _event("kernel", "triton_poi_fused_0"),
            _event("kernel", "nccl_AllReduce"),
        ],
        meta=metadata("plugin"),
        evidence=[
            evidence("gems_add", component="FlagGems", api="aten::add"),
            evidence("nccl_AllReduce"),
            evidence("never_executed", component="FlagGems", api="aten::mul"),
        ],
    )
    summary = generate_coverage(baseline, plugin, tmp_path / "coverage")
    assert summary["total_operator_count"] == 3
    assert summary["covered_operator_count"] == 1
    assert summary["coverage_percent(%)"] == "33.33"


def test_fusion_intersection_deduplicates_and_rejects_unproven(tmp_path):
    baseline = report(
        tmp_path,
        "native",
        [
            _event("cpu_op", "aten::add", 1),
            _event("kernel", "a", 1),
            _event("cpu_op", "aten::mul", 2),
            _event("kernel", "m", 2),
        ],
    )
    plugin = report(
        tmp_path,
        "plugin",
        [_event("kernel", "fused")],
        meta=metadata("plugin"),
        evidence=[evidence("fused", component="FlagGems")],
    )
    mappings = [
        {
            "plugin_kernel": "fused",
            "baseline_identities": [
                ["aten", "aten::add"],
                ["aten", "aten::mul"],
                ["aten", "aten::abs"],
            ],
            "evidence": "source.py:12",
        }
    ]
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps(mappings * 2))
    summary = generate_coverage(
        baseline, plugin, tmp_path / "coverage", fusion_mappings=path
    )
    assert summary["covered_operator_count"] == 2
    assert summary["coverage_percent(%)"] == "100.00"
    mappings[0]["plugin_kernel"] = "not_executed"
    path.write_text(json.dumps(mappings))
    with pytest.raises(ValueError, match="no successful"):
        generate_coverage(baseline, plugin, tmp_path / "bad", fusion_mappings=path)


def test_native_reference_baseline_role_and_fallback_reason(tmp_path):
    plugin = report(tmp_path, "plugin", _sample_events(), meta=metadata("plugin"))
    with pytest.raises(ValueError, match="implementation=native"):
        generate_coverage(plugin, plugin, tmp_path / "a")
    with pytest.raises(ValueError, match="requires an explicit reason"):
        generate_coverage(plugin, plugin, tmp_path / "b", denominator="B")
    result = generate_coverage(
        plugin,
        plugin,
        tmp_path / "c",
        denominator="B",
        fallback_reason="test platform lacks native runtime",
    )
    assert result["denominator_set"] == "B"


def test_only_active_flagcx_marked_collectives_are_proven(tmp_path):
    marker = _operator_marker(
        {
            "operator_name": "flagcx::all_reduce",
            "source_category": "third_party",
            "source_library": "flagcx",
            "classification_rule": "active_flagcx_communicator",
        }
    )
    events = [
        _event("user_annotation", marker, timestamp=0, duration=10),
        _event(
            "cuda_runtime",
            "cudaLaunchKernel",
            timestamp=1,
            extra_args={"correlation": 1},
        ),
        _event("kernel", "nccl_AllReduce", timestamp=11, extra_args={"correlation": 1}),
        _event(
            "cuda_runtime",
            "cudaLaunchKernel",
            timestamp=2,
            extra_args={"correlation": 2},
        ),
        _event("kernel", "copy_kernel", timestamp=12, extra_args={"correlation": 2}),
        _event("kernel", "nccl_AllReduce_other", timestamp=15),
    ]
    results = report(tmp_path, "plugin", events, meta=metadata("plugin"))
    items = json.loads((results / "runtime_evidence.json").read_text())
    assert [item["kernel_name"] for item in items] == ["nccl_AllReduce"]
    rows = read_csv(results / "operator_list.csv")
    assert (
        next(r for r in rows if r["kernel_name"] == "copy_kernel")["operator_kind"]
        != "communication"
    )


def test_preparation_api_and_previous_window_do_not_cover_aten(tmp_path):
    baseline = report(
        tmp_path,
        "native",
        [_event("cpu_op", "aten::add", 1), _event("kernel", "native_add", 1)],
    )
    meta = metadata("plugin")
    meta["runtime_window"] = {"start_unix_ns": 20, "end_unix_ns": 30}
    plugin = report(
        tmp_path,
        "plugin",
        [_event("kernel", "gems_add")],
        meta=meta,
        evidence=[
            evidence(
                "gems_add",
                component="FlagGems",
                api="aten::add",
                phase="preparation",
                timestamp_ns=10,
            ),
            evidence(
                "gems_add",
                component="FlagGems",
                api="aten::add",
                phase="runtime",
                timestamp_ns=15,
            ),
        ],
    )
    summary = generate_coverage(baseline, plugin, tmp_path / "coverage")
    assert summary["covered_operator_count"] == 0


def test_reference_rejects_different_blacklist(tmp_path):
    meta = metadata(mode="eager")
    meta["environment"] = {"SGLANG_FL_FLAGOS_BLACKLIST": "mm"}
    eager = report(tmp_path, "eager", _sample_events(), meta=meta)
    with pytest.raises(ValueError, match="configuration"):
        report(tmp_path, "graph", _sample_events(), reference=eager)


def test_zero_duration_is_unavailable_not_zero_percent(tmp_path):
    results = report(tmp_path, "zero", [_event("kernel", "no_timer", duration=0)])
    assert (
        read_csv(results / "operator_list.csv")[0]["kernel_time_percent(%)"] == "null"
    )
    assert json.loads((results / "summary.json").read_text())[
        "timing_unavailable_reason"
    ]


def test_fusion_does_not_count_its_own_identity_again(tmp_path):
    baseline = report(
        tmp_path,
        "native",
        [
            _event("cpu_op", "aten::add", 1),
            _event("kernel", "a", 1),
            _event("kernel", "fused"),
        ],
    )
    plugin = report(
        tmp_path,
        "plugin",
        [_event("kernel", "fused")],
        meta=metadata("plugin"),
        evidence=[evidence("fused", component="FlagGems")],
    )
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            [
                {
                    "plugin_kernel": "fused",
                    "baseline_identities": [["aten", "aten::add"]],
                    "evidence": "source.py:12",
                }
            ]
        )
    )
    result = generate_coverage(
        baseline, plugin, tmp_path / "coverage", fusion_mappings=path
    )
    assert result["covered_operator_count"] == 1


def test_reference_cannot_overwrite_conflicting_runtime_candidates(tmp_path):
    eager = report(
        tmp_path,
        "eager",
        [_event("cpu_op", "aten::add", 1), _event("kernel", "shared", 1)],
        meta=metadata(mode="eager"),
    )
    graph = report(
        tmp_path,
        "graph",
        [
            _event("cpu_op", "aten::mul", 2),
            _event("cpu_op", "aten::sub", 2),
            _event("kernel", "shared", 2),
        ],
        reference=eager,
    )
    assert read_csv(graph / "kernel_time.csv")[0]["operator_name"] == "null"
    assert read_csv(graph / "operator_list.csv")[0]["operator_kind"] == "unattributed"


def test_large_shape_evidence_survives_csv_and_reference(tmp_path):
    # Reproduces the formal plugin Graph run's >128 KiB CSV field failure.
    shape = list(range(40_000))
    eager = report(
        tmp_path,
        "eager",
        [
            _event("cpu_op", "aten::add", 1, shapes=[shape], dtypes=["float"]),
            _event("kernel", "large_shape_kernel", 1),
        ],
        meta=metadata(mode="eager"),
    )
    graph = report(
        tmp_path, "graph", [_event("kernel", "large_shape_kernel")], reference=eager
    )
    row = read_csv(graph / "kernel_shape_dtype.csv")[0]
    assert json.loads(row["eager_input_shapes"]) == [shape]
    assert row["input_shapes"] == "null"
    assert all(validate_runtime_report(graph).values())


def test_compile_kernel_self_marker_is_not_a_second_module(tmp_path):
    # A real Graph trace has both JIT-marked and unmarked replay events for
    # this kernel. Repeating its name supplies no additional module identity.
    kernel = "triton_poi_fused_add_mul_0"
    results = report(
        tmp_path,
        "mixed_compile",
        [
            _event("cpu_op", kernel, 1),
            _event("kernel", kernel, 1),
            _event("kernel", kernel),
        ],
    )
    summary = json.loads((results / "summary.json").read_text())
    assert summary["total_operator_count"] == 1
    assert summary["runtime_kernel_event_count"] == 2
    assert all(validate_runtime_report(results).values())
