# Copyright 2026 FlagOS Contributors

import csv
import json
from pathlib import Path

import pytest

from tools.operator_profiling.inventory_report import generate_inventory_report
from tools.operator_profiling.inventory_report import validate_inventory_package


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_source(
    path: Path,
    *,
    source_valid: bool = True,
    platform_profile: bool = True,
    selected_impl_kinds: list[str] | None = None,
) -> None:
    path.mkdir()
    operator_fields = [
        "operator_id",
        "operator_name",
        "operator_kind",
        "kernel_name",
        "source_category",
        "source_library",
        "classification_rule",
        "execution_origin",
    ]
    operators = [
        {
            "operator_id": "1",
            "operator_name": "aten::mm",
            "operator_kind": "aten",
            "kernel_name": "gemm_kernel",
            "source_category": "torch_aten",
            "source_library": "pytorch",
            "classification_rule": "torch_dispatcher_operator_name",
            "execution_origin": "eager",
        },
        {
            "operator_id": "2",
            "operator_name": "flashinfer::merge_state",
            "operator_kind": "custom",
            "kernel_name": "merge_state_kernel",
            "source_category": "third_party",
            "source_library": "flashinfer",
            "classification_rule": "upstream_runtime_namespace",
            "execution_origin": "eager",
        },
    ]
    _write_csv(path / "operator_list.csv", operator_fields, operators)

    detail_fields = [
        "operator_name",
        "kernel_name",
        "source_category",
        "variant_index",
        "mapping_status",
        "input_shapes",
        "input_dtypes",
        "candidate_operators",
        "kernel_event_count",
        "kernel_time_us",
    ]
    details = [
        {
            "operator_name": "aten::mm",
            "kernel_name": "gemm_kernel",
            "source_category": "torch_aten",
            "variant_index": "1",
            "mapping_status": "operator_shape_matched",
            "input_shapes": "[[2,4],[4,8]]",
            "input_dtypes": '["BFloat16","BFloat16"]',
            "candidate_operators": "null",
            "kernel_event_count": "2",
            "kernel_time_us": "30.0",
        },
        {
            "operator_name": "flashinfer::merge_state",
            "kernel_name": "merge_state_kernel",
            "source_category": "third_party",
            "variant_index": "1",
            "mapping_status": "operator_shape_matched",
            "input_shapes": "[[2,8]]",
            "input_dtypes": '["BFloat16"]',
            "candidate_operators": "null",
            "kernel_event_count": "3",
            "kernel_time_us": "70.0",
        },
    ]
    _write_csv(path / "kernel_shape_dtype.csv", detail_fields, details)
    (path / "profile_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "scope": {
                    "execution_mode": "eager",
                    "inventory_source_policy": "four_source_categories",
                },
                "metadata": {
                    "adaptation": {
                        "layer1_flaggems": not platform_profile,
                        "layer2_plugin_dispatch": True,
                        "layer2_flagos_implementations": False,
                        "layer2_vendor_preferred": True,
                    },
                    "server": {
                        "enable_torch_compile": False,
                        "disable_cuda_graph": True,
                        "tp_size": 1,
                    },
                    "workload": {
                        "concurrency": 64,
                        "warmup_output_sha256": "same-output",
                        "profiled_output_sha256": "same-output",
                    },
                },
                "ranks": [0],
                "compute_kernel_event_count": 5,
                "compute_kernel_time_us": 100.0,
                "unique_stable_kernel_names": 2,
                "dispatch": {
                    "selected_impl_kinds": selected_impl_kinds or [],
                },
                "source_category_event_count": {
                    "third_party": 3,
                    "vendor": 0,
                    "torch_aten": 2,
                    "torch_fused": 0,
                },
                "source_category_time_us": {
                    "third_party": 70.0,
                    "vendor": 0.0,
                    "torch_aten": 30.0,
                    "torch_fused": 0.0,
                },
                "validation": {"source_conserved": source_valid},
            }
        ),
        encoding="utf-8",
    )


def test_inventory_has_only_supported_files_and_conserves_metrics(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    summary = generate_inventory_report(source, output)

    assert {path.name for path in output.iterdir()} == {
        "operator_list.csv",
        "kernel_details_report.csv",
        "summary.json",
    }
    with (output / "operator_list.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        operators = list(reader)
    assert reader.fieldnames == [
        "operator_id",
        "operator_name",
        "operator_kind",
        "kernel_name",
        "source_category",
        "source_library",
    ]
    assert len(operators) == 2

    with (output / "kernel_details_report.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        details = list(reader)
    assert reader.fieldnames[:9] == [
        "operator_name",
        "kernel_name",
        "variant_index",
        "mapping_status",
        "input_shapes",
        "input_dtypes",
        "candidate_operators",
        "kernel_event_count",
        "kernel_time_us",
    ]
    assert sum(int(row["kernel_event_count"]) for row in details) == 5
    assert sum(float(row["kernel_time_us"]) for row in details) == 100.0
    assert sum(float(row["kernel_time_percent"]) for row in details) == 100.0
    assert all(summary["validation"].values())
    assert set(summary["files"]) == {
        "operator_list.csv",
        "kernel_details_report.csv",
    }
    assert all(validate_inventory_package(output).values())


def test_inventory_rejects_an_invalid_source_report(tmp_path):
    source = tmp_path / "source"
    _write_source(source, source_valid=False)

    with pytest.raises(ValueError, match="did not pass validation"):
        generate_inventory_report(source, tmp_path / "output")


def test_inventory_rejects_a_non_platform_profile_source(tmp_path):
    source = tmp_path / "source"
    _write_source(source, platform_profile=False)

    with pytest.raises(ValueError, match="not a verified platform_profile"):
        generate_inventory_report(source, tmp_path / "output")


def test_inventory_rejects_non_vendor_plugin_dispatch(tmp_path):
    source = tmp_path / "source"
    _write_source(source, selected_impl_kinds=["flagos"])

    with pytest.raises(ValueError, match="not a verified platform_profile"):
        generate_inventory_report(source, tmp_path / "output")


def test_inventory_validator_detects_csv_tampering(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    generate_inventory_report(source, output)
    with (output / "operator_list.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(RuntimeError, match="serialized operator inventory"):
        validate_inventory_package(output)
