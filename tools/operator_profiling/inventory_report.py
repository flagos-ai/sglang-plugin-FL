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

"""Render the public operator inventory from an audited profiler report.

``trace_report`` intentionally keeps several normalized intermediate tables
for conservation checks.  This module turns those tables into the compact
three-file package consumed by operator developers.  Every value in the public
CSVs comes from the audited platform-profile run supplied in ``source_dir``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


_SOURCE_CATEGORIES = ("third_party", "vendor", "torch_aten", "torch_fused")
_OPERATOR_FIELDS = (
    "operator_id",
    "operator_name",
    "operator_kind",
    "kernel_name",
    "source_category",
    "source_library",
)
_DETAIL_FIELDS = (
    # Keep the established report prefix stable for downstream consumers.
    "operator_name",
    "kernel_name",
    "variant_index",
    "mapping_status",
    "input_shapes",
    "input_dtypes",
    "candidate_operators",
    "kernel_event_count",
    "kernel_time_us",
    # Provenance and prioritization fields follow the compatible prefix.
    "operator_kind",
    "source_category",
    "source_library",
    "execution_origin",
    "kernel_time_percent",
    "kernel_time_percent_of_category",
)
_PUBLIC_FILES = {
    "operator_list.csv",
    "kernel_details_report.csv",
    "summary.json",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or ()), list(reader)


def _require_fields(path: Path, fields: Iterable[str], required: set[str]) -> None:
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8") as source:
        row_count = max(sum(1 for _ in source) - 1, 0)
    return {"path": path.name, "row_count": row_count, "sha256": digest.hexdigest()}


def _single_or_json(values: Iterable[str], default: str = "null") -> str:
    items = sorted({value for value in values if value and value != "null"})
    if not items:
        return default
    if len(items) == 1:
        return items[0]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _percent(value: float, total: float) -> float:
    return round(value * 100.0 / total, 6) if total else 0.0


def _json_or_null(value: str) -> bool:
    if value == "null":
        return True
    try:
        json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    return True


def _close(left: float, right: float, row_count: int) -> bool:
    # Intermediate durations are serialized in microseconds with nanosecond
    # precision.  Permit only accumulated textual rounding, not data loss.
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=max(row_count, 1) * 1e-6)


def _is_platform_profile(
    metadata: dict[str, Any], scope: dict[str, Any], dispatch: dict[str, Any]
) -> bool:
    adaptation = metadata.get("adaptation", {})
    server = metadata.get("server", {})
    selected_impl_kinds = set(dispatch.get("selected_impl_kinds", ()))
    return all(
        (
            scope.get("execution_mode") == "eager",
            adaptation.get("layer1_flaggems") is False,
            adaptation.get("layer2_plugin_dispatch") is True,
            adaptation.get("layer2_flagos_implementations") is False,
            adaptation.get("layer2_vendor_preferred") is True,
            server.get("enable_torch_compile") is False,
            server.get("disable_cuda_graph") is True,
            selected_impl_kinds <= {"vendor"},
        )
    )


def validate_inventory_package(output_dir: str | Path) -> dict[str, bool]:
    """Independently validate a serialized three-file inventory package."""

    output_root = Path(output_dir)
    existing_files = {path.name for path in output_root.iterdir() if path.is_file()}
    operator_path = output_root / "operator_list.csv"
    detail_path = output_root / "kernel_details_report.csv"
    summary_path = output_root / "summary.json"

    operator_fields, operators = _read_csv(operator_path)
    detail_fields, details = _read_csv(detail_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    relations = {
        (row["operator_name"], row["kernel_name"], row["source_category"])
        for row in operators
    }
    event_count = sum(int(row["kernel_event_count"]) for row in details)
    time_us = sum(float(row["kernel_time_us"]) for row in details)
    category_events: Counter[str] = Counter()
    category_times: Counter[str] = Counter()
    for row in details:
        category_events[row["source_category"]] += int(row["kernel_event_count"])
        category_times[row["source_category"]] += float(row["kernel_time_us"])

    summary_time_us = float(summary.get("compute_kernel_time_us", -1.0))
    summary_category_time_us = {
        category: float(summary.get("source_category_time_us", {}).get(category, 0.0))
        for category in _SOURCE_CATEGORIES
    }
    metadata = summary.get("metadata", {})
    workload = metadata.get("workload", {})
    expected_ranks = list(range(int(metadata.get("server", {}).get("tp_size", -1))))

    recorded_files = summary.get("files", {})
    actual_file_records = {
        path.name: _file_record(path) for path in (operator_path, detail_path)
    }
    validation = {
        "contains_exactly_three_supported_files": existing_files == _PUBLIC_FILES,
        "operator_header_is_current": operator_fields == list(_OPERATOR_FIELDS),
        "detail_header_is_current": detail_fields == list(_DETAIL_FIELDS),
        "operator_list_is_not_empty": bool(operators),
        "kernel_details_are_not_empty": bool(details),
        "operator_relations_are_unique": len(operators)
        == len({tuple(row[field] for field in _OPERATOR_FIELDS) for row in operators}),
        "kernel_detail_rows_are_unique": len(details)
        == len({tuple(row[field] for field in _DETAIL_FIELDS) for row in details}),
        "kernel_metrics_are_finite_and_nonnegative": all(
            int(row["kernel_event_count"]) > 0
            and math.isfinite(float(row["kernel_time_us"]))
            and float(row["kernel_time_us"]) >= 0.0
            for row in details
        ),
        "profile_mode_is_platform_profile": (
            summary.get("scope", {}).get("profile_mode") == "platform_profile"
        ),
        "all_details_have_operator_provenance": all(
            (row["operator_name"], row["kernel_name"], row["source_category"])
            in relations
            for row in details
        ),
        "all_source_categories_are_valid": all(
            row["source_category"] in _SOURCE_CATEGORIES for row in details
        ),
        "all_shape_fields_are_json_or_null": all(
            _json_or_null(row[field])
            for row in details
            for field in ("input_shapes", "input_dtypes", "candidate_operators")
        ),
        "kernel_event_count_is_conserved": (
            event_count == int(summary.get("compute_kernel_event_count", -1))
        ),
        "kernel_time_is_conserved": _close(
            time_us,
            summary_time_us,
            len(details),
        ),
        "kernel_time_percentages_are_correct": all(
            float(row["kernel_time_percent"])
            == _percent(float(row["kernel_time_us"]), summary_time_us)
            and float(row["kernel_time_percent_of_category"])
            == _percent(
                float(row["kernel_time_us"]),
                summary_category_time_us[row["source_category"]],
            )
            for row in details
        ),
        "source_category_event_count_is_conserved": all(
            category_events[category]
            == int(summary.get("source_category_event_count", {}).get(category, 0))
            for category in _SOURCE_CATEGORIES
        ),
        "source_category_time_is_conserved": all(
            _close(
                category_times[category],
                summary_category_time_us[category],
                len(details),
            )
            for category in _SOURCE_CATEGORIES
        ),
        "ranks_match_tensor_parallel_size": summary.get("ranks") == expected_ranks,
        "profiled_output_matches_warmup": (
            bool(workload.get("warmup_output_sha256"))
            and workload.get("warmup_output_sha256")
            == workload.get("profiled_output_sha256")
        ),
        "source_report_validation_passed": bool(summary.get("source_report_validation"))
        and all(summary["source_report_validation"].values()),
        "generation_validation_passed": bool(summary.get("validation"))
        and all(summary["validation"].values()),
        "csv_hashes_and_row_counts_match": (
            set(recorded_files) == set(actual_file_records)
            and all(
                recorded_files[name] == record
                for name, record in actual_file_records.items()
            )
        ),
    }
    if not all(validation.values()):
        raise RuntimeError(f"serialized operator inventory is invalid: {validation}")
    return validation


def generate_inventory_report(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate the supported three-file operator inventory.

    ``source_dir`` must be a successful output of
    :func:`tools.operator_profiling.trace_report.generate_reports`.
    """

    source_root = Path(source_dir)
    output_root = Path(output_dir)
    source_operator_path = source_root / "operator_list.csv"
    source_shape_path = source_root / "kernel_shape_dtype.csv"
    source_summary_path = source_root / "profile_summary.json"

    operator_fields, source_operators = _read_csv(source_operator_path)
    shape_fields, source_shapes = _read_csv(source_shape_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))

    _require_fields(source_operator_path, operator_fields, set(_OPERATOR_FIELDS))
    _require_fields(
        source_shape_path,
        shape_fields,
        {
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
        },
    )

    if not source_operators:
        raise ValueError(f"operator inventory is empty: {source_operator_path}")
    if not source_shapes:
        raise ValueError(f"kernel detail inventory is empty: {source_shape_path}")
    invalid_categories = sorted(
        {
            row["source_category"]
            for row in [*source_operators, *source_shapes]
            if row["source_category"] not in _SOURCE_CATEGORIES
        }
    )
    if invalid_categories:
        raise ValueError(f"unsupported source categories: {invalid_categories}")
    source_validation = source_summary.get("validation", {})
    if not source_validation or not all(source_validation.values()):
        raise ValueError(
            f"source profiler report did not pass validation: {source_validation!r}"
        )
    source_metadata = source_summary.get("metadata", {})
    source_scope = source_summary.get("scope", {})
    if not _is_platform_profile(
        source_metadata,
        source_scope,
        source_summary.get("dispatch", {}),
    ):
        raise ValueError(
            "source profiler report is not a verified platform_profile run"
        )

    public_operators = sorted(
        {tuple(row[field] for field in _OPERATOR_FIELDS) for row in source_operators},
        key=lambda row: (
            int(row[0]) if row[0].isdigit() else 2**63,
            row[1:],
        ),
    )
    operator_rows = [
        dict(zip(_OPERATOR_FIELDS, row, strict=True)) for row in public_operators
    ]

    provenance: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_operators:
        provenance[
            (row["operator_name"], row["kernel_name"], row["source_category"])
        ].append(row)

    total_time_us = float(source_summary["compute_kernel_time_us"])
    category_time_us = {
        category: float(
            source_summary.get("source_category_time_us", {}).get(category, 0.0)
        )
        for category in _SOURCE_CATEGORIES
    }
    detail_rows: list[dict[str, Any]] = []
    missing_provenance: list[tuple[str, str, str]] = []
    for row in source_shapes:
        relation = (
            row["operator_name"],
            row["kernel_name"],
            row["source_category"],
        )
        records = provenance.get(relation, [])
        if not records:
            missing_provenance.append(relation)
            continue
        duration = float(row["kernel_time_us"])
        detail_rows.append(
            {
                "operator_name": row["operator_name"],
                "kernel_name": row["kernel_name"],
                "variant_index": row["variant_index"],
                "mapping_status": row["mapping_status"],
                "input_shapes": row["input_shapes"],
                "input_dtypes": row["input_dtypes"],
                "candidate_operators": row["candidate_operators"],
                "kernel_event_count": row["kernel_event_count"],
                "kernel_time_us": row["kernel_time_us"],
                "operator_kind": _single_or_json(
                    record["operator_kind"] for record in records
                ),
                "source_category": row["source_category"],
                "source_library": _single_or_json(
                    record["source_library"] for record in records
                ),
                "execution_origin": _single_or_json(
                    record.get("execution_origin", "") for record in records
                ),
                "kernel_time_percent": _percent(duration, total_time_us),
                "kernel_time_percent_of_category": _percent(
                    duration, category_time_us[row["source_category"]]
                ),
            }
        )
    if missing_provenance:
        raise ValueError(
            "kernel details are missing operator provenance for relations: "
            f"{sorted(set(missing_provenance))[:10]}"
        )

    detail_event_count = sum(int(row["kernel_event_count"]) for row in detail_rows)
    detail_time_us = sum(float(row["kernel_time_us"]) for row in detail_rows)
    category_events: Counter[str] = Counter()
    category_times: Counter[str] = Counter()
    for row in detail_rows:
        category = str(row["source_category"])
        category_events[category] += int(row["kernel_event_count"])
        category_times[category] += float(row["kernel_time_us"])

    expected_events = int(source_summary["compute_kernel_event_count"])
    material_validation = {
        "source_report_passed": all(source_validation.values()),
        "platform_profile_configuration_is_valid": True,
        "operator_list_is_not_empty": bool(operator_rows),
        "kernel_details_are_not_empty": bool(detail_rows),
        "all_details_have_operator_provenance": not missing_provenance,
        "all_source_categories_are_valid": all(
            row["source_category"] in _SOURCE_CATEGORIES for row in detail_rows
        ),
        "all_shape_fields_are_json_or_null": all(
            _json_or_null(str(row[field]))
            for row in detail_rows
            for field in ("input_shapes", "input_dtypes", "candidate_operators")
        ),
        "kernel_event_count_is_conserved": detail_event_count == expected_events,
        "kernel_time_is_conserved": _close(
            detail_time_us, total_time_us, len(detail_rows)
        ),
        "source_category_event_count_is_conserved": all(
            category_events[category]
            == int(
                source_summary.get("source_category_event_count", {}).get(category, 0)
            )
            for category in _SOURCE_CATEGORIES
        ),
        "source_category_time_is_conserved": all(
            _close(
                category_times[category],
                category_time_us[category],
                len(detail_rows),
            )
            for category in _SOURCE_CATEGORIES
        ),
    }
    if not all(material_validation.values()):
        raise RuntimeError(
            f"operator inventory validation failed: {material_validation}"
        )

    operator_path = output_root / "operator_list.csv"
    detail_path = output_root / "kernel_details_report.csv"
    summary_path = output_root / "summary.json"
    _atomic_csv(operator_path, _OPERATOR_FIELDS, operator_rows)
    _atomic_csv(detail_path, _DETAIL_FIELDS, detail_rows)

    public_scope_fields = {
        "execution_mode",
        "rank_aggregation",
        "global_torch_compile_enabled",
        "warmup_included",
        "included_operator_kinds",
        "execution_origins",
        "source_categories",
        "vendor_classification",
        "excluded_gpu_activity",
        "time_denominator",
        "inventory_source_policy",
    }
    scope = {
        key: value
        for key, value in source_summary.get("scope", {}).items()
        if key in public_scope_fields
    }
    scope.update(
        {
            "report_type": "operator_inventory",
            "profile_mode": "platform_profile",
            "purpose": "operator API, kernel, shape, frequency, and device-time inventory",
        }
    )
    summary = {
        "schema_version": 1,
        "report_type": "operator_inventory",
        "scope": scope,
        "metadata": source_metadata,
        "ranks": source_summary.get("ranks", []),
        "compute_kernel_event_count": expected_events,
        "compute_kernel_time_us": total_time_us,
        "unique_stable_kernel_names": source_summary.get(
            "unique_stable_kernel_names", 0
        ),
        "source_category_unique_operator_count": source_summary.get(
            "source_category_unique_operator_count", {}
        ),
        "source_category_event_count": source_summary.get(
            "source_category_event_count", {}
        ),
        "source_category_time_us": source_summary.get("source_category_time_us", {}),
        "source_category_percent_of_all_compute": source_summary.get(
            "source_category_percent_of_all_compute", {}
        ),
        "source_library_event_count": source_summary.get(
            "source_library_event_count", {}
        ),
        "source_library_time_us": source_summary.get("source_library_time_us", {}),
        "execution_origin_event_count": source_summary.get(
            "execution_origin_event_count", {}
        ),
        "execution_origin_time_us": source_summary.get("execution_origin_time_us", {}),
        "mapping_event_count_by_status": source_summary.get(
            "mapping_event_count_by_status", {}
        ),
        "mapping_time_us_by_status": source_summary.get(
            "mapping_time_us_by_status", {}
        ),
        "excluded_event_count_by_reason": source_summary.get(
            "excluded_event_count_by_reason", {}
        ),
        "excluded_time_us_by_reason": source_summary.get(
            "excluded_time_us_by_reason", {}
        ),
        "dispatch": source_summary.get("dispatch", {}),
        "torch_compile": source_summary.get("torch_compile", {}),
        "per_rank": source_summary.get("per_rank", {}),
        "source_report_validation": source_validation,
        "validation": material_validation,
        "files": {},
    }
    summary["files"] = {
        "operator_list.csv": _file_record(operator_path),
        "kernel_details_report.csv": _file_record(detail_path),
    }
    _atomic_json(summary_path, summary)
    validate_inventory_package(output_root)
    return summary


__all__ = ["generate_inventory_report", "validate_inventory_package"]
