# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Rank-scoped runtime reports built on PR #97's lossless trace parser."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.operator_profiling.coverage_rules import descriptor
from tools.operator_profiling.export_schema import (
    AUDIT_FILE,
    DETAIL_FIELDS,
    OPERATOR_FIELDS,
    export_detail,
    export_operator,
    operator_kind,
)
from tools.operator_profiling.eager_reference import (
    attach_eager_reference,
    load_eager_reference,
    read_csv,
    reference_operator_name,
)
from tools.operator_profiling.trace_report import (
    TraceAggregate,
    canonical_json,
    discover_trace_files,
    rank_from_filename,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(duration: int, total: int) -> str:
    if total <= 0:
        return "null"
    value = Decimal(duration) * 100 / Decimal(total)
    if 0 < value < Decimal("0.01"):
        return "<0.01"
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compatible_metadata(
    left: dict, right: dict, *, reference: bool, cross_platform: bool = False
) -> None:
    """Reject mismatched evidence instead of trusting directory names."""
    fields = ["model", "workload_config", "engine_config", "software"]
    if not cross_platform:
        fields.append("hardware")
    if reference:
        fields += ["implementation", "provider"]
    for field in fields:
        if field not in left or field not in right:
            raise ValueError(f"missing comparison metadata: {field}")
        if left[field] != right[field]:
            raise ValueError(
                f"incompatible {field}: {left[field]!r} != {right[field]!r}"
            )
    if reference and right.get("execution_mode") != "eager":
        raise ValueError("reference must be collected in eager mode")
    if reference:
        transient = {
            "SGLANG_FL_PROFILE_EVIDENCE_DIR",
            "SGLANG_FLAGGEMS_LOG_PATH",
            "SGLANG_FL_DISPATCH_LOG",
        }
        configs = [
            {
                key: value
                for key, value in item.get("environment", {}).items()
                if key not in transient
            }
            for item in (left, right)
        ]
        if configs[0] != configs[1]:
            raise ValueError(
                "incompatible plugin/environment configuration for eager reference"
            )


def _totals(rows, count_field):
    result = defaultdict(lambda: [0, 0])
    for row in rows:
        result[row["kernel_name"]][0] += int(row[count_field])
        result[row["kernel_name"]][1] += int(Decimal(str(row["kernel_time_us"])) * 1000)
    return dict(result)


def validate_runtime_report(results: Path) -> dict:
    summary = json.loads((results / "summary.json").read_text())
    details = read_csv(results / "kernel_shape_dtype.csv")
    times = read_csv(results / "kernel_time.csv")
    inventory = read_csv(results / "operator_list.csv")
    checks = {
        "file_checksums": all(
            sha256(results / name) == digest
            for name, digest in summary["sha256"].items()
        ),
        "detail_per_kernel_conservation": _totals(details, "kernel_event_count")
        == summary["per_kernel"],
        "time_per_kernel_conservation": _totals(times, "kernel_call_count")
        == summary["per_kernel"],
        "all_kernels_in_inventory": {r["kernel_name"] for r in inventory}
        == set(summary["per_kernel"]),
        "all_operator_ids_present": all(
            r["operator_id"] not in {"", "null"} for r in inventory
        ),
        "nonempty_runtime": summary["runtime_kernel_event_count"] > 0,
        "runtime_total_count": sum(item[0] for item in summary["per_kernel"].values())
        == summary["runtime_kernel_event_count"],
        "runtime_total_time": sum(item[1] for item in summary["per_kernel"].values())
        == summary["runtime_kernel_time_ns"],
        "all_conservation_checks": bool(summary.get("conservation"))
        and all(summary["conservation"].values()),
    }
    if summary["schema_version"] == "flagos-runtime-v2":
        audit = read_csv(results / AUDIT_FILE)
        normalization = read_csv(results / "operator_normalization.csv")
        raw_status = Counter()
        for row in audit:
            raw_status[row["mapping_status"]] += int(row["kernel_event_count"])
        checks.update(
            {
                "public_operator_columns": list(inventory[0]) == OPERATOR_FIELDS,
                "public_detail_columns": list(details[0]) == DETAIL_FIELDS,
                "public_details_match_audit": details
                == [export_detail(row) for row in audit],
                "audit_per_kernel_conservation": _totals(audit, "kernel_event_count")
                == summary["per_kernel"],
                "raw_mapping_status_preserved": dict(raw_status)
                == summary["raw_mapping_status_counts"],
                "public_kinds_match_normalization": len(inventory) == len(normalization)
                and all(
                    row["operator_kind"]
                    == operator_kind(
                        raw["operator_name"], raw["kernel_name"], raw["operator_kind"]
                    )
                    and all(
                        row[field] == raw[field]
                        for field in ("operator_id", "operator_name", "kernel_name")
                    )
                    for row, raw in zip(inventory, normalization)
                ),
            }
        )
    if not all(checks.values()):
        raise ValueError(f"runtime report validation failed: {checks}")
    return checks


def generate_runtime_report(
    trace_dir: Path,
    output_dir: Path,
    *,
    rank: int = 0,
    metadata: dict,
    profile_id: str | None = None,
    eager_reference: Path | None = None,
    evidence_dir: Path | None = None,
) -> dict:
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    files = discover_trace_files(trace_dir, profile_id)
    selected = [path for path in files if rank_from_filename(path) == rank]
    if not selected:
        raise ValueError(f"no runtime trace for rank {rank}: {trace_dir}")
    aggregate = TraceAggregate(preserve_kernel_names=True)
    for path in selected:
        aggregate.add_trace(path)
    total_ns = sum(aggregate.runtime_kernel_ns.values())
    if not aggregate.runtime_kernel_event_count:
        raise ValueError("runtime trace has no kernel events")

    index = None
    reference_info = None
    if eager_reference is not None:
        reference_summary = json.loads((eager_reference / "summary.json").read_text())
        compatible_metadata(metadata, reference_summary["metadata"], reference=True)
        validate_runtime_report(eager_reference)
        index, reference_info = load_eager_reference(eager_reference, rank)

    details = []
    relations = []
    raw_status = Counter()
    for kernel, variants in sorted(aggregate.runtime_kernel_variants_count.items()):
        for variant_index, key in enumerate(sorted(variants, key=repr), start=1):
            operator = key.operator_name or "null"
            count = variants[key]
            ns = aggregate.runtime_kernel_variants_ns[kernel][key]
            kind = (
                "communication"
                if key.operator_kind == "communication"
                else (
                    "torch_compile" if key.execution_origin == "torch_compile" else ""
                )
            )
            kind, identity, reason = descriptor(operator, kernel, kind=kind)
            relation = {
                "operator_name": operator,
                "operator_kind": kind,
                "kernel_name": kernel,
            }
            row = {
                "operator_name": operator,
                "kernel_name": kernel,
                "variant_index": variant_index,
                "mapping_status": key.mapping_status,
                "input_shapes": key.input_shapes,
                "input_dtypes": key.input_dtypes,
                "candidate_operators": key.candidate_operators or "null",
                "kernel_event_count": count,
                "kernel_time_us": ns / 1000,
                "raw_operator_name": operator,
                "source_category": key.source_category,
                "source_library": key.source_library,
                "dispatch_impl_id": key.dispatch_impl_id or "null",
                "dispatch_impl_kind": key.dispatch_impl_kind or "null",
            }
            if index is not None:
                enriched, enriched_details, _ = attach_eager_reference(
                    [relation], [row], index
                )
                relation, row = enriched[0], enriched_details[0]
                # Do not change inventory attribution in the face of conflicting
                # direct candidates. Ambiguous eager APIs remain one identity.
                observed = json.loads(key.candidate_operators or "null") or []
                names = {item.get("operator_name") for item in observed}
                proposed = set(relation["operator_name"].split(","))
                if names and not proposed <= names:
                    relation = {
                        "operator_name": operator,
                        "operator_kind": kind,
                        "kernel_name": kernel,
                    }
                elif relation["operator_kind"] == "eager-refer":
                    _, identity, reason = descriptor(relation["operator_name"], kernel)
            details.append(row)
            raw_status[key.mapping_status] += count
            relations.append((identity, relation, reason, ns))

    # Identical API/kernel rows merge after reference resolution. The raw
    # nanoseconds are summed before formatting; direct labels take precedence.
    grouped = {}
    for identity, relation, reason, ns in relations:
        group_key = (identity, relation["operator_name"], relation["kernel_name"])
        if group_key not in grouped:
            grouped[group_key] = [dict(relation), reason, 0]
        item = grouped[group_key]
        item[2] += ns
        if relation["operator_kind"] != "eager-refer":
            item[0] = dict(relation)
    identities = sorted({key[0] for key in grouped})
    ids = {identity: str(i + 1) for i, identity in enumerate(identities)}
    inventory = []
    normalization = []
    for (identity, _, _), (relation, reason, ns) in sorted(grouped.items()):
        inventory.append(
            {
                "operator_id": ids[identity],
                **relation,
                "kernel_time_percent(%)": percent(ns, total_ns),
            }
        )
        normalization.append(
            {
                "operator_id": ids[identity],
                **relation,
                "normalized_identity": canonical_json(identity),
                "normalization_rule": reason,
            }
        )

    time_groups = defaultdict(lambda: [0, 0, False])
    for row in details:
        raw = row["raw_operator_name"]
        operator = reference_operator_name(
            raw, row["kernel_name"], index, json.loads(row["candidate_operators"])
        )
        item = time_groups[(operator, row["kernel_name"])]
        item[0] += row["kernel_event_count"]
        item[1] += int(Decimal(str(row["kernel_time_us"])) * 1000)
        item[2] |= operator != raw
    times = [
        {
            "operator_name": operator,
            "refer": "eager-refer" if used else "null",
            "kernel_name": kernel,
            "kernel_call_count": count,
            "kernel_time_us": ns / 1000,
            "percent": percent(ns, total_ns),
        }
        for (operator, kernel), (count, ns, used) in sorted(time_groups.items())
    ]
    evidence = []
    for kernel, variants in aggregate.runtime_kernel_variants_count.items():
        for key in variants:
            if (
                key.operator_kind == "communication"
                and (key.operator_name or "").startswith("flagcx::")
                and key.mapping_status.startswith("native_marker_")
            ):
                evidence.append(
                    {
                        "kernel_name": kernel,
                        "component": "FlagCX",
                        "rank": rank,
                        "successful_launch": True,
                        "phase": "runtime",
                        "callable": key.operator_name,
                        "evidence_kind": "active_flagcx_communicator_runtime_marker",
                    }
                )
    if evidence_dir is not None:
        for path in sorted(evidence_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                item = json.loads(line)
                window = metadata.get("runtime_window")
                if (
                    window
                    and item.get("phase") == "runtime"
                    and not (
                        window["start_unix_ns"]
                        <= item.get("timestamp_ns", -1)
                        <= window["end_unix_ns"]
                    )
                ):
                    continue
                if (
                    item.get("rank") == rank
                    and item.get("kernel_name") in aggregate.runtime_kernel_count
                ):
                    evidence.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "operator_list.csv": [export_operator(row) for row in inventory],
        "kernel_time.csv": times,
        "kernel_shape_dtype.csv": [export_detail(row) for row in details],
        AUDIT_FILE: details,
        "operator_normalization.csv": normalization,
    }
    for name, rows in reports.items():
        write_csv(output_dir / name, rows)
    (output_dir / "runtime_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    per_kernel = {
        name: [aggregate.runtime_kernel_count[name], aggregate.runtime_kernel_ns[name]]
        for name in sorted(aggregate.runtime_kernel_count)
    }
    summary = {
        "schema_version": "flagos-runtime-v2",
        "public_csv_contract": "cyber-pioneer/op_profile",
        "audit_file": AUDIT_FILE,
        "scope": {
            "rank": rank,
            "collected_ranks": sorted({rank_from_filename(p) for p in files}),
            "phase": "runtime_only",
            "warmup_included": False,
            "graph_capture_included": False,
        },
        "metadata": metadata,
        "trace_files": [str(path) for path in selected],
        "runtime_kernel_event_count": aggregate.runtime_kernel_event_count,
        "runtime_kernel_time_ns": total_ns,
        "total_operator_count": len(ids),
        "per_kernel": per_kernel,
        "raw_mapping_status_counts": dict(raw_status),
        "eager_reference": reference_info,
        "timing_unavailable_reason": None
        if total_ns > 0
        else "no positive kernel duration in trace",
        "sha256": {
            name: sha256(output_dir / name)
            for name in [*reports, "runtime_evidence.json"]
        },
        "conservation": {
            "detail_per_kernel": _totals(details, "kernel_event_count") == per_kernel,
            "time_per_kernel": _totals(times, "kernel_call_count") == per_kernel,
            "inventory_total_time": sum(item[2] for item in grouped.values())
            == total_ns,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["validation"] = validate_runtime_report(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, required=True, help="workload.json saved by the runner"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--eager-reference", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    summary = generate_runtime_report(
        args.trace_dir,
        args.output_dir,
        rank=args.rank,
        metadata=metadata["coverage_metadata"],
        eager_reference=args.eager_reference,
        evidence_dir=args.evidence_dir,
    )
    print(
        json.dumps(
            {
                "operators": summary["total_operator_count"],
                "validation": summary["validation"],
            }
        )
    )


if __name__ == "__main__":
    main()
