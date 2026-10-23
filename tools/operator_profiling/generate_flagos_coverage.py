# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Intersect runtime-proven FlagOS operators with a baseline inventory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.operator_profiling.runtime_report import (
    compatible_metadata,
    percent,
    read_csv,
    sha256,
    validate_runtime_report,
    write_csv,
)


def _load(results: Path):
    if results.is_file():
        results = results.parent
    validate_runtime_report(results)
    return (
        results,
        json.loads((results / "summary.json").read_text()),
        read_csv(results / "operator_normalization.csv"),
    )


def generate_coverage(
    baseline: Path,
    plugin: Path,
    output: Path,
    *,
    fusion_mappings: Path | None = None,
    denominator: str = "A",
    fallback_reason: str = "",
) -> dict:
    baseline_dir, baseline_summary, baseline_rows = _load(baseline)
    plugin_dir, plugin_summary, plugin_rows = _load(plugin)
    if denominator == "B" and not fallback_reason.strip():
        raise ValueError(
            "denominator B requires an explicit reason why native D is unavailable"
        )
    expected = "plugin" if denominator == "B" else "native"
    if baseline_summary["metadata"].get("implementation") != expected:
        raise ValueError(
            f"denominator {denominator} requires implementation={expected}"
        )
    if plugin_summary["metadata"].get("implementation") != "plugin":
        raise ValueError("numerator requires a plugin runtime report")
    if baseline_summary["scope"]["rank"] != plugin_summary["scope"]["rank"]:
        raise ValueError("baseline and plugin must use the same parsed rank")
    if (
        baseline_summary["metadata"]["execution_mode"]
        != plugin_summary["metadata"]["execution_mode"]
    ):
        raise ValueError("baseline and plugin execution modes differ")
    compatible_metadata(
        baseline_summary["metadata"],
        plugin_summary["metadata"],
        reference=False,
        cross_platform=denominator == "B",
    )

    baseline_ids = {
        row["normalized_identity"]: row["operator_id"] for row in baseline_rows
    }
    plugin_by_kernel = defaultdict(list)
    for row in plugin_rows:
        plugin_by_kernel[row["kernel_name"]].append(row)
    evidence = json.loads((plugin_dir / "runtime_evidence.json").read_text())
    proven = defaultdict(list)
    aten_proven = defaultdict(list)
    for item in evidence:
        kernel = item.get("kernel_name")
        component = item.get("component")
        if kernel not in plugin_by_kernel or not item.get("successful_launch"):
            continue
        if item.get("rank") != plugin_summary["scope"]["rank"]:
            continue
        if (
            component == "FlagTree"
            and item.get("provider", {}).get("name") != "flagtree"
        ):
            continue
        if component not in {"FlagGems", "FlagTree", "FlagCX"}:
            continue
        for row in plugin_by_kernel[kernel]:
            # A Triton communication implementation still needs FlagCX proof.
            if row["operator_kind"] == "communication" and component != "FlagCX":
                continue
            proven[row["normalized_identity"]].append(item)
        if (
            component == "FlagGems"
            and item.get("aten_api")
            and item.get("phase") == "runtime"
        ):
            aten_proven[item["aten_api"]].append(item)

    covered = defaultdict(list)
    for identity in baseline_ids:
        if identity in proven:
            covered[identity].extend(proven[identity])
        decoded = json.loads(identity)
        if decoded[0] == "aten" and decoded[1] in aten_proven:
            covered[identity].extend(aten_proven[decoded[1]])

    fusion_audit = []
    mapped_kernels = set()
    if fusion_mappings:
        mappings = json.loads(fusion_mappings.read_text())
        if not isinstance(mappings, list):
            raise ValueError("fusion mappings must be a JSON list")
        for mapping in mappings:
            if not mapping.get("evidence", "").strip():
                raise ValueError("fusion mapping requires a verifiable evidence source")
            kernel = mapping["plugin_kernel"]
            kernel_evidence = [
                item
                for rows in proven.values()
                for item in rows
                if item["kernel_name"] == kernel
            ]
            if not kernel_evidence:
                raise ValueError(
                    f"fusion kernel has no successful FlagOS runtime evidence: {kernel}"
                )
            for identity_parts in mapping["baseline_identities"]:
                identity = json.dumps(
                    identity_parts,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                in_baseline = identity in baseline_ids
                fusion_audit.append(
                    {
                        "plugin_kernel": kernel,
                        "baseline_identity": identity,
                        "evidence": mapping["evidence"],
                        "in_baseline": in_baseline,
                    }
                )
                if in_baseline:
                    mapped_kernels.add(kernel)
                    # Set-based intersection guarantees once-only counting,
                    # even for overlapping or repeated mappings.
                    covered[identity].extend(
                        {**item, "fusion_mapping": mapping["evidence"]}
                        for item in kernel_evidence
                    )

    # Once a fused implementation is expanded into baseline components, its
    # own direct identity must not also count as another covered operator.
    for identity, items in list(covered.items()):
        own_kernels = {
            row["kernel_name"]
            for row in plugin_rows
            if row["normalized_identity"] == identity
        }
        remaining = [
            item
            for item in items
            if item.get("fusion_mapping")
            or item["kernel_name"] not in mapped_kernels & own_kernels
        ]
        if remaining:
            covered[identity] = remaining
        else:
            del covered[identity]

    rows = []
    for row in baseline_rows:
        evidence_items = covered.get(row["normalized_identity"], [])
        rows.append(
            {
                **row,
                "flagos_covered": "true" if evidence_items else "",
                "flagos_type": ",".join(
                    sorted({item["component"] for item in evidence_items})
                ),
                "evidence": json.dumps(evidence_items, sort_keys=True)
                if evidence_items
                else "insufficient runtime evidence",
            }
        )
    summary = {
        "denominator_set": denominator,
        "fallback_reason": fallback_reason,
        "covered_operator_count": len(covered),
        "undetermined_operator_count": len(baseline_ids) - len(covered),
        "total_operator_count": len(baseline_ids),
        "coverage_percent(%)": percent(len(covered), len(baseline_ids)),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "operator_flagos_coverage.csv", rows)
    write_csv(output / "operator_flagos_coverage_summary.csv", [summary])
    write_csv(
        output / "fusion_mapping.csv",
        fusion_audit,
        ["plugin_kernel", "baseline_identity", "evidence", "in_baseline"],
    )
    flagos_rows = [
        {
            **row,
            "evidence": json.dumps(proven[row["normalized_identity"]], sort_keys=True),
        }
        for row in plugin_rows
        if row["normalized_identity"] in proven
    ]
    write_csv(
        output / "flagos_operators.csv", flagos_rows, [*plugin_rows[0], "evidence"]
    )
    audit = {
        **summary,
        "baseline": str(baseline_dir.resolve()),
        "plugin": str(plugin_dir.resolve()),
        "baseline_inventory_sha256": sha256(baseline_dir / "operator_list.csv"),
        "plugin_inventory_sha256": sha256(plugin_dir / "operator_list.csv"),
        "fusion_mappings_sha256": sha256(fusion_mappings) if fusion_mappings else None,
        "execution_mode": baseline_summary["metadata"]["execution_mode"],
        "definition": "unique baseline identities intersect runtime-proven FlagOS identities, union verified fusion mappings",
    }
    (output / "coverage_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="results directory or operator_list.csv",
    )
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion-mappings", type=Path)
    parser.add_argument("--denominator", choices=("A", "D", "B"), default="A")
    parser.add_argument("--fallback-reason", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            generate_coverage(
                args.baseline,
                args.plugin,
                args.output_dir,
                fusion_mappings=args.fusion_mappings,
                denominator=args.denominator,
                fallback_reason=args.fallback_reason,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
