#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prove that two Triton-slot images differ in PYTHONPATH only (host side).

The A/B experiment for reviewer r4627164234 (/opt/flagtree vs /opt/triton)
is only valid if the two arms are the SAME image except for the compiler
slot. Given the two arm images this script checks, from ``docker image
inspect`` and from inside each container:

  1. both carry identical provenance labels (base digest, source commit, the
     three wheel digests, the experiment id) and declare the two known slots;
  2. their layer lists are identical (same RootFS by construction);
  3. the image config differs in exactly ``PYTHONPATH`` (value = the
     declared slot) and the slot label - nothing else;
  4. the file trees the slot choice could touch (/opt/flagtree, /opt/triton,
     site-packages) hash identically in both arms;
  5. each arm matches ITS OWN slot contract (module origin, module version,
     triton distribution metadata) - recorded for the report.

It never weakens a contract: an arm that resolves the wrong tree, ships an
unexpected triton distribution or differs in any other config key fails.

Usage: verify_triton_slot_pair.py --a <image> --b <image> [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from triton_slot import FLAGTREE_SLOT, SLOT_LABEL, VENDOR_SLOT  # noqa: E402

EXPERIMENT_LABEL = "org.flagos.ci.triton-slot-experiment"
BOUND_LABELS = (
    "org.flagos.ci.base-digest",
    "org.flagos.ci.source-commit",
    "org.flagos.ci.sglang-wheel-sha256",
    "org.flagos.ci.xgrammar-wheel-sha256",
    "org.flagos.ci.sglang-fl-wheel-sha256",
    EXPERIMENT_LABEL,
)
TREES = ("/opt/flagtree", "/opt/triton", "<site-packages>")

# runs inside each arm with the arm's own image-level ENV
_ARM_PROGRAM = r"""
import hashlib, importlib.metadata as md, importlib.util, json, pathlib, sys, sysconfig
sys.path.insert(0, "/scripts")
from triton_slot import SLOT_LABEL, assert_slot_root, contract_of, dist_info_version

def tree_digest(root):
    root = pathlib.Path(root)
    manifest = hashlib.sha256()
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        manifest.update(str(path.relative_to(root)).encode() + b"\0")
        manifest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return {"files": count, "sha256": manifest.hexdigest()}

slot = __import__("os").environ["PYTHONPATH"]
observed = assert_slot_root(slot)
spec = importlib.util.find_spec("triton")
assert spec is not None and spec.origin.startswith(slot + "/"), spec
try:
    version = md.version("triton")
except md.PackageNotFoundError:
    version = None
contract = contract_of(slot)
expected = dist_info_version(contract["triton_dist_info"]) if contract["pip_check_sees_triton"] else None
assert version == expected, (version, expected)
print(json.dumps({
    "slot": slot,
    "triton_origin": spec.origin,
    "module_version": observed["module_version"],
    "backends": observed["backends"],
    "triton_distribution": version,
    "trees": {
        "/opt/flagtree": tree_digest("/opt/flagtree"),
        "/opt/triton": tree_digest("/opt/triton"),
        "<site-packages>": tree_digest(sysconfig.get_paths()["purelib"]),
    },
}))
"""


def docker(*args: str, stdin: str | None = None) -> str:
    completed = subprocess.run(
        ["docker", *args], input=stdin, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"FATAL: docker {' '.join(args)} failed:\n{completed.stderr[-800:]}"
        )
    return completed.stdout


def inspect(image: str) -> dict:
    return json.loads(docker("image", "inspect", image))[0]


def env_of(config: dict) -> dict:
    return dict(entry.split("=", 1) for entry in config["Env"])


def arm_report(image: str) -> dict:
    return json.loads(
        docker(
            "run",
            "--rm",
            "-i",
            "--platform",
            "linux/amd64",
            "--volume",
            f"{HERE}:/scripts:ro",
            "--entrypoint",
            "bash",
            image,
            "-c",
            "exec python -",
            stdin=_ARM_PROGRAM,
        )
        .strip()
        .splitlines()[-1]
    )


def compare(a: dict, b: dict) -> dict:
    slot_a = a["Config"]["Labels"].get(SLOT_LABEL)
    slot_b = b["Config"]["Labels"].get(SLOT_LABEL)
    assert slot_a == FLAGTREE_SLOT, f"arm A must be the production slot, got {slot_a!r}"
    assert slot_b == VENDOR_SLOT, f"arm B must be the vendor slot, got {slot_b!r}"

    for label in BOUND_LABELS:
        left, right = a["Config"]["Labels"].get(label), b["Config"]["Labels"].get(label)
        assert left and left == right, f"{label}: {left!r} != {right!r}"

    assert a["RootFS"]["Layers"] == b["RootFS"]["Layers"], "layer lists differ"
    assert (a["Architecture"], a["Os"]) == (b["Architecture"], b["Os"]), (
        "platform differs"
    )

    env_a, env_b = env_of(a["Config"]), env_of(b["Config"])
    differing = sorted(
        k for k in set(env_a) | set(env_b) if env_a.get(k) != env_b.get(k)
    )
    assert differing == ["PYTHONPATH"], f"ENV differs beyond PYTHONPATH: {differing}"
    assert env_a["PYTHONPATH"] == slot_a and env_b["PYTHONPATH"] == slot_b, (
        f"PYTHONPATH {env_a['PYTHONPATH']!r}/{env_b['PYTHONPATH']!r} != declared slots"
    )
    labels_a, labels_b = a["Config"]["Labels"], b["Config"]["Labels"]
    label_diff = sorted(
        k for k in set(labels_a) | set(labels_b) if labels_a.get(k) != labels_b.get(k)
    )
    assert label_diff == [SLOT_LABEL], (
        f"labels differ beyond the slot label: {label_diff}"
    )

    # every other config key (entrypoint, cmd, working dir, user, volumes,
    # ports, ...) and the platform must be identical; build timestamps and
    # ids are not part of the image contract
    def rest(image: dict) -> dict:
        config = {
            k: v for k, v in image["Config"].items() if k not in ("Env", "Labels")
        }
        return {
            "config": config,
            "architecture": image["Architecture"],
            "os": image["Os"],
        }

    assert rest(a) == rest(b), "image config differs beyond ENV/labels"
    return {"slot_a": slot_a, "slot_b": slot_b, "layers": len(a["RootFS"]["Layers"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="image selecting /opt/flagtree")
    parser.add_argument("--b", required=True, help="image selecting /opt/triton")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    meta_a, meta_b = inspect(args.a), inspect(args.b)
    structure = compare(meta_a, meta_b)
    report_a, report_b = arm_report(args.a), arm_report(args.b)
    for tree in TREES:
        assert report_a["trees"][tree] == report_b["trees"][tree], (
            f"{tree} differs between the arms: "
            f"{report_a['trees'][tree]} != {report_b['trees'][tree]}"
        )
    report = {
        "verdict": "single-variable pair: PYTHONPATH (and the slot label) only",
        "arm_a": {"image": args.a, "id": meta_a["Id"], **report_a},
        "arm_b": {"image": args.b, "id": meta_b["Id"], **report_b},
        "structure": structure,
        "bound_labels": {
            label: meta_a["Config"]["Labels"][label] for label in BOUND_LABELS
        },
    }
    print(report["verdict"])
    for arm, name in ((report_a, "A"), (report_b, "B")):
        print(
            f"  arm {name}: slot {arm['slot']} | module {arm['module_version']} | "
            f"backends {arm['backends']} | triton distribution "
            f"{arm['triton_distribution'] or 'absent'} | {arm['triton_origin']}"
        )
    for tree in TREES:
        print(
            f"  {tree}: {report_a['trees'][tree]['files']} files, tree sha256 {report_a['trees'][tree]['sha256'][:16]}… (identical in both arms)"
        )
    print(f"  layers: {structure['layers']} identical; ENV differs in PYTHONPATH only")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
