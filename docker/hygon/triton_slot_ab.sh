#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Triton compiler slot A/B experiment (reviewer r4627164234): ONE variable.
#
#   arm A  PYTHONPATH=/opt/flagtree  FlagTree 3.6.0, HCU backend, no triton
#                                  dist-info (the production default)
#   arm B  PYTHONPATH=/opt/triton    DTK vendor triton 3.5.1 with dist-info
#
# Both arms are thin layers over ONE accepted candidate image
# (docker/hygon/build_two_wheel.sh output): same resolved Runtime digest,
# same layers, same two application wheels. The arms differ in PYTHONPATH
# and the experiment/slot labels only, which
# docker/hygon/scripts/verify_triton_slot_pair.py proves before each arm
# runs the normal post-build acceptance against its own slot contract.
#
# This script never changes the production default (the containerfile keeps
# /opt/flagtree) and never pushes. It is NOT a slot conclusion: the arms
# still have to pass the Hygon hardware matrix (see docker/hygon/README.md).
#
# Usage: triton_slot_ab.sh [--experiment ID] [--context DIR] <candidate> <tag-a> <tag-b>
#   --context DIR   staging context of the candidate (build record line
#                   "build-context:"), enabling the strict content binding
#                   for the arms; without it the arms run the weaker
#                   identity gate and the report says so.

set -euo pipefail

platform_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$platform_dir/scripts"
SLOT_A="/opt/flagtree"
SLOT_B="/opt/triton"

experiment="triton-slot-ab-$(date -u +%Y%m%dT%H%M%SZ)"
context=""
positional=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --experiment) experiment="${2:?--experiment needs a value}"; shift 2 ;;
        --context) context="${2:?--context needs a value}"; shift 2 ;;
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) positional+=("$1"); shift ;;
    esac
done
[ "${#positional[@]}" -eq 3 ] || {
    echo "usage: triton_slot_ab.sh [--experiment ID] [--context DIR] <candidate> <tag-a> <tag-b>" >&2
    exit 1
}
candidate="${positional[0]}"
tag_a="${positional[1]}"
tag_b="${positional[2]}"

label() { docker image inspect --format "{{index .Config.Labels \"$1\"}}" "$candidate"; }
for required in org.flagos.ci.base-digest org.flagos.ci.source-commit \
                org.flagos.ci.sglang-wheel-sha256 org.flagos.ci.xgrammar-wheel-sha256 \
                org.flagos.ci.sglang-fl-wheel-sha256; do
    value="$(label "$required")"
    [ -n "$value" ] && [ "$value" != "<no value>" ] || {
        echo "FATAL: $candidate lacks the $required label (build it with build_two_wheel.sh)" >&2
        exit 1
    }
done
candidate_id="$(docker image inspect --format '{{.Id}}' "$candidate")"
pinned="hygon-triton-slot-ab-base:${experiment}"
docker tag "$candidate" "$pinned"

work="$(mktemp -d "${TMPDIR:-/tmp}/hygon-triton-slot-ab.XXXXXX")"
trap 'rm -rf "$work"' EXIT
# the pair report survives this script (disposable evidence; never committed)
evidence="$(mktemp -d "${TMPDIR:-/tmp}/hygon-triton-slot-ab-evidence.XXXXXX")"

build_arm() {  # <slot> <tag>
    local slot="$1" tag="$2"
    cat > "$work/Containerfile" <<EOF
FROM ${pinned}
LABEL org.flagos.ci.triton-slot="${slot}"
LABEL org.flagos.ci.triton-slot-experiment="${experiment}"
ENV PYTHONPATH=${slot}
EOF
    docker build --platform linux/amd64 -f "$work/Containerfile" -t "$tag" "$work" >/dev/null
    echo "built arm ${slot}: $tag ($(docker image inspect --format '{{.Id}}' "$tag"))"
}

echo "=== Triton slot A/B experiment ${experiment} ==="
echo "candidate: ${candidate} (${candidate_id})"
build_arm "$SLOT_A" "$tag_a"
build_arm "$SLOT_B" "$tag_b"

echo "=== Single-variable proof (PYTHONPATH only) ==="
python3 "$scripts/verify_triton_slot_pair.py" --a "$tag_a" --b "$tag_b" \
    --json "$evidence/pair-report.json"

echo "=== Post-build acceptance per arm (each against its own slot contract) ==="
verify_args=()
if [ -n "$context" ] && [ -d "$context/artifacts" ]; then
    sglang_wheel="sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl"
    xgrammar_wheel="xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    plugin_wheel="sglang_fl-0.1.0-py3-none-any.whl"
    verify_args=(
        --sglang-wheel-sha256 "$(label org.flagos.ci.sglang-wheel-sha256)"
        --sglang-provenance "$context/artifacts/${sglang_wheel%.whl}.provenance.json"
        --xgrammar-wheel-sha256 "$(label org.flagos.ci.xgrammar-wheel-sha256)"
        --xgrammar-provenance "$context/artifacts/${xgrammar_wheel%.whl}.provenance.json"
        --sglang-fl-wheel "$context/artifacts/$plugin_wheel"
        --sglang-fl-wheel-sha256 "$(label org.flagos.ci.sglang-fl-wheel-sha256)"
        --sglang-fl-provenance "$context/artifacts/${plugin_wheel%.whl}.provenance.json"
    )
    echo "strict content binding: enabled (provenance from $context)"
else
    echo "WARNING: no --context given; the arms run the weaker identity gate"
    echo "(the candidate's own strict acceptance is not repeated here)"
fi
for tag in "$tag_a" "$tag_b"; do
    bash "$scripts/verify_image.sh" "$tag" ${verify_args[@]+"${verify_args[@]}"}
done

echo "=== A/B record ==="
echo "experiment:     $experiment"
echo "candidate:      $candidate_id"
echo "arm A:          $tag_a  $(docker image inspect --format '{{.Id}}' "$tag_a")  PYTHONPATH=$SLOT_A"
echo "arm B:          $tag_b  $(docker image inspect --format '{{.Id}}' "$tag_b")  PYTHONPATH=$SLOT_B"
echo "pair report:    $evidence/pair-report.json (survives this script; remove manually)"
echo "hardware:       NOT run here - both arms still need the Hygon matrix"
