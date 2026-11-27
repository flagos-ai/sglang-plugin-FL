#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Resolver gate for retiring the Triton exclusion (reviewer r4627512386).
#
# The corrected xgrammar wheel removes the only active ``triton`` requirement
# of the closure, so the transitional ``--excludes triton`` (and the exact
# pip-check exception that documents it) must have become a no-op before they
# are deleted. This script proves that against the REAL indexes, on the
# resolved Runtime digest, with the staged artifacts of a build context:
#
#   plan A   the containerfile's exact install command + --dry-run
#            (today: still with --excludes)
#   plan B   the same command WITHOUT the --excludes option
#
# Both plans are validated by the normal plan gate (corrected xgrammar
# selected, no standalone triton, no runtime distribution replaced), plan B
# must select the datasets pin, and the two plans must be IDENTICAL.
#
# Only a green run of this gate - followed by a candidate build and the full
# post-build acceptance after the exclusion is deleted - authorizes removing
# docker/hygon/runtime-0.5.18.excludes.txt and the pip-check exemption.
# Nothing here builds or pushes an image; the uv outputs are kept as evidence.
#
# Usage: no_excludes_gate.sh <staging-context-dir> [--evidence-dir DIR]

set -euo pipefail

platform_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$platform_dir/scripts"
containerfile="$platform_dir/runtime-0.5.18.containerfile"
[ "${1:-}" ] || { echo "usage: no_excludes_gate.sh <staging-context-dir> [--evidence-dir DIR>" >&2; exit 1; }
context="${1:?}"
shift
evidence=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --evidence-dir) evidence="${2:?}"; shift 2 ;;
        *) echo "usage: no_excludes_gate.sh <staging-context-dir> [--evidence-dir DIR>" >&2; exit 1 ;;
    esac
done
[ -d "$context/artifacts" ] || {
    echo "FATAL: $context/artifacts not found (stage with docker/hygon/stage_wheels.sh)" >&2
    exit 1
}
[ -n "$evidence" ] || evidence="$(mktemp -d "${TMPDIR:-/tmp}/hygon-no-excludes-gate.XXXXXX")"
mkdir -p "$evidence"

BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -n 1)"

echo "=== [1/3] Resolving the Runtime tag and collecting the resolver inputs ==="
docker pull --platform linux/amd64 "$BASE_IMAGE" >/dev/null
BASE_REPO="${BASE_IMAGE%:*}"
RESOLVED_BASE="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$BASE_IMAGE" \
    | grep -F "$BASE_REPO@sha256:" | head -n 1)"
[ -n "$RESOLVED_BASE" ] || { echo "FATAL: could not resolve $BASE_IMAGE" >&2; exit 1; }
echo "resolved base: $RESOLVED_BASE"
docker run --rm --platform linux/amd64 \
    --volume "$evidence":/out --volume "$scripts":/scripts:ro \
    --entrypoint bash "$RESOLVED_BASE" -c 'exec python "$@"' bash \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$RESOLVED_BASE"
# the same generated constraints the build consumes
cp "$evidence/runtime-constraints.txt" "$context/artifacts/runtime-constraints.txt"

echo "=== [2/3] Two real dry-runs on the same digest and artifacts ==="
# The containerfile's install arguments, verbatim (ARG defaults substituted;
# no mapfile: the macOS host runs bash 3.2).
ARGS=()
while IFS= read -r argument; do ARGS+=("$argument"); done \
    < <(python3 "$scripts/uv_install_contract.py" "$containerfile")
[ "${#ARGS[@]}" -gt 0 ] || { echo "FATAL: no uv install arguments found in $containerfile" >&2; exit 1; }

plan_args() {  # <A|B>: prints the argument list (B drops "--excludes <file>")
    local arm="$1" skip=0 argument
    for argument in "${ARGS[@]}"; do
        if [ "$skip" -eq 1 ]; then skip=0; continue; fi
        if [ "$arm" = "B" ] && [ "$argument" = "--excludes" ]; then skip=1; continue; fi
        printf '%s\n' "$argument"
    done
}

run_plan() {  # <A|B>
    local arm="$1" plan="$evidence/plan-$1" status=0
    mkdir -p "$plan"
    plan_args "$arm" > "$plan/args"
    {
        printf 'plan %s uv arguments: --dry-run' "$arm"
        sed 's/^/ /' "$plan/args" | tr -d '\n'
        printf '\n'
    } | tee "$evidence/plan-$arm.arguments.txt"
    docker run --rm --platform linux/amd64 \
        --volume "$plan":/plan \
        --volume "$context/artifacts":/tmp/artifacts:ro \
        --volume "$scripts":/scripts:ro \
        --entrypoint bash "$RESOLVED_BASE" -c '
            mapfile -t ARGS < /plan/args
            /usr/local/bin/uv pip install --dry-run "${ARGS[@]}" \
                > /plan/uv-dry-run.txt 2>&1 || { cat /plan/uv-dry-run.txt >&2; exit 2; }
            python /scripts/validate_uv_plan.py /plan/uv-dry-run.txt /plan
        ' || status=$?
    if [ "$status" -ne 0 ]; then
        echo "NO-EXCLUDES GATE: plan $arm FAILED (uv/plan status $status)" >&2
        exit "$status"
    fi
}

run_plan A
run_plan B

echo "=== [3/3] The two plans must be identical ==="
count="$(wc -l < "$evidence/plan-B/sglang-resolver-additions.txt" | tr -d ' ')"
if ! diff -u "$evidence/plan-A/sglang-resolver-additions.txt" \
        "$evidence/plan-B/sglang-resolver-additions.txt" > "$evidence/plan-diff.txt"; then
    cat "$evidence/plan-diff.txt"
    echo "NO-EXCLUDES GATE: FAIL (the plans differ; keep the exclusion)" >&2
    exit 1
fi
grep -qx 'datasets==5.2.0' "$evidence/plan-B/sglang-resolver-additions.txt" || {
    echo "NO-EXCLUDES GATE: FAIL (plan B does not select datasets==5.2.0)" >&2
    exit 1
}
echo "plan A (with --excludes) == plan B (without --excludes): $count identical distributions"
grep -x 'xgrammar==0.2.1+flagos.hcu1' "$evidence/plan-B/sglang-resolver-additions.txt" | sed 's/^/  /'
echo "  no standalone triton; no runtime distribution replaced; datasets==5.2.0 selected"
echo "NO-EXCLUDES GATE: PASS"
echo "evidence: $evidence (disposable)"
