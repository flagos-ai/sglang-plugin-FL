#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Post-build acceptance for the NVIDIA two-wheel CI image.
#
# The runtime baseline is derived data, not source: it is generated fresh
# from the Runtime base digest pinned in the containerfile, inside that
# base image, and only then compared against the candidate. The candidate
# never supplies its own reference data, and nothing lands in the
# repository -- the baseline lives in a mktemp directory removed on exit.
#
#   stage 1  docker run <pinned base>  verify_runtime_immutable.py
#            --generate --out <temporary baseline directory>
#   stage 2  docker run <candidate>    verify_runtime_immutable.py
#            --verify --evidence-dir <the same temporary baseline>
#
# Usage: verify_image.sh <candidate-image> [base-image]
#        (base defaults to the digest pinned in ARG BASE_IMAGE of
#         runtime-0.5.18.containerfile)

set -euo pipefail

candidate="${1:?usage: verify_image.sh <candidate-image> [base-image]}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
evidence="$(cd "$here/.." && pwd)"
containerfile="$evidence/runtime-0.5.18.containerfile"

if [ "$#" -ge 2 ]; then
    base="$2"
else
    base="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -1)"
    [ -n "$base" ] || {
        echo "FATAL: cannot derive the base image from $containerfile" >&2
        exit 1
    }
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/nv-runtime-baseline.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

echo "=== Stage 1: generating the runtime baseline inside the pinned base ==="
echo "    base:      $base"
echo "    candidate: $candidate"
docker run --rm \
    --platform linux/amd64 \
    --volume "$here":/verifier:ro \
    --volume "$tmp":/baseline \
    --entrypoint python3 \
    "$base" /verifier/verify_runtime_immutable.py --generate --out /baseline

echo "=== Stage 2: verifying the candidate against that baseline ==="
docker run --rm \
    --platform linux/amd64 \
    --volume "$here":/verifier:ro \
    --volume "$tmp":/baseline:ro \
    --entrypoint python3 \
    "$candidate" /verifier/verify_runtime_immutable.py --verify --evidence-dir /baseline
