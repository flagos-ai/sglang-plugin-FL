#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build entry for the Hygon two-wheel CI image. The containerfile is pure
# image assembly; everything that used to be build-time acceptance lives
# here (pre-build, against the pinned base digest) or in
# docker/hygon/scripts/verify_image.sh (post-build):
#
#   1. docker/hygon/stage_wheels.sh - build the plugin wheel from this
#      checkout, fetch+verify the fixed flagos SGLang wheel, assemble the
#      disposable build context;
#   2. collect the Runtime reference data (freeze, site-file hashes,
#      pip-check baseline, provenance, constraints) from the PINNED base
#      digest into a mktemp dir - never committed, never in the image;
#   3. run the resolver dry-run plan gate in a container of the same base
#      digest, with the exact twin of the containerfile's install command
#      (plus --dry-run), and validate that the plan touches no runtime
#      distribution;
#   4. place the generated runtime-constraints.txt into the build context
#      and build the candidate image.
#
# Acceptance (which gates the push) is a separate step:
#   docker/hygon/scripts/verify_image.sh <tag>
#
# Usage: build_two_wheel.sh <tag>

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform_dir="$repo_root/docker/hygon"
scripts="$platform_dir/scripts"
containerfile="$platform_dir/runtime-0.5.18.containerfile"

tag="${1:?usage: build_two_wheel.sh <tag>}"

# Single source of truth for the pinned base digest: the containerfile's
# ARG default.
BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -n 1)"
[ -n "$BASE_IMAGE" ] || { echo "FATAL: no ARG BASE_IMAGE in $containerfile" >&2; exit 1; }

# Same index priority as the containerfile ARGs (asserted by the
# two-wheel contract unit tests); passed to the dry-run container with -e.
FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
PYPI_INDEX=https://pypi.org/simple
PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

echo "=== [1/4] Staging the two wheels ==="
staging_log="$(mktemp "${TMPDIR:-/tmp}/hygon-2w-stage-log.XXXXXX")"
bash "$platform_dir/stage_wheels.sh" | tee "$staging_log"
context="$(sed -n 's/^context:[[:space:]]\+\(\S*\).*/\1/p' "$staging_log" | tail -n 1)"
rm -f "$staging_log"
[ -n "$context" ] && [ -d "$context" ] || {
    echo "FATAL: could not locate the staged build context" >&2
    exit 1
}

refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-reference.XXXXXX")"
trap 'rm -rf "$refdir"' EXIT

echo "=== [2/4] Collecting the Runtime reference from the pinned base digest ==="
docker run --rm --platform linux/amd64 \
    --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$BASE_IMAGE" \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$BASE_IMAGE"

echo "=== [3/4] Resolver dry-run plan gate (same base digest) ==="
# /opt/sglang-ci inside this container is the disposable plan dir; the
# mounts reproduce the exact paths the containerfile install will see:
# the constraints at /opt/sglang-ci, the staged wheels under /tmp/wheels,
# the single-item exclusion list at /tmp. dry-run.sh is the containerfile
# install command plus --dry-run and the output redirect (compared by the
# two-wheel contract unit tests).
plan="$refdir/uvplan"
mkdir -p "$plan"
cp "$refdir/runtime-constraints.txt" "$plan/runtime-constraints.txt"
cp "$refdir/runtime-distributions-before.json" "$plan/"
cp "$platform_dir/top-level-constraints.txt" "$plan/top-level-constraints.txt"
cat > "$plan/dry-run.sh" <<'EOF'
/usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --dry-run \
      --index "$FLAGOS_INDEX" \
      --index "$PYPI_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
      --constraints /opt/sglang-ci/runtime-constraints.txt \
      --constraints /opt/sglang-ci/top-level-constraints.txt \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      '/tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[hygon]' \
      > /opt/sglang-ci/uv-dry-run.txt 2>&1 \
&& python /scripts/validate_uv_plan.py /opt/sglang-ci/uv-dry-run.txt
EOF
docker run --rm --platform linux/amd64 \
    -e FLAGOS_INDEX -e PYPI_INDEX -e PROXY_INDEX \
    --volume "$plan":/opt/sglang-ci \
    --volume "$context/artifacts":/tmp/wheels:ro \
    --volume "$context/artifacts/runtime-0.5.18.excludes.txt":/tmp/runtime-0.5.18.excludes.txt:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$BASE_IMAGE" /opt/sglang-ci/dry-run.sh
echo "resolver plan validated: no runtime distribution replaced"
echo "  plan audit: $plan/sglang-resolver-additions.txt (disposable)"

echo "=== [4/4] Building the candidate image ==="
cp "$refdir/runtime-constraints.txt" "$context/artifacts/runtime-constraints.txt"
docker build --platform linux/amd64 \
    -f "$context/runtime-0.5.18.containerfile" \
    -t "$tag" "$context"

echo "=== Reproducibility record ==="
echo "source-commit:  $(git -C "$repo_root" rev-parse HEAD)"
echo "base-digest:    $BASE_IMAGE"
echo "image-tag:      $tag"
echo "image-id:       $(docker image inspect --format '{{.Id}}' "$tag")"
echo "build-context:  $context (from stage_wheels.sh; remove manually when done)"
echo "accept next:"
echo "  docker/hygon/scripts/verify_image.sh $tag"
