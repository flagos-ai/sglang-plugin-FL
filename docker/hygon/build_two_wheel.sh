#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build entry for the Hygon two-wheel CI image. The containerfile is pure
# image assembly; the steps that need the resolved Runtime base live here:
#
#   1. docker/hygon/stage_wheels.sh prepares the disposable build context;
#   2. the containerfile's moving semver tag is resolved once to a repo
#      digest, and every later step uses that digest;
#   3. the Runtime reference (distribution freeze, site-package file hashes,
#      pip-check baseline, compiler slots) is collected from that digest into
#      a mktemp directory - never committed, never in the image - and the
#      generated runtime-constraints.txt is placed into the context, so the
#      resolver cannot replace a Runtime distribution;
#   4. docker build FROM THE RESOLVED DIGEST (passed as the BASE_IMAGE arg);
#   5. docker/hygon/scripts/verify_image.sh accepts the finished image against
#      the reference from step 3. A failed acceptance aborts this script.
#
# Nothing is pushed here: publish an accepted image with docker push.
#
# Usage: build_two_wheel.sh <tag>

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform_dir="$repo_root/docker/hygon"
scripts="$platform_dir/scripts"
recipe="$platform_dir/runtime-0.5.18.containerfile"

tag="${1:?usage: build_two_wheel.sh <tag>}"

# The plugin wheel is built from this checkout: a dirty tree would produce an
# artifact that names HEAD without being HEAD's content.
if [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
    echo "FATAL: worktree is dirty; commit or stash first" >&2
    git -C "$repo_root" status --short >&2
    exit 1
fi
source_commit="$(git -C "$repo_root" rev-parse HEAD)"

staging_log="$(mktemp "${TMPDIR:-/tmp}/hygon-2w-stage-log.XXXXXX")"
refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-reference.XXXXXX")"
cleanup() {
    local status=$?
    rm -f "$staging_log"
    rm -rf "$refdir"
    exit "$status"
}
trap cleanup EXIT

echo "=== [1/4] Staging the wheels ==="
bash "$platform_dir/stage_wheels.sh" | tee "$staging_log"
context="$(awk '/^context:/{print $2}' "$staging_log" | tail -n 1)"
[ -n "$context" ] && [ -d "$context" ] || {
    echo "FATAL: could not locate the staged build context" >&2
    exit 1
}

echo "=== [2/4] Binding the build to one Runtime digest ==="
# The Runtime owner publishes the base under a moving semver tag; it is
# resolved ONCE and the build uses the digest.
BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$recipe" | head -n 1)"
[ -n "$BASE_IMAGE" ] || { echo "FATAL: no ARG BASE_IMAGE in $recipe" >&2; exit 1; }
docker pull --platform linux/amd64 "$BASE_IMAGE" >/dev/null
RESOLVED_BASE="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$BASE_IMAGE" \
    | grep -F "${BASE_IMAGE%:*}@sha256:" | head -n 1)"
[ -n "$RESOLVED_BASE" ] || {
    echo "FATAL: could not resolve $BASE_IMAGE to a repo digest" >&2
    exit 1
}
echo "runtime base: $BASE_IMAGE -> $RESOLVED_BASE"

echo "=== [3/4] Collecting the Runtime reference from the base digest ==="
docker run --rm --platform linux/amd64 \
    --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$RESOLVED_BASE" \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$RESOLVED_BASE"
cp "$refdir/runtime-constraints.txt" "$context/artifacts/runtime-constraints.txt"

echo "=== [4/4] Building the image ==="
docker build --platform linux/amd64 \
    --build-arg "BASE_IMAGE=$RESOLVED_BASE" \
    -f "$recipe" -t "$tag" "$context"

[ "$(git -C "$repo_root" rev-parse HEAD)" = "$source_commit" ] || {
    echo "FATAL: HEAD moved during the build" >&2
    exit 1
}

echo "=== Post-build acceptance ==="
bash "$scripts/verify_image.sh" "$tag" "$RESOLVED_BASE"

echo "=== Build record ==="
echo "source-commit: $source_commit"
echo "runtime base:  $RESOLVED_BASE"
echo "image:         $tag ($(docker image inspect --format '{{.Id}}' "$tag"))"
echo "accepted (not pushed); build context: $context (remove it when done)"
