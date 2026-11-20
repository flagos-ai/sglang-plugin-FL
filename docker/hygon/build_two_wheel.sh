#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build entry for the Hygon two-wheel CI image. The containerfile is pure
# image assembly; everything that used to be build-time acceptance lives
# here (pre-build, against the resolved base digest) or in
# docker/hygon/scripts/verify_image.sh (post-build):
#
#   1. docker/hygon/stage_wheels.sh - build the plugin wheel from this
#      checkout, fetch+verify the fixed flagos SGLang wheel, assemble the
#      disposable build context;
#   2. resolve the Runtime's moving semver tag (the containerfile default)
#      to its repo digest - every later step uses that digest;
#   3. collect the Runtime reference data (freeze, site-file hashes,
#      pip-check baseline, provenance, constraints) from that digest into a
#      mktemp dir - never committed, never in the image;
#   4. run the resolver dry-run plan gate in a container of the same digest,
#      with the exact twin of the containerfile's install command (plus
#      --dry-run), and validate that the plan touches no runtime
#      distribution;
#   5. place the generated runtime-constraints.txt into the build context
#      and build the candidate image FROM THE RESOLVED DIGEST (passed as
#      the BASE_IMAGE build arg), carrying its provenance as OCI labels
#      (org.flagos.ci.base-digest / source-commit / hcu-patch-sha256) so
#      the acceptance step binds to the image itself, not to the checkout;
#   6. run the post-build acceptance (docker/hygon/scripts/verify_image.sh)
#      against the produced tag - this is the release gate.
#
# A failed acceptance aborts this script; --push (the only documented way
# to publish from here) is reached only after acceptance passes.
#
# Provenance: a releasable build requires a completely clean worktree -
# the plugin wheel is built from this checkout and the build context
# carries docker/hygon, so any staged, unstaged or untracked file could
# silently enter the artifact while the labels name the recorded HEAD.
# --allow-dirty is a dev-only escape that cannot push and whose output is
# not releasable evidence.
#
# Usage: build_two_wheel.sh [--push] [--allow-dirty] <tag>

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform_dir="$repo_root/docker/hygon"
scripts="$platform_dir/scripts"
containerfile="$platform_dir/runtime-0.5.18.containerfile"

push=0
allow_dirty=0
args=()
for arg in "$@"; do
    case "$arg" in
        --push) push=1 ;;
        --allow-dirty) allow_dirty=1 ;;
        -h|--help)
            echo "usage: build_two_wheel.sh [--push] [--allow-dirty] <tag>"
            echo "  --allow-dirty  dev build from a dirty worktree;"
            echo "                 not releasable, cannot be pushed"
            exit 0
            ;;
        *) args+=("$arg") ;;
    esac
done
[ "${#args[@]}" -eq 1 ] || { echo "usage: build_two_wheel.sh [--push] [--allow-dirty] <tag>" >&2; exit 1; }
tag="${args[0]}"

if [ "$push" -eq 1 ] && [ "$allow_dirty" -eq 1 ]; then
    echo "FATAL: --allow-dirty cannot be combined with --push (a dirty build is not releasable)" >&2
    exit 1
fi

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [ "$allow_dirty" -eq 0 ] && [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
    echo "FATAL: worktree is dirty; commit or stash first (dev builds: --allow-dirty)" >&2
    git -C "$repo_root" status --short >&2
    exit 1
fi
[ "$allow_dirty" -eq 0 ] || echo "WARNING: --allow-dirty dev build; NOT releasable"

# Single source of truth for the Runtime base: the containerfile's ARG
# default (a moving semver tag per the Runtime owner). Everything below
# resolves it ONCE to a repo digest and uses that digest, so reference
# collection, the dry-run gate and the build all see the same image and the
# record shows exactly what was built from.
BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -n 1)"
[ -n "$BASE_IMAGE" ] || { echo "FATAL: no ARG BASE_IMAGE in $containerfile" >&2; exit 1; }

# Same index priority as the containerfile ARGs (asserted by the
# two-wheel contract unit tests); passed to the dry-run container with -e.
FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple
PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

echo "=== [1/6] Staging the two wheels ==="
staging_log="$(mktemp "${TMPDIR:-/tmp}/hygon-2w-stage-log.XXXXXX")"
bash "$platform_dir/stage_wheels.sh" | tee "$staging_log"
context="$(awk '/^context:/{print $2}' "$staging_log" | tail -n 1)"
rm -f "$staging_log"
[ -n "$context" ] && [ -d "$context" ] || {
    echo "FATAL: could not locate the staged build context" >&2
    exit 1
}

echo "=== [2/6] Resolving the moving tag to a repo digest ==="
docker pull --platform linux/amd64 "$BASE_IMAGE" >/dev/null
BASE_REPO="${BASE_IMAGE%:*}"
RESOLVED_BASE="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$BASE_IMAGE" \
    | grep -F "$BASE_REPO@sha256:" | head -n 1)"
[ -n "$RESOLVED_BASE" ] || {
    echo "FATAL: could not resolve $BASE_IMAGE to a repo digest" >&2
    exit 1
}
echo "runtime base tag:     $BASE_IMAGE"
echo "resolved digest:      $RESOLVED_BASE"

refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-reference.XXXXXX")"
trap 'rm -rf "$refdir"' EXIT

echo "=== [3/6] Collecting the Runtime reference from the resolved digest ==="
docker run --rm --platform linux/amd64 \
    --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$RESOLVED_BASE" \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$RESOLVED_BASE"

echo "=== [4/6] Resolver dry-run plan gate (same resolved digest) ==="
# /plan inside this container is the disposable plan dir; the file mounts
# reproduce the exact /tmp paths the containerfile install consumes:
# the constraints and requirements as files, the staged wheels under
# /tmp/wheels, the single-item exclusion list. dry-run.sh is the containerfile
# install command plus --dry-run and the output redirect (compared by the
# two-wheel contract unit tests).
plan="$refdir/uvplan"
mkdir -p "$plan"
cp "$refdir/runtime-constraints.txt" "$plan/runtime-constraints.txt"
cp "$refdir/runtime-distributions-before.json" "$plan/"
cat > "$plan/dry-run.sh" <<'EOF'
/usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --dry-run \
      --index "$FLAGOS_INDEX" \
      --index "$TENCENT_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
      --constraints /tmp/runtime-constraints.txt \
      -r /tmp/requirements.txt \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      '/tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[hygon]' \
      > /plan/uv-dry-run.txt 2>&1 \
&& python /scripts/validate_uv_plan.py /plan/uv-dry-run.txt /plan
EOF
docker run --rm --platform linux/amd64 \
    -e FLAGOS_INDEX -e TENCENT_INDEX -e PROXY_INDEX \
    --volume "$plan":/plan \
    --volume "$plan/runtime-constraints.txt":/tmp/runtime-constraints.txt:ro \
    --volume "$context/docker/hygon/requirements.txt":/tmp/requirements.txt:ro \
    --volume "$context/artifacts":/tmp/wheels:ro \
    --volume "$context/artifacts/runtime-0.5.18.excludes.txt":/tmp/runtime-0.5.18.excludes.txt:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$RESOLVED_BASE" /plan/dry-run.sh
echo "resolver plan validated: no runtime distribution replaced"
echo "  plan audit: $plan/sglang-resolver-additions.txt (disposable)"

echo "=== [5/6] Building the candidate image ==="
patch_sha256="$(sha256sum "$context/docker/hygon/sglang-v0.5.18-hcu.patch" | awk '{print $1}')"
cp "$refdir/runtime-constraints.txt" "$context/artifacts/runtime-constraints.txt"
docker build --platform linux/amd64 \
    --build-arg "BASE_IMAGE=$RESOLVED_BASE" \
    --label "org.flagos.ci.base-digest=$RESOLVED_BASE" \
    --label "org.flagos.ci.source-commit=$source_commit" \
    --label "org.flagos.ci.hcu-patch-sha256=$patch_sha256" \
    -f "$context/runtime-0.5.18.containerfile" \
    -t "$tag" "$context"

# The build must have consumed exactly the recorded commit: the worktree
# may not have moved, and the image must carry that commit as its label.
[ "$(git -C "$repo_root" rev-parse HEAD)" = "$source_commit" ] || {
    echo "FATAL: HEAD moved during the build" >&2
    exit 1
}
labeled_commit="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.source-commit"}}' "$tag")"
[ "$labeled_commit" = "$source_commit" ] || {
    echo "FATAL: source-commit label mismatch: image=$labeled_commit worktree=$source_commit" >&2
    exit 1
}

echo "=== [6/6] Post-build acceptance (release gate) ==="
bash "$scripts/verify_image.sh" "$tag"

echo "=== Reproducibility record ==="
echo "source-commit:  $source_commit"
echo "runtime tag:    $BASE_IMAGE (moving; resolved at build time)"
echo "resolved base:  $RESOLVED_BASE (baked as org.flagos.ci.base-digest)"
echo "hcu patch:      sha256=$patch_sha256 (org.flagos.ci.hcu-patch-sha256)"
echo "image-tag:      $tag"
echo "image-id:       $(docker image inspect --format '{{.Id}}' "$tag")"
echo "build-context:  $context (from stage_wheels.sh; remove manually when done)"

if [ "$push" -eq 1 ]; then
    echo "=== Pushing (acceptance passed) ==="
    docker push "$tag"
    echo "pushed: $tag"
else
    echo "accepted (not pushed); publish later with: docker push $tag"
fi
