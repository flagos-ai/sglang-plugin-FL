#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stage the two reviewed wheel artifacts for the Hygon CI image build.
#
# This entry only prepares a build context; it never builds the full image
# and never touches the repository working tree:
#   1. build the sglang-plugin-FL wheel from the current checkout
#      (standard `python -m build --wheel`, output redirected to the
#      staging area, so no dist/ appears in the repository);
#   2. run the plugin-wheel acceptance checks (METADATA, hygon extra
#      anchors, both SGLang entry-point groups, Hygon attention backend
#      and dispatch config presence, no tests, no triton declaration);
#   3. download the fixed flagos-built SGLang 0.5.18 wheel and verify its
#      SHA256;
#   4. assemble a temporary build context under $TMPDIR with the wheels
#      under artifacts/ plus the resolver exclusion list and the minimal
#      containerfile;
#   5. print the reproducibility record (source commit, wheel paths,
#      SHA256s, context path, and the exact docker build command).
#
# Wheels are never committed; the staging directory is disposable.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SGLANG_WHEEL_NAME="sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
SGLANG_WHEEL_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/packages/sglang/0.5.18+flagos/${SGLANG_WHEEL_NAME}"
SGLANG_WHEEL_SHA256="1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800"

stage="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-stage.XXXXXX")"
# The final build context must SURVIVE this script (the printed docker
# build command references it), so it lives outside the trapped cleanup
# area; only the intermediate staging area is disposable.
context="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-context.XXXXXX")"
trap 'rm -rf "$stage"' EXIT

# Keep both areas strictly outside the repository.
for area in "$stage" "$context"; do
    case "$(cd "$area" && pwd -P)/" in
        "$(cd "$repo_root" && pwd -P)/"*)
            echo "FATAL: staging directory resolved inside the repository" >&2
            exit 1
            ;;
    esac
done

echo "=== Building the sglang-plugin-FL wheel ==="
python3 -m build --wheel --outdir "$stage" "$repo_root" >/dev/null
plugin_wheel="$stage/sglang_fl-0.1.0-py3-none-any.whl"
test -s "$plugin_wheel"

echo "=== Plugin wheel acceptance ==="
python3 "$repo_root/docker/hygon/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"

echo "=== Fetching and verifying the SGLang wheel ==="
sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
for attempt in 1 2 3 4 5; do
    if curl -fL --retry 3 --retry-delay 5 "$SGLANG_WHEEL_URL" -o "$sglang_wheel"; then
        break
    fi
    echo "wheel fetch attempt ${attempt} failed; retrying" >&2
    sleep 10
done
actual="$(sha256sum "$sglang_wheel" | awk '{print $1}')"
if [ "$actual" != "$SGLANG_WHEEL_SHA256" ]; then
    echo "FATAL: SHA256 mismatch for $SGLANG_WHEEL_NAME: expected $SGLANG_WHEEL_SHA256, got $actual" >&2
    exit 1
fi
echo "verified $SGLANG_WHEEL_NAME sha256=$actual"

echo "=== Assembling the build context ==="
mkdir -p "$context/artifacts"
cp "$plugin_wheel" "$context/artifacts/"
cp "$sglang_wheel" "$context/artifacts/"
cp "$repo_root/docker/hygon/runtime-0.5.18.excludes.txt" \
   "$context/artifacts/runtime-0.5.18.excludes.txt"
# The staged containerfile references docker/hygon/ (build scripts via
# bind mounts, stubs, the clang wrapper, the top-level constraints), so
# the whole platform directory travels with the context.
cp -R "$repo_root/docker" "$context/docker"
cp "$repo_root/docker/hygon/runtime-0.5.18.containerfile" "$context/"

echo "=== Reproducibility record ==="
commit="$(git -C "$repo_root" rev-parse HEAD)"
echo "source-commit:  $commit"
echo "plugin-wheel:   $(basename "$plugin_wheel") sha256=$(sha256sum "$plugin_wheel" | awk '{print $1}')"
echo "sglang-wheel:   $SGLANG_WHEEL_NAME sha256=$SGLANG_WHEEL_SHA256"
echo "context:        $context (survives this script; remove manually when done)"
echo "build command:"
echo "  docker build -f $context/runtime-0.5.18.containerfile -t <tag> $context"
