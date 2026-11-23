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
#   4. derive the HCU SGLang wheel from it with
#      tools/wheels/hygon/derive_sglang_wheel.py: the reviewed HCU gating
#      patch is folded in and the wheel is republished as
#      0.5.18+flagos.hcu1 BEFORE any image assembly (strict diff gates run
#      inside the derive; the image never patches site-packages);
#   5. assemble a temporary build context under $TMPDIR with the two
#      wheels under artifacts/ plus the resolver exclusion list and the
#      minimal containerfile;
#   6. print the reproducibility record (source commit, input/derived
#      SGLang wheel + patch SHA256s, plugin wheel SHA256, context path,
#      and the exact docker build command).
#
# Wheels and provenance JSONs are never committed; the staging directory
# is disposable.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SGLANG_WHEEL_NAME="sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
SGLANG_WHEEL_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/packages/sglang/0.5.18+flagos/${SGLANG_WHEEL_NAME}"
SGLANG_WHEEL_SHA256="1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800"
DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"
DERIVED_SGLANG_WHEEL_NAME="sglang-${DERIVED_SGLANG_VERSION}-cp310-cp310-linux_x86_64.whl"
HCU_PATCH="$repo_root/tools/wheels/hygon/sglang-v0.5.18-hcu.patch"

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

echo "=== Fetching and verifying the original SGLang wheel ==="
sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
if [ -n "${SGLANG_INPUT_WHEEL:-}" ] && [ -s "$SGLANG_INPUT_WHEEL" ]; then
    # Offline/dev escape with the SAME identity gate: a pre-secured local
    # copy of the pinned wheel (same override as the derive tests).
    cp "$SGLANG_INPUT_WHEEL" "$sglang_wheel"
    echo "using pre-secured input wheel: $SGLANG_INPUT_WHEEL"
else
    for attempt in 1 2 3 4 5; do
        if curl -fL --retry 3 --retry-delay 5 "$SGLANG_WHEEL_URL" -o "$sglang_wheel"; then
            break
        fi
        echo "wheel fetch attempt ${attempt} failed; retrying" >&2
        sleep 10
    done
fi
actual="$(sha256sum "$sglang_wheel" | awk '{print $1}')"
if [ "$actual" != "$SGLANG_WHEEL_SHA256" ]; then
    echo "FATAL: SHA256 mismatch for $SGLANG_WHEEL_NAME: expected $SGLANG_WHEEL_SHA256, got $actual" >&2
    exit 1
fi
echo "verified $SGLANG_WHEEL_NAME sha256=$actual"

echo "=== Deriving the HCU SGLang wheel ($DERIVED_SGLANG_VERSION) ==="
derive_out="$stage/derived"
python3 "$repo_root/tools/wheels/hygon/derive_sglang_wheel.py" \
    --input-wheel "$sglang_wheel" \
    --expected-input-sha256 "$SGLANG_WHEEL_SHA256" \
    --patch "$HCU_PATCH" \
    --output-dir "$derive_out" \
    --derived-version "$DERIVED_SGLANG_VERSION" \
    --source-commit "$(git -C "$repo_root" rev-parse HEAD)" \
    | tee "$stage/derive.log"
derived_wheel="$derive_out/$DERIVED_SGLANG_WHEEL_NAME"
test -s "$derived_wheel"
derived_provenance="${DERIVED_SGLANG_WHEEL_NAME%.whl}.provenance.json"
test -s "$derive_out/$derived_provenance"
derived_sha256="$(sha256sum "$derived_wheel" | awk '{print $1}')"
patch_sha256="$(sha256sum "$HCU_PATCH" | awk '{print $1}')"

echo "=== Assembling the build context ==="
mkdir -p "$context/artifacts"
cp "$plugin_wheel" "$context/artifacts/"
cp "$derived_wheel" "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME"
# the provenance JSON travels with the context (NOT into the image): the
# post-build acceptance binds the OCI label and the installed patched
# files against it
cp "$derive_out/$derived_provenance" "$context/artifacts/$derived_provenance"
cp "$repo_root/docker/hygon/runtime-0.5.18.excludes.txt" \
   "$context/artifacts/runtime-0.5.18.excludes.txt"
# The staged containerfile references docker/hygon/ (stubs and the clang
# wrapper), so the whole platform directory travels with
# the context. The HCU patch is NOT part of the image inputs anymore: it
# is consumed by the derive step above (its input lives in
# tools/wheels/hygon/).
cp -R "$repo_root/docker" "$context/docker"
cp "$repo_root/docker/hygon/runtime-0.5.18.containerfile" "$context/"
# defense in depth: no patch may enter the build context
if [ -e "$context/docker/hygon/sglang-v0.5.18-hcu.patch" ]; then
    echo "FATAL: HCU patch leaked into the build context" >&2
    exit 1
fi

echo "=== Reproducibility record ==="
commit="$(git -C "$repo_root" rev-parse HEAD)"
echo "source-commit:  $commit"
echo "plugin-wheel:   $(basename "$plugin_wheel") sha256=$(sha256sum "$plugin_wheel" | awk '{print $1}')"
echo "sglang-input:   $SGLANG_WHEEL_NAME sha256=$SGLANG_WHEEL_SHA256"
echo "hcu-patch:      sha256=$patch_sha256"
echo "sglang-derived: $DERIVED_SGLANG_WHEEL_NAME sha256=$derived_sha256"
echo "context:        $context (survives this script; remove manually when done)"
echo "build command:"
echo "  docker/hygon/build_two_wheel.sh <tag>"
echo "  (requires a clean worktree; resolves the runtime tag to one digest,"
echo "   collects runtime-constraints.txt from it, runs the dry-run plan"
echo "   gate, copies the constraints into this context, then builds it)"
