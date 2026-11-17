#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stage the two reviewed wheel artifacts for the NVIDIA CI image build.
#
# This entry only prepares a build context; it never builds the full image
# and never touches the repository working tree:
#   1. build the sglang-plugin-FL wheel from the current checkout
#      (standard `python -m build --wheel`, output redirected to the
#      staging area, so no dist/ appears in the repository);
#   2. download the fixed SGLang 0.5.18 wheel and verify its SHA256;
#   3. run the plugin-wheel acceptance checks (METADATA, nvidia extra
#      anchors, both SGLang entry-point groups, nvidia_runtime.py
#      presence, no tests);
#   4. assemble a temporary build context under $TMPDIR with the wheels
#      under artifacts/ and the minimal Containerfile at its root;
#   5. print the reproducibility record (source commit, wheel paths,
#      SHA256s, context path, and the exact docker build command).
#
# Wheels are never committed; the staging directory is disposable.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SGLANG_WHEEL_NAME="sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl"
SGLANG_WHEEL_URL="https://files.pythonhosted.org/packages/48/02/8c15afa2bb3f4d060d4ff7a8cd66d35bfa233f94b41f02a2d59b439dbfde/${SGLANG_WHEEL_NAME}"
SGLANG_WHEEL_SHA256="fba7bb31a5de4c014945cbc50e9357c9253cc965a423736c324b2aaccd2270dc"

stage="$(mktemp -d "${TMPDIR:-/tmp}/nv-two-wheel-stage.XXXXXX")"
trap 'rm -rf "$stage"' EXIT

# Keep the staging area strictly outside the repository.
case "$(cd "$stage" && pwd -P)/" in
    "$(cd "$repo_root" && pwd -P)/"*)
        echo "FATAL: staging directory resolved inside the repository" >&2
        exit 1
        ;;
esac

echo "=== Building sglang-plugin-FL wheel from the checkout ==="
python3 -m build --wheel --outdir "$stage/dist" "$repo_root" >/dev/null
plugin_wheel="$(ls "$stage"/dist/sglang_fl-*.whl | head -1)"
[ -n "$plugin_wheel" ] || { echo "FATAL: no sglang_fl wheel produced" >&2; exit 1; }

echo "=== Staging the pinned SGLang wheel ==="
sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
if [ -n "${SGLANG_WHEEL_FILE:-}" ]; then
    # Pre-fetched copy (offline or hostile-transit staging); the digest
    # check below still applies, so a wrong file fails loudly.
    cp "${SGLANG_WHEEL_FILE:?empty SGLANG_WHEEL_FILE}" "$sglang_wheel"
else
    # files.pythonhosted.org resets transpacific connections at times;
    # HTTP/1.1 plus retry-with-resume so one reset does not sink the run.
    curl -fsSL --http1.1 --retry 5 --retry-all-errors --continue-at - \
        "$SGLANG_WHEEL_URL" -o "$sglang_wheel"
fi

echo "=== Verifying the staged SGLang digest ==="
sglang_sha256="$(shasum -a 256 "$sglang_wheel" | cut -d' ' -f1)"
if [ "$sglang_sha256" != "$SGLANG_WHEEL_SHA256" ]; then
    echo "FATAL: SGLang wheel SHA256 mismatch: expected $SGLANG_WHEEL_SHA256, got $sglang_sha256" >&2
    exit 1
fi

echo "=== Acceptance checks on the plugin wheel ==="
python3 "$repo_root/docker/cuda/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"

echo "=== Assembling the build context ==="
context="$(mktemp -d "${TMPDIR:-/tmp}/nv-two-wheel-context.XXXXXX")"
mkdir "$context/artifacts"
cp "$plugin_wheel" "$context/artifacts/"
cp "$sglang_wheel" "$context/artifacts/"
cp "$repo_root/docker/cuda/runtime-0.5.18.containerfile" "$context/Containerfile"

echo "=== Cross-checking the Containerfile against the staged wheels ==="
containerfile="$context/Containerfile"
for staged in "$context"/artifacts/*.whl; do
    base="$(basename "$staged")"
    grep -q "artifacts/$base" "$containerfile" || {
        echo "FATAL: Containerfile does not reference artifacts/$base" >&2
        exit 1
    }
done

echo "=== Re-verifying digests inside the context ==="
# Digests are recomputed on the files the docker build will actually see.
plugin_sha256="$(shasum -a 256 "$context/artifacts/$(basename "$plugin_wheel")" | cut -d' ' -f1)"
sglang_sha256="$(shasum -a 256 "$context/artifacts/$SGLANG_WHEEL_NAME" | cut -d' ' -f1)"
if [ "$sglang_sha256" != "$SGLANG_WHEEL_SHA256" ]; then
    echo "FATAL: context SGLang wheel SHA256 mismatch: expected $SGLANG_WHEEL_SHA256, got $sglang_sha256" >&2
    exit 1
fi

echo
echo "=== staging record ==="
echo "source-commit:  $(git -C "$repo_root" rev-parse HEAD)"
echo "plugin-wheel:   $context/artifacts/$(basename "$plugin_wheel")"
echo "plugin-sha256:  $plugin_sha256   (regenerated per build; not pinned)"
echo "sglang-wheel:   $context/artifacts/$SGLANG_WHEEL_NAME"
echo "sglang-sha256:  $sglang_sha256"
echo "build-context:  $context (disposable; contains Containerfile + artifacts/)"
echo "next step:"
echo "  docker build -f $context/Containerfile -t <tag> $context"
