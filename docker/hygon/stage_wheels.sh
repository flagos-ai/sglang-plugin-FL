#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stage the wheel artifacts for the Hygon two-wheel CI image build.
#
# This entry only prepares a build context; it never builds the image. The
# plugin wheel build leaves its git-ignored build/ and *.egg-info/ in the
# checkout, nothing else there changes:
#   1. build the sglang_fl wheel from the current checkout, with zip
#      timestamps taken from the commit so two builds of one commit are
#      identical;
#   2. run the plugin-wheel acceptance checks
#      (docker/hygon/scripts/check_plugin_wheel.py);
#   3. fetch the published SGLang and xgrammar wheels and verify their
#      SHA256 (SGLANG_INPUT_WHEEL / XGRAMMAR_INPUT_WHEEL stage a pre-fetched
#      copy instead; the digest check still applies);
#   4. derive the two HCU platform wheels with tools/wheels/hygon/ and check
#      both against their pinned digests: one version, one content. An
#      already derived wheel (a published artifact, or a secured local copy)
#      is staged instead when SGLANG_DERIVED_WHEEL / XGRAMMAR_DERIVED_WHEEL
#      is set;
#   5. assemble a temporary build context under $TMPDIR and print the record.
#
# Wheels are never committed; the staging area is disposable.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tools="$repo_root/tools/wheels/hygon"

PLUGIN_WHEEL_NAME="sglang_fl-0.1.0-py3-none-any.whl"
SGLANG_WHEEL_NAME="sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
SGLANG_WHEEL_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/packages/sglang/0.5.18+flagos/${SGLANG_WHEEL_NAME}"
SGLANG_WHEEL_SHA256="1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800"
SGLANG_DERIVED_VERSION="0.5.18+flagos.hcu1"
SGLANG_DERIVED_NAME="sglang-${SGLANG_DERIVED_VERSION}-cp310-cp310-linux_x86_64.whl"
# One version carries exactly one content. A derivation that hashes to
# anything else fails here: different content needs the next local version.
SGLANG_DERIVED_SHA256="edcae8c0678290ec68b83d8ec38537d0ef7852b3f8f183984126bba9e8639b5f"
XGRAMMAR_WHEEL_NAME="xgrammar-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
XGRAMMAR_WHEEL_URL="https://mirrors.cloud.tencent.com/pypi/packages/1a/62/4fba98477159f4f79b2baa33f891708df3585ff6a1ac77b4c64895ec462b/${XGRAMMAR_WHEEL_NAME}"
XGRAMMAR_WHEEL_SHA256="5b80b8060fcf6c94324f563e1d4778029b21d35c556dca5bbf49428d307eca37"
XGRAMMAR_DERIVED_VERSION="0.2.1+flagos.hcu1"
XGRAMMAR_DERIVED_NAME="xgrammar-${XGRAMMAR_DERIVED_VERSION}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
XGRAMMAR_DERIVED_SHA256="c2deb2853ac98cbaf63ecfa7f55dd00c54e25c9c43a34e9edb234f84e316d1a2"
XGRAMMAR_TRITON_REQUIREMENT='triton; platform_system == "Linux" and platform_machine == "x86_64"'

require_sha256() {  # <file> <expected sha256> <what>
    local actual
    actual="$(sha256sum "$1" | awk '{print $1}')"
    [ "$actual" = "$2" ] || {
        echo "FATAL: SHA256 mismatch for $3: expected $2, got $actual" >&2
        exit 1
    }
}

# stage_derived <override variable> <derive output dir> <wheel name> <pinned sha256>
#             <derive command...>
# Derives into the output dir unless the override names an already derived
# wheel, then checks the pinned digest and copies the wheel into the context.
stage_derived() {
    local variable="$1" out="$2" name="$3" pinned="$4" override=""
    eval "override=\"\${$variable:-}\""
    shift 4
    mkdir -p "$out" "$context/artifacts"
    if [ -n "$override" ]; then
        cp "$override" "$out/$name"
        echo "using an already derived wheel: $override"
    elif ! "$@" > "$out/derive.log"; then
        cat "$out/derive.log" >&2
        exit 1
    fi
    require_sha256 "$out/$name" "$pinned" \
        "$name (one version, one content; different content needs the next local version)"
    cp "$out/$name" "$context/artifacts/$name"
}

stage="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-stage.XXXXXX")"
# The build context survives a successful run (build_two_wheel.sh consumes
# it); a failed or interrupted run leaves nothing behind in $TMPDIR.
context="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-context.XXXXXX")"
cleanup() {
    local status=$?
    rm -rf "$stage"
    [ "$status" -eq 0 ] || rm -rf "$context"
    exit "$status"
}
trap cleanup EXIT

# Keep both areas strictly outside the repository.
for area in "$stage" "$context"; do
    case "$(cd "$area" && pwd -P)/" in
        "$(cd "$repo_root" && pwd -P)/"*)
            echo "FATAL: staging directory resolved inside the repository" >&2
            exit 1
            ;;
    esac
done
commit="$(git -C "$repo_root" rev-parse HEAD)"

echo "=== Building the sglang_fl wheel from the checkout ==="
SOURCE_DATE_EPOCH="$(git -C "$repo_root" log -1 --format=%ct HEAD)" \
    python3 -m build --wheel --outdir "$stage/dist" "$repo_root" >/dev/null
plugin_wheel="$stage/dist/$PLUGIN_WHEEL_NAME"
[ -s "$plugin_wheel" ] || {
    echo "FATAL: the build produced no $PLUGIN_WHEEL_NAME" >&2
    exit 1
}

echo "=== Plugin wheel acceptance ==="
python3 "$repo_root/docker/hygon/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"

sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
echo "=== Fetching and verifying the published SGLang wheel ==="
if [ -n "${SGLANG_INPUT_WHEEL:-}" ]; then
    cp "$SGLANG_INPUT_WHEEL" "$sglang_wheel"
else
    curl -fsSL --retry 5 --retry-delay 5 "$SGLANG_WHEEL_URL" -o "$sglang_wheel"
fi
require_sha256 "$sglang_wheel" "$SGLANG_WHEEL_SHA256" "$SGLANG_WHEEL_NAME"

echo "=== Deriving the HCU SGLang wheel ($SGLANG_DERIVED_VERSION) ==="
stage_derived SGLANG_DERIVED_WHEEL "$stage/sglang-derived" \
    "$SGLANG_DERIVED_NAME" "$SGLANG_DERIVED_SHA256" \
    python3 "$tools/derive_sglang_wheel.py" \
        --input-wheel "$sglang_wheel" \
        --expected-input-sha256 "$SGLANG_WHEEL_SHA256" \
        --patch "$tools/sglang-v0.5.18-hcu.patch" \
        --output-dir "$stage/sglang-derived" \
        --derived-version "$SGLANG_DERIVED_VERSION" \
        --source-commit "$commit"

xgrammar_wheel="$stage/$XGRAMMAR_WHEEL_NAME"
echo "=== Fetching and verifying the published xgrammar wheel ==="
if [ -n "${XGRAMMAR_INPUT_WHEEL:-}" ]; then
    cp "$XGRAMMAR_INPUT_WHEEL" "$xgrammar_wheel"
else
    curl -fsSL --retry 5 --retry-delay 5 "$XGRAMMAR_WHEEL_URL" -o "$xgrammar_wheel"
fi
require_sha256 "$xgrammar_wheel" "$XGRAMMAR_WHEEL_SHA256" "$XGRAMMAR_WHEEL_NAME"

echo "=== Deriving the HCU xgrammar wheel ($XGRAMMAR_DERIVED_VERSION) ==="
stage_derived XGRAMMAR_DERIVED_WHEEL "$stage/xgrammar-derived" \
    "$XGRAMMAR_DERIVED_NAME" "$XGRAMMAR_DERIVED_SHA256" \
    python3 "$tools/derive_metadata_wheel.py" \
        --input-wheel "$xgrammar_wheel" \
        --expected-input-sha256 "$XGRAMMAR_WHEEL_SHA256" \
        --drop-requires-dist "$XGRAMMAR_TRITON_REQUIREMENT" \
        --drop-distribution triton \
        --derived-version "$XGRAMMAR_DERIVED_VERSION" \
        --output-dir "$stage/xgrammar-derived" \
        --source-commit "$commit"

echo "=== Assembling the build context ==="
mkdir -p "$context/artifacts"
cp "$plugin_wheel" "$context/artifacts/"
cp "$repo_root/docker/hygon/runtime-0.5.18.excludes.txt" "$context/artifacts/"
cp "$repo_root/docker/hygon/runtime-0.5.18.containerfile" "$context/Containerfile"

echo "=== Staging record ==="
echo "source-commit:  $commit"
echo "plugin-wheel:   $PLUGIN_WHEEL_NAME sha256=$(sha256sum "$plugin_wheel" | awk '{print $1}') (built from this checkout)"
echo "sglang-wheel:   $SGLANG_DERIVED_NAME sha256=$SGLANG_DERIVED_SHA256"
echo "xgrammar-wheel: $XGRAMMAR_DERIVED_NAME sha256=$XGRAMMAR_DERIVED_SHA256"
echo "context:        $context (disposable; build_two_wheel.sh adds the runtime constraints)"
echo "next step:      docker/hygon/build_two_wheel.sh <tag>"
