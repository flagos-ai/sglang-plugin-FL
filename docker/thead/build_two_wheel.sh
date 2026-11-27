#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build the T-Head two-wheel CI image: one uv resolver install of the two
# application wheels on top of the Runtime tag named in the containerfile.
#
# Everything runs inside digest-pinned images; the host needs docker + bash:
#   1. build the sglang-plugin-FL wheel from the HEAD COMMIT of this checkout
#      (git archive -> pinned builder; a dirty tree is deliberately excluded,
#      the staged wheel is the reviewed commit's payload);
#   2. run the plugin-wheel acceptance checks (scripts/check_plugin_wheel.py);
#   3. build the SGLang 0.5.18 wheel from the pinned upstream commit in the
#      same builder (the Runtime needs a Python-3.12 wheel and this commit is
#      a release-branch cherry-pick that no index serves);
#   4. resolve the Runtime's moving tag to the digest this build uses and read
#      the Runtime's constraints from that exact base (pip freeze --all);
#   5. assemble a build context, build the image, print the record.
#
# Wheels and constraints are never committed. The context survives a
# successful run; kernel_fallback.sh stages into it.
#
# Usage: build_two_wheel.sh <image-tag> [runtime-tag]
#   SGLANG_INPUT_WHEEL=<path>   reuse a wheel already built from SGLANG_COMMIT
#   THEAD_TWO_WHEEL_DIR=<dir>   parent directory of the build context

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
containerfile="$repo_root/docker/thead/runtime-0.5.18.containerfile"

# Pinned build inputs: sources by commit, toolchain by digest and version, so
# two builds of one commit produce identical wheels. The wheel SHA256s are
# outputs, recorded in the staging record.
SGLANG_COMMIT="71de97b264b04dcd514cf904003028aefe9775c8"
SGLANG_VERSION="0.5.18"
PLUGIN_WHEEL_NAME="sglang_fl-0.1.0-py3-none-any.whl"
SGLANG_WHEEL_NAME="sglang-$SGLANG_VERSION-py3-none-any.whl"
BUILD_IMAGE="python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
BUILD_TOOLCHAIN="build==1.6.1 setuptools==84.2.1 wheel==0.48.0"
SGLANG_TOOLCHAIN="$BUILD_TOOLCHAIN setuptools-scm==10.3.4 setuptools-rust==1.13.0"
RUNTIME_INDEX="https://mirrors.cloud.tencent.com/pypi/simple"

tag="${1:?usage: build_two_wheel.sh <image-tag> [runtime-tag]}"
runtime_tag="${2:-$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile")}"
commit="$(git -C "$repo_root" rev-parse HEAD)"

stage="$(mktemp -d "${THEAD_TWO_WHEEL_DIR:-${TMPDIR:-/tmp}}/thead-two-wheel-stage.XXXXXX")"
context="$(mktemp -d "${THEAD_TWO_WHEEL_DIR:-${TMPDIR:-/tmp}}/thead-two-wheel-context.XXXXXX")"
cleanup() {
    local status=$?
    rm -rf "$stage"
    [ "$status" -eq 0 ] || rm -rf "$context"
    exit "$status"
}
trap cleanup EXIT

for area in "$stage" "$context"; do
    case "$(cd "$area" && pwd -P)/" in
        "$(cd "$repo_root" && pwd -P)/"*)
            echo "FATAL: staging directory resolved inside the repository" >&2
            exit 1
            ;;
    esac
done

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# build_in_builder <output-dir> <script>
# Runs a wheel build in the pinned builder. The builder's output is kept on
# the host and shown only when the build fails.
build_in_builder() {
    local log
    log="$(mktemp "$stage/build.XXXXXX")"
    if ! docker run --rm --platform linux/amd64 -i \
            -v "$1:/out" -e PIP_INDEX_URL="$RUNTIME_INDEX" "$BUILD_IMAGE" \
            bash -ceu "$2" > "$log" 2>&1; then
        cat "$log" >&2
        exit 1
    fi
}

echo "=== [1/5] Building the sglang-plugin-FL wheel (in the pinned builder) ==="
plugin_wheel="$stage/$PLUGIN_WHEEL_NAME"
git -C "$repo_root" archive --format=tar HEAD | build_in_builder "$stage" "
    mkdir -p /src && tar -x -C /src
    pip install -q $BUILD_TOOLCHAIN
    cd /tmp
    SOURCE_DATE_EPOCH='$(git -C "$repo_root" log -1 --format=%ct "$commit")' \
        python -m build --wheel --no-isolation --outdir /out /src
    ls /out/sglang_fl-*.whl >/dev/null
"

echo "=== [2/5] Plugin wheel acceptance ==="
python3 "$repo_root/docker/thead/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"

echo "=== [3/5] Building the SGLang wheel from the pinned commit ==="
sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
if [ -n "${SGLANG_INPUT_WHEEL:-}" ]; then
    cp "$SGLANG_INPUT_WHEEL" "$sglang_wheel"
    echo "using a staged SGLang wheel: $SGLANG_INPUT_WHEEL"
else
    build_in_builder "$stage" "
        apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
        rm -rf /var/lib/apt/lists/*
        pip install -q $SGLANG_TOOLCHAIN
        # The clone fetches blobs lazily, so the checkout hits the network
        # as well: retry the whole acquisition.
        attempt=0
        until (
            rm -rf /src &&
            git clone --filter=blob:none --quiet https://github.com/sgl-project/sglang.git /src &&
            cd /src &&
            git fetch --depth 1 origin '$SGLANG_COMMIT' -q &&
            git checkout --detach FETCH_HEAD -q
        ); do
            attempt=\$((attempt + 1))
            [ \"\$attempt\" -lt 4 ] || exit 1
            sleep 10
        done
        cd /src
        test \"\$(git rev-parse HEAD)\" = '$SGLANG_COMMIT' || { echo 'commit drift' >&2; exit 1; }
        cp python/pyproject_other.toml python/pyproject.toml
        cd python
        SOURCE_DATE_EPOCH=\"\$(git log -1 --format=%ct)\" \
        SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG='$SGLANG_VERSION' \
            python -m build --wheel --no-isolation --outdir /out .
        ls /out/sglang-'$SGLANG_VERSION'-*.whl >/dev/null
    "
fi

echo "=== [4/5] Runtime constraints, resolved from the moving tag ==="
runtime_digest="$(docker buildx imagetools inspect "$runtime_tag" --format '{{.Manifest.Digest}}')"
runtime_ref="${runtime_tag%@*}@${runtime_digest}"
docker pull --platform linux/amd64 --quiet "$runtime_tag" >/dev/null
mkdir -p "$context/artifacts"
# Every distribution Python can see from the Runtime's Bash environment
# (BASH_ENV puts the compiler slot on PYTHONPATH), read through bash like
# the application does; uv alone cannot see the slot's own distribution.
docker run --rm --platform linux/amd64 "$runtime_ref" \
    bash -c 'cd /tmp && python -m pip freeze --all --disable-pip-version-check' \
    > "$context/artifacts/runtime-constraints.txt"
grep -q '^[A-Za-z0-9._-]*==' "$context/artifacts/runtime-constraints.txt"

echo "=== [5/5] Build context and image ==="
cp "$sglang_wheel" "$plugin_wheel" "$context/artifacts/"
cp "$repo_root/docker/thead/runtime-0.5.18.excludes.txt" "$context/artifacts/"
cp "$containerfile" "$context/Containerfile"
# The default image installs no platform-artifact wheel. The scan covers
# Containerfile instructions only (COPY/ADD/RUN); comments stay free to
# document the fallbacks.
if grep -E '^[[:space:]]*(COPY|ADD|RUN)\b' "$context/Containerfile" \
        | grep -qE 'sglang[-_]kernel|flag_gems[-.]'; then
    echo "FATAL: a Containerfile instruction references a platform-artifact wheel" >&2
    exit 1
fi
docker build --platform linux/amd64 --build-arg BASE_IMAGE="$runtime_ref" \
    -f "$context/Containerfile" -t "$tag" "$context" > "$context/build-image.log" 2>&1 \
    || { tail -n 40 "$context/build-image.log" >&2; exit 1; }

{
    echo "source-commit:  $commit"
    echo "runtime-tag:    $runtime_tag"
    echo "runtime-digest: $runtime_digest"
    echo "plugin-wheel:   $PLUGIN_WHEEL_NAME sha256=$(sha256 "$plugin_wheel")"
    echo "sglang-wheel:   $SGLANG_WHEEL_NAME sha256=$(sha256 "$sglang_wheel")"
    echo "constraints:    runtime-constraints.txt ($(grep -c '==' "$context/artifacts/runtime-constraints.txt") Runtime distributions, never committed)"
    echo "image:          $tag $(docker image inspect "$tag" --format '{{.Id}}')"
    echo "context:        $context"
} | tee "$context/staging-record.txt"
