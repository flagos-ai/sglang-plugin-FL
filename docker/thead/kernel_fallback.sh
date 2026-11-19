#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# DORMANT FALLBACK, not part of the default image.
#
# The default T-Head image serves `sgl_kernel` through the plugin's semantic
# placeholder (sglang_fl/dispatch/backends/vendor/thead/import_guard.py):
# is_arch_support_pdl() returns False, scalar_type stays absent, any other
# call fails loudly on real use. Switch to this fallback only when device
# logs prove the engines enter real sgl_kernel native ops
# ("... is unavailable: sgl_kernel is not installed ..." RuntimeErrors).
#
# 1. extracts the PPU sglang-kernel wheel from the digest-pinned artifact image
#    (which ships no shell: docker create + docker cp) and checks its SHA256;
# 2. stages it into an existing two-wheel build context under artifacts/.
#
# The image change is a derived containerfile, built FROM the accepted
# candidate (kept manual so the default image cannot drift into shipping the
# vendor kernel by accident):
#
#   ARG CANDIDATE_IMAGE
#   FROM ${CANDIDATE_IMAGE}
#   RUN --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
#       /usr/local/bin/uv pip install --python /flagos/bin/python3 --no-cache --no-deps \
#         '/tmp/artifacts/sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl'
#
# --no-deps: the PPU wheel is installed resolution-free, so it can never
# replace the vendor Torch / FlagTree builds. A real sgl_kernel always wins
# over the placeholder; no plugin change is needed.
#
# Usage: kernel_fallback.sh <build-context>

set -euo pipefail

context="${1:?usage: kernel_fallback.sh <build-context>}"
[ -f "$context/staging-record.txt" ] || {
    echo "FATAL: $context is not a build_two_wheel.sh context" >&2
    exit 1
}

KERNEL_WHEEL_NAME="sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl"
KERNEL_WHEEL_SHA256="d918dd235867b181cfed90908a26d9072dacb2d88255933d40dc6b35b87076fe"
ARTIFACT_IMAGE="harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:09ab4254d56708de7bd94aa706f675a39dc13af0fa9befb95215654b77636edc"

stage="$(mktemp -d "${TMPDIR:-/tmp}/thead-kernel-fallback.XXXXXX")"
cleanup() {
    docker rm -f "$cid" >/dev/null 2>&1 || true
    rm -rf "$stage"
}
cid="$(docker create --platform linux/amd64 "$ARTIFACT_IMAGE" true)" && trap cleanup EXIT

echo "=== Extracting the PPU sglang-kernel wheel from the pinned artifact image ==="
kernel_wheel="$stage/$KERNEL_WHEEL_NAME"
docker cp "$cid:/vendor/$KERNEL_WHEEL_NAME" "$kernel_wheel"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$kernel_wheel" | awk '{print $1}')"
else
    actual="$(shasum -a 256 "$kernel_wheel" | awk '{print $1}')"
fi
[ "$actual" = "$KERNEL_WHEEL_SHA256" ] || {
    echo "FATAL: kernel wheel SHA256 mismatch: expected $KERNEL_WHEEL_SHA256, got $actual" >&2
    exit 1
}
cp "$kernel_wheel" "$context/artifacts/"

echo "staged: $context/artifacts/$KERNEL_WHEEL_NAME (sha256=$actual)"
echo "Build the derived image with the Containerfile documented in this script's header."
