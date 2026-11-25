#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Turn a TsingMicro (TXDA) SGLang 0.5.11 image into a single SGLang 0.5.18
# "empty" runtime in the system Python. Used by empty-0.5.18.containerfile and
# runnable by hand inside an existing container (same steps, same result).
#
# Keeps the vendor stack (torch, torch_txda, FlagTree, FlagGems, Kuiper
# runtime/CCL, FlagCX) untouched: dependency resolution runs against a
# constraints file captured from the installed vendor packages.
#
# Steps:
#   1. remove SGLang 0.5.11, the old plugin and CUDA-only wheels
#   2. install SGLang 0.5.18 [srt_empty] (non-editable) from a release tarball
#   3. install xgrammar / compressed-tensors without dependencies
#   4. install the plugin from PLUGIN_DIR (editable by default)
#   5. verify (docker/tsingmicro/verify_environment.py --metadata-only)
#
# Environment:
#   SGLANG_VERSION         default 0.5.18
#   SGLANG_SOURCE_URL      tarball URL; file:// works (default: codeload tag)
#   SGLANG_SOURCE_SHA256   optional sha256 of the tarball
#   SGLANG_SRC_DIR         where the source is unpacked (default /opt/sglang-$VERSION)
#   PLUGIN_DIR             plugin checkout to install (default: repository root)
#   PLUGIN_EDITABLE        1 (default) = pip install -e, 0 = regular install
#   PIP_INDEX_URL          honoured by pip as usual
set -euo pipefail

SGLANG_VERSION="${SGLANG_VERSION:-0.5.18}"
SGLANG_SOURCE_URL="${SGLANG_SOURCE_URL:-https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/v${SGLANG_VERSION}}"
SGLANG_SOURCE_SHA256="${SGLANG_SOURCE_SHA256:-}"
SGLANG_SRC_DIR="${SGLANG_SRC_DIR:-/opt/sglang-${SGLANG_VERSION}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="${PLUGIN_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PLUGIN_EDITABLE="${PLUGIN_EDITABLE:-1}"
PY="${PYTHON:-python3}"
PIP=("$PY" -m pip)
export PIP_DISABLE_PIP_VERSION_CHECK=1
# SGLang 0.5.18's build discovers Rust extensions (gRPC router etc.) with
# cargo; none of them is used by srt on TXDA and the image has no toolchain.
# Same setting as the MUSA / Hygon 0.5.18 images.
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"

log() { echo "=== [install_sglang_0518] $*"; }

# ── 0. capture the vendor stack as constraints ────────────────────────────────
# torch / torch_txda / FlagTree / FlagGems are TXDA builds; never let pip
# replace them while resolving SGLang's pure-Python dependencies.
CONSTRAINTS=/opt/tsingmicro-vendor-constraints.txt
"$PY" - "$CONSTRAINTS" <<'PY'
import importlib.metadata as md
import sys

vendor = ["torch", "torch_txda", "flagtree", "flag_gems", "numpy"]
lines = []
for name in vendor:
    try:
        lines.append(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        if name in ("torch", "torch_txda", "flagtree"):
            sys.exit(f"vendor package {name} missing: not a TsingMicro base image")
with open(sys.argv[1], "w") as f:
    f.write("\n".join(lines) + "\n")
print("vendor constraints:", ", ".join(lines))
PY

# ── 1. remove SGLang 0.5.11, the old plugin and CUDA-only wheels ──────────────
# CUDA-only wheels shipped with the regular SGLang 0.5.11 install. None is
# importable on TXDA; srt_empty does not require them.
CUDA_ONLY=(
  sglang-kernel flashinfer-python flashinfer-cubin flash-attn-4 vllm kernels kernels-data
  nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cudnn-frontend nvidia-ml-py nvgpu
  cuda-python cuda-bindings cuda-core cuda-pathfinder cuda-tile
  quack-kernels torch-memory-saver torch-c-dlpack-ext
  numba lm-format-enforcer prometheus-fastapi-instrumentator py-cpuinfo tomlkit backports.strenum
  onnx onnxoptimizer onnxruntime onnxslim
)
installed=$("${PIP[@]}" list --format=freeze 2>/dev/null | sed -E 's/[=@ ].*//' | tr 'A-Z_' 'a-z-')
remove=(sglang sglang-fl sglang_fl)
for p in "${CUDA_ONLY[@]}"; do
  if grep -qx "$(echo "$p" | tr 'A-Z_' 'a-z-')" <<<"$installed"; then remove+=("$p"); fi
done
log "uninstalling: ${remove[*]}"
"${PIP[@]}" uninstall -y "${remove[@]}" 2>&1 | grep -E "Successfully uninstalled|WARNING: Skipping" || true

# Leftovers that pip does not own (stale copies would shadow the new install).
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for d in "$SITE/sglang" "$SITE/sglang_fl"; do
  if [ -d "$d" ]; then log "removing unowned leftover $d"; rm -rf -- "$d"; fi
done
# The 0.5.11 base image keeps the plugin source it was built from; remove it
# so only one plugin copy exists (unless it is the checkout being installed).
OLD_PLUGIN_SRC=/sgl-workspace/sglang-plugin-FL
if [ -d "$OLD_PLUGIN_SRC" ] && [ "$(cd "$OLD_PLUGIN_SRC" && pwd -P)" != "$(cd "$PLUGIN_DIR" && pwd -P)" ]; then
  log "removing 0.5.11-era plugin source $OLD_PLUGIN_SRC"
  rm -rf -- "$OLD_PLUGIN_SRC"
fi

# ── 2. SGLang 0.5.18 [srt_empty] ──────────────────────────────────────────────
if [ ! -f "$SGLANG_SRC_DIR/python/pyproject_other.toml" ]; then
  log "fetching SGLang v${SGLANG_VERSION} from ${SGLANG_SOURCE_URL}"
  mkdir -p "$SGLANG_SRC_DIR"
  tmp=$(mktemp /tmp/sglang-src.XXXXXX.tar.gz)
  curl -fL --retry 3 "$SGLANG_SOURCE_URL" -o "$tmp"
  if [ -n "$SGLANG_SOURCE_SHA256" ]; then
    printf '%s  %s\n' "$SGLANG_SOURCE_SHA256" "$tmp" | sha256sum -c -
  fi
  tar -xzf "$tmp" --strip-components=1 -C "$SGLANG_SRC_DIR"
  rm -f -- "$tmp"
fi
cd "$SGLANG_SRC_DIR/python"
cp pyproject_other.toml pyproject.toml
log "installing SGLang ${SGLANG_VERSION} [srt_empty]"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" \
SETUPTOOLS_SCM_PRETEND_VERSION="$SGLANG_VERSION" \
  "${PIP[@]}" install --no-cache-dir -c "$CONSTRAINTS" '.[srt_empty]'
cd /

# ── 3. optional runtime packages imported at startup ──────────────────────────
# Same pins as the other 0.5.18 vendor images. Their metadata requests a
# standalone triton wheel; FlagTree already provides the module.
"${PIP[@]}" install --no-cache-dir --no-deps xgrammar==0.2.1 compressed-tensors==0.15.0

# ── 4. plugin ─────────────────────────────────────────────────────────────────
log "installing plugin from $PLUGIN_DIR (editable=$PLUGIN_EDITABLE)"
git config --global --add safe.directory "$PLUGIN_DIR" 2>/dev/null || true
if [ "$PLUGIN_EDITABLE" = 1 ]; then
  "${PIP[@]}" install --no-cache-dir --no-deps -e "$PLUGIN_DIR"
else
  "${PIP[@]}" install --no-cache-dir --no-deps "$PLUGIN_DIR"
fi

# ── 5. verify ─────────────────────────────────────────────────────────────────
"$PY" "$SCRIPT_DIR/verify_environment.py" --plugin-root "$PLUGIN_DIR" --metadata-only
log "done"
