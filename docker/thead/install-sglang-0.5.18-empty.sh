#!/bin/bash
# Install SGLang 0.5.18 (empty / srt_empty mode) plus sglang-plugin-FL on top of a
# T-Head PPU operations runtime, without replacing the runtime stack.
#
# The operations runtime (flagos-runtime-thead-ppu2.1.0) provides the
# platform-owned builds: PPU torch, vendor torchvision/torchaudio, FlagTree
# (which supplies the active `triton` module from /opt/flagtree), FlagGems and
# the PPU SDK userspace. This script:
#   1. snapshots EVERY runtime Python distribution and turns the snapshot into
#      pip constraints (a failed resolution is always preferred over a silent
#      runtime upgrade), and asserts the documented vendor stack is present,
#   2. installs SGLang 0.5.18 from source with the torch-free `srt_empty` extra
#      under those constraints,
#   3. installs the application extras that srt_empty does not cover
#      (xgrammar, compressed-tensors, loguru) with --no-deps, because their
#      metadata reaches into the device stack (standalone Triton, public torch)
#      and must not drag second implementations beside the runtime-owned ones,
#   4. optionally installs the plugin (editable, --no-deps),
#   5. verifies zero runtime-distribution drift against the snapshot, records
#      the added closure, and checks the no-device import contracts,
#   6. on a PPU host, verifies the plugin activates as thead.
#
# Used by docker/thead/empty-0.5.18.containerfile; also runnable by hand, e.g. in a
# venv cloned from the base site-packages.
#
# Env:
#   PYTHON          interpreter to install into         (default: python3)
#   SGLANG_SRC      SGLang source dir; cloned if absent  (default: /sgl-workspace/sglang-0.5.18)
#   SGLANG_VERSION  tag to clone                         (default: 0.5.18)
#   SGLANG_REPO     git URL                              (default: https://github.com/sgl-project/sglang.git)
#   PLUGIN_SRC      plugin checkout                      (default: repository containing this script)
#   INSTALL_PLUGIN  1 = install the plugin here (default 1); the image sets 0 and
#                   builds the plugin in its own layer instead
#   SNAPSHOT_DIR    directory for the before/added snapshots (default: /opt)
#   SKIP_DEVICE_CHECK=1  skip the PPU activation check (e.g. image build without devices)
set -euo pipefail

PYTHON=${PYTHON:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}
SGLANG_SRC=${SGLANG_SRC:-/sgl-workspace/sglang-${SGLANG_VERSION}}
SGLANG_REPO=${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}
PLUGIN_SRC=${PLUGIN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-/opt}
XGRAMMAR_VERSION=0.2.1
COMPRESSED_TENSORS_VERSION=0.15.0
LOGURU_VERSION=0.7.3
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== [1/6] Snapshot runtime distributions ($PYTHON)"
"$PYTHON" - "$WORK/constraints.txt" "$WORK/vendor-before.txt" \
    "$SNAPSHOT_DIR/runtime-distributions.before" <<'EOF'
import importlib.metadata as m, sys

def norm(name):
    return name.lower().replace("_", "-")

constraints_path, vendor_before_path, snapshot_path = sys.argv[1:4]
expected = {}
for line in """torch==2.10.0+v0.1.0.ppu2.1.1
torchvision==0.25.0+v0.1.0.ppu2.1.1
torchaudio==2.10.0+v0.1.0.ppu2.1.1
flagtree==0.7.0+ppu3.6
flag-gems==5.4.0
numpy==1.26.4""".splitlines():
    name, version = line.split("==")
    expected[norm(name)] = version

inventory = {}
for dist in m.distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    prev = inventory.setdefault(norm(name), (name, dist.version))
    if prev[1] != dist.version:
        sys.exit(f"duplicate runtime distribution {name}: {prev[1]}, {dist.version}")

missing = [f"{k} (want {v})" for k, v in expected.items() if k not in inventory]
wrong = [f"{inventory[k][0]}: {inventory[k][1]} (want {v})"
         for k, v in expected.items() if k in inventory and inventory[k][1] != v]
if missing or wrong:
    sys.exit("base environment does not match the documented vendor stack: "
             + "; ".join(missing + wrong))

lines = ["# Every runtime-owned distribution: versions must not change."]
for key in sorted(inventory):
    name, version = inventory[key]
    lines.append(f"{name}=={version}")
# A standalone triton distribution would shadow FlagTree's compiler.  Only
# forbid introducing one when the runtime itself does not ship a triton dist.
if "triton" not in inventory:
    lines.append("triton<0")
open(constraints_path, "w").write("\n".join(lines) + "\n")
open(snapshot_path, "w").write(
    "".join(f"{inventory[k][0]}=={inventory[k][1]}\n" for k in sorted(inventory)))
open(vendor_before_path, "w").write(
    "".join(f"{inventory[k][0]}=={inventory[k][1]}\n" for k in sorted(inventory)))
print(f"snapshot: {len(inventory)} distributions -> {snapshot_path}")
for key in sorted(expected):
    print(f"  {inventory[key][0]}=={inventory[key][1]}")
EOF

for path in /opt/flagtree /opt/accl-p; do
  test -d "$path" || { echo "base environment is missing $path"; exit 1; }
done

echo "=== [2/6] SGLang ${SGLANG_VERSION} source -> $SGLANG_SRC"
if [ ! -f "$SGLANG_SRC/python/pyproject_other.toml" ]; then
  git clone --depth 1 --branch "v${SGLANG_VERSION}" "$SGLANG_REPO" "$SGLANG_SRC"
fi
grep -q '^srt_empty' "$SGLANG_SRC/python/pyproject_other.toml" \
  || { echo "pyproject_other.toml has no srt_empty extra"; exit 1; }

echo "=== [3/6] Install SGLang (srt_empty) under runtime constraints"
cp "$SGLANG_SRC/python/pyproject_other.toml" "$SGLANG_SRC/python/pyproject.toml"
(cd "$SGLANG_SRC/python" && \
  SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" \
  "$PYTHON" -m pip install --no-build-isolation -c "$WORK/constraints.txt" -e ".[srt_empty]")

echo "=== [4/6] Application extras (--no-deps) and plugin (optional)"
# srt_empty = sglang[runtime_base] only; runtime_common's compressed-tensors and
# xgrammar are SGLang application requirements and are installed here without
# dependency resolution so their metadata (standalone Triton, public torch)
# cannot replace the runtime-owned stack.  loguru is compressed-tensors' only
# declared runtime dependency missing from the srt_empty closure.
"$PYTHON" -m pip install --no-deps \
  "xgrammar==${XGRAMMAR_VERSION}" \
  "compressed-tensors==${COMPRESSED_TENSORS_VERSION}" \
  "loguru==${LOGURU_VERSION}"
if [ "${INSTALL_PLUGIN:-1}" = 1 ]; then
  "$PYTHON" -m pip install --no-deps --no-build-isolation -e "$PLUGIN_SRC"
fi

echo "=== [5/6] Verify zero drift, added closure, import contracts"
"$PYTHON" - "$SNAPSHOT_DIR/runtime-distributions.before" \
    "$SNAPSHOT_DIR/sglang-added-distributions.txt" "$SGLANG_VERSION" <<'EOF'
import importlib, importlib.metadata as m, sys
from pathlib import Path

snapshot_path, added_path, sglang_version = sys.argv[1:4]

def norm(name):
    return name.lower().replace("_", "-")

before = {}
for line in Path(snapshot_path).read_text().splitlines():
    if line and not line.startswith("#"):
        name, version = line.split("==", 1)
        before[norm(name)] = (name, version)
after = {}
for dist in m.distributions():
    name = dist.metadata.get("Name")
    if name:
        after.setdefault(norm(name), (name, dist.version))

drift = []
for key, (name, version) in before.items():
    if key not in after:
        drift.append(f"{name}=={version} -> REMOVED")
    elif after[key][1] != version:
        drift.append(f"{name}=={version} -> {after[key][0]}=={after[key][1]}")
if drift:
    sys.exit("runtime distribution drift:\n" + "\n".join(drift))
if "triton" not in before and "triton" in after:
    sys.exit("a standalone triton distribution was installed over FlagTree")

added = [after[k] for k in sorted(set(after) - set(before))]
Path(added_path).write_text(
    "".join(f"{name}=={version}\n" for name, version in added))
print(f"zero drift over {len(before)} runtime distributions; "
      f"{len(added)} added -> {added_path}")

# No-device import contracts: everything the image guarantees must import on a
# device-less builder, so a broken image cannot be published.
import torch, torchvision, torchaudio, xgrammar, loguru
from compressed_tensors.config import CompressionFormat
from torchvision.io import decode_jpeg
from sglang import Engine
from sglang.srt.layers import quantization
triton = importlib.import_module("triton")
assert m.version("sglang").split("+")[0] == sglang_version, m.version("sglang")
assert m.version("xgrammar") == "0.2.1"
assert m.version("compressed-tensors") == "0.15.0"
assert m.version("loguru") == "0.7.3"
assert CompressionFormat is not None
assert decode_jpeg is not None
assert quantization.QUANTIZATION_METHODS
print("sglang", m.version("sglang"),
      "| torch", torch.__version__,
      "| torchvision", torchvision.__version__,
      "| triton from", triton.__file__)
EOF

if [ "${SKIP_DEVICE_CHECK:-0}" != 1 ]; then
  echo "=== [6/6] Device activation check"
  (cd /tmp && "$PYTHON" -c "
from sglang_fl.dispatch.config.utils import get_platform_name, get_effective_config
import sglang.srt.managers.scheduler, sglang.srt.models.qwen3_5
p = get_platform_name(); assert p == 'thead', p
print('platform', p, get_effective_config())")
else
  echo "=== [6/6] Device activation check skipped (SKIP_DEVICE_CHECK=1)"
fi
echo "=== done"
