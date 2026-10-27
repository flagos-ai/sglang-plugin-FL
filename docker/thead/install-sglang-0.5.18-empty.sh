#!/bin/bash
# Install SGLang 0.5.18 (empty / srt_empty mode) plus sglang-plugin-FL on top of a
# T-Head PPU base environment, without replacing the vendor stack.
#
# The base (T-Head PPU release / 0.5.11 empty image) provides the vendor builds that
# cannot come from PyPI: PPU torch, FlagTree (ships the `triton` module), sgl-kernel
# ppu build, FlagGems, FlagCX. This script:
#   1. captures their installed versions as pip constraints (and forbids a standalone
#      `triton` distribution, which would overwrite FlagTree's compiler),
#   2. installs SGLang 0.5.18 from source with the torch-free `srt_empty` extra,
#   3. installs xgrammar 0.2.1 with --no-deps (its metadata requires `triton`),
#   4. installs the plugin (editable) without touching dependencies,
#   5. checks that the vendor stack is unchanged and the plugin activates as thead.
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
#   SKIP_DEVICE_CHECK=1  skip the PPU activation check (e.g. image build without devices)
set -euo pipefail

PYTHON=${PYTHON:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}
SGLANG_SRC=${SGLANG_SRC:-/sgl-workspace/sglang-${SGLANG_VERSION}}
SGLANG_REPO=${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}
PLUGIN_SRC=${PLUGIN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
XGRAMMAR_VERSION=0.2.1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Vendor packages that must keep their installed builds.
VENDOR_PKGS=(torch torchvision torchaudio torchcodec flagtree sglang-kernel
             cuda-python cuda-bindings apache-tvm-ffi torch_memory_saver)

echo "=== [1/5] Capture vendor stack constraints ($PYTHON)"
"$PYTHON" - "$WORK/constraints.txt" "$WORK/vendor-before.txt" "${VENDOR_PKGS[@]}" <<'EOF'
import importlib.metadata as m, sys
out, snap, pkgs = sys.argv[1], sys.argv[2], sys.argv[3:]
lines, found = ["# Vendor stack captured from the base environment; must not change."], []
for p in pkgs:
    try:
        v = m.version(p)
    except m.PackageNotFoundError:
        continue
    lines.append(f"{p}=={v}"); found.append(f"{p}=={v}")
# FlagTree provides the triton module; a standalone triton dist must never be installed.
lines.append("triton<0")
open(out, "w").write("\n".join(lines) + "\n")
open(snap, "w").write("\n".join(found) + "\n")
for name in ("torch", "flagtree", "sglang-kernel"):
    if not any(f.lower().startswith(name + "==") for f in found):
        sys.exit(f"base environment is missing vendor package {name}")
print("\n".join(found))
EOF

echo "=== [2/5] SGLang ${SGLANG_VERSION} source -> $SGLANG_SRC"
if [ ! -f "$SGLANG_SRC/python/pyproject_other.toml" ]; then
  git clone --depth 1 --branch "v${SGLANG_VERSION}" "$SGLANG_REPO" "$SGLANG_SRC"
fi
grep -q '^srt_empty' "$SGLANG_SRC/python/pyproject_other.toml" \
  || { echo "pyproject_other.toml has no srt_empty extra"; exit 1; }

echo "=== [3/5] Install SGLang (srt_empty) under vendor constraints"
cp "$SGLANG_SRC/python/pyproject_other.toml" "$SGLANG_SRC/python/pyproject.toml"
(cd "$SGLANG_SRC/python" && \
  SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" \
  "$PYTHON" -m pip install --no-build-isolation -c "$WORK/constraints.txt" -e ".[srt_empty]")

echo "=== [4/5] xgrammar ${XGRAMMAR_VERSION} (--no-deps) and plugin (editable, --no-deps)"
"$PYTHON" -m pip install --no-deps "xgrammar==${XGRAMMAR_VERSION}"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$PLUGIN_SRC"

echo "=== [5/5] Verify"
"$PYTHON" - "$WORK/vendor-before.txt" "$SGLANG_VERSION" <<'EOF'
import importlib.metadata as m, sys
before = [l for l in open(sys.argv[1]).read().split() if l]
changed = [f"{l} -> {m.version(l.split('==')[0])}" for l in before
           if m.version(l.split("==")[0]) != l.split("==")[1]]
if changed:
    sys.exit("vendor stack changed: " + ", ".join(changed))
try:
    m.distribution("triton")
    sys.exit("a standalone triton distribution was installed over FlagTree")
except m.PackageNotFoundError:
    pass
assert m.version("sglang").split("+")[0] == sys.argv[2], m.version("sglang")
print("sglang", m.version("sglang"), "| xgrammar", m.version("xgrammar"),
      "| transformers", m.version("transformers"), "| vendor stack unchanged")
EOF
if [ "${SKIP_DEVICE_CHECK:-0}" != 1 ]; then
  (cd /tmp && "$PYTHON" -c "
from sglang_fl.dispatch.config.utils import get_platform_name, get_effective_config
import sglang.srt.managers.scheduler, sglang.srt.models.qwen3_5
p = get_platform_name(); assert p == 'thead', p
print('platform', p, get_effective_config())")
fi
echo "=== done"
