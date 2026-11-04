#!/usr/bin/env bash
# Install SGLang 0.5.18 empty mode + sglang-plugin-FL on top of the MetaX empty
# image (torch 2.10.0+metax3.9.0.8, FlagTree 0.7.0.post2+metax3.6, FlagGems
# 5.3.0rc2, FlagCX 0.13.0). Used by docker/metax/empty-0.5.18.containerfile and
# can be run by hand inside an existing container of that image.
#
# The vendor stack is pinned from what is installed, so pip can neither replace
# it nor pull a standalone triton over FlagTree.
#
# Environment (all optional):
#   PYTHON           interpreter to install into          (default: python3 on PATH)
#   PLUGIN_DIR       sglang-plugin-FL checkout             (default: repo containing this script)
#   SGLANG_DIR       where the SGLang source is unpacked   (default: /sgl-workspace/sglang)
#   SGLANG_VERSION   SGLang release tag without "v"        (default: 0.5.18)
#   SGLANG_TARBALL   local SGLang source tarball; skips the download
#   SGLANG_SHA256    expected tarball sha256               (default: the v0.5.18 codeload archive)
#   EDITABLE         1 = pip install -e (dev / CI checkout), 0 = regular install (default: 1)
#   PIP_INDEX_URL    honoured by pip as usual
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
PLUGIN_DIR=${PLUGIN_DIR:-$(cd "$HERE/../.." && pwd)}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}
SGLANG_URL="https://codeload.github.com/sgl-project/sglang/tar.gz/refs/tags/v${SGLANG_VERSION}"
SGLANG_SHA256=${SGLANG_SHA256:-4ed28e4a5bb0bee3a2fe905cea6a4db95299a9c289275bd7f75e617ff7cad361}
SGLANG_PATCH=$HERE/sglang-v${SGLANG_VERSION}-empty-metax.patch
FLAGTREE_MARKER="MACA: skip max_threads check"
EDITABLE=${EDITABLE:-1}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-300} PIP_RETRIES=${PIP_RETRIES:-10}
# SGLang's Rust extensions (router etc.) are not needed for empty mode.
export SGLANG_BUILD_RUST_EXTS=none
pip_install() { "$PYTHON" -m pip install --no-cache-dir "$@"; }
# printf, not echo: bash's echo swallows "-e" as its own option.
editable_flag() { if [ "$EDITABLE" = 1 ]; then printf -- '-e'; fi; }

echo "== [1/6] preflight"
[ -f "$PLUGIN_DIR/pyproject.toml" ] && [ -d "$PLUGIN_DIR/sglang_fl" ] || { echo "PLUGIN_DIR=$PLUGIN_DIR is not a sglang-plugin-FL checkout"; exit 1; }
[ -f "$SGLANG_PATCH" ] || { echo "missing $SGLANG_PATCH"; exit 1; }
"$PYTHON" - "$WORK/constraints.txt" "$WORK/vendor.txt" <<'EOF'
import importlib.metadata as m, sys
required = {"torch": "+metax", "flagtree": "+metax", "flag_gems": "", "flagcx": ""}
optional = ["torchvision", "torchaudio", "numpy", "scipy"]
pins = []
for name, marker in required.items():
    try:
        version = m.version(name)
    except m.PackageNotFoundError:
        sys.exit(f"{name} is not installed; this script expects the MetaX empty base image")
    if marker not in version:
        sys.exit(f"{name}=={version} is not the MetaX build (expected '{marker}' in the version)")
    pins.append(f"{name}=={version}")
for name in optional:
    try:
        pins.append(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        pass
open(sys.argv[2], "w").write("\n".join(pins) + "\n")
# FlagTree provides the ``triton`` module; forbid the standalone wheel.
open(sys.argv[1], "w").write("\n".join(pins + ["triton<0"]) + "\n")
print("vendor pins:", ", ".join(pins))
EOF

# The MetaX empty image ships FlagTree with the C550 max_threads check skipped
# (the driver always reports 512 threads); without it num_warps=16 kernels fail.
COMPILER=$("$PYTHON" -c "import triton.compiler.compiler as c; print(c.__file__)")
grep -q "$FLAGTREE_MARKER" "$COMPILER" || {
  echo "FlagTree in $COMPILER lacks the MetaX max_threads fix; use the MetaX empty base image"; exit 1; }

# Fetch and verify the source before touching the installed packages, so a
# failed download leaves the environment as it was.
echo "== [2/6] SGLang v${SGLANG_VERSION} source -> $SGLANG_DIR"
if [ -e "$SGLANG_DIR" ] && [ -n "$(ls -A "$SGLANG_DIR" 2>/dev/null)" ]; then
  echo "$SGLANG_DIR exists and is not empty; move it away first"; exit 1
fi
TARBALL=${SGLANG_TARBALL:-$WORK/sglang.tar.gz}
if [ -z "${SGLANG_TARBALL:-}" ]; then
  # curl is not part of the MetaX base image; use the interpreter instead.
  "$PYTHON" - "$SGLANG_URL" "$TARBALL" <<'EOF'
import shutil, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=300) as src, open(sys.argv[2], "wb") as dst:
    shutil.copyfileobj(src, dst)
EOF
fi
ACTUAL_SHA256=$(sha256sum "$TARBALL" | cut -d' ' -f1)
if [ "$ACTUAL_SHA256" != "$SGLANG_SHA256" ]; then
  echo "sha256 mismatch for $TARBALL: got $ACTUAL_SHA256, expected $SGLANG_SHA256"; exit 1
fi
echo "sha256 OK ($ACTUAL_SHA256)"
mkdir -p "$SGLANG_DIR"
tar -xzf "$TARBALL" --strip-components=1 -C "$SGLANG_DIR"
# MetaX: is_cuda()/is_cuda_alike() follow the OOT platform; fla device fallback.
git -C "$SGLANG_DIR" apply --check "$SGLANG_PATCH"
git -C "$SGLANG_DIR" apply "$SGLANG_PATCH"

echo "== [3/6] remove previously installed SGLang / plugin"
"$PYTHON" -m pip uninstall -y sglang sglang_fl 2>&1 | grep -E "Successfully uninstalled|not installed" || true

echo "== [4/6] install SGLang .[srt_empty]"
cd "$SGLANG_DIR/python"
cp pyproject_other.toml pyproject.toml
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=$SGLANG_VERSION SETUPTOOLS_SCM_PRETEND_VERSION=$SGLANG_VERSION \
  pip_install --no-build-isolation -c "$WORK/constraints.txt" $(editable_flag) ".[srt_empty]"
# SGLang 0.5.18 expects xgrammar 0.2.1. Its wheel requires the standalone
# triton package, which FlagTree replaces, so install it without dependencies.
pip_install --no-deps "xgrammar==0.2.1"

echo "== [5/6] install sglang-plugin-FL from $PLUGIN_DIR"
pip_install --no-deps --no-build-isolation $(editable_flag) "$PLUGIN_DIR"

echo "== [6/6] verify"
"$PYTHON" - "$WORK/vendor.txt" "$SGLANG_VERSION" "$EDITABLE" <<'EOF'
import importlib.metadata as m, json, sys
for name in ("sglang", "sglang_fl"):
    raw = m.distribution(name).read_text("direct_url.json") or "{}"
    editable = json.loads(raw).get("dir_info", {}).get("editable", False)
    assert editable == (sys.argv[3] == "1"), f"{name}: editable={editable}, EDITABLE={sys.argv[3]}"
changed = []
for line in open(sys.argv[1]).read().split():
    name, version = line.split("==")
    if m.version(name) != version:
        changed.append(f"{line} -> {m.version(name)}")
if changed:
    sys.exit("vendor stack changed: " + ", ".join(changed))
try:
    m.distribution("triton")
    sys.exit("a standalone triton wheel was installed over FlagTree")
except m.PackageNotFoundError:
    pass
assert m.version("sglang") == sys.argv[2], m.version("sglang")
for name in ("sglang", "sglang_fl", "transformers", "xgrammar", "torch", "flagtree", "flag_gems", "flagcx"):
    print(f"  {name}: {m.version(name)}")
EOF
# Importing SGLang initialises the platform; that needs a visible MetaX GPU,
# which a docker build does not have.
if "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  (cd /tmp && "$PYTHON" - <<'EOF'
import sglang.launch_server  # noqa: F401
from sglang.srt.utils import is_cuda, is_cuda_alike
from sglang_fl.dispatch.config.utils import get_platform_name
assert get_platform_name() == "metax", get_platform_name()
assert not is_cuda() and is_cuda_alike(), (is_cuda(), is_cuda_alike())
print("  platform=metax, is_cuda=False, is_cuda_alike=True")
EOF
  )
else
  echo "  no MetaX GPU visible; skipped the runtime import check"
fi
echo "MetaX SGLang ${SGLANG_VERSION} empty-mode install done"
