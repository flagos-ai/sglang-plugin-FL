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
#   2. clones SGLang at the digest-pinned v0.5.18 commit and asserts HEAD,
#   3. installs SGLang from source with the torch-free `srt_empty` extra under
#      those constraints,
#   4. installs the application extras that were selected by actual startup
#      failures on PPU (xgrammar, compressed-tensors, loguru) with --no-deps,
#      because their metadata reaches into the device stack (standalone Triton,
#      public torch) and must not drag second implementations beside the
#      runtime-owned ones,
#   5. verifies zero runtime-distribution drift against the snapshot, the exact
#      added closure against the committed additions lock, the reviewed
#      `pip check` baseline, and the no-device import contracts,
#   6. on a PPU host, verifies the plugin activates as thead.
#
# VERIFY_ONLY=1 skips steps 2-4 and re-runs step 5 against the current
# environment, for use by the final image layer and CI setup.
#
# Used by docker/thead/empty-0.5.18.containerfile; also runnable by hand, e.g. in a
# venv cloned from the base site-packages.
#
# Env:
#   PYTHON          interpreter to install into         (default: python3)
#   SGLANG_SRC      SGLang source dir; cloned if absent  (default: /sgl-workspace/sglang-0.5.18)
#   SGLANG_VERSION  version tag to clone                 (default: 0.5.18)
#   SGLANG_COMMIT   exact commit to check out and assert (default: the v0.5.18 tag commit)
#   SGLANG_REPO     git URL                              (default: https://github.com/sgl-project/sglang.git)
#   PLUGIN_SRC      plugin checkout                      (default: repository containing this script)
#   ADDITIONS_LOCK  exact added-closure lock file        (default: <script dir>/empty-0.5.18.additions.lock)
#   LEDGER          append-only added-distributions ledger (default: /opt/added-distributions.txt)
#   INSTALL_PLUGIN  1 = install the plugin here (default 1); the image sets 0 and
#                   builds the plugin in its own layer instead
#   SNAPSHOT_DIR    directory for the runtime snapshot    (default: /opt)
#   VERIFY_ONLY=1   verify the current environment only (no installs)
#   SKIP_DEVICE_CHECK=1  skip the PPU activation check (e.g. image build without devices)
set -euo pipefail

PYTHON=${PYTHON:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}
SGLANG_SRC=${SGLANG_SRC:-/sgl-workspace/sglang-${SGLANG_VERSION}}
SGLANG_COMMIT=${SGLANG_COMMIT:-71de97b264b04dcd514cf904003028aefe9775c8}  # == tag v0.5.18
SGLANG_REPO=${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}
PLUGIN_SRC=${PLUGIN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ADDITIONS_LOCK=${ADDITIONS_LOCK:-$(dirname "${BASH_SOURCE[0]}")/empty-0.5.18.additions.lock}
LEDGER=${LEDGER:-/opt/added-distributions.txt}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-/opt}
XGRAMMAR_VERSION=0.2.1
COMPRESSED_TENSORS_VERSION=0.15.0
LOGURU_VERSION=0.7.3
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== [1/6] Snapshot runtime distributions ($PYTHON)"
"$PYTHON" - "$SNAPSHOT_DIR/runtime-distributions.before" "$SNAPSHOT_DIR/runtime-constraints.txt" \
    ${VERIFY_ONLY:+--if-absent} <<'EOF'
import importlib.metadata as m, sys
from pathlib import Path

def norm(name):
    return name.lower().replace("_", "-")

snapshot_path, constraints_path = sys.argv[1:3]
if_absent = "--if-absent" in sys.argv[3:]
if if_absent and Path(snapshot_path).exists():
    print(f"snapshot already present: {snapshot_path}")
    sys.exit(0)

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
Path(constraints_path).write_text("\n".join(lines) + "\n")
Path(snapshot_path).write_text(
    "".join(f"{inventory[k][0]}=={inventory[k][1]}\n" for k in sorted(inventory)))
print(f"snapshot: {len(inventory)} distributions -> {snapshot_path}")
for key in sorted(expected):
    print(f"  {inventory[key][0]}=={inventory[key][1]}")
EOF

for path in /opt/flagtree /opt/accl-p; do
  test -d "$path" || { echo "base environment is missing $path"; exit 1; }
done

if [ "${VERIFY_ONLY:-0}" = 1 ]; then
  echo "VERIFY_ONLY=1: skipping installs (steps 2-4)"
else
echo "=== [2/6] SGLang ${SGLANG_VERSION} (${SGLANG_COMMIT}) -> $SGLANG_SRC"
if [ ! -d "$SGLANG_SRC/.git" ]; then
  git clone --depth 1 --branch "v${SGLANG_VERSION}" "$SGLANG_REPO" "$SGLANG_SRC"
fi
git -C "$SGLANG_SRC" fetch --depth 1 origin "${SGLANG_COMMIT}"
git -C "$SGLANG_SRC" checkout --detach "${SGLANG_COMMIT}"
head_commit="$(git -C "$SGLANG_SRC" rev-parse HEAD)"
[ "$head_commit" = "${SGLANG_COMMIT}" ] \
  || { echo "SGLang HEAD ${head_commit} != pinned ${SGLANG_COMMIT}"; exit 1; }
grep -q '^srt_empty' "$SGLANG_SRC/python/pyproject_other.toml" \
  || { echo "pyproject_other.toml has no srt_empty extra"; exit 1; }

echo "=== [3/6] Install SGLang (srt_empty) under runtime constraints"
cp "$SGLANG_SRC/python/pyproject_other.toml" "$SGLANG_SRC/python/pyproject.toml"
(cd "$SGLANG_SRC/python" && \
  SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" \
  "$PYTHON" -m pip install --no-build-isolation -c "$SNAPSHOT_DIR/runtime-constraints.txt" -e ".[srt_empty]")

echo "=== [4/6] Application extras (--no-deps) and plugin (optional)"
# srt_empty = sglang[runtime_base] only.  The three extras below are a SUBSET of
# sglang's runtime_common, selected by observed startup failures on PPU hardware
# (xgrammar: probe runs 9d00440/e066a6a; compressed-tensors + loguru: probe runs
# 4752063/14e282a, TP2 e2e).  runtime_common additionally declares outlines and
# timm, which are NOT installed; the unit/functional/TP2 e2e stages on the new
# image are the proof that this minimal closure is sufficient.  They are
# installed --no-deps so their metadata (standalone Triton, public torch) cannot
# replace the runtime-owned stack.
"$PYTHON" -m pip install --no-deps \
  "xgrammar==${XGRAMMAR_VERSION}" \
  "compressed-tensors==${COMPRESSED_TENSORS_VERSION}" \
  "loguru==${LOGURU_VERSION}"
if [ "${INSTALL_PLUGIN:-1}" = 1 ]; then
  "$PYTHON" -m pip install --no-deps --no-build-isolation -e "$PLUGIN_SRC"
fi
fi  # end of the non-VERIFY_ONLY block

echo "=== [5/6] Verify zero drift, added closure, import contracts"
"$PYTHON" - "$SNAPSHOT_DIR/runtime-distributions.before" "${ADDITIONS_LOCK}" "${LEDGER}" \
    "$SGLANG_VERSION" ${VERIFY_ONLY:+--verify-only} <<'EOF'
import importlib, importlib.metadata as m, sys
from pathlib import Path

def norm(name):
    return name.lower().replace("_", "-")

snapshot_path, lock_path, ledger_path, sglang_version = sys.argv[1:5]
verify_only = "--verify-only" in sys.argv[5:]

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

added_now = {k: after[k] for k in sorted(set(after) - set(before))}
if verify_only:
    # Final-layer / CI-time check: every distribution outside the runtime
    # snapshot must be covered by the committed lock plus the append-only
    # ledger (image layers and CI setup append their additions there).
    allowed = {}
    for path in (lock_path, ledger_path):
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line and not line.startswith("#"):
                    name, version = line.split("==", 1)
                    allowed[norm(name)] = version
    unexpected = [f"{n}=={v[1]}" for n, v in sorted(added_now.items())
                  if allowed.get(n) != v[1]]
    if unexpected:
        sys.exit("distributions outside lock+ledger:\n" + "\n".join(unexpected))
    print(f"zero drift over {len(before)} runtime distributions; "
          f"all {len(added_now)} additions covered by lock+ledger")
else:
    lock = {}
    for line in Path(lock_path).read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            lock[norm(name)] = version
    only_live = [f"{n}=={added_now[n][1]}" for n in added_now if n not in lock]
    only_lock = [f"{n}=={lock[n]}" for n in lock if n not in added_now]
    mismatched = [f"{n}: live {added_now[n][1]} != lock {lock[n]}"
                  for n in added_now if n in lock and added_now[n][1] != lock[n]]
    if only_live or only_lock or mismatched:
        sys.exit("added closure does not match the additions lock:\n"
                 + "\n".join(only_live + only_lock + mismatched))
    Path(ledger_path).write_text(
        "".join(f"{added_now[n][0]}=={added_now[n][1]}\n" for n in added_now))
    print(f"zero drift over {len(before)} runtime distributions; "
          f"added closure matches the lock exactly ({len(added_now)} packages)")

# Reviewed pip-check baseline.  xgrammar's metadata declares a standalone
# `triton` distribution which we deliberately do not install: FlagTree provides
# the triton module from /opt/flagtree.  Every other violation is a failure.
import subprocess
check = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True)
allowed_violation = "xgrammar 0.2.1 requires triton, which is not installed."
violations = [l for l in (check.stdout + check.stderr).splitlines()
              if l.strip() and l.strip() != allowed_violation]
print("pip check:", "clean" if not violations else "violations found")
if violations:
    sys.exit("unexpected pip check violations:\n" + "\n".join(violations))
print("reviewed metadata exception:", allowed_violation)

# No-device import contracts: everything the image guarantees must import on a
# device-less builder, so a broken image cannot be published.  The vendor
# torchvision/torchaudio wheels are EXEMPT for now: they fail to import on a
# device-less builder (torchvision: "operator torchvision::nms does not exist";
# torchaudio: libtorchaudio.so will not load) under every tested
# PYTHONPATH/LD_LIBRARY_PATH combination.  The cause is NOT yet determined --
# that requires on-device verification; do not treat device dependence as
# proven.  The same limitation blocks compressed-tensors (its config import
# pulls transformers' PreTrainedModel chain, which imports torchvision) and any
# full sglang import.  Those packages are asserted as installed distributions
# here; the on-device probe and the unit/functional/e2e stages decide.
import torch, xgrammar, loguru
triton = importlib.import_module("triton")
assert m.version("sglang").split("+")[0] == sglang_version, m.version("sglang")
assert m.version("xgrammar") == "0.2.1"
assert m.version("compressed-tensors") == "0.15.0"
assert m.version("loguru") == "0.7.3"
assert m.version("torchvision") == "0.25.0+v0.1.0.ppu2.1.1"
assert m.version("torchaudio") == "2.10.0+v0.1.0.ppu2.1.1"
print("sglang", m.version("sglang"),
      "| torch", torch.__version__,
      "| torchvision", m.version("torchvision"), "(dist)",
      "| torchaudio", m.version("torchaudio"), "(dist)",
      "| compressed-tensors", m.version("compressed-tensors"), "(dist)",
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
