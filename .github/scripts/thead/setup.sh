#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on T-Head PPU.
# The CI image is built from the operations Runtime and adds SGLang 0.5.18
# (srt_empty plus its application requirements). This script only reinstalls the
# plugin from the checkout under test.
# --no-deps: the plugin's dependencies are already in the image, and pip must not
# resolve against (and replace) the vendor torch / FlagTree builds.
# The runtime owns pip and setuptools; they are deliberately NOT upgraded or
# downgraded here (the previous `pip install --upgrade pip "setuptools>=68,<82"`
# line replaced runtime-owned distributions at job time).  Only `wheel`, which
# the runtime does not ship, is added.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (T-Head PPU) ==="
pip install wheel
pip install --no-deps --no-build-isolation -e .
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import importlib.metadata as m, sglang_fl; print('sglang', m.version('sglang'), '| sglang_fl loaded from', sglang_fl.__file__)"
# Re-verify the runtime-preservation contract after CI setup: every
# runtime-owned distribution unchanged, every other distribution covered by the
# image's additions ledger (+ this setup's own additions, recorded below).
python - /opt/runtime-distributions.before /opt/added-distributions.txt <<'PY'
import importlib.metadata as m, sys
from pathlib import Path

def norm(name):
    return name.lower().replace("_", "-")

before = {}
for l in Path(sys.argv[1]).read_text().splitlines():
    if l and not l.startswith("#"):
        name, version = l.split("==", 1)
        before[norm(name)] = version
ledger = Path(sys.argv[2])
known = {}
for l in ledger.read_text().splitlines():
    if l and not l.startswith("#"):
        name, version = l.split("==", 1)
        known[norm(name)] = version
after = {}
for d in m.distributions():
    name = d.metadata.get("Name")
    if name:
        after.setdefault(norm(name), (name, d.version))

drift = [f"{before[k]} -> {after.get(k, ('REMOVED',))[1]}"
         for k in before if after.get(k, (None, None))[1] != before[k]]
if drift:
    sys.exit("runtime distribution drift after CI setup:\n" + "\n".join(drift))
with ledger.open("a") as fh:
    for k, (name, version) in sorted(after.items()):
        if k in before:
            continue
        if k in known:
            if known[k] != version:
                sys.exit(f"added distribution version changed: {name} {known[k]} -> {version}")
            continue
        fh.write(f"{name}=={version}\n")
        print("ledger + (job)", name, version)
        known[k] = version
print(f"runtime distributions preserved after CI setup: {len(before)}; "
      f"total covered additions: {len(known)}")
PY
# Model store inventory: e2e cases resolve /data/models/Qwen/<model> from the
# host store bind-mounted here; log what is actually visible so a "Model not
# found" collection failure can be told apart from a wrong mount level.
echo "=== Model store inventory (/data/models/Qwen) ==="
ls -la /data/models/Qwen/ 2>&1 || true

# PROBE BRANCH ONLY (drop before folding): on-device verdict for the vendor
# vision/audio wheels.  They fail to import on a device-less builder; this
# block records the imports, the shared-library diagnosis, the libcudart
# resolution and the CUDA device probe on real PPU hardware.  Non-fatal: the
# unit/functional/e2e stages decide.
echo "=== PPU vendor stack import + linkage check (probe) ==="
python - <<'PY' || true
import os, sys
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
for mod in ("torch", "torchvision", "torchaudio", "triton",
            "compressed_tensors", "xgrammar", "sglang", "sglang_fl"):
    try:
        m = __import__(mod)
        print(f"import {mod}: OK {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"import {mod}: FAILED {e!r}")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
except Exception as e:
    print("torch.cuda probe failed:", repr(e))
PY
site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for lib in "$site"/torchvision/lib/_C*.so* \
           "$site"/torchvision/lib/libtorchvision*.so* \
           "$site"/torchaudio/lib/libtorchaudio*.so*; do
  [ -e "$lib" ] || continue
  echo "--- ldd $(basename "$lib")"
  ldd "$lib" 2>&1 | grep -E "not found" || echo "  (all NEEDED libs resolve)"
done
echo "--- libcudart.so.12 resolution:"
ldconfig -p 2>/dev/null | grep -i "libcudart" || echo "  ldconfig: no libcudart entries"
found="$(ldconfig -p 2>/dev/null | grep -i "libcudart.so.12" | head -1 | awk '{print $NF}')"
if [ -n "$found" ]; then
  echo "  resolved: $found"
  case "$found" in
    /data/*|/mnt/*) echo "  provenance: host bind mount" ;;
    *) echo "  provenance: inside the container filesystem" ;;
  esac
else
  find /opt /usr /flagos -maxdepth 4 -name 'libcudart.so.12*' 2>/dev/null | head -5 || true
  echo "  (not resolvable via ldconfig)"
fi
