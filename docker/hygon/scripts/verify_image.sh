#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Post-build acceptance for the Hygon two-wheel CI image. Nothing here
# runs during image construction; every phase is a docker run against the
# finished image, and the reference data is collected dynamically from the
# same pinned base digest that built it (no committed snapshots, nothing
# COPYed into the image - baselines live in this script's mktemp dir and
# die with it).
#
# A non-zero exit means the candidate image must NOT be pushed.
#
# Phases:
#   0. collect the Runtime reference (freeze, site-file hashes, pip-check
#      baseline, provenance) from the pinned base digest
#   1. runtime immutability: distributions exact, site files unchanged,
#      pip-check only the documented (xgrammar, triton) exception,
#      compiler slots + no standalone triton + /opt/flagtree resolution,
#      two-wheel contract, HCU compat markers   [verify_runtime_immutable]
#   2. vendor module provenance vs the base reference
#   3. kernel-wheel stub import smoke
#   4. wrapper HSACO compile through the DTK device libraries (GPU-less)
#   5. SGLang import smoke (sentinel verdict from the log)
#   6. Triton entrypoint matrix (sh, bash, child process) + no triton dist
#   7. installed-wheel resolution contract from an empty working
#      directory (parent and child process; full import deferred to HCU CI)
#   8. image-level ENV contract (the five compatibility variables)
#
# Usage: verify_image.sh <image>

set -euo pipefail

image="${1:?usage: verify_image.sh <image>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$here"
repo_root="$(cd "$here/../../.." && pwd)"
containerfile="$repo_root/docker/hygon/runtime-0.5.18.containerfile"

BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -n 1)"
[ -n "$BASE_IMAGE" ] || { echo "FATAL: no ARG BASE_IMAGE in $containerfile" >&2; exit 1; }

refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-verify.XXXXXX")"
trap 'rm -rf "$refdir"' EXIT

# -i: several phases feed their program to the container over stdin.
run() { docker run --rm -i --platform linux/amd64 "$@"; }

echo "=== [0/8] Collecting the Runtime reference from the pinned base digest ==="
run --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$BASE_IMAGE" \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$BASE_IMAGE"

echo "=== [1/8] Runtime immutability (six gates) ==="
run --volume "$refdir":/evidence:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$image" \
    /scripts/verify_runtime_immutable.py --evidence-dir /evidence

echo "=== [2/8] Vendor module provenance ==="
run --volume "$refdir":/evidence:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint python "$image" \
    /scripts/verify_module_provenance.py \
    --reference /evidence/runtime-0.5.18.provenance.json

echo "=== [3/8] Kernel-wheel stub import smoke ==="
run --entrypoint python "$image" - <<'PY'
from sgl_kernel.kvcacheio import save_kv_cache
from flashinfer.sampling import top_k_renorm_prob

import sgl_kernel
import sgl_kernel.utils
import flashinfer

assert sgl_kernel.utils.is_arch_support_pdl() is False
try:
    save_kv_cache()
except RuntimeError as e:
    assert "stub" in str(e)
print("kernel-wheel import stubs + submodule finder installed")
PY

echo "=== [4/8] Wrapper HSACO compile through the DTK device libraries ==="
run --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" <<'EOF'
/opt/dtk/aillvm/bin/clang-18 -target amdgcn-amd-amdhsa -mcpu=gfx936:xnack- -O3 \
       /scripts/hcu-device-lib-check.ll -S -o /tmp/hcu-device-lib-check.s \
&& /opt/hcu-clang/clang-18-device-libs -target amdgcn-amd-amdhsa -mcpu=gfx936:xnack- -O3 \
       /tmp/hcu-device-lib-check.s -x assembler -o /tmp/hcu-device-lib-check.hsaco \
&& test -s /tmp/hcu-device-lib-check.hsaco \
&& echo "wrapper hsaco compile through DTK device libraries: OK"
EOF

echo "=== [5/8] SGLang import smoke ==="
smoke_log="$refdir/sglang-import-smoke.log"
run --volume "$scripts":/scripts:ro \
    --entrypoint python "$image" /scripts/sglang_import_smoke.py \
    > "$smoke_log" 2>&1 || true
cat "$smoke_log"
grep -qE "SMOKE-PASS|SMOKE-TOLERATED|no ROCm-capable device is detected" "$smoke_log"

echo "=== [6/8] Triton entrypoint matrix ==="
run --entrypoint bash "$image" <<'EOF'
python -c 'import triton; p = triton.__file__; assert p and p.startswith("/opt/flagtree"), p; print("triton (sh):", p)' \
&& bash -c 'python -c "import triton; assert triton.__file__.startswith(\"/opt/flagtree\"), triton.__file__; print(\"triton (bash):\", triton.__file__)"' \
&& python -c 'import importlib.metadata as md; names=[(d.metadata["Name"] or "").lower() for d in md.distributions()]; assert "triton" not in names, "standalone triton distribution still installed"; print("standalone triton distribution absent: OK")'
EOF

echo "=== [7/8] Installed-wheel resolution contract (empty cwd, no shadowing) ==="
run --entrypoint bash "$image" <<'EOF'
mkdir -p /tmp/empty-cwd && cd /tmp/empty-cwd \
&& python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("sglang_fl")
assert spec is not None and spec.origin, "sglang_fl not resolvable"
assert "site-packages" in spec.origin, f"resolves outside site-packages: {spec.origin}"
print("parent process:", spec.origin)
PY
python - <<'PY'
import importlib.util
import subprocess
import sys

child = subprocess.run(
    [sys.executable, "-c",
     "import importlib.util; s = importlib.util.find_spec('sglang_fl');"
     "print(s.origin if s else None)"],
    capture_output=True, text=True,
)
origin = child.stdout.strip()
assert child.returncode == 0 and "site-packages" in origin, origin + child.stderr
print("child process:", origin)
PY
EOF

echo "=== [8/8] Image-level ENV contract ==="
env_json="$(docker image inspect --format '{{json .Config.Env}}' "$image")"
for var in PYTHONPATH=/opt/flagtree \
           SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
           ROCM_PATH=/opt/dtk \
           TRITON_HIP_CLANG_PATH=/opt/hcu-clang/clang-18-device-libs; do
    grep -q "\"$var\"" <<<"$env_json" || {
        echo "FATAL: image ENV lacks $var" >&2
        exit 1
    }
done
grep -q '"LD_LIBRARY_PATH=/opt/dtk' <<<"$env_json" || {
    echo "FATAL: image ENV lacks the DTK LD_LIBRARY_PATH prefix" >&2
    exit 1
}
echo "ENV contract: PYTHONPATH, LD_LIBRARY_PATH, ROCM_PATH, TRITON_HIP_CLANG_PATH, SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK present"

echo "ACCEPTED: $image passed all post-build gates (reference data was collected"
echo "dynamically from $BASE_IMAGE and disposed with $refdir)"
