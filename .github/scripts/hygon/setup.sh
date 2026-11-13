#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# The operations-runtime CI image ships the vendor stack, the SGLang wheel,
# and its build-time evidence pack under /opt/sglang-ci; the plugin and the
# test runner install into the ephemeral job container.
set -euo pipefail
PLUGIN_ROOT="$(pwd)"
# checkout's safe.directory uses a temporary HOME that build subprocesses do
# not inherit. Trust only this checkout for editable metadata's Git inspection.
git config --global --add safe.directory "$PLUGIN_ROOT"
EVIDENCE="${HYGON_RUNTIME_EVIDENCE:-/opt/sglang-ci}"
if [ ! -f "$EVIDENCE/runtime-distributions-before.json" ]; then
  echo "::error::CI image lacks its runtime evidence pack: $EVIDENCE" >&2
  exit 1
fi
# The operations runtime intentionally ships no test packages; install the
# pinned runner into the ephemeral job container, then verify the runner
# landed (check.sh's --check-image runs earlier and cannot see it yet).
python -m pip install pytest==9.1.1 pytest-timeout==2.4.0 \
  iniconfig==2.3.1 pluggy==1.7.0
python .github/scripts/hygon/ci_environment.py --check-ci-packages
# Never resolve dependencies: xgrammar pulls a standalone triton wheel whose
# import is shadowed by FlagTree's /opt/flagtree entry, and the runtime's
# own distributions must never move.
python .github/scripts/hygon/ci_environment.py --check-image -- \
  python -m pip install --no-deps --no-build-isolation -e "$PLUGIN_ROOT"
# Temporary fast-path validation for the appfix5 image: its broad import
# finder must not fabricate sgl_kernel.scalar_type, because SGLang already
# provides the correct ImportError fallback for this optional NVIDIA module.
# Once hardware validation passes, this exception is baked into the image and
# this job-time rewrite is removed.
python .github/scripts/hygon/enable_sglang_scalar_fallback.py
export PYTHONPATH="$PLUGIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
python .github/scripts/hygon/ci_environment.py -- python - <<'PY'
import importlib

try:
    importlib.import_module("sgl_kernel.scalar_type")
except ModuleNotFoundError as error:
    assert error.name == "sgl_kernel.scalar_type", error
else:
    raise AssertionError("sgl_kernel.scalar_type must remain absent on HCU")

from sglang.srt.layers.quantization.utils import get_scalar_types

_, scalar_types = get_scalar_types()
assert scalar_types.uint4b8 == "uint4b8", scalar_types.uint4b8
importlib.import_module(
    "sglang_fl.dispatch.backends.vendor.hygon.impl.attention_backend"
)
print("[hygon-ci] SGLang scalar fallback and Hygon attention import passed")
PY
python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --runtime-evidence "$EVIDENCE" \
    --require-gpu --min-gpus 2
# Diagnostic-only unblocker for build-infra#1786. Apply it after the runtime
# evidence check above so the verifier still proves that the image contains the
# untouched Runtime-owned FlagTree files. Remove this rewrite when the Runtime
# digest provides a working JIT path.
python .github/scripts/hygon/enable_flagtree_device_lib_fallback.py
# SGLang 0.5.18's HIP JIT builder uses the conventional ROCm prefix for both
# hipcc and headers. DTK provides the complete compatible tree at /opt/dtk but
# the Runtime has no /opt/rocm alias. Keep this as a diagnostic-only link while
# checking for failures beyond the Runtime/application prefix mismatch.
# Actions' later test step does not source the Runtime's bash profile. Persist
# the real DTK prefix so FlagTree keeps selecting its HCU clang 18 under
# /opt/dtk/aillvm instead of following the compatibility link to clang 17.
export ROCM_PATH=/opt/dtk
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'ROCM_PATH=%s\n' "$ROCM_PATH" >> "$GITHUB_ENV"
fi
if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then
  ln -s /opt/dtk /opt/rocm
fi
if [ "$(readlink -f /opt/rocm)" != "$(readlink -f /opt/dtk)" ]; then
  echo "::error::/opt/rocm exists but does not resolve to the DTK SDK" >&2
  exit 1
fi
python - <<'PY'
from pathlib import Path

from triton.backends.hcu.compiler_hcu import HIPBackend

expected = Path("/opt/dtk/aillvm/bin/clang-18")
actual = Path(HIPBackend.path_to_rocm_clang())
if actual != expected:
    raise RuntimeError(f"FlagTree selected {actual}, expected HCU compiler {expected}")
print(f"[hygon-ci] FlagTree HCU compiler: {actual}")
PY
/opt/rocm/bin/hipcc --version | head -n 1
