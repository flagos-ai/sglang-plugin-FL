#!/bin/bash
# Copyright 2026 BAAI. All rights reserved.
# Prepare the Hygon two-wheel image for CI validation.
#
# The image ships the real application wheels (sglang 0.5.18+flagos and
# sglang_fl 0.1.0); this job only installs the pinned test runner and
# verifies the installed-wheel contract. The checkout source tree must
# never shadow the installed plugin: SGLANG_FL_TEST_SOURCE=wheel makes
# tests/run.py stage every pytest child (and its engine grandchildren)
# from an isolated directory with PYTHONSAFEPATH on.
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

# Select the installed-wheel test mode for every later pytest subprocess;
# tests/run.py keeps the default checkout mode when this is unset.
export SGLANG_FL_TEST_SOURCE=wheel
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'SGLANG_FL_TEST_SOURCE=wheel\n' >> "$GITHUB_ENV"
fi

# Diagnostic-only bypass for build-infra#1787. Keep the checked-in model
# contract on strict HCU attention; only e2e construction sees this switch.
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'SGLANG_HYGON_DIAGNOSTIC_TRITON_ATTENTION=1\n' >> "$GITHUB_ENV"
fi

# Installed-wheel contract: the two application distributions exist with
# the expected versions, the entry points are registered, and sglang_fl
# resolves from the installed distribution root - never from the checkout.
PLUGIN_ROOT="$PLUGIN_ROOT" python .github/scripts/hygon/ci_environment.py -- python - <<'PY'
import importlib
import importlib.metadata as md
import os
from importlib import metadata
from pathlib import Path

import sglang_fl

assert md.version("sglang") == "0.5.18+flagos", md.version("sglang")
assert md.version("sglang-fl") == "0.1.0", md.version("sglang-fl")

expected_entry_points = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}
for group, (name, value) in expected_entry_points.items():
    registered = {(ep.name, ep.value) for ep in md.entry_points(group=group)}
    assert (name, value) in registered, (
        f"{group}: ({name!r}, {value!r}) not in {sorted(registered)}"
    )

module_file = Path(sglang_fl.__file__).resolve()
distribution_root = Path(metadata.distribution("sglang-fl").locate_file("")).resolve()
checkout_root = Path(os.environ["PLUGIN_ROOT"]).resolve()
if not module_file.is_relative_to(distribution_root):
    raise RuntimeError(
        f"sglang_fl resolved at {module_file}, which is not inside the "
        f"installed distribution root {distribution_root}"
    )
if checkout_root == module_file.parent or checkout_root in module_file.parents:
    raise RuntimeError(f"sglang_fl resolved from the checkout: {module_file}")
print(f"[hygon-ci] installed plugin: {module_file} (distribution root {distribution_root})")

importlib.import_module(
    "sglang_fl.dispatch.backends.vendor.hygon.impl.attention_backend"
)
print("[hygon-ci] Hygon attention backend importable from the installed wheel")

try:
    importlib.import_module("sgl_kernel.scalar_type")
except ModuleNotFoundError as error:
    assert error.name == "sgl_kernel.scalar_type", error
else:
    raise AssertionError("sgl_kernel.scalar_type must remain absent on HCU")

from sglang.srt.layers.quantization.utils import get_scalar_types

_, scalar_types = get_scalar_types()
assert scalar_types.uint4b8 == "uint4b8", scalar_types.uint4b8
print("[hygon-ci] SGLang scalar fallback intact")
PY

# The image bakes the DTK prefix contract (ROCM_PATH, the /opt/rocm alias
# and the FlagTree clang wrapper, see the containerfile); this job only
# verifies it.
if [ "$(readlink -f /opt/rocm)" != "$(readlink -f /opt/dtk)" ]; then
  echo "::error::/opt/rocm is missing or does not resolve to the DTK SDK" >&2
  exit 1
fi
python - <<'PY'
import os
from pathlib import Path

from triton.backends.hcu.compiler_hcu import HIPBackend

actual = Path(HIPBackend.path_to_rocm_clang())
expected = Path(os.environ["TRITON_HIP_CLANG_PATH"])
if actual != expected:
    raise RuntimeError(f"FlagTree selected {actual}, expected the baked wrapper {expected}")
resolved = Path("/opt/dtk/aillvm/bin/clang-18")
print(f"[hygon-ci] FlagTree HCU compiler: {actual} -> {resolved}")
PY
/opt/rocm/bin/hipcc --version | head -n 1

python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --plugin-source wheel \
    --runtime-evidence "$EVIDENCE" \
    --require-gpu --min-gpus 2
