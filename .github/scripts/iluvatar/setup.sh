#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install only this checkout and its explicit empty-runtime import addon.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
mkdir -p artifacts/iluvatar

setup_corex() {
    local minimum="${SGLANG_ILUVATAR_MIN_DEVICES:-4}"
    if ! [[ "$minimum" =~ ^[1-9][0-9]*$ ]]; then
        echo "SGLANG_ILUVATAR_MIN_DEVICES must be a positive integer" >&2
        return 2
    fi
    echo "=== Installing the checked-out Iluvatar plugin and import addon ==="
    # Preserve the image's CoreX/FlagTree/FlagGems/SGLang dependencies. The two
    # editable builds are local and must not contact a package index.
    python -m pip install --no-index --disable-pip-version-check \
        --no-deps --no-build-isolation --force-reinstall \
        -e . -e ./addon/iluvatar-compat

    # Prefer both explicit checkouts over older image copies while retaining
    # every existing vendor path needed by CoreX in this and subsequent steps.
    export PYTHONPATH="$repo_root:$repo_root/addon/iluvatar-compat/src${PYTHONPATH:+:$PYTHONPATH}"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
    fi
    python - <<'PY'
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

root = Path.cwd().resolve()
for name, directory in (
    ("sglang_fl", root / "sglang_fl"),
    ("_iluvatar_empty_compat", root / "addon/iluvatar-compat/src/_iluvatar_empty_compat"),
    ("sgl_kernel", root / "addon/iluvatar-compat/src/sgl_kernel"),
    ("flashinfer", root / "addon/iluvatar-compat/src/flashinfer"),
):
    module = import_module(name)
    actual = Path(module.__file__).resolve().parent
    assert actual == directory, f"{name}: expected checkout {directory}, imported {actual}"
    if name in ("sgl_kernel", "flashinfer"):
        assert module.__flagos_import_shim__ is True, f"Expected the explicit {name} import addon"
    print(f"{name}: {module.__file__}")
for package in ("sglang", "torch", "flagtree", "flag_gems", "sglang_fl", "sglang-iluvatar-compat"):
    print(f"{package}: {version(package)}")
PY

    echo "=== Verifying the default Iluvatar vendor profile ==="
    python scripts/iluvatar/verify_environment.py \
        --min-devices "$minimum" --fused-profile vendor \
        2>&1 | tee artifacts/iluvatar/vendor-environment.log

    echo "=== Verifying the optional FlagOS profile ==="
    # This override is limited to one child process: vendor defaults and the
    # following Actions steps must not inherit the optional FlagOS profile.
    SGLANG_FL_PER_OP='gemma_rms_norm=reference;fused_moe=flagos;chunk_gated_delta_rule=flagos;fused_recurrent_gated_delta_rule=flagos;fused_recurrent_gated_delta_rule_packed_decode=flagos' \
        python scripts/iluvatar/verify_environment.py \
        --min-devices "$minimum" --fused-profile flagos \
        2>&1 | tee artifacts/iluvatar/flagos-environment.log
    echo "=== Iluvatar installation and both profile preflights completed ==="
}

setup_corex 2>&1 | tee artifacts/iluvatar/setup.log
