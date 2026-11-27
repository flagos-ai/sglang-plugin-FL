# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Ascend sgl_kernel_npu import placeholder.

The placeholder mutates ``sys.meta_path`` process-wide, so the semantics run
in a subprocess (the same pattern the unit suite uses for vendor guards on
other platforms) while the install-decision logic is tested in-process
against a restored ``sys.meta_path``.
"""

from __future__ import annotations

import importlib.machinery
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_VENDOR_DIR = "sglang_fl/dispatch/backends/vendor/ascend"

_PROBE = """
import sys
sys.path.insert(0, {repo_root!r})

from sglang_fl.dispatch.backends.vendor.ascend import import_guard

active = import_guard.install_import_guards()
assert active == ["sgl_kernel_npu"], active
assert import_guard.install_import_guards() == ["sgl_kernel_npu"], "not idempotent"

# The names the CI matrix calls carry real implementations.
import sgl_kernel_npu.norm.l1_norm as l1_module
import sgl_kernel_npu.fla.chunk as chunk_module
import sgl_kernel_npu.fla.fused_gdn_gating as gating_module
assert callable(l1_module.l1_norm)
assert callable(chunk_module.chunk_gated_delta_rule_npu)
assert callable(gating_module.fused_gdn_gating_npu)

# Everything else imports but fails loudly when called.
from sgl_kernel_npu.attention.sinks_attention import attention_sinks_triton
try:
    attention_sinks_triton()
except RuntimeError as exc:
    assert "sglang_fl placeholder" in str(exc), exc
else:
    raise SystemExit("unknown symbol did not fail loudly")

# SGLang's try/except-ImportError fallback selection keeps working.
try:
    from sgl_kernel_npu.norm import fused_rope_qk_mqa
except ImportError:
    pass
else:
    raise SystemExit("absent symbol did not raise ImportError")

# No distribution metadata is fabricated.
from importlib import metadata
try:
    metadata.version("sgl-kernel-npu")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("placeholder fabricated distribution metadata")

# Only sgl_kernel_npu is guarded; deep_ep/torchair are not guarded roots
# (their import errors must stay loud for their own call sites).
assert set(import_guard._PACKAGES) == {{"sgl_kernel_npu"}}
print("probe-ok")
"""


def test_placeholder_semantics() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo_root=str(_REPO_ROOT))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe-ok" in result.stdout


def test_opt_out_env_disables_the_placeholder() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, {repo_root!r});\n"
                "import os\n"
                "os.environ['SGLANG_FL_ASCEND_SGL_KERNEL_NPU_STUB'] = '0'\n"
                "from sglang_fl.dispatch.backends.vendor.ascend import import_guard\n"
                "assert import_guard.install_import_guards() == []\n"
                "import importlib.util\n"
                "assert importlib.util.find_spec('sgl_kernel_npu') is None\n"
                "print('probe-ok')\n"
            ).format(repo_root=str(_REPO_ROOT)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe-ok" in result.stdout


def _stub_real_finder(root: str):
    class _RealFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == root or fullname.startswith(root + "."):
                return importlib.machinery.ModuleSpec(
                    fullname, importlib.machinery.BuiltinImporter
                )
            return None

    return _RealFinder()


def test_a_real_package_blocks_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend import import_guard

    finder = _stub_real_finder("sgl_kernel_npu")
    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    # A previous activation in this process may already have appended the
    # placeholder finder; the premise here is a process without it.
    real_meta_path = [
        f for f in sys.meta_path if not isinstance(f, import_guard._PlaceholderFinder)
    ]
    monkeypatch.setattr(sys, "meta_path", [*real_meta_path, finder])
    try:
        assert import_guard.install_import_guards() == []
    finally:
        sys.meta_path = real_meta_path


def test_fallback_table_covers_the_loaded_backend_surface() -> None:
    """Every sgl_kernel_npu name the NPU backend binds module-level is covered.

    ``ascend_gdn_backend.py``, ``linear/gdn_backend.py``, ``moe/topk.py`` and
    ``layernorm.py`` bind these names at import time on NPU; a gap here would
    bind the loud placeholder and only surface at the first model step.
    """
    from sglang_fl.dispatch.backends.vendor.ascend import sgl_kernel_npu_fallbacks

    table = sgl_kernel_npu_fallbacks.SGL_KERNEL_NPU_FALLBACKS
    expected = {
        "sgl_kernel_npu.norm.l1_norm": {"l1_norm"},
        "sgl_kernel_npu.fla.fused_gdn_gating": {"fused_gdn_gating_npu"},
        "sgl_kernel_npu.fla.chunk": {"chunk_gated_delta_rule_npu"},
        "sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent": {
            "fused_sigmoid_gating_delta_rule_update_npu"
        },
        "sgl_kernel_npu.mamba.mamba_state_update_triton": {
            "move_cache_dynamic_last_kernel_h_block",
            "move_intermediate_cache",
        },
    }
    for module_path, names in expected.items():
        assert set(table.get(module_path, {})) == names, module_path


def test_absent_symbols_cover_the_importerror_fallback_sites() -> None:
    """The guard keeps absent exactly the symbols SGLang selects via ImportError."""
    from sglang_fl.dispatch.backends.vendor.ascend import import_guard

    paths = import_guard._ABSENT_MODULE_PATHS
    assert "sgl_kernel_npu.norm.fused_rope_qk_mqa" in paths
    assert "sgl_kernel_npu.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu" in paths
