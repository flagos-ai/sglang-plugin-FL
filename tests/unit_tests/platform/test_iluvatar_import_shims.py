# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the explicitly installed Iluvatar empty import packages.

Run each probe in an isolated interpreter so a developer's real NVIDIA
packages and this test's compatibility imports never replace one another.
"""

from pathlib import Path
import subprocess
import sys
import textwrap


_ADDON = Path(__file__).resolve().parents[3] / "addon" / "iluvatar-compat"


def _probe(code):
    prefix = f"import sys\nsys.path[:0] = {[str(_ADDON / 'src'), str(_ADDON)]!r}\n"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", prefix + textwrap.dedent(code)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reviewed_import_surface_and_every_kernel_call_fails():
    _probe(
        """
        import importlib
        from import_manifest import IMPORTS, UPSTREAM_REF

        assert UPSTREAM_REF == "v0.5.18"
        for module_name, names in IMPORTS.items():
            module = importlib.import_module(module_name)
            assert module.__flagos_import_shim__ is True
            assert set(names) <= set(module.__all__)
            for name in names:
                function = getattr(module, name)
                assert callable(function), (module_name, name)
                try:
                    function()
                except RuntimeError as error:
                    assert f"{module_name}.{name}" in str(error)
                    assert "Iluvatar empty runtime" in str(error)
                else:
                    raise AssertionError(f"{module_name}.{name} silently succeeded")
        """
    )


def test_unknown_symbols_and_standard_module_attributes_are_honest():
    _probe(
        """
        import importlib
        from import_manifest import IMPORTS

        for module_name in IMPORTS:
            module = importlib.import_module(module_name)
            assert isinstance(module.__file__, str)
            assert module.__spec__.name == module_name
            assert isinstance(module.__all__, list)
            assert not hasattr(module, "unreviewed_kernel")
            assert not hasattr(module, "__unreviewed_dunder__")
            try:
                exec(f"from {module_name} import unreviewed_kernel")
            except ImportError:
                pass
            else:
                raise AssertionError("Unknown import unexpectedly succeeded")
            if "." not in module_name:
                assert isinstance(module.__path__, list)
            else:
                assert not hasattr(module, "__path__")
        """
    )


def test_critical_cuda_alias_imports_work_before_plugin_loading():
    _probe(
        """
        import sys
        # These unconditional CUDA-alias imports occur in SGLang before the
        # general plugin lifecycle runs, including spawned scheduler workers.
        from sgl_kernel import fp8_scaled_mm, top_k_renorm_prob, top_p_renorm_prob
        from sgl_kernel import moe_sum_reduce, build_tree_kernel_efficient
        from sgl_kernel.kvcacheio import transfer_kv_per_layer, transfer_kv_all_layer
        from sgl_kernel.flash_attn import flash_attn_varlen_func
        from flashinfer import bmm_fp8
        from flashinfer.sampling import min_p_sampling_from_probs
        from flashinfer.sampling import top_k_top_p_sampling_from_probs
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
        assert "sglang_fl" not in sys.modules
        assert "torch" not in sys.modules
        assert "cuda" not in sys.modules
        assert "sgl_kernel_npu" not in sys.modules
        """
    )


def test_real_scalar_metadata_and_rejected_mla_metadata_construction():
    _probe(
        """
        from dataclasses import dataclass
        from typing import get_type_hints
        from sgl_kernel.flash_mla import FlashMLASchedMeta
        from sgl_kernel.scalar_type import ScalarType, NanRepr, scalar_types

        assert isinstance(scalar_types.uint4, ScalarType)
        assert scalar_types.uint4.size_bits == 4
        assert scalar_types.uint4b8.bias == 8
        assert scalar_types.uint8b128.bias == 128
        assert NanRepr.NONE.value == 0
        assert scalar_types.uint4 != scalar_types.uint8
        # Real, hashable metadata can appear in import-time quantization tables.
        table = {scalar_types.uint4: 4, scalar_types.uint8: 8}
        assert table[scalar_types.uint4] == 4

        @dataclass
        class AnnotationOnly:
            metadata: FlashMLASchedMeta

        assert get_type_hints(AnnotationOnly)["metadata"] is FlashMLASchedMeta
        try:
            FlashMLASchedMeta()
        except RuntimeError as error:
            assert "Iluvatar empty runtime" in str(error)
        else:
            raise AssertionError("Unavailable MLA metadata was instantiated")
        """
    )


def test_pdl_is_false_and_unavailable_features_are_not_advertised():
    _probe(
        """
        import importlib.util
        import sgl_kernel
        import flashinfer
        from sgl_kernel.utils import is_arch_support_pdl
        from sgl_kernel.version import __version__

        assert is_arch_support_pdl() is False
        assert __version__ == sgl_kernel.__version__ == "0.0.0+iluvatar.empty"
        assert sgl_kernel.__flagos_import_shim__ is True
        assert flashinfer.__flagos_import_shim__ is True
        assert importlib.util.find_spec("sgl_kernel.allreduce") is None
        assert importlib.util.find_spec("flashinfer.decode") is None
        assert not hasattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper")
        assert not hasattr(sgl_kernel, "convert_weight_packed")
        """
    )


def test_main_plugin_does_not_automatically_install_compatibility_packages():
    # This addon is deliberately a separate distribution, with no entry points
    # and no dependency on the NVIDIA libraries whose import names it supplies.
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    repo = _ADDON.parents[1]
    main = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    addon = tomllib.loads((_ADDON / "pyproject.toml").read_text(encoding="utf-8"))
    assert addon["project"]["name"] == "sglang-iluvatar-compat"
    assert addon["project"]["dependencies"] == []
    assert "entry-points" not in addon["project"]
    assert not any(
        "sglang-iluvatar-compat" in dependency
        for dependency in main["project"]["dependencies"]
    )
