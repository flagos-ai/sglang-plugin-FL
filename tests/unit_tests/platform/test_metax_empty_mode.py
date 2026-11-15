# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tests for MetaX Empty-mode support (no sgl_kernel / flashinfer installed).

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Preserve the production import order (SGLang platform package first).
pytest.importorskip("sglang")

from sglang_fl.platform import PlatformFL  # noqa: E402
from tests.utils.metax_patch import load_metax_patch  # noqa: E402


def _make_platform_stub(vendor_name, device_type="cuda"):
    """Build a PlatformFL stub that bypasses ``__init__`` (needs real hardware)."""
    p = PlatformFL.__new__(PlatformFL)
    p._vendor_name = vendor_name
    p._device_type = device_type
    return p


class TestMetaxPlatform:
    def test_metax_is_not_cuda_but_cuda_alike(self):
        p = _make_platform_stub("metax")
        assert p.is_cuda() is False
        assert p.is_cuda_alike() is True

    def test_nvidia_is_cuda(self):
        p = _make_platform_stub("nvidia")
        assert p.is_cuda() is True
        assert p.is_cuda_alike() is True

    def test_metax_defaults_mm_attention_to_triton_attn(self):
        p = _make_platform_stub("metax")
        args = SimpleNamespace(
            mm_attention_backend=None, attention_backend=None, device="cuda"
        )
        p.apply_server_args_defaults(args)
        assert args.mm_attention_backend == "triton_attn"
        assert args.attention_backend == "triton"

    def test_metax_keeps_user_mm_attention_backend(self):
        p = _make_platform_stub("metax")
        args = SimpleNamespace(
            mm_attention_backend="sdpa", attention_backend="flashinfer", device="cuda"
        )
        p.apply_server_args_defaults(args)
        assert args.mm_attention_backend == "sdpa"
        assert args.attention_backend == "flashinfer"


class TestMetaxRegisterOps:
    @pytest.mark.parametrize("has_sgl_kernel", [True, False])
    def test_sgl_kernel_ops_follow_sgl_kernel_presence(self, has_sgl_kernel):
        from sglang_fl.dispatch.backends.vendor.metax import register_ops
        from sglang_fl.dispatch.backends.vendor.metax.metax import MetaxBackend

        registry = MagicMock()
        with (
            patch.object(MetaxBackend, "_available", True),
            patch.object(register_ops, "_has_sgl_kernel", return_value=has_sgl_kernel),
        ):
            register_ops.register_builtins(registry)
            impls = {i.op_name: i for i in registry.register_many.call_args[0][0]}
            for op in ("silu_and_mul", "rms_norm", "gemma_rms_norm", "topk"):
                assert impls[op].is_available() is has_sgl_kernel, op
            # Triton-based impls do not need sgl_kernel.
            for op in ("fused_moe", "chunk_gated_delta_rule"):
                assert impls[op].is_available() is True, op


class TestMetaxPatchSkipping:
    def test_patches_requiring_missing_packages_are_skipped(self, caplog):
        metax_patch = load_metax_patch()

        called = []
        fake = (
            ("flashinfer_utils_comm_backend", "flashinfer", "patch_x"),
            ("sampler", "sampler", "patch_y"),
            ("triton_attention", "triton_attention", "patch_z"),
        )

        def load(submodule, func):
            return lambda: called.append(submodule)

        with (
            patch.object(metax_patch, "_required_patches", return_value=fake),
            patch.object(metax_patch, "_has_module", return_value=False),
            patch.object(metax_patch, "_load_patch", side_effect=load),
        ):
            with caplog.at_level("INFO"):
                metax_patch._apply_required_patches()
        # flashinfer_utils_comm_backend and sampler need flashinfer; only the triton patch runs.
        assert called == ["triton_attention"]
        assert "flashinfer_utils_comm_backend(no flashinfer)" in caplog.text
        assert "sampler(no flashinfer)" in caplog.text

    def test_broken_patch_raises_runtime_error(self):
        metax_patch = load_metax_patch()

        def load(submodule, func):
            raise ImportError("symbol removed in this SGLang")

        with (
            patch.object(
                metax_patch,
                "_required_patches",
                return_value=(("triton_attention", "triton_attention", "patch_z"),),
            ),
            patch.object(metax_patch, "_load_patch", side_effect=load),
        ):
            # Must not surface as ImportError: the plugin loader treats that as
            # an absent vendor module and silently skips every MetaX patch.
            with pytest.raises(
                RuntimeError, match="MetaX patch failed: triton_attention"
            ):
                metax_patch._apply_required_patches()


# SGLang symbols each MetaX patch reads or replaces. These patches run in Empty
# mode, so an upstream rename breaks MetaX at runtime; checking the targets
# here catches that on any host with SGLang installed.
_PATCH_TARGETS = {
    "pynccl_wrapper": {
        "sglang.srt.distributed.device_communicators.pynccl_wrapper": (
            "NCCLLibrary.__init__",
            "NCCLLibrary.exported_functions",
            "NCCLLibrary.exported_functions_symm_mem",
            "NCCLLibrary.path_to_library_cache",
            "NCCLLibrary.path_to_dict_mapping",
        ),
    },
    "fused_moe": {
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": (
            "inplace_fused_experts",
            "outplace_fused_experts",
            "fused_experts",
        ),
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size": (
            "moe_align_block_size",
        ),
        "sglang.kernels.ops.moe.fused_moe_triton_kernels": ("moe_sum_reduce_triton",),
    },
    "prefill_attention": {
        "sglang.kernels.ops.attention.prefill_attention": ("context_attention_fwd",),
    },
    "fp32_tp_allreduce": {
        "sglang.srt.distributed.parallel_state": ("GroupCoordinator.all_reduce",),
        "sglang.srt.plugins.hook_registry": ("HookRegistry.register", "HookType.AROUND"),
    },
    "triton_attention": {
        "sglang.kernels.ops.attention.decode_attention": ("_fwd_grouped_kernel_stage1",),
        "sglang.kernels.ops.attention.extend_attention": (
            "_get_block_sizes_for_extend_attention",
            "_fwd_kernel",
            "_fwd_kernel_unified",
        ),
        "sglang.srt.layers.attention.triton_backend": (
            "TritonAttnBackend.__init__",
            "TritonAttnBackend.init_forward_metadata",
        ),
    },
    "qwen_vl_image_resize": {
        "sglang.srt.multimodal.processors.qwen_vl": (
            "QwenVLImageProcessor.load_mm_data",
        ),
    },
    "mamba_extra_buffer": {
        "sglang.srt.arg_groups.overrides": ("supports_mamba_cache_extra_buffer",),
    },
    "speculative_triton": {
        "sglang.srt.speculative.eagle_utils": (
            "sgl_build_tree_kernel_triton",
            "verify_tree_greedy_triton",
            "verify_tree_greedy_func",
        ),
        "sglang.kernels.ops.speculative.cache_locs": (
            "assign_extend_cache_locs",
            "assign_extend_cache_locs_uniform",
            "next_power_of_2",
            "assign_extend_cache_locs_func",
            "assign_extend_cache_locs_uniform_func",
        ),
    },
}

# Patches without a target entry: vendor-package patches (skipped in Empty mode)
# and the torchvision import hook, which only wraps functions when they exist.
_PATCHES_WITHOUT_TARGETS = {"torchvision_decode_jpeg_cuda_to_cpu"}


class TestMetaxPatchTargets:
    def test_every_empty_mode_patch_declares_targets(self):
        metax_patch = load_metax_patch()
        names = {name for name, _, _ in metax_patch._PATCHES}
        expected = names - set(metax_patch._PATCH_REQUIRES) - _PATCHES_WITHOUT_TARGETS
        assert set(_PATCH_TARGETS) == expected

    def test_patch_entry_points_exist(self):
        import ast
        from pathlib import Path

        metax_patch = load_metax_patch()
        patches_dir = Path(metax_patch.__file__).parent / "patches"
        for name, submodule, func in metax_patch._PATCHES:
            tree = ast.parse((patches_dir / f"{submodule}.py").read_text("utf-8"))
            defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            assert func in defs, f"{name}: {submodule}.{func} not defined"

    @pytest.mark.parametrize(
        "module_name, attrs",
        [(m, a) for targets in _PATCH_TARGETS.values() for m, a in targets.items()],
    )
    def test_sglang_targets_exist(self, module_name, attrs):
        import functools
        import importlib

        module = importlib.import_module(module_name)
        for attr in attrs:
            try:
                functools.reduce(getattr, attr.split("."), module)
            except AttributeError:
                pytest.fail(f"{module_name}.{attr} is missing")


class TestMetaxPynccl:
    def _fake_wrapper_module(self):
        def fn(name):
            return SimpleNamespace(name=name, restype=None, argtypes=[])

        class FakeNCCLLibrary:
            exported_functions = [fn("ncclGetVersion"), fn("ncclAllReduce")]
            exported_functions_symm_mem = [fn("ncclCommWindowRegister")]
            path_to_library_cache = {}
            path_to_dict_mapping = {}

            def __init__(self, so_file=None):
                raise AssertionError("upstream __init__ must be replaced")

        return SimpleNamespace(NCCLLibrary=FakeNCCLLibrary)

    def test_mccl_symbol(self):
        from sglang_fl.dispatch.backends.vendor.metax.patches import pynccl_wrapper

        assert pynccl_wrapper.mccl_symbol("ncclAllReduce") == "mcclAllReduce"
        assert pynccl_wrapper.mccl_symbol("other") == "other"

    def test_loads_libmccl_and_binds_mccl_symbols(self, monkeypatch):
        from sglang_fl.dispatch.backends.vendor.metax.patches import pynccl_wrapper

        module = self._fake_wrapper_module()
        lib = SimpleNamespace(
            mcclGetVersion=MagicMock(),
            mcclAllReduce=MagicMock(),
            mcclCommWindowRegister=MagicMock(),
        )
        cdll = MagicMock(return_value=lib)
        monkeypatch.delenv("SGLANG_NCCL_SO_PATH", raising=False)
        # Replace the module-level names only; the real importlib / ctypes stay intact.
        monkeypatch.setattr(
            pynccl_wrapper, "importlib", SimpleNamespace(import_module=lambda _: module)
        )
        monkeypatch.setattr(pynccl_wrapper, "ctypes", SimpleNamespace(CDLL=cdll))

        pynccl_wrapper.patch_pynccl_wrapper()
        pynccl_wrapper.patch_pynccl_wrapper()  # idempotent
        nccl = module.NCCLLibrary()

        cdll.assert_called_once_with("libmccl.so")
        assert nccl._funcs == {
            "ncclGetVersion": lib.mcclGetVersion,
            "ncclAllReduce": lib.mcclAllReduce,
            "ncclCommWindowRegister": lib.mcclCommWindowRegister,
        }
        # The upstream class-level function list must not be mutated.
        assert len(module.NCCLLibrary.exported_functions) == 2
