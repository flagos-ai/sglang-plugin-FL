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
        from sglang_fl.dispatch.backends.vendor.metax import patch as metax_patch

        called = []
        fake = (
            ("flashinfer_backend_classes", "flashinfer_backend", "patch_x"),
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
        # flashinfer_backend_classes and sampler need flashinfer; only the triton patch runs.
        assert called == ["triton_attention"]
        assert "flashinfer_backend_classes(no flashinfer)" in caplog.text
        assert "sampler(no flashinfer)" in caplog.text

    def test_broken_patch_raises_runtime_error(self):
        from sglang_fl.dispatch.backends.vendor.metax import patch as metax_patch

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
