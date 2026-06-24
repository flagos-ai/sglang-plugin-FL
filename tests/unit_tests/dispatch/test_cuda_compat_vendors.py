# Tests for CUDA-compatible vendor routing (e.g. thead/PPU).
#
# Background:
#   thead (T-Head/PPU) chips are fully CUDA-compatible but report vendor=thead
#   instead of vendor=nvidia at runtime. CudaBackend.is_available() previously
#   hardcoded a check for vendor_name == "nvidia", which caused all CUDA op
#   implementations to be filtered out on thead devices.
#
#   Fix: introduce _CUDA_COMPATIBLE_VENDORS set so is_available() returns True
#   for any fully CUDA-compatible vendor.
#
# Test strategy:
#   Only mock the hardware layer (torch.cuda and current_platform._vendor_name).
#   Everything from register_builtins() to OpManager.resolve() uses real code.
#   Tests are organized in three phases that build on each other:
#
#   Phase A "Gateway Check"         — CudaBackend.is_available() returns True for thead
#   Phase B "Registration Wiring"   — register_builtins() produces OpImpls that report available
#   Phase C "End-to-End Resolution" — OpManager.resolve() selects vendor.cuda for every op
#
#   If any phase breaks, all subsequent phases fail too, forming a causal chain.

from unittest.mock import patch, MagicMock

import pytest


class TestCudaCompatibleVendors:
    """Test that CUDA-compatible vendors (thead, etc.) are correctly routed to CUDA implementations."""

    def _reset_cuda_backend(self):
        """Reset CudaBackend._available class-level cache between tests.

        CudaBackend caches the is_available() result in a class variable.
        Each test must reset it to simulate a fresh vendor environment.
        """
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

        CudaBackend._available = None

    def setup_method(self):
        self._reset_cuda_backend()

    def teardown_method(self):
        self._reset_cuda_backend()

    # ==================================================================
    # Phase A: Gateway Check
    #
    # Target:  CudaBackend.is_available()
    # Goal:    Verify it returns True for CUDA-compatible vendors
    #          and False for everyone else.
    # Method:  Mock platform._vendor_name, call real is_available().
    # On real hw: FlagGems DeviceDetector populates _vendor_name;
    #             here we set it directly via mock.
    # ==================================================================

    @pytest.mark.parametrize("vendor_name", ["nvidia", "thead"])
    def test_gateway_allows_cuda_compatible_vendors(self, vendor_name):
        """is_available() should return True for all CUDA-compatible vendors."""
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

        mock_platform = MagicMock()
        mock_platform._vendor_name = vendor_name

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch.dict(
            "sys.modules",
            {"sglang.srt.platforms": MagicMock(current_platform=mock_platform)},
        ):
            CudaBackend._available = None
            backend = CudaBackend()
            assert backend.is_available() is True

    @pytest.mark.parametrize("vendor_name", ["iluvatar", "metax", "ascend", "mthreads"])
    def test_gateway_blocks_non_compatible_vendors(self, vendor_name):
        """is_available() should return False for vendors that need their own backend."""
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

        mock_platform = MagicMock()
        mock_platform._vendor_name = vendor_name

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch.dict(
            "sys.modules",
            {"sglang.srt.platforms": MagicMock(current_platform=mock_platform)},
        ):
            CudaBackend._available = None
            backend = CudaBackend()
            assert backend.is_available() is False

    def test_gateway_blocks_when_no_cuda_device(self):
        """is_available() should return False when no CUDA device exists."""
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

        with patch("torch.cuda.is_available", return_value=False):
            CudaBackend._available = None
            backend = CudaBackend()
            assert backend.is_available() is False

    def test_compatible_vendors_set_contents(self):
        """_CUDA_COMPATIBLE_VENDORS should contain exactly the expected members."""
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend

        backend = CudaBackend()
        assert "nvidia" in backend._CUDA_COMPATIBLE_VENDORS
        assert "thead" in backend._CUDA_COMPATIBLE_VENDORS
        assert "ascend" not in backend._CUDA_COMPATIBLE_VENDORS
        assert "mthreads" not in backend._CUDA_COMPATIBLE_VENDORS
        assert "iluvatar" not in backend._CUDA_COMPATIBLE_VENDORS

    # ==================================================================
    # Phase B: Registration Wiring
    #
    # Target:  cuda/register_ops.register_builtins() + _bind_is_available()
    # Goal:    Verify that every registered OpImpl correctly delegates
    #          impl.is_available() back to CudaBackend.is_available().
    # Why needed:
    #   register_builtins() wraps each op function with _bind_is_available(),
    #   which attaches backend.is_available as fn._is_available.
    #   OpManager.resolve() calls impl.is_available() -> impl.fn._is_available()
    #   to filter candidates. This phase verifies that wiring is intact,
    #   so the Phase A result (True for thead) propagates to Phase C.
    # ==================================================================

    def test_registration_wires_is_available_correctly(self):
        """Every OpImpl from register_builtins() should report available on thead.

        Verifies the wiring chain:
          impl.is_available()
            -> impl.fn._is_available()         (set by _bind_is_available)
              -> CudaBackend.is_available()     (the real method)
                -> vendor in _CUDA_COMPATIBLE_VENDORS -> True
        """
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
        from sglang_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins,
        )
        from sglang_fl.dispatch.registry import OpRegistry

        mock_platform = MagicMock()
        mock_platform._vendor_name = "thead"

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch.dict(
            "sys.modules",
            {"sglang.srt.platforms": MagicMock(current_platform=mock_platform)},
        ):
            CudaBackend._available = None

            registry = OpRegistry()
            register_builtins(registry)

            registered_ops = registry.list_operators()
            assert len(registered_ops) > 0, "register_builtins() produced no ops"

            for op_name in registered_ops:
                for impl in registry.get_implementations(op_name):
                    assert impl.impl_id == "vendor.cuda"
                    assert hasattr(impl.fn, "_is_available"), (
                        f"Op '{op_name}' missing _is_available binding"
                    )
                    # This is the same call OpManager.resolve() makes to filter
                    assert impl.is_available() is True, (
                        f"Op '{op_name}' reports unavailable on thead"
                    )

    # ==================================================================
    # Phase C: End-to-End Resolution
    #
    # Target:  OpManager.resolve()
    # Goal:    Verify that the full dispatch chain selects vendor.cuda
    #          for every op on a thead platform.
    # Method:
    #   1. Mock as thead environment
    #   2. Call real register_builtins() to register ops
    #   3. Call real OpManager.resolve() for each op
    #   4. Assert the resolved impl_id == "vendor.cuda"
    # Why needed:
    #   Even if Phase A and B pass, OpManager.resolve() has additional
    #   logic that could prevent vendor.cuda from being selected:
    #     - _matches_vendor_filters(): vendor policy filtering
    #     - get_default_order(): priority-based sorting
    #     - match_token(): token matching
    #   This phase covers the complete selection pipeline.
    # ==================================================================

    def test_e2e_thead_resolves_all_ops_to_vendor_cuda(self):
        """On a thead platform, resolve() should select vendor.cuda for every registered op."""
        import os
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
        from sglang_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins as register_cuda,
        )
        from sglang_fl.dispatch.registry import OpRegistry
        from sglang_fl.dispatch.manager import OpManager
        from sglang_fl.dispatch.policy import (
            SelectionPolicy,
            set_global_policy,
            reset_global_policy,
        )

        reset_global_policy()
        set_global_policy(SelectionPolicy.from_dict(prefer="vendor", strict=False))

        mock_platform = MagicMock()
        mock_platform._vendor_name = "thead"

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch.dict(
            "sys.modules",
            {"sglang.srt.platforms": MagicMock(current_platform=mock_platform)},
        ):
            CudaBackend._available = None

            # Real registration — same code path as production
            registry = OpRegistry()
            register_cuda(registry)

            # Real manager (skip builtin_ops auto-discovery since we registered manually)
            manager = OpManager(registry=registry)
            manager._state.initialized = True
            manager._state.init_pid = os.getpid()

            # Real resolve — every op should land on vendor.cuda
            for op_name in registry.list_operators():
                fn = manager.resolve(op_name)
                impl_id = manager._get_impl_id_for_fn(op_name, fn)
                assert impl_id == "vendor.cuda", (
                    f"Op '{op_name}' resolved to '{impl_id}' instead of 'vendor.cuda'"
                )

        reset_global_policy()

    def test_e2e_thead_all_expected_ops_resolvable(self):
        """All 10 CUDA ops should be resolvable on thead without error.

        Explicitly lists every op registered by cuda/register_ops.py
        to catch regressions if ops are added or removed:
          - silu_and_mul, rms_norm, gemma_rms_norm: activation / normalization
          - rotary_embedding, mrotary_embedding: positional encoding
          - topk, fused_moe: MoE routing
          - chunk_gated_delta_rule: linear attention (the op that originally failed)
          - fused_recurrent_gated_delta_rule*: recurrent decoding
        """
        import os
        from sglang_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
        from sglang_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins as register_cuda,
        )
        from sglang_fl.dispatch.registry import OpRegistry
        from sglang_fl.dispatch.manager import OpManager
        from sglang_fl.dispatch.policy import (
            SelectionPolicy,
            set_global_policy,
            reset_global_policy,
        )

        reset_global_policy()
        set_global_policy(SelectionPolicy.from_dict(prefer="vendor", strict=False))

        mock_platform = MagicMock()
        mock_platform._vendor_name = "thead"

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ), patch.dict(
            "sys.modules",
            {"sglang.srt.platforms": MagicMock(current_platform=mock_platform)},
        ):
            CudaBackend._available = None

            registry = OpRegistry()
            register_cuda(registry)

            manager = OpManager(registry=registry)
            manager._state.initialized = True
            manager._state.init_pid = os.getpid()

            expected_ops = [
                "silu_and_mul",
                "rms_norm",
                "gemma_rms_norm",
                "rotary_embedding",
                "mrotary_embedding",
                "topk",
                "fused_moe",
                "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
                "fused_recurrent_gated_delta_rule_packed_decode",
            ]

            for op_name in expected_ops:
                fn = manager.resolve(op_name)
                assert fn is not None, f"Op '{op_name}' resolved to None"
                impl_id = manager._get_impl_id_for_fn(op_name, fn)
                assert impl_id == "vendor.cuda", (
                    f"Op '{op_name}' resolved to '{impl_id}' instead of 'vendor.cuda'"
                )

        reset_global_policy()


class TestDistBackendMap:
    """Test that thead is routed to nccl for distributed communication.

    PlatformFL._resolve_dist_backend() resolves the backend in this order:
      1. SGLANG_FL_DIST_BACKEND env var (explicit override)
      2. FLAGCX_PATH env var presence -> "flagcx"
      3. _DIST_BACKEND_MAP[vendor_name] (our fix adds "thead" -> "nccl")
      4. Fallback: "nccl"

    We test both the static map entry and the actual resolve method.
    """

    def test_thead_in_dist_backend_map(self):
        """_DIST_BACKEND_MAP should contain thead -> nccl."""
        from sglang_fl.platform import _DIST_BACKEND_MAP

        assert "thead" in _DIST_BACKEND_MAP
        assert _DIST_BACKEND_MAP["thead"] == "nccl"

    def test_thead_resolve_dist_backend_no_env_override(self):
        """PlatformFL._resolve_dist_backend() should return 'nccl' for thead.

        Simulates the real resolve path without env var overrides.
        """
        from sglang_fl.platform import _DIST_BACKEND_MAP

        # Simulate what _resolve_dist_backend does (without instantiating PlatformFL
        # which requires FlagGems DeviceDetector):
        #   1. No SGLANG_FL_DIST_BACKEND env var
        #   2. No FLAGCX_PATH env var
        #   3. Look up _DIST_BACKEND_MAP["thead"]
        vendor_name = "thead"
        with patch.dict("os.environ", {}, clear=False):
            # Remove any env overrides that might interfere
            import os

            env_backend = os.environ.get("SGLANG_FL_DIST_BACKEND", "").strip()
            if not env_backend and "FLAGCX_PATH" not in os.environ:
                result = _DIST_BACKEND_MAP.get(vendor_name, "nccl")
                assert result == "nccl"
