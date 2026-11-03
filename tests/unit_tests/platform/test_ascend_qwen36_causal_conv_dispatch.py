# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""One prefill wrapper: branch routing, fallback, alias and API contracts."""

import sys
from types import SimpleNamespace
from unittest import mock

import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import qwen36_causal_conv as conv


def _inputs(batch=1, initial=False, tail=0):
    dim, length = 4096, 3
    return dict(
        x=torch.ones(dim, batch * length + tail, dtype=torch.bfloat16),
        weight=torch.ones(dim, 4, dtype=torch.bfloat16),
        query_start_loc=torch.arange(0, batch * length + 1, length, dtype=torch.int32),
        cache_indices=torch.arange(batch, dtype=torch.int64),
        has_initial_state=torch.full((batch,), initial, dtype=torch.bool),
        conv_states=torch.zeros(batch + 1, dim, 3, dtype=torch.bfloat16),
        seq_lens_cpu=[length] * batch,
    )


def _native(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation="silu",
):
    batch = 1 if x.ndim == 2 else x.shape[0]
    final = torch.ones(batch, weight.shape[0], 3, dtype=x.dtype)
    if final_states_out is not None:
        final_states_out.copy_(final)
        final = final_states_out
    return x, final


def test_one_wrapper_routes_single_and_batched_with_one_validation():
    stock = mock.Mock(side_effect=AssertionError("unexpected fallback"))
    optimized = conv._make_optimized_causal_conv1d_fn(stock, _native)
    assert optimized.__wrapped__ is stock
    for batch in (1, 2, 4):
        for initial in (False, True):
            for tail in (0, 2):
                args = _inputs(batch, initial, tail)
                with (
                    mock.patch.object(
                        conv, "_get_prefill_metadata", wraps=conv._get_prefill_metadata
                    ) as validate,
                    mock.patch.object(
                        conv, "_run_single_sequence", wraps=conv._run_single_sequence
                    ) as single,
                    mock.patch.object(
                        conv,
                        "_run_equal_length_batched",
                        wraps=conv._run_equal_length_batched,
                    ) as batched,
                ):
                    out = optimized(**args)
                    validate.assert_called_once()
                    assert single.call_count == (batch == 1)
                    assert batched.call_count == (batch > 1)
                    assert torch.equal(out[:, : batch * 3], args["x"][:, : batch * 3])
                    assert not torch.count_nonzero(out[:, batch * 3 :])
                    assert torch.all(args["conv_states"][:batch] == 1)
                    assert not torch.count_nonzero(args["conv_states"][batch:])
    stock.assert_not_called()


def test_unified_wrapper_falls_back_once_and_preserves_kwargs():
    cases = []
    args = _inputs(1)
    args["cache_indices"][0] = -1
    cases.append(args)
    args = _inputs(2)
    args["seq_lens_cpu"] = [2, 4]
    cases.append(args)
    args = _inputs(2)
    args["has_initial_state"][1] = True
    cases.append(args)
    args = _inputs(2)
    args["weight"] = args["weight"].float()
    cases.append(args)
    for args in cases:
        sentinel = object()
        stock = mock.Mock(return_value=sentinel)
        optimized = conv._make_optimized_causal_conv1d_fn(stock, _native)
        assert optimized(**args, future_argument=sentinel) is sentinel
        stock.assert_called_once()
        assert stock.call_args.kwargs["future_argument"] is sentinel
        assert stock.call_args.kwargs["seq_lens_cpu"] is args["seq_lens_cpu"]


def test_single_native_api_detection_is_once_per_wrapper():
    with mock.patch.object(
        conv, "_native_api_supported", wraps=conv._native_api_supported
    ) as detect:
        optimized = conv._make_optimized_causal_conv1d_fn(mock.Mock(), _native)
        for batch in (1, 2, 1):
            optimized(**_inputs(batch))
        detect.assert_called_once_with(_native)


def test_unavailable_native_signature_preserves_batched_fast_path():
    stock = mock.Mock(return_value="fallback")
    with mock.patch.object(
        conv.inspect, "signature", side_effect=ValueError("opaque API")
    ):
        optimized = conv._make_optimized_causal_conv1d_fn(stock, _native)
    assert optimized(**_inputs(1)) == "fallback"
    assert isinstance(optimized(**_inputs(2)), torch.Tensor)
    stock.assert_called_once()


def test_metadata_native_api_batches_and_keeps_single_fallback(cpu_causal_conv):
    calls = []

    def native(
        x,
        weight,
        bias=None,
        seqlens=None,
        has_initial_state=None,
        initial_states=None,
        return_final_states=False,
        activation="silu",
    ):
        calls.append((seqlens, has_initial_state))
        assert torch.equal(seqlens, torch.full((2,), 3, dtype=torch.int32))
        assert torch.equal(
            has_initial_state, torch.full((2,), initial_states is not None)
        )
        return cpu_causal_conv(
            x, weight, bias, initial_states, return_final_states, activation=activation
        )

    stock = mock.Mock(return_value="fallback")
    optimized = conv._make_optimized_causal_conv1d_fn(stock, native)
    for initial in (False, True):
        args = _inputs(2, initial)
        result = optimized(**args)
        expected, _ = cpu_causal_conv(
            args["x"][:, :3],
            args["weight"],
            initial_states=torch.zeros(1, 4096, 3, dtype=torch.bfloat16)
            if initial
            else None,
        )
        assert torch.equal(result, torch.cat((expected, expected), dim=-1))
        assert torch.all(args["conv_states"][:2] == 1)
        assert not torch.count_nonzero(args["conv_states"][2:])
    assert len(calls) == 2
    stock.assert_not_called()
    assert optimized(**_inputs(1)) == "fallback"
    stock.assert_called_once()
    assert len(calls) == 2


def test_installer_rebinds_early_and_late_aliases_without_stacking(mock_npu_mamba):
    for backend_loaded in (False, True):
        stock = lambda *args, **kwargs: None
        kernel = SimpleNamespace(
            causal_conv1d_fn_npu=stock, causal_conv1d_fn_native=_native
        )
        backend = SimpleNamespace(causal_conv1d_fn_npu=stock, causal_conv1d_fn=stock)
        with (
            mock.patch.object(mock_npu_mamba, "causal_conv1d", kernel, create=True),
            mock.patch.dict(sys.modules),
        ):
            if backend_loaded:
                sys.modules[conv._BACKEND_MODULE] = backend
            else:
                sys.modules.pop(conv._BACKEND_MODULE, None)
            assert conv.patch_qwen36_causal_conv_prefill()
            installed = kernel.causal_conv1d_fn_npu
            assert installed.__wrapped__ is stock
            assert getattr(installed, conv._PATCH_MARKER)
            sys.modules[conv._BACKEND_MODULE] = backend
            assert conv.patch_qwen36_causal_conv_prefill()
            assert kernel.causal_conv1d_fn_npu is installed
            assert backend.causal_conv1d_fn_npu is installed
            assert backend.causal_conv1d_fn is installed
