# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FP32 communication must survive the active FlagCX hook's in-place contract."""

from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.patch import (
    _patch_flagcx_fp32_all_reduce,
)
from sglang_fl.distributed.communicator import CommunicatorFL


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.bfloat16), ("musa", torch.bfloat16), ("musa", torch.float32)],
)
def test_fp32_communicator_scope_and_inplace_result(monkeypatch, device, dtype):
    if device == "musa" and (
        not hasattr(torch, "musa") or not torch.musa.is_available()
    ):
        pytest.skip("MUSA hardware is required")

    calls = []

    def collective(self, tensor):
        calls.append((tensor, tensor.dtype))
        tensor.add_(0.125)
        return tensor

    monkeypatch.setattr(CommunicatorFL, "all_reduce", collective)
    _patch_flagcx_fp32_all_reduce()
    comm = CommunicatorFL.__new__(CommunicatorFL)
    # This is the route taken even by an eagerly imported upstream function:
    # the active GroupCoordinator hook looks up comm.all_reduce at call time.
    coordinator = SimpleNamespace(fl_communicator=comm)

    def eager_upstream_reference(value):
        return coordinator.fl_communicator.all_reduce(value)

    value = torch.tensor([1, 2, 3], device=device, dtype=dtype)
    result = eager_upstream_reference(value)
    assert result is value
    torch.testing.assert_close(
        value.cpu(), torch.tensor([1.125, 2.125, 3.125], dtype=dtype)
    )
    assert len(calls) == 1
    expected_dtype = torch.float32 if device == "musa" else dtype
    assert calls[0][1] == expected_dtype
    if device == "musa" and dtype == torch.bfloat16:
        assert calls[0][0] is not value
    else:
        assert calls[0][0] is value
