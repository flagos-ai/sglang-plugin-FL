# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Multimodal masks must use the live helper and exact integer comparisons."""

from importlib import import_module
from pathlib import Path

import pytest
import torch
import yaml

from sglang_fl.dispatch.backends.vendor.mthreads.patch import _patch_multimodal_mask


@pytest.mark.parametrize("owner", ["mm_utils", "mm_schedule"])
def test_mask_patch_follows_embedding_function_owner(monkeypatch, owner):
    mm_utils = pytest.importorskip("sglang.srt.managers.mm_utils")
    module = pytest.importorskip(f"sglang.srt.managers.{owner}")

    def embedding_entry():
        pass

    embedding_entry.__module__ = module.__name__
    monkeypatch.setattr(mm_utils, "get_embedding_and_mask", embedding_entry)
    monkeypatch.setattr(module, "_get_multimodal_mask", None, raising=False)
    _patch_multimodal_mask()
    actual = module._get_multimodal_mask(torch.tensor([1, 7, 9, 7]), torch.tensor([7]))
    torch.testing.assert_close(actual, torch.tensor([[False], [True], [False], [True]]))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_large_placeholder_equality_and_live_embedding_mask(
    monkeypatch, dtype, strided
):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    pytest.importorskip("sglang.srt.managers.mm_schedule")
    import flag_gems
    from sglang.srt.managers import mm_utils

    module = import_module(mm_utils.get_embedding_and_mask.__module__)
    _patch_multimodal_mask()
    config_path = (
        Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/config/musa.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    ids_cpu = torch.tensor([2**24, 2**24 + 1, 2**24 + 2, 2**24 + 3, 123], dtype=dtype)
    placeholders_cpu = torch.tensor([2**24 + 1, 2**24 + 2], dtype=dtype)
    ids = ids_cpu.to("musa")
    if strided:
        storage = torch.empty(10, dtype=dtype, device="musa")
        storage[::2] = ids
        ids = storage[::2]
    placeholders = placeholders_cpu.to("musa")
    embedding = torch.zeros((2, 3), device="musa")
    monkeypatch.setattr(module, "_get_precomputed_embedding", lambda *args: embedding)

    with flag_gems.use_gems(exclude=config["flagos_blacklist"]):
        # Both Tensor and Scalar overloads used with MM IDs must stay exact.
        eq_tensor = ids == placeholders[0]
        eq_scalar = ids == (2**24 + 1)
        distinct = torch.equal(ids[:1], placeholders[:1])
        _, mask, _ = mm_utils.get_embedding_and_mask(
            data_embedding_func=None,
            embedding_items=[],
            placeholder_tensor=placeholders,
            input_ids=ids,
            items_size=[0, 0],
            prefix_length=[0],
            extend_length=[5],
            items_offset_list=[[(1, 2)]],
        )
    expected_eq = torch.tensor([False, True, False, False, False])
    torch.testing.assert_close(eq_tensor.cpu(), expected_eq)
    torch.testing.assert_close(eq_scalar.cpu(), expected_eq)
    assert not distinct
    torch.testing.assert_close(
        mask.cpu(), torch.tensor([[False], [True], [True], [False], [False]])
    )
