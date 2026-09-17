# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The MUSA FlagGems configuration preserves sampling-mask assignment."""

from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.gpu
@pytest.mark.parametrize("threshold", [-1, 63, 128])
def test_sampling_mask_assignment(threshold):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    import flag_gems

    config_path = (
        Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/config/musa.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    reference = torch.arange(128, dtype=torch.float32).reshape(4, 32)
    actual = reference.to("musa")
    reference[reference < threshold] = 0
    # Top-k sampling with no truncation produces an entirely false mask.
    # FlagGems 01433e830 broadcasts its empty nonzero indices incorrectly.
    with flag_gems.use_gems(exclude=config["flagos_blacklist"]):
        actual[actual < threshold] = 0
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("length", [0, 2])
def test_eager_boolean_buffer_slice(length):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    import flag_gems

    config_path = (
        Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/config/musa.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    buffer = torch.zeros(4, dtype=torch.bool, device="musa")
    source = torch.ones(length, dtype=torch.bool, device="musa")
    with flag_gems.use_gems(exclude=config["flagos_blacklist"]):
        # EagerRunner fills a slice of its persistent graph-buffer registry.
        # This must accept bool and update the original buffer through a view.
        buffer[:length].copy_(source)
    reference = torch.zeros(4, dtype=torch.bool)
    reference[:length] = True
    torch.testing.assert_close(buffer.cpu(), reference)
