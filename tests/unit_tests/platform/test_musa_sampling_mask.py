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
