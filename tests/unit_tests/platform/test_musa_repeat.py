# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33, 256])
def test_native_repeat_for_decode_mrope_positions(tokens):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    import flag_gems

    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/config/musa.yaml"
        ).read_text()
    )
    cpu_positions = torch.arange(tokens * 2, dtype=torch.int64)[::2]
    storage = torch.arange(tokens * 2, dtype=torch.int64, device="musa")
    positions = storage[::2]
    with flag_gems.use_gems(exclude=config["flagos_blacklist"]):
        actual = positions.unsqueeze(0).repeat(3, 1)
    torch.testing.assert_close(actual.cpu(), cpu_positions.unsqueeze(0).repeat(3, 1))
