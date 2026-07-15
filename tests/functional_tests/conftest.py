# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for SGLang-FL functional tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Return a CUDA device for functional operator tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")
