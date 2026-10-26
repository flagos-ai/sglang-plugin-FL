# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_fl import dispatch
from sglang_fl.dispatch.backends.vendor.mthreads.musa import MusaBackend


@pytest.fixture
def musa_dispatch(monkeypatch):
    """Use the actual registry/manager with simulated MUSA availability."""
    monkeypatch.setattr(MusaBackend, "_available", True)
    manager = dispatch.OpManager()
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    with dispatch.policy_context(dispatch.SelectionPolicy()):
        yield manager
