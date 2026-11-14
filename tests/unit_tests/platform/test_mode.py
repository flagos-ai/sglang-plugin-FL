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

import pytest

from sglang_fl.mode import (
    ADAPT_MODE,
    PLATFORM_PROFILE_MODE,
    get_sglang_fl_mode,
)


def test_default_mode_is_adapt(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_MODE", raising=False)
    assert get_sglang_fl_mode() == ADAPT_MODE


@pytest.mark.parametrize("value", ["platform_profile", "platform-profile"])
def test_platform_profile_aliases(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FL_MODE", value)
    assert get_sglang_fl_mode() == PLATFORM_PROFILE_MODE


def test_unknown_mode_fails_fast(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_MODE", "unexpected")
    with pytest.raises(ValueError, match="Unsupported SGLANG_FL_MODE"):
        get_sglang_fl_mode()
