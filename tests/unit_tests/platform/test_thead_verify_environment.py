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

"""Unit tests for the pure checks of .github/scripts/thead/verify_environment.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "scripts"
    / "thead"
    / "verify_environment.py"
)


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("thead_verify_environment", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_distributions(monkeypatch, verify, rows):
    """rows: (metadata Name, version, distribution root) triples."""

    def distribution(name, version, root):
        return SimpleNamespace(
            metadata={"Name": name},
            version=version,
            locate_file=lambda path, _root=root: _root / path,
        )

    monkeypatch.setattr(
        verify.metadata,
        "distributions",
        lambda: [distribution(*row) for row in rows],
    )


class TestRuntimeInventory:
    def test_keyed_by_canonical_name(self, monkeypatch, verify):
        _install_distributions(
            monkeypatch,
            verify,
            [("sglang_fl", "0.1.0", Path("/site")), ("PyYAML", "6.0.1", Path("/site"))],
        )
        assert verify.runtime_inventory() == {"sglang-fl": "0.1.0", "pyyaml": "6.0.1"}

    def test_checkout_is_excluded(self, monkeypatch, verify):
        checkout = Path("/home/runner/work/checkout")
        _install_distributions(
            monkeypatch,
            verify,
            [("sglang_fl", "0.1.0", checkout), ("torch", "2.10.0", Path("/site"))],
        )
        assert verify.runtime_inventory(checkout) == {"torch": "2.10.0"}

    def test_nameless_entries_are_skipped(self, monkeypatch, verify):
        _install_distributions(monkeypatch, verify, [("", "1.0", Path("/site"))])
        assert verify.runtime_inventory() == {}


class TestInventoryDrift:
    def test_unchanged_inventory_has_no_drift(self, verify):
        assert verify.inventory_drift({"torch": "2.10.0"}, {"torch": "2.10.0"}) == []

    def test_added_test_tooling_is_not_drift(self, verify):
        after = {"torch": "2.10.0", "pytest": "9.2.0"}
        assert verify.inventory_drift({"torch": "2.10.0"}, after) == []

    def test_change_and_removal_are_drift(self, verify):
        before = {"torch": "2.10.0", "numpy": "1.26.4"}
        after = {"torch": "2.10.1"}
        assert verify.inventory_drift(before, after) == [
            "numpy: 1.26.4 -> <REMOVED>",
            "torch: 2.10.0 -> 2.10.1",
        ]


class TestEntryPointFailures:
    def test_exact_contract_passes(self, verify):
        groups = {
            "sglang.srt.platforms": {"sglang_fl": "sglang_fl:activate_platform"},
            "sglang.srt.plugins": {"sglang_fl": "sglang_fl:load_plugin"},
        }
        assert verify.entry_point_failures(groups) == []

    def test_value_typo_fails(self, verify):
        groups = {
            "sglang.srt.platforms": {"sglang_fl": "sglang_fl.activate_platform"},
            "sglang.srt.plugins": {"sglang_fl": "sglang_fl:load_plugin"},
        }
        failures = verify.entry_point_failures(groups)
        assert len(failures) == 1 and "activate_platform" in failures[0]

    def test_missing_group_fails(self, verify):
        groups = {"sglang.srt.platforms": {"sglang_fl": "sglang_fl:activate_platform"}}
        failures = verify.entry_point_failures(groups)
        assert len(failures) == 1 and "sglang.srt.plugins" in failures[0]
