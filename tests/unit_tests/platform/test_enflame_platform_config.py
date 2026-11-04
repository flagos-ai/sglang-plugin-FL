"""GCU detection and production dispatch configuration, independent of hardware."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def config_utils(monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'sglang_fl/dispatch/config/utils.py'
    spec = importlib.util.spec_from_file_location('enflame_config_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        gcu=SimpleNamespace(is_available=lambda: True),
        cuda=SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.delenv('SGLANG_FL_CONFIG', raising=False)
    monkeypatch.delenv('SGLANG_FL_PLATFORM', raising=False)
    return module


def test_gcu_detection_selects_own_config(config_utils):
    assert config_utils.get_platform_name() == 'gcu'
    path = config_utils.get_config_path()
    assert path.name == 'gcu.yaml'
    assert path.is_file()


def test_gcu_config_loads_without_other_platform_policy(config_utils):
    config = config_utils.load_platform_config('gcu')
    assert config['prefer'] == 'vendor'
    assert isinstance(config['flagos_blacklist'], list)
