"""Dependency-light tests for the MThreads Xing4 model adapter."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Xing4ModelAdapterTests(TestCase):
    def test_vendor_override_is_explicit_and_normalized(self):
        registry = _load_module(
            "xing4_native_registry", "sglang_fl/model_adapters/registry.py"
        )
        with mock.patch.dict(
            os.environ, {"SGLANG_FL_MODEL_VENDOR": " MThreads "}
        ):
            self.assertEqual(registry._detect_vendor(), "mthreads")

    def test_model_registration_is_mthreads_scoped(self):
        registry = _load_module(
            "xing4_native_registry_routing", "sglang_fl/model_adapters/registry.py"
        )
        calls = []

        class FakeModelRegistry:
            @staticmethod
            def register(package, **kwargs):
                calls.append((package, kwargs))

        modules = {
            name: ModuleType(name)
            for name in (
                "sglang",
                "sglang.srt",
                "sglang.srt.models",
                "sglang.srt.models.registry",
            )
        }
        modules["sglang.srt.models.registry"].ModelRegistry = FakeModelRegistry
        with mock.patch.dict(sys.modules, modules):
            registry._register_sglang_models("mthreads")
            registry._register_sglang_models("unknown")

        self.assertEqual(
            calls,
            [
                (
                    "sglang_fl.models.mthreads_registry",
                    {"overwrite": True, "strict": True},
                )
            ],
        )

    def test_mthreads_registration_does_not_import_ascend_hooks(self):
        registry = _load_module(
            "sglang_fl.model_adapters.registry_runtime",
            "sglang_fl/model_adapters/registry.py",
        )
        calls = []
        xingchen4 = ModuleType("sglang_fl.model_adapters.xingchen4")
        xingchen4.register_xingchen4_hooks = lambda: calls.append("xingchen4")

        registry._registered = False
        registry._detect_vendor = lambda: "mthreads"
        registry._register_transformers_configs = lambda vendor: None
        registry._register_sglang_models = lambda vendor: None
        with mock.patch.dict(
            sys.modules,
            {"sglang_fl.model_adapters.xingchen4": xingchen4},
        ):
            registry.register_model_adapters()

        self.assertEqual(calls, ["xingchen4"])

    def test_detokenizer_matching_does_not_affect_other_models(self):
        adapter = _load_module(
            "xing4_detokenizer", "sglang_fl/model_adapters/xingchen4_detokenizer.py"
        )

        class Xing4_0Tokenizer:
            __module__ = "tokenization_xing4_0"

        class OtherTokenizer:
            name_or_path = "other/model"

        self.assertTrue(adapter._is_xingchen4_tokenizer(Xing4_0Tokenizer()))
        self.assertFalse(adapter._is_xingchen4_tokenizer(OtherTokenizer()))

    def test_registry_package_exposes_separate_model_entry_points(self):
        registry = ROOT / "sglang_fl/models/mthreads_registry"
        self.assertTrue((registry / "xingchen4.py").is_file())
        self.assertTrue((registry / "xing4_0.py").is_file())
