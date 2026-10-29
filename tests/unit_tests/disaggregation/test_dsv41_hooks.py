# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model hooks register once and leave the actual transport/configuration intact."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[3]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelHookTests(unittest.TestCase):
    def setUp(self):
        self.module = load(
            "isolated_dsv41_hooks.patch", "sglang_fl/dsv41_runtime/patch.py"
        )

    def test_entry_point_installs_hooks_without_environment_configuration(self):
        entry = load("isolated_dsv41_entry", "sglang_fl/dsv41_runtime/__init__.py")
        hooks = types.ModuleType("isolated_dsv41_entry.patch")
        hooks.apply_dsv41_patches = unittest.mock.Mock()
        with patch.dict(sys.modules, {hooks.__name__: hooks}):
            entry.load_plugin()
        hooks.apply_dsv41_patches.assert_called_once_with()

    def test_registration_uses_repository_hook_registry_once(self):
        registry = types.SimpleNamespace(register=unittest.mock.Mock())
        kinds = types.SimpleNamespace(AROUND="around", REPLACE="replace")
        native = types.ModuleType("sglang_fl.disaggregation.patch")
        native.apply_disaggregation_patch = unittest.mock.Mock()
        prefill = types.ModuleType("isolated_dsv41_hooks.prefill")
        for name in ("candidate_block_ids", "candidate_blocks", "extend_indexer"):
            setattr(prefill, name, lambda *args: None)
        indexer = types.ModuleType("isolated_dsv41_hooks.indexer")
        indexer.producer = lambda *args: None
        indexer.compact = lambda *args: None
        layout = types.ModuleType("isolated_dsv41_hooks.pd")
        layout.install_native_layout_hooks = unittest.mock.Mock()
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.plugins.hook_registry": types.SimpleNamespace(
                    HookRegistry=registry, HookType=kinds
                ),
                "sglang_fl.disaggregation.patch": native,
                "isolated_dsv41_hooks.prefill": prefill,
                "isolated_dsv41_hooks.indexer": indexer,
                "isolated_dsv41_hooks.pd": layout,
            },
        ):
            self.module.apply_dsv41_patches()
            self.module.apply_dsv41_patches()
        self.assertEqual(registry.register.call_count, 7)
        self.assertEqual(registry.register.call_args_list[0].args[2], kinds.AROUND)
        native.apply_disaggregation_patch.assert_called_once_with()
        targets = [call.args[0] for call in registry.register.call_args_list]
        self.assertTrue(all(target.startswith("sglang.srt.") for target in targets))

    def test_indexer_respects_oot_disable_and_filters(self):
        registry = types.SimpleNamespace(register=unittest.mock.Mock())
        kinds = types.SimpleNamespace(AROUND="around")
        hook = lambda *args: None
        config = {"oot_whitelist": [], "oot_blacklist": []}
        with patch("sglang_fl._build_config", return_value=config):
            with patch.dict("os.environ", {"SGLANG_FL_OOT_ENABLED": "0"}):
                self.module._register_indexer_hooks(registry, kinds, hook, hook)
            registry.register.assert_not_called()
            with patch.dict("os.environ", {"SGLANG_FL_OOT_ENABLED": "1"}):
                config["oot_whitelist"] = ["dsv41_indexer_compact"]
                self.module._register_indexer_hooks(registry, kinds, hook, hook)
        self.assertEqual(registry.register.call_count, 1)
        self.assertTrue(
            registry.register.call_args.args[0].endswith("candidate_blocks")
        )

    def test_draft_stats_are_isolated_and_target_stats_restore(self):
        from contextlib import contextmanager

        class Recorder:
            disabled = False

            @contextmanager
            def disable_this_region(recorder):
                previous = recorder.disabled
                recorder.disabled = True
                try:
                    yield
                finally:
                    recorder.disabled = previous

        recorder = Recorder()
        stats = types.ModuleType("sglang.srt.eplb.expert_distribution")
        stats.get_global_expert_distribution_recorder = lambda: recorder
        sentinel = object()

        def native(value):
            self.assertTrue(recorder.disabled)
            self.assertIs(value, sentinel)
            return value

        with patch.dict(sys.modules, {stats.__name__: stats}):
            self.assertIs(self.module._draft_forward(native, sentinel), sentinel)
            self.assertFalse(recorder.disabled)

            def fail():
                raise ValueError("native failure")

            with self.assertRaises(ValueError):
                self.module._draft_forward(fail)
            self.assertFalse(recorder.disabled)


if __name__ == "__main__":
    unittest.main()
