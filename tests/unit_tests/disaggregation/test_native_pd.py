# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native control-protocol selection is generic and leaves legacy PD intact."""

import importlib.util
import os
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[3]


class NativePDTests(unittest.TestCase):
    def module(self, name):
        if name in self.modules:
            return self.modules[name]
        module = types.ModuleType(name)
        module.__path__ = []
        self.modules[name] = module
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(self.module(parent), child, module)
        return module

    def load(self, name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(self.module(parent), child, module)
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.modules = {}
        self.module_patch = patch.dict(sys.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        uvicorn = self.module("uvicorn._subprocess")
        uvicorn.subprocess_started = lambda **kwargs: kwargs
        utils = self.module("sglang.srt.disaggregation.utils")
        utils.TransferBackend = Enum("TransferBackend", {"MOONCAKE": "mooncake"})
        utils.KVClassType = Enum(
            "KVClassType", "KVARGS MANAGER SENDER RECEIVER BOOTSTRAP_SERVER"
        )
        self.original = lambda backend, kind: ("original", backend, kind)
        utils.get_kv_class = self.original
        self.utils = utils
        self.module("sglang.srt.disaggregation.prefill").get_kv_class = self.original
        server_args = self.module("sglang.srt.server_args")
        server_args.DISAGG_TRANSFER_BACKEND_CHOICES = ["mooncake"]
        server_args.add_disagg_transfer_backend_choices = (
            server_args.DISAGG_TRANSFER_BACKEND_CHOICES.extend
        )
        base = self.module("sglang.srt.disaggregation.base")
        base.KVArgs = type("KVArgs", (), {})
        self.module("sglang.srt.utils").is_npu = lambda: False
        self.module("sglang.srt.utils.network").get_local_ip_auto = lambda: "127.0.0.1"

        class NativeManager:
            def _registerable_regions(manager):
                return [(11, 8), (22, 16), (33, 32)]

        native = self.module("sglang.srt.disaggregation.mooncake.conn")
        native.MooncakeKVManager = NativeManager
        for role in ("Sender", "Receiver", "BootstrapServer"):
            setattr(native, "MooncakeKV" + role, type(role, (), {}))
        legacy = self.module("sglang_fl.disaggregation")
        for role in ("Manager", "Sender", "Receiver", "BootstrapServer"):
            setattr(legacy, "FlagcxKV" + role, type("Legacy" + role, (), {}))
        self.legacy = legacy
        self.native_base = native

        class TransferEngine:
            def __init__(engine, hostname, gpu_id, ib_device):
                engine.hostname = hostname
                engine.gpu_id = gpu_id
                engine.ib_device = ib_device
                engine.calls = []
                engine.result = 0

            def batch_register(engine, ptrs, sizes):
                engine.calls.append(("device", ptrs, sizes))
                return engine.result

            def batch_register_host(engine, ptrs, sizes):
                engine.calls.append(("host", ptrs, sizes))
                return engine.result

            def batch_transfer_sync(engine, *args):
                return engine.result

        self.transfer_engine = TransferEngine
        self.module("sglang_fl.disaggregation.transfer_engine").FlagCXTransferEngine = (
            TransferEngine
        )

        class Counter:
            def __init__(counter, *args, **kwargs):
                counter.total = 0

            def labels(counter, **labels):
                return counter

            def inc(counter, amount=1):
                counter.total += amount

        self.module("prometheus_client").Counter = Counter
        self.load(
            "sglang_fl.disaggregation.rolling", "sglang_fl/disaggregation/rolling.py"
        )
        self.native = self.load(
            "sglang_fl.disaggregation.native", "sglang_fl/disaggregation/native.py"
        )
        self.bridge = self.load(
            "sglang_fl.disaggregation.patch", "sglang_fl/disaggregation/patch.py"
        )

    def test_spawn_entry_registers_before_native_http_worker(self):
        events = []
        self.bridge._http_worker_original = lambda **kw: events.append("native") or kw
        cfg = types.SimpleNamespace(app="sglang.srt.entrypoints.http_server:app")
        with patch.object(
            self.bridge,
            "apply_disaggregation_patch",
            side_effect=lambda: events.append("register"),
        ):
            result = self.bridge._flagcx_http_worker_started(
                cfg, "target", ["socket"], 4
            )
        self.assertEqual(events, ["register", "native"])
        self.assertEqual(result["stdin_fileno"], 4)

    def test_spawn_entry_leaves_other_uvicorn_apps_unchanged(self):
        self.bridge._http_worker_original = lambda **kw: "native"
        with patch.object(self.bridge, "apply_disaggregation_patch") as apply:
            result = self.bridge._flagcx_http_worker_started(
                types.SimpleNamespace(app="other:app"), None, []
            )
        self.assertEqual(result, "native")
        apply.assert_not_called()

    def test_worker_bootstrap_install_is_idempotent(self):
        subprocess = self.modules["uvicorn._subprocess"]
        original = subprocess.subprocess_started
        self.bridge._patch_http_worker_bootstrap()
        self.bridge._patch_http_worker_bootstrap()
        self.assertIs(self.bridge._http_worker_original, original)
        self.assertIs(
            subprocess.subprocess_started, self.bridge._flagcx_http_worker_started
        )

    def manager(self, role="decode"):
        manager = self.native.FlagcxNativeKVManager()
        manager.disaggregation_mode = types.SimpleNamespace(value=role)
        manager.kv_args = types.SimpleNamespace(
            gpu_id=2, engine_rank=0, aux_data_ptrs=[22]
        )
        manager.server_args = types.SimpleNamespace(
            served_model_name="another-model", disaggregation_ib_device="mlx5_0"
        )
        return manager

    def test_old_manager_api_uses_existing_generic_backend(self):
        del self.native_base.MooncakeKVManager._registerable_regions
        self.bridge.apply_disaggregation_patch()
        cls = self.utils.get_kv_class(
            self.utils.TransferBackend.FLAGCX, self.utils.KVClassType.MANAGER
        )
        self.assertIs(cls, self.legacy.FlagcxKVManager)

    def test_native_route_works_without_model_flags_and_preserves_aliases(self):
        self.bridge.apply_disaggregation_patch()
        wrapper = self.utils.get_kv_class
        self.assertIs(
            wrapper(self.utils.TransferBackend.FLAGCX, self.utils.KVClassType.MANAGER),
            self.native.FlagcxNativeKVManager,
        )
        self.assertIs(
            wrapper(self.utils.TransferBackend.FLAGCX, self.utils.KVClassType.SENDER),
            self.native_base.MooncakeKVSender,
        )
        alias = self.modules["sglang.srt.disaggregation.prefill"].get_kv_class
        self.assertIs(alias, wrapper)
        self.assertEqual(
            wrapper(
                self.utils.TransferBackend.MOONCAKE, self.utils.KVClassType.MANAGER
            )[0],
            "original",
        )
        self.bridge.apply_disaggregation_patch()
        self.assertIs(self.utils.get_kv_class, wrapper)

    def test_native_manager_reuses_shared_transfer_engine(self):
        self.assertTrue(
            issubclass(self.native._MeteredFlagCXEngine, self.transfer_engine)
        )
        manager = self.manager()
        manager.init_engine()
        self.assertIsNone(manager.engine.legacy)
        self.assertEqual(manager.engine.flagcx.gpu_id, 2)
        self.assertEqual(manager.engine.flagcx.hostname, "127.0.0.1")

    def test_cpu_auxiliary_buffers_use_host_registration(self):
        manager = self.manager()
        manager.init_engine()
        manager.register_buffer_to_engine()
        self.assertEqual(
            manager.engine.flagcx.calls,
            [("device", [11, 33], [8, 32]), ("host", [22], [16])],
        )

    def test_npu_auxiliary_buffers_use_device_registration(self):
        self.native.is_npu = lambda: True
        manager = self.manager()
        manager.init_engine()
        manager.register_buffer_to_engine()
        self.assertEqual(
            manager.engine.flagcx.calls, [("device", [11, 22, 33], [8, 16, 32])]
        )

    def test_registration_failure_is_not_ignored(self):
        manager = self.manager()
        manager.init_engine()
        manager.engine.flagcx.result = -1
        with self.assertRaises(RuntimeError):
            manager.register_buffer_to_engine()

    def test_shared_engine_close_is_idempotent(self):
        self.module("sglang.srt.platforms").current_platform = object()
        self.module("sglang.srt.utils.network").NetworkAddress = object
        module = self.load(
            "isolated_transfer_engine", "sglang_fl/disaggregation/transfer_engine.py"
        )
        engine = module.FlagCXTransferEngine.__new__(module.FlagCXTransferEngine)
        engine.flagcx = types.SimpleNamespace(
            flagcxP2pEngineDestroy=unittest.mock.Mock()
        )
        engine.engine = object()
        engine._closed = False
        engine.close()
        engine.close()
        engine.flagcx.flagcxP2pEngineDestroy.assert_called_once_with(engine.engine)


if __name__ == "__main__":
    unittest.main()
