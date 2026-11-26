# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native control-protocol selection is generic and leaves legacy PD intact."""

import concurrent.futures
import importlib.util
import os
import sys
import threading
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
        self.module(
            "sglang_fl.disaggregation.transfer_engine"
        ).FlagCXTransferEngine = TransferEngine

        class Counter:
            def __init__(counter, *args, **kwargs):
                counter.total = 0

            def labels(counter, **labels):
                return counter

            def inc(counter, amount=1):
                counter.total += amount

        self.module("prometheus_client").Counter = Counter
        self.load(
            "sglang_fl.disaggregation.peer_queues",
            "sglang_fl/disaggregation/peer_queues.py",
        )
        self.load(
            "sglang_fl.disaggregation.rolling", "sglang_fl/disaggregation/rolling.py"
        )
        self.native = self.load(
            "sglang_fl.disaggregation.native", "sglang_fl/disaggregation/native.py"
        )
        self.bridge = self.load(
            "sglang_fl.disaggregation.patch", "sglang_fl/disaggregation/patch.py"
        )

    def queue_modules(self):
        poll = self.module("sglang.srt.disaggregation.base.conn")
        poll.KVPoll = types.SimpleNamespace(Failed="failed")
        common = self.module("sglang.srt.disaggregation.common.utils")

        class Chunk:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        common.TransferKVChunk = Chunk
        self.utils.DisaggregationMode = types.SimpleNamespace(PREFILL="prefill")
        trace = self.module("sglang.srt.observability.trace")
        trace.TraceNullContext = type("TraceNullContext", (), {})
        return poll.KVPoll, Chunk, trace.TraceNullContext

    def test_balanced_transfer_requires_native_behavior_not_just_signature(self):
        poll, chunk, null_trace = self.queue_modules()

        def native(
            self,
            bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            aux_index=None,
            state_indices=None,
            num_kv_tokens=None,
            trace_ctx=None,
        ):
            assert self.disaggregation_mode == "prefill"
            assert not is_last_chunk or aux_index is not None
            if (
                bootstrap_room not in self.request_status
                or self.check_status(bootstrap_room) == poll.Failed
            ):
                return
            if bootstrap_room not in self.transfer_infos:
                return
            queue = sum(
                int(s.rsplit(":", 1)[1]) for s in self.transfer_infos[bootstrap_room]
            ) % len(self.transfer_queues)
            self.transfer_queues[queue].put(
                chunk(
                    room=bootstrap_room,
                    prefill_kv_indices=kv_indices,
                    index_slice=index_slice,
                    is_last_chunk=is_last_chunk,
                    prefill_aux_index=aux_index,
                    state_indices=state_indices,
                    num_kv_tokens=num_kv_tokens,
                    trace_ctx=null_trace() if trace_ctx is None else trace_ctx,
                )
            )

        self.native_base.MooncakeKVManager.add_transfer_request = native
        self.assertTrue(self.native._supports_balanced_transfer())
        self.assertIs(
            self.native.class_mapping(self.utils.KVClassType.MANAGER),
            self.native._BalancedFlagcxNativeKVManager,
        )

        def incompatible(
            self,
            bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            aux_index=None,
            state_indices=None,
            num_kv_tokens=None,
            trace_ctx=None,
        ):
            return None

        self.native_base.MooncakeKVManager.add_transfer_request = incompatible
        self.assertFalse(self.native._supports_balanced_transfer())
        self.assertIs(
            self.native.class_mapping(self.utils.KVClassType.MANAGER),
            self.native.FlagcxNativeKVManager,
        )

    def test_balanced_transfer_preserves_payload_and_guards(self):
        poll, _, null_trace = self.queue_modules()
        manager = self.native._BalancedFlagcxNativeKVManager()
        manager.disaggregation_mode = "prefill"
        manager.request_status = {1: "ok", 2: poll.Failed, 3: "ok"}
        manager.check_status = lambda room: manager.request_status[room]
        manager.transfer_infos = {
            1: {"peer:50001": object()},
            2: {"peer:50003": object()},
        }
        queue = []
        manager.transfer_queues = [types.SimpleNamespace(put=queue.append)]
        trace, indices, state = object(), object(), object()
        manager.add_transfer_request(1, indices, slice(2, 4), True, 9, state, 12, trace)
        self.assertEqual(len(queue), 1)
        result = queue[0]
        self.assertIs(result.prefill_kv_indices, indices)
        self.assertEqual(result.index_slice, slice(2, 4))
        self.assertEqual(
            (result.room, result.is_last_chunk, result.prefill_aux_index), (1, True, 9)
        )
        self.assertIs(result.state_indices, state)
        self.assertEqual(result.num_kv_tokens, 12)
        self.assertIs(result.trace_ctx, trace)
        manager.add_transfer_request(1, indices, slice(0, 1), False)
        self.assertIsInstance(queue[1].trace_ctx, null_trace)
        manager.add_transfer_request(2, indices, slice(0, 1), False)
        manager.add_transfer_request(3, indices, slice(0, 1), False)
        manager.add_transfer_request(99, indices, slice(0, 1), False)
        self.assertEqual(len(queue), 2)
        with self.assertRaises(AssertionError):
            manager.add_transfer_request(1, indices, slice(0, 1), True)

    def test_concurrent_first_use_keeps_peer_groups_stable(self):
        self.queue_modules()
        manager = self.native._BalancedFlagcxNativeKVManager()
        manager.disaggregation_mode = "prefill"
        manager.request_status = {n: "ok" for n in range(8)}
        manager.check_status = lambda room: "ok"
        manager.transfer_infos = {
            n: {f"peer:{50001 + 2 * n}": object()} for n in range(8)
        }
        items = [[] for _ in range(4)]
        lock = threading.Lock()

        def put(which, chunk):
            with lock:
                items[which].append(chunk.room)

        manager.transfer_queues = [
            types.SimpleNamespace(put=lambda item, n=n: put(n, item)) for n in range(4)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(
                pool.map(
                    lambda n: manager.add_transfer_request(
                        n % 8, None, slice(0, 1), False
                    ),
                    range(800),
                )
            )
        self.assertEqual(sum(map(len, items)), 800)
        self.assertEqual(sorted(map(len, items)), [200] * 4)
        for n in range(8):
            self.assertEqual(sum(n in q for q in items), 1)

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
