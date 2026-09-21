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

"""CPU-only contract tests for the MUSA custom-AR graph slot layout.

The plugin's runtime dependencies are intentionally not imported here.  The
test loads the communicator with small import stubs and exercises only its
layout and disabled graph-registration control flow.  No GPU, process group,
JIT compiler, or distributed process is needed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "sglang_fl/dispatch/backends/vendor/mthreads/jit_custom_ar/communicator.py"
)
_MODULE_NAME = "_cpu_test_musa_jit_custom_ar_communicator"
_GRAPH_INPUT_ENV = "SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT"
_GRAPH_INPUT_ENV_ALIAS = "SGL_CUSTOM_AR_GRAPH_REGISTERED_INPUT"


class _FakeDevice:
    def __init__(self, value):
        self.value = value

    def __repr__(self):  # pragma: no cover - only used in assertion failures
        return f"FakeDevice({self.value!r})"


class _FakeTensor:
    def __init__(self, shape=(), *, dtype=None, element_size=2):
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(shape)
        self.dtype = dtype
        self._element_size = element_size

    def numel(self):
        import math
        return math.prod(self._shape)

    def element_size(self):
        return self._element_size

    def is_contiguous(self):
        return True

    @property
    def shape(self):
        return self._shape

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]


class _FakeProcessGroup:
    pass


class _ForbiddenSlotTensor:
    """Raise if a disabled graph path touches device-slot storage."""

    def __getattribute__(self, name):
        raise AssertionError(f"disabled graph path touched slot storage: {name}")


def _module(name: str, **attrs):
    value = types.ModuleType(name)
    for key, item in attrs.items():
        setattr(value, key, item)
    return value


def _load_communicator():
    """Load the target module with dependency stubs and restore sys.modules."""

    fake_torch = _module("torch")
    fake_torch.__path__ = []
    fake_torch.device = _FakeDevice
    fake_torch.Tensor = _FakeTensor
    fake_torch.int64 = object()
    fake_torch.uint8 = object()
    fake_torch.float16 = object()
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_torch.float64 = object()
    fake_torch.empty = lambda shape, **_: _FakeTensor(shape)
    fake_torch.tensor = lambda values, **_: _FakeTensor(len(values))
    fake_torch.empty_like = lambda value: _FakeTensor(value.shape)
    fake_torch.zeros_like = lambda value: _FakeTensor(value.shape)
    fake_torch.get_device_module = lambda: types.SimpleNamespace(
        is_current_stream_capturing=lambda: False,
        synchronize=lambda: None,
    )

    fake_dist = _module(
        "torch.distributed",
        ProcessGroup=_FakeProcessGroup,
        get_rank=lambda group=None: 0,
        get_world_size=lambda group=None: 2,
        is_initialized=lambda: False,
    )
    fake_torch.distributed = fake_dist

    false_flag = types.SimpleNamespace(get=lambda: False, is_set=lambda: False)
    fake_envs = _module(
        "sglang.srt.environ",
        envs=types.SimpleNamespace(
            SGLANG_USE_JIT_ALL_REDUCE=false_flag,
            SGLANG_MUSA_USE_JIT_ALL_REDUCE=false_flag,
            SGLANG_MEMORY_SAVER_CUDA_GRAPH=false_flag,
        ),
    )
    fake_ops = _module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_ops",
        IS_CUSTOM_AR_AVAILABLE=True,
    )
    fake_piecewise = _module(
        "sglang.srt.compilation.piecewise_context_manager",
        is_in_piecewise_cuda_graph=lambda: False,
    )
    fake_cuda_wrapper = _module(
        "sglang.srt.distributed.device_communicators.cuda_wrapper",
        CudaRTLibrary=type("CudaRTLibrary", (), {}),
    )
    fake_utils = _module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_utils",
        can_use_custom_all_reduce_with_nvlink=lambda **_: True,
        is_weak_contiguous=lambda _: True,
    )
    fake_srt_utils = _module(
        "sglang.srt.utils",
        get_bool_env_var=lambda _: False,
        is_cuda=lambda: False,
        is_hip=lambda: False,
        is_musa=lambda: True,
        log_info_on_rank0=lambda *args, **kwargs: None,
    )
    fake_fused = _module(
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.fused_rmsnorm",
        MusaJitCustomAllreduceRMSNorm=type(
            "MusaJitCustomAllreduceRMSNorm",
            (),
            {"__init__": lambda self, comm: None},
        ),
    )
    fake_csrc = _module(
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc"
    )
    fake_jit_ar = _module(
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc.allreduce",
        meta_size=lambda world_size: 128,
        ensure_compiled=lambda world_size: None,
    )
    fake_csrc.allreduce = fake_jit_ar

    stubs = {
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.compilation": _module("sglang.srt.compilation"),
        "sglang.srt.distributed": _module("sglang.srt.distributed"),
        "sglang.srt.distributed.device_communicators": _module(
            "sglang.srt.distributed.device_communicators"
        ),
        "sglang_fl": _module("sglang_fl"),
        "sglang_fl.dispatch": _module("sglang_fl.dispatch"),
        "sglang_fl.dispatch.backends": _module("sglang_fl.dispatch.backends"),
        "sglang_fl.dispatch.backends.vendor": _module(
            "sglang_fl.dispatch.backends.vendor"
        ),
        "sglang_fl.dispatch.backends.vendor.mthreads": _module(
            "sglang_fl.dispatch.backends.vendor.mthreads"
        ),
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar": _module(
            "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar"
        ),
        "torch": fake_torch,
        "torch.distributed": fake_dist,
        "sglang.srt.environ": fake_envs,
        "sglang.srt.distributed.device_communicators.custom_all_reduce_ops": fake_ops,
        "sglang.srt.compilation.piecewise_context_manager": fake_piecewise,
        "sglang.srt.distributed.device_communicators.cuda_wrapper": fake_cuda_wrapper,
        "sglang.srt.distributed.device_communicators.custom_all_reduce_utils": fake_utils,
        "sglang.srt.utils": fake_srt_utils,
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.fused_rmsnorm": fake_fused,
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc": fake_csrc,
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc.allreduce": fake_jit_ar,
    }
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.compilation",
        "sglang.srt.distributed",
        "sglang.srt.distributed.device_communicators",
        "sglang_fl",
        "sglang_fl.dispatch",
        "sglang_fl.dispatch.backends",
        "sglang_fl.dispatch.backends.vendor",
        "sglang_fl.dispatch.backends.vendor.mthreads",
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar",
        "sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc",
    ):
        stubs[name].__path__ = []
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        # Exercise the production guard instead of an empty RMSNorm stub.
        fused_path = _SOURCE.parent / "fused_rmsnorm.py"
        exec(compile(fused_path.read_text(), str(fused_path), "exec"), fake_fused.__dict__)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot load communicator source: {_SOURCE}")
        target = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = target
        spec.loader.exec_module(target)
        target._CPU_TEST_STUBS = stubs
        return target
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


COMMUNICATOR = _load_communicator()


def _make_communicator(
    *,
    max_size: int | None = None,
    env_value: str | None = None,
    allocation_sizes: list[int] | None = None,
    constructor=None,
):
    """Construct with fake collectives and a CPU-shaped fake tensor."""

    def create_shared_buffer(size_in_bytes, group=None):
        if allocation_sizes is not None:
            allocation_sizes.append(size_in_bytes)
        return [101, 202]

    COMMUNICATOR.CustomAllreduce.create_shared_buffer = staticmethod(create_shared_buffer)
    runtime_stubs = COMMUNICATOR._CPU_TEST_STUBS
    previous = {name: sys.modules.get(name) for name in runtime_stubs}
    try:
        sys.modules.update(runtime_stubs)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_GRAPH_INPUT_ENV, None)
            os.environ.pop(_GRAPH_INPUT_ENV_ALIAS, None)
            if env_value is not None:
                os.environ[_GRAPH_INPUT_ENV] = env_value
            kwargs = {}
            if max_size is not None:
                kwargs["max_size"] = max_size
            if constructor is None:
                constructor = COMMUNICATOR.MusaJitCustomAllreduce
            return constructor(_FakeProcessGroup(), "cpu", **kwargs)
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class TestGraphRankDataLayout(unittest.TestCase):
    def test_default_workspace_bound_and_persistent_allocations(self):
        allocations = []
        comm = _make_communicator(allocation_sizes=allocations)

        self.assertEqual(comm.max_size, 512 * 1024 * 1024)
        # world=2's metadata prefix is supplied by the test stub as 128 B;
        # both persistent allocations must use the broad class default.
        self.assertEqual(
            allocations,
            [128 + 512 * 1024 * 1024, 512 * 1024 * 1024],
        )

    def test_factory_default_is_512(self):
        env_name = COMMUNICATOR._MUSA_CUSTOM_AR_MAX_SIZE_MB_ENV
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_name, None)
            with mock.patch.object(
                COMMUNICATOR, "_use_jit_all_reduce", return_value=True
            ):
                constructor = COMMUNICATOR.dispatch_custom_allreduce()

        allocations = []
        comm = _make_communicator(
            constructor=constructor, allocation_sizes=allocations
        )
        self.assertEqual(comm.max_size, 512 * 1024 * 1024)
        self.assertEqual(
            allocations, [128 + 512 * 1024 * 1024, 512 * 1024 * 1024]
        )

    def test_factory_opt_in_128(self):
        env_name = COMMUNICATOR._MUSA_CUSTOM_AR_MAX_SIZE_MB_ENV
        with mock.patch.dict(os.environ, {env_name: "128"}, clear=False):
            with mock.patch.object(
                COMMUNICATOR, "_use_jit_all_reduce", return_value=True
            ):
                constructor = COMMUNICATOR.dispatch_custom_allreduce()

        allocations = []
        comm = _make_communicator(
            constructor=constructor, allocation_sizes=allocations
        )
        self.assertEqual(comm.max_size, 128 * 1024 * 1024)
        self.assertEqual(
            allocations, [128 + 128 * 1024 * 1024, 128 * 1024 * 1024]
        )

    def test_factory_rejects_malformed_or_out_of_range_values_before_allocating(self):
        env_name = COMMUNICATOR._MUSA_CUSTOM_AR_MAX_SIZE_MB_ENV
        invalid_values = (
            "",
            "0",
            "513",
            "128.0",
            " 128",
            "128 ",
            "-1",
            "+128",
            "１２８",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {env_name: value}, clear=False):
                    with mock.patch.object(
                        COMMUNICATOR, "_use_jit_all_reduce", return_value=True
                    ):
                        with self.assertRaisesRegex(ValueError, env_name):
                            COMMUNICATOR.dispatch_custom_allreduce()

    def test_qwen36_tp2_payload_and_workspace_boundaries(self):
        hidden = 2048
        bf16_bytes = 2
        workspace = 128 * 1024 * 1024

        def payload_bytes(rows):
            return rows * hidden * bf16_bytes

        self.assertEqual(payload_bytes(16_384), 64 * 1024 * 1024)
        self.assertEqual(payload_bytes(64), 256 * 1024)
        self.assertEqual(payload_bytes(32_768), workspace)
        self.assertGreater(payload_bytes(32_769), workspace)

        comm = _make_communicator(max_size=workspace)
        bfloat16 = COMMUNICATOR._CPU_TEST_STUBS["torch"].bfloat16
        self.assertTrue(
            comm.should_custom_ar(
                _FakeTensor((16_384, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        self.assertTrue(
            comm.should_custom_ar(
                _FakeTensor((32_768, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        self.assertFalse(
            comm.should_custom_ar(
                _FakeTensor((32_769, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        # A payload above the opt-in workspace must return to the caller's
        # fallback path without touching any device launcher.
        self.assertIsNone(
            comm.custom_all_reduce(
                _FakeTensor((32_769, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        self.assertFalse(
            comm.should_custom_ar(
                _FakeTensor((1, 1), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        self.assertFalse(
            comm.should_custom_ar(
                _FakeTensor(
                    (16_384, hidden),
                    dtype=COMMUNICATOR._CPU_TEST_STUBS["torch"].float64,
                    element_size=8,
                )
            )
        )
        self.assertTrue(
            comm._should_fused_rmsnorm_custom_ar(
                _FakeTensor((32_768, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )
        self.assertFalse(
            comm._should_fused_rmsnorm_custom_ar(
                _FakeTensor((32_769, hidden), dtype=bfloat16, element_size=bf16_bytes)
            )
        )

    def test_large_prefill_uses_nonpush_fallback_and_forced_push_is_rejected(self):
        comm = _make_communicator(max_size=128 * 1024 * 1024)

        class _Resolver:
            SHOT_PUSH = 0
            SHOT_ONE_STAGE = 1
            SHOT_TWO_STAGE = 2
            SHOT_PUSH_WIDE = 3
            SHOT_TWO_STAGE_512 = 4

            def __init__(self, forced):
                self.forced = forced

            def is_shot_forced(self):
                return self.forced

            def use_push_in_graph(self):
                return False

            def preferred_shot(self, world_size, nbytes):
                if self.forced:
                    return self.SHOT_PUSH
                return (
                    self.SHOT_PUSH
                    if nbytes <= 256 * 1024
                    else self.SHOT_ONE_STAGE
                )

            def preferred_graph_fallback_shot(self, world_size, nbytes):
                return self.SHOT_ONE_STAGE

            def preferred_fallback_shot(self, world_size, nbytes):
                return self.SHOT_ONE_STAGE

            def push_buffer_bytes(self, nbytes, world_size):
                return nbytes * 2 * world_size

        comm._jit_ar = _Resolver(forced=False)
        self.assertEqual(
            comm._preferred_shot_cached(64 * 1024 * 1024, False, False),
            _Resolver.SHOT_ONE_STAGE,
        )
        comm._shot_decision_cache.clear()
        self.assertEqual(
            comm._preferred_shot_cached(256 * 1024, False, False),
            _Resolver.SHOT_PUSH,
        )

        comm._jit_ar = _Resolver(forced=True)
        comm._shot_decision_cache.clear()
        with self.assertRaisesRegex(RuntimeError, r"staging buffer"):
            comm._preferred_shot_cached(64 * 1024 * 1024, False, False)

    def test_disabled_uses_one_slot_even_when_max_size_is_small(self):
        comm = _make_communicator(max_size=32, env_value="0")

        self.assertFalse(comm._graph_registered_input_enabled)
        self.assertEqual(comm._graph_rank_data_slot_count, 1)
        self.assertEqual(comm._graph_rank_data_bytes, 64)
        self.assertEqual(comm._graph_rank_data_slots.shape, (1, 8))

    def test_enabled_capacity_is_exact_floor_of_max_size_over_64(self):
        comm = _make_communicator(max_size=64 * 5 + 7, env_value="1")

        self.assertTrue(comm._graph_registered_input_enabled)
        self.assertEqual(comm._graph_rank_data_slot_count, 5)
        self.assertEqual(comm._graph_rank_data_bytes, 5 * 64)
        self.assertEqual(comm._graph_rank_data_slots.shape, (5, 8))

    def test_enabled_rejects_max_size_smaller_than_one_slot(self):
        with self.assertRaisesRegex(ValueError, r"rank-data slot \(64 bytes\)"):
            _make_communicator(max_size=63, env_value="1")

    def test_environment_switches_layout_without_a_max_size_override(self):
        disabled = _make_communicator(max_size=128, env_value="0")
        enabled = _make_communicator(max_size=128, env_value="1")

        self.assertEqual(disabled._graph_rank_data_slot_count, 1)
        self.assertEqual(enabled._graph_rank_data_slot_count, 2)

    def test_disabled_registration_paths_do_not_touch_slots(self):
        comm = COMMUNICATOR.MusaJitCustomAllreduce.__new__(
            COMMUNICATOR.MusaJitCustomAllreduce
        )
        comm._graph_registered_input_enabled = False
        comm._graph_rank_data_slots = _ForbiddenSlotTensor()
        comm._graph_inputs = ["input"]
        comm._graph_registered_input_sequence = ["sequence"]
        comm._graph_registered_sequence_signature = ("signature",)
        comm._graph_registered_cursor = 9
        comm._graph_registered_miss = True

        self.assertEqual(comm.register_graph_buffers(), 0)
        self.assertEqual(comm._graph_inputs, [])
        self.assertEqual(comm._graph_registered_input_sequence, [])
        self.assertEqual(comm._graph_registered_sequence_signature, ())
        self.assertEqual(comm._graph_registered_cursor, 0)

        # begin/end are no-ops for the compatibility path and must not inspect
        # the slot tensor or advance graph-registration state.
        comm._graph_registered_input_sequence = ["sequence"]
        comm._graph_registered_sequence_signature = ("signature",)
        comm._graph_registered_cursor = 4
        comm._graph_registered_miss = True
        comm.begin_graph_capture_registration()
        comm.end_graph_capture_registration()
        self.assertEqual(comm._graph_registered_input_sequence, ["sequence"])
        self.assertEqual(comm._graph_registered_sequence_signature, ("signature",))
        self.assertEqual(comm._graph_registered_cursor, 4)
        self.assertTrue(comm._graph_registered_miss)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
