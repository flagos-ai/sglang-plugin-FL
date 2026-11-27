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

import logging
import sys
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    profiler as musa_profiler,
)
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    profiler_lifecycle as lifecycle,
)


class _FakeFunction:
    def __init__(self, name, calls, result=0, callback=None):
        self.name = name
        self.calls = calls
        self.result = result
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(self.name)
        if self.callback is not None:
            return self.callback(*args)
        return self.result


class _FakeMusart:
    def __init__(self, result=801, cleared_result=None, include_stop=True):
        self.calls = []
        cleared_result = result if cleared_result is None else cleared_result
        self.musaProfilerStart = _FakeFunction(
            "musaProfilerStart", self.calls, result=result
        )
        if include_stop:
            self.musaProfilerStop = _FakeFunction(
                "musaProfilerStop", self.calls, result=result
            )
        self.musaGetLastError = _FakeFunction(
            "musaGetLastError", self.calls, result=cleared_result
        )
        self.musaRuntimeGetVersion = _FakeFunction(
            "musaRuntimeGetVersion",
            self.calls,
            callback=self._set_runtime_version,
        )
        self.musaGetErrorString = _FakeFunction(
            "musaGetErrorString",
            self.calls,
            callback=lambda _result: b"fake MUSA error",
        )

    @staticmethod
    def _set_runtime_version(version_pointer):
        version_pointer._obj.value = 40305
        return 0


def test_musa_profiler_api_special_cases_801(monkeypatch, caplog):
    library = _FakeMusart()
    monkeypatch.setattr(musa_profiler.ctypes, "CDLL", lambda _name: library)

    controller = musa_profiler.MusaProfilerApi()
    with caplog.at_level(
        logging.WARNING,
        logger="sglang_fl.dispatch.backends.vendor.mthreads.patches.profiler",
    ):
        assert controller.start() == 0
        assert controller.stop() == 0

    assert library.calls.count("musaGetLastError") == 2
    assert caplog.text.count("returned MUSA error 801") == 2
    assert "runtime version 40305" in caplog.text


def test_musa_profiler_api_raises_other_errors_after_clearing(monkeypatch):
    library = _FakeMusart(result=700, cleared_result=700)
    monkeypatch.setattr(musa_profiler.ctypes, "CDLL", lambda _name: library)

    with pytest.raises(musa_profiler.MusaProfilerError, match="MUSA error 700"):
        musa_profiler.MusaProfilerApi().start()

    assert library.calls.index("musaGetLastError") == (
        library.calls.index("musaProfilerStart") + 1
    )


def test_musa_profiler_api_rejects_missing_runtime_symbols(monkeypatch):
    library = _FakeMusart(include_stop=False)
    monkeypatch.setattr(musa_profiler.ctypes, "CDLL", lambda _name: library)

    with pytest.raises(
        musa_profiler.MusaProfilerError,
        match="missing required profiler symbols: musaProfilerStop",
    ):
        musa_profiler.MusaProfilerApi().validate()


def test_cudart_proxy_redirects_profiler_and_delegates_other_symbols():
    events = []

    class FakeApi:
        def start(self):
            events.append("musa_start")
            return 0

        def stop(self):
            events.append("musa_stop")
            return 0

    delegate = SimpleNamespace(cudaHostRegister="original_host_register")
    proxy = musa_profiler.MusaCudartProxy(lambda: delegate, FakeApi())

    assert proxy.cudaProfilerStart() == 0
    assert proxy.cudaProfilerStop() == 0
    assert proxy.cudaHostRegister == "original_host_register"
    assert events == ["musa_start", "musa_stop"]


def test_torch_redirects_reuse_sglang_gpu_and_cuda_profiler_paths(monkeypatch):
    original_cudart = lambda: SimpleNamespace(cudaHostRegister="host_register")
    fake_activity = SimpleNamespace(CUDA="cuda", PrivateUse1="privateuse1")
    fake_torch = SimpleNamespace(
        profiler=SimpleNamespace(
            ProfilerActivity=fake_activity, profile=lambda **kw: kw
        ),
        cuda=SimpleNamespace(cudart=original_cudart),
    )
    monkeypatch.setattr(musa_profiler, "torch", fake_torch)

    lifecycle.install_patches(musa_profiler._prepare_torch_profiler_redirects())

    assert fake_activity.CUDA == "cuda"
    assert fake_torch.profiler.profile(activities=["cuda"])["activities"] == [
        "privateuse1"
    ]
    assert fake_torch.profiler.profile(activities=["cpu", "cuda"])["activities"] == [
        "cpu",
        "privateuse1",
    ]
    assert fake_torch.profiler.profile(activities=["cpu"])["activities"] == ["cpu"]
    assert fake_torch.cuda.cudart().cudaHostRegister == "host_register"
    assert isinstance(fake_torch.cuda.cudart(), musa_profiler.MusaCudartProxy)


def test_patch_install_keeps_musart_loading_lazy(monkeypatch):
    library_loads = []
    monkeypatch.setattr(musa_profiler, "_patches_applied", False)
    monkeypatch.setattr(
        musa_profiler,
        "prepare_profiler_lifecycle_patch",
        lambda _error_type: [],
    )
    monkeypatch.setattr(musa_profiler, "_prepare_torch_profiler_redirects", lambda: [])
    monkeypatch.setattr(
        musa_profiler.ctypes,
        "CDLL",
        lambda name: library_loads.append(name),
    )

    musa_profiler.apply_musa_profiler_patches()

    assert library_loads == []


def test_legacy_marker_start_failure_uses_sglang_stop_for_rollback():
    events = []

    class MarkerError(RuntimeError):
        pass

    class FakeSchedulerProfilerMixin:
        def start_profile(self, stage=None):
            events.extend(["rpd_start", "mem_start", "marker_start"])
            self.profile_in_progress = True
            raise MarkerError("marker failed")

        def stop_profile(self, stage=None):
            events.extend(["rpd_stop", "mem_stop", "marker_stop"])
            self.profile_in_progress = False
            self.torch_profiler = None

    lifecycle._wrap_legacy_profiler_lifecycle(FakeSchedulerProfilerMixin, MarkerError)
    scheduler = FakeSchedulerProfilerMixin()
    scheduler.torch_profiler = object()
    scheduler.profile_in_progress = False
    scheduler.profiler_start_forward_ct = 10

    with pytest.raises(MarkerError, match="marker failed"):
        scheduler.start_profile()

    assert events == [
        "rpd_start",
        "mem_start",
        "marker_start",
        "rpd_stop",
        "mem_stop",
        "marker_stop",
    ]
    assert scheduler.torch_profiler is None
    assert scheduler.profile_in_progress is False
    assert scheduler.profiler_start_forward_ct is None


def test_profile_v2_list_rolls_back_started_profilers_in_reverse_order():
    events = []

    class FakeProfiler:
        def __init__(self, name, fail_start=False):
            self.name = name
            self.fail_start = fail_start

        def start(self):
            events.append(f"{self.name}_start")
            if self.fail_start:
                raise RuntimeError(f"{self.name} failed")

        def stop(self):
            events.append(f"{self.name}_stop")

    class FakeProfilerList:
        pass

    lifecycle._wrap_profiler_list_lifecycle(FakeProfilerList)
    profiler_list = FakeProfilerList()
    profiler_list.inners = [
        FakeProfiler("torch"),
        FakeProfiler("mem"),
        FakeProfiler("marker", fail_start=True),
    ]

    with pytest.raises(RuntimeError, match="marker failed"):
        profiler_list.start()

    assert events == [
        "torch_start",
        "mem_start",
        "marker_start",
        "mem_stop",
        "torch_stop",
    ]


def test_profile_v2_list_attempts_every_stop_before_raising():
    events = []

    class FakeProfiler:
        def __init__(self, name, fail_stop=False):
            self.name = name
            self.fail_stop = fail_stop

        def stop(self):
            events.append(f"{self.name}_stop")
            if self.fail_stop:
                raise RuntimeError(f"{self.name} failed")

    class FakeProfilerList:
        pass

    lifecycle._wrap_profiler_list_lifecycle(FakeProfilerList)
    profiler_list = FakeProfilerList()
    profiler_list.inners = [
        FakeProfiler("torch"),
        FakeProfiler("marker", fail_stop=True),
        FakeProfiler("rpd"),
    ]

    with pytest.raises(RuntimeError, match="marker failed"):
        profiler_list.stop()

    assert events == ["torch_stop", "marker_stop", "rpd_stop"]


@pytest.fixture
def profiler_contract(monkeypatch):
    class SchedulerProfilerMixin:
        def profile(self, recv_req):
            return "configured"

        def start_profile(self, stage=None):
            pass

        def stop_profile(self, stage=None):
            pass

    class ProfilerList:
        def __init__(self, inners):
            self.inners = inners

        def start(self):
            pass

        def stop(self):
            pass

    scheduler_module = ModuleType("sglang.srt.managers.scheduler_profiler_mixin")
    scheduler_module.SchedulerProfilerMixin = SchedulerProfilerMixin
    scheduler_module.ProfileReqOutput = SimpleNamespace
    scheduler_module.ProfileReqType = SimpleNamespace(START_PROFILE="start")
    list_module = ModuleType("sglang.srt.utils.profile_utils")
    list_module._ProfilerList = ProfilerList
    list_module._ProfilerTorch = type("FakeTorchProfiler", (), {})
    list_module._ProfilerRPD = type("FakeRPD", (), {})
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, list_module.__name__, list_module)
    monkeypatch.setattr(musa_profiler, "_patches_applied", False)
    return SchedulerProfilerMixin, ProfilerList


@pytest.mark.parametrize("version", ["release-build", "vendor-build"])
def test_contract_install_does_not_gate_redirects_by_version(
    monkeypatch, profiler_contract, version
):
    # These are API doubles, not validation of another SGLang distribution.
    monkeypatch.setattr(metadata, "version", lambda _name: version)
    fake_activity = SimpleNamespace(CUDA="cuda", PrivateUse1="privateuse1")
    original_cudart = lambda: SimpleNamespace(cudaHostRegister="host_register")
    fake_torch = SimpleNamespace(
        profiler=SimpleNamespace(
            ProfilerActivity=fake_activity, profile=lambda **kw: kw
        ),
        cuda=SimpleNamespace(cudart=original_cudart),
    )
    monkeypatch.setattr(musa_profiler, "torch", fake_torch)

    musa_profiler.apply_musa_profiler_patches()

    assert fake_activity.CUDA == "cuda"
    assert isinstance(fake_torch.cuda.cudart(), musa_profiler.MusaCudartProxy)
    assert musa_profiler._patches_applied


@pytest.mark.parametrize(
    "invalid_api",
    ["legacy_start", "legacy_stop", "list_start", "list_stop", "inner_order"],
)
def test_invalid_contract_fails_before_any_patch(
    monkeypatch, profiler_contract, invalid_api
):
    scheduler, profiler_list = profiler_contract
    if invalid_api == "legacy_start":
        scheduler.start_profile = lambda self: None
    elif invalid_api == "legacy_stop":
        scheduler.stop_profile = lambda self, required_argument: None
    elif invalid_api == "list_start":
        profiler_list.start = None
    elif invalid_api == "list_stop":
        profiler_list.stop = lambda self, required_argument: None
    else:
        profiler_list.__init__ = lambda self, inners: setattr(
            self, "inners", list(reversed(inners))
        )
    monkeypatch.setattr(musa_profiler, "_prepare_torch_profiler_redirects", lambda: [])
    original_start = scheduler.start_profile
    original_list_start = profiler_list.start

    with pytest.raises(RuntimeError, match="SGLang profiler cleanup requires"):
        musa_profiler.apply_musa_profiler_patches()

    assert scheduler.start_profile is original_start
    assert profiler_list.start is original_list_start
    assert not getattr(scheduler, "_musa_profiler_lifecycle_patched", False)
    assert not getattr(profiler_list, "_musa_profiler_lifecycle_patched", False)
    assert not musa_profiler._patches_applied


@pytest.mark.parametrize("missing_api", ["activity", "cudart"])
def test_missing_torch_api_fails_before_any_patch(
    monkeypatch, profiler_contract, missing_api
):
    scheduler, profiler_list = profiler_contract
    activity = SimpleNamespace(CUDA="cuda")
    if missing_api != "activity":
        activity.PrivateUse1 = "privateuse1"
    cuda = SimpleNamespace()
    if missing_api != "cudart":
        cuda.cudart = lambda: object()
    monkeypatch.setattr(
        musa_profiler,
        "torch",
        SimpleNamespace(
            profiler=SimpleNamespace(
                ProfilerActivity=activity, profile=lambda **kw: kw
            ),
            cuda=cuda,
        ),
    )
    with pytest.raises(musa_profiler.MusaProfilerError):
        musa_profiler.apply_musa_profiler_patches()

    assert activity.CUDA == "cuda"
    assert not getattr(scheduler, "_musa_profiler_lifecycle_patched", False)
    assert not getattr(profiler_list, "_musa_profiler_lifecycle_patched", False)
    assert not musa_profiler._patches_applied


def test_missing_lifecycle_api_fails_before_any_patch(monkeypatch, profiler_contract):
    scheduler, profiler_list = profiler_contract
    monkeypatch.delattr(sys.modules["sglang.srt.utils.profile_utils"], "_ProfilerList")
    monkeypatch.setattr(musa_profiler, "_prepare_torch_profiler_redirects", lambda: [])

    with pytest.raises(RuntimeError, match="profiler cleanup APIs are unavailable"):
        musa_profiler.apply_musa_profiler_patches()

    assert not getattr(scheduler, "_musa_profiler_lifecycle_patched", False)
    assert not getattr(profiler_list, "_musa_profiler_lifecycle_patched", False)
    assert not musa_profiler._patches_applied


def test_legacy_rollback_preserves_state_when_native_cleanup_fails():
    class MarkerError(RuntimeError):
        pass

    class SchedulerProfilerMixin:
        def start_profile(self, stage=None):
            self.profile_in_progress = True
            raise MarkerError("marker start failed")

        def stop_profile(self, stage=None):
            raise RuntimeError("torch stop failed")

    lifecycle._wrap_legacy_profiler_lifecycle(SchedulerProfilerMixin, MarkerError)
    scheduler = SchedulerProfilerMixin()
    profiler = object()
    scheduler.torch_profiler = profiler
    scheduler.profiler_start_forward_ct = 10

    with pytest.raises(RuntimeError, match="torch stop failed") as exc:
        scheduler.start_profile()

    assert isinstance(exc.value.__context__, MarkerError)
    assert scheduler.torch_profiler is profiler
    assert scheduler.profile_in_progress
    assert scheduler.profiler_start_forward_ct == 10


def test_legacy_rollback_forwards_only_stage_to_stop_after_marker_error():
    events = []

    class MarkerError(RuntimeError):
        pass

    class SchedulerProfilerMixin:
        def start_profile(self, stage=None, *, trace_option=None):
            self.profile_in_progress = True
            raise MarkerError("marker start failed")

        def stop_profile(self, stage=None):
            events.extend([("torch_stop", stage), ("marker_stop", stage)])
            raise MarkerError("marker stop failed")

    lifecycle._wrap_legacy_profiler_lifecycle(SchedulerProfilerMixin, MarkerError)
    scheduler = SchedulerProfilerMixin()
    scheduler.torch_profiler = object()
    scheduler.profiler_start_forward_ct = 10

    with pytest.raises(MarkerError, match="marker start failed"):
        scheduler.start_profile("prefill", trace_option=True)

    assert events == [("torch_stop", "prefill"), ("marker_stop", "prefill")]
    assert scheduler.torch_profiler is None
    assert not scheduler.profile_in_progress
    assert scheduler.profiler_start_forward_ct is None


def test_legacy_marker_stop_failure_clears_state_after_native_stops():
    events = []

    class MarkerError(RuntimeError):
        pass

    class SchedulerProfilerMixin:
        def start_profile(self, stage=None):
            pass

        def stop_profile(self, stage=None):
            events.extend(["torch_stop", "memory_stop", "marker_stop"])
            raise MarkerError("marker stop failed")

    lifecycle._wrap_legacy_profiler_lifecycle(SchedulerProfilerMixin, MarkerError)
    scheduler = SchedulerProfilerMixin()
    scheduler.torch_profiler = object()
    scheduler.profile_in_progress = True
    scheduler.profiler_start_forward_ct = 10

    with pytest.raises(MarkerError, match="marker stop failed"):
        scheduler.stop_profile()

    assert events == ["torch_stop", "memory_stop", "marker_stop"]
    assert scheduler.torch_profiler is None
    assert not scheduler.profile_in_progress
    assert scheduler.profiler_start_forward_ct is None


def test_multi_rank_mixed_marker_rejected_before_arming(profiler_contract, monkeypatch):
    scheduler, _ = profiler_contract
    sizes = []
    monkeypatch.setattr(
        "torch.distributed.get_world_size",
        lambda group: sizes.append(group) or group,
    )
    lifecycle.install_patches(
        lifecycle.prepare_profiler_lifecycle_patch(musa_profiler.MusaProfilerError)
    )
    worker = scheduler()
    worker.dp_tp_cpu_group = 2
    for activities in (["GPU", "CUDA_PROFILER"], ["CPU", "GPU", "CUDA_PROFILER"]):
        result = worker.profile(SimpleNamespace(type="start", activities=activities))
        assert not result.success and "separately" in result.message
    assert (
        worker.profile(SimpleNamespace(type="start", activities=["GPU"]))
        == "configured"
    )
    assert (
        worker.profile(SimpleNamespace(type="start", activities=["CUDA_PROFILER"]))
        == "configured"
    )
    assert sizes == [2, 2]
    worker.dp_tp_cpu_group = 1
    assert (
        worker.profile(
            SimpleNamespace(type="start", activities=["CPU", "GPU", "CUDA_PROFILER"])
        )
        == "configured"
    )


@pytest.mark.parametrize("failure_index", range(9))
def test_atomic_patch_install_restores_entries_and_retries(
    profiler_contract, monkeypatch, failure_index
):
    scheduler, profiler_list = profiler_contract
    original = lambda **kwargs: kwargs
    base_cuda = lambda: SimpleNamespace(cudaHostRegister=object())
    fake_activity = SimpleNamespace(CUDA="cuda", PrivateUse1="musa")
    fake = SimpleNamespace(
        profiler=SimpleNamespace(ProfilerActivity=fake_activity, profile=original),
        cuda=SimpleNamespace(cudart=base_cuda),
    )
    monkeypatch.setattr(musa_profiler, "torch", fake)
    targets = [scheduler, profiler_list, fake.profiler, fake.cuda]
    before = [dict(vars(target)) for target in targets]
    install = lifecycle.install_patches

    class FailOnce:
        def __setattr__(self, name, value):
            raise RuntimeError("injected mount failure")

    def fail(changes):
        failing = list(changes)
        failing.insert(failure_index, (FailOnce(), "entry", None))
        install(failing)

    monkeypatch.setattr(musa_profiler, "install_patches", fail)
    with pytest.raises(RuntimeError, match="injected mount failure"):
        musa_profiler.apply_musa_profiler_patches()
    assert [dict(vars(target)) for target in targets] == before
    assert not musa_profiler._patches_applied
    assert fake_activity.CUDA == "cuda"

    monkeypatch.setattr(musa_profiler, "install_patches", install)
    musa_profiler.apply_musa_profiler_patches()
    wrapped = fake.profiler.profile
    proxy = fake.cuda.cudart()
    musa_profiler.apply_musa_profiler_patches()
    assert fake.profiler.profile is wrapped
    assert fake.cuda.cudart() is proxy
    assert fake_activity.CUDA == "cuda"


def test_multi_rank_startup_rollback_stops_torch_locally(profiler_contract):
    _, profiler_list = profiler_contract
    framework = sys.modules["sglang.srt.utils.profile_utils"]
    events = []

    class TorchProbe(framework._ProfilerTorch):
        def start(self):
            self.torch_profiler = SimpleNamespace(
                stop=lambda: events.append("local torch stop")
            )

        def stop(self):
            pytest.fail("must not enter upstream collective-based stop")

    class FailingMemory:
        def start(self):
            raise RuntimeError("local startup problem")

    lifecycle.install_patches(
        lifecycle.prepare_profiler_lifecycle_patch(musa_profiler.MusaProfilerError)
    )
    with pytest.raises(RuntimeError, match="local startup problem"):
        profiler_list([TorchProbe(), FailingMemory()]).start()
    assert events == ["local torch stop"]


def test_isolated_redirect_install_does_not_replace_prior_proxy(monkeypatch):
    def facade(value):
        return SimpleNamespace(
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CUDA="cuda", PrivateUse1="musa"),
                profile=lambda **kw: kw,
            ),
            cuda=SimpleNamespace(cudart=lambda: SimpleNamespace(delegated=value)),
        )

    real = facade("existing")
    monkeypatch.setattr(musa_profiler, "torch", real)
    lifecycle.install_patches(musa_profiler._prepare_torch_profiler_redirects())
    original_proxy = real.cuda.cudart()
    original_profile = real.profiler.profile
    with monkeypatch.context() as isolated:
        fake = facade("test")
        isolated.setattr(musa_profiler, "torch", fake)
        lifecycle.install_patches(musa_profiler._prepare_torch_profiler_redirects())
        assert fake.cuda.cudart().delegated == "test"
        assert real.cuda.cudart() is original_proxy
    assert real.cuda.cudart().delegated == "existing"
    assert real.profiler.profile is original_profile
