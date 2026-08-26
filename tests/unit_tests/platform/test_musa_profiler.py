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
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    profiler as musa_profiler,
)
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    sglang_0_5_11_profiler_lifecycle as lifecycle,
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
        profiler=SimpleNamespace(ProfilerActivity=fake_activity),
        cuda=SimpleNamespace(cudart=original_cudart),
    )
    monkeypatch.setattr(musa_profiler, "torch", fake_torch)

    musa_profiler._install_torch_profiler_redirects()

    assert fake_activity.CUDA == "privateuse1"
    assert fake_torch.cuda.cudart().cudaHostRegister == "host_register"
    assert isinstance(fake_torch.cuda.cudart(), musa_profiler.MusaCudartProxy)


def test_patch_install_keeps_musart_loading_lazy(monkeypatch):
    library_loads = []
    monkeypatch.setattr(musa_profiler, "_patches_applied", False)
    monkeypatch.setattr(
        musa_profiler,
        "apply_sglang_0_5_11_profiler_lifecycle_patch",
        lambda _error_type: None,
    )
    monkeypatch.setattr(
        musa_profiler, "_install_torch_profiler_redirects", lambda: None
    )
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


def test_sglang_profiler_lifecycle_patch_rejects_other_versions(monkeypatch):
    monkeypatch.setattr(lifecycle, "_installed_sglang_version", lambda: "0.5.12")

    with pytest.raises(RuntimeError, match="requires SGLang 0.5.11"):
        lifecycle._require_sglang_0_5_11()
