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

from types import SimpleNamespace

from sglang_fl.dispatch.backends.vendor.mthreads import eventfd_completion
from sglang_fl.dispatch.backends.vendor.mthreads.eventfd_completion import provider
from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    eventfd_completion as patch,
)


class _FallbackEvent:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class _Completion:
    def __init__(self, error=None):
        self.error = error
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        if self.error is not None:
            raise self.error


class _Result:
    def __init__(self):
        self.next_token_ids = SimpleNamespace(
            device=SimpleNamespace(type="musa")
        )
        self.copy_done = None


def test_copy_wrapper_replaces_recorded_event_with_completion_proxy(monkeypatch):
    fallback = _FallbackEvent()
    completion = _Completion()

    def original(result, return_logprob):
        assert return_logprob is False
        result.next_token_ids = "cpu-output"
        result.copy_done = fallback
        return "copied"

    monkeypatch.setattr(
        eventfd_completion,
        "try_enqueue_musa_completion",
        lambda device: completion,
    )
    result = _Result()
    value = patch._wrap_copy_to_cpu(original)(result, False)

    assert value == "copied"
    assert isinstance(result.copy_done, provider.MusaCompletionEventProxy)
    result.copy_done.synchronize()
    result.copy_done.synchronize()
    assert completion.wait_calls == 1
    assert fallback.sync_calls == 0


def test_completion_wait_failure_uses_original_event():
    fallback = _FallbackEvent()
    completion = _Completion(RuntimeError("wait failed"))
    proxy = provider.MusaCompletionEventProxy(completion, fallback)

    proxy.synchronize()
    proxy.synchronize()

    assert completion.wait_calls == 1
    assert fallback.sync_calls == 1


def test_pool_exhaustion_keeps_original_event(monkeypatch):
    fallback = _FallbackEvent()

    def original(result, _return_logprob):
        result.copy_done = fallback

    monkeypatch.setattr(
        eventfd_completion,
        "try_enqueue_musa_completion",
        lambda device: None,
    )
    result = _Result()
    patch._wrap_copy_to_cpu(original)(result, False)

    assert result.copy_done is fallback


def test_auto_mode_is_s5000_only(monkeypatch):
    monkeypatch.delenv(patch._ENV_NAME, raising=False)
    monkeypatch.setattr(patch, "_device_is_s5000", lambda: True)
    assert patch._enabled()
    monkeypatch.setattr(patch, "_device_is_s5000", lambda: False)
    assert not patch._enabled()


def test_explicit_off_wins_over_device(monkeypatch):
    monkeypatch.setenv(patch._ENV_NAME, "off")
    monkeypatch.setattr(patch, "_device_is_s5000", lambda: True)
    assert not patch._enabled()
