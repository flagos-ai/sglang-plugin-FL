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

from sglang_fl.dispatch.backends.vendor.mthreads.patches import topk_schedule


def _tensor_shape(*shape):
    return SimpleNamespace(ndim=len(shape), shape=shape)


def test_topk_schedule_is_enabled_by_default_but_can_be_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_MUSA_TOPK_SCHEDULE", raising=False)
    assert topk_schedule._enabled()

    monkeypatch.setenv("SGLANG_MUSA_TOPK_SCHEDULE", "off")
    assert not topk_schedule._enabled()


def test_target_shape_requires_measured_dimensions():
    weights = _tensor_shape(64, 8)
    assert topk_schedule._is_target_shape(weights, _tensor_shape(64, 256), 0, None)
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(65, 256), 0, None)
    assert topk_schedule._is_target_shape(
        _tensor_shape(4095, 8), _tensor_shape(4095, 256), 0, None
    )
    assert topk_schedule._is_target_shape(
        _tensor_shape(15360, 8), _tensor_shape(15360, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        _tensor_shape(16385, 8), _tensor_shape(16385, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(64, 128), 0, None)
    assert not topk_schedule._is_target_shape(
        _tensor_shape(64, 4), _tensor_shape(64, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 1.0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 0, object()
    )


def test_wrapper_temporarily_pins_target_and_restores_configs():
    configs = [object(), object()]
    selected = object()
    kernel = SimpleNamespace(configs=configs)
    seen = []

    def original(*args):
        seen.append(kernel.configs)
        return "result"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, selected)
    result = wrapped(_tensor_shape(64, 8), object(), _tensor_shape(64, 256))

    assert result == "result"
    assert seen == [[selected]]
    assert kernel.configs is configs
    assert getattr(wrapped, topk_schedule._PATCH_MARKER)


def test_wrapper_leaves_unmeasured_shape_on_autotuner():
    configs = [object(), object()]
    kernel = SimpleNamespace(configs=configs)
    seen = []

    def original(*args):
        seen.append(kernel.configs)

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    wrapped(_tensor_shape(65, 8), object(), _tensor_shape(65, 256))

    assert seen == [configs]
    assert kernel.configs is configs
