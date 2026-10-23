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

from contextlib import contextmanager
from types import SimpleNamespace
import base64
import json

import torch

from sglang_fl import profiling_hooks
from sglang_fl.profiling_hooks import (
    _OPERATOR_MARKER_PREFIX,
    _load_sglang_jit_around_hook,
    _operator_marker_name,
    _shape_and_dtype,
    _tvm_function_call_around_hook,
    _tvm_function_names,
    _tvm_get_function_around_hook,
    profile_dispatch_impl_call,
)


def _decode_marker(marker: str) -> dict:
    assert marker.startswith(_OPERATOR_MARKER_PREFIX)
    encoded = marker[len(_OPERATOR_MARKER_PREFIX) :]
    assert '"' not in encoded
    return json.loads(base64.urlsafe_b64decode(encoded))


def test_tensor_description_contains_shape_and_dtype():
    tensor = torch.empty((2, 3), dtype=torch.bfloat16)

    assert _shape_and_dtype(tensor) == ([2, 3], "torch.bfloat16")


def test_operator_marker_is_trace_safe_and_contains_input_metadata():
    marker = _operator_marker_name(
        "sglang.kernel.example",
        (torch.empty((2, 3), dtype=torch.bfloat16), 64),
        {},
        source="unit_test",
    )

    payload = _decode_marker(marker)
    assert payload == {
        "operator_name": "sglang.kernel.example",
        "input_shapes": [[2, 3], []],
        "input_dtypes": ["torch.bfloat16", "Scalar[int]"],
        "source": "unit_test",
    }


def test_sglang_jit_function_is_named_and_unprofiled_call_is_transparent(
    monkeypatch,
):
    module = object()
    function = object()
    result = _load_sglang_jit_around_hook(lambda *args: module, "clamp_position")
    assert result is module
    result = _tvm_get_function_around_hook(
        lambda instance, name: function, module, "clamp_position"
    )
    assert result is function
    assert (
        _tvm_function_names[id(function)]
        == "sglang.jit_kernel.clamp_position.clamp_position"
    )

    monkeypatch.setattr(profiling_hooks, "_profiler_enabled", lambda: False)
    result = _tvm_function_call_around_hook(
        lambda instance, value: value + 1, function, 4
    )
    assert result == 5


def test_dispatch_call_is_transparent_without_profiler(monkeypatch):
    monkeypatch.setattr(profiling_hooks, "_profiler_enabled", lambda: False)
    impl = SimpleNamespace(fn=lambda value: value + 1)

    assert profile_dispatch_impl_call(impl, 4) == 5


def test_dispatch_marker_records_selected_implementation(monkeypatch):
    markers = []

    @contextmanager
    def record_function(marker):
        markers.append(marker)
        yield

    monkeypatch.setattr(profiling_hooks, "_profiler_enabled", lambda: True)
    monkeypatch.setattr(torch.profiler, "record_function", record_function)
    impl = SimpleNamespace(
        op_name="rms_norm",
        impl_id="vendor.cuda",
        kind=SimpleNamespace(value="vendor"),
        vendor="cuda",
        runtime_source_category="third_party",
        runtime_source_library="sgl_kernel",
        fn=lambda value: value + 1,
    )

    assert profile_dispatch_impl_call(impl, 4) == 5
    assert len(markers) == 1
    payload = _decode_marker(markers[0])
    assert payload["dispatch_operator_name"] == "rms_norm"
    assert payload["dispatch_impl_id"] == "vendor.cuda"
    assert payload["dispatch_impl_kind"] == "vendor"
    assert payload["dispatch_vendor"] == "cuda"
    assert payload["dispatch_runtime_source_category"] == "third_party"
    assert payload["dispatch_runtime_source_library"] == "sgl_kernel"
