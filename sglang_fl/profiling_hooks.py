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

"""Correlation markers for eager operator profiling.

Torch Profiler records ATen input metadata directly, but calls through SGLang
JIT kernels and plugin dispatch do not always retain an unambiguous Python API
name.  These hooks add compact ``record_function`` markers only while the
profiler is active.  The report parser uses them to associate device kernels
with API names, shapes, dtypes, and vendor-dispatch provenance.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import torch

_OPERATOR_MARKER_PREFIX = "sglang_fl.operator_profile.kernel:"
_tvm_module_names: dict[int, str] = {}
_tvm_function_names: dict[int, str] = {}


def _callable_name(fn: Any) -> str:
    target = getattr(fn, "__func__", fn)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return getattr(target, "__name__", type(target).__name__)


def _shape_and_dtype(value: Any, depth: int = 0) -> tuple[Any, Any]:
    """Describe call inputs without retaining tensors or scalar values."""

    if isinstance(value, torch.Tensor):
        return list(value.shape), str(value.dtype)
    if value is None:
        return [], "None"
    if isinstance(value, (bool, int, float, complex, str)):
        return [], f"Scalar[{type(value).__name__}]"
    if depth >= 3:
        return [], f"Object[{type(value).__module__}.{type(value).__qualname__}]"
    if isinstance(value, (list, tuple)):
        described = [_shape_and_dtype(item, depth + 1) for item in value[:32]]
        shapes = [item[0] for item in described]
        dtypes = [item[1] for item in described]
        if len(value) > 32:
            shapes.append({"truncated_items": len(value) - 32})
            dtypes.append("Truncated")
        return shapes, dtypes
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))[:32]
        shapes: dict[str, Any] = {}
        dtypes: dict[str, Any] = {}
        for key, item in items:
            shapes[str(key)], dtypes[str(key)] = _shape_and_dtype(item, depth + 1)
        if len(value) > 32:
            shapes["__truncated_items__"] = len(value) - 32
            dtypes["__truncated_items__"] = "Truncated"
        return shapes, dtypes
    return [], f"Object[{type(value).__module__}.{type(value).__qualname__}]"


def _ordered_call_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parameter_names: list[str] | None = None,
) -> list[Any]:
    if not parameter_names:
        return [*args, *(kwargs[name] for name in sorted(kwargs))]

    by_name = dict(zip(parameter_names, args, strict=False))
    by_name.update(kwargs)
    values = [by_name[name] for name in parameter_names if name in by_name]
    declared = set(parameter_names)
    values.extend(kwargs[name] for name in sorted(kwargs) if name not in declared)
    if len(args) > len(parameter_names):
        values.extend(args[len(parameter_names) :])
    return values


def _profiler_enabled() -> bool:
    enabled = getattr(torch.autograd, "_profiler_enabled", None)
    return bool(enabled and enabled())


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _operator_marker_name(
    operator_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    source: str,
    parameter_names: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    values = _ordered_call_values(args, kwargs, parameter_names)
    metadata = [_shape_and_dtype(value) for value in values]
    payload = {
        "operator_name": operator_name,
        "input_shapes": [item[0] for item in metadata],
        "input_dtypes": [item[1] for item in metadata],
        "source": source,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    encoded = base64.urlsafe_b64encode(_json_key(payload).encode()).decode("ascii")
    return _OPERATOR_MARKER_PREFIX + encoded


def _record_call(
    original_fn,
    self,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    operator_name: str,
    source: str,
    parameter_names: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
):
    if not _profiler_enabled():
        return original_fn(self, *args, **kwargs)
    marker = _operator_marker_name(
        operator_name,
        args,
        kwargs,
        source=source,
        parameter_names=parameter_names,
        extra_metadata=extra_metadata,
    )
    with torch.profiler.record_function(marker):
        return original_fn(self, *args, **kwargs)


def profile_dispatch_impl_call(impl: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute one selected vendor implementation under an audit marker."""

    if not _profiler_enabled():
        return impl.fn(*args, **kwargs)
    kind = getattr(getattr(impl, "kind", None), "value", None)
    marker = _operator_marker_name(
        str(impl.op_name),
        args,
        kwargs,
        source="sglang_fl_dispatch",
        extra_metadata={
            "dispatch_operator_name": str(impl.op_name),
            "dispatch_impl_id": str(impl.impl_id),
            "dispatch_impl_kind": str(kind or impl.kind),
            "dispatch_vendor": impl.vendor,
            "dispatch_runtime_source_category": impl.runtime_source_category,
            "dispatch_runtime_source_library": impl.runtime_source_library,
        },
    )
    with torch.profiler.record_function(marker):
        return impl.fn(*args, **kwargs)


def _forward_around_hook(original_fn, self, *args, **kwargs):
    selected = getattr(self, "_forward_method", None)
    return _record_call(
        original_fn,
        self,
        args,
        kwargs,
        operator_name=_callable_name(selected),
        source="sglang_multi_platform_op",
    )


def _triton_run_around_hook(original_fn, self, *args, grid, warmup, **kwargs):
    def invoke(instance, *call_args, **call_kwargs):
        return original_fn(
            instance,
            *call_args,
            grid=grid,
            warmup=warmup,
            **call_kwargs,
        )

    if warmup:
        return invoke(self, *args, **kwargs)
    parameters = getattr(self, "params", None)
    parameter_names = (
        [str(parameter.name) for parameter in parameters]
        if parameters is not None
        else None
    )
    return _record_call(
        invoke,
        self,
        args,
        kwargs,
        operator_name=_callable_name(getattr(self, "fn", None)),
        source="triton_jit",
        parameter_names=parameter_names,
    )


def _load_sglang_jit_around_hook(original_fn, *args, **kwargs):
    module = original_fn(*args, **kwargs)
    if args:
        _tvm_module_names[id(module)] = str(args[0])
    return module


def _tvm_get_function_around_hook(original_fn, self, name, *args, **kwargs):
    function = original_fn(self, name, *args, **kwargs)
    module_name = _tvm_module_names.get(id(self))
    if module_name is not None:
        _tvm_function_names[id(function)] = f"sglang.jit_kernel.{module_name}.{name}"
    return function


def _tvm_function_call_around_hook(original_fn, self, *args, **kwargs):
    operator_name = _tvm_function_names.get(id(self))
    if operator_name is None:
        return original_fn(self, *args, **kwargs)
    return _record_call(
        original_fn,
        self,
        args,
        kwargs,
        operator_name=operator_name,
        source="sglang_jit_kernel",
    )


def setup_operator_profile_hooks() -> None:
    """Register eager-only correlation hooks."""

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    multi_platform_op = "sglang.srt.layers.utils.multi_platform.MultiPlatformOp"
    HookRegistry.register(
        f"{multi_platform_op}.forward", _forward_around_hook, HookType.AROUND
    )
    HookRegistry.register(
        "triton.runtime.jit.JITFunction.run",
        _triton_run_around_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.jit_kernel.utils.load_jit",
        _load_sglang_jit_around_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "tvm_ffi.module.Module.get_function",
        _tvm_get_function_around_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "tvm_ffi.core.Function.__call__",
        _tvm_function_call_around_hook,
        HookType.AROUND,
    )


__all__ = ["profile_dispatch_impl_call", "setup_operator_profile_hooks"]
