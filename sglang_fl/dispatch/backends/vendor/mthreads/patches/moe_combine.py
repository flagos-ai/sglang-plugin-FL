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

"""Shape-gated deterministic Qwen3.6 shared-expert combine for MUSA.

This module keeps the product worktree untouched: it swaps the already-imported
``moe_sum_reduce`` symbol in SGLang's fused-MoE module and wraps the Qwen2Moe
block that is reused by Qwen3.5/3.6.

The model wrapper computes the shared expert *without* its sigmoid gate and
places both tensors in a short-lived ContextVar.  The combine wrapper consumes
that context only when the actual routed down-output and destination satisfy
the measured contract.  Any miss or kernel exception delegates to the original
routed reduction and the model wrapper performs the original sigmoid-weighted
in-place add, so a shared expert can never be dropped.  The launch writes the
complete TP-local output and preserves the caller's in-place destination and
stream/event semantics.

For exact decode-graph buckets M=40/M=64, an independent opt-in preserves
Qwen's existing two-stream overlap.  The shared branch remains on the graph's
primary stream and routed experts remain on ``alt_stream``.  At the routed
reduce seam, the alternate stream waits for the already-enqueued shared branch
and launches the same one-kernel combine, so the fused consumer joins both
producers immediately before reading them.  The framework's existing final
alternate-to-primary join is retained.  Python and ContextVar state are used
only while SGLang performs its two eager warmups and captures the graph; replay
contains only the recorded stream dependencies and GPU kernels.

See the MThreads plugin README section "Patch background and equivalence
notes" for the combine screen behind this candidate.
"""

from __future__ import annotations

import importlib
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any, Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Keep CPU/plugin imports independent of Triton.  The lazy kernel builder
# populates this module global immediately before defining the JIT function so
# Triton's annotation evaluator can resolve ``tl.constexpr``.
tl: Any = None

_ENV_NAME = "SGLANG_MUSA_DETERMINISTIC_MOE_COMBINE"
_DECODE_GRAPH_ENV_NAME = "SGLANG_MUSA_DETERMINISTIC_MOE_COMBINE_DECODE_GRAPH"
_DISABLED_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_PATCH_MARKER = "_sglang_fl_musa_deterministic_moe_combine"
_MODEL_CONTRACT_MARKER = "_sglang_fl_musa_deterministic_moe_contract"

_HIDDEN = 2048
_TOPK = 8
_NUM_EXPERTS = 256
_SUPPORTED_TOKENS = frozenset((2048, 4096, 6144, 8192, 16384))
_DECODE_GRAPH_SUPPORTED_TOKENS = frozenset((40, 64))


@dataclass(frozen=True)
class _MoeModelContract:
    """Immutable constructor-time contract, independent of the config object."""

    model_type: str | None
    hidden_size: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    moe_intermediate_size: int | None
    shared_expert_intermediate_size: int | None
    matches: bool


def _typed_config_field(config: Any, name: str, expected_type: type) -> Any:
    """Read one config field, keeping exact type constraints and no coercion."""

    value = getattr(config, name, None)
    return value if type(value) is expected_type else None


def _model_contract_from_config(config: Any) -> _MoeModelContract:
    """Copy only primitive contract fields; never retain the config object.

    All config reads are captured at this one extraction boundary: a missing
    or unreadable field yields the non-matching contract, so an optional
    optimization probe can never become a model construction failure.  Exact
    field types are preserved and no implicit conversion is applied.
    """

    try:
        model_type = _typed_config_field(config, "model_type", str)
        hidden_size = _typed_config_field(config, "hidden_size", int)
        num_experts = _typed_config_field(config, "num_experts", int)
        num_experts_per_tok = _typed_config_field(
            config, "num_experts_per_tok", int
        )
        moe_intermediate_size = _typed_config_field(
            config, "moe_intermediate_size", int
        )
        shared_expert_intermediate_size = _typed_config_field(
            config, "shared_expert_intermediate_size", int
        )
    except Exception:  # noqa: BLE001 - unreadable config must fall back
        logger.debug(
            "MUSA deterministic combine config contract unavailable", exc_info=True
        )
        return _MoeModelContract(None, None, None, None, None, None, False)

    matches = (
        model_type in {"qwen3_5_moe_text", "qwen3_5_moe"}
        and hidden_size == _HIDDEN
        and num_experts == _NUM_EXPERTS
        and num_experts_per_tok == _TOPK
        and moe_intermediate_size == 512
        and shared_expert_intermediate_size == 512
    )
    return _MoeModelContract(
        model_type=model_type,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        matches=matches,
    )


def _init_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Get Qwen2MoeSparseMoeBlock's config from positional or keyword args."""

    if len(args) >= 2:
        return args[1]
    return kwargs.get("config")


def _clear_model_contract(instance: Any) -> None:
    try:
        delattr(instance, _MODEL_CONTRACT_MARKER)
    except AttributeError:
        pass


def _make_qwen_init(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap construction and publish a marker only after original init succeeds."""

    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(self, *args: Any, **kwargs: Any):
        # A repeated __init__ must not retain a marker from an earlier success
        # if the new constructor invocation fails.
        _clear_model_contract(self)
        result = original(self, *args, **kwargs)
        try:
            marker = _model_contract_from_config(_init_config(args, kwargs))
            setattr(self, _MODEL_CONTRACT_MARKER, marker)
        except Exception:
            _clear_model_contract(self)
            raise
        return result

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


@dataclass
class MoeCombineContext:
    """Per-forward shared-expert inputs and consumption state."""

    shared_unweighted: torch.Tensor
    gate_logits: torch.Tensor
    routed_scaling_factor: float = 1.0
    decode_graph_dual_stream: bool = False
    shared_stream: Any = None
    used: bool = False


_ACTIVE_CONTEXT: ContextVar[MoeCombineContext | None] = ContextVar(
    "sglang_fl_moe_combine_context", default=None
)
_TRITON_KERNEL = None
# A Triton compile/launch failure is a process-level capability failure for
# this optional path. Stop retrying it on every MoE layer/request; the
# original reduction remains available for the rest of the process.
_CANDIDATE_DISABLED = False
_DECODE_GRAPH_CANDIDATE_DISABLED = False
# One launch-submitted record per combine execution path (eager/prefill and
# decode-graph); the two entries are independent.
_SUCCESS_LOGGED: set[str] = set()
_SUCCESS_LOG_LOCK = Lock()
_DEVICE_NAME_CACHE: dict[tuple[str | None, str], str] = {}
_DEVICE_NAME_CACHE_LOCK = Lock()


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in _DISABLED_VALUES


def _decode_graph_enabled() -> bool:
    """Keep the new capture path independently reversible and off by default."""

    return (
        os.environ.get(_DECODE_GRAPH_ENV_NAME, "off").strip().lower()
        not in _DISABLED_VALUES
    )


def _distributed_rank() -> int:
    """Return the local process rank without making distributed a dependency."""

    try:
        distributed = getattr(torch, "distributed", None)
        if (
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            return int(distributed.get_rank())
    except Exception as exc:  # noqa: BLE001 - rank evidence must never affect execution
        logger.debug("MUSA deterministic MoE combine rank unavailable: %s", exc)
    return -1


def _log_success_once(path: str) -> None:
    """Emit one launch-submitted marker for each combine execution path.

    This records that the launch was submitted, not that asynchronous
    execution or graph replay has been verified.
    """

    with _SUCCESS_LOG_LOCK:
        if path in _SUCCESS_LOGGED:
            return
        _SUCCESS_LOGGED.add(path)
    logger.info(
        "MUSA deterministic MoE %s combine launch submitted: rank=%s",
        path,
        _distributed_rank(),
    )


def _device_cache_key(device: Any) -> tuple[str | None, str]:
    return device.type, str(device)


def _device_name(tensor: torch.Tensor) -> str:
    device = tensor.device
    if device.type != "musa":
        return ""
    cache_key = _device_cache_key(device)
    with _DEVICE_NAME_CACHE_LOCK:
        if cache_key in _DEVICE_NAME_CACHE:
            return _DEVICE_NAME_CACHE[cache_key]
        try:
            musa = getattr(torch, "musa", None)
            if musa is None or not musa.is_available():
                name = ""
            else:
                name = str(musa.get_device_name(device))
        except Exception:  # noqa: BLE001 - a missing runtime must fall back safely
            name = ""
        # Cache both successful and unknown results.  A runtime/query failure
        # must not become a per-layer host-side cost or repeatedly emit work.
        _DEVICE_NAME_CACHE[cache_key] = name
        return name


def _is_capture_or_graph(qwen2_module: Any, forward_batch: Any) -> bool:
    """Conservatively reject every known capture/graph indication."""

    capture_fn = getattr(qwen2_module, "get_is_capture_mode", None)
    if capture_fn is None:
        return True
    try:
        if bool(capture_fn()):
            return True
    except Exception:  # noqa: BLE001 - unknown runtime state is unsafe to patch
        return True

    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    for name in ("is_cuda_graph", "is_piecewise_cuda_graph", "is_capture"):
        value = getattr(mode, name, None)
        if value is None:
            continue
        try:
            if bool(value() if callable(value) else value):
                return True
        except Exception:  # noqa: BLE001 - unknown mode is unsafe to patch
            return True
    return False


def _is_decode_capture(qwen2_module: Any, forward_batch: Any) -> bool:
    """Accept only SGLang's startup capture of a real decode ForwardBatch."""

    capture_fn = getattr(qwen2_module, "get_is_capture_mode", None)
    if capture_fn is None:
        return False
    try:
        if not bool(capture_fn()):
            return False
    except Exception:  # noqa: BLE001 - unknown runtime state must use old path
        return False

    mode = getattr(forward_batch, "forward_mode", None)
    is_decode = getattr(mode, "is_decode", None)
    if is_decode is None:
        return False
    try:
        return bool(is_decode() if callable(is_decode) else is_decode)
    except Exception:  # noqa: BLE001 - unknown forward mode must use old path
        return False


def _is_deepep(qwen2_module: Any) -> bool:
    backend_fn = getattr(qwen2_module, "get_moe_a2a_backend", None)
    if backend_fn is None:
        return True
    try:
        backend = backend_fn()
        return bool(backend.is_deepep())
    except Exception:  # noqa: BLE001 - unknown backend must use the old path
        return True


def _is_bf16(tensor: Any) -> bool:
    return tensor.dtype == torch.bfloat16


def _is_contiguous(tensor: Any) -> bool:
    return bool(tensor.is_contiguous())


def _same_device(*tensors: Any) -> bool:
    devices = [tensor.device for tensor in tensors]
    return bool(devices) and all(device == devices[0] for device in devices[1:])


def _is_tensor(value: Any) -> bool:
    """Single entry object-category check before direct tensor attribute reads.

    The contract paths read ``shape``/``dtype``/``device``/``is_contiguous``
    directly.  Non-tensor inputs are rejected here and keep the original path,
    instead of raising from an attribute read that sits outside the launch
    guard.
    """

    return isinstance(value, torch.Tensor)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _contract_matches(
    routed: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float | None,
    context: MoeCombineContext,
) -> bool:
    """Check the complete measured combine contract before launching Triton."""

    if not _enabled() or _CANDIDATE_DISABLED:
        return False
    if context.decode_graph_dual_stream and _DECODE_GRAPH_CANDIDATE_DISABLED:
        return False
    if not (
        _is_tensor(routed)
        and _is_tensor(output)
        and _is_tensor(context.shared_unweighted)
        and _is_tensor(context.gate_logits)
    ):
        return False
    routed_shape = _shape(routed)
    output_shape = _shape(output)
    shared_shape = _shape(context.shared_unweighted)
    gate_shape = _shape(context.gate_logits)
    token_num = routed_shape[0] if routed_shape else -1
    supported_tokens = (
        _DECODE_GRAPH_SUPPORTED_TOKENS
        if context.decode_graph_dual_stream
        else _SUPPORTED_TOKENS
    )
    if (
        token_num not in supported_tokens
        or routed_shape != (token_num, _TOPK, _HIDDEN)
        or output_shape != (token_num, _HIDDEN)
        or shared_shape != (token_num, _HIDDEN)
        or gate_shape != (token_num, 1)
    ):
        return False
    if context.used or context.routed_scaling_factor != 1.0:
        return False
    if routed_scaling_factor is not None:
        try:
            if float(routed_scaling_factor) != 1.0:
                return False
        except (TypeError, ValueError):
            return False
    if not all(
        _is_bf16(tensor) and _is_contiguous(tensor)
        for tensor in (
            routed,
            output,
            context.shared_unweighted,
            context.gate_logits,
        )
    ):
        return False
    if not _same_device(routed, output, context.shared_unweighted, context.gate_logits):
        return False
    if routed.device.type != "musa":
        return False
    return "S5000" in _device_name(routed).upper()


def _shared_context_matches(
    hidden_states: torch.Tensor,
    context: MoeCombineContext,
) -> bool:
    """Validate model-side shared tensors without allocating a routed buffer."""

    hidden_shape = _shape(hidden_states)
    token_num = hidden_shape[0] if hidden_shape else -1
    return (
        _shape(context.shared_unweighted) == (token_num, _HIDDEN)
        and _shape(context.gate_logits) == (token_num, 1)
        and _is_bf16(context.shared_unweighted)
        and _is_bf16(context.gate_logits)
        and _is_contiguous(context.shared_unweighted)
        and _is_contiguous(context.gate_logits)
        and _same_device(hidden_states, context.shared_unweighted, context.gate_logits)
    )


def _candidate_for_tokens(token_num: int) -> tuple[int, int, int]:
    if token_num == 40:
        return 2, 512, 8
    if token_num == 64:
        return 1, 2048, 16
    if token_num == 2048:
        return 2, 512, 8
    if token_num in _SUPPORTED_TOKENS:
        return 1, 2048, 16
    raise ValueError(f"unsupported deterministic combine token count: {token_num}")


def _get_triton_kernel():
    """Lazily define the kernel so CPU/plugin imports need no Triton runtime."""

    global _TRITON_KERNEL, tl
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def deterministic_moe_combine_kernel(
        routed_ptr,
        routed_stride_0,
        routed_stride_1,
        routed_stride_2,
        gate_logits_ptr,
        gate_stride_0,
        gate_stride_1,
        shared_ptr,
        shared_stride_0,
        shared_stride_1,
        output_ptr,
        output_stride_0,
        output_stride_1,
        token_num: int,
        hidden_dim: int,
        routed_scaling_factor: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        routed_stride_0 = tl.cast(routed_stride_0, dtype=tl.int64)
        routed_stride_1 = tl.cast(routed_stride_1, dtype=tl.int64)
        routed_stride_2 = tl.cast(routed_stride_2, dtype=tl.int64)
        gate_stride_0 = tl.cast(gate_stride_0, dtype=tl.int64)
        gate_stride_1 = tl.cast(gate_stride_1, dtype=tl.int64)
        shared_stride_0 = tl.cast(shared_stride_0, dtype=tl.int64)
        shared_stride_1 = tl.cast(shared_stride_1, dtype=tl.int64)
        output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
        output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

        token_block_id = tl.program_id(0)
        dim_block_id = tl.program_id(1)
        offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask_token = offs_token < token_num
        mask_dim = offs_dim < hidden_dim
        mask = mask_token[:, None] & mask_dim[None, :]

        routed_base = (
            routed_ptr
            + offs_token[:, None] * routed_stride_0
            + offs_dim[None, :] * routed_stride_2
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
        for topk_idx in tl.range(0, TOPK, num_stages=1):
            routed_tile = tl.load(
                routed_base + topk_idx * routed_stride_1,
                mask=mask,
                other=0.0,
            )
            accumulator += routed_tile.to(tl.float32)
        accumulator *= routed_scaling_factor

        gate_logits = tl.load(
            gate_logits_ptr + offs_token * gate_stride_0 + 0 * gate_stride_1,
            mask=mask_token,
            other=0.0,
        ).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-gate_logits))
        shared_ptrs = (
            shared_ptr
            + offs_token[:, None] * shared_stride_0
            + offs_dim[None, :] * shared_stride_1
        )
        accumulator += gate[:, None] * tl.load(shared_ptrs, mask=mask, other=0.0).to(
            tl.float32
        )

        output_ptrs = (
            output_ptr
            + offs_token[:, None] * output_stride_0
            + offs_dim[None, :] * output_stride_1
        )
        tl.store(
            output_ptrs,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    _TRITON_KERNEL = deterministic_moe_combine_kernel
    return _TRITON_KERNEL


def _launch_candidate(
    routed: torch.Tensor,
    output: torch.Tensor,
    context: MoeCombineContext,
) -> None:
    import triton

    token_num = int(routed.shape[0])
    block_m, block_dim, num_warps = _candidate_for_tokens(token_num)
    kernel = _get_triton_kernel()
    grid = (triton.cdiv(token_num, block_m), triton.cdiv(_HIDDEN, block_dim))
    kernel[grid](
        routed,
        *routed.stride(),
        context.gate_logits,
        *context.gate_logits.stride(),
        context.shared_unweighted,
        *context.shared_unweighted.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        hidden_dim=_HIDDEN,
        routed_scaling_factor=1.0,
        TOPK=_TOPK,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
        num_stages=1,
    )


def _wrap_moe_sum_reduce(original: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        routed: torch.Tensor,
        output: torch.Tensor,
        routed_scaling_factor: float | None,
        *args: Any,
        **kwargs: Any,
    ):
        context = _ACTIVE_CONTEXT.get()
        if context is None or not _contract_matches(
            routed, output, routed_scaling_factor, context
        ):
            return original(routed, output, routed_scaling_factor, *args, **kwargs)

        try:
            if context.decode_graph_dual_stream:
                # Python reaches this seam only after all routed-expert work
                # has been enqueued on ``alt_stream``.  The shared branch was
                # enqueued earlier on the primary stream, so this tail wait
                # preserves overlap and joins the two producers immediately
                # before their fused consumer.
                consumer_stream = torch.musa.current_stream()
                if (
                    context.shared_stream is None
                    or consumer_stream == context.shared_stream
                ):
                    raise RuntimeError("decode-graph combine stream contract mismatch")
                context.shared_unweighted.record_stream(consumer_stream)
                context.gate_logits.record_stream(consumer_stream)
                consumer_stream.wait_stream(context.shared_stream)
            _launch_candidate(routed, output, context)
        except Exception as exc:  # noqa: BLE001 - preserve old chain on any failure
            global _CANDIDATE_DISABLED, _DECODE_GRAPH_CANDIDATE_DISABLED
            context.used = False
            if context.decode_graph_dual_stream:
                first_failure = not _DECODE_GRAPH_CANDIDATE_DISABLED
                _DECODE_GRAPH_CANDIDATE_DISABLED = True
                path = "decode-graph"
            else:
                first_failure = not _CANDIDATE_DISABLED
                _CANDIDATE_DISABLED = True
                path = "eager/prefill"
            if first_failure:
                logger.warning(
                    "MUSA deterministic MoE %s combine disabled after first failure; "
                    "using original reduction: %s",
                    path,
                    exc,
                )
            return original(routed, output, routed_scaling_factor, *args, **kwargs)

        # The launch is asynchronous but the caller's stream/event semantics
        # are unchanged.  The model wrapper skips the shared add only after the
        # launch was submitted.
        context.used = True
        _log_success_once(
            "decode-graph" if context.decode_graph_dual_stream else "eager/prefill"
        )
        return None

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _extract_linear_output(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


def _model_contract_matches(
    qwen2_module: Any,
    module: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    use_reduce_scatter: bool = False,
    should_allreduce_fusion: bool = False,
    *,
    decode_graph: bool = False,
) -> bool:
    if not _enabled() or _CANDIDATE_DISABLED:
        return False
    if decode_graph:
        if (
            not _decode_graph_enabled()
            or _DECODE_GRAPH_CANDIDATE_DISABLED
            or not _is_decode_capture(qwen2_module, forward_batch)
            or getattr(module, "alt_stream", None) is None
        ):
            return False
        supported_tokens = _DECODE_GRAPH_SUPPORTED_TOKENS
    else:
        if _is_capture_or_graph(qwen2_module, forward_batch):
            return False
        supported_tokens = _SUPPORTED_TOKENS
    if _is_deepep(qwen2_module):
        return False
    # The tested combine writes the complete TP-local output.  Keep the
    # reduce-scatter variant on the original path until it has its own proof.
    # ``should_allreduce_fusion`` is supported: the candidate preserves the
    # in-place destination and lets the downstream custom-AR owner retain it.
    if use_reduce_scatter:
        return False
    if not _is_tensor(hidden_states):
        return False
    # Qwen3.5/3.6 constructs ``alt_stream`` on MUSA even for eager execution.
    # The framework only consumes it in the capture-mode branch, which is
    # rejected above. Presence of the handle alone must therefore not make
    # the measured eager path miss the contract.
    hidden_shape = _shape(hidden_states)
    if hidden_shape[1:] != (_HIDDEN,):
        return False
    if hidden_shape[0] not in supported_tokens:
        return False
    if not _is_bf16(hidden_states) or not _is_contiguous(hidden_states):
        return False
    if hidden_states.device.type != "musa":
        return False
    if "S5000" not in _device_name(hidden_states).upper():
        return False

    marker = getattr(module, _MODEL_CONTRACT_MARKER, None)
    if not isinstance(marker, _MoeModelContract) or not marker.matches:
        return False
    if getattr(module, "tp_size", None) != 2:
        return False
    if getattr(module, "num_experts", None) != _NUM_EXPERTS:
        return False
    if getattr(module, "num_shared_experts", 0) <= 0:
        return False
    if getattr(module, "num_fused_shared_experts", 0) != 0:
        return False
    if getattr(module, "enable_shared_expert_fusion", False):
        return False
    if getattr(module, "shared_expert", None) is None:
        return False
    if getattr(module, "shared_expert_gate", None) is None:
        return False

    topk_config = getattr(getattr(module, "topk", None), "topk_config", None)
    if getattr(topk_config, "top_k", None) != _TOPK:
        return False
    experts = getattr(module, "experts", None)
    if getattr(experts, "num_experts", None) != _NUM_EXPERTS:
        return False
    runner_config = getattr(experts, "moe_runner_config", None)
    if getattr(runner_config, "inplace", None) is not True:
        return False
    if getattr(runner_config, "no_combine", False):
        return False
    scale = getattr(runner_config, "routed_scaling_factor", None)
    if scale is not None:
        try:
            if float(scale) != 1.0:
                return False
        except (TypeError, ValueError):
            return False
    # The existing SGLang all-reduce combine path writes routed results with
    # BF16 atomics and bypasses ``moe_sum_reduce``.  Do not alter that path or
    # precompute shared tensors when it is selected; this candidate is only
    # the measured non-atomic reduction branch.
    server_args_fn = getattr(qwen2_module, "get_global_server_args", None)
    if server_args_fn is None:
        return False
    try:
        if bool(getattr(server_args_fn(), "enable_fused_moe_sum_all_reduce", False)):
            return False
    except Exception:  # noqa: BLE001 - unknown server mode must use old path
        return False
    return True


def _weighted_shared(context: MoeCombineContext) -> torch.Tensor:
    return F.sigmoid(context.gate_logits) * context.shared_unweighted


def _decode_graph_model_contract_matches(
    qwen2_module: Any,
    module: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    use_reduce_scatter: bool = False,
    should_allreduce_fusion: bool = False,
) -> bool:
    return _model_contract_matches(
        qwen2_module,
        module,
        hidden_states,
        forward_batch,
        use_reduce_scatter,
        should_allreduce_fusion,
        decode_graph=True,
    )


def _forward_decode_graph_combine(
    qwen2_module: Any,
    module: Any,
    hidden_states: torch.Tensor,
    use_reduce_scatter: bool,
    should_allreduce_fusion: bool,
) -> torch.Tensor:
    """Capture the exact B40/B64 combine while retaining Qwen's two streams."""

    num_tokens, hidden_dim = hidden_states.shape
    flat_hidden_states = hidden_states.view(-1, hidden_dim)
    current_stream = torch.musa.current_stream()

    # Match Qwen2MoeSparseMoeBlock.forward_normal_dual_stream: fork the
    # alternate stream before enqueuing the shared branch, and use one cloned
    # input for both shared projections.
    module.alt_stream.wait_stream(current_stream)
    shared_input = flat_hidden_states.clone()
    shared_unweighted = module.shared_expert(shared_input)
    shared_gate_logits = _extract_linear_output(module.shared_expert_gate(shared_input))
    if not isinstance(shared_unweighted, torch.Tensor) or not isinstance(
        shared_gate_logits, torch.Tensor
    ):
        raise TypeError("decode-graph shared expert/gate did not return tensors")
    context = MoeCombineContext(
        shared_unweighted=shared_unweighted,
        gate_logits=shared_gate_logits,
        routed_scaling_factor=1.0,
        decode_graph_dual_stream=True,
        shared_stream=current_stream,
    )
    if not _shared_context_matches(flat_hidden_states, context):
        raise ValueError("decode-graph shared expert context is outside contract")

    token = _ACTIVE_CONTEXT.set(context)
    try:
        with torch.musa.stream(module.alt_stream):
            final_hidden_states = module._forward_router_experts(flat_hidden_states)

        # This is the original alternate-to-primary join.  On a candidate
        # hit, moe_sum_reduce has already inserted the reciprocal tail wait
        # (shared producer to alternate consumer) before the fused kernel.
        current_stream.wait_stream(module.alt_stream)

        if not context.used:
            # A contract miss or launch failure has already run the original
            # routed reduction on alt_stream.  Preserve the exact in-place
            # destination by applying the shared term only after the join.
            final_hidden_states += _weighted_shared(context)
    finally:
        _ACTIVE_CONTEXT.reset(token)

    if module.tp_size > 1 and not qwen2_module.should_skip_post_experts_all_reduce(
        is_tp_path=True,
        use_reduce_scatter=use_reduce_scatter,
        should_allreduce_fusion=should_allreduce_fusion,
    ):
        final_hidden_states = qwen2_module.tensor_model_parallel_all_reduce(
            final_hidden_states
        )

    return final_hidden_states.view(num_tokens, hidden_dim)


def _make_qwen_forward(
    qwen2_module: Any,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Any = None,
        use_reduce_scatter: bool = False,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        if _decode_graph_model_contract_matches(
            qwen2_module,
            self,
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        ):
            return _forward_decode_graph_combine(
                qwen2_module,
                self,
                hidden_states,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        if not _model_contract_matches(
            qwen2_module,
            self,
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        ):
            return original(
                self,
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        num_tokens, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        try:
            shared_unweighted = self.shared_expert(flat_hidden_states)
            shared_gate_logits = _extract_linear_output(
                self.shared_expert_gate(flat_hidden_states)
            )
            if not isinstance(shared_unweighted, torch.Tensor) or not isinstance(
                shared_gate_logits, torch.Tensor
            ):
                raise TypeError("shared expert/gate did not return tensors")
            context = MoeCombineContext(
                shared_unweighted=shared_unweighted,
                gate_logits=shared_gate_logits,
                routed_scaling_factor=1.0,
            )
            if not _shared_context_matches(flat_hidden_states, context):
                raise ValueError("shared expert context is outside combine contract")
        except Exception as exc:  # noqa: BLE001 - exact old model path is safest
            logger.debug("MUSA deterministic MoE context skipped: %s", exc)
            return original(
                self,
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        token = _ACTIVE_CONTEXT.set(context)
        try:
            final_hidden_states = self._forward_router_experts(flat_hidden_states)
        finally:
            # Never leave one layer's shared tensors visible to a later layer,
            # coroutine, or request after the router returns or raises.
            _ACTIVE_CONTEXT.reset(token)

        if not context.used:
            # Candidate miss/failure: the original fused-MoE reduction has
            # already populated final_hidden_states, so preserve the exact
            # in-place shared-expert ownership contract.
            final_hidden_states += _weighted_shared(context)

        if self.tp_size > 1 and not qwen2_module.should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = qwen2_module.tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _restore_installation(
    fused_module: Any,
    qwen_cls: Any,
    old_fused: Any,
    old_init: Any,
    old_forward: Any,
) -> bool:
    """Restore the three pre-install objects and report full success.

    The restored values are exactly the objects that were live immediately
    before installation, including any wrapper already in place.  The code
    never unwraps through ``__wrapped__`` to an earlier function.
    """

    restored = True
    try:
        fused_module.moe_sum_reduce = old_fused
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore fused_moe.moe_sum_reduce",
            exc_info=True,
        )
    try:
        qwen_cls.__init__ = old_init
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore Qwen2MoeSparseMoeBlock.__init__",
            exc_info=True,
        )
    try:
        qwen_cls.forward = old_forward
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore Qwen2MoeSparseMoeBlock.forward",
            exc_info=True,
        )
    return restored


def apply_musa_deterministic_moe_combine_patch() -> bool:
    """Apply the idempotent local combine patch when the MUSA runtime exists."""

    if not _enabled():
        logger.info("MUSA deterministic MoE combine disabled by %s", _ENV_NAME)
        return False
    if not hasattr(torch, "musa"):
        logger.info("MUSA deterministic MoE combine skipped: torch.musa unavailable")
        return False

    # Snapshot all three seams so a failed import/assignment cannot expose a
    # half-patched product path.  Existing wrappers are restored as-is, which
    # keeps repeated application idempotent.
    try:
        fused_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        )
        qwen_module = importlib.import_module("sglang.srt.models.qwen2_moe")
        qwen_cls = qwen_module.Qwen2MoeSparseMoeBlock
        old_fused = fused_module.moe_sum_reduce
        old_init = qwen_cls.__init__
        old_forward = qwen_cls.forward
    except (ImportError, AttributeError) as exc:
        logger.warning("MUSA deterministic combine patch targets unavailable: %s", exc)
        return False

    # Build every wrapper before touching an install point.  The wrapper
    # builders are idempotent via their own marker check, so repeated
    # installation returns the existing wrappers and never adds a layer.
    try:
        new_fused = _wrap_moe_sum_reduce(old_fused)
        new_init = _make_qwen_init(old_init)
        new_forward = _make_qwen_forward(qwen_module, old_forward)
    except Exception as exc:
        logger.warning("MUSA deterministic combine wrapper build failed: %s", exc)
        return False

    # Publish the three wrappers together; restore the exact pre-install
    # objects if any assignment fails.
    try:
        fused_module.moe_sum_reduce = new_fused
        qwen_cls.__init__ = new_init
        qwen_cls.forward = new_forward
    except Exception as exc:
        if _restore_installation(
            fused_module, qwen_cls, old_fused, old_init, old_forward
        ):
            logger.warning(
                "MUSA deterministic combine rolled back after install error: %s",
                exc,
            )
        else:
            logger.error(
                "MUSA deterministic combine install failed and rollback was "
                "incomplete: %s",
                exc,
            )
        return False

    logger.info(
        "MUSA deterministic MoE combine applied for Qwen3.6 TP2 BF16 "
        "M=2048/4096/6144/8192/16384 "
        "(M2048 BM2/BD512/W8/S1; larger BM1/BD2048/W16/S1); "
        "decode-graph B40/B64=%s",
        "enabled" if _decode_graph_enabled() else "disabled",
    )
    return True


__all__ = [
    "MoeCombineContext",
    "apply_musa_deterministic_moe_combine_patch",
]
