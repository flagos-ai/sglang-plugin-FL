"""Lossless top-k=1 GDN target verification on Ascend.

The CANN 8.5 native multi-token convolution and recurrent GDN operators do
not round and persist their intermediate state at the same boundaries as a
sequence of ordinary one-token decode calls.  That changes later target
logits even when every accepted token is identical.

For the linear EAGLE chain used by Qwen3.6 MTP, run the already-supported
one-token decode path once per draft position.  Save the resulting SSM and
convolution state after every position in SGLang's existing speculative
scratch buffers, then commit the snapshot selected by the accept result.
Tree verification and non-Ascend backends keep their native implementation.
"""

from __future__ import annotations

from functools import wraps
import importlib
import inspect
import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)

_GDN_MODULE = "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"
_GDN_CLASS = "AscendGDNAttnBackend"
_HYBRID_MODULE = (
    "sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend"
)
_HYBRID_CLASS = "AscendHybridLinearAttnBackend"
_FORWARD_MARKER = "_sglang_fl_sequential_gdn_verify"
_COMMIT_MARKER = "_sglang_fl_snapshot_gdn_commit"
_ACTIVE_FLAG = "_sglang_fl_sequential_verify_active"

_FORWARD_PARAMETERS = (
    "self",
    "layer",
    "forward_batch",
    "mixed_qkv",
    "a",
    "b",
    "kwargs",
)
_COMMIT_PARAMETERS = (
    "self",
    "last_correct_step_indices",
    "mamba_track_indices",
    "mamba_steps_to_track",
    "model",
    "req_pool_indices",
)


def _require_signature(function: Any, expected: tuple[str, ...], label: str) -> None:
    actual = tuple(inspect.signature(function).parameters)
    if actual != expected:
        raise RuntimeError(
            "Unsupported SGLang interface for the Ascend 0.5.18 GDN patch: "
            f"{label}{actual}, expected {expected}."
        )


def _causal_conv1d(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Indirection kept small so the exact native call is unit-testable."""

    return torch.ops.npu.causal_conv1d(*args, **kwargs)


def _is_linear_target_verify(forward_batch: Any) -> bool:
    spec_info = getattr(forward_batch, "spec_info", None)
    return bool(
        forward_batch.forward_mode.is_target_verify()
        and spec_info is not None
        and getattr(spec_info, "topk", None) == 1
        and getattr(spec_info, "ragged_verify_layout", None) is None
    )


def _sequential_target_verify(
    backend: Any,
    layer: Any,
    forward_batch: Any,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Execute a fixed-width verify chain through the one-token decode ABI."""

    if not isinstance(mixed_qkv, torch.Tensor):
        raise TypeError("Ascend GDN target verify requires tensor mixed_qkv")

    metadata = backend.forward_metadata
    cache_indices = metadata.mamba_cache_indices
    draft_token_num = int(forward_batch.spec_info.draft_token_num)
    num_token_padding = mixed_qkv.shape[0]

    if (
        not backend.graph_mode
        and forward_batch.num_token_non_padded_cpu != num_token_padding
    ):
        non_padded = forward_batch.num_token_non_padded_cpu
        mixed_qkv = mixed_qkv[:non_padded]
        a = a[:non_padded]
        b = b[:non_padded]

    batch_size = cache_indices.shape[0]
    expected_tokens = batch_size * draft_token_num
    if mixed_qkv.shape[0] != expected_tokens:
        raise RuntimeError(
            "Ascend sequential GDN verify requires a dense top-k=1 chain: "
            f"got {mixed_qkv.shape[0]} tokens for batch={batch_size}, "
            f"draft_token_num={draft_token_num}."
        )

    mixed_by_request = mixed_qkv.reshape(batch_size, draft_token_num, -1)
    a_by_request = a.reshape(batch_size, draft_token_num, *a.shape[1:])
    b_by_request = b.reshape(batch_size, draft_token_num, *b.shape[1:])

    layer_cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    intermediate_ssm = layer_cache.intermediate_ssm
    intermediate_conv = layer_cache.intermediate_conv_window[0]
    if intermediate_ssm is None:
        raise RuntimeError(
            "Ascend sequential GDN verify requires SGLang's speculative SSM cache"
        )

    if intermediate_ssm.shape[1] < draft_token_num:
        raise RuntimeError("speculative SSM cache is smaller than draft_token_num")
    if intermediate_conv.shape[1] < draft_token_num:
        raise RuntimeError("speculative conv cache is smaller than draft_token_num")

    # Fixed-width target metadata uses [0, D, 2D, ...].  Dividing by D creates
    # the exact [0, 1, 2, ...] metadata used by ordinary one-token decode and
    # avoids allocating another device arange inside graph capture.
    one_token_query_start = metadata.query_start_loc // draft_token_num
    safe_cache_indices = cache_indices.to(torch.int64).clamp(min=0)
    conv_window = layer.conv_weights.shape[-1] - 1

    # Speculative NPU pools deliberately use layouts that the native multi-token
    # kernels consume as raw storage: conv has D-1 extra rows and temporal is a
    # transposed view.  Ordinary decode instead expects compact contiguous
    # [B, K-1, C] conv and [B, HV, K, V] SSM pools.  Materializing logical views
    # here is essential: passing the speculative pool directly would make the
    # Triton recurrent kernel reinterpret V-major storage as K-major.
    work_conv = conv_states.index_select(0, safe_cache_indices)[
        :, -conv_window:, :
    ].contiguous()
    work_ssm = ssm_states.index_select(0, safe_cache_indices).contiguous()
    work_cache_indices = torch.arange(
        batch_size,
        dtype=cache_indices.dtype,
        device=cache_indices.device,
    )
    outputs = []

    for step in range(draft_token_num):
        step_mixed = _causal_conv1d(
            mixed_by_request[:, step].contiguous(),
            backend._get_conv_weights_t(layer),
            conv_states=work_conv,
            bias=layer.bias,
            query_start_loc=one_token_query_start,
            cache_indices=work_cache_indices,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

        query, key, value = torch.split(
            step_mixed,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        query = query.view(1, batch_size, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, batch_size, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, batch_size, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = backend.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a_by_request[:, step].contiguous(),
            b=b_by_request[:, step].contiguous(),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=work_ssm,
            cache_indices=work_cache_indices,
            query_start_loc=one_token_query_start,
        )

        # The decode operators have now crossed exactly the same BF16/FP32
        # persistence boundary as normal generation.  Copy the logical K,V
        # snapshot back through SGLang's speculative cache view; commit later
        # handles the inverse logical-to-physical layout conversion.
        intermediate_ssm[:batch_size, step].copy_(work_ssm)
        conv_snapshot = work_conv.transpose(-1, -2)
        intermediate_conv[:batch_size, step].copy_(conv_snapshot)

        outputs.append(
            core_attn_out.reshape(
                batch_size,
                layer.num_v_heads,
                layer.head_v_dim,
            )
        )

    output = torch.stack(outputs, dim=1).reshape(
        expected_tokens,
        layer.num_v_heads,
        layer.head_v_dim,
    )
    if not backend.graph_mode and output.shape[0] < num_token_padding:
        output = torch.cat(
            [
                output,
                output.new_zeros(
                    num_token_padding - output.shape[0],
                    *output.shape[1:],
                ),
            ],
            dim=0,
        )

    setattr(backend, _ACTIVE_FLAG, True)
    return output


def _restore_selected_snapshots(
    hybrid_backend: Any,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None,
    mamba_steps_to_track: torch.Tensor | None,
) -> None:
    """Commit accepted sequential snapshots for every local GDN layer."""

    linear_backend = hybrid_backend.linear_attn_backend
    request_number = last_correct_step_indices.shape[0]
    state_indices = linear_backend.forward_metadata.mamba_cache_indices[
        :request_number
    ].to(torch.int64)
    source_indices = torch.arange(
        request_number,
        dtype=torch.int64,
        device=state_indices.device,
    )
    steps = last_correct_step_indices.to(torch.int64)

    caches = linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
    ssm_states = caches.temporal
    intermediate_ssm = caches.intermediate_ssm
    conv_states = caches.conv[0]
    intermediate_conv = caches.intermediate_conv_window[0]
    if intermediate_ssm is None:
        raise RuntimeError("speculative SSM cache disappeared before MTP commit")

    conv_window = intermediate_conv.shape[-1]
    valid_main = steps >= 0
    main_sources = source_indices[valid_main]
    main_steps = steps[valid_main]
    main_destinations = state_indices[valid_main]
    if main_destinations.numel() > 0:
        ssm_states[:, main_destinations] = intermediate_ssm[:, main_sources, main_steps]
        conv_states[:, main_destinations, -conv_window:, :] = intermediate_conv[
            :, main_sources, main_steps
        ].transpose(-1, -2)

    if mamba_track_indices is None:
        return
    if mamba_steps_to_track is None:
        raise RuntimeError("mamba track indices require per-request track steps")

    track_indices = mamba_track_indices.to(torch.int64)
    track_steps = mamba_steps_to_track.to(torch.int64)
    track_mask = track_steps >= 0
    valid_sources = source_indices[track_mask]
    valid_steps = track_steps[track_mask]
    valid_destinations = track_indices[track_mask]
    if valid_destinations.numel() == 0:
        return

    ssm_states[:, valid_destinations] = intermediate_ssm[:, valid_sources, valid_steps]
    conv_states[:, valid_destinations, -conv_window:, :] = intermediate_conv[
        :, valid_sources, valid_steps
    ].transpose(-1, -2)


def patch_gdn_target_verify() -> bool:
    """Install the lossless linear-chain verify and snapshot commit wrappers."""

    try:
        gdn_module = importlib.import_module(_GDN_MODULE)
        hybrid_module = importlib.import_module(_HYBRID_MODULE)
        gdn_class = getattr(gdn_module, _GDN_CLASS)
        hybrid_class = getattr(hybrid_module, _HYBRID_CLASS)
    except (AttributeError, ImportError):
        logger.debug("SGLang Ascend GDN backends are unavailable", exc_info=True)
        return False

    original_forward = gdn_class.forward_extend
    original_commit = hybrid_class.update_mamba_state_after_mtp_verify
    if getattr(original_forward, _FORWARD_MARKER, False):
        return True

    _require_signature(original_forward, _FORWARD_PARAMETERS, "forward_extend")
    _require_signature(
        original_commit,
        _COMMIT_PARAMETERS,
        "update_mamba_state_after_mtp_verify",
    )

    @wraps(original_forward)
    def forward_extend(
        self: Any,
        layer: Any,
        forward_batch: Any,
        mixed_qkv: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not _is_linear_target_verify(forward_batch):
            return original_forward(
                self,
                layer,
                forward_batch,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        return _sequential_target_verify(
            self,
            layer,
            forward_batch,
            mixed_qkv,
            a,
            b,
        )

    @wraps(original_commit)
    def update_mamba_state_after_mtp_verify(
        self: Any,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: torch.Tensor | None,
        mamba_steps_to_track: torch.Tensor | None,
        model: Any,
        req_pool_indices: torch.Tensor | None = None,
    ) -> None:
        linear_backend = self.linear_attn_backend
        if not getattr(linear_backend, _ACTIVE_FLAG, False):
            return original_commit(
                self,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
                model,
                req_pool_indices,
            )
        try:
            _restore_selected_snapshots(
                self,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
            )
        finally:
            setattr(linear_backend, _ACTIVE_FLAG, False)

    setattr(forward_extend, _FORWARD_MARKER, True)
    setattr(forward_extend, "_sglang_fl_original", original_forward)
    setattr(update_mamba_state_after_mtp_verify, _COMMIT_MARKER, True)
    setattr(update_mamba_state_after_mtp_verify, "_sglang_fl_original", original_commit)
    gdn_class.forward_extend = forward_extend
    hybrid_class.update_mamba_state_after_mtp_verify = (
        update_mamba_state_after_mtp_verify
    )
    logger.info("Enabled lossless sequential GDN target verification on Ascend")
    return True
