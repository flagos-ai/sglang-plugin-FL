"""Tune SGLang triton attention kernels for MetaX shared-memory limits.

C550 has 64 KiB shared memory per block. SGLang's default decode config
(BLOCK=32, num_stages=2) needs ~73 KiB for head_dim 256 and fails with
``OutOfResources``. The values below follow the MetaX-tuned SGLang build.

SGLang >= 0.5.16 moved the kernels to ``sglang.kernels.ops.attention`` and
changed the decode launcher signature, so the decode stage-1 kernel is wrapped
instead of the launcher there. Older SGLang keeps the launcher replacement.
"""

from __future__ import annotations

import importlib
import logging

import triton

logger = logging.getLogger(__name__)

_MODULES = (
    # SGLang >= 0.5.16
    (
        "sglang.kernels.ops.attention.decode_attention",
        "sglang.kernels.ops.attention.extend_attention",
    ),
    # SGLang <= 0.5.15
    (
        "sglang.srt.layers.attention.triton_ops.decode_attention",
        "sglang.srt.layers.attention.triton_ops.extend_attention",
    ),
)
_SCENARIO = "flashattn-fwd"
_patched = False


def _import_modules():
    for decode_name, extend_name in _MODULES:
        try:
            return importlib.import_module(decode_name), importlib.import_module(
                extend_name
            )
        except ImportError:
            continue
    raise ImportError("SGLang triton decode/extend attention modules not found")


class _ScenarioKernel:
    """Proxy for a triton kernel that adds ``scenario`` to every launch.

    The MetaX-tuned SGLang launches attention kernels with
    ``scenario="flashattn-fwd"``, which selects FlagTree/mxcc compile options
    tuned for attention. ``overrides`` replaces launch arguments (e.g. the
    decode tile sizes). Attribute access is forwarded to the real kernel.
    """

    def __init__(self, kernel, scenario: str, overrides=None, override_if=None):
        self._kernel = kernel
        self._scenario = scenario
        self._overrides = overrides or {}
        self._override_if = override_if

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def run(*args, **kwargs):
            kwargs.setdefault("scenario", self._scenario)
            if self._overrides and (
                self._override_if is None or self._override_if(kwargs)
            ):
                kwargs.update(self._overrides)
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def _wrap(module, name, **kwargs) -> None:
    kernel = getattr(module, name, None)
    if kernel is not None and not isinstance(kernel, _ScenarioKernel):
        setattr(module, name, _ScenarioKernel(kernel, _SCENARIO, **kwargs))


def _make_decode_grouped_att_m_fwd_legacy(mod):
    """SGLang <= 0.5.15: replace the grouped decode launcher."""
    kernel = mod._fwd_grouped_kernel_stage1
    min_block_kv = mod._MIN_BLOCK_KV

    def _decode_grouped_att_m_fwd_metax(
        q,
        k_buffer,
        v_buffer,
        att_out,
        att_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len=-1,
    ):
        BLOCK = 16
        Lk = k_buffer.shape[-1]
        Lv = v_buffer.shape[-1]

        if Lk == 576:
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        elif Lk == 288:
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lk)
            BLOCK_DPE = 0
        BLOCK_DV = triton.next_power_of_2(Lv)

        batch, head_num = q.shape[0], q.shape[1]
        kv_group_num = q.shape[1] // k_buffer.shape[1]

        BLOCK_H = 16
        grid = (
            batch,
            triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
            max_kv_splits,
        )

        kernel[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale_withk,
            kv_indptr,
            kv_indices,
            att_out,
            att_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=min_block_kv,
            logit_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
            num_warps=4,
            num_stages=1,
            Lk=Lk,
            Lv=Lv,
            scenario=_SCENARIO,
        )

    return _decode_grouped_att_m_fwd_metax


def _get_block_sizes_for_extend_attention_metax(Lq: int, Lv: int):
    if Lq == 576:
        BLOCK_DMODEL, BLOCK_DPE = 512, 64
    elif Lq == 288:
        BLOCK_DMODEL, BLOCK_DPE = 256, 32
    elif Lq == 192:
        BLOCK_DMODEL, BLOCK_DPE = 128, 64
    else:
        BLOCK_DMODEL, BLOCK_DPE = triton.next_power_of_2(Lq), 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Sized for the C550 64 KiB shared-memory limit.
    if Lq <= 128:
        BLOCK_M, BLOCK_N = (64, 64)
    elif Lq <= 256:
        BLOCK_M, BLOCK_N = (64, 32)
    else:
        BLOCK_M, BLOCK_N = (16, 16)
    num_warps = 4

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


def _is_normal_extend(mode) -> bool:
    """Extend batch that is neither decode/idle nor speculative."""
    fn = getattr(mode, "is_extend_without_speculative", None)
    if fn is not None:  # SGLang >= 0.5.16
        return fn()
    return not (
        mode.is_decode_or_idle() or mode.is_target_verify() or mode.is_draft_extend()
    )


def _patch_triton_backend_kv_splits() -> None:
    """Match the MetaX-tuned SGLang decode kv split policy.

    That build hard-codes ``max_kv_splits = 16`` and, after a normal extend
    batch, sets it to 16 for long (>256 tokens) and 2 for short extends.
    """
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    orig_init = TritonAttnBackend.__init__
    orig_meta = TritonAttnBackend.init_forward_metadata

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.max_kv_splits = 16

    def init_forward_metadata(self, forward_batch):
        if _is_normal_extend(forward_batch.forward_mode):
            lens = forward_batch.extend_seq_lens_cpu
            if lens:
                self.max_kv_splits = 16 if max(lens) > 256 else 2
        return orig_meta(self, forward_batch)

    TritonAttnBackend.__init__ = __init__
    TritonAttnBackend.init_forward_metadata = init_forward_metadata


def patch_triton_attention() -> None:
    """Replace triton decode/extend attention launch configs with MetaX-tuned ones."""
    global _patched
    if _patched:
        return

    decode, extend = _import_modules()

    if decode.__name__.startswith("sglang.kernels."):
        # Override the GQA decode tile; MLA decode keeps SGLang's own tuning.
        _wrap(
            decode,
            "_fwd_grouped_kernel_stage1",
            overrides={"BLOCK_N": 16, "num_stages": 1, "num_warps": 4},
            override_if=lambda kw: not kw.get("HAS_MLA", False),
        )
    else:
        decode._decode_grouped_att_m_fwd = _make_decode_grouped_att_m_fwd_legacy(decode)
    extend._get_block_sizes_for_extend_attention = (
        _get_block_sizes_for_extend_attention_metax
    )
    _wrap(extend, "_fwd_kernel")
    _wrap(extend, "_fwd_kernel_unified")
    _patch_triton_backend_kv_splits()

    _patched = True
    logger.info(
        "patched triton attention for MetaX (%s): decode BLOCK=16 num_stages=1, "
        "extend block sizes for 64KiB smem + scenario=flashattn-fwd, max_kv_splits=16/2",
        decode.__name__,
    )
