"""Tune SGLang triton attention kernels for MetaX shared-memory limits.

C550 has 64 KiB shared memory per block. SGLang's default decode config
(BLOCK=32, num_stages=2) needs ~73 KiB for head_dim 256 and fails with
``OutOfResources``. The values below follow the MetaX-tuned SGLang build.
"""

from __future__ import annotations

import importlib
import logging

import triton

logger = logging.getLogger(__name__)

_DECODE_MODULE = "sglang.srt.layers.attention.triton_ops.decode_attention"
_EXTEND_MODULE = "sglang.srt.layers.attention.triton_ops.extend_attention"
_patched = False


def _make_decode_grouped_att_m_fwd(mod):
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
            scenario="flashattn-fwd",
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


class _ScenarioKernel:
    """Proxy for a triton kernel that adds ``scenario`` to every launch.

    The MetaX-tuned SGLang launches the extend kernel with
    ``scenario="flashattn-fwd"``, which selects FlagTree/mxcc compile options
    tuned for attention. Attribute access is forwarded to the real kernel.
    """

    def __init__(self, kernel, scenario: str):
        self._kernel = kernel
        self._scenario = scenario

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def run(*args, **kwargs):
            kwargs.setdefault("scenario", self._scenario)
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self._kernel, name)


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
        mode = forward_batch.forward_mode
        if not (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend()
        ):
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

    decode = importlib.import_module(_DECODE_MODULE)
    extend = importlib.import_module(_EXTEND_MODULE)

    decode._decode_grouped_att_m_fwd = _make_decode_grouped_att_m_fwd(decode)
    extend._get_block_sizes_for_extend_attention = (
        _get_block_sizes_for_extend_attention_metax
    )
    if not isinstance(extend._fwd_kernel, _ScenarioKernel):
        extend._fwd_kernel = _ScenarioKernel(extend._fwd_kernel, "flashattn-fwd")
    _patch_triton_backend_kv_splits()

    _patched = True
    logger.info(
        "patched triton attention for MetaX: decode BLOCK=16 num_stages=1, "
        "extend block sizes for 64KiB smem + scenario=flashattn-fwd, max_kv_splits=16/2"
    )
