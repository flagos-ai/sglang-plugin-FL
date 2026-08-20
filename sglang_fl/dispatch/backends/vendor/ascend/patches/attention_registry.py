"""Ascend/NPU patch for SGLang attention backend wrapper.

Ports the source edit from
``sglang/srt/layers/attention/attention_registry.py``: on NPU the CUDA-only
``KDAAttnBackend`` / ``LightningAttentionBackend`` modules fail to import, so
their imports must be deferred into the ``not is_npu()`` branch instead of
running at the top of the ``mambaish_config`` block.

``attn_backend_wrapper`` is a module-level function that ``model_runner``
imports by name (``from ... import attn_backend_wrapper``). We therefore rebind
it both on ``attention_registry`` and, if already imported, on ``model_runner``.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def patch_attn_backend_wrapper() -> None:
    try:
        from sglang.srt.layers.attention import attention_registry as ar
    except Exception as e:
        logger.warning("Ascend attn_backend_wrapper patch skipped: %s", e)
        return

    def attn_backend_wrapper(runner, full_attn_backend):
        """
        Wrapper for special models like hybrid GDN, so we don't
        need to change the code of the original attention backend.
        """
        assert not (
            runner.hybrid_gdn_config is not None and runner.use_mla_backend
        ), "hybrid_gdn can only be used with non-MLA models."

        if cfg := runner.mambaish_config:
            from sglang.srt.configs.linear_attn_model_registry import (
                get_linear_attn_config,
                import_backend_class,
            )
            from sglang.srt.layers.attention.fla.utils import check_environments
            from sglang.srt.layers.attention.linear.utils import (
                initialize_linear_attn_config,
            )
            from sglang.srt.utils import is_blackwell, is_npu

            if not is_npu():
                from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                    HybridLinearAttnBackend,
                    Mamba2AttnBackend,
                )
                from sglang.srt.layers.attention.linear.gdn_backend import (
                    GDNAttnBackend,
                )
                from sglang.srt.layers.attention.linear.kda_backend import (
                    KDAAttnBackend,
                )
                from sglang.srt.layers.attention.linear.lightning_backend import (
                    LightningAttentionBackend,
                )
            else:
                from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
                    AscendGDNAttnBackend as GDNAttnBackend,
                )
                from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                    AscendHybridLinearAttnBackend as HybridLinearAttnBackend,
                )
                from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                    AscendMamba2AttnBackend as Mamba2AttnBackend,
                )

            check_environments()
            initialize_linear_attn_config(runner.server_args)
            if runner.hybrid_gdn_config is not None:
                if is_blackwell():
                    assert (
                        runner.server_args.attention_backend == "triton"
                        or runner.server_args.attention_backend == "trtllm_mha"
                        or runner.server_args.attention_backend == "fa4"
                        or runner.server_args.attention_backend == "flashinfer"
                    ), "triton, trtllm_mha, fa4, or flashinfer backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend to specify the backend."
                if is_npu():
                    assert (
                        runner.server_args.attention_backend == "ascend"
                    ), "ascend backend is the only supported backend on NPU for hybrid GDN models, use --attention-backend ascend to specify the backend."
                logger.info(
                    "Using hybrid linear attention backend for hybrid GDN models."
                )
                linear_attn_backend = GDNAttnBackend(runner)
            elif runner.mamba2_config is not None:
                linear_attn_backend = Mamba2AttnBackend(runner)
            elif runner.kimi_linear_config is not None:
                linear_attn_backend = KDAAttnBackend(runner)
            elif runner.hybrid_lightning_config is not None:
                linear_attn_backend = LightningAttentionBackend(runner)
            else:
                spec_result = get_linear_attn_config(runner.model_config.hf_config)
                if spec_result is not None:
                    spec, _ = spec_result
                    BackendClass = import_backend_class(spec.backend_class_name)
                    linear_attn_backend = BackendClass(runner)
                else:
                    raise ValueError(
                        "Expected hybrid GDN or NemotronH models, but got unknown model. "
                        "If this is a custom hybrid model, use register_linear_attn_model() "
                        "from sglang.srt.configs.linear_attn_model_registry."
                    )
            full_attn_layers = cfg.full_attention_layer_ids
            return HybridLinearAttnBackend(
                full_attn_backend, linear_attn_backend, full_attn_layers
            )

        return full_attn_backend

    ar.attn_backend_wrapper = attn_backend_wrapper

    # model_runner imports the function by name at its module top level; if it
    # is already imported, rebind its reference too so the patch takes effect.
    mr = sys.modules.get("sglang.srt.model_executor.model_runner")
    if mr is not None and hasattr(mr, "attn_backend_wrapper"):
        mr.attn_backend_wrapper = attn_backend_wrapper

    logger.info("Ascend attn_backend_wrapper patch applied")
