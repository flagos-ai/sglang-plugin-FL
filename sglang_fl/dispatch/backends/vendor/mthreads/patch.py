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

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)

_patches_applied = False


def _patch_pp_send_recv_order() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    except Exception as e:
        logger.warning("MUSA PP send/recv order patch skipped: %s", e)
        return

    import torch

    orig_fn = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors

    @wraps(orig_fn)
    def pp_send_recv_and_preprocess_output_tensors_with_musa_order(
        self,
        next_first_rank_mb_id,
        next_mb_id,
        mbs,
        mb_metadata,
        last_rank_comm_queue,
        pp_outputs,
    ):
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            if mbs[next_mb_id] is None or mbs[next_mb_id].forward_mode.is_prebuilt():
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if (self.pp_rank % 2) == 0:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = (
        pp_send_recv_and_preprocess_output_tensors_with_musa_order
    )
    logger.info("MUSA PP send/recv ordering patch applied")


def _patch_pp_launch_batch_add_sync() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    except Exception as e:
        logger.warning("MUSA PP launch sync patch skipped: %s", e)
        return

    orig_fn = SchedulerPPMixin._pp_launch_batch

    @wraps(orig_fn)
    def pp_launch_batch_with_forward_stream_sync(self, *args, **kwargs):
        result, event = orig_fn(self, *args, **kwargs)
        self.forward_stream.synchronize()
        return result, event

    SchedulerPPMixin._pp_launch_batch = pp_launch_batch_with_forward_stream_sync
    logger.info("MUSA PP launch forward_stream sync patch applied")


def _patch_qwen3_5_model() -> None:
    """Adapt sglang to the bare Qwen3_5MoeForCausalLM entry class.

    Upstream sglang (as of 2026-08) registers only the wrapper classes
    Qwen3_5MoeForConditionalGeneration / Qwen3_5ForConditionalGeneration.
    Models with ``architectures=["Qwen3_5MoeForCausalLM"]`` (e.g. the
    Qwen3.8-2.4T-A95B-FP8) hit two bugs on the bare path:

      1. Checkpoint keys carry a "model." prefix while the bare path's
         params_dict has none -> KeyError on load_weights.
      2. The bare path builds no lm_head / logits_processor on the last
         PP rank -> AttributeError ('Tensor' has no 'next_token_logits').

    The same fixes are being submitted upstream to sgl-project/sglang;
    this patch self-skips once the loaded sglang already contains them.
    """
    import inspect

    try:
        import sglang.srt.models.qwen3_5 as qwen3_5
    except Exception as e:
        logger.warning("MUSA qwen3_5 model patch skipped: %s", e)
        return

    # Self-skip once upstream sglang carries the fixes.
    try:
        if "logits_processor" in inspect.getsource(qwen3_5.Qwen3_5ForCausalLM):
            logger.info("MUSA qwen3_5 model patch skipped (upstream already fixed)")
            return
    except (OSError, TypeError):
        pass

    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.server_args import get_global_server_args

    MoeCLM = qwen3_5.Qwen3_5MoeForCausalLM
    CLM = qwen3_5.Qwen3_5ForCausalLM

    # --- 1) bare path: build lm_head + logits_processor on prefix == "" ---
    # (verbatim-equivalent to the upstream fix in Qwen3_5ForCausalLM.__init__,
    #  applied after super().__init__ so it runs for both CLM and MoeCLM)
    orig_init = MoeCLM.__init__

    @wraps(orig_init)
    def init_with_lm_head(self, config, quant_config=None, prefix=""):
        orig_init(self, config, quant_config, prefix)
        if prefix == "":
            if self.pp_group.is_last_rank:
                if (
                    self.config.tie_word_embeddings
                    and self.pp_group.world_size == 1
                ):
                    self.lm_head = self.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        quant_config=quant_config,
                        prefix="lm_head",
                        use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    )
            else:
                self.lm_head = PPMissingLayer()
            self.logits_processor = LogitsProcessor(config)

    MoeCLM.__init__ = init_with_lm_head

    # --- 2) bare path: strip "model." prefix from checkpoint keys ---
    # Equivalent to the upstream strip_model_prefix fix: the "language_model"
    # -> "model." replace is unconditional (it is also inside orig_load's own
    # loop), the "model." prefix strip is gated on params_dict lacking any
    # "model." keys so the wrapped path (prefix="model") is untouched.
    orig_load = MoeCLM.load_weights

    @wraps(orig_load)
    def load_weights_with_strip(self, weights):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        strip = not any(k.startswith("model.") for k in params_dict)
        if strip:

            def _strip(weights):
                for name, loaded_weight in weights:
                    if "language_model" in name:
                        name = name.replace(r"model.language_model.", r"model.")
                    if name.startswith("model."):
                        name = name[len("model.") :]
                    yield name, loaded_weight

            weights = _strip(weights)
        return orig_load(self, weights)

    MoeCLM.load_weights = load_weights_with_strip

    # --- 3) bare path: run logits_processor on forward when present ---
    # Mirrors the upstream fix's placement: the logits branch runs only on the
    # last PP rank (orig_forward returns PPProxyTensors on other ranks, and
    # logits_processor exists on every rank after patch 1, so an ungated
    # hasattr check would feed PPProxyTensors to LogitsProcessor and crash).
    orig_forward = CLM.forward

    @wraps(orig_forward)
    def forward_with_logits(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
    ):
        hidden_states = orig_forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
            input_deepstack_embeds,
        )
        if hasattr(self, "logits_processor") and self.pp_group.is_last_rank:
            # The upstream fix passes aux_hidden_states here; on the bare path
            # (no deepstack heads) it is always [].
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, []
            )
        return hidden_states

    CLM.forward = forward_with_logits

    # --- 4) register the bare entry class ---
    if MoeCLM not in qwen3_5.EntryClass:
        qwen3_5.EntryClass.append(MoeCLM)

    logger.info("MUSA qwen3_5 model patch applied (Qwen3_5MoeForCausalLM)")


def apply_musa_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return

    _patch_pp_send_recv_order()
    _patch_pp_launch_batch_add_sync()
    _patch_qwen3_5_model()
    _patches_applied = True
    logger.info("All MUSA PP patches applied successfully")


apply_musa_patches()
