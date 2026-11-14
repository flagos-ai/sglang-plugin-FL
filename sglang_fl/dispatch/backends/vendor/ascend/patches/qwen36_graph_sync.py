"""Avoid two unnecessary NPU/CPU transfers in text-only mRoPE graph decode."""

import functools
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)
_MARKER = "_sglang_fl_qwen36_graph_sync"


def patch_qwen36_graph_sync() -> bool:
    if os.getenv("SGLANG_FL_QWEN36_GRAPH_SYNC", "0") != "1":
        return False
    try:
        from sglang.srt.hardware_backend.npu.graph_runner import (
            npu_graph_runner as graph_module,
        )
        from sglang.srt.model_executor import forward_batch_info as batch_module

        batch_class = batch_module.ForwardBatch
        graph_class = graph_module.NPUGraphRunner
        original_mrope = batch_class._compute_mrope_positions
        original_replay = graph_class.replay
        if getattr(original_replay, _MARKER, False):
            return False
        mrope_source = inspect.getsource(original_mrope)
        replay_source = inspect.getsource(original_replay)
        if not (
            "self.seq_lens_cpu[batch_idx] - 1" in mrope_source
            and "ret.positions = clamp_position(batch.seq_lens)"
            in inspect.getsource(batch_class.init_new)
            and "forward_batch.seq_lens.cpu().tolist()" in replay_source
            and "thread.join()" in replay_source
        ):
            return False
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Qwen3.6 graph sync patch skipped: %s", exc)
        return False

    @functools.wraps(original_mrope)
    def compute_mrope(self, model_runner, batch):
        cpu_lens = self.seq_lens_cpu
        if (
            self.forward_mode.is_decode()
            and self.spec_info is None
            and self.spec_algorithm.is_none()
            and batch.dllm_config is None
            and batch.multimodal_inputs is not None
            and all(item is None for item in batch.multimodal_inputs)
            and isinstance(cpu_lens, torch.Tensor)
            and cpu_lens.device.type == "cpu"
            and isinstance(self.positions, torch.Tensor)
            and cpu_lens.numel() == self.positions.numel()
            and bool((cpu_lens > 0).all())
            and self.positions.device.type == "npu"
            and self.positions.dtype == torch.int64
        ):
            self.mrope_positions = self.positions.unsqueeze(0).repeat(3, 1)
            if not getattr(compute_mrope, "_logged", False):
                logger.info("Text-only mRoPE NPU fast path active")
                compute_mrope._logged = True
            return
        return original_mrope(self, model_runner, batch)

    @functools.wraps(original_replay)
    def replay(
        self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None
    ):
        cpu_lens = forward_batch.seq_lens_cpu
        if not (
            not skip_attn_backend_init
            and forward_batch.forward_mode.is_decode()
            and forward_batch.spec_info is None
            and forward_batch.spec_algorithm.is_none()
            and getattr(self.model_runner, "model_is_mrope", False)
            and isinstance(cpu_lens, torch.Tensor)
            and cpu_lens.device.type == "cpu"
            and cpu_lens.numel() == forward_batch.batch_size
            and bool((cpu_lens > 0).all())
        ):
            return original_replay(
                self, forward_batch, skip_attn_backend_init, pp_proxy_tensors
            )

        self.replay_prepare(forward_batch, pp_proxy_tensors)
        if not getattr(replay, "_logged", False):
            logger.info("NPU graph replay CPU length mirror active")
            replay._logged = True
        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        if not graph_module.is_deepseek_nsa(self.model_runner.model_config.hf_config):
            seq_lens = cpu_lens.tolist() + [0] * (self.bs - self.raw_bs)
            # Preserve replay order while letting the CPU prepare the next batch.
            # The task handle must not be rewritten before the previous graph ends.
            previous_done = getattr(self, "_sglang_fl_previous_graph_done", None)

            def update_after_previous():
                if previous_done is not None:
                    previous_done.synchronize()
                self._update_inputs(seq_lens)

            thread = graph_module.threading.Thread(target=update_after_previous)
            thread.start()
            self.graphs[self.bs].replay()
            done = torch.npu.Event()
            done.record(torch.npu.current_stream())
            thread.join()
            self._sglang_fl_previous_graph_done = done
        else:
            self.graphs[self.bs].replay()

        output = self.output_buffers[self.bs]
        if isinstance(output, graph_module.LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = output.full_logits[: self.raw_num_token]
            else:
                full_logits = None
                next_token_logits = output.next_token_logits[: self.raw_num_token]
            return graph_module.LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        assert isinstance(output, graph_module.PPProxyTensors)
        return graph_module.PPProxyTensors(
            {key: value[: self.bs] for key, value in output.tensors.items()}
        )

    batch_class._compute_mrope_positions = compute_mrope
    setattr(replay, _MARKER, True)
    graph_class.replay = replay
    logger.info("Enabled ordered Qwen3.6 graph sync patch")
    return True
