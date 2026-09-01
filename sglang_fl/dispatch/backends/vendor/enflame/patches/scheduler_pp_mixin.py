from collections import deque
from typing import List, Optional

import torch

from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata

from sglang.srt.managers.scheduler import Scheduler

def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(self.cur_batch, pp_proxy_tensors)
                # GCU limits
                major, _ = torch.gcu.get_device_capability("gcu")
                if major == 3:
                    torch.cuda.synchronize() # add sync, before next batch
                set_time_batch(
                    self.cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, self.cur_batch)
                            ),
                        )
                    )
        return result, event

def patch_scheduler_pp_mixin():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    SCHEDULER_PP_MIXIN = "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin"
    HookRegistry.register(f"{SCHEDULER_PP_MIXIN}._pp_launch_batch", _pp_launch_batch, HookType.REPLACE)