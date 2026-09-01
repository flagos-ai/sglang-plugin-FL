from __future__ import annotations

import torch

from functools import partial

from sglang.srt.environ import envs
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import get_bool_env_var
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)

# ---------------------------------------------------------------------------
# Replacement: _capture_graph
# ---------------------------------------------------------------------------

def _capture_graph(self, graph, pool, stream, run_once_fn):
    if self.model_runner.server_args.debug_cuda_graph:
        assert (
            envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get()
        ), "Breakable CUDA graph is not enabled in debug mode"

    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=self.model_runner.server_args.enable_memory_saver
        and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
    )

    if envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get():
        if memory_saver_adapter.enabled:
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )
        graph_ctx = BreakableCUDAGraphCapture
    else:
        graph_ctx = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else self.device_module.graph
        )

    if self.model_runner.server_args.debug_cuda_graph:
        captured_fn = eager_on_graph(True)(run_once_fn)
    else:
        captured_fn = run_once_fn

    with graph_ctx(graph, pool=pool, stream=stream):
        out = captured_fn()
    return out

# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------

_CUDA_GRAPH_CLASS = "sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner"

def patch_cuda_graph_runner():
    """Register GCU-compatible overrides for CudaGraphRunner."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        f"{_CUDA_GRAPH_CLASS}._capture_graph",
        _capture_graph,
        HookType.REPLACE,
    )
