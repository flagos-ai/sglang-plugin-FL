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

"""Query stream capture state through torch on PPU.

SGLang 0.5.18's breakable CUDA graph (prefill graph) asks cuda-python's
``cudaStreamGetCaptureInfo`` whether a stream is capturing. On PPU (HGGC) that
call returns ``cudaErrorInvalidValue`` for every stream, idle or capturing,
so prefill graph capture aborts. ``torch.cuda.is_current_stream_capturing()``
reports correctly, which is the same fallback SGLang uses on HIP/XPU.
"""

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_MODULE = (
    "sglang.srt.model_executor.runner_backend_utils."
    "breakable_cuda_graph.breakable_cuda_graph"
)


def _is_stream_capturing(stream) -> bool:
    with torch.cuda.stream(stream):
        return torch.cuda.is_current_stream_capturing()


def patch_graph_capture_status() -> None:
    try:
        mod = importlib.import_module(_MODULE)
    except ImportError:
        return
    if not hasattr(mod, "_is_stream_capturing"):
        logger.warning("thead: %s._is_stream_capturing not found", _MODULE)
        return
    # Callers resolve this name from module globals at call time.
    mod._is_stream_capturing = _is_stream_capturing
    logger.info("thead: breakable CUDA graph capture status via torch")
