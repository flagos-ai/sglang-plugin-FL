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

"""Keep MATE's paged decode split capacity dynamic under graph replay.

The supported vendor backend inherits capture initialization from SGLang's
FlashAttentionBackend. Its capture-time sequence lengths are ones, not the
eventual replay lengths. MATE uses max_seq_len_k on the host to select its
maximum split fanout, so capture must use the allocated page-table capacity.
Actual per-request lengths remain device inputs to the metadata producer.
"""

import importlib
import inspect
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_METHOD = "init_forward_metadata_capture_cuda_graph"
_PATCH_MARKER = "_sglang_fl_musa_fa3_graph_capacity"
_PARAMETERS = (
    "self",
    "bs",
    "num_tokens",
    "req_pool_indices",
    "seq_lens",
    "encoder_lens",
    "forward_mode",
    "spec_info",
)


def _patch_backend(backend_cls):
    original = getattr(backend_cls, _METHOD, None)
    if getattr(original, _PATCH_MARKER, False):
        return True
    # An override may already implement this fix or a different metadata ABI.
    # Do not replace it, or mutate the parent shared by other device backends.
    if original is None or _METHOD in vars(backend_cls):
        logger.warning("MUSA FA3 graph capacity patch skipped: backend override/ABI")
        return False
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    if tuple(parameters) != _PARAMETERS or any(
        p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values()
    ):
        logger.warning("MUSA FA3 graph capacity patch skipped: capture signature")
        return False

    @wraps(original)
    def capture_with_paged_capacity(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        result = original(
            self,
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        metadata = self.forward_metadata
        if (
            not self.use_mla
            and forward_mode.is_decode()
            and spec_info is None
            and metadata.page_table is not None
        ):
            metadata.max_seq_len_k = metadata.page_table.shape[1] * self.page_size
        return result

    setattr(capture_with_paged_capacity, _PATCH_MARKER, True)
    setattr(backend_cls, _METHOD, capture_with_paged_capacity)
    logger.info("MUSA FA3 paged decode graph capacity patch applied")
    return True


def apply_musa_fa3_graph_metadata_patch():
    if os.getenv("SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY", "1").lower() in (
        "0",
        "false",
        "off",
    ):
        return False
    try:
        module = importlib.import_module(
            "sglang.srt.hardware_backend.musa.attention.flashattention_backend"
        )
        backend_cls = module.MusaFlashAttentionBackend
    except (ImportError, AttributeError) as exc:
        logger.warning("MUSA FA3 graph capacity patch unavailable: %s", exc)
        return False
    return _patch_backend(backend_cls)
