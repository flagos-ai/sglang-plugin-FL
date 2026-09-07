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

"""MetaX vision guard: preset the CUDA-only flashinfer cudnn symbol to None.

``sglang/srt/layers/attention/vision.py`` imports
``flashinfer.prefill.cudnn_batch_prefill_with_kv_cache`` unconditionally in its
``if _is_cuda:`` branch, which a CUDA-alias runtime (MetaX) executes; the
MetaX flashinfer build strips the symbol, so the bare import would raise.
Presetting the attribute to ``None`` on the module makes the import bind
``None`` instead — the symbol is only ever *called* on the flashinfer_cudnn
vision-attention path, which MetaX never selects.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_vision_cudnn_guard() -> None:
    """Fill the missing cudnn_batch_prefill_with_kv_cache symbol with None."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import flashinfer.prefill as _prefill
    except Exception as e:  # pragma: no cover - guard only
        logger.warning("Metax vision cudnn guard skipped: %s", e)
        return

    if not hasattr(_prefill, "cudnn_batch_prefill_with_kv_cache"):
        _prefill.cudnn_batch_prefill_with_kv_cache = None
        logger.info("Metax vision cudnn guard applied")
