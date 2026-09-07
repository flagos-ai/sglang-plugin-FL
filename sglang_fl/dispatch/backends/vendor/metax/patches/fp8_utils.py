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

"""MetaX fp8 guard: preset the CUDA-only flashinfer bmm_fp8 symbol to None.

``sglang/srt/layers/quantization/fp8_utils.py`` imports ``flashinfer.bmm_fp8``
in its ``if _is_cuda:`` branch, which a CUDA-alias runtime (MetaX) executes;
the MetaX flashinfer build strips the symbol, so the bare import would raise.
Presetting the attribute to ``None`` makes the import bind ``None`` instead —
the op is only reached on fp8-quantized model paths, which MetaX does not run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_fp8_bmm_guard() -> None:
    """Fill the missing bmm_fp8 symbol with None."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import flashinfer
    except Exception as e:  # pragma: no cover - guard only
        logger.warning("Metax fp8 bmm guard skipped: %s", e)
        return

    if not hasattr(flashinfer, "bmm_fp8"):
        flashinfer.bmm_fp8 = None
        logger.info("Metax fp8 bmm guard applied")
