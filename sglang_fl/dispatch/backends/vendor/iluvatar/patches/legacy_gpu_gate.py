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

"""Iluvatar: don't apply sglang's NVIDIA sm75 floor to a corex device.

`maybe_downgrade_dtype_for_legacy_gpu()` reads
`torch.cuda.get_device_capability()` and, below sm80, switches the model dtype
to float16 — then raises when the minor version is under 5:

    if torch.cuda.get_device_capability()[1] < 5:
        raise RuntimeError("SGLang only supports sm75 and above.")

On iluvatar the capability reads (7, 1): corex reports the CUDA
*compatibility* identity its torch build targets, not an NVIDIA SM level, so
the floor rejects every iluvatar device and every serve dies during
`ModelRunner.load_model()`.

Rebind the module global to a corex version that keeps the float16 downgrade
(BI-V150 has no bfloat16, so the requested dtype has to fall) and drops the
floor. `model_runner` imports the name directly, so rebinding the defining
module before that import runs is what takes effect; the already-imported case
is handled as well, for an import order we did not anticipate.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_TARGET_MODULE = "sglang.srt.model_executor.model_runner_components.load_model_utils"
_MODEL_RUNNER_MODULE = "sglang.srt.model_executor.model_runner"
_FN_NAME = "maybe_downgrade_dtype_for_legacy_gpu"

_patched = False


def _downgrade_for_iluvatar(*, server_args, model_config) -> None:
    """sglang's downgrade without the NVIDIA sm floor."""
    import torch

    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80 on corex. Use float16 due to lack of "
            "bfloat16 support."
        )
        from sglang.srt.runtime_context import get_context

        get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
        model_config.dtype = torch.float16


def patch_legacy_gpu_gate() -> None:
    """Rebind the sm-floored legacy-GPU downgrade to the corex version."""
    global _patched
    if _patched:
        return
    _patched = True

    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    original = getattr(module, _FN_NAME)
    setattr(module, _FN_NAME, _downgrade_for_iluvatar)

    # model_runner does `from ... import maybe_downgrade_dtype_for_legacy_gpu`,
    # which copies the reference — refresh it if that import already happened.
    model_runner = sys.modules.get(_MODEL_RUNNER_MODULE)
    if model_runner is not None and getattr(model_runner, _FN_NAME, None) is original:
        setattr(model_runner, _FN_NAME, _downgrade_for_iluvatar)

    logger.info("iluvatar legacy-GPU dtype gate patched (sm75 floor dropped)")
