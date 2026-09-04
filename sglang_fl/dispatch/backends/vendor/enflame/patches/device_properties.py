"""GCU device-property compatibility for sglang's triton_load_watch hook.

torch_gcu aliases ``torch.cuda.*`` to gcu and reports ``is_available()``
True, but its ``torch_gcu._C._GcuDeviceProperties`` carries no
``is_integrated`` attribute. sglang 0.5.18 installs a triton
kernel_load_start hook (triton_load_watch) that, on every post-ready kernel
load, calls ``sglang/srt/utils/common.py get_available_gpu_memory``; on the
"cuda" branch it reads ``props.is_integrated`` unconditionally. On GCU that
read raises AttributeError (only RuntimeError is caught at the call site),
killing the scheduler at the first kernel load after serving starts.

GCU is a discrete accelerator with dedicated HBM — ``is_integrated=False``
is the correct value and selects the ``torch.cuda.mem_get_info()`` path,
which torch_gcu aliases to the working gcu implementation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_applied = False


def patch_gcu_device_properties() -> None:
    """Expose ``is_integrated`` on gcu device properties (idempotent)."""
    global _applied
    if _applied:
        return
    try:
        import torch_gcu

        torch_gcu._C._GcuDeviceProperties.is_integrated = False
        _applied = True
    except Exception as e:  # noqa: BLE001 - never let a compat patch take sglang down
        logger.warning("gcu device-properties patch failed: %r", e)


patch_gcu_device_properties()
