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

"""Keep flagtree's notion of the current XPU in sync with torch's.

Problem
-------
flagtree's ``XPUDriver`` tracks the active XPU in the ``TRITON_XPU_DEVICE``
environment variable instead of asking torch
(``triton/backends/xpu/driver.py``, marked ``FLAGTREE XPU SYNC MARK``)::

    self.get_current_device = self._get_current_xpu_device   # reads the env var
    self.set_current_device = self._set_current_xpu_device   # writes the env var

which overrides upstream ``GPUDriver`` (``triton/backends/driver.py``)::

    self.get_current_device = torch.cuda.current_device
    self.set_current_device = torch.cuda.set_device

``torch.cuda.set_device(n)`` never writes that env var, so any process running on
device index != 0 has flagtree believing it is on device 0. Every triton launch
then resolves its stream from that wrong index (``triton/runtime/jit.py``::
``device = get_current_device(); stream = get_current_stream(device)``), so the
kernel is submitted to **device 0's stream**.

Eager tolerates this. CUDA-graph capture does not: the recording stream belongs
to device n, and submitting to a non-recording stream either

* dies with ``xpuLaunchKernel(...) -> Unknown error(err_code: -900)`` when
  ``CUDA_LAUNCH_BLOCKING=1``, or
* **silently records an EMPTY graph** otherwise, so replay is a no-op and the
  results are wrong with no error at all (torch warns "The CUDA Graph is empty
  ... captured on wrong device or stream").

With TP > 1 this hits every rank whose device index is non-zero. It masquerades
as "a broken operator", because the op reported is merely whichever triton kernel
that rank captures first -- blacklisting it just promotes the next one.

Native kunlunxin triton 3.0.0 keeps the upstream torch-based accessors and is not
affected.

Why not just set the env var
----------------------------
A shell-level ``export TRITON_XPU_DEVICE=n`` cannot work for TP > 1: it is one
process-wide value while each rank needs its own. Exporting ``0`` leaves rank1
broken; exporting ``1`` breaks rank0 instead. Setting it per-process would work,
but requires every launch site to know its rank; rebinding the accessor does not.

What this does
--------------
Rebinds the accessors to consult torch, falling back to the env var only while
torch is still uninitialised (flagtree's standalone smoke path, which is what the
override was added for). ``torch.cuda.*`` is redirected onto the XPU runtime by
torch_xmlir on this stack ("SYMBOL_REWRITE torch success"), so this does **not**
pull in an NVIDIA driver.

Load ordering: this runs from ``_apply_vendor_patches()`` inside
``load_plugin()``, which ``run_scheduler_process()`` calls first thing -- i.e.
inside every spawned TP worker, before any model code. That is before
``ModelRunner.init_torch_distributed()`` calls ``set_device(gpu_id)``, which is
fine: what gets installed is a *function* evaluated at each launch, so it picks up
the device as of then.

Set ``SGLANG_FL_TRITON_XPU_DEVICE_SYNC=0`` to disable.
"""

import logging
import os

logger = logging.getLogger(__name__)

_applied = False


def _current_xpu_device():
    """Active XPU index: ask torch first, fall back to the env var."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    return int(
        os.environ.get("TRITON_XPU_DEVICE", os.environ.get("XPU_VISIBLE_DEVICE", "0"))
    )


def _set_xpu_device(device):
    """Move torch AND keep the env var in sync, so both views agree."""
    os.environ["TRITON_XPU_DEVICE"] = str(device)
    try:
        import torch

        torch.cuda.set_device(int(device))
    except Exception:
        pass


def patch_triton_xpu_device_sync():
    """Rebind flagtree's XPU device accessors onto torch. Idempotent."""
    global _applied
    if _applied:
        return
    if os.environ.get("SGLANG_FL_TRITON_XPU_DEVICE_SYNC", "1").strip() in (
        "0",
        "false",
        "False",
    ):
        logger.info(
            "triton XPU device sync disabled (SGLANG_FL_TRITON_XPU_DEVICE_SYNC=0)"
        )
        return

    try:
        from triton.backends.xpu.driver import XPUDriver
    except Exception as e:
        # Native triton without the xpu backend module, or triton absent.
        logger.debug("triton XPU device sync skipped: %s", e)
        return

    # Only flagtree has the env-var-based accessors. Their absence means this
    # build already tracks torch, so there is nothing to fix.
    if not hasattr(XPUDriver, "_get_current_xpu_device"):
        logger.debug(
            "triton XPU device sync skipped: build already tracks torch device"
        )
        return

    XPUDriver._get_current_xpu_device = staticmethod(_current_xpu_device)
    XPUDriver._set_current_xpu_device = staticmethod(_set_xpu_device)

    # If a driver instance already exists, ``XPUDriver.__init__`` has copied the
    # old bound methods onto it, so patching the class alone would not take
    # effect. Rebind on the live instance too -- but never read the lazy
    # ``.active`` property, since that would force driver creation as a side
    # effect of applying a patch.
    #
    # Note ``from triton.runtime import driver`` yields the *DriverConfig
    # instance* (triton/runtime/driver.py ends with ``driver = DriverConfig()``),
    # not the module -- so resolve both spellings.
    try:
        from triton.runtime import driver as _drv

        cfg = getattr(_drv, "driver", _drv)  # module -> .driver; instance -> itself
        inst = getattr(cfg, "_active", None)
        if inst is None:
            inst = getattr(cfg, "_default", None)
        if inst is not None:
            inst.get_current_device = _current_xpu_device
            inst.set_current_device = _set_xpu_device
            logger.debug("triton XPU device sync: rebound on live driver instance")
    except Exception as e:
        logger.debug("triton XPU device sync: instance rebind skipped: %s", e)

    _applied = True
    logger.info(
        "patched flagtree XPUDriver device accessors -> torch "
        "(fixes cuda graph capture on non-zero device index)"
    )
