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

"""Cambricon MLU vendor patches for sglang — entrypoint.

Auto-imported by ``sglang_fl._apply_vendor_patches()`` when FlagGems'
DeviceDetector reports cambricon hardware. The module body runs at import,
so each patch applies once per process at plugin load time.

torch_mlu's CUDA-migration layer masquerades as CUDA but is stale relative to
real CUDA, so code that assumes CUDA semantics trips on MLU. These patches fix
the torch_mlu facade (not the sglang wheel) so unmodified sglang code works:

1. ``_MLUDeviceProperties.is_integrated`` — sglang's
   ``get_available_gpu_memory()`` reads ``props.is_integrated`` off
   ``torch.cuda.get_device_properties()``; torch_mlu's properties class lacks
   it. MLU590 has dedicated HBM, so False is correct. Class-level injection is
   required: many sglang modules bind ``get_available_gpu_memory`` via
   ``from sglang.srt.utils import ...`` before the plugin loads, so a getattr
   guard on the sglang function would not reach every call site.
2. ``torch.cuda.get_device_capability`` — MLU590 is bf16-capable but torch_mlu
   reports ``(5,0)`` through the CUDA alias; sglang arch gates read that as
   sm50 legacy, force a float16 downgrade and raise "SGLang only supports sm75
   and above". Presenting an sm80-class capability keeps dtype untouched and
   clears the gate (stays below (10,0) so sm100-only paths stay off).
3. SDPA ``enable_gqa`` — sglang's torch_native backend calls
   ``F.scaled_dot_product_attention(..., enable_gqa=True)`` for GQA models.
   torch_mlu's fused SDPA kernels reject the gathered KV shape and torch falls
   back to the mlu MATH backend, whose signature predates the ``enable_gqa``
   kwarg (TypeError); it also cannot expand KV heads itself. Expand the KV
   heads here (as torch's own GQA math path does) and drop ``enable_gqa``.
"""

import logging

logger = logging.getLogger(__name__)

_patches_applied = False

_CAPABILITY = (8, 0)  # see module docstring concern 2


def _patch_device_properties_is_integrated() -> None:
    from torch_mlu._MLUC import _MLUDeviceProperties

    # pybind class attribute assignment is allowed (verified on neuware4.7.2 /
    # torch_mlu 1.33.1+torch2.11.0). Instances resolve is_integrated through
    # the class dict.
    _MLUDeviceProperties.is_integrated = False


def _patch_cuda_capability_facade() -> None:
    import torch

    fn = getattr(torch.cuda, "get_device_capability", None)
    if fn is None or not fn.__module__.startswith("torch_mlu"):
        logger.warning("cambricon capability facade skipped: %r", fn)
        return

    def _mlu_device_capability(device=None):
        return _CAPABILITY

    torch.cuda.get_device_capability = _mlu_device_capability


def _patch_sdpa_gqa_fallback() -> None:
    import torch
    import torch.nn.functional as _F

    _orig = _F.scaled_dot_product_attention

    def _wrapped(query, key, value, *args, **kwargs):
        if kwargs.get("enable_gqa") and getattr(query, "device", None) is not None:
            if query.device.type == "mlu" and query.dim() >= 3:
                nq = query.shape[-3]
                nkv = key.shape[-3]
                if nkv != nq:
                    rep = nq // nkv
                    key = key.repeat_interleave(rep, -3)
                    value = value.repeat_interleave(rep, -3)
                kwargs["enable_gqa"] = False
        return _orig(query, key, value, *args, **kwargs)

    _F.scaled_dot_product_attention = _wrapped
    # torch_native_backend binds the function at module import
    # (``from torch.nn.functional import scaled_dot_product_attention``), which
    # happens at model-runner init — after load_plugins() in every process, so
    # the binding picks up the wrapper. Rebinding here additionally covers a
    # process where the backend was imported before this patch ran.
    try:
        import sglang.srt.layers.attention.torch_native_backend as _tnb

        if getattr(_tnb, "scaled_dot_product_attention", None) is not _wrapped:
            _tnb.scaled_dot_product_attention = _wrapped
    except Exception as e:  # pragma: no cover - best-effort rebind
        logger.info("cambricon sdpa rebind skipped: %r", e)


def apply_cambricon_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    _patch_device_properties_is_integrated()
    _patch_cuda_capability_facade()
    _patch_sdpa_gqa_fallback()
    logger.info("cambricon patches applied")


apply_cambricon_patches()
