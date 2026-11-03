# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select framework-owned dynamic INT8 schemes on MUSA.

SGLang v0.5.18 (71de97b264b04dcd514cf904003028aefe9775c8) rejects
compressed-tensors INT8 MoE outside NPU and selects a CUDA-only dense kernel.
These selectors are installed on a vendor subclass returned by the public
platform configuration hook. Remove them when upstream schemes support MUSA.
They do not replace weight loaders, numerical kernels, or non-MUSA schemes.
"""

from functools import wraps
import inspect
import logging

logger = logging.getLogger(__name__)
MARKER = "_sglang_fl_musa_w8a8"


def _musa_selected():
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    return info is not None and info.vendor_name == "mthreads"


def _factories():
    from ..quantization import build_w8a8_schemes

    return build_w8a8_schemes()


def _validate_contract(value):
    weight = value["weights"]
    activation = value["input_activations"]
    strategy = getattr(weight.strategy, "value", weight.strategy)
    if strategy != "channel" or not weight.symmetric or not activation.symmetric:
        raise NotImplementedError(
            "MUSA dynamic INT8 requires symmetric token activations and channel weights"
        )
    if value.get("format") not in (None, "int-quantized"):
        raise NotImplementedError("MUSA dynamic INT8 requires int-quantized storage")
    return weight, activation


def install_w8a8_patches(upstream=None, is_musa=None, factories=None):
    """Install two selection wrappers atomically; failed selection never retries."""
    if upstream is None:
        from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors

        upstream = compressed_tensors
    is_musa = is_musa or _musa_selected
    factories = factories or _factories
    cls = upstream.CompressedTensorsConfig
    required = {
        "get_linear_scheme": ("self", "layer", "layer_name"),
        "get_moe_scheme": ("self", "layer", "layer_name"),
        "get_scheme_dict": ("self", "layer", "layer_name"),
        "_is_dynamic_token_w8a8": ("self", "weight_quant", "input_quant"),
        "_add_fused_moe_to_target_scheme_map": ("self",),
    }
    for name, parameters in required.items():
        function = getattr(cls, name, None)
        if not callable(function):
            raise RuntimeError(f"MUSA W8A8 requires SGLang API {name}")
        actual = inspect.signature(function).parameters
        for parameter in parameters:
            if parameter not in actual:
                raise RuntimeError(f"MUSA W8A8 API {name} is missing {parameter}")
        if set(actual) != set(parameters):
            raise RuntimeError(f"MUSA W8A8 API {name} has an unsupported signature")
    original_linear = cls.get_linear_scheme
    original_moe = cls.get_moe_scheme
    marked = [getattr(f, MARKER, False) for f in (original_linear, original_moe)]
    if all(marked):
        return False
    if any(marked):
        raise RuntimeError("MUSA W8A8 found an incomplete previous patch installation")

    def dynamic(self, value):
        if value is None:
            return False
        weight, activation = value.get("weights"), value.get("input_activations")
        if weight is None or activation is None:
            return False
        for quant in (weight, activation):
            quant_type = getattr(quant.type, "value", quant.type)
            if quant_type != "int":
                return False
        return self._is_dynamic_token_w8a8(weight, activation)

    @wraps(original_linear)
    def linear(self, layer, layer_name=None):
        if is_musa():
            value = self.get_scheme_dict(layer, layer_name)
            if dynamic(self, value):
                weight, activation = _validate_contract(value)
                if self.sparsity_scheme_map:
                    raise NotImplementedError(
                        "MUSA dynamic INT8 does not support sparse weights"
                    )
                return factories().linear(weight, activation)
        return original_linear(self, layer=layer, layer_name=layer_name)

    @wraps(original_moe)
    def moe(self, layer, layer_name=None):
        if is_musa() and layer_name is not None:
            self._add_fused_moe_to_target_scheme_map()
            values = [
                self.get_scheme_dict(layer, f"{layer_name}.0.{projection}")
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
            if any(dynamic(self, value) for value in values):
                if not all(value == values[0] for value in values):
                    raise ValueError(
                        "All MoE projections need the same quantization scheme"
                    )
                weight, activation = _validate_contract(values[0])
                return factories().moe(weight, activation)
        return original_moe(self, layer=layer, layer_name=layer_name)

    for function in (linear, moe):
        setattr(function, MARKER, True)
    try:
        cls.get_linear_scheme = linear
        cls.get_moe_scheme = moe
    except BaseException:
        cls.get_linear_scheme = original_linear
        cls.get_moe_scheme = original_moe
        raise
    logger.info("MUSA dynamic INT8 scheme selection installed")
    return True
