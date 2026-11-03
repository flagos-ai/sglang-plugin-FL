# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Thin scheme adapters; all numerical work stays in public operator APIs."""

from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize=1)
def build_w8a8_schemes():
    from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsMoEScheme,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8MoEMethod

    from sglang_fl.dispatch import resolve_op

    class MusaLinear(CompressedTensorsW8A8Int8):
        def __init__(self, weight, activation):
            super().__init__(weight.strategy, False, activation.symmetric)
            # Resolve before loading weights or launching quantization. A missing
            # public API is a configuration error, never an execution fallback.
            resolve_op("int8_scaled_mm")

        @classmethod
        def get_min_capability(cls):
            return 0

        def apply_weights(self, layer, x, bias):
            if bias is not None:
                raise NotImplementedError(
                    "MUSA dynamic INT8 linear does not yet support bias"
                )
            kernel = resolve_op("int8_scaled_mm")
            x_quantized, x_scale = per_token_quant_int8(x)
            return kernel(
                x_quantized,
                layer.weight,
                x_scale,
                layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
            )

    class MusaMoE(CompressedTensorsMoEScheme):
        def __init__(self, weight, activation):
            self.weight_quant = weight
            self.input_quant = activation
            self.static_input_scales = False
            self._method = W8A8Int8MoEMethod(None)

        @classmethod
        def get_min_capability(cls):
            return 0

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            return self._method.create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

        def process_weights_after_loading(self, layer):
            return self._method.process_weights_after_loading(layer)

        def create_moe_runner(self, layer, moe_runner_config):
            self.moe_runner_config = moe_runner_config
            return self._method.create_moe_runner(layer, moe_runner_config)

        def apply_weights(self, layer, dispatch_output):
            return self._method.apply(layer, dispatch_output)

    return SimpleNamespace(linear=MusaLinear, moe=MusaMoE)


@lru_cache(maxsize=1)
def get_musa_compressed_tensors_config():
    """Return a vendor configuration through the public platform hook."""
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    from .patches import require_supported_sglang
    from .patches.w8a8 import _validate_contract, install_w8a8_patches

    require_supported_sglang()

    class MusaCompressedTensorsConfig(CompressedTensorsConfig):
        @classmethod
        def get_min_capability(cls):
            # MUSA capability numbers are not CUDA SM versions. Scheme-level
            # validation below constrains support instead of spoofing hardware.
            return 0

        def get_scheme_dict(self, layer, layer_name=None):
            # The native model declares a three-way packed mapping even when
            # its index value projection has zero width. Match only the two
            # real checkpoint components, retaining the original ignore rules.
            if (
                layer_name is not None
                and layer_name.endswith(".index_qkv_proj")
                and getattr(layer, "v_head_size", None) == 0
            ):
                from copy import copy

                local = copy(self)
                local.packed_modules_mapping = {
                    "index_qkv_proj": ["index_q_proj", "index_k_proj"],
                    **{
                        k: v
                        for k, v in self.packed_modules_mapping.items()
                        if k != "index_qkv_proj"
                    },
                }
                return super(MusaCompressedTensorsConfig, local).get_scheme_dict(
                    layer, layer_name
                )
            return super().get_scheme_dict(layer, layer_name)

        @classmethod
        def from_config(cls, config):
            result = super().from_config(config)
            if result.sparsity_scheme_map:
                raise NotImplementedError("MUSA INT8 does not support sparse weights")
            for value in result.target_scheme_map.values():
                weight = value.get("weights")
                activation = value.get("input_activations")
                if (
                    weight is None
                    or activation is None
                    or not result._is_dynamic_token_w8a8(weight, activation)
                ):
                    raise NotImplementedError(
                        "MUSA compressed-tensors requires dynamic token W8A8"
                    )
                if any(
                    getattr(q.type, "value", q.type) != "int"
                    for q in (weight, activation)
                ):
                    raise NotImplementedError("MUSA compressed-tensors requires INT8")
                _validate_contract(value)
            return result

    # Install selectors only on the new vendor class. The upstream class and
    # other vendors retain their original methods and capability requirements.
    install_w8a8_patches(
        SimpleNamespace(CompressedTensorsConfig=MusaCompressedTensorsConfig),
        is_musa=lambda: True,
    )
    return MusaCompressedTensorsConfig
