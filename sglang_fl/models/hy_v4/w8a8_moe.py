# SPDX-License-Identifier: Apache-2.0
"""HYV4 compressed-tensors W8A8 MoE bridge for SGLang 0.5.18 CUDA.

The host provides the INT8 Triton runner but gates this checkpoint scheme to
NPU. Preserve the checkpoint's channel scales and use the existing INT8 runner,
including its routed-expert SwiGLU clamp. No FP8 or weight dequantization.
"""

import logging

import torch
from compressed_tensors.quantization import QuantizationStrategy
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


class HYV4W8A8Int8MoE(CompressedTensorsMoEScheme):
    def __init__(self, weight_quant, input_quant):
        self.weight_quant, self.input_quant = (weight_quant, input_quant)
        self.static_input_scales = not input_quant.dynamic
        if (
            weight_quant.strategy != QuantizationStrategy.CHANNEL
            or input_quant.strategy != QuantizationStrategy.TOKEN
            or self.static_input_scales
            or (not weight_quant.symmetric)
            or (not input_quant.symmetric)
        ):
            raise ValueError(
                "HYV4 INT8 MoE requires symmetric channel weights and dynamic token activations"
            )

    @classmethod
    def get_min_capability(cls):
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        if layer.w13_weight.dtype != torch.int8 or layer.w2_weight.dtype != torch.int8:
            raise ValueError("HYV4 W8A8 expert weights must remain INT8")

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply_weights(self, layer, dispatch_output):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return self.runner.run(
            dispatch_output,
            TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            ),
        )


def install_w8a8_moe():
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    if getattr(CompressedTensorsConfig, "_hy4_moe_registered", False):
        return
    previous = CompressedTensorsConfig.get_moe_scheme

    def get_moe_scheme(self, layer, layer_name):
        if getattr(self, "_hy4_model", False):
            self._add_fused_moe_to_target_scheme_map()
            names = [
                layer_name + suffix
                for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
            ]
            schemes = [self.get_scheme_dict(layer, name) for name in names]
            scheme = schemes[0]
            if scheme and all(item == scheme for item in schemes):
                weights, inputs = (
                    scheme.get("weights"),
                    scheme.get("input_activations"),
                )
                if self._is_dynamic_token_w8a8(weights, inputs):
                    return HYV4W8A8Int8MoE(weights, inputs)
        return previous(self, layer, layer_name)

    CompressedTensorsConfig.get_moe_scheme = get_moe_scheme
    CompressedTensorsConfig._hy4_moe_registered = True
