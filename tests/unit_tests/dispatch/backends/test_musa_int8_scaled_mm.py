# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real MUSA numerical tests, independent of the operator's intermediates."""

import pytest
import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA hardware",
)


def test_quantization_config_uses_the_public_platform_hook():
    import sglang  # noqa: F401 - initialize the framework before its OOT platform
    from sglang_fl.platform import PlatformFL
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = "mthreads"
    cls = platform.get_quantization_config("compressed-tensors")
    assert issubclass(cls, CompressedTensorsConfig)
    assert cls is not CompressedTensorsConfig
    assert cls.get_min_capability() == 0
    assert CompressedTensorsConfig.get_min_capability() == 70
    assert platform.get_quantization_config("fp8") is None
    platform._vendor_name = "nvidia"
    assert platform.get_quantization_config("compressed-tensors") is None


def test_vendor_config_validates_supported_scheme_before_loading():
    from copy import deepcopy
    from sglang_fl.dispatch.backends.vendor.mthreads.quantization import (
        get_musa_compressed_tensors_config,
    )

    cls = get_musa_compressed_tensors_config()
    config = {
        "format": "int-quantized",
        "ignore": ["lm_head"],
        "config_groups": {
            "w8a8": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }
    instance = cls.from_config(config)
    assert isinstance(instance, cls)
    assert instance.ignore == ["lm_head"]
    fused = deepcopy(config)
    fused["config_groups"]["w8a8"]["targets"] = [
        "model.self_attn.index_q_proj",
        "model.self_attn.index_k_proj",
    ]
    fused["packed_modules_mapping"] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "index_qkv_proj": ["index_q_proj", "index_k_proj", "index_v_proj"]
    }
    instance = cls.from_config(fused)
    layer = torch.nn.Module()
    layer.v_head_size = 0
    assert instance.get_scheme_dict(layer, "model.self_attn.index_qkv_proj") is not None
    assert len(instance.packed_modules_mapping["index_qkv_proj"]) == 3
    layer.v_head_size = 128
    with pytest.raises(ValueError, match="matching target"):
        instance.get_scheme_dict(layer, "model.self_attn.index_qkv_proj")
    invalid = deepcopy(config)
    invalid["config_groups"]["w8a8"]["input_activations"].update(
        dynamic=False, strategy="tensor"
    )
    with pytest.raises(NotImplementedError, match="dynamic token"):
        cls.from_config(invalid)


@pytest.mark.parametrize("m,n,k", [(1, 13, 31), (16, 768, 6144), (64, 6144, 384)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_scaled_mm_matches_independent_integer_reference(m, n, k, with_bias):
    from sglang_fl.dispatch.backends.flagos.impl.int8_scaled_mm import (
        int8_scaled_mm_flagos,
    )

    generator = torch.Generator().manual_seed(771)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=generator)
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=generator)
    sa = torch.full((m, 1), 2.0**-8)
    sb = torch.full((1, n), 2.0**-7)
    bias = torch.linspace(-1, 1, n).bfloat16() if with_bias else None
    reference = (a.int() @ b.t().int()).float() * sa * sb
    if bias is not None:
        with pytest.raises(NotImplementedError, match="support bias"):
            int8_scaled_mm_flagos(
                a.to("musa"),
                b.to("musa").t(),
                sa.to("musa"),
                sb.to("musa"),
                out_dtype=torch.bfloat16,
                bias=bias.to("musa"),
            )
        return
    output = int8_scaled_mm_flagos(
        a.to("musa"),
        b.to("musa").t(),
        sa.to("musa"),
        sb.to("musa"),
        out_dtype=torch.bfloat16,
        bias=None if bias is None else bias.to("musa"),
    )
    torch.musa.synchronize()
    torch.testing.assert_close(output.cpu(), reference.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("width", [384, 6144])
def test_upstream_quantization_rounding_zero_and_sum(width):
    from sglang.kernels.ops.quantization import int8_kernel
    from sglang_fl.dispatch.backends.vendor.mthreads.patches.int8_rounding import (
        install_int8_rounding_patch,
    )

    install_int8_rounding_patch(int8_kernel)
    x = torch.zeros(4, width, dtype=torch.bfloat16)
    x[0, :8] = torch.tensor([127, -127, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
    x[1] = torch.linspace(-5, 5, width)
    x[2, :4] = torch.tensor([0.25, -0.25, 0.75, -0.75])
    f = x.float()
    maximum = f.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scaled = f * (127 / maximum)
    reference = (scaled.sign() * torch.floor(scaled.abs() + 0.5)).to(torch.int8)
    quantized, scales, sums = int8_kernel.per_token_quant_int8(
        x.to("musa"),
        cal_sum=True,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(quantized.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), maximum / 127, rtol=1e-6, atol=0)
    torch.testing.assert_close(sums.cpu(), f.sum(-1).bfloat16(), rtol=0, atol=0)
