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

"""Unit coverage for the SGLang 0.5.18 Ascend vision fallback."""

import sys
from types import ModuleType

import pytest
import torch


def _install_fake_vision(monkeypatch, *, incompatible_signature=False):
    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    layers = ModuleType("sglang.srt.layers")
    attention = ModuleType("sglang.srt.layers.attention")
    vision = ModuleType("sglang.srt.layers.attention.vision")

    class VisionAscendAttention:
        if incompatible_signature:

            def forward(self, renamed_parameter):
                return renamed_parameter

        else:

            def forward(
                self,
                q,
                k,
                v,
                cu_seqlens,
                bsz,
                seq_len,
                softmax_scale=None,
                forward_metadata=None,
                attention_mask=None,
                **kwargs,
            ):
                self.fused_calls.append(
                    {
                        "q": q,
                        "k": k,
                        "v": v,
                        "cu_seqlens": cu_seqlens,
                        "bsz": bsz,
                        "seq_len": seq_len,
                        "softmax_scale": softmax_scale,
                        "forward_metadata": forward_metadata,
                        "attention_mask": attention_mask,
                        **kwargs,
                    }
                )
                return q

    vision.VisionAscendAttention = VisionAscendAttention
    attention.vision = vision
    layers.attention = attention
    srt.layers = layers
    sglang.srt = srt

    for module in (sglang, srt, layers, attention, vision):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return VisionAscendAttention


def _instance(cls):
    instance = cls()
    instance.sdpa_calls = []
    instance.fused_calls = []

    def sdpa_fallback(**kwargs):
        instance.sdpa_calls.append(kwargs)
        return "sdpa"

    instance.sdpa_fallback = sdpa_fallback
    return instance


def test_qwen36_head_is_padded_through_fused_attention(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.vision import (
        patch_vision_ascend_attention,
    )

    cls = _install_fake_vision(monkeypatch)
    patch_vision_ascend_attention()
    instance = _instance(cls)
    q = torch.randn(96, 4, 72)
    k = torch.randn(96, 2, 72)
    v = torch.randn(96, 2, 72)

    result = instance.forward(
        q,
        k,
        v,
        [0, 32, 96],
        1,
        96,
        forward_metadata="metadata",
        output_ws="workspace",
    )

    assert result.shape == q.shape
    assert torch.equal(result, q)
    assert instance.sdpa_calls == []
    assert len(instance.fused_calls) == 1
    call = instance.fused_calls[0]
    assert call["q"].shape == (96, 4, 128)
    assert call["k"].shape == (96, 2, 128)
    assert call["v"].shape == (96, 2, 128)
    assert call["softmax_scale"] == pytest.approx(72**-0.5)
    assert call["cu_seqlens"] == [0, 32, 96]
    assert call["forward_metadata"] == "metadata"
    assert call["output_ws"] == "workspace"


def test_other_unsupported_head_uses_existing_sdpa_fallback(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.vision import (
        patch_vision_ascend_attention,
    )

    cls = _install_fake_vision(monkeypatch)
    patch_vision_ascend_attention()
    instance = _instance(cls)
    q = torch.randn(8, 2, 80)

    result = instance.forward(q, q, q, [0, 8], 1, 8, softmax_scale=0.25)

    assert result == "sdpa"
    assert instance.fused_calls == []
    assert instance.sdpa_calls[0]["softmax_scale"] == 0.25


def test_qwen36_masked_attention_keeps_unpadded_sdpa_fallback(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.vision import (
        patch_vision_ascend_attention,
    )

    cls = _install_fake_vision(monkeypatch)
    patch_vision_ascend_attention()
    instance = _instance(cls)
    q = torch.randn(8, 2, 72)

    result = instance.forward(q, q, q, [0, 8], 1, 8, attention_mask="mask")

    assert result == "sdpa"
    assert instance.fused_calls == []
    assert instance.sdpa_calls[0]["q"] is q
    assert instance.sdpa_calls[0]["attention_mask"] == "mask"


def test_aligned_head_keeps_fused_attention_and_patch_is_idempotent(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.vision import (
        patch_vision_ascend_attention,
    )

    cls = _install_fake_vision(monkeypatch)
    patch_vision_ascend_attention()
    patched = cls.forward
    patch_vision_ascend_attention()
    instance = _instance(cls)
    q = torch.randn(64, 4, 128)

    result = instance.forward(q, q, q, [0, 64], 1, 64)

    assert cls.forward is patched
    assert result is q
    assert instance.fused_calls[0]["q"] is q
    assert instance.sdpa_calls == []


def test_patch_rejects_unknown_upstream_signature(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches.vision import (
        patch_vision_ascend_attention,
    )

    _install_fake_vision(monkeypatch, incompatible_signature=True)

    with pytest.raises(RuntimeError, match="Unsupported SGLang VisionAscendAttention"):
        patch_vision_ascend_attention()
