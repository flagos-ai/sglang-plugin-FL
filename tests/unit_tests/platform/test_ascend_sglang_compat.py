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

"""Regression tests for the independent SGLang 0.5.18 Ascend adaptation."""

from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml


def test_ascend_chunk_disables_wheel_final_state_by_default(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.fla import (
        chunk_gated_delta_rule_ascend,
    )

    kernel_module = ModuleType("sgl_kernel_npu.fla.chunk")
    calls = []
    expected = object()
    kernel_module.chunk_gated_delta_rule_npu = lambda **kwargs: (
        calls.append(kwargs) or expected
    )
    monkeypatch.setitem(sys.modules, kernel_module.__name__, kernel_module)

    tensors = [object() for _ in range(5)]
    assert chunk_gated_delta_rule_ascend(*tensors, scale=0.5) is expected
    assert calls[0]["output_final_state"] is False


def test_ascend_moe_delegates_the_0518_dispatch_output_unchanged() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        fused_moe_ascend,
    )

    calls = []
    expected = object()
    layer = object()
    dispatch_output = SimpleNamespace(
        # AscendTPDispatchOutput has these fields and deliberately has no
        # pre-0.5.18 `topk_output` aggregate.
        hidden_states=object(),
        topk_weights=object(),
        topk_ids=object(),
    )
    method = SimpleNamespace(
        forward_npu=lambda actual_layer, actual_dispatch: (
            calls.append((actual_dispatch, actual_layer)) or expected
        )
    )

    assert fused_moe_ascend(method, layer, dispatch_output) is expected
    assert calls == [(dispatch_output, layer)]


def test_ascend_normalization_delegates_native_0518_fallbacks() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.normalization import (
        gemma_rms_norm_ascend,
        rms_norm_ascend,
    )

    calls = []
    expected = object()
    x = object()
    residual = object()
    norm = SimpleNamespace(
        forward_npu=lambda actual_x, actual_residual: (
            calls.append((actual_x, actual_residual)) or expected
        )
    )

    assert rms_norm_ascend(norm, x, residual) is expected
    assert gemma_rms_norm_ascend(norm, x, None) is expected
    assert calls == [(x, residual), (x, None)]


def test_ascend_topk_delegates_native_0518_routing() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.impl.topk import topk_ascend

    calls = []
    expected = object()
    hidden_states = object()
    router_logits = object()
    non_padded = object()
    dispatch_info = object()
    topk = SimpleNamespace(
        forward_npu=lambda *args, **kwargs: calls.append((args, kwargs)) or expected
    )

    assert (
        topk_ascend(
            topk,
            hidden_states,
            router_logits,
            num_token_non_padded=non_padded,
            expert_location_dispatch_info=dispatch_info,
        )
        is expected
    )
    assert calls == [
        (
            (hidden_states, router_logits),
            {
                "num_token_non_padded": non_padded,
                "expert_location_dispatch_info": dispatch_info,
            },
        )
    ]


def test_ascend_qwen36_policy_uses_0518_native_contracts() -> None:
    config_path = (
        Path(__file__).parents[3] / "sglang_fl" / "dispatch" / "config" / "ascend.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backends = config["op_backends"]

    for op_name in ("topk", "gemma_rms_norm", "fused_moe", "chunk_gated_delta_rule"):
        assert backends[op_name][0] == "vendor"
    assert backends["silu_and_mul"][0] == "flagos"
    assert backends["mrotary_embedding"][0] == "flagos"
    assert "fused_recurrent_gated_delta_rule" not in backends
    assert "masked_scatter_" in config["flagos_blacklist"]
    assert "sum" not in config["flagos_blacklist"]


def test_ascend_entrypoints_do_not_export_removed_spec_v2_toggle() -> None:
    root = Path(__file__).parents[3]
    entrypoints = [
        root / "scripts" / "ascend" / "acceptance_common.sh",
        *sorted((root / "examples").glob("qwen3_6_*.py")),
    ]

    for entrypoint in entrypoints:
        assert "SGLANG_ENABLE_SPEC_V2" not in entrypoint.read_text(encoding="utf-8"), (
            f"{entrypoint.relative_to(root)} exports the toggle removed by SGLang 0.5.18"
        )


def test_ascend_mtp_correctness_uses_strict_semantic_gate() -> None:
    root = Path(__file__).parents[3]
    source = (root / "examples" / "qwen3_6_27b_mtp_inference.py").read_text(
        encoding="utf-8"
    )

    assert 'print("    FAIL: <100% semantic contract agreement")' in source
    assert '"    FAIL: speculative decode/target prefill mismatch "' in source
    assert 'print("    FAIL: stats not available")' in source
    assert 'print("    WARN: <90% exact match;' in source
    assert 'print("    SKIP: stats not available")' not in source


def test_ascend_mtp_semantic_contracts_reject_superficial_matches() -> None:
    root = Path(__file__).parents[3]
    namespace = runpy.run_path(root / "examples" / "qwen3_6_27b_mtp_inference.py")
    validate = namespace["_validate_output"]

    assert validate({"expected_number": 50}, "50")[0] is True
    assert validate({"expected_number": 50}, "There are not 50 but 51.")[0] is False
    assert validate({"expected_exact": ["paris"]}, "**Paris.**")[0] is True
    assert validate({"expected_exact": ["paris"]}, "Paris, France")[0] is False
    assert (
        validate(
            {"expected_number_sequence": [2, 3, 5, 7, 11]},
            "2, 3, 5, 7, 11",
        )[0]
        is True
    )
    assert (
        validate(
            {"expected_number_sequence": [2, 3, 5, 7, 11]},
            "The first 5 are 2, 3, 5, 7, 11",
        )[0]
        is False
    )

    recursive_spec = {
        "python_function": "factorial",
        "python_contract": "recursive",
    }
    assert (
        validate(
            recursive_spec,
            "def factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)",
        )[0]
        is True
    )
    assert (
        validate(recursive_spec, "def factorial(n):\n    return factorial(n - 1)")[0]
        is False
    )
    assert (
        validate(
            recursive_spec,
            "def factorial(n):\n    return 1 if n < 2 else 120",
        )[0]
        is False
    )

    palindrome_spec = {
        "python_function": "is_palindrome",
        "python_contract": "reverse_slice_comparison",
    }
    assert (
        validate(
            palindrome_spec,
            "def is_palindrome(s):\n    return s == s[::-1]",
        )[0]
        is True
    )
    assert (
        validate(palindrome_spec, "def is_palindrome(s):\n    return True")[0] is False
    )
    assert (
        validate(
            palindrome_spec,
            "def is_palindrome(s):\n    return s != 'abc'",
        )[0]
        is False
    )


def test_ascend_mtp_logprob_conformance_rejects_false_positives() -> None:
    root = Path(__file__).parents[3]
    namespace = runpy.run_path(root / "examples" / "qwen3_6_27b_mtp_inference.py")
    check = namespace["run_logprob_conformance"]
    check.__globals__["_text_prompt"] = lambda prompt: prompt

    class FakeEngine:
        def __init__(self, mode: str):
            self.mode = mode

        def generate(
            self,
            prompt=None,
            sampling_params=None,
            input_ids=None,
            **_kwargs,
        ):
            del sampling_params
            if prompt is not None:
                value = -0.03 if self.mode == "all_near_ties" else -0.1
                return {
                    "meta_info": {
                        "prompt_tokens": 3,
                        "input_token_logprobs": [
                            (-0.2, 1, None),
                            (-0.2, 2, None),
                            (-0.2, 3, None),
                        ],
                        "output_token_logprobs": [
                            (value, token_id, None) for token_id in range(100, 132)
                        ],
                        "output_top_logprobs": [
                            [(value, token_id, None), (-1.0, 999, None)]
                            for token_id in range(100, 132)
                        ],
                    }
                }

            assert input_ids is not None
            output_ids = input_ids[-32:]
            value = -0.03 if self.mode == "all_near_ties" else -0.1
            score_ids = list(output_ids)
            if self.mode == "token_mismatch":
                score_ids[0] += 1

            top_logprobs = []
            for token_id in output_ids:
                if self.mode == "nan_top":
                    top_logprobs.append(
                        [(float("nan"), token_id, None), (-1.0, 999, None)]
                    )
                elif self.mode == "all_near_ties":
                    top_logprobs.append([(0.0, 999, None), (value, token_id, None)])
                else:
                    top_logprobs.append([(value, token_id, None), (-1.0, 999, None)])

            return {
                "meta_info": {
                    "prompt_tokens": len(input_ids),
                    "input_token_logprobs": [
                        *[(-0.2, token_id, None) for token_id in input_ids[:-32]],
                        *[(value, token_id, None) for token_id in score_ids],
                    ],
                    "input_top_logprobs": [
                        *[[] for _ in input_ids[:-32]],
                        *top_logprobs,
                    ],
                }
            }

    good = check(FakeEngine("good"))
    assert good["passed"] is True
    assert good["scored_tokens"] == 64
    assert good["near_ties"] == 0

    token_mismatch = check(FakeEngine("token_mismatch"))
    assert token_mismatch["passed"] is False
    assert any("token-id mismatch" in error for error in token_mismatch["errors"])

    nan_top = check(FakeEngine("nan_top"))
    assert nan_top["passed"] is False
    assert any("non-finite top-logprob" in error for error in nan_top["errors"])

    systematic_second_best = check(FakeEngine("all_near_ties"))
    assert systematic_second_best["passed"] is False
    assert systematic_second_best["near_ties"] == 64


def test_ascend_mamba_state_update_disables_multibuffer(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches import mamba_state_update

    calls = []

    class FakeKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

        def run(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "result"

    kernel = FakeKernel()
    module = SimpleNamespace(move_cache_dynamic_last_kernel_h_block=kernel)
    monkeypatch.setattr(
        mamba_state_update.importlib,
        "import_module",
        lambda name: (
            module
            if name == mamba_state_update._KERNEL_MODULE
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    assert mamba_state_update.patch_mamba_state_update_multibuffer() is True
    patched_run = kernel.run
    assert mamba_state_update.patch_mamba_state_update_multibuffer() is True
    assert kernel.run is patched_run

    assert (
        kernel[(1,)](
            "payload",
            H_BLOCK_SIZE=2,
            BLOCK_V=128,
            BLOCK_K=128,
            multibuffer=True,
            num_warps=4,
        )
        == "result"
    )
    assert (
        kernel[(2,)](
            "other-shape",
            H_BLOCK_SIZE=1,
            BLOCK_V=128,
            BLOCK_K=128,
            multibuffer=True,
        )
        == "result"
    )
    assert calls == [
        (
            ("payload",),
            {
                "grid": (1,),
                "H_BLOCK_SIZE": 2,
                "BLOCK_V": 128,
                "BLOCK_K": 128,
                "multibuffer": False,
                "num_warps": 4,
            },
        ),
        (
            ("other-shape",),
            {
                "grid": (2,),
                "H_BLOCK_SIZE": 1,
                "BLOCK_V": 128,
                "BLOCK_K": 128,
                "multibuffer": True,
            },
        ),
    ]


def test_ascend_mamba_state_update_patch_is_optional(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches import mamba_state_update

    def missing(_name):
        raise ImportError("kernel wheel is not installed")

    monkeypatch.setattr(mamba_state_update.importlib, "import_module", missing)
    assert mamba_state_update.patch_mamba_state_update_multibuffer() is False


def test_ascend_gdn_target_verify_replays_decode_and_commits_snapshots(
    monkeypatch,
) -> None:
    import torch

    from sglang_fl.dispatch.backends.vendor.ascend.patches import gdn_target_verify

    batch_size = 2
    draft_token_num = 2
    channels = 3
    conv_window = 2
    slots = 4

    conv_all = torch.zeros(1, slots, draft_token_num + conv_window - 1, channels)
    conv_all[0, 1, -conv_window:] = torch.tensor(
        [[-2.0, -20.0, -200.0], [-1.0, -10.0, -100.0]]
    )
    conv_all[0, 2, -conv_window:] = torch.tensor(
        [[-4.0, -40.0, -400.0], [-3.0, -30.0, -300.0]]
    )
    # The speculative NPU pool is a non-contiguous logical [HV, K, V] view of
    # V-major physical storage.  Use unequal K/V dimensions so an accidental
    # raw reinterpretation cannot hide behind Qwen3.6's square 128x128 state.
    ssm_physical = torch.zeros(1, slots, 1, 3, 2)
    ssm_physical[0, 1, 0] = torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
    ssm_physical[0, 2, 0] = torch.tensor([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]])
    ssm_all = ssm_physical.transpose(-1, -2)
    assert not ssm_all.is_contiguous()
    initial_ssm = ssm_all.clone()
    intermediate_ssm_all = torch.zeros(
        1,
        batch_size + 1,
        draft_token_num,
        1,
        2,
        3,
    )
    intermediate_conv_all = torch.zeros(
        1,
        batch_size + 1,
        draft_token_num,
        channels,
        conv_window,
    )
    per_layer_cache = SimpleNamespace(
        conv=[conv_all[0]],
        temporal=ssm_all[0],
        intermediate_ssm=intermediate_ssm_all[0],
        intermediate_conv_window=[intermediate_conv_all[0]],
    )
    all_layer_cache = SimpleNamespace(
        conv=[conv_all],
        temporal=ssm_all,
        intermediate_ssm=intermediate_ssm_all,
        intermediate_conv_window=[intermediate_conv_all],
    )
    pool = SimpleNamespace(
        mamba2_layer_cache=lambda _layer_id: per_layer_cache,
        get_speculative_mamba2_params_all_layers=lambda: all_layer_cache,
    )
    cache_indices = torch.tensor([1, 2], dtype=torch.int32)
    metadata = SimpleNamespace(
        mamba_cache_indices=cache_indices,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    conv_calls = []

    def fake_causal_conv1d(x, _weight, **kwargs):
        conv_calls.append(kwargs["query_start_loc"].clone())
        states = kwargs["conv_states"]
        indices = kwargs["cache_indices"].to(torch.int64)
        for row, slot in enumerate(indices.tolist()):
            states[slot, : conv_window - 1].copy_(states[slot, 1:conv_window].clone())
            states[slot, conv_window - 1].copy_(x[row])
        return x

    monkeypatch.setattr(gdn_target_verify, "_causal_conv1d", fake_causal_conv1d)

    decode_calls = []
    decode_state_contiguity = []
    decode_initial_states = []

    def fake_decode(*, q, ssm_states, cache_indices, query_start_loc, **kwargs):
        del kwargs
        decode_calls.append(query_start_loc.clone())
        decode_state_contiguity.append(ssm_states.is_contiguous())
        decode_initial_states.append(ssm_states.clone())
        indices = cache_indices.to(torch.int64)
        ssm_states[indices] += q.reshape(batch_size, 1, 1, 1)
        return ssm_states.index_select(0, indices)[:, 0, 0, 0].reshape(
            1, batch_size, 1, 1
        )

    backend = SimpleNamespace(
        forward_metadata=metadata,
        graph_mode=False,
        req_to_token_pool=pool,
        kernel_dispatcher=SimpleNamespace(decode=fake_decode),
        _get_conv_weights_t=lambda _layer: torch.empty(0),
    )
    layer = SimpleNamespace(
        layer_id=0,
        conv_weights=torch.empty(channels, conv_window + 1),
        bias=None,
        q_dim=1,
        k_dim=1,
        v_dim=1,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_q_dim=1,
        head_k_dim=1,
        head_v_dim=1,
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
    )
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(draft_token_num=draft_token_num, topk=1),
        num_token_non_padded_cpu=batch_size * draft_token_num,
    )
    mixed_qkv = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )

    output = gdn_target_verify._sequential_target_verify(
        backend,
        layer,
        forward_batch,
        mixed_qkv,
        torch.zeros(4, 1),
        torch.zeros(4, 1),
    )

    assert output.flatten().tolist() == [11.0, 13.0, 23.0, 27.0]
    assert all(call.tolist() == [0, 1, 2] for call in conv_calls)
    assert all(call.tolist() == [0, 1, 2] for call in decode_calls)
    assert decode_state_contiguity == [True, True]
    assert torch.equal(decode_initial_states[0], initial_ssm[0, cache_indices])
    assert intermediate_ssm_all[0, 0, :, 0, 0, 0].tolist() == [11.0, 13.0]
    assert intermediate_ssm_all[0, 1, :, 0, 0, 0].tolist() == [23.0, 27.0]
    assert torch.equal(ssm_all, initial_ssm)
    assert intermediate_conv_all[0, 0, 0, 0].tolist() == [-1.0, 1.0]
    assert intermediate_conv_all[0, 0, 1, 0].tolist() == [1.0, 2.0]
    assert intermediate_conv_all[0, 1, 0, 0].tolist() == [-3.0, 3.0]
    assert intermediate_conv_all[0, 1, 1, 0].tolist() == [3.0, 4.0]

    hybrid = SimpleNamespace(linear_attn_backend=backend)
    gdn_target_verify._restore_selected_snapshots(
        hybrid,
        torch.tensor([0, 1]),
        torch.tensor([3, 3]),
        torch.tensor([-1, 0]),
    )

    assert torch.equal(ssm_all[0, 1], initial_ssm[0, 1] + 1.0)
    assert torch.equal(ssm_all[0, 2], initial_ssm[0, 2] + 7.0)
    assert torch.equal(ssm_all[0, 3], initial_ssm[0, 2] + 3.0)
    assert torch.equal(
        ssm_physical[0, 1, 0],
        (initial_ssm[0, 1, 0] + 1.0).transpose(-1, -2),
    )
    assert torch.equal(
        conv_all[0, 1, -conv_window:].transpose(-1, -2),
        intermediate_conv_all[0, 0, 0],
    )
    assert torch.equal(
        conv_all[0, 2, -conv_window:].transpose(-1, -2),
        intermediate_conv_all[0, 1, 1],
    )
    assert torch.equal(
        conv_all[0, 3, -conv_window:].transpose(-1, -2),
        intermediate_conv_all[0, 1, 0],
    )

    # A -1 main step is a no-op, matching SGLang's native move/rollback
    # kernels; it must not accidentally select Python's last draft index.
    ssm_all[0, 1].fill_(42.0)
    preserved_conv = torch.full_like(conv_all[0, 1, -conv_window:], 43.0)
    conv_all[0, 1, -conv_window:].copy_(preserved_conv)
    gdn_target_verify._restore_selected_snapshots(
        hybrid,
        torch.tensor([-1]),
        None,
        None,
    )
    assert torch.equal(ssm_all[0, 1], torch.full_like(ssm_all[0, 1], 42.0))
    assert torch.equal(conv_all[0, 1, -conv_window:], preserved_conv)


def test_ascend_logsumexp_topk_uses_sglang_fallback(monkeypatch) -> None:
    import torch

    from sglang_fl.dispatch.backends.vendor.ascend.patches import logsumexp

    fused_calls = []
    row_lse_calls = []

    def fused_topk(*_args, **_kwargs):
        fused_calls.append(True)
        pytest.fail("the fused top-k helper must not be called")

    def row_lse(x):
        row_lse_calls.append(x)
        x32 = x.float()
        row_max = x32.amax(dim=-1)
        row_log_sum = torch.logsumexp(x32 - row_max[:, None], dim=-1)
        return row_max, row_log_sum

    module = SimpleNamespace(
        FUSED_TOPK_MAX_K=8,
        row_logsumexp=row_lse,
        row_logsumexp_topk=fused_topk,
    )
    monkeypatch.setattr(
        logsumexp.importlib,
        "import_module",
        lambda name: (
            module
            if name == logsumexp._KERNEL_MODULE
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    assert logsumexp.patch_logsumexp_topk_fallback() is True
    patched_topk = module.row_logsumexp_topk
    assert module.FUSED_TOPK_MAX_K == 8

    logits = torch.tensor(
        [[1.0, 7.0, -2.0, 4.0], [3.0, -1.0, 9.0, 2.0]],
        dtype=torch.bfloat16,
    )
    row_max, row_log_sum, top_vals, top_idx = patched_topk(logits, 2)
    expected_vals, expected_idx = torch.topk(logits, 2, dim=-1)

    assert row_lse_calls == [logits]
    assert fused_calls == []
    assert row_max.dtype == torch.float32
    assert row_log_sum.dtype == torch.float32
    assert top_vals.dtype == torch.float32
    assert top_idx.dtype == torch.int64
    assert torch.equal(top_vals, expected_vals.float())
    assert torch.equal(top_idx, expected_idx)

    # Re-applying the patch is a no-op and preserves the split helper.
    assert logsumexp.patch_logsumexp_topk_fallback() is True
    assert module.row_logsumexp_topk is patched_topk


def test_ascend_logsumexp_topk_patch_is_optional(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.patches import logsumexp

    def missing(_name):
        raise ImportError("SGLang logsumexp module is unavailable")

    monkeypatch.setattr(logsumexp.importlib, "import_module", missing)
    assert logsumexp.patch_logsumexp_topk_fallback() is False


def test_ascend_single_node_entrypoints_default_gloo_to_loopback() -> None:
    root = Path(__file__).parents[3]
    examples = root / "examples"
    single_node_entrypoints = [
        examples / "qwen3_6_27b_concurrent.py",
        examples / "qwen3_6_27b_mtp_inference.py",
        examples / "qwen3_6_27b_offline_inference.py",
        examples / "qwen3_6_35b_a3b_concurrent.py",
        examples / "qwen3_6_35b_a3b_offline_inference.py",
    ]

    for entrypoint in single_node_entrypoints:
        source = entrypoint.read_text(encoding="utf-8")
        assert 'os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")' in source
        assert 'os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"' in source
        assert 'os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"' in source
        assert 'os.environ.pop("HCCL_HOST_SOCKET_PORT_RANGE", None)' not in source
        assert 'os.environ.setdefault("HCCL_IF_BASE_PORT"' not in source

    common = (root / "scripts" / "ascend" / "acceptance_common.sh").read_text(
        encoding="utf-8"
    )
    assert 'GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"' in common
    assert '-z "${HCCL_HOST_SOCKET_PORT_RANGE+x}"' in common
    assert '-z "${HCCL_NPU_SOCKET_PORT_RANGE+x}"' in common
    assert '-z "${HCCL_IF_BASE_PORT+x}"' in common
    assert "export HCCL_HOST_SOCKET_PORT_RANGE=auto" in common
    assert "export HCCL_NPU_SOCKET_PORT_RANGE=auto" in common
    assert "HCCL_HOST_SOCKET_PORT_RANGE=%s" in common
    assert "HCCL_NPU_SOCKET_PORT_RANGE=%s" in common
    assert 'export GLOO_SOCKET_IFNAME="${interface}"' in common

    for model in ("27b", "35b_a3b"):
        multinode = (examples / f"qwen3_6_{model}_multinode.py").read_text(
            encoding="utf-8"
        )
        assert 'os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"' in multinode
        assert 'os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"' in multinode
        assert 'os.environ.pop("HCCL_HOST_SOCKET_PORT_RANGE", None)' not in multinode
        assert 'os.environ.setdefault("HCCL_IF_BASE_PORT"' not in multinode


def test_ascend_fla_patch_preserves_native_gdn_state_contract(monkeypatch) -> None:
    from sglang_fl.dispatch import fla_patch
    from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge

    native_chunk = object()
    native_recurrent = object()
    native_packed = object()
    chunk_module = SimpleNamespace(chunk_gated_delta_rule=native_chunk)
    recurrent_module = SimpleNamespace(
        fused_recurrent_gated_delta_rule=native_recurrent,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )
    gdn_module = SimpleNamespace(
        chunk_gated_delta_rule=native_chunk,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )

    modules = {
        "sglang.kernels.ops.attention.fla.chunk": chunk_module,
        "sglang.kernels.ops.attention.fla.fused_recurrent": recurrent_module,
        "sglang.srt.layers.attention.linear.kernels.gdn_triton": gdn_module,
    }
    monkeypatch.setattr(
        fla_patch.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(fla_patch, "_originals", {})

    originals = fla_patch.patch_fla_functions(
        excluded_ops={
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
        excluded_gdn_ops={
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
    )

    # The public chunk function keeps its SGLang `(output, None, h)` contract
    # through the bridge, while the internal NPU alias retains the wheel's
    # `(output, final_state, h)` contract used for cache writeback.
    assert chunk_module.chunk_gated_delta_rule is chunk_gated_delta_rule_bridge
    assert gdn_module.chunk_gated_delta_rule is native_chunk
    assert recurrent_module.fused_recurrent_gated_delta_rule is native_recurrent
    assert (
        recurrent_module.fused_recurrent_gated_delta_rule_packed_decode is native_packed
    )
    assert gdn_module.fused_recurrent_gated_delta_rule_packed_decode is native_packed
    assert originals == {"chunk_gated_delta_rule": native_chunk}


def test_ascend_fla_patch_first_gdn_import_keeps_npu_rebinding(monkeypatch) -> None:
    from sglang_fl.dispatch import fla_patch
    from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge

    native_public_chunk = object()
    native_npu_chunk = object()
    native_recurrent = object()
    native_packed = object()
    chunk_module = SimpleNamespace(chunk_gated_delta_rule=native_public_chunk)
    recurrent_module = SimpleNamespace(
        fused_recurrent_gated_delta_rule=native_recurrent,
        fused_recurrent_gated_delta_rule_packed_decode=native_packed,
    )
    imported_gdn = []

    def import_module(name):
        if name == "sglang.kernels.ops.attention.fla.chunk":
            return chunk_module
        if name == "sglang.kernels.ops.attention.fla.fused_recurrent":
            return recurrent_module
        if name == "sglang.srt.layers.attention.linear.kernels.gdn_triton":
            # This mimics a first import on NPU: gdn_triton initially imports
            # the public name, then its is_npu() branch rebinds to the wheel.
            assert chunk_module.chunk_gated_delta_rule is chunk_gated_delta_rule_bridge
            module = SimpleNamespace(
                chunk_gated_delta_rule=native_npu_chunk,
                fused_recurrent_gated_delta_rule_packed_decode=native_packed,
            )
            imported_gdn.append(module)
            return module
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(fla_patch.importlib, "import_module", import_module)
    monkeypatch.setattr(fla_patch, "_originals", {})

    fla_patch.patch_fla_functions(
        excluded_ops={
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
        excluded_gdn_ops={
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        },
    )

    assert len(imported_gdn) == 1
    assert imported_gdn[0].chunk_gated_delta_rule is native_npu_chunk
    assert (
        imported_gdn[0].fused_recurrent_gated_delta_rule_packed_decode is native_packed
    )


def test_flagos_generic_recurrent_is_not_advertised_on_ascend(monkeypatch) -> None:
    import sglang_fl.utils as utils
    from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend

    backend = FlagOSBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(
        utils,
        "get_device_info",
        lambda: SimpleNamespace(vendor_name="ascend"),
    )

    assert backend.is_fused_recurrent_available() is False


def test_ascend_page_range_chunks_the_torch_npu_launch(monkeypatch) -> None:
    pytest.importorskip("sglang")
    import torch

    from sglang_fl.dispatch.backends.vendor.ascend import allocator

    calls = []
    native_arange = torch.arange

    def recording_arange(start, stop, **kwargs):
        calls.append((start, stop))
        kwargs["device"] = "cpu"
        return native_arange(start, stop, **kwargs)

    monkeypatch.setattr(allocator.torch, "arange", recording_arange)
    pages = allocator.build_ascend_page_range(
        131_075,
        device="npu",
        max_elements=65_535,
    )

    assert calls == [(1, 65_536), (65_536, 131_071), (131_071, 131_076)]
    assert all(stop - start <= 65_535 for start, stop in calls)
    assert pages.dtype == torch.int64
    assert pages.tolist() == list(range(1, 131_076))


def test_ascend_allocator_rejects_invalid_ranges() -> None:
    pytest.importorskip("sglang")
    from sglang_fl.dispatch.backends.vendor.ascend.allocator import (
        build_ascend_page_range,
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_ascend_page_range(-1, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        build_ascend_page_range(1, device="cpu", max_elements=0)
