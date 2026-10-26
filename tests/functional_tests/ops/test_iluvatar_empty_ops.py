# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CoreX GPU acceptance for PR132 vendor and optional FlagOS empty adapters.

Run with the documented Iluvatar runtime:
    python -m pytest tests/functional_tests/ops/test_iluvatar_empty_ops.py -q

Only absence of CoreX skips this module. Missing kernels, compilation failures,
numerical mismatches and graph-capture failures on CoreX are test failures.
The oracles below use explicit FP32 CPU equations, never SGLang or FlagGems.
"""

from __future__ import annotations

from importlib import import_module
from typing import NamedTuple
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

_DTYPE = torch.float16
_RTOL = 1.5e-2
_ATOL = 3e-3


@pytest.mark.parametrize("mask_mode", [0, 1], ids=["full-mask", "qlen-only"])
def test_speculative_tree_upstream_triton_branch_acceptance(mask_mode):
    """Compile the real tree/verify kernels without loading a model."""
    from sglang_fl.dispatch.backends.vendor.iluvatar.patches.speculative_tree import (
        patch_speculative_tree,
    )

    patch_speculative_tree()
    eagle = import_module("sglang.srt.speculative.eagle_utils")
    assert eagle.sgl_build_tree_kernel_efficient is eagle.sgl_build_tree_kernel_triton
    device = "cuda"
    lengths = torch.tensor([2, 3], device=device, dtype=torch.int64)
    # Root -> children 1,2; the last node belongs to 1 in row 0 and 2 in row 1.
    parent = torch.tensor([[0, 0], [0, 1]], device=device, dtype=torch.int64)
    selected = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device, dtype=torch.int64)
    draft = torch.tensor([[11, 22, 33], [44, 55, 66]], device=device, dtype=torch.int64)
    bonus = torch.tensor([100, 200], device=device, dtype=torch.int64)
    mask = torch.ones(52 if mask_mode == 0 else 32, device=device, dtype=torch.bool)
    output = eagle.build_tree_kernel_efficient(
        bonus,
        parent,
        selected,
        draft,
        lengths,
        5,
        2,
        2,
        4,
        tree_mask_mode=mask_mode,
        tree_mask_buf=mask,
    )
    tree_mask, positions, retrieve, next_token, next_sibling, candidates = output
    assert tree_mask is mask
    torch.testing.assert_close(positions.cpu(), torch.tensor([2, 3, 3, 4, 3, 4, 4, 5]))
    torch.testing.assert_close(retrieve.cpu(), torch.arange(8).reshape(2, 4))
    torch.testing.assert_close(
        next_token.cpu(), torch.tensor([[1, 3, -1, -1], [1, -1, 3, -1]])
    )
    torch.testing.assert_close(
        next_sibling.cpu(), torch.tensor([[-1, 2, -1, -1], [-1, 2, -1, -1]])
    )
    ancestry = [
        torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.bool
        ),
        torch.tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]], dtype=torch.bool
        ),
    ]
    prefixes = [2, 3] if mask_mode == 0 else [0, 0]
    expected_mask = torch.cat(
        [
            torch.cat(
                (torch.ones(4, prefix, dtype=torch.bool), branch), dim=1
            ).flatten()
            for prefix, branch in zip(prefixes, ancestry)
        ]
    )
    torch.testing.assert_close(tree_mask.cpu(), expected_mask)
    candidates = candidates.reshape(2, 4)
    torch.testing.assert_close(
        candidates.cpu(), torch.tensor([[100, 11, 22, 33], [200, 44, 55, 66]])
    )
    predicts = torch.full((8,), -7, device=device, dtype=torch.int64)
    accepted = torch.full((2, 3), -1, device=device, dtype=torch.int64)
    count = torch.full((2,), -1, device=device, dtype=torch.int64)
    targets = torch.tensor(
        [[11, 33, 90, 777], [55, 80, 123, 70]], device=device, dtype=torch.int64
    )
    result = eagle.verify_tree_greedy_func(
        predicts,
        accepted,
        count,
        candidates,
        retrieve,
        next_token,
        next_sibling,
        targets,
        topk=2,
    )
    assert all(
        actual is original
        for actual, original in zip(result, (predicts, accepted, count))
    )
    torch.testing.assert_close(accepted.cpu(), torch.tensor([[0, 1, 3], [4, 6, -1]]))
    torch.testing.assert_close(count.cpu(), torch.tensor([2, 1]))
    torch.testing.assert_close(
        predicts.cpu(), torch.tensor([11, 33, -7, 777, 55, -7, 123, -7])
    )
    spec_utils = import_module("sglang.srt.speculative.spec_utils")
    scores, indices = spec_utils.fast_topk(
        torch.tensor([[0.1, 0.8, 0.4]], device=device), 1, -1
    )
    torch.testing.assert_close(scores.cpu(), torch.tensor([[0.8]]))
    torch.testing.assert_close(indices.cpu(), torch.tensor([[1]]))


@pytest.fixture(autouse=True)
def _require_corex():
    if not hasattr(torch, "corex"):
        pytest.skip("Iluvatar CoreX torch is required")
    assert torch.cuda.is_available(), (
        "CoreX torch is installed but its GPU is unavailable"
    )


@pytest.fixture(
    params=[
        pytest.param("vendor", id="vendor_pr132"),
        pytest.param("flagos", id="flagos"),
    ]
)
def iluvatar_ops(request, _require_corex):
    """Run the same public SGLang contracts against each implementation."""
    backend = request.param
    package = "vendor.iluvatar" if backend == "vendor" else "flagos"
    suffix = "iluvatar" if backend == "vendor" else "flagos"
    base = f"sglang_fl.dispatch.backends.{package}.impl"
    fla = import_module(f"{base}.fla")
    moe = import_module(f"{base}.fused_moe")
    return SimpleNamespace(
        backend=backend,
        chunk=getattr(fla, f"chunk_gated_delta_rule_{suffix}"),
        recurrent=getattr(fla, f"fused_recurrent_gated_delta_rule_{suffix}"),
        packed_decode=getattr(
            fla, f"fused_recurrent_gated_delta_rule_packed_decode_{suffix}"
        ),
        fused_moe=getattr(moe, f"fused_moe_{suffix}"),
    )


class _TopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None = None


def _random(shape, *, seed, scale=0.2, dtype=_DTYPE):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


def _assert_close(actual, expected, *, label):
    actual_cpu = actual.detach().float().cpu()
    assert actual_cpu.shape == expected.shape, label
    assert torch.isfinite(actual_cpu).all(), label
    torch.testing.assert_close(
        actual_cpu, expected.float(), rtol=_RTOL, atol=_ATOL, msg=label
    )


def _cpu_fla(q, k, v, g, beta, state, *, scale, cu_seqlens=None, normalize=False):
    """Token-by-token delta recurrence with SGLang's explicit [V,K] state."""
    q, k, v, g, beta = (x.detach().float().cpu() for x in (q, k, v, g, beta))
    state = state.detach().float().cpu().clone()
    if normalize:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    batch, length, query_heads, _ = q.shape
    value_heads = v.shape[2]
    head_groups = value_heads // query_heads
    assert head_groups * query_heads == value_heads
    output = torch.zeros_like(v)
    if cu_seqlens is None:
        spans = [(row, 0, length) for row in range(batch)]
    else:
        bounds = cu_seqlens.tolist()
        spans = [(0, start, stop) for start, stop in zip(bounds[:-1], bounds[1:])]
    chunks = []
    for sequence, (row, start, stop) in enumerate(spans):
        history = []
        current = state[sequence].clone()
        for offset, token in enumerate(range(start, stop)):
            if offset % 64 == 0:
                history.append(current.clone())
            for value_head in range(value_heads):
                query_head = value_head // head_groups
                key = k[row, token, query_head]
                query = q[row, token, query_head] * scale
                decayed = current[value_head] * torch.exp(g[row, token, value_head])
                prediction = torch.mv(decayed, key)
                delta = (v[row, token, value_head] - prediction) * beta[
                    row, token, value_head
                ]
                current[value_head] = decayed + delta[:, None] * key[None, :]
                output[row, token, value_head] = torch.mv(current[value_head], query)
        state[sequence] = current
        chunks.append(torch.stack(history))
    chunk_states = (
        torch.cat(chunks).unsqueeze(0)
        if cu_seqlens is not None
        else torch.stack(chunks)
    )
    return output, state, chunk_states


def _fla_inputs(batch, length, query_heads=1):
    value_heads, key_dim, value_dim = 2, 32, 16
    return {
        "q": _random((batch, length, query_heads, key_dim), seed=11),
        "k": _random((batch, length, query_heads, key_dim), seed=12),
        "v": _random((batch, length, value_heads, value_dim), seed=13),
        "g": -_random((batch, length, value_heads), seed=14, scale=0.08).abs().float(),
        "beta": _random((batch, length, value_heads), seed=15, scale=0.7)
        .float()
        .sigmoid(),
    }


def _to_gpu(inputs):
    return {name: tensor.cuda() for name, tensor in inputs.items()}


@pytest.mark.parametrize("normalize", [False, True])
def test_chunk_varlen_indexed_cache_output_and_64_token_states(iluvatar_ops, normalize):
    # Nonsequential valid slots, one padding sentinel, and a sequence spanning
    # two 64-token chunks detect cache layout and state-history errors.
    bounds = torch.tensor([0, 5, 72, 76], dtype=torch.int32)
    slots = torch.tensor([4, -1, 1], dtype=torch.int64)
    inputs = _fla_inputs(1, 76)
    cache_cpu = _random((6, 2, 16, 32), seed=20, scale=0.06, dtype=torch.float32)
    initial = cache_cpu[slots.clamp_min(0)].clone()
    initial[slots < 0] = 0
    expected, final, history = _cpu_fla(
        **inputs, state=initial, scale=0.5, cu_seqlens=bounds, normalize=normalize
    )
    expected_cache = cache_cpu.clone()
    expected_cache[4], expected_cache[1] = final[0], final[2]
    cache = cache_cpu.cuda()

    actual, final_output, actual_history = iluvatar_ops.chunk(
        **_to_gpu(inputs),
        scale=0.5,
        initial_state=cache,
        initial_state_indices=slots.cuda(),
        cu_seqlens=bounds.cuda(),
        use_qk_l2norm_in_kernel=normalize,
    )

    assert final_output is None
    assert actual.dtype == _DTYPE
    _assert_close(actual, expected, label="chunk output")
    _assert_close(actual_history, history, label="64-token chunk start states in [V,K]")
    _assert_close(cache, expected_cache, label="indexed chunk state writeback")
    untouched = torch.tensor([0, 2, 3, 5], device="cuda")
    assert torch.equal(cache[untouched], cache_cpu.cuda()[untouched])


@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_dense_history_shape_and_initial_state_ownership(
    iluvatar_ops, with_state
):
    inputs = _fla_inputs(2, 67, query_heads=2)
    initial = _random((2, 2, 16, 32), seed=21, scale=0.06, dtype=torch.float32)
    if not with_state:
        initial.zero_()
    expected, _, history = _cpu_fla(**inputs, state=initial, scale=0.25)
    state = initial.cuda() if with_state else None
    before = state.clone() if state is not None else None
    actual, final_output, actual_history = iluvatar_ops.chunk(
        **_to_gpu(inputs),
        scale=0.25,
        initial_state=state,
    )
    assert final_output is None
    assert actual_history.shape == (2, 2, 2, 16, 32)
    _assert_close(actual, expected, label="dense chunk output")
    _assert_close(actual_history, history, label="dense [B,ceil(T/64),HV,V,K] history")
    if state is not None:
        assert torch.equal(state, before), (
            "unindexed public chunk must not mutate initial_state"
        )


@pytest.mark.parametrize(
    "batch,length,query_heads,with_state,return_state",
    [(2, 2, 1, True, True), (2, 65, 2, True, True), (2, 3, 2, False, False)],
)
def test_public_recurrent_state_layout_and_input_ownership(
    iluvatar_ops, batch, length, query_heads, with_state, return_state
):
    inputs = _fla_inputs(batch, length, query_heads)
    initial = _random((batch, 2, 16, 32), seed=30, scale=0.06, dtype=torch.float32)
    if not with_state:
        initial.zero_()
    expected, final, _ = _cpu_fla(**inputs, state=initial, scale=0.25, normalize=True)
    state = initial.cuda() if with_state else None
    before = state.clone() if state is not None else None
    actual, actual_state = iluvatar_ops.recurrent(
        **_to_gpu(inputs),
        scale=0.25,
        initial_state=state,
        output_final_state=return_state,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close(actual, expected, label="public recurrent output")
    assert actual.dtype == _DTYPE
    if return_state:
        assert actual_state.dtype == torch.float32
        _assert_close(actual_state, final, label="public recurrent final [V,K] state")
    else:
        assert actual_state is None
    if state is not None:
        assert torch.equal(state, before), (
            "public recurrent must not mutate initial_state"
        )
        assert actual_state.data_ptr() != state.data_ptr()


def test_public_recurrent_varlen_sequences_do_not_share_state(iluvatar_ops):
    inputs = _fla_inputs(1, 8, query_heads=2)
    bounds = torch.tensor([0, 3, 8], dtype=torch.int32)
    initial = _random((2, 2, 16, 32), seed=31, scale=0.06, dtype=torch.float32)
    expected, final, _ = _cpu_fla(
        **inputs, state=initial, scale=0.25, cu_seqlens=bounds
    )
    state = initial.cuda()
    actual, actual_state = iluvatar_ops.recurrent(
        **_to_gpu(inputs),
        scale=0.25,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=bounds.cuda(),
    )
    _assert_close(actual, expected, label="varlen recurrent output")
    _assert_close(actual_state, final, label="varlen recurrent final state")
    assert torch.equal(state, initial.cuda())


def _packed_case(batch=3):
    query_heads, value_heads, key_dim, value_dim = 1, 2, 32, 16
    inputs = _fla_inputs(batch, 1, query_heads)
    mixed = torch.cat(
        [inputs["q"].flatten(1), inputs["k"].flatten(1), inputs["v"].flatten(1)], -1
    )
    a = _random((batch, value_heads), seed=40, scale=0.8)
    b = _random((batch, value_heads), seed=41, scale=0.9)
    log_decay = _random((value_heads,), seed=42, scale=0.4, dtype=torch.float32)
    bias = _random((value_heads,), seed=43, scale=0.3)
    cache = _random(
        (batch + 3, value_heads, value_dim, key_dim),
        seed=44,
        scale=0.06,
        dtype=torch.float32,
    )
    slots = torch.arange(batch, dtype=torch.int64).roll(1) + 1
    slots[0], slots[1] = batch + 1, -1
    return mixed, a, b, log_decay, bias, cache, slots


def _cpu_packed(mixed, a, b, log_decay, bias, cache, slots, out):
    batch, value_heads, value_dim, key_dim = len(slots), *cache.shape[-3:]
    query_heads = (mixed.shape[-1] - value_heads * value_dim) // (2 * key_dim)
    query_width = query_heads * key_dim
    q, k, v = torch.split(
        mixed, [query_width, query_width, value_heads * value_dim], -1
    )
    q = q.reshape(batch, 1, query_heads, key_dim)
    k = k.reshape(batch, 1, query_heads, key_dim)
    v = v.reshape(batch, 1, value_heads, value_dim)
    # Gate equations intentionally run in FP32 before entering the recurrence.
    g = -torch.exp(log_decay.float()) * F.softplus(
        a.float() + bias.float(), threshold=20
    )
    beta = torch.sigmoid(b.float())
    initial = cache[slots.clamp_min(0)].clone()
    expected, final, _ = _cpu_fla(
        q, k, v, g[:, None], beta[:, None], initial, scale=0.25, normalize=True
    )
    next_cache, next_out = cache.clone(), out.clone()
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            next_cache[slot] = final[row]
            next_out[row] = expected[row]
        else:
            # Official SGLang 0.5.18 packed decode writes zero for PAD_SLOT_ID.
            next_out[row].zero_()
    return next_out, next_cache


def _packed_gpu_buffers(case):
    mixed, a, b, log_decay, bias, cache, slots = case
    return dict(
        mixed_qkv=mixed.cuda(),
        a=a.cuda(),
        b=b.cuda(),
        A_log=log_decay.cuda(),
        dt_bias=bias.cuda(),
        scale=0.25,
        initial_state=cache.cuda(),
        out=torch.full(
            (len(slots), 1, *cache.shape[-3:-1]), -3.25, device="cuda", dtype=_DTYPE
        ),
        ssm_state_indices=slots.cuda(),
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.parametrize("batch", [3, 65])
def test_packed_decode_updates_only_selected_slots_and_caller_output(
    iluvatar_ops, batch
):
    case = _packed_case(batch)
    initial_out = torch.full((batch, 1, 2, 16), -3.25, dtype=torch.float32)
    expected_out, expected_cache = _cpu_packed(*case, initial_out)
    buffers = _packed_gpu_buffers(case)
    output, state = iluvatar_ops.packed_decode(**buffers)
    assert output is buffers["out"]
    assert state is buffers["initial_state"]
    _assert_close(
        output, expected_out, label="packed decode output including -1 sentinel"
    )
    _assert_close(
        state, expected_cache, label="packed decode indexed [V,K] state writeback"
    )
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    used_slots = set(case[6].tolist())
    for slot in set(range(case[5].shape[0])) - used_slots:
        assert torch.equal(state[slot].cpu(), case[5][slot])


def test_packed_graph_replay_reads_and_updates_live_cache(iluvatar_ops):
    case = _packed_case()
    buffers = _packed_gpu_buffers(case)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        iluvatar_ops.packed_decode(**buffers)
        iluvatar_ops.packed_decode(**buffers)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_state = iluvatar_ops.packed_decode(**buffers)
    assert captured_out is buffers["out"]
    assert captured_state is buffers["initial_state"]

    # Ignore any capture/warmup state and start replay from a known live cache.
    buffers["initial_state"].copy_(case[5])
    buffers["out"].fill_(-3.25)
    expected_out = torch.full((3, 1, 2, 16), -3.25, dtype=torch.float32)
    expected_cache = case[5].clone()
    for replay in range(2):
        reference_case = (*case[:5], expected_cache, case[6])
        expected_out, expected_cache = _cpu_packed(*reference_case, expected_out)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(
            buffers["out"], expected_out, label=f"graph replay {replay} output"
        )
        _assert_close(
            buffers["initial_state"],
            expected_cache,
            label=f"graph replay {replay} cache writeback",
        )
    assert torch.equal(buffers["out"][1], torch.zeros_like(buffers["out"][1]))
    used_slots = set(case[6].tolist())
    for slot in set(range(case[5].shape[0])) - used_slots:
        assert torch.equal(buffers["initial_state"][slot].cpu(), case[5][slot])


@pytest.mark.parametrize(
    "with_bias,route_on_input,scale",
    [
        pytest.param(False, False, 1.0, id="standard"),
        pytest.param(False, False, 0.7, id="routed_scale"),
        pytest.param(True, False, 0.7, id="bias"),
        pytest.param(True, True, 0.7, id="bias_and_input_routing"),
    ],
)
def test_moe_gated_silu_bias_routing_and_scaling_match_cpu(
    iluvatar_ops, with_bias, route_on_input, scale
):
    tokens, experts, hidden, intermediate, topk = 5, 4, 32, 64, 2
    # Keep outputs above the absolute tolerance so omitting routed scaling or
    # either expert GEMM cannot pass just because every result is near zero.
    x = _random((tokens, hidden), seed=50, scale=0.5)
    w13 = _random((experts, 2 * intermediate, hidden), seed=51, scale=0.2)
    w2 = _random((experts, hidden, intermediate), seed=52, scale=0.2)
    bias13 = (
        _random((experts, 2 * intermediate), seed=53, scale=0.03) if with_bias else None
    )
    bias2 = _random((experts, hidden), seed=54, scale=0.03) if with_bias else None
    ids = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1], [1, 2]], dtype=torch.int32)
    weights = _random((tokens, topk), seed=55, scale=0.5, dtype=torch.float32).softmax(
        -1
    )
    expected = torch.zeros((tokens, hidden), dtype=torch.float32)
    for token in range(tokens):
        for choice in range(topk):
            expert = ids[token, choice].item()
            route_weight = weights[token, choice]
            expert_input = x[token].float()
            gate_up = torch.mv(w13[expert].float(), expert_input)
            if bias13 is not None:
                gate_up = gate_up + bias13[expert].float()
            # SGLang/FlagGems apply this flag to the GEMM1 result, including
            # its bias, before the gated activation (rather than weighting x).
            if route_on_input:
                gate_up = gate_up * route_weight
            gate, up = gate_up.chunk(2)
            expert_output = torch.mv(w2[expert].float(), F.silu(gate) * up)
            if bias2 is not None:
                expert_output = expert_output + bias2[expert].float()
            expected[token] += expert_output * (1 if route_on_input else route_weight)
    expected *= scale
    assert expected.abs().max() > 10 * _ATOL, (
        "MoE oracle signal is below its error budget"
    )

    obj = SimpleNamespace(
        moe_runner_config=SimpleNamespace(
            num_experts=experts,
            num_local_experts=experts,
            activation="silu",
            is_gated=True,
            no_combine=False,
            apply_router_weight_on_input=route_on_input,
            routed_scaling_factor=scale,
        )
    )
    layer = SimpleNamespace(
        w13_weight=w13.cuda(),
        w2_weight=w2.cuda(),
        w13_weight_bias=bias13.cuda() if bias13 is not None else None,
        w2_weight_bias=bias2.cuda() if bias2 is not None else None,
    )
    dispatch = SimpleNamespace(
        hidden_states=x.cuda(),
        topk_output=_TopKOutput(topk_ids=ids.cuda(), topk_weights=weights.cuda()),
    )
    if iluvatar_ops.backend == "vendor" and (with_bias or route_on_input):
        # PR132 intentionally advertises this narrower contract. Unsupported
        # configurations must fail explicitly instead of producing partial math.
        with pytest.raises(NotImplementedError):
            iluvatar_ops.fused_moe(obj, layer, dispatch)
        return
    before = dispatch.hidden_states.clone()
    result = iluvatar_ops.fused_moe(obj, layer, dispatch)
    _assert_close(
        result.hidden_states, expected, label="MoE CPU gated-SiLU routing equation"
    )
    assert result.hidden_states.dtype == _DTYPE
    assert torch.equal(dispatch.hidden_states, before), (
        "MoE must not overwrite dispatch input"
    )


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "dtype_name"),
    [((256, 32), (1, 32), "float32"), ((16, 256, 288), (1, 1, 288), "bfloat16")],
    ids=["broadcast-2d-fp32", "broadcast-nd-bf16"],
)
def test_iluvatar_cold_mul_inplace_preserves_value_and_alias(
    tmp_path, a_shape, b_shape, dtype_name
):
    """Exercise the production ATen registration in an isolated cold process."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = os.environ.copy()
    for key in (
        "SGLANG_FL_CONFIG",
        "SGLANG_FL_FLAGOS_WHITELIST",
        "SGLANG_FL_FLAGOS_BLACKLIST",
        "FLAGGEMS_DB_URL",
    ):
        env.pop(key, None)
    env["USE_FLAGGEMS"] = "1"
    env["FLAGGEMS_CACHE_DIR"] = str(tmp_path / "private-gems-cache")
    log_dir = Path(env.get("ILUVATAR_CI_ARTIFACT_DIR", tmp_path)) / "mul-inplace"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{dtype_name}-{os.getpid()}.log"
    child = r"""
import json
import os
from pathlib import Path
import sys

cache = Path(os.environ["FLAGGEMS_CACHE_DIR"])
assert not cache.exists(), "A new process must begin with a private cold cache"
cache.mkdir()
assert not list(cache.iterdir())
import torch
assert hasattr(torch, "corex") and torch.cuda.is_available()
dtype = getattr(torch, sys.argv[1])
a_shape = tuple(json.loads(sys.argv[2]))
b_shape = tuple(json.loads(sys.argv[3]))
cpu_a = torch.ones(a_shape, dtype=dtype)
cpu_b = torch.full(b_shape, 0.5, dtype=dtype)
expected = (cpu_a.float() * cpu_b.float()).to(dtype)
a, b = cpu_a.cuda(), cpu_b.cuda()
import sglang_fl
config = sglang_fl._build_config()
sglang_fl._setup_flaggems(config)
import flag_gems
keys = flag_gems.all_registered_keys()
native_route = torch._C._dispatch_dump_table("aten::mul_.Tensor")
assert "mul.Tensor" in keys, "The out-of-place FlagGems route must remain enabled"
print(json.dumps({"private_cache": str(cache), "cold_before_import": True,
                  "blacklist": config["flagos_blacklist"],
                  "mul_registered": "mul.Tensor" in keys,
                  "mul_inplace_registered": "mul_.Tensor" in keys,
                  "actual_mul_inplace_cuda_dispatch": native_route}), flush=True)
pointer = a.data_ptr()
result = a.mul_(b)
torch.cuda.synchronize()
print(json.dumps({"cold_first_value": a.flatten()[0].item(),
                  "expected_first_value": expected.flatten()[0].item()}), flush=True)
torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)
assert result is a and result.data_ptr() == pointer
torch.testing.assert_close(b.cpu(), cpu_b, rtol=0, atol=0)
assert "mul_.Tensor" not in keys, "Iluvatar must retain native in-place multiply"
fresh_a = cpu_a.cuda()
out = torch.mul(fresh_a, b)
torch.cuda.synchronize()
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
torch.testing.assert_close(fresh_a.cpu(), cpu_a, rtol=0, atol=0)
torch.testing.assert_close(b.cpu(), cpu_b, rtol=0, atol=0)
assert out.data_ptr() != fresh_a.data_ptr()
print("Production cold mul_ value/alias/input and out-of-place checks passed", flush=True)
"""
    import json

    with log.open("x", encoding="utf-8") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                child,
                dtype_name,
                json.dumps(a_shape),
                json.dumps(b_shape),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")
