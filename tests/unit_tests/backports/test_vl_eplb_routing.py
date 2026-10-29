"""VL routing must preserve logical expert outputs after physical relocation."""

import ast
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def routing():
    spec = PathFinder.find_spec("sglang")
    if spec is None:
        pytest.skip("SGLang source is required")
    source = Path(spec.origin).parent / "srt/multimodal/dsv41/vl_routing.py"
    if not source.exists():
        pytest.skip("apply the V4.1 source backport first")
    records = []
    state = NS(enabled=True, info=None)

    def mask(ids, count):
        if count is not None:
            ids[torch.arange(ids.shape[0]) >= count.item()] = -1

    def zero(weights, count):
        if count is not None:
            weights[torch.arange(weights.shape[0]) >= count.item()] = 0

    namespace = {
        "torch": torch,
        "F": F,
        "is_cuda": lambda: False,
        "has_per_rank_fused_shared_slots": lambda count: False,
        "get_exec": lambda: NS(moe=NS(enable_eplb=state.enabled)),
        "ExpertLocationDispatchInfo": NS(init_new=lambda layer: state.info),
        "topk_ids_logical_to_physical": lambda ids, info: info.inverse[ids],
        "get_global_expert_distribution_recorder": lambda: NS(
            on_select_experts=lambda **kw: records.append(kw["topk_ids"].clone())
        ),
        "_mask_topk_ids_padded_region": mask,
        "_zero_topk_weights_padded_region": zero,
        "_RENORMALIZE_SUM_EPSILON": 1e-20,
        "StandardTopKOutput": lambda weights, ids, logits: NS(
            topk_weights=weights, topk_ids=ids
        ),
    }
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    config = NS(
        num_fused_shared_experts=0,
        top_k=2,
        renormalize=True,
        apply_routed_scaling_factor_on_output=True,
        routed_scaling_factor=2.0,
        fused_shared_experts_scaling_factor=None,
    )
    moe = NS(
        topk=NS(topk_config=config),
        layer_id=3,
        is_nextn=False,
        config=NS(image_token_id=99),
        gate=NS(
            e_score_correction_bias=torch.tensor([0.1, -0.2, 0.3, 0.0]),
            e_score_correction_bias_vl=torch.tensor([-0.2, 0.4, 0.0, 0.2]),
        ),
    )
    return namespace, state, records, moe


@pytest.mark.parametrize("shared", [0, 1])
def test_expert_relocation_preserves_weighted_output(routing, shared):
    ns, state, records, moe = routing
    moe.topk.topk_config.num_fused_shared_experts = shared
    moe.topk.topk_config.top_k = 2 + shared
    logits = torch.tensor([[2.0, 0.1, 0.9, 1.1], [0.5, 1.3, 0.2, 0.1]])
    token_ids = torch.tensor([4, 99])
    logical_values = torch.tensor([11.0, 23.0, 37.0, 53.0, 71.0])
    identity = torch.arange(4)
    state.info = NS(inverse=identity, num_physical_experts=4)
    before = ns["vision_topk"](moe, logits, token_ids)
    expected = (logical_values[before.topk_ids] * before.topk_weights).sum(-1)
    for permutation in [torch.tensor([2, 0, 3, 1]), torch.tensor([1, 3, 0, 2])]:
        state.info = NS(inverse=permutation.argsort(), num_physical_experts=4)
        physical_values = torch.cat([logical_values[permutation], logical_values[4:]])
        after = ns["vision_topk"](moe, logits, token_ids)
        actual = (physical_values[after.topk_ids] * after.topk_weights).sum(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            after.topk_weights, before.topk_weights, rtol=0, atol=0
        )
        assert records[-1].shape == (2, 2)
        assert torch.equal(records[-1], after.topk_ids[:, :2])


def test_padding_is_not_remapped_to_a_real_expert(routing):
    ns, state, records, moe = routing
    state.info = NS(inverse=torch.tensor([2, 3, 1, 0]), num_physical_experts=4)
    output = ns["vision_topk"](
        moe, torch.ones(3, 4), torch.tensor([1, 99, 1]), torch.tensor(1)
    )
    assert (output.topk_ids[1:] == -1).all()
    assert (output.topk_weights[1:] == 0).all()
    assert (records[-1][1:] == -1).all()


@pytest.mark.parametrize("draft", [False, True])
def test_no_target_mapping_without_target_eplb(routing, draft):
    ns, state, records, moe = routing
    state.info = NS(inverse=torch.tensor([2, 3, 1, 0]), num_physical_experts=4)
    state.enabled = draft
    moe.is_nextn = draft
    output = ns["vision_topk"](moe, torch.tensor([[2.0, 0.1, 0.9, 1.1]]), None)
    assert output.topk_ids.tolist() == [[0, 2]]
    assert records == []
