"""CPU contracts for the applied DSpark backport, independent of GPU kernels."""

import ast
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def source():
    spec = PathFinder.find_spec("sglang")
    if spec is None:
        pytest.skip("SGLang source is required")
    root = Path(spec.origin).parent / "srt/speculative/dspark_components"
    if not (root / "tp_sync.py").exists():
        pytest.skip("apply the V4.1 source backport before running these tests")
    return root


def function(root, file, name, namespace):
    tree = ast.parse((root / file).read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    for n in ast.walk(node):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, ast.FunctionDef):
            n.returns = None
            n.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("dp_attention", [False, True])
@pytest.mark.parametrize("size", [1, 2])
def test_sync_uses_the_replica_group(source, dp_attention, size):
    calls = []

    def group(label):
        return NS(
            world_size=size,
            broadcast=lambda values, src: calls.append((label, src)),
        )

    parallel = NS(
        enable_dp_attention=dp_attention,
        attn_tp_group=group("attention"),
        tp_group=group("tensor"),
    )
    sync = function(
        source,
        "tp_sync.py",
        "sync_speculative_decision",
        {
            "get_parallel": lambda: parallel,
        },
    )
    values = torch.tensor([3, 9])
    assert sync(values) is values
    expected = [("attention" if dp_attention else "tensor", 0)] if size > 1 else []
    assert calls == expected


def sampler_namespace(fast):
    calls = []

    def sync(values):
        calls.append(values.clone())
        return values.fill_(7)

    ns = {
        "torch": torch,
        "sync_speculative_decision": sync,
        "expect": lambda contract, value, **kw: value,
        "_DRAFT_STEP_LOGITS": None,
        "_DRAFT_PROBS": None,
        "DraftBlockResult": NS,
        "resolve_greedy_mask": lambda **kw: torch.zeros(2, dtype=torch.bool),
        "envs": NS(SGLANG_DSPARK_FAST_SAMPLING=NS(get=lambda: fast)),
        "SampleStepTokens": NS(execute=lambda **kw: kw["step_logits"].argmax(-1)),
    }
    return ns, calls


def markov(logits, *, sampler, **kwargs):
    return sampler(logits[:, 0], 0)[:, None], logits


@pytest.mark.parametrize("greedy,fast", [(True, False), (False, False), (False, True)])
def test_eager_proposal_uses_the_synced_token(source, greedy, fast):
    ns, calls = sampler_namespace(fast)
    sample = function(source, "dspark_draft.py", "sample_draft_block", ns)
    output = sample(
        base_logits=torch.zeros(2, 1, 16),
        anchor_tokens=torch.zeros(2, dtype=torch.long),
        draft_hidden=torch.zeros(2, 1, 4),
        sampling_info=NS(is_all_greedy=greedy, temperatures=torch.ones(2)),
        markov_head=NS(sample_block=markov),
        device=torch.device("cpu"),
    )
    assert len(calls) == 1
    assert output.draft_tokens.tolist() == [[7], [7]]


@pytest.mark.parametrize("folded", [False, True])
def test_captured_proposal_uses_the_synced_token(source, folded):
    ns, calls = sampler_namespace(True)
    function(source, "dspark_draft_sampler.py", "greedy_step_sampler", ns)
    sample = function(source, "dspark_draft_sampler.py", "__call__", ns)
    state = NS(
        gamma=1,
        model=NS(compute_base_logits=lambda x: (torch.zeros(2, 16), None)),
        markov_head=NS(sample_block=markov),
        folded_sampling=folded,
        exp_noise=torch.empty(2, 16),
        temperatures=torch.ones(2),
        greedy_mask=torch.zeros(2, dtype=torch.bool),
        out=torch.empty(2, dtype=torch.long),
        corrected_out=torch.empty(2, 16),
        confidence_out=None,
    )
    sample(state, torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
    assert len(calls) == 1
    assert state.out.tolist() == [7, 7]


def test_accept_sync_precedes_kv_commit(source):
    decisions = iter([2, 42, 5])

    def sync(value):
        value.fill_(next(decisions))
        return value

    def finalize(**kw):
        assert kw["correct_len"].item() == 2
        assert kw["cap_trim_lens"].item() == 5
        commit = kw["correct_len"] + 1
        return NS(
            cap_trim_lens=kw["cap_trim_lens"],
            commit_lens=commit,
            new_seq_lens=kw["prefix_lens"] + commit,
        )

    ns = {
        "sync_speculative_decision": sync,
        "accept_draft_tokens": lambda **kw: tuple(
            torch.tensor([x]) for x in [0, 12, 6]
        ),
        "FinalizeAcceptLens": NS(execute=finalize),
        "BuildOutTokens": NS(execute=lambda **kw: kw["bonus"]),
        "AcceptOuts": NS,
    }
    accept = function(source, "dspark_verify.py", "accept_and_finalize", ns)
    output = accept(
        NS(_simulate_acc_len=0, gamma=5, verify_num_draft_tokens=6),
        folded_accept=False,
        bs=1,
        verify_ids_2d=None,
        target_logits=None,
        draft_block=None,
        sampling_info=None,
        draft_input=None,
        layout=None,
        prefix_lens=torch.tensor([100]),
        draft_tokens=None,
    )
    assert output.bonus.item() == 42
    assert output.commit_lens.item() == 3
    assert output.new_seq_lens.item() == 103
