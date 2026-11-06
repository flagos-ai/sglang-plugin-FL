"""CPU contracts for changed runtime-memory, vision TP and FLA entry points."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _load(relative):
    path = ROOT / "sglang_fl/dispatch/backends/vendor/enflame" / relative
    spec = importlib.util.spec_from_file_location("enflame_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_modules(monkeypatch, definitions):
    # Isolate package roots too: accidentally importing removed SGLang paths must fail.
    names = set()
    for name in definitions:
        parts = name.split(".")
        names.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    for name in sorted(names, key=lambda value: value.count(".")):
        module = ModuleType(name)
        module.__path__ = []
        for attr, value in definitions.get(name, {}).items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("device,distributed,empty_cache", [
    ("gcu", False, True), ("gcu", True, False), ("cuda", True, True),
])
def test_memory_query_units_cache_and_distributed_min(monkeypatch, device, distributed, empty_cache):
    module = _load("patches/runtime_memory.py")
    group, calls = object(), []
    free_bytes = 8.5 * (1 << 30)
    reduced_bytes = 3.25 * (1 << 30)

    def query(gpu_id):
        assert gpu_id == 2
        calls.append("query")
        return free_bytes, 16 * (1 << 30)

    def reduce(tensor, op, group):
        assert distributed and group is expected_group
        assert tensor.device.type == "cpu"
        assert op is torch.distributed.ReduceOp.MIN
        assert tensor.item() == free_bytes
        calls.append("reduce")
        tensor.fill_(reduced_bytes)

    expected_group = group
    monkeypatch.setattr(torch, "gcu", SimpleNamespace(empty_cache=lambda: calls.append("empty"), mem_get_info=query), raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    original = Mock()
    result = module.get_available_gpu_memory_gcu(original, device, 2, distributed, empty_cache, group)
    assert result == (3.25 if distributed else 8.5)
    assert calls == (["empty"] if empty_cache else []) + ["query"] + (["reduce"] if distributed else [])
    original.assert_not_called()


def test_memory_other_device_delegates_all_options():
    module = _load("patches/runtime_memory.py")
    group, result = object(), object()
    original = Mock(return_value=result)
    assert module.get_available_gpu_memory_gcu(original, "cpu", 2, True, False, group) is result
    original.assert_called_once_with("cpu", 2, True, False, group)


@pytest.mark.parametrize("data_parallel", [False, True])
def test_vision_reads_new_runtime_context_tp_api(monkeypatch, data_parallel):
    parallel = Mock(return_value=SimpleNamespace(attn_tp_size=4))
    _fake_modules(monkeypatch, {
        "sglang.srt.layers.attention.vision": {"SingletonCache": type("Cache", (), {}), "resolve_seqlens": Mock()},
        "sglang.srt.environ": {"envs": SimpleNamespace()},
        "sglang.srt.runtime_context": {"get_parallel": parallel},
    })
    module = _load("patches/vision.py")
    # Upstream's replacement target is a torch Module, unlike the patch holder class.
    instance = torch.nn.Module.__new__(torch.nn.Module)
    module.GCU_VisionFlash3Attention.__init__(instance, use_data_parallel=data_parallel)
    assert instance.tp_size == (1 if data_parallel else 4)
    assert parallel.call_count == (0 if data_parallel else 1)


def test_fla_chunk_new_entry_point_and_return_contract(monkeypatch):
    inputs = [object() for _ in range(5)]
    state, indices, lengths, output, final_state = (object() for _ in range(5))
    calls = []

    def apply(q, k, v, g, beta, scale, initial_state, initial_state_indices,
              cu_seqlens, use_qk_l2norm_in_kernel):
        assert all(a is b for a, b in zip((q, k, v, g, beta), inputs))
        assert scale == 0.5 and initial_state is state
        assert initial_state_indices is indices and cu_seqlens is lengths
        assert use_qk_l2norm_in_kernel is True
        calls.append(q)
        return output, final_state

    _fake_modules(monkeypatch, {
        "sglang.kernels.ops.attention.fla.chunk": {"ChunkGatedDeltaRuleFunction": SimpleNamespace(apply=apply)},
    })
    path = ROOT / "sglang_fl/dispatch/backends/vendor/enflame/impl/fla.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "chunk_gated_delta_rule_gcu")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    actual = namespace[node.name](*inputs, scale=0.5, initial_state=state,
                                  initial_state_indices=indices, cu_seqlens=lengths,
                                  use_qk_l2norm_in_kernel=True)
    assert actual[0] is output and actual[1] is None and actual[2] is final_state
    assert len(calls) == 1


def test_gdn_mtp_verify_matches_upstream_signature():
    # Compare source signatures without importing Triton or initializing devices.
    import sglang

    relative = "kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"
    upstream = Path(sglang.__file__).resolve().parent / relative
    replacement = ROOT / (
        "sglang_fl/dispatch/backends/vendor/enflame/impl/gdn_mtp_verify.py"
    )

    def signature(path, name):
        tree = ast.parse(path.read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for arg in (
            function.args.posonlyargs + function.args.args
            + function.args.kwonlyargs
        ):
            arg.annotation = None
        return ast.dump(function.args, include_attributes=False)

    assert signature(
        replacement, "fused_sigmoid_gating_delta_rule_update_gcu"
    ) == signature(upstream, "fused_sigmoid_gating_delta_rule_update")
