"""CPU contracts for GCU logprob selection and non-GCU fallback."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load(name, relative):
    root = Path(__file__).resolve().parents[3]
    path = root / "sglang_fl/dispatch/backends/vendor/enflame/patches" / relative
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logprob = _load("enflame_logprob_under_test", "logprob.py")


def test_logprob_leaves_other_devices_to_original():
    x = torch.tensor([[0.1, 0.9]])
    original = Mock(return_value=([0.9], [1]))
    result = logprob.top_logprobs_raw_gcu(original, x, [1], "decode")
    assert result == ([0.9], [1])
    assert original.call_args.args[0] is x


def test_gcu_logprob_cpu_selection_preserves_values_and_indices():
    x = torch.tensor([[0.5, 1.5, -1.0]])

    class DeviceInput:
        device = SimpleNamespace(type="gcu")

        def cpu(self):
            return x

    def select(tensor, counts, stage, pruned, no_copy):
        assert tensor.device.type == "cpu"
        values, indices = tensor.topk(max(counts), dim=-1)
        return values.tolist(), indices.tolist()

    values, indices = logprob.top_logprobs_raw_gcu(select, DeviceInput(), [2], "decode")
    assert values == [[1.5, 0.5]]
    assert indices == [[1, 0]]


@pytest.mark.parametrize("no_copy", [False, True])
def test_raw_optional_arguments_and_nested_device_return(monkeypatch, no_copy):
    module = _load("enflame_logprob_return_under_test", "logprob.py")
    target = SimpleNamespace(type="gcu")
    moves, calls = [], []

    class ResultTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            assert device is target
            moves.append(self)
            return SimpleNamespace(value=self.value, device=device)

    monkeypatch.setattr(module, "torch", SimpleNamespace(Tensor=ResultTensor))
    cpu_input, counts, pruned = object(), object(), object()
    input_ = SimpleNamespace(device=target, cpu=lambda: cpu_input)
    values, indices = ResultTensor(0.75), ResultTensor(2)
    result = ([values], (indices, None, "retained"))

    def original(actual_input, actual_counts, stage, actual_pruned, actual_no_copy):
        assert actual_input is cpu_input and actual_counts is counts
        assert stage == "extend" and actual_pruned is pruned
        assert actual_no_copy is no_copy
        calls.append(actual_input)
        return result

    actual = module.top_logprobs_raw_gcu(original, input_, counts, "extend", pruned, no_copy)
    assert len(calls) == 1
    if no_copy:
        assert moves == [values, indices]
        assert isinstance(actual, tuple) and isinstance(actual[0], list)
        assert isinstance(actual[1], tuple)
        assert actual[0][0].device is target and actual[0][0].value == 0.75
        assert actual[1][0].device is target and actual[1][0].value == 2
        assert actual[1][1:] == (None, "retained")
    else:
        assert actual is result and moves == []


@pytest.mark.parametrize("gcu", [False, True])
@pytest.mark.parametrize("optional", [False, True])
def test_chunk_preserves_optional_data_and_output_containers(gcu, optional):
    module = _load("enflame_logprob_chunk_under_test", "logprob.py")
    moves, calls = [], []

    class Input:
        device = SimpleNamespace(type="gcu" if gcu else "cpu")

        def __init__(self):
            self.host = object()

        def cpu(self):
            assert gcu
            moves.append(self)
            return self.host

    logprobs = Input()
    normalizer = tuple(Input() for _ in range(2)) if optional else None
    precomputed = tuple(Input() for _ in range(2)) if optional else None
    counts, pruned, split, result = (object() for _ in range(4))
    out_values, out_indices = [], []

    def original(actual, actual_counts, actual_pruned, values, indices, actual_split,
                 *, log_normalizer, precomputed_topk):
        assert actual is (logprobs.host if gcu else logprobs)
        assert actual_counts is counts and actual_pruned is pruned
        assert values is out_values and indices is out_indices and actual_split is split
        if optional:
            for received, source in ((log_normalizer, normalizer), (precomputed_topk, precomputed)):
                assert all(a is (b.host if gcu else b) for a, b in zip(received, source))
                assert len(received) == len(source)
                if not gcu:
                    assert received is source
        else:
            assert log_normalizer is None and precomputed_topk is None
        values.append("value")
        indices.append("index")
        calls.append(actual)
        return result

    assert module.top_logprobs_chunk_gcu(original, logprobs, counts, pruned,
                                         out_values, out_indices, split,
                                         log_normalizer=normalizer,
                                         precomputed_topk=precomputed) is result
    assert len(calls) == 1
    assert out_values == ["value"] and out_indices == ["index"]
    assert len(moves) == ((5 if optional else 1) if gcu else 0)
