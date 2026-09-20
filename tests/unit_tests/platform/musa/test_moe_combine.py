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

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.patches import moe_combine


class _FakeDevice:
    type = "musa"

    def __init__(self, index=0):
        self.index = index

    def __str__(self):
        return f"musa:{self.index}"


class _FakeTensor:
    def __init__(self, shape, dtype=torch.bfloat16, device=None, contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device or _FakeDevice()
        self._contiguous = contiguous
        self.recorded_streams = []

    def is_contiguous(self):
        return self._contiguous

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


@pytest.fixture(autouse=True)
def _accept_fake_contract_tensors(monkeypatch):
    """The entry guard checks ``torch.Tensor``; these unit tests use fakes.

    Only ``_FakeTensor`` is widened.  Plain non-tensor objects still hit the
    real ``isinstance`` check, so the rejection path stays testable.
    """

    real_is_tensor = moe_combine._is_tensor

    def is_tensor(value):
        return real_is_tensor(value) or isinstance(value, _FakeTensor)

    monkeypatch.setattr(moe_combine, "_is_tensor", is_tensor)


class _FakeStream:
    def __init__(self, name):
        self.name = name
        self.waited_for = []

    def wait_stream(self, stream):
        self.waited_for.append(stream)


def _fake_contract_tensors(tokens=2048, device_index=0):
    device = _FakeDevice(device_index)
    routed = _FakeTensor((tokens, 8, 2048), device=device)
    output = _FakeTensor((tokens, 2048), device=device)
    shared = _FakeTensor((tokens, 2048), device=device)
    gate = _FakeTensor((tokens, 1), device=device)
    return routed, output, shared, gate


def _context(tokens=2048, device_index=0):
    routed, output, shared, gate = _fake_contract_tensors(tokens, device_index)
    return moe_combine.MoeCombineContext(shared, gate), (routed, output, shared, gate)


def _valid_config(**overrides):
    values = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _set_model_marker(block, config=None):
    setattr(
        block,
        moe_combine._MODEL_CONTRACT_MARKER,
        moe_combine._model_contract_from_config(config or _valid_config()),
    )


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disable", "disabled"])
def test_env_off(monkeypatch, value):
    monkeypatch.setenv(moe_combine._ENV_NAME, value)
    assert not moe_combine._enabled()


def test_env_auto_is_enabled(monkeypatch):
    monkeypatch.delenv(moe_combine._ENV_NAME, raising=False)
    assert moe_combine._enabled()
    monkeypatch.setenv(moe_combine._ENV_NAME, "auto")
    assert moe_combine._enabled()


def test_decode_graph_env_is_independent_and_off_by_default(monkeypatch):
    monkeypatch.delenv(moe_combine._DECODE_GRAPH_ENV_NAME, raising=False)
    assert not moe_combine._decode_graph_enabled()
    monkeypatch.setenv(moe_combine._DECODE_GRAPH_ENV_NAME, "auto")
    assert moe_combine._decode_graph_enabled()
    monkeypatch.setenv(moe_combine._DECODE_GRAPH_ENV_NAME, "off")
    assert not moe_combine._decode_graph_enabled()


def test_model_contract_marker_is_immutable_and_exact():
    marker = moe_combine._model_contract_from_config(_valid_config())
    assert marker.matches
    assert marker.model_type == "qwen3_5_moe_text"
    with pytest.raises(AttributeError):
        marker.matches = False

    mismatch = moe_combine._model_contract_from_config(
        _valid_config(shared_expert_intermediate_size=1024)
    )
    assert not mismatch.matches


def test_constructor_marker_supports_positional_and_keyword_config():
    class RealLikeQwen2MoeSparseMoeBlock:
        def __init__(self, layer_id, config, **kwargs):
            self.layer_id = layer_id
            # The framework class intentionally does not retain config.
            self.kwargs = kwargs

    original = RealLikeQwen2MoeSparseMoeBlock.__init__
    RealLikeQwen2MoeSparseMoeBlock.__init__ = moe_combine._make_qwen_init(original)
    try:
        positional = RealLikeQwen2MoeSparseMoeBlock(0, _valid_config())
        keyword = RealLikeQwen2MoeSparseMoeBlock(1, config=_valid_config())
        for instance in (positional, keyword):
            marker = getattr(instance, moe_combine._MODEL_CONTRACT_MARKER)
            assert marker.matches
            assert not hasattr(instance, "config")
    finally:
        RealLikeQwen2MoeSparseMoeBlock.__init__ = original


def test_constructor_marker_mismatch_and_init_failure():
    class RealLikeQwen2MoeSparseMoeBlock:
        def __init__(self, layer_id, config):
            self.layer_id = layer_id

    original = RealLikeQwen2MoeSparseMoeBlock.__init__
    RealLikeQwen2MoeSparseMoeBlock.__init__ = moe_combine._make_qwen_init(original)
    try:
        mismatch = RealLikeQwen2MoeSparseMoeBlock(
            0, _valid_config(model_type="qwen2_moe")
        )
        assert not getattr(mismatch, moe_combine._MODEL_CONTRACT_MARKER).matches

        def failing_init(self, layer_id, config):
            raise RuntimeError("constructor failure")

        wrapped_failure = moe_combine._make_qwen_init(failing_init)
        failed = RealLikeQwen2MoeSparseMoeBlock.__new__(RealLikeQwen2MoeSparseMoeBlock)
        with pytest.raises(RuntimeError, match="constructor failure"):
            wrapped_failure(failed, 0, _valid_config())
        assert not hasattr(failed, moe_combine._MODEL_CONTRACT_MARKER)
    finally:
        RealLikeQwen2MoeSparseMoeBlock.__init__ = original


def test_unreadable_config_field_yields_non_matching_contract():
    class ExplodingConfig:
        @property
        def hidden_size(self):
            raise RuntimeError("cannot read")

    marker = moe_combine._model_contract_from_config(ExplodingConfig())

    assert not marker.matches
    assert marker.hidden_size is None


def test_unreadable_config_does_not_break_construction():
    class ExplodingConfig:
        @property
        def hidden_size(self):
            raise RuntimeError("cannot read")

    class RealLikeQwen2MoeSparseMoeBlock:
        def __init__(self, layer_id, config):
            self.layer_id = layer_id

    wrapped = moe_combine._make_qwen_init(
        RealLikeQwen2MoeSparseMoeBlock.__init__
    )
    instance = RealLikeQwen2MoeSparseMoeBlock.__new__(RealLikeQwen2MoeSparseMoeBlock)
    wrapped(instance, 0, ExplodingConfig())

    marker = getattr(instance, moe_combine._MODEL_CONTRACT_MARKER)
    assert not marker.matches


def test_full_install_is_idempotent(monkeypatch):
    class FakeQwenBlock:
        def __init__(self, layer_id, config):
            self.layer_id = layer_id

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    original_reduce = lambda *args, **kwargs: "reduce"
    fused_module = _apply_targets(monkeypatch, original_reduce, FakeQwenBlock)

    assert moe_combine.apply_musa_deterministic_moe_combine_patch()
    first = (
        fused_module.moe_sum_reduce,
        FakeQwenBlock.__init__,
        FakeQwenBlock.forward,
    )
    assert all(getattr(obj, moe_combine._PATCH_MARKER, False) for obj in first)

    # Repeated full installation must reuse the existing wrappers.
    assert moe_combine.apply_musa_deterministic_moe_combine_patch()
    assert (
        fused_module.moe_sum_reduce,
        FakeQwenBlock.__init__,
        FakeQwenBlock.forward,
    ) == first

    instance = FakeQwenBlock(0, _valid_config())
    assert getattr(instance, moe_combine._MODEL_CONTRACT_MARKER).matches


def test_real_qwen_class_constructor_marker_without_self_config(monkeypatch):
    qwen_module = pytest.importorskip("sglang.srt.models.qwen2_moe")
    if not hasattr(qwen_module, "get_tensor_model_parallel_world_size"):
        pytest.skip("constructor integration requires the pinned MUSA SGLang 0.5.11 API")
    cls = qwen_module.Qwen2MoeSparseMoeBlock

    class FakeTopK:
        def __init__(self, top_k, renormalize, layer_id):
            self.topk_config = SimpleNamespace(top_k=top_k)

    class FakeExperts(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.num_experts = kwargs["num_experts"]
            self.moe_runner_config = SimpleNamespace(
                inplace=True,
                no_combine=False,
                routed_scaling_factor=None,
            )

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeMLP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(qwen_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(qwen_module, "get_moe_impl_class", lambda quant: FakeExperts)
    monkeypatch.setattr(qwen_module, "TopK", FakeTopK)
    monkeypatch.setattr(qwen_module, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(qwen_module, "Qwen2MoeMLP", FakeMLP)
    monkeypatch.setattr(
        qwen_module,
        "get_global_server_args",
        lambda: SimpleNamespace(ep_num_redundant_experts=0),
    )
    monkeypatch.setattr(
        qwen_module,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: False),
    )
    monkeypatch.setattr(qwen_module, "_use_aiter", False)

    original_init = cls.__init__
    original_forward = cls.forward
    fused_module = SimpleNamespace(moe_sum_reduce=lambda *args, **kwargs: None)
    real_import = moe_combine.importlib.import_module

    def import_module(name):
        if name.endswith("triton_utils.fused_moe"):
            return fused_module
        return real_import(name)

    monkeypatch.setattr(moe_combine.importlib, "import_module", import_module)
    monkeypatch.setattr(moe_combine.torch, "musa", SimpleNamespace(), raising=False)
    assert moe_combine.apply_musa_deterministic_moe_combine_patch()
    try:
        config = _valid_config(norm_topk_prob=False, hidden_act="silu")
        instance = cls(layer_id=0, config=config)
        marker = getattr(instance, moe_combine._MODEL_CONTRACT_MARKER)
        assert marker.matches
        assert not hasattr(instance, "config")
    finally:
        cls.__init__ = original_init
        cls.forward = original_forward


def test_wrapper_build_failure_leaves_targets_unmodified(monkeypatch, caplog):
    class FakeQwenBlock:
        def __init__(self, layer_id, config):
            self.layer_id = layer_id

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    original_reduce = lambda *args, **kwargs: None
    fused_module = _apply_targets(monkeypatch, original_reduce, FakeQwenBlock)
    old_init, old_forward = FakeQwenBlock.__init__, FakeQwenBlock.forward

    def boom(module, original):
        raise RuntimeError("wrapper construction failed")

    monkeypatch.setattr(moe_combine, "_make_qwen_forward", boom)

    with caplog.at_level(logging.WARNING, logger=moe_combine.__name__):
        assert not moe_combine.apply_musa_deterministic_moe_combine_patch()

    # The build stage runs before any install point is touched.
    assert fused_module.moe_sum_reduce is original_reduce
    assert FakeQwenBlock.__init__ is old_init
    assert FakeQwenBlock.forward is old_forward
    assert "wrapper build failed" in caplog.text


def _apply_targets(monkeypatch, fused_reduce, qwen_cls):
    fused_module = SimpleNamespace(moe_sum_reduce=fused_reduce)
    qwen_module = SimpleNamespace(Qwen2MoeSparseMoeBlock=qwen_cls)

    def import_module(name):
        if name.endswith("triton_utils.fused_moe"):
            return fused_module
        return qwen_module

    monkeypatch.setattr(moe_combine.importlib, "import_module", import_module)
    monkeypatch.setattr(moe_combine.torch, "musa", SimpleNamespace(), raising=False)
    return fused_module


def test_partial_assignment_failure_restores_pre_install_objects(monkeypatch, caplog):
    state = {"failed": False}

    class _FailForwardOnce(type):
        def __setattr__(cls, name, value):
            if name == "forward" and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("forward assignment failed")
            super().__setattr__(name, value)

    class FakeQwenBlock(metaclass=_FailForwardOnce):
        def __init__(self, layer_id, config):
            self.layer_id = layer_id

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    # The pre-install seam may already hold a wrapper; it must be restored
    # as-is rather than unwrapped to an earlier function.
    existing_wrapper = lambda *args, **kwargs: "already wrapped"
    fused_module = _apply_targets(monkeypatch, existing_wrapper, FakeQwenBlock)
    old_init, old_forward = FakeQwenBlock.__init__, FakeQwenBlock.forward

    with caplog.at_level(logging.WARNING, logger=moe_combine.__name__):
        assert not moe_combine.apply_musa_deterministic_moe_combine_patch()

    assert fused_module.moe_sum_reduce is existing_wrapper
    assert FakeQwenBlock.__init__ is old_init
    assert FakeQwenBlock.forward is old_forward
    assert "rolled back" in caplog.text


def test_restore_installation_reports_incomplete_rollback(caplog):
    fused_module = SimpleNamespace(moe_sum_reduce="installed")

    class _RefusingTarget:
        def __setattr__(self, name, value):
            raise RuntimeError("assignment refused")

    with caplog.at_level(logging.WARNING, logger=moe_combine.__name__):
        restored = moe_combine._restore_installation(
            fused_module, _RefusingTarget(), "old", object(), object()
        )

    assert restored is False
    assert fused_module.moe_sum_reduce == "old"
    assert "failed to restore" in caplog.text


def test_device_name_cache_is_per_device(monkeypatch):
    calls = []
    musa = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda device: calls.append(str(device)) or "MTT S5000",
    )
    monkeypatch.setattr(moe_combine.torch, "musa", musa, raising=False)
    monkeypatch.setattr(moe_combine, "_DEVICE_NAME_CACHE", {})
    context0, tensors0 = _context()
    routed0, output0, _, _ = tensors0
    assert moe_combine._contract_matches(routed0, output0, 1.0, context0)
    assert moe_combine._contract_matches(routed0, output0, 1.0, context0)
    context1, tensors1 = _context(device_index=1)
    routed1, output1, _, _ = tensors1
    assert moe_combine._contract_matches(routed1, output1, 1.0, context1)
    assert calls == ["musa:0", "musa:1"]


def test_env_off_forces_combine_fallback(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setenv(moe_combine._ENV_NAME, "false")
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, (routed, output, _, _) = _context()
    launch_calls = []
    monkeypatch.setattr(
        moe_combine,
        "_launch_candidate",
        lambda *args, **kwargs: launch_calls.append((args, kwargs)),
    )

    def original(*args, **kwargs):
        return "baseline"

    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        result = wrapped(routed, output, 1.0)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert result == "baseline"
    assert not launch_calls
    assert not context.used


@pytest.mark.parametrize("invalid_input", ["routed", "output", "shared", "gate"])
def test_non_tensor_reduction_inputs_keep_original_path(monkeypatch, invalid_input):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setenv(moe_combine._ENV_NAME, "auto")
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, (routed, output, shared, gate) = _context()
    assert moe_combine._contract_matches(routed, output, 1.0, context)
    inputs = dict(routed=routed, output=output, shared=shared, gate=gate)
    inputs[invalid_input] = SimpleNamespace()
    context.shared_unweighted = inputs["shared"]
    context.gate_logits = inputs["gate"]
    original_calls = []
    launch_calls = []
    monkeypatch.setattr(
        moe_combine, "_launch_candidate", lambda *a: launch_calls.append(a)
    )

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "baseline"

    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        # Objects with no tensor attributes must not raise from the contract
        # read that sits outside the launch guard.
        result = wrapped(inputs["routed"], inputs["output"], 1.0)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert result == "baseline"
    assert original_calls == [((inputs["routed"], inputs["output"], 1.0), {})]
    assert not launch_calls
    assert not context.used
    assert not moe_combine._CANDIDATE_DISABLED


def test_non_tensor_hidden_states_do_not_enter_optimization(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    block, _ = _fake_block()

    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(), block, SimpleNamespace(), None
    )


def test_shape_and_layout_mismatch_falls_back(monkeypatch):
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, _ = _context(tokens=1024)
    routed, output, _, _ = _fake_contract_tensors(tokens=1024)
    original_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "baseline"

    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        result = wrapped(routed, output, 1.0)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert result == "baseline"
    assert original_calls
    assert not context.used


@pytest.mark.parametrize("tokens", [2048, 4096, 6144, 8192, 16384])
def test_combine_contract_accepts_supported_token_shapes(monkeypatch, tokens):
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, (routed, output, _, _) = _context(tokens=tokens)

    assert moe_combine._contract_matches(routed, output, 1.0, context)


def test_exact_m16k_combine_uses_large_shape_kernel_geometry():
    assert moe_combine._candidate_for_tokens(16384) == (1, 2048, 16)
    with pytest.raises(ValueError, match="unsupported"):
        moe_combine._candidate_for_tokens(16385)


def test_decode_graph_combine_uses_measured_small_shape_geometries():
    assert moe_combine._candidate_for_tokens(40) == (2, 512, 8)
    assert moe_combine._candidate_for_tokens(64) == (1, 2048, 16)


def test_model_marker_missing_or_reduce_scatter_falls_back(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    block, _ = _fake_block()
    delattr(block, moe_combine._MODEL_CONTRACT_MARKER)
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(), block, _FakeTensor((2048, 2048)), None
    )

    block, _ = _fake_block()
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(),
        block,
        _FakeTensor((2048, 2048)),
        None,
        use_reduce_scatter=True,
    )


def test_model_contract_accepts_exact_m16k_and_rejects_adjacent_shape(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    block, _ = _fake_block(tokens=1)

    assert moe_combine._model_contract_matches(
        _fake_qwen_module(),
        block,
        _FakeTensor((16384, 2048)),
        None,
    )
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(),
        block,
        _FakeTensor((16385, 2048)),
        None,
    )


def test_combine_consumed_marks_context(monkeypatch):
    monkeypatch.setattr(moe_combine, "_SUCCESS_LOGGED", set())
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, (routed, output, _, _) = _context()
    launch_calls = []
    original_calls = []

    def launch(routed_arg, output_arg, context_arg):
        launch_calls.append((routed_arg, output_arg, context_arg))

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "baseline"

    monkeypatch.setattr(moe_combine, "_launch_candidate", launch)
    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        result = wrapped(routed, output, None)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert result is None
    assert len(launch_calls) == 1
    assert context.used
    assert not original_calls


def test_combine_exception_uses_original_and_does_not_consume(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_SUCCESS_LOGGED", set())
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    context, (routed, output, _, _) = _context()
    original_calls = []
    launch_calls = []

    def launch(*args, **kwargs):
        launch_calls.append((args, kwargs))
        raise RuntimeError("compile failure")

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "baseline"

    monkeypatch.setattr(moe_combine, "_launch_candidate", launch)
    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        result = wrapped(routed, output, 1.0)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert result == "baseline"
    assert original_calls
    assert not context.used
    assert moe_combine._CANDIDATE_DISABLED

    # Once compilation/launch fails, later layers/requests must not retry the
    # candidate or emit one warning per layer.
    context2, (routed2, output2, _, _) = _context()
    token = moe_combine._ACTIVE_CONTEXT.set(context2)
    try:
        result2 = wrapped(routed2, output2, 1.0)
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)
    assert result2 == "baseline"
    assert len(launch_calls) == 1
    assert len(original_calls) == 2


def _fake_qwen_module(capture=False, deepep=False):
    return SimpleNamespace(
        get_is_capture_mode=lambda: capture,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_deepep=lambda: deepep),
        get_global_server_args=lambda: SimpleNamespace(
            enable_fused_moe_sum_all_reduce=False
        ),
        should_skip_post_experts_all_reduce=lambda **kwargs: True,
        tensor_model_parallel_all_reduce=lambda value: value,
    )


def _decode_forward_batch(is_decode=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: is_decode),
    )


def _fake_block(tokens=2048):
    hidden = torch.zeros((tokens, 2048), dtype=torch.bfloat16)
    block = SimpleNamespace(
        tp_size=2,
        num_experts=256,
        num_shared_experts=1,
        num_fused_shared_experts=0,
        enable_shared_expert_fusion=False,
        shared_expert=lambda value: torch.ones_like(value),
        shared_expert_gate=lambda value: torch.zeros(
            (value.shape[0], 1), dtype=value.dtype, device=value.device
        ),
        topk=SimpleNamespace(topk_config=SimpleNamespace(top_k=8)),
        experts=SimpleNamespace(
            num_experts=256,
            moe_runner_config=SimpleNamespace(
                inplace=True,
                no_combine=False,
                routed_scaling_factor=None,
            ),
        ),
        alt_stream=None,
    )
    _set_model_marker(block)
    return block, hidden


def test_model_wrapper_consumed_skips_shared_add(monkeypatch):
    qwen_module = _fake_qwen_module()
    block, hidden = _fake_block()

    def router(value):
        context = moe_combine._ACTIVE_CONTEXT.get()
        assert context is not None
        context.used = True
        return torch.zeros_like(value)

    block._forward_router_experts = router
    monkeypatch.setattr(moe_combine, "_model_contract_matches", lambda *args, **kwargs: not kwargs.get("decode_graph", False))
    wrapped = moe_combine._make_qwen_forward(
        qwen_module, lambda *args, **kwargs: pytest.fail("unexpected fallback")
    )

    result = wrapped(block, hidden)

    assert torch.equal(result, torch.zeros_like(hidden))
    assert moe_combine._ACTIVE_CONTEXT.get() is None


def test_model_wrapper_unconsumed_adds_original_shared_term(monkeypatch):
    qwen_module = _fake_qwen_module()
    block, hidden = _fake_block()
    block._forward_router_experts = lambda value: torch.zeros_like(value)
    monkeypatch.setattr(moe_combine, "_model_contract_matches", lambda *args, **kwargs: not kwargs.get("decode_graph", False))
    wrapped = moe_combine._make_qwen_forward(
        qwen_module, lambda *args, **kwargs: pytest.fail("unexpected fallback")
    )

    result = wrapped(block, hidden)

    expected = torch.full_like(hidden, 0.5)
    assert torch.equal(result, expected)
    assert moe_combine._ACTIVE_CONTEXT.get() is None


def test_capture_and_deepep_are_fallbacks(monkeypatch):
    block, hidden = _fake_block()
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(capture=True), block, hidden, None
    )
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(deepep=True), block, hidden, None
    )


@pytest.mark.parametrize("tokens", [40, 64])
def test_decode_graph_model_contract_is_exact_and_independent(monkeypatch, tokens):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_DECODE_GRAPH_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    monkeypatch.setenv(moe_combine._ENV_NAME, "auto")
    monkeypatch.setenv(moe_combine._DECODE_GRAPH_ENV_NAME, "auto")
    block, _ = _fake_block(tokens=1)
    block.alt_stream = _FakeStream("alt")
    hidden = _FakeTensor((tokens, 2048))

    assert moe_combine._decode_graph_model_contract_matches(
        _fake_qwen_module(capture=True),
        block,
        hidden,
        _decode_forward_batch(),
    )
    assert not moe_combine._model_contract_matches(
        _fake_qwen_module(capture=True),
        block,
        hidden,
        _decode_forward_batch(),
    )


def test_decode_graph_model_contract_rejects_adjacent_and_non_capture(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_DECODE_GRAPH_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    monkeypatch.setenv(moe_combine._ENV_NAME, "auto")
    monkeypatch.setenv(moe_combine._DECODE_GRAPH_ENV_NAME, "auto")
    block, _ = _fake_block(tokens=1)
    block.alt_stream = _FakeStream("alt")
    hidden = _FakeTensor((39, 2048))

    assert not moe_combine._decode_graph_model_contract_matches(
        _fake_qwen_module(capture=True),
        block,
        hidden,
        _decode_forward_batch(),
    )
    assert not moe_combine._decode_graph_model_contract_matches(
        _fake_qwen_module(capture=False),
        block,
        _FakeTensor((40, 2048)),
        _decode_forward_batch(),
    )
    assert not moe_combine._decode_graph_model_contract_matches(
        _fake_qwen_module(capture=True),
        block,
        _FakeTensor((40, 2048)),
        _decode_forward_batch(is_decode=False),
    )


def test_decode_graph_reduce_joins_shared_tail_before_launch(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_DECODE_GRAPH_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    producer_stream = _FakeStream("primary")
    consumer_stream = _FakeStream("alt")
    context, (routed, output, shared, gate) = _context(tokens=40)
    context.decode_graph_dual_stream = True
    context.shared_stream = producer_stream
    launch_calls = []
    original_calls = []
    monkeypatch.setattr(
        moe_combine.torch.cuda, "current_stream", lambda: consumer_stream
    )
    monkeypatch.setattr(
        moe_combine,
        "_launch_candidate",
        lambda *args: launch_calls.append(args),
    )

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))

    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        assert wrapped(routed, output, 1.0) is None
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert context.used
    assert len(launch_calls) == 1
    assert not original_calls
    assert consumer_stream.waited_for == [producer_stream]
    assert shared.recorded_streams == [consumer_stream]
    assert gate.recorded_streams == [consumer_stream]


def test_decode_graph_launch_failure_falls_back_without_disabling_eager(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_DECODE_GRAPH_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    producer_stream = _FakeStream("primary")
    consumer_stream = _FakeStream("alt")
    context, (routed, output, _, _) = _context(tokens=64)
    context.decode_graph_dual_stream = True
    context.shared_stream = producer_stream
    original_calls = []
    monkeypatch.setattr(
        moe_combine.torch.cuda, "current_stream", lambda: consumer_stream
    )
    monkeypatch.setattr(
        moe_combine,
        "_launch_candidate",
        lambda *args: (_ for _ in ()).throw(RuntimeError("compile failure")),
    )

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "baseline"

    wrapped = moe_combine._wrap_moe_sum_reduce(original)
    token = moe_combine._ACTIVE_CONTEXT.set(context)
    try:
        assert wrapped(routed, output, 1.0) == "baseline"
    finally:
        moe_combine._ACTIVE_CONTEXT.reset(token)

    assert len(original_calls) == 1
    assert not context.used
    assert moe_combine._DECODE_GRAPH_CANDIDATE_DISABLED
    assert not moe_combine._CANDIDATE_DISABLED


def test_decode_graph_forward_preserves_dual_stream_fork_and_join(monkeypatch):
    qwen_module = _fake_qwen_module(capture=True)
    block, hidden = _fake_block(tokens=40)
    primary_stream = _FakeStream("primary")
    alternate_stream = _FakeStream("alt")
    block.alt_stream = alternate_stream

    def router(value):
        context = moe_combine._ACTIVE_CONTEXT.get()
        assert context is not None
        assert context.decode_graph_dual_stream
        assert context.shared_stream is primary_stream
        context.used = True
        return torch.zeros_like(value)

    block._forward_router_experts = router
    monkeypatch.setattr(
        moe_combine.torch.cuda, "current_stream", lambda: primary_stream
    )
    monkeypatch.setattr(moe_combine.torch.cuda, "stream", lambda stream: nullcontext())

    result = moe_combine._forward_decode_graph_combine(
        qwen_module,
        block,
        hidden,
        use_reduce_scatter=False,
        should_allreduce_fusion=False,
    )

    assert torch.equal(result, torch.zeros_like(hidden))
    assert alternate_stream.waited_for == [primary_stream]
    assert primary_stream.waited_for == [alternate_stream]
    assert moe_combine._ACTIVE_CONTEXT.get() is None


def test_eager_alt_stream_handle_does_not_force_fallback(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", False)
    monkeypatch.setattr(moe_combine, "_device_name", lambda tensor: "MTT S5000")
    block, _ = _fake_block()
    block.alt_stream = object()
    hidden = _FakeTensor((2048, 2048), device=_FakeDevice())

    assert moe_combine._model_contract_matches(_fake_qwen_module(), block, hidden, None)


def test_model_exception_resets_context(monkeypatch):
    qwen_module = _fake_qwen_module()
    block, hidden = _fake_block()

    def router(value):
        raise RuntimeError("router failure")

    block._forward_router_experts = router
    monkeypatch.setattr(moe_combine, "_model_contract_matches", lambda *args, **kwargs: not kwargs.get("decode_graph", False))
    wrapped = moe_combine._make_qwen_forward(qwen_module, lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="router failure"):
        wrapped(block, hidden)
    assert moe_combine._ACTIVE_CONTEXT.get() is None


def test_model_wrapper_after_candidate_failure_uses_original(monkeypatch):
    monkeypatch.setattr(moe_combine, "_CANDIDATE_DISABLED", True)
    qwen_module = _fake_qwen_module()
    block, hidden = _fake_block()
    fallback_calls = []
    launch_calls = []

    def original(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return "baseline"

    monkeypatch.setattr(
        moe_combine,
        "_launch_candidate",
        lambda *args, **kwargs: launch_calls.append((args, kwargs)),
    )
    wrapped = moe_combine._make_qwen_forward(qwen_module, original)

    result = wrapped(block, hidden)

    assert result == "baseline"
    assert len(fallback_calls) == 1
    assert not launch_calls


def test_wrappers_are_idempotent():
    original_reduce = lambda *args, **kwargs: None
    wrapped_reduce = moe_combine._wrap_moe_sum_reduce(original_reduce)
    assert moe_combine._wrap_moe_sum_reduce(wrapped_reduce) is wrapped_reduce

    module = _fake_qwen_module()
    original_forward = lambda *args, **kwargs: None
    wrapped_forward = moe_combine._make_qwen_forward(module, original_forward)
    assert moe_combine._make_qwen_forward(module, wrapped_forward) is wrapped_forward
