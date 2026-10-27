# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the native recurrence and MoE wrapper contracts.

The small NumPy interpreter executes the actual Triton function body, including
its pointer masks. The numerical oracle is a separate token recurrence. It
verifies padding/cache semantics without claiming GPU compilation coverage.
"""

from collections import namedtuple
from itertools import product
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_fl.dispatch.backends.vendor.iluvatar.impl import (
    fused_moe,
    gdn_triton,
    moe_triton,
)
from sglang_fl.dispatch.backends.vendor.iluvatar.patches import awq_dequantize


class _Array(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype):
        return self.astype(dtype).view(_Array)

    def __iadd__(self, other):
        # Triton values are SSA values: assigning token += 1 must not mutate
        # the separately held begin value of the sequence.
        return _Array(np.asarray(self) + other)


class _Pointer:
    def __init__(self, buffer, offset=0):
        self.buffer = buffer
        self.offset = np.asarray(offset)
        self.dtype = SimpleNamespace(element_ty=buffer.dtype)

    def __add__(self, offset):
        return _Pointer(self.buffer, self.offset + offset)


def _pointer(tensor):
    # Retain the whole backing storage: a state pool can be an envelope-strided
    # layer view whose slot stride is larger than HV*V*K.
    count = tensor.untyped_storage().nbytes() // tensor.element_size()
    storage = tensor.as_strided((count,), (1,), storage_offset=0).numpy()
    return _Pointer(storage, tensor.storage_offset())


def _load(pointer, mask=True, other=0.0):
    offsets, mask = np.broadcast_arrays(pointer.offset, mask)
    values = pointer.buffer[np.where(mask, offsets, 0)]
    return _Array(np.where(mask, values, other).astype(pointer.buffer.dtype))


def _store(pointer, values, mask=True):
    offsets, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
    pointer.buffer[offsets[mask]] = values[mask]


@pytest.fixture
def cpu_triton(monkeypatch):
    language = ModuleType("triton.language")
    language.constexpr = object()
    language.float32, language.int64 = np.float32, np.int64
    language.arange = lambda start, stop: _Array(np.arange(start, stop))
    language.load, language.store = _load, _store
    language.exp = lambda value: _Array(np.exp(value))
    language.log = lambda value: _Array(np.log(value))
    language.where = lambda condition, a, b: _Array(np.where(condition, a, b))
    language.sigmoid = lambda value: _Array(1.0 / (1.0 + np.exp(-value)))
    language.rsqrt = lambda value: _Array(1.0 / np.sqrt(value))
    language.sum = lambda value, axis: _Array(np.sum(value, axis=axis))
    program_ids = [0, 0, 0]
    language.program_id = lambda axis: program_ids[axis]

    class Jit:
        def __init__(self, function):
            self.function = function

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kwargs.pop("num_warps", None)
                kwargs.pop("num_stages", None)
                args = tuple(
                    _pointer(x) if isinstance(x, torch.Tensor) else x for x in args
                )
                for ids in product(*(range(size) for size in grid)):
                    program_ids[: len(ids)] = ids
                    self.function(*args, **kwargs)

            return launch

    triton = ModuleType("triton")
    triton.jit = Jit
    triton.language = language
    triton.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
    triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    gdn_triton._kernel.cache_clear()
    gdn_triton._packed_decode_kernel.cache_clear()
    yield
    gdn_triton._kernel.cache_clear()
    gdn_triton._packed_decode_kernel.cache_clear()


def _oracle(q, k, v, g, beta, initial, bounds, scale, normalize):
    output = torch.empty_like(v)
    final, checkpoints = [], []
    query_heads, value_heads = q.shape[1], v.shape[1]
    for sequence, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = initial[sequence].clone()
        for offset, token in enumerate(range(start, stop)):
            if offset % 64 == 0:
                checkpoints.append(state.clone())
            for value_head in range(value_heads):
                query_head = value_head // (value_heads // query_heads)
                query, key = q[token, query_head], k[token, query_head]
                if normalize:
                    query = query / (query.square().sum() + 1e-6).sqrt()
                    key = key / (key.square().sum() + 1e-6).sqrt()
                decayed = state[value_head] * g[token, value_head].exp()
                delta = (v[token, value_head] - decayed.mv(key)) * beta[
                    token, value_head
                ]
                state[value_head] = decayed + delta[:, None] * key[None, :]
                output[token, value_head] = state[value_head].mv(query) * scale
        final.append(state)
    return output, torch.stack(final), torch.stack(checkpoints)[None]


@pytest.mark.parametrize("normalize", [False, True])
def test_chunk_padding_has_zero_initial_state_and_outputs_without_cache_writeback(
    cpu_triton, normalize
):
    generator = torch.Generator().manual_seed(132)

    def random(shape):
        return torch.randn(shape, generator=generator) * 0.2

    tokens, heads, value_heads, key_dim, value_dim = 130, 1, 2, 4, 3
    q, k = random((tokens, heads, key_dim)), random((tokens, heads, key_dim))
    v = random((tokens, value_heads, value_dim))
    g = -random((tokens, value_heads)).abs()
    beta = random((tokens, value_heads)).sigmoid()
    backing = random((5, 2, value_heads, value_dim, key_dim))
    pool = backing[:, 1]
    before = backing.clone()
    slots = torch.tensor([3, -1], dtype=torch.int32)
    bounds = torch.tensor([0, 65, 130], dtype=torch.int32)
    initial = torch.stack([pool[3], torch.zeros_like(pool[0])])
    expected_output, final, expected_checkpoints = _oracle(
        q, k, v, g, beta, initial, bounds.tolist(), 0.5, normalize
    )
    out = torch.full_like(v, -99.0)
    actual, checkpoints = gdn_triton.recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        pool,
        slots,
        bounds,
        0.5,
        normalize,
        out=out,
        return_checkpoints=True,
    )

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        checkpoints[:, :4], expected_checkpoints, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(pool[3], final[0], rtol=2e-5, atol=2e-6)
    expected_backing = before.clone()
    expected_backing[3, 1] = final[0]
    torch.testing.assert_close(backing, expected_backing, rtol=2e-5, atol=2e-6)
    assert torch.equal(checkpoints[0, 2], torch.zeros_like(checkpoints[0, 2]))
    assert checkpoints[0, 3].abs().sum() > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_packed_padding_writes_zero_output_without_touching_cache(
    cpu_triton, normalize
):
    generator = torch.Generator().manual_seed(18)

    def random(shape):
        return torch.randn(shape, generator=generator) * 0.2

    batch, heads, value_heads, key_dim, value_dim = 2, 1, 2, 4, 3
    q, k = random((batch, heads, key_dim)), random((batch, heads, key_dim))
    v = random((batch, value_heads, value_dim))
    packed = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=1)
    a, b = random((batch, value_heads)), random((batch, value_heads))
    a_log, dt_bias = random((value_heads,)), random((value_heads,))
    g = -a_log.exp() * torch.nn.functional.softplus(a + dt_bias)
    beta = b.sigmoid()
    backing = random((5, 2, value_heads, value_dim, key_dim))
    pool = backing[:, 1]
    before = backing.clone()
    expected, final, _ = _oracle(
        q[:1], k[:1], v[:1], g[:1], beta[:1], pool[3:4], [0, 1], 0.5, normalize
    )
    out = torch.full_like(v, -99.0)
    actual, state = gdn_triton.packed_decode_gdn(
        packed,
        a,
        b,
        a_log,
        dt_bias,
        0.5,
        pool,
        out,
        torch.tensor([3, -1], dtype=torch.int32),
        normalize,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert state.data_ptr() == pool.data_ptr()
    torch.testing.assert_close(out[:1], expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    expected_backing = before.clone()
    expected_backing[3, 1] = final[0]
    torch.testing.assert_close(backing, expected_backing, rtol=2e-5, atol=2e-6)


class _CudaIdentity:
    is_cuda = True
    dtype = torch.float16
    shape = (2, 3)

    def dim(self):
        return 2

    def contiguous(self):
        return self


def _moe_inputs(factor):
    config = SimpleNamespace(
        num_experts=2,
        num_local_experts=2,
        activation="silu",
        routed_scaling_factor=factor,
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 8, 3), w2_weight=torch.empty(2, 3, 4)
    )
    dispatch = SimpleNamespace(
        hidden_states=_CudaIdentity(),
        topk_output=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), None),
    )
    return SimpleNamespace(moe_runner_config=config), layer, dispatch


@pytest.mark.parametrize("factor", [None, 1.0, 0.7])
def test_moe_applies_routed_scaling_after_expert_combine(monkeypatch, factor):
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    calls = []

    def experts(*args):
        calls.append(args)
        return base.clone()

    monkeypatch.setattr(moe_triton, "fused_experts", experts)
    dispatcher = ModuleType("sglang.srt.layers.moe.token_dispatcher")
    dispatcher.StandardCombineInput = namedtuple(
        "StandardCombineInput", "hidden_states"
    )
    monkeypatch.setitem(sys.modules, dispatcher.__name__, dispatcher)
    result = fused_moe.fused_moe_iluvatar(*_moe_inputs(factor))
    torch.testing.assert_close(
        result.hidden_states, base * (1.0 if factor is None else factor)
    )
    assert len(calls) == 1


@pytest.mark.parametrize("unsupported", ["router_input", "bias"])
def test_moe_unsupported_configuration_is_rejected_before_kernel(
    monkeypatch, unsupported
):
    obj, layer, dispatch = _moe_inputs(0.7)
    if unsupported == "router_input":
        obj.moe_runner_config.apply_router_weight_on_input = True
    else:
        layer.w13_weight_bias = torch.empty(2, 8)
    monkeypatch.setattr(
        moe_triton,
        "fused_experts",
        lambda *args: pytest.fail("Unsupported configuration reached the kernel"),
    )
    with pytest.raises(NotImplementedError, match="unsupported Iluvatar fused MoE"):
        fused_moe.fused_moe_iluvatar(obj, layer, dispatch)


@pytest.fixture
def awq_patch_modules(monkeypatch):
    """Isolate installer contracts; these stubs do not validate SGLang cold load."""
    from sglang_fl import utils

    monkeypatch.setattr(awq_dequantize, "_applied", False)
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name="iluvatar")
    )
    target = ModuleType("sglang.srt.hardware_backend.gpu.quantization.awq_kernels")
    triton = ModuleType("sglang.kernels.ops.quantization.awq_triton")
    custom_op = ModuleType("sglang.kernels.ops.quantization.awq_dequantize")
    original = lambda *args: ("original", args)
    replacement = lambda *args: ("triton", args)
    target.awq_dequantize = original
    triton.awq_dequantize_triton = replacement
    # The registered op and other AWQ kernels retain their own bindings.
    custom_op.awq_dequantize = original
    monkeypatch.setitem(sys.modules, custom_op.__name__, custom_op)
    target.AWQLinearKernel = SimpleNamespace(
        apply=lambda *args: target.awq_dequantize(*args)
    )
    target.AWQMarlinLinearKernel = SimpleNamespace(apply=object())
    target.AWQMoEKernel = SimpleNamespace(apply=object())
    imports = []

    def import_module(name):
        imports.append(name)
        # An unexpected eager import must fail instead of reaching real kernels.
        return {target.__name__: target, triton.__name__: triton}[name]

    monkeypatch.setattr(
        awq_dequantize, "importlib", SimpleNamespace(import_module=import_module)
    )
    return SimpleNamespace(
        target=target,
        triton=triton,
        custom_op=custom_op,
        original=original,
        replacement=replacement,
        imports=imports,
    )


@pytest.mark.parametrize("vendor", [None, "nvidia"])
def test_awq_patch_ignores_other_or_missing_device(
    monkeypatch, awq_patch_modules, vendor
):
    from sglang_fl import utils

    info = None if vendor is None else SimpleNamespace(vendor_name=vendor)
    monkeypatch.setattr(utils, "get_device_info", lambda: info)
    awq_dequantize.patch_awq_dequantize()
    assert awq_patch_modules.imports == []
    assert awq_patch_modules.target.awq_dequantize is awq_patch_modules.original
    assert not awq_dequantize._applied


def test_awq_patch_changes_only_ordinary_binding_and_is_idempotent(awq_patch_modules):
    modules = awq_patch_modules
    before = vars(modules.target).copy()
    ordinary_apply = modules.target.AWQLinearKernel.apply
    marlin_apply = modules.target.AWQMarlinLinearKernel.apply
    moe_apply = modules.target.AWQMoEKernel.apply
    assert ordinary_apply(1, 2, 3) == ("original", (1, 2, 3))

    awq_dequantize.patch_awq_dequantize()
    assert ordinary_apply(1, 2, 3) == ("triton", (1, 2, 3))
    assert modules.target.awq_dequantize is modules.replacement
    assert vars(modules.target).keys() == before.keys()
    for name, value in before.items():
        if name != "awq_dequantize":
            assert vars(modules.target)[name] is value
    assert modules.target.AWQLinearKernel.apply is ordinary_apply
    assert modules.target.AWQMarlinLinearKernel.apply is marlin_apply
    assert modules.target.AWQMoEKernel.apply is moe_apply
    assert modules.custom_op.awq_dequantize is modules.original
    assert awq_dequantize._applied

    awq_dequantize.patch_awq_dequantize()
    assert modules.imports == [modules.target.__name__, modules.triton.__name__]
    assert modules.target.awq_dequantize is modules.replacement


@pytest.mark.parametrize("binding", ["original", "replacement"])
@pytest.mark.parametrize("invalid", ["missing", "noncallable"])
def test_awq_patch_rejects_missing_binding_without_partial_install(
    awq_patch_modules, binding, invalid
):
    modules = awq_patch_modules
    module, name, valid = (
        (modules.target, "awq_dequantize", modules.original)
        if binding == "original"
        else (modules.triton, "awq_dequantize_triton", modules.replacement)
    )
    if invalid == "missing":
        delattr(module, name)
    else:
        setattr(module, name, None)
    before = vars(modules.target).copy()
    with pytest.raises(
        RuntimeError, match="requires SGLang's AWQ dequantization bindings"
    ):
        awq_dequantize.patch_awq_dequantize()
    assert vars(modules.target) == before
    assert not awq_dequantize._applied
    assert modules.custom_op.awq_dequantize is modules.original

    setattr(module, name, valid)
    awq_dequantize.patch_awq_dequantize()
    assert modules.target.awq_dequantize is modules.replacement
    assert awq_dequantize._applied


def test_awq_patch_propagates_import_failure(monkeypatch, awq_patch_modules):
    def unavailable(name):
        raise ImportError("missing AWQ runtime dependency")

    monkeypatch.setattr(awq_dequantize.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="missing AWQ runtime dependency"):
        awq_dequantize.patch_awq_dequantize()
    assert not awq_dequantize._applied
    assert awq_patch_modules.target.awq_dequantize is awq_patch_modules.original
