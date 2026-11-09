"""CPU contracts for ordinary GCU graph API wrappers, without device capture."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def graph_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "sglang_fl/dispatch/backends/vendor/enflame/patches/mtp_graph.py"
    spec = importlib.util.spec_from_file_location("enflame_graph_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name))
    return module


@pytest.mark.parametrize("keyword", [False, True])
def test_graph_keyword_becomes_positional(graph_module, keyword):
    graph, pool, stream, result = object(), object(), object(), object()
    calls = []

    def positional_graph(graph, /, **kwargs):
        calls.append((graph, kwargs))
        return result

    proxy = graph_module._GraphDeviceModule(SimpleNamespace(graph=positional_graph))
    options = dict(pool=pool, stream=stream, capture_error_mode="thread_local")
    actual = proxy.graph(cuda_graph=graph, **options) if keyword else proxy.graph(graph, **options)
    assert actual is result
    assert len(calls) == 1
    received, kwargs = calls[0]
    assert received is graph
    assert kwargs["pool"] is pool and kwargs["stream"] is stream
    assert kwargs["capture_error_mode"] == "thread_local"
    assert "cuda_graph" not in kwargs


@pytest.mark.parametrize("failure", [None, "enter", "body"])
def test_capture_forwards_arguments_and_restores_after_context(graph_module, failure):
    graph, pool, stream, shape, options, result = (object() for _ in range(6))
    error = RuntimeError("simulated capture failure")
    events = []

    class CaptureContext:
        def __enter__(self):
            events.append("enter")
            if failure == "enter":
                raise error
            return self

        def __exit__(self, kind, value, traceback):
            events.append(("exit", value))
            return False

    def positional_graph(received_graph, /, *, pool, stream, capture_error_mode):
        assert received_graph is graph
        assert pool is expected_pool and stream is expected_stream
        assert capture_error_mode == "thread_local"
        return CaptureContext()

    expected_pool, expected_stream = pool, stream
    synchronize = lambda: None
    device = SimpleNamespace(graph=positional_graph, synchronize=synchronize)
    backend = SimpleNamespace(_cuda_graph_runner=SimpleNamespace(device="gcu"), _device_module=device)
    other_backend = SimpleNamespace(_device_module=device)

    def original(self, received_shape, *, config):
        assert self is backend and received_shape is shape and config is options
        assert isinstance(self._device_module, graph_module._GraphDeviceModule)
        assert self._device_module.synchronize is synchronize
        assert other_backend._device_module is device
        with self._device_module.graph(cuda_graph=graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
            events.append("body")
            if failure == "body":
                raise error
        return result

    if failure:
        with pytest.raises(RuntimeError) as caught:
            graph_module.capture_one_gcu(original, backend, shape, config=options)
        assert caught.value is error
    else:
        assert graph_module.capture_one_gcu(original, backend, shape, config=options) is result
    assert backend._device_module is device
    assert other_backend._device_module is device
    if failure == "enter":
        assert events == ["enter"]
    else:
        assert events == ["enter", "body", ("exit", error if failure else None)]


@pytest.mark.parametrize("device_type", ["cuda", "cpu"])
def test_non_gcu_capture_delegates_without_wrapping(graph_module, device_type):
    device, positional, keyword, result = (object() for _ in range(4))
    backend = SimpleNamespace(_cuda_graph_runner=SimpleNamespace(device=device_type), _device_module=device)
    calls = []

    def original(self, arg, *, config):
        assert self is backend and self._device_module is device
        assert arg is positional and config is keyword
        calls.append(self)
        return result

    assert graph_module.capture_one_gcu(original, backend, positional, config=keyword) is result
    assert calls == [backend]
    assert backend._device_module is device


def test_repeated_capture_does_not_accumulate_wrappers(graph_module):
    device = SimpleNamespace(synchronize=lambda: None)
    backend = SimpleNamespace(_cuda_graph_runner=SimpleNamespace(device="gcu"), _device_module=device)
    seen = []

    def original(self):
        proxy = self._device_module
        assert isinstance(proxy, graph_module._GraphDeviceModule)
        assert proxy._module is device
        seen.append(proxy)

    for _ in range(2):
        graph_module.capture_one_gcu(original, backend)
        assert backend._device_module is device
    assert len(seen) == 2 and seen[0] is not seen[1]


def _buffer_module():
    path = Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/backends/vendor/enflame/patches/buffer_copy.py"
    spec = importlib.util.spec_from_file_location("enflame_buffer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grouped_copy_updates_existing_strided_storage():
    import torch
    from unittest.mock import Mock

    module = _buffer_module()

    class GCUBuffer:
        device = SimpleNamespace(type="gcu")

        def __init__(self, tensor):
            self.tensor = tensor

        def copy_(self, source):
            self.tensor.copy_(source)

    storage = torch.full((3, 4), -1.0)
    dst1, dst2 = storage[:, ::2], torch.zeros(2)
    refs = dst1.view_as(dst1), dst2.view_as(dst2)
    addresses = dst1.data_ptr(), dst2.data_ptr()
    src1, src2 = torch.arange(6.0).reshape(3, 2), torch.tensor([8.0, 9.0])
    original = Mock()
    for delta in (0, 10):
        module.grouped_copy_gcu(original, [GCUBuffer(dst1), dst2], [src1 + delta, src2 + delta])
        torch.testing.assert_close(refs[0], src1 + delta)
        torch.testing.assert_close(refs[1], src2 + delta)
        assert (dst1.data_ptr(), dst2.data_ptr()) == addresses
    torch.testing.assert_close(storage[:, 1::2], torch.full((3, 2), -1.0))
    original.assert_not_called()


def test_grouped_copy_other_devices_preserves_original_call():
    import torch
    from unittest.mock import Mock

    destinations, sources, result = [torch.zeros(2)], [torch.ones(2)], object()
    original = Mock(return_value=result)
    assert _buffer_module().grouped_copy_gcu(original, destinations, sources) is result
    original.assert_called_once_with(destinations, sources)
    torch.testing.assert_close(destinations[0], torch.zeros(2))
