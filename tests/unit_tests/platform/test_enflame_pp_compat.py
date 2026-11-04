"""CPU contracts for the GCU PP batch argument and stream synchronization."""

import ast
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize("major", [3, 4])
def test_pp_launch_uses_argument_batch_and_scopes_sync(last_rank, major):
    path = Path(__file__).resolve().parents[3] / "sglang_fl/dispatch/backends/vendor/enflame/patches/scheduler_pp_mixin.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_pp_launch_batch")
    batch = SimpleNamespace(reqs=object())
    stale_batch = SimpleNamespace(reqs=object())
    proxy, schedule_stream, current_stream, tensors = (object() for _ in range(4))
    result = SimpleNamespace(can_run_cuda_graph=True)
    events = []

    def wait(stream):
        assert stream is schedule_stream
        events.append("wait")

    def time(reqs, name, **kwargs):
        assert reqs is batch.reqs and reqs is not stale_batch.reqs
        assert kwargs["trace_only"] is True
        if "end" in name:
            assert kwargs["attrs"] == {"pp_mb_id": 1}
            events.append("end")
        else:
            events.append("start")

    def run(actual_batch, actual_proxy):
        assert actual_batch is batch and actual_proxy is proxy
        events.append("run")
        return result

    def prepare(actual_result, actual_batch):
        assert actual_result is result and actual_batch is batch
        events.append("prepare")
        return tensors

    def record(stream):
        assert stream is current_stream
        events.append("record")

    event = SimpleNamespace(record=record)
    backend = SimpleNamespace(
        cur_batch=stale_batch, forward_stream_ctx=nullcontext(),
        forward_stream=SimpleNamespace(wait_stream=wait), schedule_stream=schedule_stream,
        run_batch=run, _pp_prepare_tensor_dict=prepare,
        device_module=SimpleNamespace(Event=lambda: event, current_stream=lambda: current_stream),
        pp_group=SimpleNamespace(is_last_rank=last_rank),
    )
    namespace = dict(
        torch=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
                              gcu=SimpleNamespace(get_device_capability=lambda device: (major, 0)),
                              cuda=SimpleNamespace(synchronize=lambda: events.append("sync"))),
        set_time_batch=time, PPBatchMetadata=SimpleNamespace,
        PPProxyTensors=lambda value: SimpleNamespace(tensors=value),
    )
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    metadata = [object(), None, object()]
    retained = metadata[0], metadata[2]
    queue = deque([object()])
    returned_result, returned_event = namespace[node.name](backend, 1, batch, proxy, metadata, queue)
    assert returned_result is result and returned_event is event
    assert metadata[1].can_run_cuda_graph is True
    assert metadata[0] is retained[0] and metadata[2] is retained[1]
    assert backend.cur_batch is stale_batch
    expected = ["wait", "start", "run"] + (["sync"] if major == 3 else []) + ["end", "record"]
    if last_rank:
        expected.append("prepare")
        assert len(queue) == 2
        assert queue[-1][0] is event and queue[-1][1].tensors is tensors
    else:
        assert len(queue) == 1
    assert events == expected
