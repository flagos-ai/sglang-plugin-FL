"""Bridge HCU CUDA graph replay completion to sampling and PP communication."""

import itertools
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class _EventRecord:
    sequence: int
    event: object


_LOCK = threading.Lock()
_SEQUENCE = itertools.count(1)
_LATEST_REPLAY_BY_DEVICE_AND_STREAM = {}
_SAMPLE_WAITED_REPLAY_SEQUENCE = {}
_COMM_WAITED_REPLAY_SEQUENCE = {}
_SCHEDULER_SEND_EVENT_ATTR = "_hcu_pp_send_done_event"


def _stream_handle(stream) -> int:
    for attribute in ("cuda_stream", "hip_stream"):
        handle = getattr(stream, attribute, None)
        if handle is not None:
            return int(handle)
    return id(stream)


def _device_key(device, device_module, stream) -> str:
    stream_device = getattr(stream, "device", None)
    if stream_device is not None and ":" in str(stream_device):
        return str(stream_device)

    device_key = str(device)
    if ":" not in device_key:
        try:
            device_key = f"{device_key}:{int(device_module.current_device())}"
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    return device_key


def _record_current_stream_event(device_module, device, records) -> int:
    stream = device_module.current_stream(device=device)
    event = device_module.Event()
    event.record(stream)
    stream_handle = _stream_handle(stream)
    device_key = _device_key(device, device_module, stream)
    sequence = next(_SEQUENCE)
    with _LOCK:
        records[(device_key, stream_handle)] = _EventRecord(
            sequence=sequence,
            event=event,
        )
    return sequence


def _record_replay_completion(device_module, device) -> int:
    return _record_current_stream_event(
        device_module,
        device,
        _LATEST_REPLAY_BY_DEVICE_AND_STREAM,
    )


def _wait_for_replay_before_sample(device_module, device) -> tuple[int, ...]:
    current_stream = device_module.current_stream(device=device)
    device_key = _device_key(device, device_module, current_stream)
    with _LOCK:
        records = tuple(
            (stream_handle, record)
            for (record_device, stream_handle), record in (
                _LATEST_REPLAY_BY_DEVICE_AND_STREAM.items()
            )
            if record_device == device_key
            and _SAMPLE_WAITED_REPLAY_SEQUENCE.get(
                (device_key, stream_handle), 0
            )
            < record.sequence
        )

    for _, record in records:
        record.event.synchronize()

    if records:
        with _LOCK:
            for stream_handle, record in records:
                key = (device_key, stream_handle)
                _SAMPLE_WAITED_REPLAY_SEQUENCE[key] = max(
                    record.sequence,
                    _SAMPLE_WAITED_REPLAY_SEQUENCE.get(key, 0),
                )
    return tuple(record.sequence for _, record in records)


def enqueue_replay_dependency_for_pp(device_module, device) -> tuple[int, ...]:
    """Make the current PP stream wait for unseen replay completion events."""
    target_stream = device_module.current_stream(device=device)
    target_handle = _stream_handle(target_stream)
    device_key = _device_key(device, device_module, target_stream)
    with _LOCK:
        records = tuple(
            (source_handle, record)
            for (record_device, source_handle), record in (
                _LATEST_REPLAY_BY_DEVICE_AND_STREAM.items()
            )
            if record_device == device_key
            and _COMM_WAITED_REPLAY_SEQUENCE.get(
                (device_key, target_handle, source_handle), 0
            )
            < record.sequence
        )

    for source_handle, record in records:
        if source_handle != target_handle:
            target_stream.wait_event(record.event)

    if records:
        with _LOCK:
            for source_handle, record in records:
                key = (device_key, target_handle, source_handle)
                _COMM_WAITED_REPLAY_SEQUENCE[key] = max(
                    record.sequence,
                    _COMM_WAITED_REPLAY_SEQUENCE.get(key, 0),
                )
    return tuple(record.sequence for _, record in records)


def _cuda_graph_replay_hook(original_fn, runner, forward_batch, *args, **kwargs):
    result = original_fn(runner, forward_batch, *args, **kwargs)
    _record_replay_completion(runner.device_module, runner.device)
    return result


def _scheduler_uses_flagcx(scheduler) -> bool:
    pp_group = getattr(scheduler, "pp_group", None)
    communicator = getattr(pp_group, "fl_communicator", None)
    flagcx_comm = getattr(communicator, "_flagcx_comm", None)
    return flagcx_comm is not None and not getattr(flagcx_comm, "disabled", False)


def _pp_send_dict_hook(original_fn, scheduler, *args, **kwargs):
    if not _scheduler_uses_flagcx(scheduler):
        return original_fn(scheduler, *args, **kwargs)

    enqueue_replay_dependency_for_pp(scheduler.device_module, scheduler.device)
    result = original_fn(scheduler, *args, **kwargs)
    event = scheduler.device_module.Event()
    event.record(scheduler.device_module.current_stream())
    setattr(scheduler, _SCHEDULER_SEND_EVENT_ATTR, event)
    return result


def _pp_launch_batch_hook(original_fn, scheduler, *args, **kwargs):
    if _scheduler_uses_flagcx(scheduler):
        event = getattr(scheduler, _SCHEDULER_SEND_EVENT_ATTR, None)
        if event is not None:
            scheduler.forward_stream.wait_event(event)
            setattr(scheduler, _SCHEDULER_SEND_EVENT_ATTR, None)
    return original_fn(scheduler, *args, **kwargs)


def _model_runner_sample_hook(original_fn, runner, *args, **kwargs):
    device_module = getattr(runner, "device_module", None)
    if device_module is None:
        import torch

        device_module = torch.get_device_module(runner.device)

    _wait_for_replay_before_sample(device_module, runner.device)
    return original_fn(runner, *args, **kwargs)


def setup_cuda_graph_pre_sample_fence() -> None:
    """Register the required GalaxyHIP/ROCR replay dependency hooks."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner.replay",
        _cuda_graph_replay_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_pp_mixin."
        "SchedulerPPMixin._pp_send_dict_to_next_stage",
        _pp_send_dict_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_pp_mixin."
        "SchedulerPPMixin._pp_launch_batch",
        _pp_launch_batch_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.sample",
        _model_runner_sample_hook,
        HookType.AROUND,
    )
