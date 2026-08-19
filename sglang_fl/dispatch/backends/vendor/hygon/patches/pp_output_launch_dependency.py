"""Order HCU PP output forwarding after local model execution."""

import logging


logger = logging.getLogger(__name__)


def _scheduler_uses_flagcx(scheduler) -> bool:
    pp_group = getattr(scheduler, "pp_group", None)
    communicator = getattr(pp_group, "fl_communicator", None)
    flagcx_comm = getattr(communicator, "_flagcx_comm", None)
    return flagcx_comm is not None and not getattr(flagcx_comm, "disabled", False)


def _pp_send_dict_hook(original_fn, scheduler, *args, **kwargs):
    if not _scheduler_uses_flagcx(scheduler):
        return original_fn(scheduler, *args, **kwargs)

    msg_type = kwargs.get("msg_type", "default")
    if len(args) >= 3:
        msg_type = args[2]

    if msg_type == "output" and not scheduler.pp_group.is_last_rank:
        launch_event = getattr(scheduler, "launch_event", None)
        if launch_event is not None:
            scheduler.device_module.current_stream(
                device=scheduler.device
            ).wait_event(launch_event)

    return original_fn(scheduler, *args, **kwargs)


def setup_pp_output_launch_dependency() -> None:
    """Register the HCU ordering dependency for non-final PP output sends."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.managers.scheduler_pp_mixin."
        "SchedulerPPMixin._pp_send_dict_to_next_stage",
        _pp_send_dict_hook,
        HookType.AROUND,
    )
    logger.warning("HCU PP output launch dependency enabled")
