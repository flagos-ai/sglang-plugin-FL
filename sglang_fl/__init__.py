"""SGLang OOT Plugin — FlagGems-based multi-chip adaptation.

Two entry_points are registered:
  1. Platform Plugin (sglang.srt.platforms): activate_platform()
     Provides PlatformFL — device identity, memory, dist backend, graph capture.
  2. General Plugin (sglang.srt.plugins): load_plugin()
     Registers ATen ops (FlagGems), fused kernel dispatch (AROUND hook),
     and communicator hooks.

Loading:
  Plugin is auto-discovered via setuptools entry_points after `pip install`.
  No env vars needed. Optional: SGLANG_PLUGINS to filter, SGLANG_PLATFORM to select.

This plugin registers:
  Layer 1: FlagGems ATen operator replacement (flag_gems.enable)
  Layer 2: SGLang fused kernels via MultiPlatformOp.register_oot_forward +
           HookRegistry AROUND hook on dispatch_forward
  Layer 3: FlagCX communicator (via Platform Plugin get_communicator_class)

Environment variables:
  USE_FLAGGEMS=1|0                    Master switch for Layer 1 (default: 1)
  SGLANG_FL_FLAGOS_WHITELIST=op1,op2  Only these ATen ops use FlagGems
  SGLANG_FL_FLAGOS_BLACKLIST=op1,op2  These ATen ops don't use FlagGems
  SGLANG_FLAGGEMS_RECORD=1|0          Enable FlagGems logging (default: 0)
  SGLANG_FLAGGEMS_LOG_ONCE=1|0        Log only once per op (default: 1)
  SGLANG_FLAGGEMS_LOG_PATH=<path>     Log output path
  SGLANG_FL_OOT_ENABLED=1|0           Master switch for Layer 2 (default: 1)
  SGLANG_FL_PREFER=flagos|vendor|reference  Global backend priority
  SGLANG_FL_PER_OP=op=kind|kind;...         Per-op backend override
  SGLANG_FL_STRICT=1|0                Fallback on error (default: 1=enabled)
  SGLANG_FL_OOT_WHITELIST=op1,op2     Only dispatch listed ops through OOT
  SGLANG_FL_OOT_BLACKLIST=op1,op2     Skip listed ops from OOT dispatch
  SGLANG_FL_DENY_VENDORS=v1,v2       Deny specific vendor backends
  SGLANG_FL_ALLOW_VENDORS=v1,v2      Only allow these vendor backends
  SGLANG_FL_DISPATCH_LOG=<path>       Dispatch log file path
  SGLANG_FL_DISPATCH_DEBUG=1|0        Verbose operator selection logging
  SGLANG_FL_LOG_LEVEL=DEBUG|INFO|...  Dispatch module log level
  SGLANG_FL_CONFIG=<path>             YAML config file (overrides platform defaults)
  SGLANG_FL_DIST_BACKEND=nccl|hccl|flagcx   Override distributed backend
  FLAGCX_PATH=<path>                         If set, default to flagcx backend
"""

import logging
import multiprocessing
import os

import torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _is_rank0() -> bool:
    """Return True if this is the main process (rank 0) — safe to call at any stage."""
    # During load_plugin(): dist not initialized yet, use multiprocessing parent check
    if multiprocessing.parent_process() is not None:
        return False  # We are a spawned subprocess (tp_rank >= 1)
    return True


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _parse_bool(val: str, default: bool = False) -> bool:
    return val.strip().lower() in ("1", "true", "yes") if val else default


def _parse_list(val: str) -> list:
    return [x.strip() for x in val.split(",") if x.strip()] if val else []


def _parse_op_prefer(val: str) -> dict:
    """Parse SGLANG_FL_PER_OP="silu_and_mul=flagos|vendor;rms_norm=vendor"
    Returns dict: {op_name -> list[str]}
    e.g. {"silu_and_mul": ["flagos", "vendor"], "rms_norm": ["vendor"]}
    """
    result = {}
    for item in [p.strip() for p in val.split(";") if p.strip()]:
        if "=" not in item:
            continue
        op_name, order_str = item.split("=", 1)
        op_name = op_name.strip()
        order = [x.strip() for x in order_str.split("|") if x.strip()]
        if op_name and order:
            result[op_name] = order
    return result


# ─── Config builder ───────────────────────────────────────────────────────────


def _build_config() -> dict:
    """Build unified config: env vars > YAML (via SGLANG_FL_CONFIG) > defaults.

    Priority chain:
      env vars (SGLANG_FL_*, SGLANG_FLAGGEMS_*)
        > YAML config (SGLANG_FL_CONFIG or platform auto-detect)
          > code defaults
    """
    from sglang_fl.dispatch.config import get_effective_config

    yaml_cfg = get_effective_config()

    # prefer: SGLANG_FL_PREFER > yaml.prefer > "flagos"
    prefer = (
        os.environ.get("SGLANG_FL_PREFER", "").strip()
        or yaml_cfg.get("prefer", "")
        or "flagos"
    ).lower()

    # op_backends: SGLANG_FL_PER_OP > yaml.op_backends > {}
    per_op_str = os.environ.get("SGLANG_FL_PER_OP", "").strip()
    if per_op_str:
        op_backends = _parse_op_prefer(per_op_str)
    else:
        op_backends = yaml_cfg.get("op_backends", {})
        # Normalize: yaml values may be lists already or strings
        if isinstance(op_backends, dict):
            for k, v in op_backends.items():
                if isinstance(v, str):
                    op_backends[k] = [v]

    # blacklist: SGLANG_FL_OOT_BLACKLIST > yaml.oot_blacklist > []
    blacklist_str = os.environ.get("SGLANG_FL_OOT_BLACKLIST", "").strip()
    if blacklist_str:
        oot_blacklist = _parse_list(blacklist_str)
    else:
        oot_blacklist = yaml_cfg.get("oot_blacklist", []) or []

    oot_whitelist = _parse_list(os.environ.get("SGLANG_FL_OOT_WHITELIST", ""))

    # flagos_blacklist: SGLANG_FL_FLAGOS_BLACKLIST > yaml.flagos_blacklist > []
    flagos_bl_str = os.environ.get("SGLANG_FL_FLAGOS_BLACKLIST", "").strip()
    if flagos_bl_str:
        flagos_blacklist = _parse_list(flagos_bl_str)
    else:
        flagos_blacklist = yaml_cfg.get("flagos_blacklist", []) or []

    # strict: SGLANG_FL_STRICT > yaml.strict > True (default: fallback enabled)
    strict_str = os.environ.get("SGLANG_FL_STRICT", "").strip()
    if strict_str:
        strict = strict_str == "1"
    else:
        strict = yaml_cfg.get("strict", True)

    # deny_vendors: SGLANG_FL_DENY_VENDORS > yaml.deny_vendors > []
    deny_str = os.environ.get("SGLANG_FL_DENY_VENDORS", "").strip()
    if deny_str:
        deny_vendors = set(_parse_list(deny_str))
    else:
        dv_raw = yaml_cfg.get("deny_vendors", []) or []
        deny_vendors = set(dv_raw) if dv_raw else set()

    # allow_vendors: SGLANG_FL_ALLOW_VENDORS > yaml.allow_vendors > None
    allow_str = os.environ.get("SGLANG_FL_ALLOW_VENDORS", "").strip()
    if allow_str:
        allow_vendors = set(_parse_list(allow_str))
    else:
        av_raw = yaml_cfg.get("allow_vendors", None)
        allow_vendors = set(av_raw) if av_raw else None

    # dispatch_log: SGLANG_FL_DISPATCH_LOG
    dispatch_log = os.environ.get("SGLANG_FL_DISPATCH_LOG", "").strip()

    # flaggems settings (unchanged env vars, no YAML mapping)
    flaggems_record = _parse_bool(os.environ.get("SGLANG_FLAGGEMS_RECORD", "0"))
    flaggems_log_path = os.environ.get("SGLANG_FLAGGEMS_LOG_PATH", "").strip()

    return {
        "prefer": prefer,
        "op_backends": op_backends,
        "strict": strict,
        "deny_vendors": deny_vendors,
        "allow_vendors": allow_vendors,
        "oot_blacklist": oot_blacklist,
        "oot_whitelist": oot_whitelist,
        "flagos_blacklist": flagos_blacklist,
        "dispatch_log": dispatch_log,
        "flaggems_record": flaggems_record,
        "flaggems_log_path": flaggems_log_path,
    }


# ─── Dispatch initialization ──────────────────────────────────────────────────


def _init_dispatch(config: dict) -> None:
    """Initialize the dispatch system with config-derived policy.

    Maps the legacy config dict to the new dispatch policy system,
    then triggers OpManager initialization (which auto-discovers backends).
    """
    from sglang_fl.dispatch import (
        SelectionPolicy,
        get_default_manager,
        set_global_policy,
    )

    # Map our config to dispatch policy
    prefer = config.get("prefer", "flagos")
    op_backends = config.get("op_backends", {})

    # Convert op_backends format to per_op_order
    # op_backends: {"rms_norm": ["vendor", "flagos"]} -> per_op_order
    # Also accepts class names for backward compatibility: {"RMSNorm": [...]}
    per_op_order = {}
    if op_backends:
        _CLASS_TO_OP = {
            "SiluAndMul": "silu_and_mul",
            "RMSNorm": "rms_norm",
            "RotaryEmbedding": "rotary_embedding",
            "TopK": "topk",
            "UnquantizedFusedMoEMethod": "fused_moe",
        }
        for key, val in op_backends.items():
            op_name = _CLASS_TO_OP.get(key, key)  # class name -> op_name, or keep as-is
            per_op_order[op_name] = val if isinstance(val, list) else [val]

    policy = SelectionPolicy.from_dict(
        prefer=prefer,
        strict=config.get("strict", True),
        per_op_order=per_op_order if per_op_order else None,
        deny_vendors=config.get("deny_vendors"),
        allow_vendors=config.get("allow_vendors"),
    )
    set_global_policy(policy)

    # Force initialization (registers all backends)
    get_default_manager().ensure_initialized()


# ─── Dispatch AROUND hook ────────────────────────────────────────────────────


def _make_dispatch_hook(config: dict = None):
    """Build the AROUND hook for MultiPlatformOp.dispatch_forward.

    The hook intercepts dispatch_forward() for ops that have bridge-layer
    implementations. For all other ops it falls through to
    the original SGLang dispatch logic (CUDA/HIP/etc.).

    Respects config["oot_whitelist"] / config["oot_blacklist"]:
      - OOT_WHITELIST: only listed ops use OOT dispatch (comma-separated class names)
      - OOT_BLACKLIST: listed ops skip OOT dispatch (comma-separated class names)
      - Cannot set both simultaneously.
      - Empty (default): all registered ops use OOT dispatch.
    """
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm, GemmaRMSNorm
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang_fl.dispatch.bridge import (
        silu_and_mul_bridge,
        rms_norm_bridge,
        gemma_rms_norm_bridge,
        rotary_embedding_bridge,
        mrotary_embedding_bridge,
        topk_bridge,
        fused_moe_bridge,
    )

    if config is None:
        dispatch_log_path = os.environ.get("SGLANG_FL_DISPATCH_LOG", "").strip()
        whitelist = _parse_list(os.environ.get("SGLANG_FL_OOT_WHITELIST", ""))
        blacklist = _parse_list(os.environ.get("SGLANG_FL_OOT_BLACKLIST", ""))
    else:
        dispatch_log_path = config.get("dispatch_log", "")
        whitelist = config.get("oot_whitelist", [])
        blacklist = config.get("oot_blacklist", [])

    _log_file = open(dispatch_log_path, "a") if dispatch_log_path else None
    if whitelist and blacklist:
        raise ValueError(
            "Cannot set both SGLANG_FL_OOT_WHITELIST and SGLANG_FL_OOT_BLACKLIST. "
            "Use one or the other."
        )
    if whitelist:
        logger.info(f"OOT dispatch whitelist: {whitelist}")
    if blacklist:
        logger.info(f"OOT dispatch blacklist: {blacklist}")

    # Map SGLang op classes to their bridge functions (via MRO inheritance)
    _BRIDGE_MAP = {
        SiluAndMul: silu_and_mul_bridge,
        RMSNorm: rms_norm_bridge,
        GemmaRMSNorm: gemma_rms_norm_bridge,
        RotaryEmbedding: rotary_embedding_bridge,
        MRotaryEmbedding: mrotary_embedding_bridge,
        TopK: topk_bridge,
        UnquantizedFusedMoEMethod: fused_moe_bridge,
    }

    def _find_bridge(cls):
        """Walk MRO to find a bridge function for the given class."""
        for parent in cls.__mro__:
            if parent in _BRIDGE_MAP:
                return _BRIDGE_MAP[parent]
        return None

    def _dispatch_hook(original_fn, self):
        # Fast path: object-level cache (avoids MRO walk + bound method creation)
        cached = getattr(self, "_fl_cached_fn", None)
        if cached is not None:
            return cached

        op_cls = type(self)
        op_name = op_cls.__name__

        # P1: Whitelist/blacklist gate
        if whitelist and op_name not in whitelist:
            return original_fn(self)
        if blacklist and op_name in blacklist:
            return original_fn(self)

        # Find bridge function for this op (supports subclasses via MRO)
        bridge_fn = _find_bridge(op_cls)
        if bridge_fn is None:
            # No bridge registered: return forward_cuda directly to bypass dispatch_forward.
            # This allows unregistered OOT ops (e.g., MoE before bridge is added) to use
            # their native CUDA implementation instead of falling into forward_native.
            result = self.forward_cuda
            self._fl_cached_fn = result
            return result

        if _log_file:
            _log_file.write(f"[OOT-DISPATCH] {op_name} → dispatch\n")
            _log_file.flush()

        # Return a bound method that calls the bridge function
        # The bridge function has the same signature as forward_cuda
        result = bridge_fn.__get__(self, op_cls)
        self._fl_cached_fn = result
        return result

    return _dispatch_hook


# ─── FlagGems setup ───────────────────────────────────────────────────────────


def _setup_flaggems(config: dict = None):
    use_fg = _parse_bool(os.environ.get("USE_FLAGGEMS", "1"), default=True)

    if not use_fg:
        logger.info("FlagGems disabled (USE_FLAGGEMS=0)")
        return

    import flag_gems

    # Whitelist/blacklist: env > YAML config
    if config is not None:
        record = config.get("flaggems_record", False)
        log_path = config.get("flaggems_log_path", "")
        blacklist = config.get("flagos_blacklist", [])
    else:
        record = _parse_bool(os.environ.get("SGLANG_FLAGGEMS_RECORD", "0"))
        log_path = os.environ.get("SGLANG_FLAGGEMS_LOG_PATH", "").strip()
        blacklist = []

    whitelist_str = os.environ.get("SGLANG_FL_FLAGOS_WHITELIST", "").strip()
    blacklist_str = os.environ.get("SGLANG_FL_FLAGOS_BLACKLIST", "").strip()

    if whitelist_str and blacklist_str:
        raise ValueError(
            "Cannot set both SGLANG_FL_FLAGOS_WHITELIST and SGLANG_FL_FLAGOS_BLACKLIST. "
            "Use one or the other."
        )

    whitelist = _parse_list(whitelist_str)
    if blacklist_str:
        blacklist = _parse_list(blacklist_str)

    fg_kwargs = dict(
        record=record,
        once=_parse_bool(os.environ.get("SGLANG_FLAGGEMS_LOG_ONCE", "1"), default=True),
    )
    if log_path:
        fg_kwargs["path"] = log_path

    if whitelist:
        flag_gems.only_enable(include=whitelist, **fg_kwargs)
        logger.info(f"FlagGems only_enable: {whitelist}")
    elif blacklist:
        flag_gems.enable(unused=blacklist, **fg_kwargs)
        logger.info(f"FlagGems enable (excluding: {blacklist})")
    else:
        flag_gems.enable(**fg_kwargs)
        logger.info("FlagGems enable: ALL ATen ops replaced with Triton kernels")

    # Filter gems_aten log to only record Layer 1 (ATen dispatch) calls.
    # Exclude Layer 2 flagos calls (flag_gems.fused.*, flag_gems.modules.*)
    # so the log purely reflects ATen operator replacement.
    if fg_kwargs.get("record"):
        import logging as _logging

        class _AtenOnlyFilter(_logging.Filter):
            """Only pass log records from ATen-replacement ops (Layer 1).

            Matches flag_gems.ops.* (standard Triton ops) and
            flag_gems.runtime.backend.<vendor>.ops.* (OOT vendor ops).
            Excludes fused/modules loggers which are Layer 2 internal calls.
            """

            def filter(self, record):
                name = record.name
                return name.startswith("flag_gems.ops") or ".ops." in name

        fg_logger = _logging.getLogger("flag_gems")
        for h in fg_logger.handlers:
            if getattr(h, "_flaggems_owned", False):
                h.addFilter(_AtenOnlyFilter())


# ─── Vendor-specific sglang patches ───────────────────────────────────────────


def _apply_vendor_patches() -> None:
    """Import vendor/<vendor_name>/patch.py to apply vendor monkey-patches
    on sglang internals. Called last in load_plugin(), after every sglang_fl
    layer (FlagGems ATen, dispatch system, AROUND hooks, communicator).
    Resolves vendor_name via FlagGems' DeviceDetector — no PlatformFL needed,
    so this still runs before sglang's model_runner is imported. Silently
    skips when the vendor module is absent or hardware is unrecognised.
    """
    import importlib

    try:
        try:
            # FlagGems<=5.0.2: DeviceDetector lives in device.
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            # FlagGems>5.0.2: DeviceDetector lives in device_finder.
            from flag_gems.runtime.backend.device_finder import DeviceDetector

        vendor = DeviceDetector().vendor_name
    except Exception as e:
        logger.warning("vendor patch skipped: DeviceDetector failed (%s)", e)
        return

    module = f"sglang_fl.dispatch.backends.vendor.{vendor}.patch"
    try:
        importlib.import_module(module)
        logger.info("vendor patch loaded: %s", module)
    except ImportError:
        logger.info("vendor patch absent: %s", module)


# ─── Communicator AROUND hooks ────────────────────────────────────────────────


def _setup_communicator_hooks():
    """Register AROUND hooks on GroupCoordinator to inject CommunicatorFL.

    Auto-activates when the Platform Plugin is OOT. The CommunicatorFL
    transparently routes through FlagCX (if available) or torch.distributed.

    Hooks (12 total):
      - __init__: inject self.fl_communicator, suppress PyNccl when FlagCX active
      - all_reduce, _reduce_scatter_tensor, _all_gather_into_tensor,
        reduce_scatterv, all_gatherv, send, recv, broadcast: delegate to fl_communicator
      - broadcast_tensor_dict, send_tensor_dict, recv_tensor_dict:
        full method intercept for FlagCX coverage on composite operations
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _GC_TARGET = "sglang.srt.distributed.parallel_state.GroupCoordinator"

    # ── __init__ hook: create and attach CommunicatorFL ──

    def _init_hook(original_fn, self, *args, **kwargs):
        # Run original __init__ first
        original_fn(self, *args, **kwargs)
        # Attach CommunicatorFL if world_size > 1
        if getattr(self, "world_size", 1) > 1:
            try:
                from sglang_fl.distributed.communicator import CommunicatorFL

                self.fl_communicator = CommunicatorFL(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    world_size=self.world_size,
                    rank_in_group=self.rank_in_group,
                    ranks=self.ranks,
                )
                # Suppress PyNccl when FlagCX is active to avoid conflicts
                if (
                    self.fl_communicator
                    and self.fl_communicator._flagcx_comm
                    and hasattr(self, "pynccl_comm")
                    and self.pynccl_comm is not None
                ):
                    self.pynccl_comm.disabled = True
            except Exception as e:
                logger.warning(f"CommunicatorFL creation failed: {e}")
                self.fl_communicator = None
        else:
            self.fl_communicator = None

    # ── all_reduce hook ──

    def _all_reduce_hook(original_fn, self, input_):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            return comm.all_reduce(input_)
        return original_fn(self, input_)

    # ── reduce_scatter_tensor hook ──

    def _reduce_scatter_tensor_hook(original_fn, self, output, input_):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            comm.reduce_scatter(output, input_)
            return
        return original_fn(self, output, input_)

    # ── all_gather_into_tensor hook ──

    def _all_gather_into_tensor_hook(original_fn, self, output, input_):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            comm.all_gather(output, input_)
            return
        return original_fn(self, output, input_)

    # ── reduce_scatterv hook ──

    def _reduce_scatterv_hook(original_fn, self, input_, output=None, sizes=None):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            return comm.reduce_scatterv(input_, output=output, sizes=sizes)
        return original_fn(self, input_, output=output, sizes=sizes)

    # ── all_gatherv hook ──

    def _all_gatherv_hook(original_fn, self, input_, sizes=None):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            return comm.all_gatherv(input_, sizes=sizes)
        return original_fn(self, input_, sizes=sizes)

    # ── send hook ──

    def _send_hook(original_fn, self, tensor, dst=None):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            if dst is None:
                dst = (self.rank_in_group + 1) % self.world_size
            comm.send(tensor, dst)
            return
        return original_fn(self, tensor, dst=dst)

    # ── recv hook ──

    def _recv_hook(original_fn, self, size, dtype, src=None):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            if src is None:
                src = (self.rank_in_group - 1) % self.world_size
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            comm.recv(tensor, src)
            return tensor
        return original_fn(self, size, dtype, src=src)

    # ── broadcast hook ──

    def _broadcast_hook(original_fn, self, input_, src=0):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            return comm.broadcast(input_, src)
        return original_fn(self, input_, src)

    # ── broadcast_tensor_dict hook ──

    def _broadcast_tensor_dict_hook(
        original_fn, self, tensor_dict=None, src=0, group=None, metadata_group=None
    ):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return tensor_dict
            return comm.broadcast_tensor_dict(
                tensor_dict=tensor_dict,
                src=src,
                rank_in_group=self.rank_in_group,
                broadcast_object_fn=self.broadcast_object,
            )
        return original_fn(
            self,
            tensor_dict=tensor_dict,
            src=src,
            group=group,
            metadata_group=metadata_group,
        )

    # ── send_tensor_dict hook ──

    def _send_tensor_dict_hook(
        original_fn,
        self,
        tensor_dict,
        dst=None,
        all_gather_group=None,
        async_send=False,
    ):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            if self.world_size == 1:
                return tensor_dict
            if dst is None:
                dst = (self.rank_in_group + 1) % self.world_size
            return comm.send_tensor_dict(
                tensor_dict=tensor_dict,
                dst=dst,
                send_object_fn=self.send_object,
                all_gather_group=all_gather_group,
            )
        return original_fn(
            self,
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            async_send=async_send,
        )

    # ── recv_tensor_dict hook ──

    def _recv_tensor_dict_hook(
        original_fn,
        self,
        src=None,
        all_gather_group=None,
    ):
        comm = getattr(self, "fl_communicator", None)
        if comm is not None and not comm.disabled:
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return None
            if src is None:
                src = (self.rank_in_group - 1) % self.world_size
            return comm.recv_tensor_dict(
                src=src,
                recv_object_fn=self.recv_object,
                all_gather_group=all_gather_group,
            )
        return original_fn(self, src=src, all_gather_group=all_gather_group)

    # ── Register all hooks ──

    HookRegistry.register(f"{_GC_TARGET}.__init__", _init_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.all_reduce", _all_reduce_hook, HookType.AROUND)
    HookRegistry.register(
        f"{_GC_TARGET}._reduce_scatter_tensor",
        _reduce_scatter_tensor_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_GC_TARGET}._all_gather_into_tensor",
        _all_gather_into_tensor_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_GC_TARGET}.reduce_scatterv", _reduce_scatterv_hook, HookType.AROUND
    )
    HookRegistry.register(
        f"{_GC_TARGET}.all_gatherv", _all_gatherv_hook, HookType.AROUND
    )
    HookRegistry.register(f"{_GC_TARGET}.send", _send_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.recv", _recv_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.broadcast", _broadcast_hook, HookType.AROUND)
    HookRegistry.register(
        f"{_GC_TARGET}.broadcast_tensor_dict",
        _broadcast_tensor_dict_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_GC_TARGET}.send_tensor_dict", _send_tensor_dict_hook, HookType.AROUND
    )
    HookRegistry.register(
        f"{_GC_TARGET}.recv_tensor_dict", _recv_tensor_dict_hook, HookType.AROUND
    )

    logger.info(
        "CommunicatorFL AROUND hooks registered on GroupCoordinator "
        "(12 hooks: init + 8 collectives + broadcast_tensor_dict + send/recv_tensor_dict)"
    )


# ─── Platform Plugin entry point ─────────────────────────────────────────────


def activate_platform() -> str | None:
    """Entry point for sglang.srt.platforms — called during platform discovery.

    Returns the fully-qualified class path of PlatformFL if hardware is detected,
    or None if FlagGems DeviceDetector fails (no supported hardware).
    """
    try:
        try:
            # FlagGems<=5.0.2: DeviceDetector lives in device.
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            # FlagGems>5.0.2: DeviceDetector lives in device_finder.
            from flag_gems.runtime.backend.device_finder import DeviceDetector

        detector = DeviceDetector()
        logger.info(
            "sglang_fl platform activating: vendor=%s, device=%s",
            detector.vendor_name,
            detector.name,
        )
        return "sglang_fl.platform:PlatformFL"
    except Exception as e:
        logger.warning("sglang_fl platform activation failed: %s", e)
        return None


# ─── General Plugin entry point ──────────────────────────────────────────────

_plugin_loaded = False


def load_plugin():
    """Entry point for sglang.srt.plugins — called by SGLang's load_plugins().

    Idempotent: safe to call multiple times.
    """
    global _plugin_loaded
    if _plugin_loaded:
        return
    _plugin_loaded = True

    # Suppress info logs on non-rank-0 processes to avoid duplicate output
    if not _is_rank0():
        logger.setLevel(logging.WARNING)

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # 0. Build unified config (YAML + env vars)
    config = _build_config()

    # 1. FlagGems ATen ops
    _setup_flaggems(config)

    # 2. Initialize dispatch system (OpManager + backends + policy)
    _init_dispatch(config)

    # 3. Install dispatch AROUND hook (bridge layer → dispatch.call_op)
    oot_enabled = _parse_bool(
        os.environ.get("SGLANG_FL_OOT_ENABLED", "1"), default=True
    )
    if oot_enabled:
        HookRegistry.register(
            "sglang.srt.layers.utils.multi_platform.MultiPlatformOp.dispatch_forward",
            _make_dispatch_hook(config),
            HookType.AROUND,
        )
        # Patch FLA functions to use dispatch mechanism
        from sglang_fl.dispatch.fla_patch import patch_fla_functions

        patch_fla_functions()

        # Patch RotaryEmbedding.__init__ to restore bridge after MUSA _forward_method stomp
        # Must be called after the AROUND hook on dispatch_forward is registered (above).
        from sglang_fl.dispatch.rotary_patch import patch_rotary_embedding_init

        patch_rotary_embedding_init()
    else:
        logger.info("Layer 2 (Fused Ops) disabled (SGLANG_FL_OOT_ENABLED=0)")

    # 4. Communicator hooks (CommunicatorFL with FlagCX/torch.distributed)
    _setup_communicator_hooks()

    # 5. Vendor-specific patches — final overlay on top of all sglang_fl layers
    _apply_vendor_patches()

    # 6. Summary banner — confirm plugin is active (rank 0 only)
    if _is_rank0():
        use_fg = _parse_bool(os.environ.get("USE_FLAGGEMS", "1"), default=True)
        aten_status = "OFF" if not use_fg else "ON"
        oot_status = f"prefer={config['prefer']}" if oot_enabled else "OFF"
        # Reflects env-var configuration only; vendor-specific defaults (e.g. hccl
        # on Ascend) are not shown here — see PlatformFL init log for the full result.
        dist_backend = os.environ.get("SGLANG_FL_DIST_BACKEND", "").strip() or (
            "flagcx" if os.environ.get("FLAGCX_PATH", "").strip() else "nccl"
        )
        banner = (
            "\n"
            "+" + "=" * 58 + "+\n"
            "|" + "sglang_fl activated".center(58) + "|\n"
            "+" + "-" * 58 + "+\n"
            "|" + f"  Layer 1 (ATen -> FlagGems):  {aten_status}".ljust(58) + "|\n"
            "|" + f"  Layer 2 (Fused Ops):         {oot_status}".ljust(58) + "|\n"
            "|" + f"  Layer 3 (Communication):     {dist_backend}".ljust(58) + "|\n"
            "+" + "=" * 58 + "+"
        )
        logger.info(banner)
