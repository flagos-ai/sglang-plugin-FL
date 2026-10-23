# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 PD attention layouts admitted by config and wire geometry."""

# (tp_size, ep_size, dp_size, attn_tp_size, attn_dp_size)
_ADMITTED_DECODE_LAYOUTS = (
    (8, 8, 8, 1, 8),  # attention TP1 x DP8
    (8, 8, 4, 2, 4),  # attention TP2 x DP4
)

# (local_attn_tp, local_attn_dp, remote_attn_tp, remote_dp)
_ADMITTED_PEER_PAIRS = (
    (1, 8, 8, 1),  # decode attn TP1 x DP8  <- prefill attn TP8 / DP1
    (2, 4, 8, 1),  # decode attn TP2 x DP4  <- prefill attn TP8 / DP1
)


def _attention_widths(cfg):
    """(attn_tp_size, attn_dp_size) from the configured leaves.

    attn_cp_size is enforced to 1 by the caller (the V4.1 feature hook), so the
    quotient below is the attention tensor width.
    """
    attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1
    return cfg.tp_size // attn_dp_size, attn_dp_size


def allow_decode_config(cfg) -> bool:
    if cfg.disaggregation_mode != "decode":
        return False
    if not (cfg.enable_dp_attention and cfg.enable_dp_lm_head):
        return False
    if cfg.moe_a2a_backend != "none":
        return False
    attn_tp_size, attn_dp_size = _attention_widths(cfg)
    return (
        cfg.tp_size,
        cfg.ep_size,
        cfg.dp_size,
        attn_tp_size,
        attn_dp_size,
    ) in _ADMITTED_DECODE_LAYOUTS


def allow_peer_topology(
    *,
    is_mla,
    local_tp,
    local_ep,
    local_attn_tp,
    local_attn_dp,
    remote_attn_tp,
    remote_dp,
) -> bool:
    return (
        is_mla
        and local_tp == local_ep == 8
        and (local_attn_tp, local_attn_dp, remote_attn_tp, remote_dp)
        in _ADMITTED_PEER_PAIRS
    )


def validate_c2_strides(source, destination) -> None:
    # Generic state transfer uses the sender's item size to address both sides.
    # A different receiver stride must fail before issuing an RDMA/TCP write.
    if list(source) != list(destination):
        raise ValueError(
            "DeepSeek-V4.1 C2 state strides differ between P and D: "
            f"source={list(source)}, destination={list(destination)}"
        )


def _resolved_layout_init(original):
    """Remove an obsolete opt-in operand in a reviewed native constructor.

    The remaining TP/DP checks and the numerical implementation are preserved.
    This is local function replacement; process environment is never modified.
    Newer sources without the obsolete operand keep their original constructor.
    """
    import ast
    import hashlib
    import inspect
    import textwrap
    from types import FunctionType

    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)

    class ResolveLayout(ast.NodeTransformer):
        count = 0

        def visit_Compare(self, node):
            call = node.left
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "environ"
                and isinstance(call.func.value.value, ast.Name)
                and call.func.value.value.id == "os"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
                and any(
                    suffix in call.args[0].value
                    for suffix in ("DPA_PD", "ATTN_TP_REDUCE")
                )
            ):
                self.count += 1
                return ast.copy_location(ast.Constant(True), node)
            return self.generic_visit(node)

    resolve = ResolveLayout()
    function = resolve.visit(tree).body[0]
    if not resolve.count:
        return original
    reviewed = {
        "ecc2bdb96cea8a31991c4a77b026090bab49fcd2f2ea28184f62c1cb199155eb",
        "5dabbbaecf7144a88c714e3682b3f7d86a0baa78734083d2827874e4efba70d2",
    }
    if (
        resolve.count != 1
        or hashlib.sha256(source.encode()).hexdigest() not in reviewed
    ):
        raise RuntimeError("Unreviewed native DSV4.1 PD layout constructor")
    # Compiling in a class preserves zero-argument super()'s __class__ cell.
    function.decorator_list = []
    definition = ast.ClassDef(
        name="_NativeLayout", bases=[], keywords=[], body=[function], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    scope = dict(original.__globals__)
    exec(compile(module, original.__code__.co_filename, "exec"), scope)
    code = scope["_NativeLayout"].__init__.__code__
    if code.co_freevars != original.__code__.co_freevars:
        raise RuntimeError("Native PD constructor closure changed")
    replacement = FunctionType(
        code,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    replacement.__annotations__ = original.__annotations__
    return replacement


def install_native_layout_hooks():
    """Adapt the native model API without requiring the release backport."""
    import importlib
    import importlib.util
    import sys

    replacements = {}
    for name in ("dsv41_pd", "dsv41_dpa_experiment"):
        path = "sglang.srt.disaggregation." + name
        if importlib.util.find_spec(path) is None:
            continue
        native = importlib.import_module(path)
        for attr in (
            "allow_decode_config",
            "allow_peer_topology",
            "validate_c2_strides",
        ):
            old = getattr(native, attr, None)
            if old is not None:
                replacements[old] = globals()[attr]
    # Refresh aliases already bound by the native managers as well.
    for name, module in list(sys.modules.items()):
        if not name.startswith("sglang.") or module is None:
            continue
        for attr, value in list(vars(module).items()):
            if not callable(value):
                continue
            try:
                new = replacements.get(value)
            except TypeError:
                continue
            if new is not None:
                setattr(module, attr, new)
    for module, cls_name in (
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", "DSparkWorkerV2"),
        ("sglang.srt.models.deepseek_v4_dspark", "DSparkAttention"),
    ):
        cls = getattr(importlib.import_module(module), cls_name)
        cls.__init__ = _resolved_layout_init(cls.__init__)
