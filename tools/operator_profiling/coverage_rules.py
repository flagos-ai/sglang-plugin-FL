# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Operator identities for the FlagOS inference coverage specification.

Full kernel names are retained in reports. Only identities are normalized;
different numbered stages and compile kernels remain distinct.
"""

from __future__ import annotations

import re

from .trace_report import stable_kernel_name

INDUCTOR_PREFIXES = ("triton_poi_", "triton_red_", "triton_per_", "triton_for_")


def normalized_kernel(name: str) -> str:
    # A generic launcher's template argument identifies the computation.
    # Dropping it would collapse unrelated CUTLASS/elementwise functors.
    if any(token in name for token in ("device_kernel<", "elementwise_kernel<")):
        result = name.removeprefix("void ").strip()
    else:
        result = stable_kernel_name(name)
        if "::" not in result:
            result = result.split("<", 1)[0].split("(", 1)[0]
    return re.sub(r"(?:_kernel)?_(?:rank|device)_?\d+$", "", result)


def descriptor(operator: str, kernel: str, *, kind: str = "", compile_module: str = ""):
    """Return (display kind, normalized identity, normalization reason)."""
    normalized = normalized_kernel(kernel)
    if kind == "communication":
        return (
            kind,
            (kind, operator, normalized),
            "communication API + normalized kernel",
        )
    if (
        kind != "torch_compile"
        and not kernel.startswith(INDUCTOR_PREFIXES)
        and operator.startswith("aten::")
        and "," not in operator
    ):
        return (
            "aten",
            ("aten", operator.split(".", 1)[0]),
            "ATen API (overloads merged)",
        )
    if kind == "torch_compile" or kernel.startswith(INDUCTOR_PREFIXES):
        # JIT markers may repeat the kernel name as the operator. That is not
        # module provenance: use the same unknown module as an unmarked replay.
        module = compile_module or ("null" if operator == kernel else operator)
        return (
            "torch_compile",
            ("torch_compile", module, normalized),
            "compile module + kernel (stages preserved)",
        )
    resolved_kind = kind or ("unattributed" if operator == "null" else "custom")
    return resolved_kind, ("kernel", normalized), "normalized kernel (stages preserved)"
