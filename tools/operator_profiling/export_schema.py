# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public CSV contract shared with cyber-pioneer/op_profile.

Presentation only: identities and raw evidence remain in the normalization
and audit files. Taxonomy follows vllm-plugin-FL PR #593 rule_map.py.
"""

from __future__ import annotations

import json
import re

OPERATOR_FIELDS = [
    "operator_id",
    "operator_name",
    "operator_kind",
    "kernel_name",
    "kernel_time_percent(%)",
]
DETAIL_FIELDS = [
    "operator_name",
    "kernel_name",
    "variant_index",
    "mapping_status",
    "input_shapes",
    "input_dtypes",
    "candidate_operators",
    "kernel_event_count",
    "kernel_time_us",
    "eager_operator_name",
    "eager_input_shapes",
    "eager_input_dtypes",
    "eager_candidates",
    "mapping_source",
]
AUDIT_FILE = "kernel_shape_dtype_audit.csv"
COLLECTIVES = (
    "all_gather",
    "all_reduce",
    "allgather",
    "allreduce",
    "cross_device_reduce",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
)
BACKENDS = ("flagcx", "nccl", "rccl", "symm_mem")
COMPUTE = ("fused", "fusion", "gemm", "matmul", "norm")
DISTRIBUTED = ("c10d::", "collective", "distributed::", "processgroup")
TORCH_DTYPES = {
    "bfloat16": "c10::BFloat16",
    "float16": "c10::Half",
    "half": "c10::Half",
    "float32": "float",
    "float": "float",
    "float64": "double",
    "double": "double",
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short int",
    "short": "short int",
    "int32": "int",
    "int": "int",
    "int64": "long int",
    "long": "long int",
    "uint16": "short unsigned int",
    "uint32": "unsigned int",
    "uint64": "long unsigned int",
    "bool": "bool",
    "complex32": "c10::complex<c10::Half>",
    "complex64": "c10::complex<float>",
    "complex128": "c10::complex<double>",
    "float8_e4m3fn": "c10::Float8_e4m3fn",
    "float8_e4m3fnuz": "c10::Float8_e4m3fnuz",
    "float8_e5m2": "c10::Float8_e5m2",
    "float8_e5m2fnuz": "c10::Float8_e5m2fnuz",
}


def operator_kind(operator: str, kernel: str, original_kind: str) -> str:
    if original_kind == "eager-refer":
        return original_kind
    if (
        operator.startswith("flagcx::")
        and original_kind != "communication"
        and not any(token in kernel.lower() for token in COLLECTIVES + BACKENDS)
    ):
        # A communicator marker can enclose allocation/copy as well as NCCL.
        # Preserve the parser's non-communication evidence for those kernels.
        return "custom"
    if re.match(r"^moe_align_block_size_stage_?\d+(?:_[A-Za-z0-9]+)*$", kernel):
        return "custom"
    combined = f"{operator} {kernel}".lower()
    collective = any(token in combined for token in COLLECTIVES)
    if collective and any(token in combined for token in COMPUTE):
        return "fused_communication_compute"
    if (
        collective
        or any(token in combined for token in BACKENDS)
        or ("broadcast" in combined and any(token in combined for token in DISTRIBUTED))
    ):
        return "communication"
    if operator.startswith("aten::"):
        return "aten"
    if operator.startswith("triton_") or kernel.startswith("triton_"):
        return "triton_compiled"
    if operator == "null":
        return "unattributed"
    return "custom" if "::" in operator else "runtime_operator"


def export_operator(row: dict) -> dict:
    result = {field: row[field] for field in OPERATOR_FIELDS}
    result["operator_kind"] = operator_kind(
        row["operator_name"], row["kernel_name"], row["operator_kind"]
    )
    return result


def _dtypes(value):
    if isinstance(value, list):
        return [_dtypes(item) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("torch."):
        # Unknown types stay unavailable in the public schema; audit is lossless.
        return TORCH_DTYPES.get(value[6:], "")
    if value.startswith("Scalar["):
        return "Scalar"
    if value == "None" or value.startswith("Object["):
        return ""
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _candidates(value: str, *, eager: bool) -> str:
    candidates = json.loads(value)
    if candidates is None:
        return "null"
    converted = []
    for item in candidates:
        row = {
            "operator_name" if eager else "operator": item.get(
                "operator_name", item.get("operator")
            ),
            "input_shapes": item.get("input_shapes"),
            "input_dtypes": _dtypes(item.get("input_dtypes")),
        }
        if not eager and "metadata_status" in item:
            row["metadata_status"] = item["metadata_status"]
        converted.append(row)
    return _json(converted)


def export_detail(row: dict) -> dict:
    result = {field: row.get(field, "null") for field in DETAIL_FIELDS}
    result["mapping_status"] = row["mapping_status"].removeprefix("native_marker_")
    for field in ("input_dtypes", "eager_input_dtypes"):
        result[field] = _json(_dtypes(json.loads(result[field])))
    result["candidate_operators"] = _candidates(
        result["candidate_operators"], eager=False
    )
    result["eager_candidates"] = _candidates(result["eager_candidates"], eager=True)
    return result
