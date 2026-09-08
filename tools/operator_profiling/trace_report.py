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

"""Build operator inventories from SGLang runtime ``torch.profiler`` traces.

The report is deliberately kernel-backed: every row in the primary reports is
derived from a physical GPU kernel event.  CPU-only framework bookkeeping,
collectives, memcpy and memset activity are kept in the audit summary but do
not enter the compute-operator inventory.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import concurrent.futures
import csv
import gzip
import hashlib
import heapq
import json
import multiprocessing
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple, TextIO

_TRACE_SUFFIXES = (".trace.json", ".trace.json.gz")
_COMPUTE_CATEGORY = "kernel"
_MEMORY_CATEGORIES = {"gpu_memcpy", "gpu_memset"}
_CPU_OWNER_CATEGORIES = {"cpu_op", "user_annotation"}
_STEP_RE = re.compile(r"^step\[(?P<mode>[A-Z0-9_]+)\b")
_OPERATOR_MARKER_PREFIX = "sglang_fl.operator_profile.kernel:"
_RANK_PATTERNS = (
    re.compile(r"(?:^|[-_.])TP[-_](\d+)(?:[-_.]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])rank[-_]?(\d+)(?:[-_.]|$)", re.IGNORECASE),
)
_COMMUNICATION_RE = re.compile(
    r"(?:nccl|rccl|flagcx|hccl|msccl|gloo|c10d|"
    r"record_param_comms|custom_all_reduce|custom_ar|"
    r"two_shot_all_reduce|cross_device_reduce|allreduce|all_reduce|"
    r"allgather|all_gather|reduce_scatter)",
    re.IGNORECASE,
)
_FUSED_COMMUNICATION_COMPUTE_RE = re.compile(
    r"(?:fused[_:. -]*)?(?:all[_ -]?reduce|reduce[_ -]?scatter).*(?:norm|gemm|matmul|fusion)"
    r"|(?:norm|gemm|matmul).*(?:all[_ -]?reduce|reduce[_ -]?scatter)"
    r"|allreduce_fusion",
    re.IGNORECASE,
)
_MOE_ALIGN_STAGE_RE = re.compile(
    r"^(?:_)?moe_align_block_size_stage(?:1|2(?:_vec)?|3|4)$"
)
_ADDRESS_RE = re.compile(r"(?:0x)?[0-9a-f]{12,}", re.IGNORECASE)
_CLONE_SUFFIX_RE = re.compile(r"\s*\[clone [^]]+]$")


class SourceClassification(NamedTuple):
    category: str
    library: str
    rule: str


class OperatorMetadata(NamedTuple):
    operator_name: str
    input_shapes: str | None
    input_dtypes: str | None
    metadata_status: str
    source_category: str | None = None
    source_library: str | None = None
    classification_rule: str | None = None
    dispatch_operator_name: str | None = None
    dispatch_impl_id: str | None = None
    dispatch_impl_kind: str | None = None
    dispatch_vendor: str | None = None
    dispatch_runtime_source_category: str | None = None
    dispatch_runtime_source_library: str | None = None


class MappingKey(NamedTuple):
    mapping_status: str
    operator_name: str | None
    input_shapes: str
    input_dtypes: str
    candidate_operators: str | None
    operator_kind: str
    execution_origin: str
    source_category: str
    source_library: str
    classification_rule: str
    dispatch_operator_name: str
    dispatch_impl_id: str
    dispatch_impl_kind: str
    dispatch_vendor: str
    dispatch_runtime_source_category: str
    dispatch_runtime_source_library: str


class DispatchCallKey(NamedTuple):
    operator_name: str
    impl_id: str
    impl_kind: str
    vendor: str
    runtime_source_category: str
    runtime_source_library: str


_EAGER_ORIGIN = "eager"
_TORCH_COMPILE_ORIGIN = "torch_compile"
_SOURCE_CATEGORIES = (
    "third_party",
    "vendor",
    "torch_aten",
    "torch_fused",
)
_THIRD_PARTY_LIBRARY_PATTERNS = (
    ("flashinfer", ("flashinfer",)),
    ("flashattention", ("flash_attn", "flashattention", "flash_attention")),
    ("deep_gemm", ("deep_gemm", "deepgemm")),
    ("sgl_kernel", ("sgl_kernel", "sglang::")),
    ("sglang", ("sglang.",)),
)


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON suitable for a CSV cell or key."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_json(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_trace_events(
    path: Path, chunk_size: int = 1024 * 1024
) -> Iterator[dict[str, Any]]:
    """Stream the ``traceEvents`` array without loading a full trace into RAM.

    PyTorch normally writes one pretty-printed event per group of lines, but
    that layout is not part of the Chrome trace contract.  Incremental JSON
    decoding keeps this reader valid for both compact and pretty traces.
    """

    marker = '"traceEvents"'
    decoder = json.JSONDecoder()
    with _open_text(path) as source:
        buffer = ""
        while marker not in buffer:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError(f"traceEvents not found in {path}")
            buffer += chunk
            if len(buffer) > 64 * 1024 * 1024:
                raise ValueError(f"trace header is unexpectedly large in {path}")

        buffer = buffer.split(marker, 1)[1]
        while "[" not in buffer:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError(f"traceEvents array not found in {path}")
            buffer += chunk
        buffer = buffer.split("[", 1)[1]
        cursor = 0
        eof = False

        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor] in " \t\r\n,":
                    cursor += 1
                if cursor < len(buffer):
                    break
                if eof:
                    raise ValueError(f"unterminated traceEvents array in {path}")
                buffer = ""
                cursor = 0
                chunk = source.read(chunk_size)
                eof = not chunk
                buffer += chunk

            if buffer[cursor] == "]":
                return

            try:
                event, end = decoder.raw_decode(buffer, cursor)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"invalid trace event JSON in {path}") from None
                buffer = buffer[cursor:]
                cursor = 0
                chunk = source.read(chunk_size)
                eof = not chunk
                buffer += chunk
                continue

            cursor = end
            if isinstance(event, dict):
                yield event
            if cursor >= chunk_size:
                buffer = buffer[cursor:]
                cursor = 0


def discover_trace_files(path: Path, profile_id: str | None = None) -> list[Path]:
    """Return runtime trace files under ``path`` in deterministic order."""

    candidates = [path] if path.is_file() else sorted(path.rglob("*.trace.json*"))
    files = [
        item
        for item in candidates
        if item.is_file()
        and item.name.endswith(_TRACE_SUFFIXES)
        and "graph_capture" not in item.name.lower()
        and (profile_id is None or profile_id in item.name)
    ]
    return sorted(dict.fromkeys(files))


def rank_from_filename(path: Path) -> int:
    for pattern in _RANK_PATTERNS:
        match = pattern.search(path.name)
        if match:
            return int(match.group(1))
    return -1


def stable_kernel_name(raw_name: str) -> str:
    """Normalize volatile launch decoration while retaining implementation identity."""

    name = " ".join(str(raw_name).strip().split())
    name = _CLONE_SUFFIX_RE.sub("", name)
    name = _ADDRESS_RE.sub("<address>", name)
    name = name.replace("(anonymous namespace)", "<anonymous>")
    # Torch sometimes sanitizes a Python object repr into the kernel symbol.
    for separator in ("_object_at__", "_object_at_0x", " object at 0x"):
        if separator in name:
            name = name.split(separator, 1)[0]
    if name.startswith("void "):
        name = name[5:]
    # Demangled CUDA C++ symbols include launch arguments and template
    # specializations.  The owning function is the stable implementation unit.
    if "::" in name:
        depth = 0
        leaf_start = 0
        argument_start: int | None = None
        for index, character in enumerate(name):
            if character == "<":
                depth += 1
            elif character == ">" and depth:
                depth -= 1
            elif character == "(" and depth == 0:
                argument_start = index
                break
            elif (
                character == ":"
                and depth == 0
                and index + 1 < len(name)
                and name[index + 1] == ":"
            ):
                leaf_start = index + 2
        if argument_start is not None:
            name = name[:argument_start]
        template_start = name.find("<", leaf_start)
        if template_start >= 0:
            name = name[:template_start]
    return name.strip() or str(raw_name).strip()


def _external_id(event: dict[str, Any]) -> int | str | None:
    value = event.get("args", {}).get("External id")
    return value if isinstance(value, (int, str)) else None


def _correlation_id(event: dict[str, Any]) -> int | str | None:
    value = event.get("args", {}).get("correlation")
    return value if isinstance(value, (int, str)) else None


def _event_ns(event: dict[str, Any], field_name: str) -> int:
    return round(float(event.get(field_name, 0.0) or 0.0) * 1000)


def _duration_ns(event: dict[str, Any]) -> int:
    return _event_ns(event, "dur")


def _event_metadata(event: dict[str, Any]) -> OperatorMetadata:
    args = event.get("args", {})
    has_shapes = "Input Dims" in args
    has_dtypes = "Input type" in args
    shapes = canonical_json(args["Input Dims"]) if has_shapes else None
    dtypes = canonical_json(args["Input type"]) if has_dtypes else None
    status = (
        "shape_and_dtype"
        if has_shapes and has_dtypes
        else "shape_only"
        if has_shapes
        else "dtype_only"
        if has_dtypes
        else "no_input_metadata"
    )
    return OperatorMetadata(str(event.get("name", "")), shapes, dtypes, status)


def _is_torch_compile_kernel_owner(event: dict[str, Any]) -> bool:
    """Identify profiler events emitted by TorchInductor kernel wrappers.

    Torch profiler emits these fields from the generated wrapper itself.  The
    check intentionally avoids kernel-name patterns because user-authored
    Triton/FlagTree kernels may use similar names without involving compile.
    """

    if event.get("cat") != "cpu_op":
        return False
    args = event.get("args", {})
    kernel_file = args.get("kernel_file")
    return (
        isinstance(args.get("kernel_hash"), str)
        and bool(args["kernel_hash"])
        and isinstance(args.get("kernel_backend"), str)
        and isinstance(kernel_file, str)
        and "torchinductor" in kernel_file.lower()
    )


def _is_step_span(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and _STEP_RE.match(str(event.get("name", ""))) is not None
    )


def _operator_marker_metadata(event: dict[str, Any]) -> OperatorMetadata | None:
    if event.get("ph") != "X" or event.get("cat") != "user_annotation":
        return None
    name = str(event.get("name", ""))
    if not name.startswith(_OPERATOR_MARKER_PREFIX):
        return None
    try:
        encoded = name[len(_OPERATOR_MARKER_PREFIX) :]
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    operator_name = payload.get("operator_name")
    shapes = payload.get("input_shapes")
    dtypes = payload.get("input_dtypes")
    if not isinstance(operator_name, str) or not operator_name:
        return None
    source_category = payload.get("source_category")
    source_library = payload.get("source_library")
    classification_rule = payload.get("classification_rule")
    dispatch_operator_name = payload.get("dispatch_operator_name")
    dispatch_impl_id = payload.get("dispatch_impl_id")
    dispatch_impl_kind = payload.get("dispatch_impl_kind")
    dispatch_vendor = payload.get("dispatch_vendor")
    dispatch_runtime_source_category = payload.get("dispatch_runtime_source_category")
    dispatch_runtime_source_library = payload.get("dispatch_runtime_source_library")
    if source_category is not None and source_category not in _SOURCE_CATEGORIES:
        return None
    if any(
        value is not None and (not isinstance(value, str) or not value)
        for value in (source_library, classification_rule)
    ):
        return None
    if source_category == "vendor" and (
        source_library is None or classification_rule != "plugin_dispatch_op_impl_kind"
    ):
        return None
    dispatch_values = (
        dispatch_operator_name,
        dispatch_impl_id,
        dispatch_impl_kind,
    )
    if any(value is not None for value in dispatch_values):
        if not all(isinstance(value, str) and value for value in dispatch_values):
            return None
        if dispatch_impl_kind not in {"flagos", "reference", "vendor"}:
            return None
        if dispatch_impl_kind == "vendor" and not (
            isinstance(dispatch_vendor, str) and dispatch_vendor
        ):
            return None
    elif dispatch_vendor is not None:
        return None
    if dispatch_runtime_source_category is not None:
        if dispatch_runtime_source_category not in {"vendor", "third_party"}:
            return None
        if not (
            isinstance(dispatch_runtime_source_library, str)
            and dispatch_runtime_source_library
        ):
            return None
    elif dispatch_runtime_source_library is not None:
        return None
    return OperatorMetadata(
        operator_name=operator_name,
        input_shapes=canonical_json(shapes),
        input_dtypes=canonical_json(dtypes),
        metadata_status="shape_and_dtype",
        source_category=source_category,
        source_library=source_library,
        classification_rule=classification_rule,
        dispatch_operator_name=dispatch_operator_name,
        dispatch_impl_id=dispatch_impl_id,
        dispatch_impl_kind=dispatch_impl_kind,
        dispatch_vendor=dispatch_vendor,
        dispatch_runtime_source_category=dispatch_runtime_source_category,
        dispatch_runtime_source_library=dispatch_runtime_source_library,
    )


def _step_phase(name: str) -> str:
    match = _STEP_RE.match(name)
    if not match:
        return "other"
    mode = match.group("mode")
    if mode in {"EXTEND", "MIXED", "SPLIT_PREFILL", "DLLM_EXTEND"}:
        return "prefill"
    if mode in {"DECODE", "TARGET_VERIFY", "DRAFT_EXTEND_V2"}:
        return "decode"
    return mode.lower()


@dataclass(frozen=True)
class _StepSpan:
    start_ns: int
    end_ns: int
    phase: str


@dataclass
class _StepIndex:
    spans: dict[tuple[str, str], list[_StepSpan]] = field(
        default_factory=lambda: defaultdict(list)
    )
    starts: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    @classmethod
    def from_trace(cls, path: Path) -> "_StepIndex":
        result = cls()
        for event in iter_trace_events(path):
            result.add_event(event)
        result.finalize()
        return result

    def add_event(self, event: dict[str, Any]) -> None:
        if not _is_step_span(event):
            return
        start = _event_ns(event, "ts")
        key = (str(event.get("pid", "")), str(event.get("tid", "")))
        self.spans[key].append(
            _StepSpan(
                start_ns=start,
                end_ns=start + _duration_ns(event),
                phase=_step_phase(str(event.get("name", ""))),
            )
        )

    def finalize(self) -> None:
        for key, spans in self.spans.items():
            spans.sort(key=lambda span: (span.start_ns, span.end_ns))
            self.starts[key] = [span.start_ns for span in spans]

    def phase_for(self, event: dict[str, Any]) -> str:
        key = (str(event.get("pid", "")), str(event.get("tid", "")))
        return self.phase_at(key, _event_ns(event, "ts"))

    def phase_at(self, key: tuple[str, str], timestamp: int) -> str:
        spans = self.spans.get(key)
        if not spans:
            return "other"
        index = bisect.bisect_right(self.starts[key], timestamp) - 1
        while index >= 0:
            span = spans[index]
            if span.start_ns <= timestamp <= span.end_ns:
                return span.phase
            if span.end_ns < timestamp:
                break
            index -= 1
        return "other"


@dataclass(frozen=True)
class _OperatorMarkerSpan:
    start_ns: int
    end_ns: int
    metadata: OperatorMetadata


@dataclass
class _OperatorMarkerIndex:
    spans: dict[tuple[str, str], list[_OperatorMarkerSpan]] = field(
        default_factory=lambda: defaultdict(list)
    )
    starts: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    @classmethod
    def from_trace(cls, path: Path) -> "_OperatorMarkerIndex":
        result = cls()
        for event in iter_trace_events(path):
            result.add_event(event)
        result.finalize()
        return result

    def add_event(self, event: dict[str, Any]) -> None:
        metadata = _operator_marker_metadata(event)
        if metadata is None:
            return
        start = _event_ns(event, "ts")
        key = (str(event.get("pid", "")), str(event.get("tid", "")))
        self.spans[key].append(
            _OperatorMarkerSpan(
                start_ns=start,
                end_ns=start + _duration_ns(event),
                metadata=metadata,
            )
        )

    def finalize(self) -> None:
        for key, spans in self.spans.items():
            spans.sort(key=lambda span: (span.start_ns, -span.end_ns))
            self.starts[key] = [span.start_ns for span in spans]

    def metadata_for(self, event: dict[str, Any]) -> OperatorMetadata | None:
        key = (str(event.get("pid", "")), str(event.get("tid", "")))
        return self.metadata_at(key, _event_ns(event, "ts"))

    def metadata_at(
        self, key: tuple[str, str], timestamp: int
    ) -> OperatorMetadata | None:
        spans = self.spans.get(key)
        if not spans:
            return None
        index = bisect.bisect_right(self.starts[key], timestamp) - 1
        # The latest enclosing marker is the innermost marker.  This makes a
        # direct Triton/FlagTree marker win over an enclosing MultiPlatformOp.
        while index >= 0:
            span = spans[index]
            if timestamp <= span.end_ns:
                return span.metadata
            index -= 1
        return None

    def dispatch_metadata_for(self, event: dict[str, Any]) -> OperatorMetadata | None:
        """Return the innermost enclosing plugin-dispatch marker, if any."""

        key = (str(event.get("pid", "")), str(event.get("tid", "")))
        return self.dispatch_metadata_at(key, _event_ns(event, "ts"))

    def dispatch_metadata_at(
        self, key: tuple[str, str], timestamp: int
    ) -> OperatorMetadata | None:
        """Return the innermost plugin-dispatch marker at a trace location."""

        spans = self.spans.get(key)
        if not spans:
            return None
        index = bisect.bisect_right(self.starts[key], timestamp) - 1
        while index >= 0:
            span = spans[index]
            if timestamp <= span.end_ns and span.metadata.dispatch_impl_id:
                return span.metadata
            index -= 1
        return None

    def resolve_launches(
        self,
        launches: Sequence[tuple[int | str, tuple[str, str], int]],
    ) -> Iterator[
        tuple[
            int | str,
            tuple[str, str],
            int,
            OperatorMetadata | None,
            OperatorMetadata | None,
        ]
    ]:
        """Resolve enclosing operator and dispatch markers for many launches.

        A point lookup may need to walk through many expired inner spans before
        finding a still-active outer span.  Repeating that lookup for every
        CUDA launch becomes quadratic on long decode traces.  This sweep keeps
        active spans in latest-start heaps and resolves every thread in
        timestamp order.
        """

        launches_by_thread: dict[tuple[str, str], list[tuple[int | str, int]]] = (
            defaultdict(list)
        )
        for correlation_id, thread_key, timestamp in launches:
            launches_by_thread[thread_key].append((correlation_id, timestamp))

        for thread_key, thread_launches in launches_by_thread.items():
            spans = self.spans.get(thread_key, [])
            span_index = 0
            active: list[tuple[int, int, int, OperatorMetadata]] = []
            active_dispatch: list[tuple[int, int, int, OperatorMetadata]] = []
            for correlation_id, timestamp in sorted(
                thread_launches, key=lambda item: item[1]
            ):
                while (
                    span_index < len(spans) and spans[span_index].start_ns <= timestamp
                ):
                    span = spans[span_index]
                    entry = (
                        -span.start_ns,
                        span_index,
                        span.end_ns,
                        span.metadata,
                    )
                    heapq.heappush(active, entry)
                    if span.metadata.dispatch_impl_id:
                        heapq.heappush(active_dispatch, entry)
                    span_index += 1
                while active and active[0][2] < timestamp:
                    heapq.heappop(active)
                while active_dispatch and active_dispatch[0][2] < timestamp:
                    heapq.heappop(active_dispatch)
                yield (
                    correlation_id,
                    thread_key,
                    timestamp,
                    active[0][3] if active else None,
                    active_dispatch[0][3] if active_dispatch else None,
                )


def _mapping(
    event_external_id: int | str | None,
    cpu_by_external_id: dict[int | str, set[OperatorMetadata]],
) -> dict[str, Any]:
    if event_external_id is None:
        return {
            "mapping_status": "missing_external_id",
            "operator_name": None,
            "input_shapes": None,
            "input_dtypes": None,
            "candidate_operators": None,
        }
    candidates = sorted(
        cpu_by_external_id.get(event_external_id, set()),
        key=lambda item: tuple(value or "" for value in item),
    )
    if not candidates:
        return {
            "mapping_status": "no_cpu_op_match",
            "operator_name": None,
            "input_shapes": None,
            "input_dtypes": None,
            "candidate_operators": None,
        }
    if len(candidates) > 1:
        names = {item.operator_name for item in candidates}
        candidate_rows = [
            {
                "operator_name": item.operator_name,
                "input_shapes": _decode_json(item.input_shapes),
                "input_dtypes": _decode_json(item.input_dtypes),
                "metadata_status": item.metadata_status,
                "source_category": item.source_category,
                "source_library": item.source_library,
                "classification_rule": item.classification_rule,
                "dispatch_operator_name": item.dispatch_operator_name,
                "dispatch_impl_id": item.dispatch_impl_id,
                "dispatch_impl_kind": item.dispatch_impl_kind,
                "dispatch_vendor": item.dispatch_vendor,
                "dispatch_runtime_source_category": (
                    item.dispatch_runtime_source_category
                ),
                "dispatch_runtime_source_library": item.dispatch_runtime_source_library,
            }
            for item in candidates
        ]
        return {
            "mapping_status": (
                "shape_ambiguous" if len(names) == 1 else "operator_ambiguous"
            ),
            "operator_name": next(iter(names)) if len(names) == 1 else None,
            "input_shapes": None,
            "input_dtypes": None,
            "candidate_operators": candidate_rows,
        }

    metadata = candidates[0]
    statuses = {
        "shape_and_dtype": "operator_shape_matched",
        "shape_only": "operator_matched_dtype_missing",
        "dtype_only": "operator_matched_shape_missing",
        "no_input_metadata": "operator_matched_metadata_missing",
    }
    return {
        "mapping_status": statuses[metadata.metadata_status],
        "operator_name": metadata.operator_name,
        "input_shapes": _decode_json(metadata.input_shapes),
        "input_dtypes": _decode_json(metadata.input_dtypes),
        "candidate_operators": None,
        "source_category": metadata.source_category,
        "source_library": metadata.source_library,
        "classification_rule": metadata.classification_rule,
        "dispatch_operator_name": metadata.dispatch_operator_name,
        "dispatch_impl_id": metadata.dispatch_impl_id,
        "dispatch_impl_kind": metadata.dispatch_impl_kind,
        "dispatch_vendor": metadata.dispatch_vendor,
        "dispatch_runtime_source_category": (metadata.dispatch_runtime_source_category),
        "dispatch_runtime_source_library": metadata.dispatch_runtime_source_library,
    }


def _marker_mapping(
    correlation_id: int | str | None,
    marker_by_correlation_id: dict[int | str, set[OperatorMetadata]],
) -> dict[str, Any] | None:
    if correlation_id is None or correlation_id not in marker_by_correlation_id:
        return None
    result = _mapping(correlation_id, marker_by_correlation_id)
    result["mapping_status"] = f"native_marker_{result['mapping_status']}"
    return result


def _operator_kind(source: SourceClassification) -> str:
    """Return the aggregation grain implied by the source category."""

    return "aten" if source.category == "torch_aten" else "fused"


def _classify_source(
    *,
    operator_name: str | None,
    kernel_name: str,
    raw_kernel_name: str,
    execution_origin: str,
    explicit_category: str | None = None,
    explicit_library: str | None = None,
    explicit_rule: str | None = None,
    dispatch_impl_kind: str | None = None,
    dispatch_vendor: str | None = None,
    dispatch_runtime_source_category: str | None = None,
    dispatch_runtime_source_library: str | None = None,
) -> SourceClassification:
    """Classify implementation provenance using runtime evidence.

    ``vendor`` is intentionally not inferred from CUDA/CUTLASS/cuBLAS symbols.
    It is reserved for an sglang-plugin-FL dispatch event whose selected
    ``OpImpl.kind`` is ``VENDOR``.
    """

    # The inventory describes the implementation path selected by the
    # framework.  Once a physical kernel is observed inside an executed
    # plugin VENDOR OpImpl, the vendor dispatch owns that event regardless of
    # whether the adapter delegates to SGLang, an ATen op, or a chip library.
    # The delegate remains available through dispatch_runtime_source_* and is
    # reported separately; it must not override the dispatch-path category.
    if dispatch_impl_kind == "vendor" and dispatch_vendor:
        return SourceClassification("vendor", dispatch_vendor, "plugin_vendor_dispatch")

    if explicit_category is not None:
        if explicit_category not in _SOURCE_CATEGORIES:
            raise ValueError(f"unknown explicit source category: {explicit_category}")
        if explicit_category == "vendor" and (
            explicit_library is None or explicit_rule != "plugin_dispatch_op_impl_kind"
        ):
            raise ValueError(
                "vendor classification requires plugin dispatch OpImpl evidence"
            )
        return SourceClassification(
            explicit_category,
            explicit_library or "unspecified",
            explicit_rule or "explicit_runtime_marker",
        )

    if execution_origin == _TORCH_COMPILE_ORIGIN:
        return SourceClassification(
            "torch_fused",
            "torchinductor",
            "torch_profiler_inductor_metadata",
        )
    if operator_name and operator_name.startswith("aten::"):
        return SourceClassification(
            "torch_aten", "pytorch", "torch_dispatcher_operator_name"
        )

    evidence = " ".join(
        item for item in (operator_name, kernel_name, raw_kernel_name) if item
    ).lower()
    for library, patterns in _THIRD_PARTY_LIBRARY_PATTERNS:
        if any(pattern in evidence for pattern in patterns):
            return SourceClassification(
                "third_party", library, "upstream_runtime_namespace"
            )
    if kernel_name.startswith(("at::", "c10::")):
        library = "pytorch"
    elif "cutlass" in evidence:
        library = "cutlass"
    elif dispatch_runtime_source_category == "third_party":
        return SourceClassification(
            "third_party",
            dispatch_runtime_source_library or "upstream_unknown",
            "dispatch_implementation_provenance",
        )
    elif dispatch_runtime_source_category == "vendor":
        if dispatch_impl_kind != "vendor" or not dispatch_vendor:
            raise ValueError(
                "vendor runtime provenance requires an executed vendor OpImpl"
            )
        return SourceClassification(
            "vendor",
            dispatch_runtime_source_library or dispatch_vendor,
            "dispatch_implementation_provenance",
        )
    else:
        library = "upstream_unknown"
    return SourceClassification("third_party", library, "runtime_kernel_evidence")


def _is_communication(kernel_name: str, operator_name: str | None) -> bool:
    return bool(
        _COMMUNICATION_RE.search(kernel_name)
        or (operator_name is not None and _COMMUNICATION_RE.search(operator_name))
    )


def _is_fused_communication_compute(
    kernel_name: str, operator_name: str | None
) -> bool:
    evidence = " ".join(
        value for value in (kernel_name, operator_name) if value is not None
    )
    return bool(_FUSED_COMMUNICATION_COMPUTE_RE.search(evidence))


def _communication_library(kernel_name: str, operator_name: str | None) -> str:
    evidence = " ".join(
        value for value in (kernel_name, operator_name) if value is not None
    ).lower()
    for library in ("flagcx", "nccl", "rccl", "hccl", "msccl", "gloo", "c10d"):
        if library in evidence:
            return library
    if any(token in evidence for token in ("custom_ar", "custom_all_reduce")):
        return "sglang_custom_all_reduce"
    return "communication_runtime"


def _mapping_key(
    link: dict[str, Any],
    execution_origin: str,
    source: SourceClassification,
    *,
    operator_kind: str | None = None,
) -> MappingKey:
    candidates = link.get("candidate_operators")
    return MappingKey(
        mapping_status=str(link["mapping_status"]),
        operator_name=link.get("operator_name"),
        input_shapes=canonical_json(link.get("input_shapes")),
        input_dtypes=canonical_json(link.get("input_dtypes")),
        candidate_operators=(
            canonical_json(candidates) if candidates is not None else None
        ),
        operator_kind=operator_kind or _operator_kind(source),
        execution_origin=execution_origin,
        source_category=source.category,
        source_library=source.library,
        classification_rule=source.rule,
        dispatch_operator_name=str(link.get("dispatch_operator_name") or ""),
        dispatch_impl_id=str(link.get("dispatch_impl_id") or ""),
        dispatch_impl_kind=str(link.get("dispatch_impl_kind") or ""),
        dispatch_vendor=str(link.get("dispatch_vendor") or ""),
        dispatch_runtime_source_category=str(
            link.get("dispatch_runtime_source_category") or ""
        ),
        dispatch_runtime_source_library=str(
            link.get("dispatch_runtime_source_library") or ""
        ),
    )


@dataclass
class TraceAggregate:
    """Lossless aggregate of every included physical compute-kernel event."""

    trace_files: list[str] = field(default_factory=list)
    ranks: set[int] = field(default_factory=set)
    cpu_event_count: int = 0
    cpu_operator_names: set[str] = field(default_factory=set)
    cpu_metadata_count: Counter[str] = field(default_factory=Counter)
    gpu_category_count: Counter[str] = field(default_factory=Counter)
    gpu_category_ns: Counter[str] = field(default_factory=Counter)
    excluded_count: Counter[str] = field(default_factory=Counter)
    excluded_ns: Counter[str] = field(default_factory=Counter)
    kernel_count: Counter[str] = field(default_factory=Counter)
    kernel_ns: Counter[str] = field(default_factory=Counter)
    kernel_variants_count: dict[str, Counter[MappingKey]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    kernel_variants_ns: dict[str, Counter[MappingKey]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    raw_kernel_count: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    raw_kernel_ns: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # Keep all runtime kernels for the audit summary.  Public CSVs use the
    # compute collections below and therefore exclude pure communication.
    runtime_kernel_count: Counter[str] = field(default_factory=Counter)
    runtime_kernel_ns: Counter[str] = field(default_factory=Counter)
    runtime_kernel_variants_count: dict[str, Counter[MappingKey]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    runtime_kernel_variants_ns: dict[str, Counter[MappingKey]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    phase_count: Counter[tuple[str, str]] = field(default_factory=Counter)
    phase_ns: Counter[tuple[str, str]] = field(default_factory=Counter)
    execution_origin_count: Counter[str] = field(default_factory=Counter)
    execution_origin_ns: Counter[str] = field(default_factory=Counter)
    source_category_count: Counter[str] = field(default_factory=Counter)
    source_category_ns: Counter[str] = field(default_factory=Counter)
    source_library_count: Counter[str] = field(default_factory=Counter)
    source_library_ns: Counter[str] = field(default_factory=Counter)
    classification_rule_count: Counter[str] = field(default_factory=Counter)
    classification_rule_ns: Counter[str] = field(default_factory=Counter)
    source_category_operator_ids: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    dispatch_call_count: Counter[DispatchCallKey] = field(default_factory=Counter)
    dispatch_kernel_count: Counter[DispatchCallKey] = field(default_factory=Counter)
    dispatch_kernel_ns: Counter[DispatchCallKey] = field(default_factory=Counter)

    def _add_runtime_kernel(
        self,
        kernel_name: str,
        key: MappingKey,
        duration: int,
    ) -> None:
        self.runtime_kernel_count[kernel_name] += 1
        self.runtime_kernel_ns[kernel_name] += duration
        self.runtime_kernel_variants_count[kernel_name][key] += 1
        self.runtime_kernel_variants_ns[kernel_name][key] += duration

    def merge(self, other: "TraceAggregate") -> None:
        """Merge an independently parsed trace aggregate without re-reading it."""

        self.trace_files.extend(other.trace_files)
        self.ranks.update(other.ranks)
        self.cpu_event_count += other.cpu_event_count
        self.cpu_operator_names.update(other.cpu_operator_names)
        for attribute in (
            "cpu_metadata_count",
            "gpu_category_count",
            "gpu_category_ns",
            "excluded_count",
            "excluded_ns",
            "kernel_count",
            "kernel_ns",
            "runtime_kernel_count",
            "runtime_kernel_ns",
            "phase_count",
            "phase_ns",
            "execution_origin_count",
            "execution_origin_ns",
            "source_category_count",
            "source_category_ns",
            "source_library_count",
            "source_library_ns",
            "classification_rule_count",
            "classification_rule_ns",
            "dispatch_call_count",
            "dispatch_kernel_count",
            "dispatch_kernel_ns",
        ):
            getattr(self, attribute).update(getattr(other, attribute))
        for attribute in (
            "kernel_variants_count",
            "kernel_variants_ns",
            "raw_kernel_count",
            "raw_kernel_ns",
            "runtime_kernel_variants_count",
            "runtime_kernel_variants_ns",
        ):
            target = getattr(self, attribute)
            for key, values in getattr(other, attribute).items():
                target[key].update(values)
        for category, operator_ids in other.source_category_operator_ids.items():
            self.source_category_operator_ids[category].update(operator_ids)

    def add_trace(self, path: Path) -> None:
        rank = rank_from_filename(path)
        self.trace_files.append(str(path))
        self.ranks.add(rank)
        steps = _StepIndex()
        operator_markers = _OperatorMarkerIndex()
        cpu_by_external_id: dict[int | str, set[OperatorMetadata]] = defaultdict(set)
        marker_by_correlation_id: dict[int | str, set[OperatorMetadata]] = defaultdict(
            set
        )
        dispatch_by_correlation_id: dict[int | str, set[OperatorMetadata]] = (
            defaultdict(set)
        )
        torch_compile_external_ids: set[int | str] = set()
        concrete_cpu_ids: set[int | str] = set()
        phase_by_external_id: dict[int | str, str] = {}
        phase_by_correlation_id: dict[int | str, str] = {}
        external_id_locations: dict[int | str, tuple[tuple[str, str], int]] = {}
        launch_locations: list[tuple[int | str, tuple[str, str], int]] = []
        gpu_events: list[tuple[str, int, str, int | str | None, int | str | None]] = []

        for event in iter_trace_events(path):
            steps.add_event(event)
            operator_markers.add_event(event)
            category = str(event.get("cat", ""))
            if category in {_COMPUTE_CATEGORY, *_MEMORY_CATEGORIES}:
                gpu_events.append(
                    (
                        category,
                        _duration_ns(event),
                        str(event.get("name", "")),
                        _external_id(event),
                        _correlation_id(event),
                    )
                )
                continue
            if category in {"cuda_runtime", "cuda_driver"}:
                correlation_id = _correlation_id(event)
                if correlation_id is not None:
                    launch_locations.append(
                        (
                            correlation_id,
                            (
                                str(event.get("pid", "")),
                                str(event.get("tid", "")),
                            ),
                            _event_ns(event, "ts"),
                        )
                    )
                continue
            if event.get("cat") not in _CPU_OWNER_CATEGORIES:
                continue
            marker_metadata = _operator_marker_metadata(event)
            if marker_metadata is not None and marker_metadata.dispatch_impl_id:
                dispatch_key = DispatchCallKey(
                    marker_metadata.dispatch_operator_name or "",
                    marker_metadata.dispatch_impl_id,
                    marker_metadata.dispatch_impl_kind or "",
                    marker_metadata.dispatch_vendor or "",
                    marker_metadata.dispatch_runtime_source_category or "",
                    marker_metadata.dispatch_runtime_source_library or "",
                )
                self.dispatch_call_count[dispatch_key] += 1
            if _is_step_span(event) or marker_metadata is not None:
                continue
            metadata = _event_metadata(event)
            if event.get("cat") == "cpu_op":
                self.cpu_event_count += 1
                self.cpu_operator_names.add(metadata[0])
                self.cpu_metadata_count[metadata[3]] += 1
            event_id = _external_id(event)
            if event_id is None:
                continue
            if _is_torch_compile_kernel_owner(event):
                torch_compile_external_ids.add(event_id)
            # Prefer concrete cpu_op metadata over record_function annotations.
            if event.get("cat") == "cpu_op":
                if event_id not in concrete_cpu_ids:
                    cpu_by_external_id[event_id].clear()
                    concrete_cpu_ids.add(event_id)
                cpu_by_external_id[event_id].add(metadata)
                external_id_locations[event_id] = (
                    (str(event.get("pid", "")), str(event.get("tid", ""))),
                    _event_ns(event, "ts"),
                )
            elif event_id not in concrete_cpu_ids and "Input Dims" in event.get(
                "args", {}
            ):
                cpu_by_external_id[event_id].add(metadata)
                external_id_locations[event_id] = (
                    (str(event.get("pid", "")), str(event.get("tid", ""))),
                    _event_ns(event, "ts"),
                )

        steps.finalize()
        operator_markers.finalize()
        for event_id, (thread_key, timestamp) in external_id_locations.items():
            phase_by_external_id[event_id] = steps.phase_at(thread_key, timestamp)
        for (
            correlation_id,
            thread_key,
            timestamp,
            marker,
            dispatch_marker,
        ) in operator_markers.resolve_launches(launch_locations):
            if marker is not None:
                marker_by_correlation_id[correlation_id].add(marker)
                phase_by_correlation_id[correlation_id] = steps.phase_at(
                    thread_key, timestamp
                )
            if dispatch_marker is not None:
                dispatch_by_correlation_id[correlation_id].add(dispatch_marker)

        for category, duration, raw_name, event_id, correlation_id in gpu_events:
            self.gpu_category_count[category] += 1
            self.gpu_category_ns[category] += duration
            if category in _MEMORY_CATEGORIES:
                self.excluded_count[category] += 1
                self.excluded_ns[category] += duration
                continue

            kernel_name = stable_kernel_name(raw_name)
            link = _mapping(event_id, cpu_by_external_id)
            marker_link = _marker_mapping(correlation_id, marker_by_correlation_id)
            dispatch_link = _marker_mapping(correlation_id, dispatch_by_correlation_id)
            if marker_link is not None and marker_link.get("source_category"):
                # An explicit runtime provenance marker is stronger evidence than
                # a generic CPU launch owner, especially for plugin vendor ops.
                link = marker_link
            elif link["mapping_status"] in {
                "missing_external_id",
                "no_cpu_op_match",
                "operator_matched_shape_missing",
                "operator_matched_metadata_missing",
            }:
                if marker_link is not None:
                    link = marker_link
            if dispatch_link is not None:
                for field_name in (
                    "dispatch_operator_name",
                    "dispatch_impl_id",
                    "dispatch_impl_kind",
                    "dispatch_vendor",
                    "dispatch_runtime_source_category",
                    "dispatch_runtime_source_library",
                ):
                    link[field_name] = dispatch_link.get(field_name)
            operator_name = link.get("operator_name")
            is_communication = _is_communication(raw_name, operator_name)
            is_fused_communication_compute = (
                is_communication
                and _is_fused_communication_compute(raw_name, operator_name)
            )
            execution_origin = (
                _TORCH_COMPILE_ORIGIN
                if event_id in torch_compile_external_ids
                else _EAGER_ORIGIN
            )
            if is_communication and not is_fused_communication_compute:
                communication_source = SourceClassification(
                    "communication",
                    _communication_library(raw_name, operator_name),
                    "runtime_communication_kernel",
                )
                runtime_key = _mapping_key(
                    link,
                    execution_origin,
                    communication_source,
                    operator_kind="communication",
                )
                self._add_runtime_kernel(kernel_name, runtime_key, duration)
                self.excluded_count["communication"] += 1
                self.excluded_ns["communication"] += duration
                continue

            source = _classify_source(
                operator_name=operator_name,
                kernel_name=kernel_name,
                raw_kernel_name=raw_name,
                execution_origin=execution_origin,
                explicit_category=link.get("source_category"),
                explicit_library=link.get("source_library"),
                explicit_rule=link.get("classification_rule"),
                dispatch_impl_kind=link.get("dispatch_impl_kind"),
                dispatch_vendor=link.get("dispatch_vendor"),
                dispatch_runtime_source_category=link.get(
                    "dispatch_runtime_source_category"
                ),
                dispatch_runtime_source_library=link.get(
                    "dispatch_runtime_source_library"
                ),
            )
            key = _mapping_key(
                link,
                execution_origin,
                source,
                operator_kind=(
                    "fused_communication_compute"
                    if is_fused_communication_compute
                    else None
                ),
            )
            self._add_runtime_kernel(kernel_name, key, duration)
            self.kernel_count[kernel_name] += 1
            self.kernel_ns[kernel_name] += duration
            self.kernel_variants_count[kernel_name][key] += 1
            self.kernel_variants_ns[kernel_name][key] += duration
            self.raw_kernel_count[kernel_name][raw_name] += 1
            self.raw_kernel_ns[kernel_name][raw_name] += duration
            phase = phase_by_external_id.get(
                event_id, phase_by_correlation_id.get(correlation_id, "other")
            )
            self.phase_count[(phase, key.operator_kind)] += 1
            self.phase_ns[(phase, key.operator_kind)] += duration
            self.execution_origin_count[execution_origin] += 1
            self.execution_origin_ns[execution_origin] += duration
            self.source_category_count[source.category] += 1
            self.source_category_ns[source.category] += duration
            self.source_library_count[source.library] += 1
            self.source_library_ns[source.library] += duration
            self.classification_rule_count[source.rule] += 1
            self.classification_rule_ns[source.rule] += duration
            operator_id = (
                operator_name
                if source.category == "torch_aten" and operator_name is not None
                else kernel_name
            )
            self.source_category_operator_ids[source.category].add(operator_id)
            if key.dispatch_impl_id:
                dispatch_key = DispatchCallKey(
                    key.dispatch_operator_name,
                    key.dispatch_impl_id,
                    key.dispatch_impl_kind,
                    key.dispatch_vendor,
                    key.dispatch_runtime_source_category,
                    key.dispatch_runtime_source_library,
                )
                self.dispatch_kernel_count[dispatch_key] += 1
                self.dispatch_kernel_ns[dispatch_key] += duration

    @property
    def compute_kernel_count(self) -> int:
        return sum(self.kernel_count.values())

    @property
    def compute_kernel_ns(self) -> int:
        return sum(self.kernel_ns.values())

    @property
    def runtime_kernel_event_count(self) -> int:
        return sum(self.runtime_kernel_count.values())

    @property
    def runtime_kernel_time_ns(self) -> int:
        return sum(self.runtime_kernel_ns.values())


def collect_traces(files: Sequence[Path]) -> TraceAggregate:
    if not files:
        raise ValueError("at least one profiler trace is required")
    aggregate = TraceAggregate()
    for path in files:
        aggregate.add_trace(path)
    return aggregate


def _format_percent(value_ns: int, total_ns: int) -> str:
    percent = value_ns / total_ns * 100 if total_ns else 0.0
    if 0 < percent < 0.001:
        return "<0.001%"
    return f"{percent:.3f}%"


def _ns_to_us(value: int) -> float:
    return value / 1000


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _ordered_kernel_names(aggregate: TraceAggregate) -> list[str]:
    return sorted(
        aggregate.kernel_count,
        key=lambda name: (-aggregate.kernel_ns[name], name),
    )


def _ordered_runtime_kernel_names(aggregate: TraceAggregate) -> list[str]:
    return sorted(
        aggregate.runtime_kernel_count,
        key=lambda name: (-aggregate.runtime_kernel_ns[name], name),
    )


def _report_operator_kind(key: MappingKey, kernel_name: str) -> str:
    """Return the reference-compatible runtime operator taxonomy."""

    if key.operator_kind == "communication":
        return "communication"
    if key.operator_kind == "fused_communication_compute":
        return "fused_communication_compute"
    if key.execution_origin == _TORCH_COMPILE_ORIGIN:
        return "torch_compile"
    if key.operator_kind == "aten":
        return "aten"
    if key.operator_name is None:
        return (
            "unattributed_nvjet" if "nvjet" in kernel_name.lower() else "unattributed"
        )
    return "custom"


def _canonical_report_operator_name(key: MappingKey, kernel_name: str) -> str:
    if _MOE_ALIGN_STAGE_RE.match(kernel_name):
        return "moe_align_block_size"
    return key.operator_name or "null"


def _report_operator_identity(
    operator_name: str,
    operator_kind: str,
    kernel_name: str,
) -> tuple[str, str] | None:
    if operator_kind == "communication":
        return None
    if operator_kind == "aten":
        return ("aten", operator_name)
    if _MOE_ALIGN_STAGE_RE.match(kernel_name):
        return ("family", "moe_align_block_size")
    # Custom, compile-generated and unattributed implementations are counted
    # at stable physical implementation granularity.
    return ("kernel", kernel_name)


def kernel_shape_dtype_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    """Build the compute-only shape/dtype report used by operator developers."""

    rows: list[dict[str, Any]] = []
    for kernel_name in _ordered_kernel_names(aggregate):
        keys = sorted(
            aggregate.kernel_variants_count[kernel_name],
            key=lambda key: (
                -aggregate.kernel_variants_ns[kernel_name][key],
                key.mapping_status,
                _canonical_report_operator_name(key, kernel_name),
                key.input_shapes,
                key.input_dtypes,
                key.candidate_operators or "",
            ),
        )
        for variant_index, key in enumerate(keys, start=1):
            rows.append(
                {
                    "operator_name": _canonical_report_operator_name(key, kernel_name),
                    "kernel_name": kernel_name,
                    "source_category": key.source_category,
                    "variant_index": variant_index,
                    "mapping_status": key.mapping_status,
                    "input_shapes": key.input_shapes,
                    "input_dtypes": key.input_dtypes,
                    "candidate_operators": key.candidate_operators or "null",
                    "kernel_event_count": aggregate.kernel_variants_count[kernel_name][
                        key
                    ],
                    "kernel_time_us": _ns_to_us(
                        aggregate.kernel_variants_ns[kernel_name][key]
                    ),
                }
            )
    return rows


def kernel_time_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    """Aggregate compute time by owner, stable kernel and source category."""

    counts: Counter[tuple[str, str, str]] = Counter()
    durations: Counter[tuple[str, str, str]] = Counter()
    for kernel_name, variants in aggregate.kernel_variants_count.items():
        for key, count in variants.items():
            report_key = (
                _canonical_report_operator_name(key, kernel_name),
                kernel_name,
                key.source_category,
            )
            counts[report_key] += count
            durations[report_key] += aggregate.kernel_variants_ns[kernel_name][key]
    total_ns = aggregate.compute_kernel_ns
    return [
        {
            "operator_name": operator_name,
            "kernel_name": kernel_name,
            "source_category": source_category,
            "kernel_call_count": counts[(operator_name, kernel_name, source_category)],
            "kernel_time_us": _ns_to_us(
                durations[(operator_name, kernel_name, source_category)]
            ),
            "percent": _format_percent(
                durations[(operator_name, kernel_name, source_category)], total_ns
            ),
            "percent_of_category": _format_percent(
                durations[(operator_name, kernel_name, source_category)],
                aggregate.source_category_ns[source_category],
            ),
        }
        for operator_name, kernel_name, source_category in sorted(
            durations,
            key=lambda item: (-durations[item], item[2], item[0], item[1]),
        )
    ]


def reference_operator_list_rows(
    aggregate: TraceAggregate,
) -> list[dict[str, Any]]:
    """Build the enriched, compute-only operator/kernel inventory.

    The first four columns remain compatible with the reference report.  The
    appended fields make implementation provenance and plugin dispatch auditable
    without repeating them for every shape variant.
    """

    relations: set[tuple[Any, ...]] = set()
    for kernel_name, variants in aggregate.kernel_variants_count.items():
        for key in variants:
            operator_name = _canonical_report_operator_name(key, kernel_name)
            operator_kind = _report_operator_kind(key, kernel_name)
            identity = _report_operator_identity(
                operator_name, operator_kind, kernel_name
            )
            relations.add(
                (
                    identity,
                    operator_name,
                    operator_kind,
                    kernel_name,
                    key.source_category,
                    key.source_library,
                    key.classification_rule,
                    key.execution_origin,
                    key.dispatch_operator_name,
                    key.dispatch_impl_id,
                    key.dispatch_impl_kind,
                    key.dispatch_vendor,
                    key.dispatch_runtime_source_category,
                    key.dispatch_runtime_source_library,
                )
            )

    kind_order = {
        "custom": 0,
        "fused_communication_compute": 1,
        "aten": 2,
        "runtime_operator": 3,
        "torch_compile": 4,
        "triton_compiled": 5,
        "unattributed": 6,
        "unattributed_nvjet": 7,
    }
    identities: dict[tuple[str, str], tuple[int, str, str]] = {}
    for identity, _operator_name, operator_kind, _kernel_name, *_rest in relations:
        if identity is None:
            continue
        candidate = (kind_order.get(operator_kind, 99), identity[0], identity[1])
        previous = identities.get(identity)
        if previous is None or candidate < previous:
            identities[identity] = candidate
    ordered_identities = sorted(identities, key=lambda item: identities[item])
    operator_ids = {
        identity: str(index)
        for index, identity in enumerate(ordered_identities, start=1)
    }

    def relation_order(
        relation: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        identity, operator_name, operator_kind, kernel_name, *provenance = relation
        if identity is None:
            operator_id = len(operator_ids) + 1
        else:
            operator_id = int(operator_ids[identity])
        return (operator_id, operator_kind, operator_name, kernel_name, *provenance)

    return [
        {
            "operator_id": "null" if identity is None else operator_ids[identity],
            "operator_name": operator_name,
            "operator_kind": operator_kind,
            "kernel_name": kernel_name,
            "source_category": source_category,
            "source_library": source_library,
            "classification_rule": classification_rule,
            "execution_origin": execution_origin,
            "dispatch_operator_name": dispatch_operator_name or "null",
            "dispatch_impl_id": dispatch_impl_id or "null",
            "dispatch_impl_kind": dispatch_impl_kind or "null",
            "dispatch_vendor": dispatch_vendor or "null",
            "runtime_delegate_category": (dispatch_runtime_source_category or "null"),
            "runtime_delegate_library": dispatch_runtime_source_library or "null",
        }
        for (
            identity,
            operator_name,
            operator_kind,
            kernel_name,
            source_category,
            source_library,
            classification_rule,
            execution_origin,
            dispatch_operator_name,
            dispatch_impl_id,
            dispatch_impl_kind,
            dispatch_vendor,
            dispatch_runtime_source_category,
            dispatch_runtime_source_library,
        ) in sorted(relations, key=relation_order)
    ]


def _kernel_execution_origin(aggregate: TraceAggregate, kernel_name: str) -> str:
    origins = {
        key.execution_origin for key in aggregate.kernel_variants_count[kernel_name]
    }
    return next(iter(origins)) if len(origins) == 1 else "mixed"


def _kernel_classification_value(
    aggregate: TraceAggregate, kernel_name: str, field_name: str
) -> str:
    values = {
        str(getattr(key, field_name))
        for key in aggregate.kernel_variants_count[kernel_name]
    }
    return next(iter(values)) if len(values) == 1 else "mixed"


def kernel_summary_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    total_ns = aggregate.compute_kernel_ns
    return [
        {
            "kernel_name": name,
            "execution_origin": _kernel_execution_origin(aggregate, name),
            "source_category": _kernel_classification_value(
                aggregate, name, "source_category"
            ),
            "source_library": _kernel_classification_value(
                aggregate, name, "source_library"
            ),
            "total_call_count": aggregate.kernel_count[name],
            "total_time_us": _ns_to_us(aggregate.kernel_ns[name]),
            "percent": _format_percent(aggregate.kernel_ns[name], total_ns),
        }
        for name in _ordered_kernel_names(aggregate)
    ]


def details_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kernel_name in _ordered_kernel_names(aggregate):
        keys = sorted(
            aggregate.kernel_variants_count[kernel_name],
            key=lambda key: (
                -aggregate.kernel_variants_ns[kernel_name][key],
                key.mapping_status,
                key.operator_name or "",
                key.input_shapes,
                key.input_dtypes,
                key.candidate_operators or "",
                key.operator_kind,
                key.execution_origin,
                key.source_category,
                key.source_library,
                key.dispatch_operator_name,
                key.dispatch_impl_id,
            ),
        )
        for variant_index, key in enumerate(keys, start=1):
            rows.append(
                {
                    "operator_name": key.operator_name or "null",
                    "kernel_name": kernel_name,
                    "operator_kind": key.operator_kind,
                    "source_category": key.source_category,
                    "source_library": key.source_library,
                    "classification_rule": key.classification_rule,
                    "execution_origin": key.execution_origin,
                    "dispatch_operator_name": key.dispatch_operator_name or "null",
                    "dispatch_impl_id": key.dispatch_impl_id or "null",
                    "dispatch_impl_kind": key.dispatch_impl_kind or "null",
                    "dispatch_vendor": key.dispatch_vendor or "null",
                    "dispatch_runtime_source_category": (
                        key.dispatch_runtime_source_category or "null"
                    ),
                    "dispatch_runtime_source_library": (
                        key.dispatch_runtime_source_library or "null"
                    ),
                    "variant_index": variant_index,
                    "mapping_status": key.mapping_status,
                    "input_shapes": key.input_shapes,
                    "input_dtypes": key.input_dtypes,
                    "candidate_operators": key.candidate_operators or "null",
                    "kernel_event_count": aggregate.kernel_variants_count[kernel_name][
                        key
                    ],
                    "kernel_time_us": _ns_to_us(
                        aggregate.kernel_variants_ns[kernel_name][key]
                    ),
                }
            )
    return rows


def operator_provenance_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    rows: set[tuple[str, ...]] = set()
    for kernel_name, variants in aggregate.kernel_variants_count.items():
        for key in variants:
            operator_name = key.operator_name or "null"
            operator_id = operator_name if key.operator_kind == "aten" else kernel_name
            rows.add(
                (
                    operator_id,
                    operator_name,
                    key.operator_kind,
                    kernel_name,
                    key.source_category,
                    key.source_library,
                    key.classification_rule,
                    key.execution_origin,
                    key.dispatch_operator_name,
                    key.dispatch_impl_id,
                    key.dispatch_impl_kind,
                    key.dispatch_vendor,
                    key.dispatch_runtime_source_category,
                    key.dispatch_runtime_source_library,
                )
            )
    ordered = sorted(
        rows,
        key=lambda row: (
            _SOURCE_CATEGORIES.index(row[4]),
            row[0],
            row[1],
            row[3],
            row[5],
            row[7],
        ),
    )
    return [
        {
            "operator_id": operator_id,
            "operator_name": operator_name,
            "operator_kind": kind,
            "kernel_name": kernel_name,
            "source_category": category,
            "source_library": library,
            "classification_rule": rule,
            "execution_origin": origin,
            "dispatch_operator_name": dispatch_operator_name or "null",
            "dispatch_impl_id": dispatch_impl_id or "null",
            "dispatch_impl_kind": dispatch_impl_kind or "null",
            "dispatch_vendor": dispatch_vendor or "null",
            "dispatch_runtime_source_category": (
                dispatch_runtime_source_category or "null"
            ),
            "dispatch_runtime_source_library": (
                dispatch_runtime_source_library or "null"
            ),
        }
        for (
            operator_id,
            operator_name,
            kind,
            kernel_name,
            category,
            library,
            rule,
            origin,
            dispatch_operator_name,
            dispatch_impl_id,
            dispatch_impl_kind,
            dispatch_vendor,
            dispatch_runtime_source_category,
            dispatch_runtime_source_library,
        ) in ordered
    ]


def operator_summary_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    times: Counter[tuple[str, ...]] = Counter()
    for kernel_name, variants in aggregate.kernel_variants_count.items():
        for key, count in variants.items():
            if key.operator_kind == "aten":
                operator_name = key.operator_name or "null"
                operator_id = operator_name
            else:
                # Fused operators use stable physical implementation names as
                # their aggregation grain.
                operator_name = kernel_name
                operator_id = kernel_name
            summary_key = (
                operator_id,
                operator_name,
                key.operator_kind,
                key.source_category,
                key.source_library,
                key.execution_origin,
                key.dispatch_operator_name,
                key.dispatch_impl_id,
                key.dispatch_impl_kind,
                key.dispatch_vendor,
                key.dispatch_runtime_source_category,
                key.dispatch_runtime_source_library,
            )
            counts[summary_key] += count
            times[summary_key] += aggregate.kernel_variants_ns[kernel_name][key]
    total_ns = aggregate.compute_kernel_ns
    ordered = sorted(times, key=lambda key: (-times[key], key))
    return [
        {
            "operator_id": key[0],
            "operator_name": key[1],
            "operator_kind": key[2],
            "source_category": key[3],
            "source_library": key[4],
            "execution_origin": key[5],
            "dispatch_operator_name": key[6] or "null",
            "dispatch_impl_id": key[7] or "null",
            "dispatch_impl_kind": key[8] or "null",
            "dispatch_vendor": key[9] or "null",
            "dispatch_runtime_source_category": key[10] or "null",
            "dispatch_runtime_source_library": key[11] or "null",
            "kernel_event_count": counts[key],
            "kernel_time_us": _ns_to_us(times[key]),
            "percent": _format_percent(times[key], total_ns),
        }
        for key in ordered
    ]


def torch_compile_operator_rows(
    aggregate: TraceAggregate,
) -> list[dict[str, Any]]:
    """Summarize TorchInductor output at stable implementation granularity."""

    rows: list[dict[str, Any]] = []
    for kernel_name in _ordered_kernel_names(aggregate):
        variants = [
            key
            for key in aggregate.kernel_variants_count[kernel_name]
            if key.execution_origin == _TORCH_COMPILE_ORIGIN
        ]
        if not variants:
            continue
        count = sum(
            aggregate.kernel_variants_count[kernel_name][key] for key in variants
        )
        duration = sum(
            aggregate.kernel_variants_ns[kernel_name][key] for key in variants
        )
        operator_names = sorted(
            {
                key.operator_name if key.operator_name is not None else "null"
                for key in variants
            }
        )
        rows.append(
            {
                "kernel_name": kernel_name,
                "operator_names": canonical_json(operator_names),
                "input_variant_count": len(variants),
                "kernel_event_count": count,
                "kernel_time_us": _ns_to_us(duration),
                "percent_of_all_compute": _format_percent(
                    duration, aggregate.compute_kernel_ns
                ),
            }
        )
    return rows


def dispatch_summary_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    """Audit logical Layer-2 calls separately from their physical kernels."""

    keys = set(aggregate.dispatch_call_count) | set(aggregate.dispatch_kernel_count)
    return [
        {
            "dispatch_operator_name": key.operator_name,
            "dispatch_impl_id": key.impl_id,
            "dispatch_impl_kind": key.impl_kind,
            "dispatch_vendor": key.vendor or "null",
            "dispatch_runtime_source_category": (key.runtime_source_category or "null"),
            "dispatch_runtime_source_library": key.runtime_source_library or "null",
            "logical_call_count": aggregate.dispatch_call_count[key],
            "physical_kernel_event_count": aggregate.dispatch_kernel_count[key],
            "physical_kernel_time_us": _ns_to_us(aggregate.dispatch_kernel_ns[key]),
            "percent_of_all_compute": _format_percent(
                aggregate.dispatch_kernel_ns[key], aggregate.compute_kernel_ns
            ),
        }
        for key in sorted(keys)
    ]


def kernel_name_variant_rows(aggregate: TraceAggregate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stable_name in _ordered_kernel_names(aggregate):
        raw_names = sorted(
            aggregate.raw_kernel_count[stable_name],
            key=lambda name: (-aggregate.raw_kernel_ns[stable_name][name], name),
        )
        for raw_name in raw_names:
            rows.append(
                {
                    "kernel_name": stable_name,
                    "raw_kernel_name": raw_name,
                    "kernel_event_count": aggregate.raw_kernel_count[stable_name][
                        raw_name
                    ],
                    "kernel_time_us": _ns_to_us(
                        aggregate.raw_kernel_ns[stable_name][raw_name]
                    ),
                }
            )
    return rows


def _csv_us_to_ns(value: str) -> int:
    return int(Decimal(value) * 1000)


def _validate_outputs(
    aggregate: TraceAggregate,
    shape_dtype_path: Path,
    kernel_time_path: Path,
    operator_list_path: Path,
) -> dict[str, bool]:
    with shape_dtype_path.open(encoding="utf-8", newline="") as source:
        shape_dtype = list(csv.DictReader(source))
    with kernel_time_path.open(encoding="utf-8", newline="") as source:
        kernel_time = list(csv.DictReader(source))
    with operator_list_path.open(encoding="utf-8", newline="") as source:
        operator_list = list(csv.DictReader(source))
    reference_shape_count: Counter[tuple[str, str, str]] = Counter()
    reference_shape_ns: Counter[tuple[str, str, str]] = Counter()
    for row in shape_dtype:
        key = (
            row["operator_name"],
            row["kernel_name"],
            row["source_category"],
        )
        reference_shape_count[key] += int(row["kernel_event_count"])
        reference_shape_ns[key] += _csv_us_to_ns(row["kernel_time_us"])
    reference_time_count: Counter[tuple[str, str, str]] = Counter()
    reference_time_ns: Counter[tuple[str, str, str]] = Counter()
    source_count: Counter[str] = Counter()
    source_ns: Counter[str] = Counter()
    for row in kernel_time:
        key = (
            row["operator_name"],
            row["kernel_name"],
            row["source_category"],
        )
        count = int(row["kernel_call_count"])
        duration = _csv_us_to_ns(row["kernel_time_us"])
        reference_time_count[key] += count
        reference_time_ns[key] += duration
        source_count[row["source_category"]] += count
        source_ns[row["source_category"]] += duration
    reference_relations = {
        (row["operator_name"], row["kernel_name"], row["source_category"])
        for row in operator_list
    }
    numbered_ids = sorted(
        {
            int(row["operator_id"])
            for row in operator_list
            if row["operator_id"] != "null"
        }
    )
    expected_ids = list(range(1, len(numbered_ids) + 1))
    return {
        "shape_time_event_count_matches": (
            dict(reference_shape_count) == reference_time_count
            and sum(reference_shape_count.values()) == aggregate.compute_kernel_count
        ),
        "shape_time_duration_matches": (
            dict(reference_shape_ns) == reference_time_ns
            and sum(reference_shape_ns.values()) == aggregate.compute_kernel_ns
        ),
        "operator_relations_cover_kernel_time": (
            set(reference_time_count).issubset(reference_relations)
        ),
        "operator_ids_are_contiguous": numbered_ids == expected_ids,
        "excluded_activity_absent_from_csv": all(
            row["operator_kind"] != "communication" for row in operator_list
        ),
        "operator_source_categories_valid": all(
            row["source_category"] in _SOURCE_CATEGORIES for row in operator_list
        ),
        "source_category_event_count_matches": all(
            source_count[category] == aggregate.source_category_count[category]
            for category in _SOURCE_CATEGORIES
        ),
        "source_category_time_matches": all(
            source_ns[category] == aggregate.source_category_ns[category]
            for category in _SOURCE_CATEGORIES
        ),
        "physical_kernel_events_counted_once": (
            sum(source_count.values()) == aggregate.compute_kernel_count
            and sum(source_ns.values()) == aggregate.compute_kernel_ns
        ),
    }


def write_report(
    aggregate: TraceAggregate,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the three public CSVs plus one audited JSON summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    shape_dtype_path = output_dir / "kernel_shape_dtype.csv"
    kernel_time_path = output_dir / "kernel_time.csv"
    operator_list_path = output_dir / "operator_list.csv"

    _atomic_write_csv(
        shape_dtype_path,
        [
            "operator_name",
            "kernel_name",
            "source_category",
            "variant_index",
            "mapping_status",
            "input_shapes",
            "input_dtypes",
            "candidate_operators",
            "kernel_event_count",
            "kernel_time_us",
        ],
        kernel_shape_dtype_rows(aggregate),
    )
    _atomic_write_csv(
        kernel_time_path,
        [
            "operator_name",
            "kernel_name",
            "source_category",
            "kernel_call_count",
            "kernel_time_us",
            "percent",
            "percent_of_category",
        ],
        kernel_time_rows(aggregate),
    )
    _atomic_write_csv(
        operator_list_path,
        [
            "operator_id",
            "operator_name",
            "operator_kind",
            "kernel_name",
            "source_category",
            "source_library",
            "classification_rule",
            "execution_origin",
            "dispatch_operator_name",
            "dispatch_impl_id",
            "dispatch_impl_kind",
            "dispatch_vendor",
            "runtime_delegate_category",
            "runtime_delegate_library",
        ],
        reference_operator_list_rows(aggregate),
    )

    mapping_count: Counter[str] = Counter()
    mapping_ns: Counter[str] = Counter()
    for kernel_name, variants in aggregate.kernel_variants_count.items():
        for key, count in variants.items():
            mapping_count[key.mapping_status] += count
            mapping_ns[key.mapping_status] += aggregate.kernel_variants_ns[kernel_name][
                key
            ]

    torch_compile_rows = torch_compile_operator_rows(aggregate)
    torch_compile_count = aggregate.execution_origin_count[_TORCH_COMPILE_ORIGIN]
    torch_compile_ns = aggregate.execution_origin_ns[_TORCH_COMPILE_ORIGIN]

    validation = _validate_outputs(
        aggregate,
        shape_dtype_path,
        kernel_time_path,
        operator_list_path,
    )
    if not all(validation.values()):
        raise RuntimeError(f"operator report conservation failed: {validation}")
    summary = {
        "schema_version": 6,
        "scope": {
            "execution_mode": "eager",
            "rank_aggregation": "operator_and_shape_union; event_count_and_time_sum",
            "global_torch_compile_enabled": bool(
                (metadata or {}).get("server", {}).get("enable_torch_compile", False)
            ),
            "warmup_included": False,
            "included_operator_kinds": ["aten", "fused"],
            "execution_origins": [_EAGER_ORIGIN, _TORCH_COMPILE_ORIGIN],
            "source_categories": list(_SOURCE_CATEGORIES),
            "vendor_classification": (
                "an included kernel inside an executed plugin VENDOR OpImpl scope is "
                "vendor; the adapter's delegate is reported separately"
            ),
            "excluded_gpu_activity": ["communication", "gpu_memcpy", "gpu_memset"],
            "time_denominator": "sum_of_included_compute_kernel_durations",
            "inventory_source_policy": "four_source_categories",
        },
        "trace_files": [Path(path).name for path in aggregate.trace_files],
        "ranks": sorted(aggregate.ranks),
        "cpu_operator_event_count": aggregate.cpu_event_count,
        "unique_cpu_operator_names": len(aggregate.cpu_operator_names),
        "cpu_metadata_count": dict(sorted(aggregate.cpu_metadata_count.items())),
        "gpu_event_count_by_category": dict(
            sorted(aggregate.gpu_category_count.items())
        ),
        "gpu_time_us_by_category": {
            key: _ns_to_us(value)
            for key, value in sorted(aggregate.gpu_category_ns.items())
        },
        "excluded_event_count_by_reason": dict(
            sorted(aggregate.excluded_count.items())
        ),
        "excluded_time_us_by_reason": {
            key: _ns_to_us(value)
            for key, value in sorted(aggregate.excluded_ns.items())
        },
        "runtime_kernel_event_count": aggregate.runtime_kernel_event_count,
        "runtime_kernel_time_us": _ns_to_us(aggregate.runtime_kernel_time_ns),
        "compute_kernel_event_count": aggregate.compute_kernel_count,
        "compute_kernel_time_us": _ns_to_us(aggregate.compute_kernel_ns),
        "unique_stable_kernel_names": len(aggregate.kernel_count),
        "execution_origin_event_count": dict(
            sorted(aggregate.execution_origin_count.items())
        ),
        "execution_origin_time_us": {
            key: _ns_to_us(value)
            for key, value in sorted(aggregate.execution_origin_ns.items())
        },
        "source_category_unique_operator_count": {
            category: len(aggregate.source_category_operator_ids[category])
            for category in _SOURCE_CATEGORIES
        },
        "source_category_event_count": {
            category: aggregate.source_category_count[category]
            for category in _SOURCE_CATEGORIES
        },
        "source_category_time_us": {
            category: _ns_to_us(aggregate.source_category_ns[category])
            for category in _SOURCE_CATEGORIES
        },
        "source_category_percent_of_all_compute": {
            category: _format_percent(
                aggregate.source_category_ns[category], aggregate.compute_kernel_ns
            )
            for category in _SOURCE_CATEGORIES
        },
        "source_library_event_count": dict(
            sorted(aggregate.source_library_count.items())
        ),
        "source_library_time_us": {
            library: _ns_to_us(value)
            for library, value in sorted(aggregate.source_library_ns.items())
        },
        "classification_rule_event_count": dict(
            sorted(aggregate.classification_rule_count.items())
        ),
        "classification_rule_time_us": {
            rule: _ns_to_us(value)
            for rule, value in sorted(aggregate.classification_rule_ns.items())
        },
        "dispatch": {
            "logical_call_count": sum(aggregate.dispatch_call_count.values()),
            "physical_kernel_event_count": sum(
                aggregate.dispatch_kernel_count.values()
            ),
            "selected_impl_ids": sorted(
                {key.impl_id for key in aggregate.dispatch_call_count}
            ),
            "selected_impl_kinds": sorted(
                {key.impl_kind for key in aggregate.dispatch_call_count}
            ),
            "operators": dispatch_summary_rows(aggregate),
        },
        "torch_compile": {
            "detection": (
                "torch profiler cpu_op metadata: kernel_hash + kernel_backend + "
                "kernel_file under torchinductor"
            ),
            "unique_stable_kernel_names": len(torch_compile_rows),
            "kernel_event_count": torch_compile_count,
            "kernel_time_us": _ns_to_us(torch_compile_ns),
            "percent_of_all_compute": _format_percent(
                torch_compile_ns, aggregate.compute_kernel_ns
            ),
            "inventory_policy": "included_as_torch_fused",
        },
        "mapping_event_count_by_status": dict(sorted(mapping_count.items())),
        "mapping_time_us_by_status": {
            key: _ns_to_us(value) for key, value in sorted(mapping_ns.items())
        },
        "phase_event_count": {
            f"{phase}:{kind}": value
            for (phase, kind), value in sorted(aggregate.phase_count.items())
        },
        "phase_time_us": {
            f"{phase}:{kind}": _ns_to_us(value)
            for (phase, kind), value in sorted(aggregate.phase_ns.items())
        },
        "validation": validation,
        "metadata": metadata or {},
        "files": {},
    }
    for name, path in (
        ("operator_list", operator_list_path),
        ("kernel_shape_dtype", shape_dtype_path),
        ("kernel_time", kernel_time_path),
    ):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with path.open(encoding="utf-8") as source:
            row_count = max(sum(1 for _ in source) - 1, 0)
        summary["files"][name] = {
            "path": path.name,
            "row_count": row_count,
            "sha256": digest,
        }
    _atomic_write_json(output_dir / "profile_summary.json", summary)
    return summary


def generate_reports(
    trace_path: Path,
    output_dir: Path,
    profile_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    jobs: int = 1,
) -> dict[str, Any]:
    """Generate one compact all-ranks report and embed per-rank totals in JSON."""

    if jobs <= 0:
        raise ValueError("jobs must be a positive integer")

    files = discover_trace_files(trace_path, profile_id)
    if not files:
        raise FileNotFoundError(
            f"no runtime torch profiler traces found under {trace_path}"
        )
    by_rank: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_rank[rank_from_filename(path)].append(path)

    ordered_rank_files = sorted(by_rank.items())
    if jobs == 1 or len(ordered_rank_files) == 1:
        rank_aggregates = {
            rank: collect_traces(rank_files) for rank, rank_files in ordered_rank_files
        }
    else:
        # Use spawn rather than fork: generate_reports may be called by a
        # process that already initialized CUDA through SGLang.
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(jobs, len(ordered_rank_files)),
            mp_context=context,
        ) as executor:
            futures = {
                rank: executor.submit(collect_traces, rank_files)
                for rank, rank_files in ordered_rank_files
            }
            rank_aggregates = {
                rank: futures[rank].result() for rank, _ in ordered_rank_files
            }
    merged_aggregate = TraceAggregate()
    for aggregate in rank_aggregates.values():
        merged_aggregate.merge(aggregate)
    summary = write_report(merged_aggregate, output_dir, metadata)
    summary["per_rank"] = {
        str(rank): {
            "runtime_kernel_event_count": aggregate.runtime_kernel_event_count,
            "runtime_kernel_time_us": _ns_to_us(aggregate.runtime_kernel_time_ns),
            "compute_kernel_event_count": aggregate.compute_kernel_count,
            "compute_kernel_time_us": _ns_to_us(aggregate.compute_kernel_ns),
            "unique_stable_kernel_names": len(aggregate.kernel_count),
            "source_category_unique_operator_count": {
                category: len(aggregate.source_category_operator_ids[category])
                for category in _SOURCE_CATEGORIES
            },
            "source_category_event_count": {
                category: aggregate.source_category_count[category]
                for category in _SOURCE_CATEGORIES
            },
            "source_category_time_us": {
                category: _ns_to_us(aggregate.source_category_ns[category])
                for category in _SOURCE_CATEGORIES
            },
        }
        for rank, aggregate in sorted(rank_aggregates.items())
    }
    _atomic_write_json(output_dir / "profile_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SGLang eager operator reports from torch profiler traces"
    )
    parser.add_argument("--trace-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of TP-rank trace groups to parse concurrently",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional JSON object copied into profile_summary.json",
    )
    args = parser.parse_args()
    metadata = None
    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("--metadata-json must contain a JSON object")
    summary = generate_reports(
        args.trace_path,
        args.output_dir,
        profile_id=args.profile_id,
        metadata=metadata,
        jobs=args.jobs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
