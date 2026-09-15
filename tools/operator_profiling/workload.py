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

"""Model-independent helpers for exact-token profiling workloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LengthResolution:
    requested_input_tokens: int
    requested_output_tokens: int
    actual_input_tokens: int
    actual_output_tokens: int
    model_max_context: int
    reserved_tokens: int
    input_truncated: bool
    truncated_token_count: int
    length_overflow_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_runtime_reserved_tokens(
    model_max_context: int,
    *,
    framework_reserved_tokens: int = 0,
    max_req_input_len: int | None = None,
) -> int:
    """Return a conservative context margin exposed by the SGLang runtime.

    ``num_reserved_tokens`` covers model-specific reservations such as EAGLE.
    SGLang can impose a smaller scheduler request limit independently.  Treating
    the gap between the model context and ``max_req_input_len`` as another
    reservation keeps an exact-output workload inside both limits without
    depending on SGLang's private scheduler constants.
    """

    if model_max_context <= 0:
        raise ValueError(f"model_max_context must be positive, got {model_max_context}")
    if framework_reserved_tokens < 0:
        raise ValueError(
            "framework_reserved_tokens must be non-negative, got "
            f"{framework_reserved_tokens}"
        )

    scheduler_reserved_tokens = 0
    if max_req_input_len is not None:
        if not 0 < max_req_input_len <= model_max_context:
            raise ValueError(
                "max_req_input_len must be within model context, got "
                f"{max_req_input_len} for context {model_max_context}"
            )
        scheduler_reserved_tokens = model_max_context - max_req_input_len

    return max(framework_reserved_tokens, scheduler_reserved_tokens)


def resolve_lengths(
    requested_input_tokens: int,
    requested_output_tokens: int,
    model_max_context: int,
    *,
    reserved_tokens: int = 0,
    overflow_policy: str = "truncate_input",
) -> LengthResolution:
    """Resolve an exact workload while preserving requested output length."""

    values = {
        "requested_input_tokens": requested_input_tokens,
        "requested_output_tokens": requested_output_tokens,
        "model_max_context": model_max_context,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if reserved_tokens < 0:
        raise ValueError(f"reserved_tokens must be non-negative, got {reserved_tokens}")
    if overflow_policy not in {"truncate_input", "error"}:
        raise ValueError(
            "overflow_policy must be 'truncate_input' or 'error', "
            f"got {overflow_policy!r}"
        )

    input_budget = model_max_context - requested_output_tokens - reserved_tokens
    if input_budget <= 0:
        raise ValueError(
            f"output ({requested_output_tokens}) plus reserved tokens "
            f"({reserved_tokens}) leaves no input capacity in model context "
            f"({model_max_context})"
        )
    overflow = requested_input_tokens > input_budget
    if overflow and overflow_policy == "error":
        raise ValueError(
            f"requested {requested_input_tokens} input + "
            f"{requested_output_tokens} output + {reserved_tokens} reserved tokens, "
            f"but model context is {model_max_context}"
        )
    actual_input = min(requested_input_tokens, input_budget)
    return LengthResolution(
        requested_input_tokens=requested_input_tokens,
        requested_output_tokens=requested_output_tokens,
        actual_input_tokens=actual_input,
        actual_output_tokens=requested_output_tokens,
        model_max_context=model_max_context,
        reserved_tokens=reserved_tokens,
        input_truncated=overflow,
        truncated_token_count=requested_input_tokens - actual_input,
        length_overflow_policy=overflow_policy,
    )


def build_exact_token_ids(tokenizer: Any, target_tokens: int) -> list[int]:
    """Create deterministic valid token IDs without constructing a huge string."""

    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    seed_text = (
        "A distributed inference system should document its assumptions, "
        "measure every relevant compute operator, and validate its results. "
    )
    seed_ids = list(tokenizer.encode(seed_text, add_special_tokens=False))
    if not seed_ids:
        raise RuntimeError("tokenizer produced no IDs for the deterministic seed text")
    prefix: list[int] = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos_token_id, int):
        prefix.append(bos_token_id)
    body_length = target_tokens - len(prefix)
    if body_length < 0:
        return prefix[:target_tokens]
    repetitions, remainder = divmod(body_length, len(seed_ids))
    return prefix + seed_ids * repetitions + seed_ids[:remainder]


def completion_token_counts(result: Any) -> list[int]:
    """Extract SGLang completion-token counts from scalar or batched results."""

    rows = result if isinstance(result, list) else [result]
    counts: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"result {index} is not a mapping: {type(row).__name__}")
        value = row.get("meta_info", {}).get("completion_tokens")
        if not isinstance(value, int):
            raise ValueError(f"result {index} has no integer completion_tokens")
        counts.append(value)
    return counts


def validate_generation_result(
    result: Any, *, concurrency: int, output_tokens: int
) -> list[int]:
    counts = completion_token_counts(result)
    if len(counts) != concurrency:
        raise RuntimeError(f"expected {concurrency} responses, received {len(counts)}")
    failures = [
        f"request {index}: {count}"
        for index, count in enumerate(counts)
        if count != output_tokens
    ]
    if failures:
        raise RuntimeError(
            f"expected exactly {output_tokens} output tokens per request; "
            + ", ".join(failures[:16])
        )
    return counts


def generation_text_digest(result: Any) -> str:
    """Hash the ordered response texts emitted by one deterministic batch."""

    rows = result if isinstance(result, list) else [result]
    texts = [str(row.get("text", "")) for row in rows]
    encoded = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generation_token_prefixes(result: Any, prefix_tokens: int = 10) -> list[list[int]]:
    """Return ordered generated-token prefixes for run diagnostics."""

    if prefix_tokens <= 0:
        raise ValueError(f"prefix_tokens must be positive, got {prefix_tokens}")
    rows = result if isinstance(result, list) else [result]
    prefixes: list[list[int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"result {index} is not a mapping: {type(row).__name__}")
        output_ids = row.get("output_ids")
        if not isinstance(output_ids, (list, tuple)):
            raise ValueError(f"result {index} has no output_ids sequence")
        prefix = list(output_ids[:prefix_tokens])
        if not all(isinstance(token_id, int) for token_id in prefix):
            raise ValueError(f"result {index} contains a non-integer output token")
        prefixes.append(prefix)
    return prefixes
