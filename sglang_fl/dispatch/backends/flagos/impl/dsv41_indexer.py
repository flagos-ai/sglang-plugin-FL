# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bridge SGLang indexer calls to public FlagGems-sglang operators."""


def available():
    try:
        import flaggems_sglang as gems
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "flaggems_sglang"
            or exc.name.startswith("flag_gems")
            or exc.name.startswith("triton")
        ):
            return False
        raise
    get_op = getattr(gems, "get_op", None)
    if not callable(get_op):
        return False
    return all(
        callable(get_op(name))
        for name in (
            "fp4_index_logits_req_to_token",
            "fp4_index_logits_candidate_blocks",
            "dsv41_long_context_producer",
            "dsv41_long_context_compact",
        )
    )


def _invoke(original, long_name, common_name, *args, **kwargs):
    import flaggems_sglang as gems
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "decode":
        result = gems.get_op(long_name)(*args, **kwargs)
        if result is not None:
            return result
    # Long-context operators return None only before launching any work.
    return gems.get_op(common_name)(*args, **kwargs)


def producer(original, *args, **kwargs):
    return _invoke(
        original,
        "dsv41_long_context_producer",
        "fp4_index_logits_req_to_token",
        *args,
        **kwargs,
    )


def compact(original, *args, **kwargs):
    return _invoke(
        original,
        "dsv41_long_context_compact",
        "fp4_index_logits_candidate_blocks",
        *args,
        **kwargs,
    )
