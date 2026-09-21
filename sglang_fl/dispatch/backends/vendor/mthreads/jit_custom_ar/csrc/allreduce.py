from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace

import torch

from sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc.jit import (
    load_musa_jit,
    load_musa_pybind,
)

SHOT_PUSH = 0
SHOT_ONE_STAGE = 1
SHOT_TWO_STAGE = 2
SHOT_PUSH_WIDE = 3
SHOT_TWO_STAGE_512 = 4


@dataclass(frozen=True)
class CompileConfig:
    threads: int
    blocks: int
    vector_load: int
    atomic_barrier: int
    max_blocks: int
    dynamic_blocks: int
    push_polling: int
    push_16b_asm: int
    double_store_2shot: int
    one_shot_2rank_special: int
    push_skip_start_barrier: int
    push_16b_asm_forced: bool


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _env_auto_bool(names: tuple[str, ...], default: bool) -> tuple[bool, bool]:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = value.lower()
        if normalized == "auto":
            return True, False
        if normalized in ("1", "true", "yes", "on"):
            return True, True
        if normalized in ("0", "false", "no", "off"):
            return False, True
        raise ValueError(f"Unsupported boolean/auto value for {name}: {value}")
    return default, False


def _has_env(names: tuple[str, ...]) -> bool:
    return any(name in os.environ for name in names)


def _force_shot() -> int | None:
    value = os.environ.get("SGLANG_CUSTOM_AR_FORCE_SHOT") or os.environ.get(
        "SGL_CUSTOM_AR_FORCE_SHOT"
    )
    if value is None:
        return None
    normalized = value.lower()
    aliases = {
        "push": SHOT_PUSH,
        "one_shot_push": SHOT_PUSH,
        "0": SHOT_PUSH,
        "push_wide": SHOT_PUSH_WIDE,
        "wide_push": SHOT_PUSH_WIDE,
        "3": SHOT_PUSH_WIDE,
        "one_shot": SHOT_ONE_STAGE,
        "1": SHOT_ONE_STAGE,
        "two_shot": SHOT_TWO_STAGE,
        "2": SHOT_TWO_STAGE,
        "two_shot_512": SHOT_TWO_STAGE_512,
        "2_512": SHOT_TWO_STAGE_512,
        "4": SHOT_TWO_STAGE_512,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported MUSA custom AR shot: {value}")
    return aliases[normalized]


def is_shot_forced() -> bool:
    return _force_shot() is not None


def push_buffer_bytes(nbytes: int, world_size: int) -> int:
    # Push data uses a v2-style 2-stage ring: stages * source-rank slots.
    return int(nbytes) * (2 * int(world_size))


def _push_threshold_bytes() -> int:
    return int(
        os.environ.get(
            "SGLANG_CUSTOM_AR_PUSH_THRESHOLD_BYTES",
            os.environ.get("SGL_CUSTOM_AR_PUSH_THRESHOLD_BYTES", str(256 * 1024)),
        )
    )


def _push_wide_threshold_bytes() -> int:
    return int(
        os.environ.get(
            "SGLANG_CUSTOM_AR_PUSH_WIDE_THRESHOLD_BYTES",
            os.environ.get("SGL_CUSTOM_AR_PUSH_WIDE_THRESHOLD_BYTES", "0"),
        )
    )


def _push_wide_min_bytes() -> int:
    return int(
        os.environ.get(
            "SGLANG_CUSTOM_AR_PUSH_WIDE_MIN_BYTES",
            os.environ.get("SGL_CUSTOM_AR_PUSH_WIDE_MIN_BYTES", str(1024 * 1024)),
        )
    )


def use_push_in_graph() -> bool:
    value = os.environ.get("SGLANG_CUSTOM_AR_PUSH_IN_GRAPH") or os.environ.get(
        "SGL_CUSTOM_AR_PUSH_IN_GRAPH", "0"
    )
    return value.lower() in ("1", "true", "yes", "on")


def _default_threads_blocks(world_size: int, shot: int) -> tuple[int, int]:
    if shot == SHOT_PUSH:
        return 512, 14
    if shot == SHOT_PUSH_WIDE:
        return 512, 36
    if shot == SHOT_TWO_STAGE_512:
        if world_size == 4:
            return 512, 120
        if world_size == 8:
            return 512, 120
        return 512, 48
    if world_size == 2:
        if shot == SHOT_ONE_STAGE:
            return 512, 56
        if shot == SHOT_TWO_STAGE:
            return 1024, 56
        return 512, 36
    if world_size == 4:
        if shot == SHOT_TWO_STAGE:
            return 1024, 40
        return 1024, 60
    if world_size == 8:
        return 768, 80
    return 512, 36


def preferred_shot(world_size: int, nbytes: int) -> int:
    force_shot = _force_shot()
    if force_shot is not None:
        return force_shot

    if world_size == 2 and 256 * 1024 < nbytes <= 512 * 1024:
        return SHOT_TWO_STAGE
    if world_size == 4 and nbytes <= 512 * 1024:
        return SHOT_TWO_STAGE_512
    push_threshold = _push_threshold_bytes()
    if world_size <= 4 and push_threshold > 0 and nbytes <= push_threshold:
        return SHOT_PUSH
    if world_size == 2 and 512 * 1024 < nbytes <= 2 * 1024 * 1024:
        return SHOT_ONE_STAGE
    push_wide_threshold = _push_wide_threshold_bytes()
    if (
        push_wide_threshold > 0
        and nbytes >= _push_wide_min_bytes()
        and nbytes <= push_wide_threshold
    ):
        return SHOT_PUSH_WIDE

    return preferred_fallback_shot(world_size, nbytes)


def preferred_fallback_shot(world_size: int, nbytes: int) -> int:
    one_shot_threshold = int(
        os.environ.get(
            "SGLANG_CUSTOM_AR_1SHOT_THRESHOLD_BYTES",
            os.environ.get("SGL_CUSTOM_AR_1SHOT_THRESHOLD_BYTES", str(4 * 1024 * 1024)),
        )
    )
    if world_size == 2 and nbytes >= one_shot_threshold:
        return SHOT_ONE_STAGE
    return SHOT_TWO_STAGE


def preferred_graph_fallback_shot(world_size: int, nbytes: int) -> int:
    if world_size == 2 and 256 * 1024 < nbytes <= 512 * 1024:
        return SHOT_TWO_STAGE
    if world_size == 4 and 512 * 1024 < nbytes <= 1024 * 1024:
        return SHOT_TWO_STAGE_512
    if world_size == 8 and nbytes <= 256 * 1024:
        return SHOT_TWO_STAGE_512
    one_shot_threshold = int(
        os.environ.get(
            "SGLANG_CUSTOM_AR_GRAPH_1SHOT_THRESHOLD_BYTES",
            os.environ.get(
                "SGL_CUSTOM_AR_GRAPH_1SHOT_THRESHOLD_BYTES", str(256 * 1024)
            ),
        )
    )
    if world_size == 2 and nbytes >= one_shot_threshold:
        return SHOT_ONE_STAGE
    return preferred_fallback_shot(world_size, nbytes)


def _compile_config(world_size: int, shot: int) -> CompileConfig:
    default_threads, default_blocks = _default_threads_blocks(world_size, shot)
    threads = _env_int(
        (
            f"SGLANG_CUSTOM_AR_{shot}SHOT_THREADS",
            f"SGL_CUSTOM_AR_{shot}SHOT_THREADS",
            "SGLANG_CUSTOM_AR_THREADS",
            "SGL_CUSTOM_AR_THREADS",
        ),
        default_threads,
    )
    block_env_names = (
        f"SGLANG_CUSTOM_AR_{shot}SHOT_BLOCKS",
        f"SGL_CUSTOM_AR_{shot}SHOT_BLOCKS",
        "SGLANG_CUSTOM_AR_BLOCKS",
        "SGL_CUSTOM_AR_BLOCKS",
    )
    blocks = _env_int(
        block_env_names,
        default_blocks,
    )
    vector_load = _env_int(
        (
            "SGLANG_CUSTOM_AR_VECTOR_LOAD",
            "SGL_CUSTOM_AR_VECTOR_LOAD",
        ),
        0,
    )
    default_atomic_barrier = 0 if shot in (SHOT_PUSH, SHOT_ONE_STAGE) else 1
    atomic_barrier = _env_int(
        (
            f"SGLANG_CUSTOM_AR_{shot}SHOT_ATOMIC_BARRIER",
            f"SGL_CUSTOM_AR_{shot}SHOT_ATOMIC_BARRIER",
            "SGLANG_CUSTOM_AR_ATOMIC_BARRIER",
            "SGL_CUSTOM_AR_ATOMIC_BARRIER",
        ),
        default_atomic_barrier,
    )
    max_blocks = _env_int(
        (
            f"SGLANG_CUSTOM_AR_{shot}SHOT_MAX_BLOCKS",
            f"SGL_CUSTOM_AR_{shot}SHOT_MAX_BLOCKS",
            "SGLANG_CUSTOM_AR_MAX_BLOCKS",
            "SGL_CUSTOM_AR_MAX_BLOCKS",
        ),
        max(120, blocks),
    )
    default_dynamic_blocks = (
        0
        if shot in (SHOT_PUSH, SHOT_PUSH_WIDE)
        or (
            world_size == 2
            and shot in (SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512)
        )
        or (world_size == 4 and shot in (SHOT_TWO_STAGE, SHOT_TWO_STAGE_512))
        or _has_env(block_env_names)
        else 1
    )
    dynamic_blocks = _env_int(
        (
            f"SGLANG_CUSTOM_AR_{shot}SHOT_DYNAMIC_BLOCKS",
            f"SGL_CUSTOM_AR_{shot}SHOT_DYNAMIC_BLOCKS",
            "SGLANG_CUSTOM_AR_DYNAMIC_BLOCKS",
            "SGL_CUSTOM_AR_DYNAMIC_BLOCKS",
        ),
        default_dynamic_blocks,
    )
    push_polling = _env_int(
        (
            "SGLANG_CUSTOM_AR_PUSH_POLLING",
            "SGL_CUSTOM_AR_PUSH_POLLING",
        ),
        0,
    )
    push_16b_asm, push_16b_asm_forced = _env_auto_bool(
        (
            "SGLANG_CUSTOM_AR_PUSH_16B_ASM",
            "SGL_CUSTOM_AR_PUSH_16B_ASM",
        ),
        True,
    )
    if shot not in (SHOT_PUSH, SHOT_PUSH_WIDE) or push_polling == 0:
        push_16b_asm = False
        push_16b_asm_forced = False
    double_store_2shot = _env_int(
        (
            "SGLANG_CUSTOM_AR_2SHOT_DOUBLE_STORE",
            "SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE",
        ),
        0,
    )
    one_shot_2rank_special = _env_int(
        (
            "SGLANG_CUSTOM_AR_1SHOT_2RANK_SPECIAL",
            "SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL",
        ),
        1,
    )
    push_skip_start_barrier = _env_int(
        (
            "SGLANG_CUSTOM_AR_PUSH_SKIP_START_BARRIER",
            "SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER",
        ),
        0,
    )
    return CompileConfig(
        threads=threads,
        blocks=blocks,
        vector_load=vector_load,
        atomic_barrier=atomic_barrier,
        max_blocks=max_blocks,
        dynamic_blocks=dynamic_blocks,
        push_polling=push_polling,
        push_16b_asm=int(push_16b_asm),
        double_store_2shot=double_store_2shot,
        one_shot_2rank_special=one_shot_2rank_special,
        push_skip_start_barrier=push_skip_start_barrier,
        push_16b_asm_forced=push_16b_asm_forced,
    )


def _load_custom_ar_module(
    world_size: int,
    shot: int,
    config: CompileConfig,
):
    name = _compile_name(shot, config)
    return load_musa_jit(
        name,
        ("distributed/custom_all_reduce.mu",),
        extra_musa_cflags=_musa_cflags(config),
    )


def _fused_rmsnorm_token_2stage(
    world_size: int, row_hidden: int = 0, override: int = -1
) -> int:
    default = 1 if world_size in (4, 8) else 0
    if override >= 0:
        default = int(override)
    return default


def _fused_rmsnorm_default_blocks(
    world_size: int,
    token_2stage: int,
    config: CompileConfig,
    short_rows: bool = False,
    row_hidden: int = 0,
    small_rows: bool = False,
) -> int:
    if world_size == 2 and row_hidden == 1024 and not short_rows:
        return 80
    if world_size == 8 and token_2stage:
        if row_hidden == 1536:
            return 180
        if row_hidden == 3072:
            return 120
        return 120
    if world_size == 4 and token_2stage:
        if row_hidden % 8 != 0:
            return 40
        if row_hidden == 1536:
            return 120
        if row_hidden == 1024 and short_rows:
            return 240
        if row_hidden == 4096 and short_rows:
            return 80
        if row_hidden == 4096:
            return 56
        if row_hidden in (1024, 2048, 4096, 8192) and not short_rows:
            return 80
        return 56
    if world_size == 8 and row_hidden == 1536 and not short_rows:
        return 60
    return config.blocks


def _fused_rmsnorm_abi_max_blocks(
    world_size: int, token_2stage: int, config: CompileConfig
) -> int:
    blocks = [
        config.max_blocks,
        _fused_rmsnorm_default_blocks(world_size, token_2stage, config),
    ]
    for row_hidden in (512, 1024, 2048, 4096, 8192):
        for short_rows in (False, True):
            for small_rows in (False, True):
                blocks.append(
                    _fused_rmsnorm_default_blocks(
                        world_size,
                        token_2stage,
                        config,
                        short_rows=short_rows,
                        row_hidden=row_hidden,
                        small_rows=small_rows,
                    )
                )
    return max(blocks)


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _fused_rmsnorm_config(
    world_size: int,
    config: CompileConfig,
    token_2stage: int,
    short_rows: bool = False,
    row_hidden: int = 0,
    small_rows: bool = False,
) -> CompileConfig:
    fused_vector_load = config.vector_load
    fused_threads = (
        (row_hidden // 8)
        if small_rows
        and world_size in (2, 4, 8)
        and (world_size == 2 or token_2stage)
        and row_hidden in (512, 1024, 2048, 4096)
        else (
            (row_hidden // 8)
            if world_size == 2
            and short_rows
            and 512 <= row_hidden <= 8192
            and row_hidden % 8 == 0
            and not _is_power_of_2(row_hidden)
            else (
                512
                if world_size == 2
                and 6144 <= row_hidden <= 8192
                and row_hidden % 8 == 0
                and not _is_power_of_2(row_hidden)
                else (
                    256
                    if world_size == 2 and row_hidden == 512
                    else (
                        512
                        if world_size == 4 and row_hidden % 8 != 0
                        else (
                            512
                            if world_size == 2 and (row_hidden == 1024 and short_rows)
                            else (
                                1024
                                if world_size == 4
                                and token_2stage
                                and row_hidden in (4096, 8192)
                                else (
                                    512
                                    if world_size == 8
                                    and row_hidden == 1536
                                    and not token_2stage
                                    and not short_rows
                                    else (
                                        (row_hidden // 8)
                                        if world_size == 2
                                        and row_hidden in (1024, 2048)
                                        else (
                                            512
                                            if short_rows and world_size == 2
                                            else (
                                                1024
                                                if world_size == 2
                                                else (
                                                    512
                                                    if world_size in (4, 8)
                                                    and token_2stage
                                                    else config.threads
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    fused_blocks = _fused_rmsnorm_default_blocks(
        world_size, token_2stage, config, short_rows, row_hidden, small_rows
    )
    fused_dynamic_blocks = (
        0
        if (
            (world_size == 8 and token_2stage)
            or (
                world_size == 8
                and row_hidden == 1536
                and not token_2stage
                and not short_rows
            )
        )
        else config.dynamic_blocks
    )
    fused_max_blocks = _fused_rmsnorm_abi_max_blocks(world_size, token_2stage, config)
    fused_max_blocks = max(fused_max_blocks, fused_blocks)
    return replace(
        config,
        threads=fused_threads,
        blocks=fused_blocks,
        vector_load=fused_vector_load,
        max_blocks=fused_max_blocks,
        dynamic_blocks=fused_dynamic_blocks,
    )


def _load_custom_ar_fused_rmsnorm_module(
    world_size: int,
    config: CompileConfig,
    short_rows: bool = False,
    row_hidden: int = 0,
    small_rows: bool = False,
    cache_policy: int = 0,
    row_skip_end_barrier_override: int = -1,
    push_polling_override: int = -1,
    lamport_push_override: int = -1,
    token_2stage_override: int = -1,
    row_warp_inv_rms_override: int = -1,
):
    token_2stage = _fused_rmsnorm_token_2stage(
        world_size, row_hidden, token_2stage_override
    )
    config = _fused_rmsnorm_config(
        world_size, config, token_2stage, short_rows, row_hidden, small_rows
    )
    shfl_2stage = 0
    packed_1stage = (
        1 if world_size == 2 and row_hidden == 8192 and not token_2stage else 0
    )
    safe_packed_1stage = 0
    vec2rank_1stage = 1
    partial_packed_non8 = 1
    vec2_non8 = 0
    vec4_non8 = 0
    regcache_2rank = 1 if short_rows and world_size == 2 else 0
    no_cache_policy = cache_policy == 1
    cache_hidden_limit_default = (
        0
        if no_cache_policy
        else (
            1536
            if world_size == 8
            and row_hidden == 1536
            and not token_2stage
            and not short_rows
            else (
                8192
                if (
                    world_size == 4
                    and token_2stage
                    and row_hidden > 4096
                    and row_hidden % 8 != 0
                )
                else (
                    4096
                    if world_size == 2 or (world_size in (4, 8) and token_2stage)
                    else 2048
                )
            )
        )
    )
    cache_hidden_limit = cache_hidden_limit_default
    typed_cache_hidden_limit = 0 if no_cache_policy else 8192
    h8192_blocks = 16
    weight_cache_hidden_limit = (
        0
        if (
            (
                world_size == 8
                and row_hidden == 1536
                and not token_2stage
                and not short_rows
            )
            or (world_size == 4 and row_hidden == 8192 and (short_rows or small_rows))
        )
        else 8192
    )
    row_weight_cache = (
        1 if world_size == 8 and token_2stage and row_hidden == 6144 else 0
    )
    row_weight_cache_min_rows = (
        4096
        if world_size == 8 and token_2stage and row_hidden == 6144
        else 512 if world_size == 8 and token_2stage else 2048
    )
    row_shared_inv_rms = (
        1 if world_size == 8 and token_2stage and row_hidden == 1024 else 0
    )
    row_warp_inv_rms = 0
    if row_warp_inv_rms_override >= 0:
        row_warp_inv_rms = int(row_warp_inv_rms_override)
    row_skip_end_barrier = 0
    if row_skip_end_barrier_override >= 0:
        row_skip_end_barrier = int(row_skip_end_barrier_override)
    use_row_bypass_load = (
        (world_size == 4 and token_2stage)
        or (world_size == 8 and token_2stage and row_hidden == 1536)
        or (world_size == 8 and token_2stage and row_hidden == 8192)
    )
    sums_bypass_load = 3 if use_row_bypass_load else 0
    tmp_with_residual = (
        1
        if token_2stage
        and (
            (
                world_size == 4
                and (
                    row_hidden in (512, 1024) or (row_hidden == 4096 and not short_rows)
                )
            )
            or (
                world_size == 8
                and (row_hidden == 1536 or (row_hidden in (1024, 2048) and small_rows))
            )
        )
        else 0
    )
    warp_rows = 0
    push_polling = 0
    if push_polling_override >= 0:
        push_polling = int(push_polling_override)
    push_slots = 2
    push_skip_end_barrier = 0
    lamport_push = 0
    if lamport_push_override >= 0:
        lamport_push = int(lamport_push_override)
    push_min_rows = 128 if lamport_push else 1
    push_max_rows = 128
    lamport_end_barrier = 1
    small_rows_flag = (
        1
        if small_rows
        and world_size in (2, 4, 8)
        and (world_size == 2 or token_2stage)
        and row_hidden in (512, 1024, 2048, 4096)
        else 0
    )
    name = (
        "sgl_musa_ar_rn_"
        + _compile_name_short(SHOT_ONE_STAGE, config)
        + (
            f"_r2{token_2stage}_s2{shfl_2stage}_p1{packed_1stage}"
            f"_sp{safe_packed_1stage}_v2{vec2rank_1stage}"
            f"_rc{regcache_2rank}"
            f"_c{cache_hidden_limit}_tc{typed_cache_hidden_limit}"
            f"_cp{cache_policy}"
            f"_h8{h8192_blocks}"
            f"_wc{weight_cache_hidden_limit}_rw{row_weight_cache}"
            f"_rm{row_weight_cache_min_rows}"
            f"_si{row_shared_inv_rms}_wi{row_warp_inv_rms}"
            f"_eb{row_skip_end_barrier}"
            f"_pn8{partial_packed_non8}"
            f"_vn8{vec2_non8}"
            f"_v4n8{vec4_non8}"
            f"_sb{sums_bypass_load}"
            f"_tr{tmp_with_residual}"
            f"_wr{warp_rows}"
            f"_pp{push_polling}"
            f"_pm{push_min_rows}"
            f"_px{push_max_rows}"
            f"_psl{push_slots}"
            f"_pse{push_skip_end_barrier}"
            f"_lp{lamport_push}"
            f"_leb{lamport_end_barrier}"
            f"_sr{small_rows_flag}"
            f"_m2_r2s"
        )
    )
    return load_musa_jit(
        name,
        ("distributed/custom_all_reduce_rmsnorm.mu",),
        extra_musa_cflags=_musa_cflags(config)
        + (
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE={token_2stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SHFL_2STAGE={shfl_2stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE={packed_1stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE={safe_packed_1stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE={vec2rank_1stage}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8={partial_packed_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8={vec2_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8={vec4_non8}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK={regcache_2rank}",
            f"-DSGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT={cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT={typed_cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS={h8192_blocks}",
            f"-DSGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT={weight_cache_hidden_limit}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE={row_weight_cache}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS={row_weight_cache_min_rows}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS={row_shared_inv_rms}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS={row_warp_inv_rms}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER={row_skip_end_barrier}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD={sums_bypass_load}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL={tmp_with_residual}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_WARP_ROWS={warp_rows}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING={push_polling}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS={push_min_rows}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS={push_max_rows}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SLOTS={push_slots}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SKIP_END_BARRIER={push_skip_end_barrier}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH={lamport_push}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_END_BARRIER={lamport_end_barrier}",
            f"-DSGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS={small_rows_flag}",
        ),
    )


def _compile_name(shot: int, config: CompileConfig) -> str:
    return (
        f"sglang_musa_custom_all_reduce_s{shot}_t{config.threads}_b{config.blocks}"
        f"_v{config.vector_load}_ab{config.atomic_barrier}"
        f"_mb{config.max_blocks}_db{config.dynamic_blocks}"
        f"_pp{config.push_polling}_p16{config.push_16b_asm}"
        f"_ds{config.double_store_2shot}_r2s{config.one_shot_2rank_special}"
        f"_psb{config.push_skip_start_barrier}"
    )


def _compile_name_short(shot: int, config: CompileConfig) -> str:
    return (
        f"s{shot}_t{config.threads}_b{config.blocks}"
        f"_v{config.vector_load}_a{config.atomic_barrier}"
        f"_m{config.max_blocks}_d{config.dynamic_blocks}"
        f"_q{config.push_polling}_p{config.push_16b_asm}"
        f"_ds{config.double_store_2shot}_r{config.one_shot_2rank_special}"
        f"_ps{config.push_skip_start_barrier}"
    )


def _musa_cflags(config: CompileConfig) -> tuple[str, ...]:
    return (
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-D__MUSA_ARCH_LIST__=310",
        f"-DSGL_CUSTOM_AR_THREADS={config.threads}",
        f"-DSGL_CUSTOM_AR_BLOCKS={config.blocks}",
        f"-DSGL_CUSTOM_AR_VECTOR_LOAD={config.vector_load}",
        f"-DSGL_CUSTOM_AR_ATOMIC_BARRIER={config.atomic_barrier}",
        f"-DSGL_CUSTOM_AR_MAX_BLOCKS={config.max_blocks}",
        f"-DSGL_CUSTOM_AR_DYNAMIC_BLOCKS={config.dynamic_blocks}",
        f"-DSGL_CUSTOM_AR_PUSH_POLLING={config.push_polling}",
        f"-DSGL_CUSTOM_AR_PUSH_16B_ASM={config.push_16b_asm}",
        f"-DSGL_CUSTOM_AR_2SHOT_DOUBLE_STORE={config.double_store_2shot}",
        f"-DSGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL={config.one_shot_2rank_special}",
        f"-DSGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER={config.push_skip_start_barrier}",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
    )


@functools.lru_cache(maxsize=32)
def _custom_ar_module(world_size: int, shot: int):
    config = _compile_config(world_size, shot)
    try:
        return _load_custom_ar_module(world_size, shot, config)
    except Exception:
        if config.push_16b_asm == 0 or config.push_16b_asm_forced:
            raise
        return _load_custom_ar_module(world_size, shot, replace(config, push_16b_asm=0))


@functools.lru_cache(maxsize=96)
def _custom_ar_fused_rmsnorm_module(
    world_size: int,
    short_rows: bool = False,
    row_hidden: int = 0,
    small_rows: bool = False,
    cache_policy: int = 0,
    row_skip_end_barrier: bool = False,
    push_polling: int = -1,
    lamport_push: int = -1,
    token_2stage_override: int = -1,
    row_warp_inv_rms_override: int = -1,
):
    token_2stage = _fused_rmsnorm_token_2stage(
        world_size, row_hidden, token_2stage_override
    )
    forced_shot = _force_shot()
    if forced_shot in (SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512):
        shot = forced_shot
    else:
        shot = SHOT_TWO_STAGE if token_2stage else SHOT_ONE_STAGE
    config = _compile_config(world_size, shot)
    try:
        return _load_custom_ar_fused_rmsnorm_module(
            world_size,
            config,
            short_rows,
            row_hidden,
            small_rows,
            int(cache_policy),
            int(bool(row_skip_end_barrier)),
            int(push_polling),
            int(lamport_push),
            int(token_2stage_override),
            int(row_warp_inv_rms_override),
        )
    except Exception:
        if config.push_16b_asm == 0 or config.push_16b_asm_forced:
            raise
        return _load_custom_ar_fused_rmsnorm_module(
            world_size,
            replace(config, push_16b_asm=0),
            short_rows,
            row_hidden,
            small_rows,
            int(cache_policy),
            int(bool(row_skip_end_barrier)),
            int(push_polling),
            int(lamport_push),
            int(token_2stage_override),
            int(row_warp_inv_rms_override),
        )


@functools.lru_cache(maxsize=32)
def _custom_ar_pybind_module(world_size: int, shot: int):
    config = _compile_config(world_size, shot)

    def load(config: CompileConfig):
        name = _compile_name(shot, config)
        torch_namespace = "sglang_musa_jit_ar_" + name
        return load_musa_pybind(
            name + "_pybind",
            (
                "distributed/custom_all_reduce.mu",
                "distributed/custom_all_reduce_pybind.cpp",
            ),
            extra_cflags=(f"-DSGL_CUSTOM_AR_TORCH_NS={torch_namespace}",),
            extra_musa_cflags=_musa_cflags(config),
        )

    try:
        return load(config)
    except Exception:
        if config.push_16b_asm == 0 or config.push_16b_asm_forced:
            raise
        return load(replace(config, push_16b_asm=0))


def _custom_ar_torch_namespace(world_size: int, shot: int) -> str:
    return "sglang_musa_jit_ar_" + _compile_name(
        int(shot), _compile_config(int(world_size), int(shot))
    )


def _module_shots(world_size: int) -> tuple[int, ...]:
    world_size = int(world_size)
    forced_shot = _force_shot()
    if world_size == 2:
        shots = [SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512]
    elif world_size == 4:
        shots = [SHOT_TWO_STAGE, SHOT_TWO_STAGE_512]
    else:
        shots = [SHOT_TWO_STAGE]
    if forced_shot in (SHOT_PUSH, SHOT_PUSH_WIDE) or (
        world_size <= 4 and _push_threshold_bytes() > 0
    ):
        shots.insert(0, SHOT_PUSH)
    if (
        forced_shot == SHOT_PUSH_WIDE
        or _push_wide_threshold_bytes() > _push_threshold_bytes()
    ):
        shots.insert(1 if SHOT_PUSH in shots else 0, SHOT_PUSH_WIDE)
    return tuple(shots)


def ensure_compiled(world_size: int) -> None:
    world_size = int(world_size)
    for shot in _module_shots(world_size):
        _custom_ar_module(world_size, shot)


def meta_size(world_size: int = 8) -> int:
    # Keep this in sync with distributed/custom_all_reduce.mu::Signal.
    # Avoid loading every JIT module during communicator initialization; the
    # actual launch module is loaded lazily on first use.
    world_size = int(world_size)
    max_blocks = max(
        _compile_config(world_size, shot).max_blocks
        for shot in _module_shots(world_size)
    )
    if world_size in (2, 4, 8):
        token_2stage = _fused_rmsnorm_token_2stage(world_size)
        forced_shot = _force_shot()
        if forced_shot in (SHOT_ONE_STAGE, SHOT_TWO_STAGE, SHOT_TWO_STAGE_512):
            fused_shot = forced_shot
        else:
            fused_shot = SHOT_TWO_STAGE if token_2stage else SHOT_ONE_STAGE
        base_fused_config = _compile_config(world_size, fused_shot)
        # The fused RMSNorm kernels compute their scratch base as `signal + 1`,
        # so every shape-specific fused module in a communicator must agree on
        # the Signal ABI size.  Otherwise a smaller module writes scratch into
        # the larger module's counter area during graph replay.
        max_blocks = max(
            max_blocks,
            _fused_rmsnorm_abi_max_blocks(world_size, token_2stage, base_fused_config),
        )

    def align(value: int, alignment: int = 128) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    flag_bytes = 4
    max_ranks = 8
    offset = 0
    offset = align(offset) + flag_bytes * max_blocks * max_ranks
    offset = align(offset) + 2 * flag_bytes * max_blocks * max_ranks
    offset = align(offset) + flag_bytes * max_blocks
    offset = align(offset) + flag_bytes
    offset = align(offset) + flag_bytes
    offset = align(offset) + flag_bytes
    return align(offset)


def launch_registered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    _custom_ar_module(int(world_size), int(shot)).sgl_musa_custom_ar_launch(
        rank_data,
        signal_ptrs_cpu,
        out,
        int(self_signal_ptr),
        int(rank),
        int(world_size),
        int(shot),
    )


def launch_registered_func(world_size: int, shot: int):
    return _custom_ar_module(int(world_size), int(shot)).sgl_musa_custom_ar_launch


def launch_empty_func(world_size: int, shot: int):
    return _custom_ar_module(int(world_size), int(shot)).sgl_musa_custom_ar_launch_empty


def create_context(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    self_signal_ptr: int,
    rank: int,
    world_size: int,
    shot: int,
) -> int:
    return int(
        _custom_ar_module(int(world_size), int(shot)).sgl_musa_custom_ar_create_context(
            rank_data,
            signal_ptrs_cpu,
            int(self_signal_ptr),
            int(rank),
            int(world_size),
        )
    )


def dispose_context(context_ptr: int, world_size: int, shot: int) -> None:
    _custom_ar_module(int(world_size), int(shot)).sgl_musa_custom_ar_dispose_context(
        int(context_ptr)
    )


def launch_context_func(world_size: int, shot: int):
    return _custom_ar_module(
        int(world_size), int(shot)
    ).sgl_musa_custom_ar_launch_context


def launch_context_pybind_func(world_size: int, shot: int):
    return _custom_ar_pybind_module(int(world_size), int(shot)).launch_context


def launch_unregistered_pybind_func(world_size: int, shot: int):
    return _custom_ar_pybind_module(int(world_size), int(shot)).launch_unregistered


def create_unregistered_context_pybind_func(world_size: int, shot: int):
    return _custom_ar_pybind_module(
        int(world_size), int(shot)
    ).create_unregistered_context


def dispose_unregistered_context_pybind_func(world_size: int, shot: int):
    return _custom_ar_pybind_module(
        int(world_size), int(shot)
    ).dispose_unregistered_context


def launch_unregistered_context_pybind_func(world_size: int, shot: int):
    return _custom_ar_pybind_module(
        int(world_size), int(shot)
    ).launch_unregistered_context


def launch_context_torchop_func(world_size: int, shot: int):
    world_size = int(world_size)
    shot = int(shot)
    _custom_ar_pybind_module(world_size, shot)
    namespace = _custom_ar_torch_namespace(world_size, shot)
    return getattr(torch.ops, namespace).launch_context


def launch_unregistered_func(world_size: int, shot: int):
    return _custom_ar_module(
        int(world_size), int(shot)
    ).sgl_musa_custom_ar_launch_unregistered


def launch_fused_allreduce_rmsnorm_unregistered_func(
    world_size: int,
    hidden: int = 0,
    short_rows: bool = False,
    small_rows: bool = False,
    cache_policy: int = 0,
    row_skip_end_barrier: bool = False,
    push_polling: int = -1,
    lamport_push: int = -1,
    row_warp_inv_rms_override: int = -1,
):
    return _custom_ar_fused_rmsnorm_module(
        int(world_size),
        bool(short_rows),
        int(hidden),
        bool(small_rows),
        int(cache_policy),
        bool(row_skip_end_barrier),
        int(push_polling),
        int(lamport_push),
        -1,
        int(row_warp_inv_rms_override),
    ).sgl_musa_custom_ar_fused_allreduce_rmsnorm_unregistered


def launch_fused_allreduce_rmsnorm_registered_func(
    world_size: int,
    hidden: int = 0,
    short_rows: bool = False,
    small_rows: bool = False,
    cache_policy: int = 0,
    row_skip_end_barrier: bool = False,
    row_warp_inv_rms_override: int = -1,
):
    return _custom_ar_fused_rmsnorm_module(
        int(world_size),
        bool(short_rows),
        int(hidden),
        bool(small_rows),
        int(cache_policy),
        bool(row_skip_end_barrier),
        0,
        0,
        -1,
        int(row_warp_inv_rms_override),
    ).sgl_musa_custom_ar_fused_allreduce_rmsnorm_registered


def launch_fused_allreduce_rmsnorm_row_registered_func(
    world_size: int,
    hidden: int = 0,
    short_rows: bool = False,
    small_rows: bool = False,
    row_skip_end_barrier: bool = False,
    token_2stage_override: int = 1,
    row_warp_inv_rms_override: int = -1,
):
    return _custom_ar_fused_rmsnorm_module(
        int(world_size),
        bool(short_rows),
        int(hidden),
        bool(small_rows),
        0,
        bool(row_skip_end_barrier),
        0,
        0,
        int(token_2stage_override),
        int(row_warp_inv_rms_override),
    ).sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_registered


def launch_fused_allreduce_rmsnorm_row_unregistered_func(
    world_size: int,
    hidden: int = 0,
    short_rows: bool = False,
    small_rows: bool = False,
    row_skip_end_barrier: bool = False,
    token_2stage_override: int = 1,
    row_warp_inv_rms_override: int = -1,
):
    return _custom_ar_fused_rmsnorm_module(
        int(world_size),
        bool(short_rows),
        int(hidden),
        bool(small_rows),
        0,
        bool(row_skip_end_barrier),
        0,
        0,
        int(token_2stage_override),
        int(row_warp_inv_rms_override),
    ).sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_unregistered


def launch_unregistered(
    rank_data: torch.Tensor,
    signal_ptrs_cpu: torch.Tensor,
    inp: torch.Tensor,
    out: torch.Tensor,
    self_signal_ptr: int,
    self_buffer_ptr: int,
    max_size_bytes: int,
    rank: int,
    world_size: int,
    shot: int,
) -> None:
    _custom_ar_module(
        int(world_size), int(shot)
    ).sgl_musa_custom_ar_launch_unregistered(
        rank_data,
        signal_ptrs_cpu,
        inp,
        out,
        int(self_signal_ptr),
        int(self_buffer_ptr),
        int(max_size_bytes),
        int(rank),
        int(world_size),
        int(shot),
    )
