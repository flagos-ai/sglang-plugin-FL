#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail before loading weights if the CoreX empty runtime is incomplete."""

import argparse
import importlib
from importlib.metadata import version
import json
import os


def verify(min_devices=1, import_runtime=True, fused_profile="vendor"):
    required_env = {
        "SGLANG_PLATFORM": "sglang_fl",
        "SGLANG_PLUGINS": "sglang_fl",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_IS_FLASHINFER_AVAILABLE": "0",
    }
    for key, value in required_env.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"Set {key}={value} before starting Python")

    import torch

    if not hasattr(torch, "corex") or not torch.cuda.is_available():
        raise RuntimeError(
            "A CoreX PyTorch build and visible Iluvatar GPU are required"
        )
    count = torch.cuda.device_count()
    if count < min_devices:
        raise RuntimeError(f"Need {min_devices} visible GPUs, found {count}")
    if version("sglang") != "0.5.18":
        raise RuntimeError(f"Expected SGLang 0.5.18, found {version('sglang')}")
    for name in ("sgl_kernel", "flashinfer"):
        module = importlib.import_module(name)
        if not getattr(module, "__flagos_import_shim__", False):
            raise RuntimeError(
                f"Expected the explicit Iluvatar empty import shim: {name}"
            )

    import sglang  # noqa: F401
    import sglang_fl
    from sglang_fl.dispatch import resolve_op
    from sglang_fl.dispatch.manager import get_default_manager

    sglang_fl.load_plugin()
    if not sglang_fl.is_plugin_active():
        raise RuntimeError("SGLang-FL plugin did not finish initialization")
    from sglang_fl.dispatch.fla_patch import get_original

    for name in (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ):
        if get_original(name) is None:
            raise RuntimeError(f"The FLA dispatch bridge was not installed: {name}")
    ops = (
        "silu_and_mul",
        "rms_norm",
        "gemma_rms_norm",
        "mrotary_embedding",
        "topk",
        "fused_moe",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "fused_recurrent_gated_delta_rule_packed_decode",
    )
    manager = get_default_manager()
    resolved = {}
    for name in ops:
        fn = resolve_op(name)
        impl_id = manager._get_impl_id_for_fn(name, fn)
        if name == "gemma_rms_norm":
            expected = "reference.torch"
        elif fused_profile == "vendor" and name in {
            "fused_moe",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        }:
            expected = "vendor.iluvatar"
        else:
            expected = "default.flagos"
        if impl_id != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {impl_id}")
        resolved[name] = impl_id
    if import_runtime:
        # Covers CUDA-alias imports that a CPU-only import smoke cannot exercise.
        for name in (
            "sglang.srt.server_args",
            "sglang.srt.managers.scheduler",
            "sglang.srt.model_executor.model_runner",
        ):
            importlib.import_module(name)
    return {
        "fused_profile": fused_profile,
        "versions": {
            name: version(name)
            for name in (
                "sglang",
                "torch",
                "flag_gems",
                "flagtree",
                "sglang_fl",
            )
        },
        "devices": count,
        "gpu": torch.cuda.get_device_name(0),
        "resolved": resolved,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-devices", type=int, default=4)
    parser.add_argument("--skip-runtime-imports", action="store_true")
    parser.add_argument(
        "--fused-profile", choices=("vendor", "flagos"), default="vendor"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.min_devices, not args.skip_runtime_imports, args.fused_profile),
            indent=2,
        )
    )
