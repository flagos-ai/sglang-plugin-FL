# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental GLM-5.2 MUSA NSA Indexer Hadamard route."""

import hashlib
from importlib.util import find_spec
from pathlib import Path

import torch

INDEXER = "sglang.srt.layers.attention.nsa.nsa_indexer"
INDEXER_SOURCE_SHA256 = "512ea677a97924eae0187c3ab6ebd858f8c52fd0e92220108303559e16a1272d"


def register_hadamard_hook():
    from flaggems_sglang import hadamard_transform
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    spec = find_spec(INDEXER)
    if (
        spec is None
        or not spec.origin
        or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != INDEXER_SOURCE_SHA256
    ):
        raise RuntimeError("The experimental GLM-5.2 Hadamard hook requires the tested NSA Indexer source")

    def dispatch(original, x):
        if (
            x.device.type == "musa"
            and x.dtype == torch.bfloat16
            and x.ndim in (2, 3)
            and x.shape[-1] == 128
            and x.numel() != 0
            and x.is_contiguous()
        ):
            return hadamard_transform(x, 128 ** -0.5)
        return original(x)

    HookRegistry.register(f"{INDEXER}.rotate_activation", dispatch, HookType.AROUND)
