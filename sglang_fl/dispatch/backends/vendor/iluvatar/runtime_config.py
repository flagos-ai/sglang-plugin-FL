# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Defaults for the CoreX empty runtime, without NVIDIA kernel packages."""


def apply_iluvatar_runtime_defaults(server_args) -> None:
    # These are optional choices: retain explicit portable backend selections.
    if getattr(server_args, "attention_backend", None) is None:
        server_args.attention_backend = "triton"
    if getattr(server_args, "sampling_backend", None) is None:
        server_args.sampling_backend = "pytorch"
    if getattr(server_args, "mm_attention_backend", None) is None:
        server_args.mm_attention_backend = "triton_attn"

    # CoreX cannot run SGLang's NVIDIA IPC kernels or CUDA piecewise compiler.
    # Standard torch.cuda graph capture remains available.
    server_args.disable_custom_all_reduce = True
    server_args.disable_piecewise_cuda_graph = True
