# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Launch Hy4-preview text serving with the SGLang 0.5.18 adapter.

Install the plugin and its pinned INT8 dependency from README.md first. This
entry point targets the NVIDIA sparse-MLA backend and takes a caller-provided
checkpoint path. The VL checkpoint architecture is supported for text only.

Single node (choose TP size for the available checkpoint and GPU memory):
    python examples/hy4_preview_serve.py --model-path /models/Hy4-preview \
        --tp-size 8 --indexer-dtype int8

Additional native server arguments follow ``--``. For a two-node TP16 launch,
run once per node with ``--tp-size 16`` and append ``--nnodes 2``,
``--node-rank 0|1`` and the same ``--dist-init-addr HOST:PORT``.
"""

import argparse
import os
import sys


def build_launch_command(args, extra_args):
    """Forward supported model settings to the native server entry point."""
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--attention-backend",
        "dsa",
        "--dsa-prefill-backend",
        "flashmla_sparse",
        "--dsa-decode-backend",
        "flashmla_sparse",
        "--page-size",
        "64",
        *extra_args,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--indexer-dtype",
        choices=("auto", "fp8", "int8"),
        default=os.environ.get("HY4_INDEXER_DTYPE", "auto"),
    )
    args, extra_args = parser.parse_known_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    environment = os.environ.copy()
    environment["SGLANG_HY4_ENABLE"] = "1"
    environment["HY4_INDEXER_DTYPE"] = args.indexer_dtype
    command = build_launch_command(args, extra_args)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
