# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen3.6-35B-A3B MTP against an eager baseline on Iluvatar CoreX.

Uses the 27B example's semantic, target-logprob and acceptance-length gates
with the 35B MoE checkpoint. Both engines use TP4, eager execution and
synchronous scheduling; the baseline is mandatory.

Usage:
  python examples/qwen3_6_35b_a3b_mtp_inference.py [--max-tokens N]

Environment:
  MODEL_PATH  Local checkpoint (default: /models/Qwen3.6-35B-A3B)
  TP_SIZE     Must be 4 (default: 4)
  MAX_TOKENS  Semantic response token limit (default: 256)
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


_MODEL_LABEL = "Qwen3.6-35B-A3B"
_DEFAULT_MODEL_PATH = f"/models/{_MODEL_LABEL}"


def _require_corex() -> None:
    import torch

    vendor = os.environ.get("GEMS_VENDOR", "").strip().lower()
    if vendor and vendor != "iluvatar":
        raise RuntimeError("This MTP entrypoint requires GEMS_VENDOR=iluvatar")
    if not hasattr(torch, "corex") or not torch.cuda.is_available():
        raise RuntimeError("This MTP entrypoint requires Iluvatar CoreX torch")
    if torch.cuda.device_count() < 4:
        raise RuntimeError("TP4 MTP validation requires at least four visible devices")


def _check_checkpoint(model_path: str) -> None:
    config_path = Path(model_path) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if not isinstance(architectures, list) or (
        "Qwen3_5MoeForConditionalGeneration" not in architectures
    ):
        raise ValueError(f"Expected a Qwen3.5/3.6 MoE checkpoint: {config_path}")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"Missing MoE text_config: {config_path}")
    mtp_layers = text_config.get("mtp_num_hidden_layers")
    if type(mtp_layers) is not int or mtp_layers < 1:
        raise ValueError(
            f"Checkpoint must declare at least one MTP layer: {config_path}"
        )


def _load_harness():
    path = Path(__file__).with_name("qwen3_6_27b_mtp_inference.py")
    spec = importlib.util.spec_from_file_location("_qwen3_6_mtp_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the MTP validation harness: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-tokens", type=int, default=os.environ.get("MAX_TOKENS", "256")
    )
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    try:
        tp_size = int(os.environ.get("TP_SIZE", "4"))
    except ValueError:
        parser.error("TP_SIZE must be 4")
    if tp_size != 4:
        parser.error("This Iluvatar MTP validation requires TP_SIZE=4")

    _require_corex()
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    _check_checkpoint(model_path)
    harness = _load_harness()
    harness.MODEL_PATH = model_path
    harness.TP_SIZE = tp_size
    harness.MAX_TOKENS = args.max_tokens

    # The shared harness also serves other backends. Give this entrypoint its
    # fixed correctness mode while retaining its defaults for other entrypoints.
    original_argv = sys.argv
    try:
        sys.argv = [
            str(Path(__file__)),
            "--max-tokens",
            str(args.max_tokens),
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--disable-overlap-schedule",
        ]
        return harness.main(model_label=_MODEL_LABEL)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
