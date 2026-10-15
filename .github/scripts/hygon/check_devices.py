#!/usr/bin/env python3
"""Check allocated HCU devices without selecting or reserving other GPUs."""

from __future__ import annotations

import argparse
import json
import os


def check_devices(torch, min_gpus=2, min_free_fraction=0.95):
    if min_gpus < 2:
        raise RuntimeError("Hygon CI model cases require at least two GPUs")
    if not 0 < min_free_fraction <= 1:
        raise RuntimeError("min_free_fraction must be in (0, 1]")
    if not getattr(torch, "__hcu_version__", None) or not getattr(
        torch.version, "hip", None
    ):
        raise RuntimeError("The Hygon DTK Torch runtime is required")
    if not torch.cuda.is_available():
        raise RuntimeError("Hygon GPUs are unavailable through torch.cuda/HIP")
    count = torch.cuda.device_count()
    if count < min_gpus:
        raise RuntimeError(f"Hygon TP2 CI needs {min_gpus} visible GPUs; found {count}")
    devices = []
    for index in range(min_gpus):
        free, total = torch.cuda.mem_get_info(index)
        if total <= 0 or not 0 <= free <= total:
            raise RuntimeError(f"Invalid GPU memory report for visible GPU {index}")
        fraction = free / total
        if fraction < min_free_fraction:
            raise RuntimeError(
                f"Visible GPU {index} is occupied ({fraction:.2%} free); "
                "assign idle GPUs through HIP_VISIBLE_DEVICES before retrying"
            )
        devices.append(
            {
                "visible_index": index,
                "name": torch.cuda.get_device_name(index),
                "free_bytes": free,
                "total_bytes": total,
                "free_fraction": fraction,
            }
        )
    return {
        "torch": str(torch.__version__),
        "hcu_version": str(torch.__hcu_version__),
        "visible_gpu_count": count,
        "checked_devices": devices,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "scope": "availability and occupancy only; setup checks numerical operators",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--min-free-fraction", type=float, default=0.95)
    args = parser.parse_args()
    # Must precede importing Torch: the base pip flagcx extension is CUDA-only.
    from ci_environment import normalized_environment

    environment = normalized_environment(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    import torch

    print(
        json.dumps(
            check_devices(torch, args.min_gpus, args.min_free_fraction), indent=2
        )
    )


if __name__ == "__main__":
    main()
