# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail CI before GPU initialization if an allocated CoreX GPU has busy memory.

This checks an explicit CUDA_VISIBLE_DEVICES allocation with the vendor ixsmi
inventory. It never selects extra devices, evicts workloads, or kills processes.
The fixed 512 MiB limit permits driver bookkeeping, not a shared model workload.
A runner must reserve the chosen devices for the duration of the job.
"""

import csv
import io
import os
import re
import subprocess


def check_inventory(text: str, selected: str) -> list[dict[str, int]]:
    if not re.fullmatch(r"\d+(,\d+)*", selected):
        raise RuntimeError("Explicit numeric CUDA_VISIBLE_DEVICES is required")
    devices = [int(value) for value in selected.split(",")]
    if len(set(devices)) != len(devices):
        raise RuntimeError("Allocated GPU indices must be unique")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) != 3 or "memory.used" not in rows[0][1]:
        raise RuntimeError("Unexpected ixsmi inventory; cannot verify idle memory")
    inventory = {}
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != 3:
            raise RuntimeError(f"Unexpected ixsmi GPU row: {row!r}")
        index = int(row[0].strip())
        memory = []
        for cell in row[1:]:
            match = re.fullmatch(r"\s*(\d+)(?:\s+MiB)?\s*", cell)
            if match is None:
                raise RuntimeError(f"Unexpected ixsmi memory value: {cell!r}")
            memory.append(int(match.group(1)))
        used, total = memory
        if index in inventory or total <= 0 or used > total:
            raise RuntimeError("Invalid or duplicate GPU inventory")
        inventory[index] = {"index": index, "used_mib": used, "total_mib": total}
    checked = []
    for index in devices:
        if index not in inventory:
            raise RuntimeError(f"Allocated physical GPU {index} is absent from ixsmi")
        gpu = inventory[index]
        print(f"Allocated GPU {index}: {gpu['used_mib']} / {gpu['total_mib']} MiB")
        if gpu["used_mib"] > 512:
            raise RuntimeError(
                f"GPU {index} has busy memory; release the allocation first"
            )
        checked.append(gpu)
    return checked


if __name__ == "__main__":
    result = subprocess.run(
        ["ixsmi", "--query-gpu=index,memory.used,memory.total", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    check_inventory(result.stdout, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
