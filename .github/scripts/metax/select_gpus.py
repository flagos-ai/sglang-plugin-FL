#!/usr/bin/env python3
"""Select idle MetaX devices for the TP4 CI suite without creating GPU contexts.

mx-smi (2.3.4) has no machine-readable memory report: ``mx-smi -j`` carries
utilization but not VRAM usage, so both are parsed from the text reports.
mx-smi ``GPU#<n>`` indices match ``CUDA_VISIBLE_DEVICES`` on MetaX.
"""

import os
import re
import subprocess
import sys
import time

_MEMORY_RE = re.compile(r"GPU#(\d+)\b.*?\bvram used\s*:\s*(\d+)\s*KB", re.S)
_USAGE_RE = re.compile(r"GPU#(\d+)\b.*?\bGPU\s*:\s*(\d+)\s*%", re.S)


def _split_devices(report):
    """Split an mx-smi report into per-device blocks."""
    return re.split(r"(?=GPU#\d+\b)", report)[1:]


def parse_vram_used_mib(report):
    used = {}
    for block in _split_devices(report):
        match = _MEMORY_RE.search(block)
        if match:
            used[int(match[1])] = int(match[2]) // 1024
    return used


def parse_utilization(report):
    utilization = {}
    for block in _split_devices(report):
        match = _USAGE_RE.search(block)
        if match:
            utilization[int(match[1])] = int(match[2])
    return utilization


def allowed_gpu_ids(environ):
    """Keep selection within any device allocation supplied by the runner."""
    value = environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or value.strip().lower() == "all":
        return None
    if value.strip().lower() in ("", "none", "void", "-1"):
        return set()
    allowed = set()
    for part in value.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"Unsupported CUDA_VISIBLE_DEVICES allocation: {value!r}")
        allowed.add(int(part))
    return allowed


def idle_gpu_ids(used_mib, utilization, allowed=None, max_used_mib=2048):
    # An idle C550 reports ~858 MiB of driver-reserved VRAM.
    return sorted(
        index
        for index, used in used_mib.items()
        if (allowed is None or index in allowed)
        and used <= max_used_mib
        and utilization.get(index) == 0
    )


def choose_gpu_ids(idle, count=4):
    if len(idle) < count:
        return None
    available = set(idle)
    # Prefer an aligned contiguous group, then any contiguous group. Higher
    # indices avoid the default devices used by unrelated single-GPU jobs.
    groups = [
        list(range(start, start + count))
        for start in sorted(available, reverse=True)
        if set(range(start, start + count)) <= available
    ]
    for group in groups:
        if group[0] % count == 0:
            return group
    return groups[0] if groups else sorted(available)[-count:]


def _mx_smi(*args):
    return subprocess.run(
        ["mx-smi", *args], check=True, capture_output=True, text=True, timeout=60
    ).stdout


def main():
    count = int(os.environ.get("METAX_CI_GPU_COUNT", "4"))
    wait_seconds = int(os.environ.get("METAX_CI_GPU_WAIT_SECONDS", "1800"))
    deadline = time.monotonic() + wait_seconds
    allowed = allowed_gpu_ids(os.environ)
    while True:
        used = parse_vram_used_mib(_mx_smi("--show-memory"))
        utilization = parse_utilization(_mx_smi("--show-usage"))
        idle = idle_gpu_ids(used, utilization, allowed)
        selected = choose_gpu_ids(idle, count)
        state = ", ".join(
            f"{index}: {used[index]}MiB, util={utilization.get(index, '?')}%"
            for index in sorted(used)
        )
        print(f"MetaX device state: {state}", file=sys.stderr, flush=True)
        if selected is not None:
            print(
                f"Selected idle MetaX devices: {selected}", file=sys.stderr, flush=True
            )
            # stdout is consumed by check.sh before torch is imported.
            print(",".join(map(str, selected)))
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"TP{count} CI requires {count} idle allocated GPUs; available: {idle}"
            )
        print(f"Waiting for {count} idle MetaX devices...", file=sys.stderr, flush=True)
        time.sleep(min(30, remaining))


if __name__ == "__main__":
    main()
