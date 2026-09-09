#!/usr/bin/env python3
"""Select idle MUSA devices for the TP4 CI suite without creating GPU contexts."""

import json
import os
import re
import subprocess
import sys
import time


def allowed_gpu_ids(environ, report=None):
    """Keep selection within any device allocation supplied by the runner."""
    uuid_to_index = {
        gpu["GPU UUID"].lower(): int(gpu["Index"])
        for gpu in (report or {}).get("GPU", [])
        if "GPU UUID" in gpu
    }
    allowed = None
    for key in ("MTHREADS_VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES"):
        value = environ.get(key)
        if value is None or value.strip().lower() == "all":
            continue
        if value.strip().lower() in ("", "none", "void", "-1"):
            current = set()
        else:
            current = set()
            for part in value.split(","):
                part = part.strip().lower()
                if part.isdigit():
                    current.add(int(part))
                elif part in uuid_to_index:
                    current.add(uuid_to_index[part])
                else:
                    raise ValueError(f"Unsupported {key} allocation: {value!r}")
        allowed = current if allowed is None else allowed & current
    return allowed


def idle_gpu_ids(report, allowed=None, max_used_mib=256):
    idle = []
    for gpu in report["GPU"]:
        index = int(gpu["Index"])
        if allowed is not None and index not in allowed:
            continue
        used = re.fullmatch(r"(\d+)\s*MiB", gpu["FB Memory Usage"]["Used"])
        utilization = re.fullmatch(r"(\d+)%", gpu["Utilization"]["Gpu"])
        if (
            used
            and utilization
            and int(used[1]) <= max_used_mib
            and int(utilization[1]) == 0
        ):
            idle.append(index)
    return sorted(idle)


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


def main():
    wait_seconds = int(os.environ.get("MUSA_CI_GPU_WAIT_SECONDS", "1800"))
    deadline = time.monotonic() + wait_seconds
    while True:
        result = subprocess.run(
            ["mthreads-gmi", "-q", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads(result.stdout)
        allowed = allowed_gpu_ids(os.environ, report)
        idle = idle_gpu_ids(report, allowed)
        selected = choose_gpu_ids(idle)
        state = ", ".join(
            f"{gpu['Index']}: {gpu['FB Memory Usage']['Used']}, "
            f"util={gpu['Utilization']['Gpu']}"
            for gpu in report["GPU"]
        )
        print(f"MUSA device state: {state}", file=sys.stderr, flush=True)
        if selected is not None:
            print(
                f"Selected idle MUSA devices: {selected}",
                file=sys.stderr,
                flush=True,
            )
            # stdout is consumed by check.sh before torch_musa is imported.
            print(",".join(map(str, selected)))
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"TP4 CI requires four idle allocated GPUs; available: {idle}"
            )
        print("Waiting for four idle MUSA devices...", file=sys.stderr, flush=True)
        time.sleep(min(30, remaining))


if __name__ == "__main__":
    main()
