#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Match main's HCU device/driver mounts, with the validated TP2 minimum.
set -euo pipefail
test -e /dev/kfd
test -e /dev/mkfd
test -d /dev/dri
test -d /opt/hyhal

# CI containers see every GPU of the node, not only this pod's allocation, so
# a busy first-visible card does not mean the job cannot run. Inventory all
# visible cards, pin the two most idle ones via HIP_VISIBLE_DEVICES (the
# remedy check_devices.py itself suggests) and persist it for the later steps
# of this job. On an otherwise clean node this resolves to 0,1 and the
# behavior is unchanged; the guard below still validates the picked pair.
PICK=$(python3 - <<'PY'
import sys

import torch

n = torch.cuda.device_count()
idle = []
for index in range(n):
    try:
        free, total = torch.cuda.mem_get_info(index)
    except Exception as error:
        print(f"[gpu-pick] visible {index}: query failed ({error})", file=sys.stderr)
        continue
    fraction = free / total if total else 0.0
    print(f"[gpu-pick] visible {index}: {fraction:.0%} free", file=sys.stderr)
    if fraction >= 0.95:
        idle.append(index)
print(",".join(str(i) for i in idle[:2]))
PY
) || PICK=""
case "$PICK" in
  *,*)
    echo "[gpu-pick] pinning HIP_VISIBLE_DEVICES=$PICK"
    export HIP_VISIBLE_DEVICES="$PICK"
    if [ -n "${GITHUB_ENV:-}" ]; then
      printf 'HIP_VISIBLE_DEVICES=%s\n' "$PICK" >> "$GITHUB_ENV"
    fi
    ;;
  "")
    echo "[gpu-pick] WARNING: no idle GPU pair found; letting the guard report" >&2
    ;;
  *)
    echo "[gpu-pick] WARNING: only one idle GPU; not pinning" >&2
    ;;
esac

python .github/scripts/hygon/ci_environment.py --check-image -- \
  python .github/scripts/hygon/check_devices.py --min-gpus 2
