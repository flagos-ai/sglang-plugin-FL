#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Check TsingMicro TXDA availability and reserve one healthy idle 8-card group.
#
# Card protection (see select_cards.py): occupancy via tsm_smi, an exclusion
# list maintained by the runner host (dmesg is not readable in the container),
# and a per-card compute probe that rejects cards left broken by an earlier
# job. The chosen cards are exported as TXDA_VISIBLE_DEVICES for later steps.
set -euo pipefail
echo "=== Checking TsingMicro TXDA availability ==="
for dev in /dev/accel /dev/accel_drv_mgr; do
  test -e "$dev" || { echo "::error::$dev missing: pass --device $dev to the container" >&2; exit 1; }
done
command -v tsm_smi >/dev/null || export PATH="/usr/local/kuiper/bin:$PATH"
# (head closes the pipe early; do not let SIGPIPE fail the step under pipefail)
tsm_smi | head -6 || true

TXDA_VISIBLE_DEVICES=$(python3 "$(dirname "$0")/select_cards.py")
export TXDA_VISIBLE_DEVICES
echo "Using TXDA_VISIBLE_DEVICES=$TXDA_VISIBLE_DEVICES"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'TXDA_VISIBLE_DEVICES=%s\n' "$TXDA_VISIBLE_DEVICES" >> "$GITHUB_ENV"
fi
