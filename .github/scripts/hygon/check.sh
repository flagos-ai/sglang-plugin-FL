#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Match main's HCU device/driver mounts, with the validated TP2 minimum.
set -euo pipefail
test -e /dev/kfd
test -e /dev/mkfd
test -d /dev/dri
test -d /opt/hyhal
python .github/scripts/hygon/ci_environment.py --check-image -- \
  python .github/scripts/hygon/check_devices.py --min-gpus 2
