#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Post-build acceptance for the Hygon two-wheel CI image: run the
# runtime-immutability verifier inside the completed image with the
# committed reference data bind-mounted read-only. This never runs during
# image construction and never modifies the image.
#
# Usage: verify_image.sh <image>            (e.g. hygon-two-wheel:test)

set -euo pipefail

image="${1:?usage: verify_image.sh <image>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
evidence="$(cd "$here/.." && pwd)"

exec docker run --rm \
    --platform linux/amd64 \
    --volume "$evidence":/evidence:ro \
    --entrypoint python \
    "$image" /evidence/scripts/verify_runtime_immutable.py --evidence-dir /evidence
