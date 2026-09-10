#!/usr/bin/env bash
# Build the flashinfer import-face stub wheel. Pure Python, zero deps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${1:-out}"
python3 generate.py
python3 -m pip wheel . --no-deps -w "$OUT"
ls -l "$OUT"
