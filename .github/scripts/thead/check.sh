#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check T-Head PPU availability.
set -euo pipefail
echo "=== Checking T-Head PPU availability ==="
ppu-smi
