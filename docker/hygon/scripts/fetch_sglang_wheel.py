#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fetch the SGLang wheel and verify its SHA256 (build-time integrity gate).

Pure-Python downloader so the build depends only on the runtime's own Python.
"""

import hashlib
import os
import time
import urllib.request
from pathlib import Path

url = os.environ["SGLANG_WHEEL_URL"]
expected = os.environ["SGLANG_WHEEL_SHA256"]
dest = Path("/opt/sglang-ci/wheels") / os.environ["SGLANG_WHEEL_NAME"]
dest.parent.mkdir(parents=True, exist_ok=True)

last_error = None
for attempt in range(1, 6):
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(dest, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        break
    except Exception as error:  # noqa: BLE001 - retry any transport failure
        last_error = error
        print(f"wheel fetch attempt {attempt} failed: {error!r}; retrying", flush=True)
        time.sleep(15)
else:
    raise SystemExit(f"could not fetch {url}: {last_error!r}")

digest = hashlib.sha256(dest.read_bytes()).hexdigest()
if digest != expected:
    raise SystemExit(
        f"SHA256 mismatch for {dest.name}: expected {expected}, got {digest}"
    )
print(f"verified {dest.name} sha256={digest}")
