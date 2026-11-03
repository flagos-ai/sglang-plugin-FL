#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL from the checkout under test on MetaX C550.
# SGLang 0.5.18 (srt_empty, with docker/metax/sglang-v0.5.18-empty-metax.patch)
# + MetaX torch 2.10 + FlagTree + FlagGems + FlagCX are preinstalled in the CI
# image (see docker/metax/empty-0.5.18.containerfile); this script only reinstalls
# the plugin. --no-deps: pip must not resolve against (and replace) the vendor
# torch / FlagTree builds.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (MetaX) ==="
python -m pip install --no-deps --no-build-isolation -e .
echo "=== Installation complete ==="
python - <<'PY'
import importlib.metadata as m

import sglang_fl
from sglang.srt.utils import is_cuda, is_cuda_alike

from sglang_fl.dispatch.config.utils import get_platform_name

print("sglang", m.version("sglang"), "| sglang_fl loaded from", sglang_fl.__file__)
platform = get_platform_name()
assert platform == "metax", f"expected platform metax, got {platform}"
# MetaX is CUDA-alike but must not take SGLang's NVIDIA-only CUDA paths.
assert not is_cuda() and is_cuda_alike(), (is_cuda(), is_cuda_alike())
print("platform", platform, "| is_cuda", is_cuda(), "| is_cuda_alike", is_cuda_alike())
PY
