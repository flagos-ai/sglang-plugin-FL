#!/bin/bash
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
python3 -m pip install -e . --no-deps --no-build-isolation
python3 - <<'EOF'
from importlib.metadata import version
from pathlib import Path
import torch_gcu
import flag_gems
import flagcx
import sgl_kernel
from flash_attn.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
import sglang_fl
from sglang_fl.dispatch.config.utils import get_platform_name
v = version("sglang")
if v.split("+", 1)[0] != "0.5.18":
    raise RuntimeError(f"Expected SGLang 0.5.18, found {v}")
p = Path(sglang_fl.__file__).resolve()
if not p.is_relative_to(Path.cwd().resolve()):
    raise RuntimeError(f"Plugin outside checkout: {p}")
if get_platform_name() != "gcu":
    raise RuntimeError("Expected gcu platform")
for name in ("sglang", "flag_gems", "flagcx"):
    print(name, version(name))
print("Plugin:", p)
EOF
