#!/usr/bin/env bash
# Set up a FlagOS SGLang environment on AMD Radeon GPUs (verified: Radeon AI PRO
# R9700, gfx1201) inside a ROCm PyTorch container, e.g.
#   rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
#
# Usage: bash docker/amd/setup_env.sh <workdir>
#
# Creates <workdir>/venv layered on the container's Python (its ROCm torch and
# triton are reused and pinned, never replaced), then installs SGLang (no deps),
# FlagGems, and this plugin. See docker/amd/README.md for why each step exists.
set -euo pipefail

W=$(mkdir -p "$1" && cd "$1" && pwd)
PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.11}
FLAGGEMS_VERSION=${FLAGGEMS_VERSION:-v5.3.0}

# 1. venv on top of the base interpreter's site-packages (works when the base
#    is itself a venv, where --system-site-packages would skip it).
BASE_SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$PY" -m venv "$W/venv"
VENV_SITE=$("$W/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$BASE_SITE" > "$VENV_SITE/_base_site.pth"
. "$W/venv/bin/activate"
pip install -q -U pip
python - > "$W/constraints.txt" <<'EOF'
import importlib.metadata as m
for p in ("torch", "triton", "numpy"):
    print(f"{p}=={m.version(p)}")
EOF
export PIP_CONSTRAINT="$W/constraints.txt"
echo "pinned: $(tr '\n' ' ' < "$W/constraints.txt")"

# 2. SGLang without its CUDA dependency set, plus the pure-python runtime deps.
pip install -q --no-deps "sglang==${SGLANG_VERSION}"
pip install -q aiohttp anthropic "blobfile==3.0.0" compressed-tensors datasets decord2 easydict einops \
  fastapi gguf hf_transfer huggingface_hub interegular IPython "llguidance>=0.7.11,<0.8.0" modelscope \
  msgspec ninja "openai==2.6.1" "openai-harmony==0.0.4" orjson "outlines==0.1.11" packaging \
  partial_json_parser pillow "prometheus-client>=0.20.0" psutil py-spy pybase64 pydantic \
  python-multipart pyyaml "pyzmq>=25.1.2" requests scipy sentencepiece setproctitle \
  "smg-grpc-proto>=0.3.3" "grpcio>=1.78.0" "grpcio-reflection>=1.78.0" "soundfile==0.13.1" \
  tiktoken "timm==1.0.16" tqdm uvicorn uvloop watchfiles zstandard \
  "transformers==5.6.0" "apache-tvm-ffi>=0.1.5,<0.2"
pip install -q --no-deps "xgrammar==0.1.32"

# 3. aiter and sgl_kernel have no build for this GPU but are imported at module
#    level by SGLang on ROCm: install import-only stubs that raise on use.
mkdir -p "$W/stubs/aiter" "$W/stubs/sgl_kernel"
cp "$PLUGIN_DIR/docker/amd/fl_stub.py" "$W/stubs/_fl_stub.py"
for p in aiter sgl_kernel; do
  printf 'from _fl_stub import install\n\n__getattr__ = install(__name__)\n' > "$W/stubs/$p/__init__.py"
done
echo "$W/stubs" > "$VENV_SITE/fl_stubs.pth"

# 4. FlagGems. Its AMD backend must use Triton's HIP libdevice; older releases
#    fall back to the CUDA one, whose __nv_* calls ROCm Triton rejects.
pip install -q "setuptools<77" "setuptools-scm>=8,<10" scikit-build-core pybind11 cmake wheel \
  "sqlalchemy>=2" packaging pyyaml
[ -d "$W/FlagGems" ] || git clone -q --branch "$FLAGGEMS_VERSION" --depth 1 \
  https://github.com/flagos-ai/FlagGems "$W/FlagGems"
AMD_DESC="$W/FlagGems/src/flag_gems/runtime/backend/_amd/__init__.py"
grep -q 'triton_extra_name="hip"' "$AMD_DESC" ||
  sed -i 's/    device_query_cmd="rocm-smi",/    device_query_cmd="rocm-smi",\n    triton_extra_name="hip",/' "$AMD_DESC"
grep -q 'triton_extra_name="hip"' "$AMD_DESC"
SETUPTOOLS_SCM_PRETEND_VERSION="${FLAGGEMS_VERSION#v}" pip install -q --no-build-isolation --no-deps "$W/FlagGems"

# 5. This plugin.
pip install -q --no-deps -e "$PLUGIN_DIR"

python - <<'EOF'
import torch, flag_gems, sglang, sglang_fl
from flag_gems.utils.triton_lang_helper import tl_extra_shim
assert tl_extra_shim.__name__.endswith("hip.libdevice"), tl_extra_shim.__name__
print(f"OK: torch {torch.__version__}, sglang {sglang.__version__}, flag_gems {flag_gems.__version__}")
EOF
echo "Activate with: . $W/venv/bin/activate"
