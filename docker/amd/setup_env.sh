#!/usr/bin/env bash
# Set up a FlagOS SGLang environment on AMD Radeon GPUs (verified: Radeon AI PRO
# R9700 gfx1201, Radeon Pro W7900 gfx1100) inside a ROCm PyTorch container, e.g.
#   rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
#
# Usage: bash docker/amd/setup_env.sh <workdir>
#   FLAGTREE_SRC=/path/to/FlagTree  build FlagTree from source (or install a built
#                                   flagtree .whl) instead of the published wheel
#
# Creates <workdir>/venv layered on the container's Python (its ROCm torch is
# reused and pinned, never replaced), then installs FlagTree, SGLang (no deps),
# FlagGems, and this plugin. See docker/amd/README.md for why each step exists.
set -euo pipefail

W=$(mkdir -p "$1" && cd "$1" && pwd)
PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.11}
FLAGGEMS_VERSION=${FLAGGEMS_VERSION:-v5.3.0}
FLAGTREE_VERSION=${FLAGTREE_VERSION:-0.7.0+amd3.6}
FLAGTREE_INDEX=${FLAGTREE_INDEX:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}

# 1. venv on top of the base interpreter's site-packages (works when the base
#    is itself a venv, where --system-site-packages would skip it).
BASE_SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$PY" -m venv "$W/venv"
VENV_SITE=$("$W/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$BASE_SITE" > "$VENV_SITE/_base_site.pth"
. "$W/venv/bin/activate"
pip install -q -U pip

# 2. FlagTree, FlagOS's Triton (AMD backend). It provides the `triton` package
#    in the venv, shadowing any ROCm Triton the container ships.
#    torch's metadata still requires its own ROCm Triton build, so pip needs that
#    distribution visible to resolve packages that depend on torch. Some ROCm
#    images ship torch without it: install it into a directory placed after the
#    venv on sys.path, so FlagTree's `triton` still wins on import.
if ! "$PY" -c "import triton" 2>/dev/null; then
  TRITON_REQ=$(python -c 'import importlib.metadata as m; print(next(r.split(";")[0].strip() for r in m.requires("torch") if r.startswith("triton")))')
  ROCM_REL=$(echo "$TRITON_REQ" | sed -n 's/.*+rocm\([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p')
  pip install -q --no-deps --target "$W/rocm_triton" "$TRITON_REQ" \
    -f "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_REL}/"
  echo "$W/rocm_triton" > "$VENV_SITE/zz_rocm_triton.pth"
fi
if [ -n "${FLAGTREE_SRC:-}" ]; then
  # A FlagTree source checkout (built here) or an already-built flagtree wheel.
  if [ -d "$FLAGTREE_SRC" ]; then
    pip install -q -r "$FLAGTREE_SRC/python/requirements.txt"
  fi
  pip install --no-build-isolation "$FLAGTREE_SRC"
else
  pip install -q --no-deps "flagtree===${FLAGTREE_VERSION}" --index-url "$FLAGTREE_INDEX"
fi

python - > "$W/constraints.txt" <<'EOF'
import importlib.metadata as m
for p in ("torch", "flagtree", "numpy"):
    print(f"{p}=={m.version(p)}")
EOF
export PIP_CONSTRAINT="$W/constraints.txt"
echo "pinned: $(tr '\n' ' ' < "$W/constraints.txt")"

# 3. SGLang without its CUDA dependency set, plus the pure-python runtime deps.
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

# 4. aiter and sgl_kernel have no build for this GPU but are imported at module
#    level by SGLang on ROCm: install import-only stubs that raise on use.
mkdir -p "$W/stubs/aiter" "$W/stubs/sgl_kernel"
cp "$PLUGIN_DIR/docker/amd/fl_stub.py" "$W/stubs/_fl_stub.py"
for p in aiter sgl_kernel; do
  printf 'from _fl_stub import install\n\n__getattr__ = install(__name__)\n' > "$W/stubs/$p/__init__.py"
done
echo "$W/stubs" > "$VENV_SITE/fl_stubs.pth"

# 5. FlagGems. Its AMD backend must use Triton's HIP libdevice (flagos-ai/FlagGems#7908);
#    without it, libdevice math ops emit CUDA __nv_* calls the AMD backend rejects.
pip install -q "setuptools<77" "setuptools-scm>=8,<10" scikit-build-core pybind11 cmake wheel \
  "sqlalchemy>=2" packaging pyyaml
[ -d "$W/FlagGems" ] || git clone -q --branch "$FLAGGEMS_VERSION" --depth 1 \
  https://github.com/flagos-ai/FlagGems "$W/FlagGems"
AMD_DESC="$W/FlagGems/src/flag_gems/runtime/backend/_amd/__init__.py"
grep -q 'triton_extra_name="hip"' "$AMD_DESC" ||
  sed -i 's/    device_query_cmd="rocm-smi",/    device_query_cmd="rocm-smi",\n    triton_extra_name="hip",/' "$AMD_DESC"
grep -q 'triton_extra_name="hip"' "$AMD_DESC"
SETUPTOOLS_SCM_PRETEND_VERSION="${FLAGGEMS_VERSION#v}" pip install -q --no-build-isolation --no-deps "$W/FlagGems"

# 6. This plugin.
pip install -q --no-deps -e "$PLUGIN_DIR"

python - "$VENV_SITE" <<'EOF'
import importlib.metadata as m, sys
import torch, triton, flag_gems, sglang, sglang_fl
from flag_gems.utils.triton_lang_helper import tl_extra_shim
assert triton.__path__[0].startswith(sys.argv[1]), f"triton is not FlagTree's: {triton.__path__[0]}"
assert tl_extra_shim.__name__.endswith("hip.libdevice"), tl_extra_shim.__name__
print(f"OK: torch {torch.__version__}, flagtree {m.version('flagtree')} (triton {triton.__version__}), "
      f"sglang {sglang.__version__}, flag_gems {flag_gems.__version__}")
EOF
echo "Activate with: . $W/venv/bin/activate"
