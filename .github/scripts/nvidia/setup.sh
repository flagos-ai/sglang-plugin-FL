#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on NVIDIA CUDA.
#
# The CI image (docker/cuda/runtime-0.5.18.containerfile) ships the
# operations runtime plus the locked SGLang closure with the aligned CUDA
# 13.0.88 toolchain and the lib64 layout shim already baked in. Job-time
# work is limited to what must not live in the image: the triton version
# swap, the toolkit activation, the JIT regression probes, and the
# ephemeral test runner + plugin. Triton is not installed by the closure:
# the runtime ships both compiler slots (/opt/flagtree 3.6.0 default via
# PYTHONPATH, /opt/triton 3.7.1 via `compiler triton`); the default
# FlagTree path is what CI runs on.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Activating the standard-closure CUDA toolkit ==="
# The SGLang closure installs a complete pip CUDA toolkit under nvidia/cu13.
# The operations Runtime intentionally keeps /usr/local/cuda runtime-only,
# so point JIT consumers at the closure-owned toolkit as a unit instead of
# mutating the Runtime tree or linking nvcc alone.
PIP_CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$PIP_CUDA_HOME"
export CUDA_PATH="$PIP_CUDA_HOME"
export PATH="$PIP_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PIP_CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# setup.sh and the tests are separate GitHub Actions steps. Persist the
# activation for the later unit/functional/e2e step.
if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "CUDA_HOME=$CUDA_HOME"
        echo "CUDA_PATH=$CUDA_PATH"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    } >> "$GITHUB_ENV"
fi
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PIP_CUDA_HOME/bin" >> "$GITHUB_PATH"
fi

# JIT regression probes, shaped after the three e2e failures this image
# design already fixed: flashinfer's bundled CCCL gates on nvcc
# MAJOR.MINOR == the headers' CUDART_VERSION (cuda_toolkit.h), compiles
# with --compiler-options=-fPIC, and links the .so with
# -L$CUDA_HOME/lib64 -lcudart through the baked shim (the driver's
# -lcuda is runner-only and intentionally skipped).
FI_CCCL=/flagos/lib/python3.12/site-packages/flashinfer/data/cccl
printf '%s\n' \
    '#include <cuda/std/__cccl/cuda_toolkit.h>' \
    '#include <cub/util_type.cuh>' \
    '__global__ void probe_kernel() {}' \
    'int main() { return 0; }' \
    | "$PIP_CUDA_HOME/bin/nvcc" -x cu -std=c++17 \
        -I"$FI_CCCL/cub" -I"$FI_CCCL/libcudacxx/include" -I"$FI_CCCL/thrust" \
        -isystem "$PIP_CUDA_HOME/include" \
        --compiler-options=-fPIC --expt-relaxed-constexpr \
        -gencode=arch=compute_80,code=sm_80 -c \
        -o /tmp/sglang-jit-toolchain-probe.o -
"$PIP_CUDA_HOME/bin/nvcc" -shared -o /tmp/sglang-jit-toolchain-probe.so \
    /tmp/sglang-jit-toolchain-probe.o \
    -L"$PIP_CUDA_HOME/lib64" -lcudart
test -s /tmp/sglang-jit-toolchain-probe.so
rm -f /tmp/sglang-jit-toolchain-probe.o /tmp/sglang-jit-toolchain-probe.so
echo "CUDA JIT compile+link probe passed: $PIP_CUDA_HOME"

echo "=== Installing test runner (ephemeral) ==="
# The operations-runtime CI image intentionally ships no test packages; the
# pinned runner installs into the ephemeral job container only, at the same
# versions the previous image baked in.
python3 -m pip install --break-system-packages \
    pytest==9.1.1 pytest-timeout==2.4.0
echo "=== Installing sglang-plugin-FL (CUDA) ==="
# The operations runtime's Python is externally managed (PEP 668). This is an
# ephemeral CI container, so install only the editable plugin and explicitly
# preserve every pinned dependency.
python3 -m pip install \
    --break-system-packages --no-deps --no-build-isolation -e .
python3 .github/scripts/nvidia/verify_environment.py \
    --require-gpu --require-ci --min-gpus 4
echo "=== Installation complete ==="
python3 -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
