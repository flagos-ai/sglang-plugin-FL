#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on NVIDIA CUDA.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Minimal fast-path: swap triton only ==="
# TEMPORARY: the only functional change from the full-closure image is the
# triton version — 3.7.1 (from the standard closure) crashes kernel launches
# (SystemError in triton/backends/nvidia/driver.py). FlagTree provides 3.6.0
# at /opt/flagtree; installing the same version as a pip wheel prevents the
# unit-test sys.modules contamination while matching FlagTree's ABI.
# Everything else (CUDA toolchain, kernel stacks, all packages) stays.
python3 -m pip install --no-deps triton==3.6.0
echo "triton swapped: 3.7.1 -> 3.6.0 (FlagTree-matching version)"

echo "=== Aligning the closure CUDA toolchain to the runtime's CTK headers ==="
# The operations runtime preinstalls nvidia-cuda-runtime 13.0.96, whose
# headers own nvidia/cu13/include (CUDART_VERSION 13000). The standard
# closure added nvcc 13.5.37, and flashinfer's bundled CCCL compares the
# nvcc MAJOR.MINOR against the headers' (cuda_toolkit.h:41): 13.5 vs 13.0
# aborts every JIT compile (PR #179 e2e run 43347047055). The runtime's
# own distributions are readonly, so align the compiler to the 13.0 line
# instead (13.0.88 matches the runtime's nvrtc/cupti line).
python3 -m pip install --no-deps \
    nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88
echo "CUDA toolchain aligned: nvcc/nvvm/crt 13.5.37 -> 13.0.88"

# Standard CUDA layout shim for the JIT link step: flashinfer's ninja link
# line assumes $CUDA_HOME/lib64 plus an unversioned libcudart.so, while the
# pip cu13 tree uses lib/ and ships only the versioned libcudart.so.13
# (run 43389681469: "/usr/bin/ld: cannot find -lcudart"). Purely additive
# symlinks; no runtime-owned file is touched.
PIP_CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13
ln -sfn lib "$PIP_CUDA_HOME/lib64"
ln -sfn libcudart.so.13 "$PIP_CUDA_HOME/lib/libcudart.so"
test -e "$PIP_CUDA_HOME/lib64/libcudart.so" && echo "cu13 lib64 shim in place"

echo "=== Activating the standard-closure CUDA toolkit ==="
# The normal SGLang dependency closure installs a complete pip CUDA toolkit
# under nvidia/cu13.  The operations Runtime intentionally keeps
# /usr/local/cuda runtime-only, so point JIT consumers at the closure-owned
# toolkit as a unit instead of mutating the Runtime tree or linking nvcc alone.
PIP_CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13
for required in \
    "$PIP_CUDA_HOME/bin/nvcc" \
    "$PIP_CUDA_HOME/include/cuda_runtime.h" \
    "$PIP_CUDA_HOME/include/cuda.h"; do
    if [ ! -e "$required" ]; then
        echo "required CUDA toolkit artifact is missing: $required" >&2
        exit 1
    fi
done
export CUDA_HOME="$PIP_CUDA_HOME"
export CUDA_PATH="$PIP_CUDA_HOME"
export PATH="$PIP_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PIP_CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# setup.sh and the tests are separate GitHub Actions steps. Persist the
# activation for the later unit/functional/e2e step without editing files in
# the Runtime-owned /usr/local/cuda tree.
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

"$PIP_CUDA_HOME/bin/nvcc" --version
# Probe with the exact compile shape that failed in e2e: flashinfer's
# bundled CCCL gates on nvcc MAJOR.MINOR == the headers' CUDART_VERSION
# (cuda_toolkit.h:41). A plain cuda_runtime.h compile passes even when
# that gate would abort, so exercise the CCCL path directly.
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
test -s /tmp/sglang-jit-toolchain-probe.o
echo "CUDA toolkit CCCL-shaped compile smoke passed: $PIP_CUDA_HOME"
# Link smoke: flashinfer's final ninja step links the .so with
# -L$CUDA_HOME/lib64 -lcudart (run 43389681469 ld failure). Probe the same
# resolution; the driver's -lcuda is runner-only and intentionally skipped.
"$PIP_CUDA_HOME/bin/nvcc" -shared -o /tmp/sglang-jit-toolchain-probe.so \
    /tmp/sglang-jit-toolchain-probe.o \
    -L"$PIP_CUDA_HOME/lib64" -lcudart
test -s /tmp/sglang-jit-toolchain-probe.so
echo "CUDA toolkit link smoke passed (-lcudart via lib64 shim)"
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
