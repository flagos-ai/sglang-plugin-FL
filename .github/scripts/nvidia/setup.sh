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
printf '%s\n' \
    '#include <cuda_runtime.h>' \
    '__global__ void probe_kernel() {}' \
    'int main() { return 0; }' \
    | "$PIP_CUDA_HOME/bin/nvcc" -x cu -std=c++17 \
        -gencode=arch=compute_80,code=sm_80 -c \
        -o /tmp/sglang-jit-toolchain-probe.o -
test -s /tmp/sglang-jit-toolchain-probe.o
echo "CUDA toolkit compile smoke passed: $PIP_CUDA_HOME"
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
