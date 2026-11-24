# NVIDIA SGLang 0.5.18 CI image.
#
# The build context must stage the two reviewed wheel artifacts under
# artifacts/ plus the one-line resolver exclusion list. All dependency
# resolution comes from their METADATA; tests and validation run against
# the completed image, never during image construction.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

COPY artifacts/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
     /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl \
     /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl
COPY artifacts/runtime-0.5.18.excludes.txt \
     /tmp/runtime-0.5.18.excludes.txt

# Normal resolver install of both wheels with the Runtime's own uv. The
# excludes file is the single deliberate platform exception: the Runtime
# provides the Triton compiler as the /opt/flagtree module tree, so the
# standalone triton distribution is dropped from resolution (nothing is
# installed, uninstalled, or stubbed for it). /flagos is a uv-managed
# virtual environment, so no system-package override is needed.
RUN /usr/local/bin/uv pip install \
      --python /flagos/bin/python3 \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      /tmp/wheels/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[nvidia]'

# The `[nvidia]` extra above pulls `cuda-toolkit[nvcc]`, whose wheels install
# under `site-packages/nvidia/cu13` -- a location no consumer treats as a CUDA
# Toolkit root. The Runtime base image ships a cudart-only `/usr/local/cuda`
# with no `bin/`, so torch's `cpp_extension`, SGLang's JIT toolchain, and
# FlashInfer all fall back to it and fail with `nvcc: not found` when they
# compile a kernel. FlashInfer compiles at serve time, so the compiler is a
# runtime dependency, not a build-time one.
#
# The plugin activates a compatibility view itself at import time, but that is
# a process-level side effect: it does not reach a process that never imports
# `sglang_fl`, and `torch.utils.cpp_extension.CUDA_HOME` caches its value at
# torch import. Make the view an environment fact instead, so every consumer
# (including subprocesses and `docker exec`) resolves the same compiler. The
# symlinks are also what the plugin's own `_valid_cuda_home()` requires to
# accept the directory; with them in place it returns early and adds nothing.
ENV CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13 \
    CUDA_PATH=/flagos/lib/python3.12/site-packages/nvidia/cu13
RUN ln -sfn lib "${CUDA_HOME}/lib64" \
 && ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
