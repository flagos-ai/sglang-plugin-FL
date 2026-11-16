# SGLang 0.5.18 CI image on the NVIDIA operations runtime (CUDA 13.3).
#
# The operations runtime is immutable: no runtime distribution may be
# upgraded, downgraded, replaced, or removed. This image only ADDS the
# version-locked standard dependency closure of the sglang==0.5.18 wheel,
# installed with --no-deps against exact pins.
#
# Determinism comes from the pinned inputs: the digest-pinned base, the
# SHA256-checksummed sglang wheel (ADD --checksum, index-independent), and
# the exact lock with --no-deps. The runtime's own state is also pinned as
# committed reference data -- freeze, pip check output, a file-level
# SHA256 manifest of site-packages, and module provenance -- generated
# once from the pinned base. The gates below verify the install against
# that static truth, so a lock edit that touches a runtime package fails
# the build instead of surfacing as a behavioral CI regression.
#
# Triton is not part of the closure: the runtime already provides both
# compiler slots -- /opt/flagtree (Triton 3.6.0, default via PYTHONPATH)
# and /opt/triton (Triton 3.7.1, selected with `compiler triton`) -- and
# reinstalling either as a pip wheel would duplicate a runtime component.
# torch's Requires-Dist: triton==3.7.1 stays unsatisfied at the pip level
# exactly as in the raw runtime (see the pip-check baseline); the module
# resolves to FlagTree's tree under container-start path order. No
# sglang-plugin-FL source ships in the image; CI installs the plugin from
# the PR checkout.
#
# Build from the repository root:
#   docker build -f docker/cuda/runtime-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:nv-0.5.18-runtime-<date> .

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

# The flagos index keeps priority for internal wheels; the pypi-proxy
# intermittently 504s large transpacific fetches (pip does not retry 504)
# and pypi.org mirrors the same wheels byte-for-byte, so it stays the
# fallback. On a proxy outage, re-run the build; the pip HTTP cache and
# layer cache keep the retry cheap.
ARG PIP_MAIN_INDEX=https://resource.flagos.net/repository/flagos-pypi-nvidia/simple
ARG PIP_FALLBACK_INDEX=https://pypi.org/simple

# Static runtime reference data, generated once from the pinned base:
#   runtime-freeze.txt        every runtime distribution (pip freeze --all)
#   site-files.sha256         every runtime site-packages file's SHA256
#   pip-check-baseline.txt    the runtime's own pip check findings
#   provenance.json           where torch/triton/flag_gems/flagcx/flagtree
#                             resolve under container-start path order
COPY docker/cuda/runtime-0.5.18.runtime-freeze.txt \
     docker/cuda/runtime-0.5.18.site-files.sha256 \
     docker/cuda/runtime-0.5.18.pip-check-baseline.txt \
     docker/cuda/runtime-0.5.18.provenance.json \
     /opt/sglang-ci/
COPY docker/cuda/runtime-0.5.18.install.lock /opt/sglang-ci/
COPY docker/cuda/scripts/verify_runtime_immutable.py /opt/sglang-ci/

# The sglang wheel is artifact-pinned by URL + SHA256, independent of
# which index would serve the same version.
ADD --checksum=sha256:fba7bb31a5de4c014945cbc50e9357c9253cc965a423736c324b2aaccd2270dc \
    https://files.pythonhosted.org/packages/48/02/8c15afa2bb3f4d060d4ff7a8cd66d35bfa233f94b41f02a2d59b439dbfde/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
    /opt/sglang-ci/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl

RUN --mount=type=cache,target=/tmp/pip-cache \
    python3 -m pip install --no-deps \
        --cache-dir /tmp/pip-cache \
        --index-url "$PIP_MAIN_INDEX" \
        --extra-index-url "$PIP_FALLBACK_INDEX" \
        -r /opt/sglang-ci/runtime-0.5.18.install.lock \
        /opt/sglang-ci/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl

# All five immutability gates in one pass, against the committed static
# baseline: lock overlap, distribution freeze, site-packages file hashes,
# pip check delta (documented waiver), and module provenance.
RUN PYTHONPATH=/opt/flagtree python3 /opt/sglang-ci/verify_runtime_immutable.py

# The closure owns the CUDA dev toolkit (13.0.88, matching the runtime's
# CUDA 13.0 headers) under nvidia/cu13; the runtime's /usr/local/cuda stays
# runtime-only. The lib64/libcudart.so symlinks adapt the pip toolkit
# layout to the standard layout FlashInfer's JIT link line expects.
ENV CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13 \
    CUDA_PATH=/flagos/lib/python3.12/site-packages/nvidia/cu13
ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
RUN ln -sfn lib "${CUDA_HOME}/lib64" \
    && ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"

# Build-time JIT contract probe (GPU-less): the same compile shape whose
# failures killed three e2e rounds -- flashinfer's bundled CCCL gates on
# nvcc MAJOR.MINOR == the headers' CUDART_VERSION (cuda_toolkit.h),
# compiles with --compiler-options=-fPIC, and links the .so with
# -L$CUDA_HOME/lib64 -lcudart. The driver's -lcuda is runner-only and
# intentionally skipped. setup.sh repeats this at job time as a
# regression check against image drift.
RUN FI_CCCL=/flagos/lib/python3.12/site-packages/flashinfer/data/cccl \
    && printf '%s\n' \
         '#include <cuda/std/__cccl/cuda_toolkit.h>' \
         '#include <cub/util_type.cuh>' \
         '__global__ void probe_kernel() {}' \
         'int main() { return 0; }' \
       | "${CUDA_HOME}/bin/nvcc" -x cu -std=c++17 \
           -I"$FI_CCCL/cub" -I"$FI_CCCL/libcudacxx/include" -I"$FI_CCCL/thrust" \
           -isystem "${CUDA_HOME}/include" \
           --compiler-options=-fPIC --expt-relaxed-constexpr \
           -gencode=arch=compute_80,code=sm_80 -c \
           -o /tmp/sglang-jit-toolchain-probe.o - \
    && "${CUDA_HOME}/bin/nvcc" -shared -o /tmp/sglang-jit-toolchain-probe.so \
           /tmp/sglang-jit-toolchain-probe.o \
           -L"${CUDA_HOME}/lib64" -lcudart \
    && test -s /tmp/sglang-jit-toolchain-probe.so \
    && rm -f /tmp/sglang-jit-toolchain-probe.o /tmp/sglang-jit-toolchain-probe.so
