# SGLang 0.5.18 CI image on the NVIDIA operations runtime (CUDA 13.3).
#
# The operations runtime is immutable: no runtime distribution may be
# upgraded, downgraded, replaced, or removed. This image only ADDS the
# standard dependency closure of the sglang==0.5.18 wheel:
#
#   * the sglang wheel itself is a checksum-pinned artifact (ADD);
#   * its declared closure resolves normally (no --no-deps enumeration
#     lock), constrained by
#       - every runtime distribution, enumerated at build time from
#         `pip freeze` (the resolver cannot move any of them), and
#       - a few top-level version pins whose reasons the constraints
#         file documents (numpy1-era datasets stack, CUDA 13.0 toolchain);
#   * triton is not shipped as a pip distribution: the runtime provides it
#     exclusively as compiler slots (/opt/flagtree 3.6.0 default via
#     PYTHONPATH, /opt/triton 3.7.1 via `compiler triton`). The resolver
#     still installs one transiently (xgrammar/humming-kernels declare a
#     plain `triton` dependency); the same build step removes it, so the
#     final image carries no second triton.
#
# Determinism of the runtime comes from the pinned inputs (digest base,
# static reference data generated from it); the closure is recorded, not
# pre-declared: verify_runtime_immutable.py writes the full additions
# manifest into the image as an audit artifact and enforces zero runtime
# drift at distribution and file level. torch/xgrammar/humming-kernels'
# pip-level `requires triton` findings are waived on record (module-level
# satisfied by the runtime slots).
#
# No sglang-plugin-FL source ships in the image; CI installs the plugin
# from the PR checkout.
#
# Build from the repository root:
#   docker build -f docker/cuda/runtime-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:nv-0.5.18-runtime-<date> .

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

# The closure resolves from pypi.org only. The flagos index also serves
# torch with a local version label (2.13.0+cu130): whenever the resolver
# backtracks, such a candidate re-opens the installed runtime torch and
# pip re-downloads 526 MB for nothing. On pypi.org the installed
# 2.13.0+cu130 is itself the best candidate for torch==2.13.0 (local
# versions sort above the bare release), so backtracking keeps it; the
# post-install freeze gate fails the build loudly if that ever changes.
# Every wheel the sglang 0.5.18 closure needs is published on pypi.org.
# On a proxy outage, re-run the build; the pip HTTP cache and layer cache
# keep the retry cheap.
ARG PIP_INDEX=https://pypi.org/simple

# Static runtime reference data, generated once from the pinned base:
#   runtime-freeze.txt        every runtime distribution (pip freeze --all)
#   site-files.sha256         every runtime site-packages file's SHA256
#   pip-check-baseline.txt    the runtime's own pip check findings
#   provenance.json           vendor module origins + compiler slot versions
COPY docker/cuda/runtime-0.5.18.runtime-freeze.txt \
     docker/cuda/runtime-0.5.18.site-files.sha256 \
     docker/cuda/runtime-0.5.18.pip-check-baseline.txt \
     docker/cuda/runtime-0.5.18.provenance.json \
     /opt/sglang-ci/
COPY docker/cuda/runtime-0.5.18.top-level-constraints.txt /opt/sglang-ci/
COPY docker/cuda/scripts/verify_runtime_immutable.py \
     docker/cuda/scripts/extract_direct_deps.py /opt/sglang-ci/

# The sglang wheel is artifact-pinned by URL + SHA256, independent of
# which index would serve the same version.
ADD --checksum=sha256:fba7bb31a5de4c014945cbc50e9357c9253cc965a423736c324b2aaccd2270dc \
    https://files.pythonhosted.org/packages/48/02/8c15afa2bb3f4d060d4ff7a8cd66d35bfa233f94b41f02a2d59b439dbfde/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
    /opt/sglang-ci/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl

# Step 1: the numpy1-era datasets stack goes in first, --no-deps from the
#    reviewed top-level pins. The runtime's numpy stays 1.26.4 (readonly),
#    every pyarrow >= 21 hard-requires numpy >= 2 at import, and -- crucial
#    for the resolver -- every datasets release that accepts a numpy1-era
#    pyarrow also caps fsspec below the runtime's 2026.9.0. Installing that
#    stack under resolution is therefore impossible against the immutable
#    runtime; the fsspec mismatch is metadata-only (import and
#    sglang.bench_serving verified against the runtime fsspec) and stays
#    waived at the pip-check gate.
RUN --mount=type=cache,target=/tmp/pip-cache \
    python3 -m pip install --no-deps \
        --cache-dir /tmp/pip-cache \
        --index-url "$PIP_INDEX" \
        datasets==4.0.0 pyarrow==18.1.0 dill==0.3.8 multiprocess==0.70.16
# Step 2: the wheel's own METADATA is the top-level basis. Extract its
#    direct dependencies, minus every runtime-provided name (torch &
#    friends): runtime distributions must never re-enter the resolution
#    graph, because a resolver backtrack re-considering a torch edge
#    replaces the runtime's +cu130 build with a public-index variant.
#    Freeze lines with a local version label are filtered out of the
#    dynamic constraints for the same reason (they are satisfied by the
#    installed runtime and absent from public indexes).
# Step 3: install the sglang artifact itself without dependencies, then
#    resolve the extracted direct additions under the dynamic runtime
#    constraints plus the reviewed top-level pins. The pre-installed
#    datasets satisfies its unversioned requirement, so the fsspec
#    conflict that is unresolvable against the immutable runtime never
#    enters the graph.
#    pandas is datasets's own dependency; it resolves here (not in the
#    --no-deps step) so its sub-dependencies come along, while the
#    numpy==1.26.4 runtime constraint keeps the pick compatible.
# Step 4: remove the transient pip triton -- the runtime provides triton
#    exclusively as compiler slots (see header).
RUN --mount=type=cache,target=/tmp/pip-cache \
    python3 -m pip freeze --all | sort > /tmp/runtime.freeze \
    && grep -v '[+]' /tmp/runtime.freeze > /tmp/runtime.constraints \
    && python3 /opt/sglang-ci/extract_direct_deps.py \
        /opt/sglang-ci/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
        /tmp/runtime.freeze /tmp/direct-additions.txt \
    && python3 -m pip install --no-deps \
        --cache-dir /tmp/pip-cache \
        --index-url "$PIP_INDEX" \
        /opt/sglang-ci/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl \
    && python3 -m pip install \
        --cache-dir /tmp/pip-cache \
        --constraint /tmp/runtime.constraints \
        --constraint /opt/sglang-ci/runtime-0.5.18.top-level-constraints.txt \
        --index-url "$PIP_INDEX" \
        -r /tmp/direct-additions.txt \
        pandas \
    && python3 -m pip uninstall -y triton \
    && echo "closure resolved under runtime constraints; transient triton removed"

# Zero-drift gates + additions manifest, all against the committed static
# baseline: distribution freeze, site-packages file hashes, pip check
# delta (documented waivers), and module provenance under container-start
# path order.
RUN PYTHONPATH=/opt/flagtree python3 /opt/sglang-ci/verify_runtime_immutable.py

# The closure owns the CUDA dev toolkit under nvidia/cu13 (constrained to
# the runtime's 13.0 header line); the runtime's /usr/local/cuda stays
# runtime-only. The lib64/libcudart.so symlinks adapt the pip toolkit
# layout to the standard layout FlashInfer's JIT link line expects.
ENV CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13 \
    CUDA_PATH=/flagos/lib/python3.12/site-packages/nvidia/cu13
ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
RUN ln -sfn lib "${CUDA_HOME}/lib64" \
    && ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"

# Build-time GPU-less JIT contract probe: flashinfer's bundled CCCL gates
# on nvcc MAJOR.MINOR == the headers' CUDART_VERSION (cuda_toolkit.h),
# compiles with --compiler-options=-fPIC, and links the .so with
# -L$CUDA_HOME/lib64 -lcudart. The driver's -lcuda is runner-only and
# intentionally skipped. setup.sh repeats this at job time.
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
