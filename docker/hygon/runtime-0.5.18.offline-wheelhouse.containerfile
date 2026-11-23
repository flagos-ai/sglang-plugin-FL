# SGLang 0.5.18 two-wheel DEVELOPMENT CANDIDATE image on the Hygon operations
# runtime, resolved OFFLINE from a wheelhouse.
#
# This is the TEMP recipe of the offline-wheelhouse development candidate
# (docker/hygon/README.md, "Offline-wheelhouse development candidate"). It
# covers the periods when the FlagOS package indexes are unreachable: the
# resolver closure comes from a wheelhouse that docker/hygon/
# build_wheelhouse.sh derives from the resolver cache of a last-green
# two-wheel image, and the install RUN has no network at all.
#
# It is NOT a release recipe. build_two_wheel.sh refuses it for --push; only
# --push-candidate publishes the result, labeled release-ready=false. The
# release recipe stays docker/hygon/runtime-0.5.18.containerfile, and the
# contract tests hold the two recipes identical outside the resolver RUN
# (same base argument, same compiler slot, same three wheels, same
# constraints and exclusion, same Runtime compatibility bundle).
#
# No `# syntax=` directive on purpose: the pinned frontend image lives on
# docker.io, which an offline build must not depend on. Every instruction
# below is supported by the builder's built-in Dockerfile frontend.
#
# Build + accept (single candidate entry; acceptance is the gate,
# --push-candidate is reachable only after it passes):
#   docker/hygon/build_two_wheel.sh --offline-wheelhouse <wheelhouse> \
#       --runtime-image <name@sha256:digest> [--push-candidate] <tag>
#
# The Runtime base is always passed in as an explicitly verified digest
# (--runtime-image): this recipe carries no default and never falls back to
# a moving tag.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# TEMP: the Runtime exposes /opt/flagtree only to bash sessions (BASH_ENV),
# so direct python entrypoints get no triton; image-level PYTHONPATH fixes it.
# Evidence: base probe - find_spec("triton") is None without a bash profile.
# Owner: FlagOS Runtime (remove once /opt/flagtree is exposed OCI-wide).
# This is also the Triton compiler slot selection: build_two_wheel.sh records
# it as org.flagos.ci.triton-slot and verify_image.sh holds each slot to its
# own contract (docker/hygon/README.md).
ENV PYTHONPATH=/opt/flagtree

# --- One uv resolver install of the two application wheels, offline --------
# CONTRACT (the same as the release recipe): no standalone triton distribution
# enters the image; the Runtime's /opt/flagtree tree (Triton 3.6) is the only
# compiler. The explicit xgrammar local-version wheel removes its incorrect
# HCU triton requirement; --excludes remains temporarily, exactly as in the
# release recipe.
#
# The closure is resolved ONLY from the wheelhouse (--no-index
# --find-links): a flat directory of the upstream wheels a last-green
# two-wheel image downloaded, each bound to the digest its registry declared
# (wheelhouse-manifest.json). Runtime distributions come from the base and
# are frozen by the generated constraints; nothing is downloaded, so the RUN
# needs no network and says so. The wheelhouse, constraints and wheels are
# read-only bind mounts from the build context and never enter an image
# layer.
# build_two_wheel.sh runs the exact twin of this command (plus --dry-run)
# with the same mounts and network mode BEFORE the build
# (uv_install_contract.py reads it from this file).
RUN --network=none \
    --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    --mount=type=bind,source=wheelhouse,target=/wheelhouse,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --no-index \
      --find-links /wheelhouse \
      --constraints /tmp/artifacts/runtime-constraints.txt \
      --excludes /tmp/artifacts/runtime-0.5.18.excludes.txt \
      '/tmp/artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
      '/tmp/artifacts/sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[hygon]'

# --- TEMP Runtime compatibility bundle ---------------------------------------
# TEMP Runtime compatibility bundle；这些设置暴露 Runtime 已携带但没有对所有入口
# 稳定提供的能力；等待 Runtime owner 明确迁移方案；详细 observed failure、owner
# 和 removal gate 见 docker/hygon/README.md。
# 1/3 DTK loader path, prepended to whatever the base carries.
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 2/3 FlagTree derives its clang selection from this root.
ENV ROCM_PATH=/opt/dtk

# 3/3 The ROCm prefix alias SGLang's HIP JIT builder expects.
RUN if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then ln -s /opt/dtk /opt/rocm; fi \
    && test "$(readlink -f /opt/rocm)" = "$(readlink -f /opt/dtk)"
