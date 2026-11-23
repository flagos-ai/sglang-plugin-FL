# SGLang 0.5.18 two-wheel CI image on the Hygon operations runtime (DTK 26.04).
#
# This recipe is runtime-image ASSEMBLY only: the operations Runtime base, one
# uv resolver install of the two real application wheels, and the HCU/DTK
# compatibility layer. Every acceptance gate - the resolver dry-run plan,
# runtime zero-drift, site-packages hashes, the pip-check diff, the wheel
# contract, plugin-guard/direct-clang/import smoke and Triton provenance - runs
# OUTSIDE this recipe: docker/hygon/build_two_wheel.sh stages the inputs and
# plan-gates the build, docker/hygon/scripts/verify_image.sh accepts the
# finished image against reference data collected dynamically from the same
# base. The compatibility blocks below carry one-line comments here; their
# evidence and removal conditions are tabulated in docker/hygon/README.md.
#
# Principles (do not weaken without review):
#   * The operations runtime is immutable: no runtime distribution may be
#     upgraded, downgraded, replaced, or removed; the resolver may only ADD
#     the SGLang dependency closure. Enforcement is post-build (the
#     zero-drift gates in verify_image.sh); a failed verification blocks
#     the push instead of failing the build.
#   * Hygon installs sglang[srt_empty,runtime_common] + sglang_fl[hygon]:
#     the dependency ownership-inversion extras. The DTK torch/FlagTree/
#     FlagGems/FlagCX stack comes from the runtime, never from this build.
#   * Build inputs are read-only BuildKit bind mounts for the RUN that consumes
#     them; they never enter a final-image layer. The released image carries no
#     CI working directory, resolver cache, wheel archive, or build evidence.
#
# Build + accept (single release entry; acceptance is the gate, --push is
# reachable only after it passes):
#   docker/hygon/build_two_wheel.sh [--push] <tag>

# syntax=docker/dockerfile:1.7
# Moving semver tag per the Runtime owner (heavy-testing phase re-publishes
# without notice; digest pinning broke when pruned digests disappeared).
# build_two_wheel.sh resolves the tag to its repo digest, builds from the
# digest and records it as the org.flagos.ci.base-digest OCI label.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04:2.3.0
FROM ${BASE_IMAGE}

# TEMP: the Runtime exposes /opt/flagtree only to bash sessions (BASH_ENV),
# so direct python entrypoints get no triton; image-level PYTHONPATH fixes it.
# Evidence: base probe - find_spec("triton") is None without a bash profile.
# Owner: FlagOS Runtime (remove once /opt/flagtree is exposed OCI-wide).
# This is also the Triton compiler slot selection: build_two_wheel.sh records
# it as org.flagos.ci.triton-slot and verify_image.sh holds each slot to its
# own contract (docker/hygon/README.md).
ENV PYTHONPATH=/opt/flagtree

# Index priority with uv's real semantics. --index-strategy
# unsafe-first-match exhausts the FIRST index carrying a package NAME before
# consulting later indexes: the flagos platform index (flagos-built wheels)
# wins whenever it has a compatible version, the Tencent PyPI mirror follows
# for generic wheels (pypi.org is unreachable from the build network), and
# the pypi-proxy is the --default-index last resort. first-index was
# measured insufficient: the Hygon index carries xgrammar but not
# xgrammar==0.2.1 and stops at the name instead of falling through.
ARG FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
ARG TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple
ARG PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

# Runtime constraints are consumed by the install RUN below.
# runtime-constraints.txt freezes the pristine Runtime inventory (canonical
# name==version) and is generated from the resolved base by
# build_two_wheel.sh - never committed. Product requirements, including the
# datasets compatibility pin, are carried by the reviewed wheel metadata.

# The generated constraints, two reviewed application wheels, corrected
# xgrammar dependency wheel and the current single-item resolver exclusion are
# mounted read-only from the staging artifacts directory for the install RUN;
# none enters an image layer. Their SHA256s are part of the staging
# reproducibility record. The SGLang wheel is the DERIVED HCU
# wheel: the exact published flagos wheel with the reviewed HCU gating patch
# folded in before staging (tools/wheels/hygon/derive_sglang_wheel.py;
# strict diff gates prove only the patch's files, the METADATA Version line,
# the dist-info directory name and RECORD differ from the pinned input).
# The image therefore never patches site-packages.
# --- One uv resolver install of the two application wheels -------------------
# CONTRACT: no standalone triton distribution enters the image; the Runtime's
# /opt/flagtree tree (Triton 3.6) is the only compiler. The explicit xgrammar
# local-version wheel removes its incorrect HCU triton requirement. --excludes
# remains temporarily until the same real-index plan and built image pass the
# removal gates; it is deliberately not treated as proof that the correction
# works against the currently unavailable FlagOS indexes.
# Evidence: run 44465620210 - full Hygon hardware matrix green, no triton
# distribution installed, triton.__file__ under /opt/flagtree.
# Owner: xgrammar artifact/build tooling (remove after the corrected wheel's
# real-index no-excludes plan, candidate build and full acceptance pass).
# build_two_wheel.sh runs the exact twin of this command (plus --dry-run)
# against the resolved base digest and plan-validates it BEFORE the build.
RUN --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --index "$FLAGOS_INDEX" \
      --index "$TENCENT_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
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
