# SGLang 0.5.18 two-wheel CI image on the Hygon operations runtime (DTK 26.04).
#
# This recipe is runtime-image ASSEMBLY only: the digest-pinned operations
# Runtime, one uv resolver install of the two real application wheels, and
# the HCU/DTK compatibility layer. Every acceptance gate - the resolver
# dry-run plan, runtime zero-drift, site-packages hashes, the pip-check
# diff, the wheel contract, the stub/wrapper/import smoke and the Triton
# provenance - runs OUTSIDE this recipe: docker/hygon/build_two_wheel.sh
# stages the inputs and plan-gates the build, docker/hygon/scripts/
# verify_image.sh accepts the finished image against reference data
# collected dynamically from the same pinned base digest. Baselines and
# audit records live in mktemp/CI artifacts only; none is committed to the
# repository and none is baked into the image.
#
# Principles (do not weaken without review):
#   * The operations runtime is immutable: no runtime distribution may be
#     upgraded, downgraded, replaced, or removed; the resolver may only ADD
#     the SGLang dependency closure. Enforcement is post-build (the
#     zero-drift gates in verify_image.sh); a failed verification blocks
#     the push instead of failing the build.
#   * Hygon installs sglang[srt_empty,runtime_common] + sglang_fl[hygon]:
#     the dependency ownership-inversion extras. The DTK torch/FlagTree/
#     FlagGems/FlagCX stack comes from the runtime, never from this build;
#     runtime_common only adds compressed-tensors (unpinned) because
#     sglang.srt's quantization registry imports it eagerly at server
#     start while the wheel declares it under that extra.
#   * The image ships the two real application wheels (sglang 0.5.18+flagos
#     and sglang_fl built from this repository); CI installs only the test
#     runner and validates against the installed wheels - the checkout
#     never shadows them (SGLANG_FL_TEST_SOURCE=wheel).
#
# Build + accept (single release entry; acceptance is the gate, --push is
# reachable only after it passes):
#   docker/hygon/build_two_wheel.sh [--push] <tag>
# The image carries its provenance as OCI labels
# (org.flagos.ci.base-digest / org.flagos.ci.source-commit); the
# acceptance step reads the base digest from the image itself, not from
# the checkout.

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04@sha256:03e598b348e58e7233628394bbefa0ccef26969a091c7e6f2f02835161757f8b
FROM ${BASE_IMAGE}

# CONTRACT: every entrypoint must resolve the Runtime's Triton.
# The Runtime ships /opt/flagtree and exposes it only to bash sessions
# (BASH_ENV -> /etc/profile.d/vendor.sh); a direct python entrypoint gets
# find_spec("triton") is None. Image-level PYTHONPATH closes that
# direct-entrypoint gap.
# Evidence: direct-entrypoint contract failure probe on the base image
# (find_spec("triton") is None without a bash profile).
# Removal condition: the Runtime exposes /opt/flagtree through OCI ENV,
# a .pth file, or another mechanism effective for all entrypoints.
ENV PYTHONPATH=/opt/flagtree

# Index priority with uv's real semantics. --index-strategy
# unsafe-first-match exhausts the FIRST index carrying a package NAME
# before consulting later indexes: the flagos platform index (flagos-built
# wheels) wins whenever it has a compatible version, PyPI follows for
# generic wheels, and the pypi-proxy is the --default-index last resort
# (slow transpacific delivery, intermittent 504s). first-index was
# measured insufficient on the real runtime: the Hygon index carries
# xgrammar but not xgrammar==0.2.1, and first-index stops at the name
# instead of falling through to PyPI ("No solution found ...").
ARG FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
ARG PYPI_INDEX=https://pypi.org/simple
ARG PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

WORKDIR /opt/sglang-ci

# Resolver inputs. top-level-constraints.txt is the reviewed, hand-curated
# pin file (see its header; only evidence-backed pins). runtime-constraints.txt
# freezes the pristine Runtime inventory (canonical name==version) and is
# generated from the pinned base digest by build_two_wheel.sh - never
# committed; it makes every runtime distribution unpipinstallable for the
# resolver below.
COPY docker/hygon/top-level-constraints.txt /opt/sglang-ci/top-level-constraints.txt
COPY artifacts/runtime-constraints.txt /opt/sglang-ci/runtime-constraints.txt

# The two reviewed application wheels and the single-item resolver
# exclusion list are staged by docker/hygon/stage_wheels.sh; their SHA256s
# are part of the staging reproducibility record.
COPY artifacts/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl /tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl
COPY artifacts/runtime-0.5.18.excludes.txt /tmp/runtime-0.5.18.excludes.txt

# --- Two-wheel install with the Runtime's own uv -----------------------------
# Both application wheels resolve together in one graph:
#   sglang[srt_empty,runtime_common] + sglang_fl[hygon].
# build_two_wheel.sh runs the exact twin of this command (plus --dry-run)
# against the pinned base digest and plan-validates it BEFORE the build.
#
# CONTRACT: no standalone triton distribution enters the image. The
# Runtime's real compiler module is the /opt/flagtree tree (Triton 3.6,
# which the runtime's FlagGems 5.4.0 build targets); xgrammar's
# distribution metadata would otherwise pull standalone triton into the
# resolver and create two competing Triton sources. --excludes drops it
# before resolution - nothing is installed, uninstalled, or stubbed for
# it; the metadata gap xgrammar still declares is the pip-check gate's
# documented platform capability exception.
# Evidence: run 44465620210 - the full Hygon hardware matrix green on this
# exact combination (no triton distribution installed; triton.__file__
# under /opt/flagtree). This is a resolver/provenance contract verified
# post-build, not a functional-must-fail claim.
# Removal condition: the Runtime ships a triton distribution consistent
# with Python distribution metadata, or upstream dependencies stop
# declaring standalone triton.
RUN /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --index "$FLAGOS_INDEX" \
      --index "$PYPI_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
      --constraints /opt/sglang-ci/runtime-constraints.txt \
      --constraints /opt/sglang-ci/top-level-constraints.txt \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      '/tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[hygon]'

# --- Application compatibility: distinguish HCU from AMD ROCm ----------------
# Kept after the slow install layer so compatibility tuning reuses the
# downloaded closure.
# REQUIRED: SGLang 0.5.18 treats every HIP runtime as AMD ROCm in three
# eagerly-imported Quark modules and then imports AMD AITER on HCU. AITER
# is not an HCU dependency and must not be added to the Runtime, so the
# reviewed source patch is applied to the installed application package.
# --check makes a future wheel/source change fail visibly rather than
# silently shipping an unpatched image; the .sha256 record is the
# applied-patch marker consumed by the post-build verifier.
# Evidence: the original failure "ModuleNotFoundError: No module named
# 'aiter'"; reviewed patch commit a5d492a.
# Removal condition: an official SGLang wheel/upstream version already
# carries the equivalent HCU-vs-ROCm gating.
COPY docker/hygon/sglang-v0.5.18-hcu.patch /opt/sglang-ci/sglang-v0.5.18-hcu.patch
RUN PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && cd "$PURELIB" \
    && git apply --check -p2 /opt/sglang-ci/sglang-v0.5.18-hcu.patch \
    && git apply -p2 /opt/sglang-ci/sglang-v0.5.18-hcu.patch \
    && git apply --reverse --check -p2 /opt/sglang-ci/sglang-v0.5.18-hcu.patch \
    && python -m compileall -q \
         sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py \
         sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py \
         sglang/srt/layers/quantization/quark_int4fp8_moe.py \
         sglang/srt/layers/attention/triton_backend.py \
         sglang/srt/layers/activation.py \
         sglang/srt/layers/layernorm.py \
         sglang/srt/layers/sampler.py \
    && sha256sum /opt/sglang-ci/sglang-v0.5.18-hcu.patch \
         > /opt/sglang-ci/sglang-hcu-patch.sha256

# --- Application compatibility: import stubs for absent kernel wheels --------
# REQUIRED: SGLang 0.5.18 imports sgl_kernel / flashinfer eagerly at module
# scope in many srt files whenever the CUDA/HIP fast path is compiled in
# (eagle_utils, communicator, deepseek_v2, the cuda-graph runners, ...).
# HCU provides neither vendor wheel, and the plugin dispatches the executed
# ops to the FlagOS backends (DEFAULT priority outranks the VENDOR
# sgl_kernel implementations, which are therefore never selected). The HCU
# patch above gates the sites with real call semantics; these stubs keep
# the long tail of pure import-time references alive and turn any
# unexpected call into an explicit error. No dist-info is emitted on
# purpose: pip/importlib.metadata stay byte-compatible with the reviewed
# closure (the engine's wheel-version probe is disabled via ENV below).
# Evidence: run 43311162022 / job 131508435886 - "ModuleNotFoundError: No
# module named 'sgl_kernel.kvcacheio'" (10 failed) before the stubs.
# Removal condition: SGLang upstream turns all these imports into correct
# optional/lazy imports, or HCU gets a verified official vendor wheel.
# The stub finder deliberately does NOT fake sgl_kernel.scalar_type: the
# module must raise a real ImportError so SGLang enters its pure-Python
# fallback.
# Evidence: run 43361932722 / job 131677473082 - with the module stubbed,
# "AttributeError: 'function' object has no attribute 'uint4b8'".
# Removal condition: the wildcard stub finder is retired, or SGLang no
# longer depends on that ImportError fallback.
COPY docker/hygon/stubs/ /opt/sglang-ci/stubs/
RUN PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && mkdir -p "$PURELIB/sgl_kernel/utils" "$PURELIB/flashinfer" \
    && cp /opt/sglang-ci/stubs/sgl_kernel/__init__.py "$PURELIB/sgl_kernel/__init__.py" \
    && cp /opt/sglang-ci/stubs/sgl_kernel/utils/__init__.py "$PURELIB/sgl_kernel/utils/__init__.py" \
    && cp /opt/sglang-ci/stubs/flashinfer/__init__.py "$PURELIB/flashinfer/__init__.py" \
    && cp /opt/sglang-ci/stubs/_hcu_kernel_stub_finder.py "$PURELIB/_hcu_kernel_stub_finder.py" \
    && cp /opt/sglang-ci/stubs/zz-hcu-kernel-stub-hook.pth "$PURELIB/zz-hcu-kernel-stub-hook.pth" \
    && echo "stub files written"

# The engine compares the installed sgl_kernel wheel version against its
# expectation; a stub has no dist-info version, so skip the probe (iluvatar
# precedent). This ENV is a companion of the stub DESIGN above, not a
# Runtime toolchain defect.
# Removal condition: the stub/version probe disappears together with a
# fixed SGLang wheel or a real vendor wheel - if the corrected wheel no
# longer needs the stubs, delete this first.
ENV SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# CONTRACT: the image carries its own DTK loader path. The Runtime ships
# the DTK userspace under /opt/dtk but leaves the loader path to the
# deployment environment, and runner hosts differ in whether they inject
# it: the identical container imported torch on one host and failed with
# "libgalaxyhip.so.5: cannot open shared object file" on another. The
# paths are image-internal (plus the hyhal bind mounts), so baking them
# makes the image self-sufficient and host-independent.
# Evidence: commit 06ff229 (the recorded loader-contract failure site).
# Removal condition: the Runtime provides the full loader contract via
# OCI ENV, ld.so.conf, or another in-image stable mechanism, verified on
# different runners.
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64

# CONTRACT: FlagTree's HCU backend derives its clang selection from
# ROCM_PATH, and direct entrypoints carry no ROCM_PATH. Without it the
# clang-17 compatibility tree gets selected and fails on unsupported
# arguments; with it FlagTree selects the aillvm clang-18 under /opt/dtk.
# Evidence: run 43426581773 (compatibility clang selected, unsupported
# parameter failure) vs run 43430350486 (/opt/dtk/aillvm/bin/clang-18
# selected, green). ROCM_PATH alone does NOT solve the device-library
# problem - that is the wrapper block below.
# Removal condition: Runtime/FlagTree stops deriving clang from this
# variable, or the Runtime officially provides it via OCI ENV.
ENV ROCM_PATH=/opt/dtk

# CONTRACT: SGLang 0.5.18's HIP JIT builder looks for hipcc and headers
# under the conventional /opt/rocm prefix; the Runtime ships the complete
# DTK tree at /opt/dtk and no /opt/rocm alias.
# Evidence: run 43417967624 / job 131868790147 - "/opt/rocm/bin/hipcc:
# not found", "Failed to build JIT module
# sgl_kernel_jit_clamp_position_int64_t".
# Removal condition: the SGLang builder natively supports the DTK prefix,
# or the Runtime officially provides the compatibility alias.
RUN if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then ln -s /opt/dtk /opt/rocm; fi \
    && test "$(readlink -f /opt/rocm)" = "$(readlink -f /opt/dtk)"

# CONTRACT: the HCU compile path must pass the DTK device-library location
# to clang explicitly. The Runtime already ships the complete DTK
# toolchain - /opt/dtk/aillvm/bin/clang-18 and the ocml.bc / ockl.bc
# bitcode under /opt/dtk/dcc/dccgcn/bitcode; what is missing is only the
# explicit passing: FlagTree 0.7.0's hcu backend asks clang to discover
# the DTK device libraries on its own and that discovery fails on the
# BW1000 runner. This wrapper is injected at the seam FlagTree documents
# (TRITON_HIP_CLANG_PATH) and appends
# --rocm-device-lib-path=/opt/dtk/dcc/dccgcn/bitcode. The REQUIRED
# capability is the explicit device-lib path passing; this wrapper is one
# workaround implementation, not the only legal one. Runtime-owned
# /opt/flagtree files stay untouched; the post-build verifier compiles a
# real OCML-referencing kernel through the wrapper on the GPU-less host.
# Evidence: build-infra #1786 ("cannot find ROCm device library"); after
# the wrapper, mtp_logprobs 10/10 and the full run 44465620210 matrix.
# Removal condition: Runtime/FlagTree _get_clang_args or the equivalent
# compile path natively passes the correct DTK device-lib arguments, with
# real kernel-compile evidence.
COPY docker/hygon/hcu-clang/clang-18-device-libs /opt/hcu-clang/clang-18-device-libs
RUN chmod +x /opt/hcu-clang/clang-18-device-libs \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ocml.bc \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ockl.bc \
    && /opt/dtk/aillvm/bin/clang-18 --version | head -n 1
ENV TRITON_HIP_CLANG_PATH=/opt/hcu-clang/clang-18-device-libs
