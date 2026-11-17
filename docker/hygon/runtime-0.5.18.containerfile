# SGLang 0.5.18 CI image on the Hygon operations runtime (DTK 26.04).
#
# Principles (do not weaken without review):
#   * The operations runtime is immutable: no runtime distribution may be
#     upgraded, downgraded, replaced, or removed. The resolver must only ADD
#     the SGLang dependency closure. Any planned change to an existing
#     distribution fails the build.
#   * Hygon installs sglang[srt_empty,runtime_common]: the dependency
#     ownership-inversion extras. The DTK torch/FlagTree/FlagGems/FlagCX
#     stack comes from the runtime, never from this build. runtime_common
#     is the wheel's own extra that adds compressed-tensors (unpinned)
#     beyond runtime_base, because sglang.srt's quantization registry
#     imports compressed_tensors eagerly at server start while the wheel
#     only declares it under runtime-common. No hand-picked packages.
#   * No sglang-plugin-FL source in this image; CI installs the plugin from
#     the PR checkout via the standard setup flow.
#   * Build logic lives in docker/hygon/scripts/ (reviewable, lintable,
#     black-formatted) and runs through BuildKit bind mounts, so the scripts
#     never enter the image;
#     this file is orchestration only. Verification gates stay in the build
#     so a drifting image fails before it can be pushed, and the volatile
#     gates are layered after the heavy install so tuning them reuses the
#     downloaded closure.
#
# Build from the repository root:
#   docker build -f docker/hygon/runtime-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-runtime-ci-<date> .

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04@sha256:03e598b348e58e7233628394bbefa0ccef26969a091c7e6f2f02835161757f8b
FROM ${BASE_IMAGE}

# FlagTree (/opt/flagtree) is the runtime's authoritative Triton. The base
# image exposes it to bash sessions only (BASH_ENV -> /etc/bash_env.sh ->
# /etc/profile.d/vendor.sh), so the resolved Triton silently depended on the
# container entrypoint (bash saw FlagTree 3.6, a direct python entrypoint
# saw the closure's standalone wheel). Bake it as image-level PYTHONPATH so
# shell, direct python entrypoints and CI jobs all resolve the same
# /opt/flagtree tree; the standalone triton wheel that xgrammar's
# declaration pulls in is removed right after the install below (the cuda
# CI image uses the same convention).
ENV PYTHONPATH=/opt/flagtree

ARG SGLANG_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-hygon/packages/sglang/0.5.18+flagos/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl
ARG SGLANG_WHEEL_SHA256=1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800
ARG SGLANG_WHEEL_NAME=sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl
# resource.flagos.net's pypi-proxy intermittently 504s large upstream fetches
# and pip does not retry HTTP 504; every pip call below loops instead, because
# pip's HTTP cache keeps whatever the previous attempt already fetched.
ARG PIP_MAIN_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
# PyPI serves generic wheels at datacenter speed; the pypi-proxy mirrors
# PyPI byte-for-byte but its gateway times out delivering large wheels on
# transpacific links (sglang-kernel is 377 MB), so it stays the LAST index.
# The platform index keeps first priority: flagos-built wheels (and the
# fsspec-pin reality - the runtime ships 2026.9.0, which every datasets
# release the proxy carries caps below; datasets 5.2.0 is the first
# release allowing it) always win from flagos-pypi-hygon.
ARG PIP_FALLBACK_INDEX=https://pypi.org/simple
ARG PIP_EXTRA_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

WORKDIR /opt/sglang-ci

# Application-level constraints: only evidence-backed pins (see the file).
# The runtime inventory is frozen dynamically at build start and the SGLang
# wheel pins its own direct requirements, so the closure's transitive
# resolution stays with the resolver; the dry-run gate verifies zero runtime
# replacements and the resolved closure is recorded as a build-time audit
# artifact (/opt/sglang-ci/sglang-additions-installed.txt), never committed
# back as an input lock.
COPY docker/hygon/top-level-constraints.txt /opt/sglang-ci/top-level-constraints.txt

# --- Stable, slow: fetch and SHA256-verify the SGLang wheel ------------------
RUN --mount=type=bind,target=/ctx python /ctx/docker/hygon/scripts/fetch_sglang_wheel.py

# --- Stable: capture the pre-install runtime state ---------------------------
# Canonicalized distribution inventory, exact constraints for the resolver,
# the runtime's own pip check result (kept as the baseline so pre-existing
# runtime findings are not attributed to SGLang), the file-level
# site-packages baseline, and module-level provenance for the vendor stack.
RUN --mount=type=bind,target=/ctx python -m pip freeze --all | sort > /opt/sglang-ci/runtime-freeze-before.txt \
    && { python -m pip check > /opt/sglang-ci/runtime-pip-check-baseline.txt 2>&1; \
         printf '%s\n' "$?" > /opt/sglang-ci/runtime-pip-check-baseline.exitcode; true; } \
    && python /ctx/docker/hygon/scripts/capture_runtime_baseline.py

# --- Resolver gate: dry-run with report before touching site-packages --------
# Any runtime distribution in the install plan fails the build: a different
# version is a hard conflict, an identical version is a reinstall that needs
# explicit review. New (absent) distributions are the SGLang closure; they are
# recorded for review but allowed.
RUN --mount=type=cache,target=/tmp/pip-cache \
    --mount=type=bind,target=/ctx \
    dry_run_rc=1 \
    && for attempt in 1 2 3 4 5 6 7 8; do \
         if python -m pip install \
             --dry-run \
             --timeout 120 \
             --retries 20 \
             --cache-dir /tmp/pip-cache \
             --report /opt/sglang-ci/sglang-resolver-plan.json \
             --constraint /opt/sglang-ci/runtime-constraints.txt \
             --constraint /opt/sglang-ci/top-level-constraints.txt \
             --index-url "$PIP_MAIN_INDEX" \
             --extra-index-url "$PIP_FALLBACK_INDEX" \
             --extra-index-url "$PIP_EXTRA_INDEX" \
             "/opt/sglang-ci/wheels/${SGLANG_WHEEL_NAME}[srt_empty,runtime_common]"; then \
           dry_run_rc=0; break; \
         else \
           dry_run_rc="$?"; \
         fi; \
         echo "dry-run attempt ${attempt} failed (rc=${dry_run_rc}); retrying" >&2; \
         sleep 20; \
       done \
    && [ "$dry_run_rc" -eq 0 ] \
    && python /ctx/docker/hygon/scripts/validate_resolver_plan.py

# --- Stable, slow: the real install ------------------------------------------
# Same constraints and indexes as the reviewed dry-run plan. The standalone
# triton wheel (pulled by xgrammar's declaration) is uninstalled in the SAME
# layer: FlagTree provides the only Triton, and a later-layer uninstall would
# still ship the ~893 MB in this layer.
RUN --mount=type=cache,target=/tmp/pip-cache \
    install_rc=1 \
    && for attempt in 1 2 3 4 5 6 7 8; do \
         if python -m pip install \
             --timeout 120 \
             --retries 20 \
             --cache-dir /tmp/pip-cache \
             --constraint /opt/sglang-ci/runtime-constraints.txt \
             --constraint /opt/sglang-ci/top-level-constraints.txt \
             --index-url "$PIP_MAIN_INDEX" \
             --extra-index-url "$PIP_FALLBACK_INDEX" \
             --extra-index-url "$PIP_EXTRA_INDEX" \
             "/opt/sglang-ci/wheels/${SGLANG_WHEEL_NAME}[srt_empty,runtime_common]"; then \
           install_rc=0; break; \
         else \
           install_rc="$?"; \
         fi; \
         echo "install attempt ${attempt} failed (rc=${install_rc}); retrying" >&2; \
         sleep 20; \
       done \
    && [ "$install_rc" -eq 0 ] \
    && while python -m pip show triton >/dev/null 2>&1; do \
         python -m pip uninstall -y triton; \
       done \
    && echo "standalone triton wheel removed; FlagTree is the only Triton"

# --- Application compatibility: distinguish HCU from AMD ROCm --------------
# Keep this after the slow dependency layer so changes to the application-only
# compatibility patch reuse the downloaded and installed requirements cache.
# SGLang 0.5.18 treats every HIP runtime as AMD ROCm in three eagerly-imported
# Quark modules and consequently imports AMD AITER on HCU.  AITER is not an HCU
# requirement and must not be added to the Runtime.  Apply the reviewed SGLang
# source patch to the installed application package instead.  --check makes a
# future wheel/source change fail visibly rather than silently shipping an
# unpatched image.
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

# --- Application compatibility: import stubs for absent kernel wheels -------
# SGLang 0.5.18 imports sgl_kernel / flashinfer eagerly at module scope in
# many srt files whenever the CUDA/HIP fast path is compiled in (eagle_utils,
# communicator, deepseek_v2, the cuda-graph runners, ...).  HCU provides
# neither wheel, and the plugin dispatches the executed ops to the FlagOS
# backends (DEFAULT priority outranks the VENDOR sgl_kernel implementations,
# which are therefore never selected).  The HCU source patch above gates the
# sites with real call semantics; these stubs keep the long tail of pure
# import-time references alive and turn any unexpected call into an explicit
# error.  This mirrors the legacy dev image's empty sgl_kernel wheel and the
# iluvatar empty-image import shims.  No dist-info is emitted on purpose:
# pip/importlib.metadata stay byte-compatible with the reviewed closure, and
# the engine's wheel-version probe is disabled via ENV below.
COPY docker/hygon/stubs/ /opt/sglang-ci/stubs/
RUN PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && mkdir -p "$PURELIB/sgl_kernel/utils" "$PURELIB/flashinfer" \
    && cp /opt/sglang-ci/stubs/sgl_kernel/__init__.py "$PURELIB/sgl_kernel/__init__.py" \
    && cp /opt/sglang-ci/stubs/sgl_kernel/utils/__init__.py "$PURELIB/sgl_kernel/utils/__init__.py" \
    && cp /opt/sglang-ci/stubs/flashinfer/__init__.py "$PURELIB/flashinfer/__init__.py" \
    && cp /opt/sglang-ci/stubs/_hcu_kernel_stub_finder.py "$PURELIB/_hcu_kernel_stub_finder.py" \
    && cp /opt/sglang-ci/stubs/zz-hcu-kernel-stub-hook.pth "$PURELIB/zz-hcu-kernel-stub-hook.pth" \
    && echo "stub files written"

RUN python - <<'PY'
from sgl_kernel.kvcacheio import save_kv_cache
from flashinfer.sampling import top_k_renorm_prob

import sgl_kernel
import sgl_kernel.utils
import flashinfer

assert sgl_kernel.utils.is_arch_support_pdl() is False
try:
    save_kv_cache()
except RuntimeError as e:
    assert "stub" in str(e)
print("kernel-wheel import stubs + submodule finder installed")
PY

# The engine compares the installed sgl_kernel wheel version against its
# expectation; a stub has no version, so skip the probe (iluvatar precedent).
ENV SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# The base runtime ships the DTK userspace under /opt/dtk but leaves the
# loader path to the deployment environment; runner hosts differ in whether
# they inject LD_LIBRARY_PATH (observed: identical containers import torch on
# one host and fail with "libgalaxyhip.so.5: cannot open shared object file"
# on another).  The paths are image-internal (plus the hyhal bind mounts), so
# baking them here makes the image self-sufficient and host-independent.
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64

# The DTK prefix convention: SGLang 0.5.18's HIP JIT builder looks for hipcc
# and headers under the conventional ROCm prefix, and FlagTree's hcu backend
# derives its HCU clang-18 selection (aillvm, not the clang-17 compatibility
# tree) from ROCM_PATH. The base image ships the complete DTK tree at
# /opt/dtk but no /opt/rocm alias and no ROCM_PATH, so bake both here —
# plain ENV (not a bash-profile export) so test steps that never source a
# profile still inherit it.
ENV ROCM_PATH=/opt/dtk
RUN if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then ln -s /opt/dtk /opt/rocm; fi \
    && test "$(readlink -f /opt/rocm)" = "$(readlink -f /opt/dtk)"

# build-infra#1786: FlagTree 0.7.0's hcu backend asks clang to discover the
# DTK device libraries on its own and that discovery fails on the BW1000
# runner ("cannot find ROCm device library") even with ROCM_PATH set. Fix at
# the seam FlagTree documents for this (TRITON_HIP_CLANG_PATH): a baked
# wrapper around the real aillvm clang-18 that drops empty disabled-option
# strings and appends the explicit device-library path. Runtime-owned
# /opt/flagtree files stay untouched.
COPY docker/hygon/hcu-clang/clang-18-device-libs /opt/hcu-clang/clang-18-device-libs
RUN chmod +x /opt/hcu-clang/clang-18-device-libs \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ocml.bc \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ockl.bc \
    && /opt/dtk/aillvm/bin/clang-18 --version | head -n 1
ENV TRITON_HIP_CLANG_PATH=/opt/hcu-clang/clang-18-device-libs

# Prove the wrapper end-to-end on the GPU-less builder: the flagtree-style
# two-step compile (LLIR -> asm -> hsaco) against a kernel that references an
# OCML device-library symbol must succeed through the wrapper.
RUN --mount=type=bind,target=/ctx /opt/dtk/aillvm/bin/clang-18 -target amdgcn-amd-amdhsa -mcpu=gfx936:xnack- -O3 \
       /ctx/docker/hygon/scripts/hcu-device-lib-check.ll -S -o /tmp/hcu-device-lib-check.s \
    && /opt/hcu-clang/clang-18-device-libs -target amdgcn-amd-amdhsa -mcpu=gfx936:xnack- -O3 \
       /tmp/hcu-device-lib-check.s -x assembler -o /tmp/hcu-device-lib-check.hsaco \
    && test -s /tmp/hcu-device-lib-check.hsaco \
    && echo "wrapper hsaco compile through DTK device libraries: OK"

# --- Volatile verification (cheap layers, tuned without re-downloading) ------
# 1. Every runtime distribution survives with the exact same version and the
#    installed additions match the reviewed resolver plan.
RUN --mount=type=bind,target=/ctx python /ctx/docker/hygon/scripts/verify_distributions.py

# 1b. File-level zero-drift: every baseline site-packages file must still
#     exist with the identical SHA256 (catches in-place overwrites).
RUN --mount=type=bind,target=/ctx python /ctx/docker/hygon/scripts/verify_site_files.py

# 2. pip check: only violations NEW relative to the runtime baseline fail the
#    build; the runtime's own pre-existing findings stay on record. One
#    declared gap is exempt: xgrammar requires triton, which the runtime
#    provides out-of-band as FlagTree (pip cannot see it - same convention
#    as the cuda CI image).
RUN { python -m pip check > /opt/sglang-ci/runtime-pip-check-after.txt 2>&1; \
         printf '%s\n' "$?" > /opt/sglang-ci/runtime-pip-check-after.exitcode; true; } \
    && grep -v '^No broken requirements found\.$' /opt/sglang-ci/runtime-pip-check-baseline.txt | sort -u > /tmp/pip-check-baseline.lines \
    && grep -v '^No broken requirements found\.$' /opt/sglang-ci/runtime-pip-check-after.txt | sort -u > /tmp/pip-check-after.lines \
    && comm -13 /tmp/pip-check-baseline.lines /tmp/pip-check-after.lines \
       | { grep -v 'requires triton, which is not installed' || true; } \
       > /opt/sglang-ci/pip-check-new-violations.txt \
    && if [ -s /opt/sglang-ci/pip-check-new-violations.txt ]; then \
         echo 'FATAL: new pip check violations introduced by the SGLang install:'; \
         cat /opt/sglang-ci/pip-check-new-violations.txt; \
         exit 1; \
       fi \
    && echo 'pip check: no new violations (baseline preserved; xgrammar->triton provided by FlagTree)'

# 3. Module provenance: every vendor module must still resolve to the exact
#    same origin (see the script docstring for the one recorded exception).
RUN --mount=type=bind,target=/ctx python /ctx/docker/hygon/scripts/verify_module_provenance.py

# 3b. Triton resolution: with the standalone wheel gone and PYTHONPATH baked,
#     every entrypoint style must resolve the same FlagTree tree. Exercise
#     the plain RUN shell, an explicit bash, and a python -c child process.
RUN python -c 'import triton; p = triton.__file__; assert p and p.startswith("/opt/flagtree"), p; print("triton (sh):", p)' \
    && bash -c 'python -c "import triton; assert triton.__file__.startswith(\"/opt/flagtree\"), triton.__file__; print(\"triton (bash):\", triton.__file__)"' \
    && python -c 'import importlib.metadata as md; names=[(d.metadata["Name"] or "").lower() for d in md.distributions()]; assert "triton" not in names, "standalone triton distribution still installed"; print("standalone triton distribution absent: OK")'

# 4. Import smoke check for sglang itself (sentinel-classified; the script
#    docstring explains why the verdict comes from the log, not the exit code).
RUN --mount=type=bind,target=/ctx python /ctx/docker/hygon/scripts/sglang_import_smoke.py > /tmp/sglang-import-smoke.log 2>&1 || true
RUN cat /tmp/sglang-import-smoke.log \
    && grep -qE "SMOKE-PASS|SMOKE-TOLERATED|no ROCm-capable device is detected" \
       /tmp/sglang-import-smoke.log
