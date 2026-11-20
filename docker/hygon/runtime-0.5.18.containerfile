# SGLang 0.5.18 two-wheel CI image on the Hygon operations runtime (DTK 26.04).
#
# This recipe is runtime-image ASSEMBLY only: the operations Runtime base, one
# uv resolver install of the two real application wheels, and the HCU/DTK
# compatibility layer. Every acceptance gate - the resolver dry-run plan,
# runtime zero-drift, site-packages hashes, the pip-check diff, the wheel
# contract, the stub/wrapper/import smoke and the Triton provenance - runs
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
#   * Build inputs live under /tmp only for the RUN that consumes them and
#     are deleted in that same RUN; the released image carries no CI
#     working directory and no build evidence.
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

# Resolver inputs, consumed and removed by the install RUN below.
# runtime-constraints.txt freezes the pristine Runtime inventory (canonical
# name==version) and is generated from the resolved base by
# build_two_wheel.sh - never committed; requirements.txt is the reviewed,
# evidence-backed pin file (see its header).
COPY artifacts/runtime-constraints.txt /tmp/runtime-constraints.txt
COPY docker/hygon/requirements.txt /tmp/requirements.txt

# The two reviewed application wheels and the single-item resolver exclusion
# list are staged by docker/hygon/stage_wheels.sh; their SHA256s are part of
# the staging reproducibility record.
COPY artifacts/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl /tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl
COPY artifacts/runtime-0.5.18.excludes.txt /tmp/runtime-0.5.18.excludes.txt

# --- One uv resolver install of the two application wheels -------------------
# CONTRACT: no standalone triton distribution enters the image; the Runtime's
# /opt/flagtree tree (Triton 3.6) is the only compiler, and --excludes drops
# the standalone triton that xgrammar's metadata would otherwise pull in.
# Evidence: run 44465620210 - full Hygon hardware matrix green, no triton
# distribution installed, triton.__file__ under /opt/flagtree.
# Owner: FlagOS Runtime (remove once it ships a metadata-consistent triton).
# build_two_wheel.sh runs the exact twin of this command (plus --dry-run)
# against the resolved base digest and plan-validates it BEFORE the build.
RUN /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --index "$FLAGOS_INDEX" \
      --index "$TENCENT_INDEX" \
      --default-index "$PROXY_INDEX" \
      --index-strategy unsafe-first-match \
      --constraints /tmp/runtime-constraints.txt \
      -r /tmp/requirements.txt \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      '/tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl[srt_empty,runtime_common]' \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[hygon]' \
      && rm -rf /tmp/wheels /tmp/runtime-constraints.txt /tmp/requirements.txt /tmp/runtime-0.5.18.excludes.txt

# --- TEMP: HCU-vs-ROCm gating patch on the installed SGLang wheel ------------
# SGLang 0.5.18 treats every HIP runtime as AMD ROCm in three eagerly-imported
# Quark modules and then imports AMD AITER, which HCU does not ship; the
# reviewed patch gates those sites. --check/--reverse-check make a future
# wheel change fail visibly; the applied patch's SHA256 is recorded as the
# org.flagos.ci.hcu-patch-sha256 OCI label (build_two_wheel.sh) and verified
# post-build.
# Evidence: original "ModuleNotFoundError: No module named 'aiter'"; patch
# commit a5d492a.
# Owner: SGLang upstream (remove once a wheel carries the HCU gating).
COPY docker/hygon/sglang-v0.5.18-hcu.patch /tmp/sglang-v0.5.18-hcu.patch
RUN PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && cd "$PURELIB" \
    && git apply --check -p2 /tmp/sglang-v0.5.18-hcu.patch \
    && git apply -p2 /tmp/sglang-v0.5.18-hcu.patch \
    && git apply --reverse --check -p2 /tmp/sglang-v0.5.18-hcu.patch \
    && python -m compileall -q \
         sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py \
         sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py \
         sglang/srt/layers/quantization/quark_int4fp8_moe.py \
         sglang/srt/layers/attention/triton_backend.py \
         sglang/srt/layers/activation.py \
         sglang/srt/layers/layernorm.py \
         sglang/srt/layers/sampler.py \
    && rm -f /tmp/sglang-v0.5.18-hcu.patch

# --- TEMP: import stubs for kernel wheels HCU does not ship ------------------
# SGLang 0.5.18 imports sgl_kernel / flashinfer eagerly at module scope; HCU
# provides neither vendor wheel and the plugin dispatches the executed ops to
# the FlagOS backends. The stub finder keeps import-time references alive and
# turns unexpected calls into explicit errors; no dist-info is emitted so pip
# metadata stays byte-compatible (the wheel-version probe is disabled via the
# ENV below). The finder deliberately does NOT fake sgl_kernel.scalar_type:
# that module must raise a real ImportError so SGLang enters its pure-Python
# fallback.
# Evidence: run 43311162022 (sgl_kernel.kvcacheio ImportError, 10 failed);
# run 43361932722 (scalar_type stubbed -> uint4b8 AttributeError).
# Owner: SGLang upstream / HCU vendor wheel (remove with lazy imports or a
# real wheel; delete the ENV below together with the stubs).
COPY docker/hygon/stubs/ /tmp/stubs/
RUN PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && mkdir -p "$PURELIB/sgl_kernel/utils" "$PURELIB/flashinfer" \
    && cp /tmp/stubs/sgl_kernel/__init__.py "$PURELIB/sgl_kernel/__init__.py" \
    && cp /tmp/stubs/sgl_kernel/utils/__init__.py "$PURELIB/sgl_kernel/utils/__init__.py" \
    && cp /tmp/stubs/flashinfer/__init__.py "$PURELIB/flashinfer/__init__.py" \
    && cp /tmp/stubs/_hcu_kernel_stub_finder.py "$PURELIB/_hcu_kernel_stub_finder.py" \
    && cp /tmp/stubs/zz-hcu-kernel-stub-hook.pth "$PURELIB/zz-hcu-kernel-stub-hook.pth" \
    && rm -rf /tmp/stubs \
    && echo "stub files written"

# Companion of the stub design above: a stub has no dist-info version, so the
# engine's wheel-version probe would fail; skip it (iluvatar precedent).
ENV SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# --- TEMP: DTK loader path baked into the image ------------------------------
# Runner hosts differ in whether they inject the DTK loader path; the paths
# are image-internal (plus the hyhal bind mounts), so baking them makes the
# image self-sufficient. Prepend-only: ${VAR:+} keeps any base value and
# never produces an empty path element (verified by verify_image.sh).
# Evidence: commit 06ff229 (recorded loader-contract failure site).
# Owner: FlagOS Runtime (remove once it ships the loader contract).
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# --- TEMP: FlagTree derives its clang selection from ROCM_PATH ---------------
# Without it the clang-17 compatibility tree gets selected and fails; with it
# FlagTree selects the aillvm clang-18 under /opt/dtk.
# Evidence: run 43426581773 (compatibility clang failure) vs run 43430350486
# (aillvm clang-18 selected, green).
# Owner: FlagOS Runtime / FlagTree (remove when clang selection stops using
# this variable or the Runtime provides it).
ENV ROCM_PATH=/opt/dtk

# --- TEMP: SGLang's HIP JIT builder expects the /opt/rocm prefix -------------
# The Runtime ships the complete DTK tree at /opt/dtk and no /opt/rocm alias.
# Evidence: run 43417967624 / job 131868790147 - "/opt/rocm/bin/hipcc: not
# found", "Failed to build JIT module sgl_kernel_jit_clamp_position_int64_t".
# Owner: SGLang upstream / FlagOS Runtime (remove with native DTK-prefix
# support or an official alias).
RUN if [ ! -e /opt/rocm ] && [ ! -L /opt/rocm ]; then ln -s /opt/dtk /opt/rocm; fi \
    && test "$(readlink -f /opt/rocm)" = "$(readlink -f /opt/dtk)"

# --- TEMP: DTK device-library path passed to clang explicitly ----------------
# FlagTree 0.7.0's hcu backend asks clang to discover the DTK device libraries
# on its own and that discovery fails on the BW1000 runner; this wrapper is
# injected at the seam FlagTree documents (TRITON_HIP_CLANG_PATH) and appends
# --rocm-device-lib-path=/opt/dtk/dcc/dccgcn/bitcode. One workaround
# implementation, not the only legal one; runtime-owned /opt/flagtree files
# stay untouched and the post-build verifier compiles a real OCML-referencing
# kernel through the wrapper on a GPU-less host.
# Evidence: build-infra #1786 ("cannot find ROCm device library"); after the
# wrapper, mtp_logprobs 10/10 and the full run 44465620210 matrix.
# Owner: FlagTree / FlagOS Runtime (remove when the compile path passes the
# device-lib arguments natively, with real kernel-compile evidence).
COPY docker/hygon/hcu-clang/clang-18-device-libs /opt/hcu-clang/clang-18-device-libs
RUN chmod +x /opt/hcu-clang/clang-18-device-libs \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ocml.bc \
    && test -f /opt/dtk/dcc/dccgcn/bitcode/ockl.bc \
    && /opt/dtk/aillvm/bin/clang-18 --version | head -n 1
ENV TRITON_HIP_CLANG_PATH=/opt/hcu-clang/clang-18-device-libs
