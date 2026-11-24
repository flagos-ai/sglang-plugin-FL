# SGLang 0.5.18 two-wheel DEVELOPMENT CANDIDATE: the release recipe
# (runtime-0.5.18.containerfile) resolved offline from a wheelhouse. TEMP,
# never a release input; the base is always passed as a verified digest.
# Evidence and boundaries: docker/hygon/README.md
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# TEMP (FlagOS Runtime): /opt/flagtree is exposed to bash sessions only; also the
# Triton compiler slot. Evidence/owner/removal gate: docker/hygon/README.md
ENV PYTHONPATH=/opt/flagtree

# Same install as the release recipe, resolved only from the wheelhouse
# (--no-index --find-links) with no network. Evidence: docker/hygon/README.md
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
