# Ascend 910C CI candidate on CANN 9.2.1, Python 3.12 and Torch 2.13.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1-910c@sha256:37df7376fb9a8ed3201e0793d4f69fed1cfb12386e4edbbd88e0a58ea8030989
FROM $BASE_IMAGE AS runtime

ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
ARG SGLANG_ARCHIVE_SHA256=db28d0c29b82bc6feb2d4a9a5a5e43fe75b51dafb461a61229be5c0ced190f05
ARG SGLANG_FL_REVISION
ARG SGL_KERNEL_NPU_RELEASE=2026.11.0.post3
ARG SGL_KERNEL_NPU_COMMIT=d5107fb699fe1f0ce3fd900365264211000a1297
ARG SGL_KERNEL_NPU_SHA256=e238f5a511484cdf4352c8fda612106a2ca761716f9d4e72135cf292435cdffa
ARG SGL_CUSTOM_OPS_SHA256=827d70c9b31f4baf1c3ad08bb5f1d7ef0719ce04abd2edf1d5f0f1eb685fe048
ARG SGL_OPS_TRANSFORMER_SHA256=b65172505dd9933501de421fbca5e9fa1b61bb6598c9cfb4eb3476fac15af85b
ARG FLAGTREE_WHEEL_SHA256=335d0f124c1264aa764112d4cbfa67e6ea38ffad9d08c3ee52b3acfc2f1d4a91

SHELL ["/bin/bash", "-c"]
ENV PATH=/usr/local/Ascend/driver/tools:$PATH \
    PYTHONNOUSERSITE=1 \
    SGLANG_BUILD_RUST_EXTS=none

# This native release targets CANN 9.2.0 and carries the A3 packages for Torch
# 2.13 and Python 3.12. The CI device gate checks this pairing on CANN 9.2.1.
RUN set -euo pipefail; \
    work="$(mktemp -d)"; \
    trap 'rm -rf "$work"' EXIT; \
    cd "$work"; \
    base="https://github.com/sgl-project/sgl-kernel-npu/releases/download/$SGL_KERNEL_NPU_RELEASE"; \
    suffix="$SGL_KERNEL_NPU_RELEASE-torch2.13.0"; \
    kernel="sgl-kernel-npu-$suffix-py312-cann9.2.0-a3-aarch64.zip"; \
    custom="custom-ops-$suffix-cann9.2.0-a3-aarch64.zip"; \
    transformer="ops-transformer-$suffix-cann9.2.0-a3-aarch64.zip"; \
    for asset in "$kernel" "$custom" "$transformer"; do \
        curl --fail --location --retry 5 --retry-delay 5 "$base/$asset" --output "$asset"; \
    done; \
    printf '%s  %s\n' "$SGL_KERNEL_NPU_SHA256" "$kernel" \
        "$SGL_CUSTOM_OPS_SHA256" "$custom" "$SGL_OPS_TRANSFORMER_SHA256" "$transformer" | sha256sum --check -; \
    python3 -m zipfile -e "$kernel" kernel; \
    python3 -m zipfile -e "$custom" custom; \
    python3 -m zipfile -e "$transformer" transformer; \
    opp=/usr/local/Ascend/cann-9.2.1/opp; \
    test -d "$opp"; \
    bash custom/CANN-custom_ops-none-linux.aarch64.run --quiet --install-path="$opp"; \
    bash transformer/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path="$opp"; \
    python3 -m pip install --no-cache-dir --no-index --no-deps kernel/*.whl custom/*.whl; \
    site="$(python3 -c 'from importlib.metadata import distribution; print(distribution("deep-ep").locate_file(""))')"; \
    set -- "$site"/deep_ep/deep_ep_cpp*.so; \
    test "$#" -eq 1; \
    test -f "$1"; \
    extension="$(basename "$1")"; \
    binding="$site/$extension"; \
    if [ -e "$binding" ] || [ -L "$binding" ]; then \
        test "$1" -ef "$binding"; \
    else \
        ln -s "deep_ep/$extension" "$binding"; \
    fi; \
    test -f "$opp/vendors/customize/bin/set_env.bash"; \
    test -f "$opp/vendors/custom_transformer/bin/set_env.bash"; \
    printf '%s\n' 'source /etc/bash_env.sh' \
        "source $opp/vendors/customize/bin/set_env.bash" \
        "source $opp/vendors/custom_transformer/bin/set_env.bash" > /etc/sglang-fl-ascend-env.sh

# Preserve the installed OPP paths for CI steps that launch Python from sh.
ENV BASH_ENV=/etc/sglang-fl-ascend-env.sh \
    ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_transformer:/usr/local/Ascend/cann-9.2.1/opp/vendors/customize \
    LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_transformer/op_api/lib:/usr/local/Ascend/cann-9.2.1/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

# Make Torch NPU's bundled TorchAir visible to SGLang's top-level import.
RUN set -euo pipefail; \
    site="$(python3 -c 'from importlib.metadata import distribution; print(distribution("torch-npu").locate_file(""))')"; \
    test -d "$site/torch_npu/dynamo/torchair"; \
    printf '%s\n' 'torch_npu/dynamo' > "$site/sglang-fl-torchair.pth"

# Keep the digest-pinned Torch, FlagGems, FlagCX and SHMEM stack while
# installing the pinned SGLang source through its Torch-free empty extra.
RUN set -euo pipefail; \
    archive="$(mktemp)"; \
    trap 'rm -f "$archive"' EXIT; \
    curl --fail --location --retry 5 --retry-delay 5 \
        "https://codeload.github.com/sgl-project/sglang/tar.gz/$SGLANG_COMMIT" --output "$archive"; \
    printf '%s  %s\n' "$SGLANG_ARCHIVE_SHA256" "$archive" | sha256sum --check -; \
    mkdir -p /opt/sglang-0.5.18; \
    tar -xzf "$archive" --strip-components=1 -C /opt/sglang-0.5.18; \
    printf '%s\n' "$SGLANG_COMMIT" > /opt/sglang-0.5.18/.flagos-source-commit; \
    printf '%s\n' 'torch==2.13.0+cpu' 'torch-npu==2.13.0' 'torchvision==0.28.0+cpu' \
        'torchaudio==2.11.0+cpu' 'numpy==1.26.4' 'flag-gems==5.4.0' \
        'flagcx==0.14.1+cann9.2.1' 'cann-shmem==1.6.0' 'transformers==5.12.1' \
        'triton==0.0.0' 'triton-ascend==0.0.0' > /tmp/sglang-ascend-constraints.txt; \
    cd /opt/sglang-0.5.18/python; \
    cp pyproject_other.toml pyproject.toml; \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18 \
        python3 -m pip install --no-cache-dir -c /tmp/sglang-ascend-constraints.txt \
            '.[srt_empty]' 'setuptools>=64' PyYAML; \
    python3 -m pip install --no-cache-dir -c /tmp/sglang-ascend-constraints.txt \
        xgrammar==0.2.1 compressed-tensors==0.16.0; \
    rm /tmp/sglang-ascend-constraints.txt

RUN test -n "$SGLANG_FL_REVISION" && \
    mkdir -p /opt/sglang-plugin-fl && \
    printf '%s\n' "$SGLANG_FL_REVISION" > /opt/sglang-plugin-fl/.flagos-source-commit
COPY pyproject.toml /opt/sglang-plugin-fl/pyproject.toml
COPY sglang_fl /opt/sglang-plugin-fl/sglang_fl
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/sglang-plugin-fl

# The exact CP312 wheel was built from FlagTree commit 2b5b8132. A read-only
# context mount lets pip consume it without adding the archive to a layer. The
# vendor runtime selects /opt/flagtree and keeps its alternative in /opt/triton.
RUN --mount=type=bind,source=docker/ascend/.build-inputs/flagtree-0.7.0+ascend3.5-cp312-cp312-linux_aarch64.whl,target=/tmp/flagtree-0.7.0+ascend3.5-cp312-cp312-linux_aarch64.whl,readonly \
    set -euo pipefail; \
    wheel=/tmp/flagtree-0.7.0+ascend3.5-cp312-cp312-linux_aarch64.whl; \
    printf '%s  %s\n' "$FLAGTREE_WHEEL_SHA256" "$wheel" | sha256sum --check -; \
    python3 -m pip install --no-cache-dir --no-index --no-deps --target /opt/flagtree-new "$wheel"; \
    mv /opt/flagtree /opt/flagtree-old; \
    mv /opt/flagtree-new /opt/flagtree; \
    rm -rf /opt/flagtree-old

ENV SGLANG_FL_ASCEND_CANN921=1 \
    PYTHONPATH=/opt/flagtree

COPY .github/scripts/ascend/verify_environment.py /usr/local/bin/verify-sglang-fl-ascend
RUN chmod 0755 /usr/local/bin/verify-sglang-fl-ascend && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 /usr/local/bin/verify-sglang-fl-ascend

LABEL ai.flagos.sglang.version="0.5.18" \
      ai.flagos.sglang.revision="$SGLANG_COMMIT" \
      ai.flagos.sglang-plugin-fl.version="0.1.0" \
      ai.flagos.sglang-plugin-fl.revision="$SGLANG_FL_REVISION" \
      ai.flagos.flagtree.version="0.7.0+ascend3.5" \
      ai.flagos.flagtree.revision="2b5b81320c18dbf4008cf6ad14d3c246bd74f7ff" \
      ai.flagos.flagtree.wheel.sha256="$FLAGTREE_WHEEL_SHA256" \
      ai.flagos.flaggems.version="5.4.0" \
      ai.flagos.flagcx.version="0.14.1+cann9.2.1" \
      ai.flagos.sgl-kernel-npu.release="$SGL_KERNEL_NPU_RELEASE" \
      ai.flagos.sgl-kernel-npu.revision="$SGL_KERNEL_NPU_COMMIT" \
      ai.flagos.sgl-kernel-npu.version="2026.9.0" \
      ai.flagos.sgl-kernel-npu.archive.sha256="$SGL_KERNEL_NPU_SHA256" \
      ai.flagos.sgl-custom-ops.archive.sha256="$SGL_CUSTOM_OPS_SHA256" \
      ai.flagos.sgl-ops-transformer.archive.sha256="$SGL_OPS_TRANSFORMER_SHA256" \
      ai.flagos.cann.version="9.2.1" \
      ai.flagos.torch-npu.version="2.13.0"

FROM runtime AS ci

RUN python3 -m pip install --no-cache-dir pytest==9.1.1 pytest-timeout==2.4.0 \
        pytest-cov pytest-json-report pytest-metadata pyyaml ruff && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 /usr/local/bin/verify-sglang-fl-ascend --require-ci

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace

# Stage the exact CP312 wheel in the ignored build-inputs directory, then
# build from the repository root. Set FLAGTREE_WHEEL to the artifact path.
#   wheel=docker/ascend/.build-inputs/flagtree-0.7.0+ascend3.5-cp312-cp312-linux_aarch64.whl
#   mkdir -p "$(dirname "$wheel")"
#   cp "$FLAGTREE_WHEEL" "$wheel"
#   printf '%s  %s\n' '335d0f124c1264aa764112d4cbfa67e6ea38ffad9d08c3ee52b3acfc2f1d4a91' "$wheel" | sha256sum --check
#   DOCKER_BUILDKIT=1 docker build --platform linux/arm64 --target ci \
#     --build-arg SGLANG_FL_REVISION="$(git rev-parse HEAD)" \
#     -f docker/ascend/cann921-0.5.18.containerfile -t sglang-fl:ascend-cann921-candidate .
