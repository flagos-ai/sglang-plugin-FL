# Build from the repository root. Plugin changes invalidate only the final layer.
# --target dependencies retains the stable stack while iterating on the plugin.
FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1@sha256:679182a04d187562a3e23bacb9d0eb876dabf09ac0c0a875c2b4a141b64cac69 AS runtime
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# This is the native Kunlunxin kernel from the validated development stack.
# The sgl-kernel-shim package cannot implement its torch.ops.sgl_kernel ops.
FROM runtime AS vendor-kernel
ARG SGL_KERNEL_URL=https://klx-public.bj.bcebos.com/v1/liquanfeng01/fl/deps/20260709/sglang_kernel-0.4.2.post2%2Bd4466f4499-cp39-abi3-linux_x86_64.whl
RUN --mount=type=cache,id=kunlunxin-pip,target=/root/.cache/pip \
    python -m pip install --no-deps "${SGL_KERNEL_URL}#sha256=146d4b7a318ca7104e4152df94c3f56941006675717480f52ba3077ee5ed8315"

FROM vendor-kernel AS gems
ARG FLAGGEMS_COMMIT=7fcdbed07b82c462ec2a2a7d7da00d788d099e67
RUN git init /opt/FlagGems && \
    git -C /opt/FlagGems remote add origin https://github.com/flagos-ai/FlagGems.git && \
    git -C /opt/FlagGems fetch --depth 1 origin "${FLAGGEMS_COMMIT}" && \
    git -C /opt/FlagGems checkout --detach FETCH_HEAD
RUN --mount=type=cache,id=kunlunxin-pip,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0+g7fcdbed07 \
        python -m pip install --no-deps /opt/FlagGems

# Preserve the runtime's paired Torch/XRE/FlagTree/FlagCX builds when resolving
# generic SGLang dependencies. FlagCX's runtime wheel supplies its shared library;
# the source tree supplies the interservice wrapper used by sglang_fl.
FROM gems AS flagcx
ARG FLAGCX_COMMIT=7812d2e90057868ba5c41f47fbb45a429252b9aa
RUN git init /opt/FlagCX && \
    git -C /opt/FlagCX remote add origin https://github.com/flagos-ai/FlagCX.git && \
    git -C /opt/FlagCX fetch --depth 1 origin "${FLAGCX_COMMIT}" && \
    git -C /opt/FlagCX checkout --detach FETCH_HEAD
RUN python - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
dist = distribution('flagcx')
libraries = [Path(dist.locate_file(f)) for f in dist.files
             if str(f).endswith('/libflagcx.so')]
assert len(libraries) == 1, libraries
libdir = Path('/opt/FlagCX/build/lib')
libdir.mkdir(parents=True, exist_ok=True)
(libdir / 'libflagcx.so').symlink_to(libraries[0])
packages = ('torch', 'torchvision', 'torchaudio', 'flagtree', 'flag_gems',
            'flagcx', 'sglang-kernel', 'numpy', 'torch-plugin', 'torch-xray',
            'xtorch-ops', 'xmlir', 'flash-attn')
lines = ['triton<0']
for name in packages:
    try:
        lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
Path('/opt/vendor-constraints.txt').write_text('\n'.join(lines) + '\n')
PY

FROM flagcx AS sglang-source
ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
RUN git init /opt/sglang && \
    git -C /opt/sglang remote add origin https://github.com/sgl-project/sglang.git && \
    git -C /opt/sglang fetch --depth 1 origin "${SGLANG_COMMIT}" && \
    git -C /opt/sglang checkout --detach FETCH_HEAD

FROM sglang-source AS sglang
COPY docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch /opt/sglang-v0.5.18-kunlunxin.patch
RUN git -C /opt/sglang apply --check /opt/sglang-v0.5.18-kunlunxin.patch && \
    git -C /opt/sglang apply /opt/sglang-v0.5.18-kunlunxin.patch && \
    cp /opt/sglang/python/pyproject_other.toml /opt/sglang/python/pyproject.toml
RUN --mount=type=cache,id=kunlunxin-pip,target=/root/.cache/pip \
    SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.18 \
    python -m pip install --no-build-isolation -c /opt/vendor-constraints.txt \
        -e '/opt/sglang/python[srt_empty]'

FROM sglang AS dependencies
RUN --mount=type=cache,id=kunlunxin-pip,target=/root/.cache/pip \
    python -m pip install -c /opt/vendor-constraints.txt \
        pytest pytest-cov pytest-timeout==2.4.0 pytest-json-report \
        pytest-metadata requests ruff pyyaml && \
    python -m pip freeze > /opt/ci-requirements.freeze.txt

FROM dependencies AS ci
ENV USE_FLAGGEMS=1 \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/opt/FlagCX \
    SGLANG_FL_CONFIG=/opt/sglang-plugin-fl/sglang_fl/dispatch/config/kunlunxin.yaml \
    SGLANG_FL_PER_OP="silu_and_mul=flagos;mrotary_embedding=flagos;topk=reference;gemma_rms_norm=reference;fused_moe=vendor;chunk_gated_delta_rule=vendor;fused_recurrent_gated_delta_rule_packed_decode=vendor"
WORKDIR /opt/sglang-plugin-fl
COPY pyproject.toml ./
COPY sglang_fl/ ./sglang_fl/
RUN --mount=type=cache,id=kunlunxin-pip,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation -e . && \
    python -c "from importlib.metadata import version; assert version('sglang') == '0.5.18'; assert version('flagtree') == '0.7.1a1+xpu3.6'" && \
    test -f /opt/FlagCX/plugin/interservice/flagcx_wrapper.py
