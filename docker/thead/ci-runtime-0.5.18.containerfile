# Minimal SGLang 0.5.18 empty-mode image for T-Head PPU CI.
#
# The operations runtime owns the complete platform stack: PPU SDK, vendor
# torch, FlagTree/Triton and FlagGems. This image adds only SGLang's
# device-independent srt_empty installation closure. The PR checkout installs
# sglang-plugin-FL and CI-only test tools at job time.
FROM harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0@sha256:ee5b2b9420dedc8eb6ad6658397e9d9602533f1c6e74a53b924ff976a113d4ab AS runtime

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/opt/accl-p:${LD_LIBRARY_PATH}

# Freeze every distribution already owned by the runtime. A dependency that
# requires changing one of these versions must fail resolution instead of
# silently upgrading, downgrading or replacing the operations stack.
RUN python - <<'PY'
from importlib.metadata import distributions
from pathlib import Path

site = "/flagos/lib/python3.12/site-packages"
versions = {}
for dist in distributions(path=[site]):
    name = dist.metadata.get("Name")
    if not name:
        continue
    previous = versions.setdefault(name, dist.version)
    if previous != dist.version:
        raise SystemExit(f"duplicate runtime distribution {name}: {previous}, {dist.version}")

Path("/opt/runtime-distributions.before").write_text(
    "".join(f"{name}=={versions[name]}\n" for name in sorted(versions, key=str.lower))
)
Path("/opt/runtime-constraints.txt").write_text(
    "# Runtime-owned distributions: versions must not change.\n"
    + "".join(f"{name}=={versions[name]}\n" for name in sorted(versions, key=str.lower))
)
PY

ARG SGLANG_VERSION=0.5.18
ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
RUN git init /opt/sglang && \
    git -C /opt/sglang remote add origin https://github.com/sgl-project/sglang.git && \
    git -C /opt/sglang fetch --depth 1 origin "${SGLANG_COMMIT}" && \
    git -C /opt/sglang checkout --detach FETCH_HEAD && \
    cp /opt/sglang/python/pyproject_other.toml /opt/sglang/python/pyproject.toml && \
    grep -q '^srt_empty' /opt/sglang/python/pyproject.toml

RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    SGLANG_BUILD_RUST_EXTS=none \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION} \
    python -m pip install --no-build-isolation \
        -c /opt/runtime-constraints.txt \
        -e '/opt/sglang/python[srt_empty]'

# Prove that the SGLang install preserved every runtime-owned distribution and
# record the exact added closure for image review.
RUN python - <<'PY'
from importlib.metadata import distributions
from pathlib import Path

site = "/flagos/lib/python3.12/site-packages"

def inventory():
    result = {}
    for dist in distributions(path=[site]):
        name = dist.metadata.get("Name")
        if name:
            result[name.lower()] = (name, dist.version)
    return result

before = {}
for line in Path("/opt/runtime-distributions.before").read_text().splitlines():
    name, version = line.rsplit("==", 1)
    before[name.lower()] = (name, version)
after = inventory()

drift = []
for key, (name, version) in before.items():
    if key not in after:
        drift.append(f"{name}=={version} -> REMOVED")
    elif after[key][1] != version:
        drift.append(f"{name}=={version} -> {after[key][0]}=={after[key][1]}")
if drift:
    raise SystemExit("runtime distribution drift:\n" + "\n".join(drift))

added = [after[key] for key in sorted(set(after) - set(before))]
Path("/opt/sglang-added-distributions.txt").write_text(
    "".join(f"{name}=={version}\n" for name, version in added)
)
print(f"runtime distributions preserved: {len(before)}")
print(f"SGLang closure additions: {len(added)}")
print(Path("/opt/sglang-added-distributions.txt").read_text())
PY

# SGLang's srt_empty metadata does not declare torchvision, but importing
# sglang.Engine unconditionally imports torchvision.io.decode_jpeg.  The public
# torchvision wheel is not ABI-compatible with the PPU vendor Torch, so build
# the matching release against the runtime-owned Torch.  Build tools and JPEG
# headers stay in this throw-away stage.  Keeping this after the SGLang closure
# also preserves the expensive dependency layer when iterating on this probe.
FROM runtime AS torchvision-builder
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHVISION_COMMIT=8ac84ee75afb1c327902156b5336f56ad63b7e2f
ARG BUILD_JOBS=4
RUN apt-get update && \
    apt-get install -y --no-install-recommends libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*
RUN git init /opt/vision && \
    git -C /opt/vision remote add origin https://github.com/pytorch/vision.git && \
    git -C /opt/vision fetch --depth 1 origin "${TORCHVISION_COMMIT}" && \
    git -C /opt/vision checkout --detach FETCH_HEAD
RUN --mount=type=cache,id=thead-pip,target=/root/.cache/pip \
    python -m pip install setuptools==69.5.1 wheel && \
    BUILD_VERSION="${TORCHVISION_VERSION}" MAX_JOBS="${BUILD_JOBS}" \
    TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_WEBP=0 \
        python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /opt/vision-wheels /opt/vision

FROM runtime AS sglang-runtime
COPY --from=torchvision-builder /opt/vision-wheels/torchvision-*.whl /tmp/
# The real plugin-activated Engine path also imports xgrammar and
# compressed-tensors unconditionally.  srt_empty intentionally omits both
# because their package metadata reaches into the device stack.  Install the
# SGLang 0.5.18-compatible versions without dependency resolution so xgrammar's
# declared standalone Triton and compressed-tensors' declared Torch cannot
# introduce second implementations beside the runtime-owned stack.  Loguru is
# compressed-tensors' only declared runtime dependency not already supplied by
# the srt_empty closure.
RUN export PYTHONPATH=/opt/flagtree && \
    python -m pip install --no-deps /tmp/torchvision-*.whl && \
    PIP_INDEX_URL=https://pypi.org/simple \
        python -m pip install --no-deps \
            xgrammar==0.2.1 \
            compressed-tensors==0.15.0 \
            loguru==0.7.3 && \
    rm -f /tmp/torchvision-*.whl && \
    python - <<'PY'
from importlib.metadata import distributions, version
from pathlib import Path
import triton
import xgrammar
from compressed_tensors.config import CompressionFormat
from torchvision.io import decode_jpeg

assert version("torchvision") == "0.25.0"
assert version("xgrammar") == "0.2.1"
assert version("compressed-tensors") == "0.15.0"
assert version("loguru") == "0.7.3"
assert CompressionFormat is not None
assert decode_jpeg is not None
assert Path(triton.__file__).resolve().is_relative_to(Path("/opt/flagtree"))

before = {}
for line in Path("/opt/runtime-distributions.before").read_text().splitlines():
    name, dist_version = line.rsplit("==", 1)
    before[name.lower()] = (name, dist_version)

after = {}
for dist in distributions(path=["/flagos/lib/python3.12/site-packages"]):
    name = dist.metadata.get("Name")
    if name:
        after[name.lower()] = (name, dist.version)

drift = []
for key, (name, dist_version) in before.items():
    if key not in after:
        drift.append(f"{name}=={dist_version} -> REMOVED")
    elif after[key][1] != dist_version:
        drift.append(f"{name}=={dist_version} -> {after[key][0]}=={after[key][1]}")
if drift:
    raise SystemExit("runtime distribution drift:\n" + "\n".join(drift))

added = [after[key] for key in sorted(set(after) - set(before))]
Path("/opt/sglang-added-distributions.txt").write_text(
    "".join(f"{name}=={dist_version}\n" for name, dist_version in added)
)
PY

# These are the exact import paths that failed in the preceding runtime probes.
# Keep them as build-time contracts so an image cannot be published while the
# Engine's eager import graph is incomplete.
RUN PYTHONPATH=/opt/sglang/python:/opt/flagtree \
    python -c "from sglang import Engine; from sglang.srt.layers import quantization; assert Engine is not None and quantization.QUANTIZATION_METHODS"

ENV PYTHONPATH=/opt/sglang/python:/opt/flagtree \
    SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
