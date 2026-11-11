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
