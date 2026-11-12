# SGLang 0.5.18 empty-mode image for sglang-plugin-FL on T-Head PPU (PPU-ZW810E).
#
# Built on the operations-owned T-Head PPU runtime.  The immutable runtime carries
# PPU SDK 2.1.0, vendor Torch 2.10.0+v0.1.0.ppu2.1.1, vendor
# torchvision/torchaudio (added by build-infra PR #1738, issue #1730),
# FlagTree 0.7.0+ppu3.6 (which supplies the active `triton` module from
# /opt/flagtree), FlagGems 5.4.0 and numpy 1.26.4.  This image must preserve
# every runtime distribution and add only SGLang 0.5.18 srt_empty, its
# application requirements and sglang-plugin-FL.  Platform-specific components
# absent from the runtime (PPU sglang-kernel, FlagCX) are NOT added: the
# hardware probes on the previous runtime proved the CI matrix
# (unit/functional/qwen3 4b_tp2 e2e) does not need them; re-adding any of them
# requires a fresh hardware failure as evidence.
#
# The vendor torchvision/torchaudio wheels currently fail to import on a
# device-less builder (torchvision::nms op not registered; libtorchaudio.so
# will not load).  The cause is not yet determined; on-device verification
# decides.  Until then the build-time contracts assert them only as installed
# distributions, and the on-device probe (CI setup) plus the
# unit/functional/e2e stages exercise the real imports.
#
# Layers are ordered so that frequently changing inputs come last, and every
# layer appends its added distributions to /opt/added-distributions.txt so the
# final layer can prove a closed set:
#   1. runtime (digest-pinned) + vendor snapshot + SGLang srt_empty (pinned
#      commit) + startup-failure-selected application extras + zero-drift and
#      additions-lock verification (docker/thead/install-sglang-0.5.18-empty.sh);
#   2. CI test toolchain;
#   3. sglang-plugin-FL (metadata + package tree only; CI re-installs the
#      plugin under test from the checkout at job time);
#   4. VERIFY_ONLY pass: runtime distributions unchanged, everything else
#      covered by lock + ledger, pip check at its reviewed baseline.
#
# Build from the repository root:
#   docker build -f docker/thead/empty-0.5.18.containerfile \
#     -t sglang-plugin-fl:thead-empty-0.5.18 .
# BASE_IMAGE is the runtime rebuilt by build-infra PR #1738.  Keep it
# digest-pinned.  If the default PIP_INDEX_URL (official PyPI) is slow from the
# build host, pass
#   --build-arg PIP_INDEX_URL=https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu130/simple/
#
# Run: PPU devices are passed with --device /dev/alixpu* (see .github/configs/thead.yml).
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0@sha256:163faf04cac90612954f32b0f45eef90223708fd3de0408497570ee4fd3b408d
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION=0.5.18
# pip index: official PyPI, as in docker/thead/containerfile (the base image's
# aiext mirror is PPU/ML-focused). Override with --build-arg PIP_INDEX_URL=<url>.
# Longer timeout and retries because files.pythonhosted.org can be slow from
# PPU build hosts.
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Runtime-owned paths that must exist in the base (the install script asserts
# them): FlagTree's triton module tree (/opt/flagtree, hence PYTHONPATH) and
# the PPU collective libraries (/opt/accl-p).  Same settings as the validated
# runtime-probe image.
ENV PYTHONPATH=/opt/flagtree \
    LD_LIBRARY_PATH=/opt/accl-p:${LD_LIBRARY_PATH}

# Layer 1: vendor snapshot + SGLang srt_empty (commit-pinned) + application
# extras + zero-drift and additions-lock verification.  The install script and
# its lock stay in the image for auditability.
COPY docker/thead/install-sglang-0.5.18-empty.sh /opt/install-sglang-0.5.18-empty.sh
COPY docker/thead/empty-0.5.18.additions.lock /opt/empty-0.5.18.additions.lock
RUN echo "pip index: ${PIP_INDEX_URL}" && \
    SGLANG_VERSION=${SGLANG_VERSION} SGLANG_SRC=/opt/sglang-${SGLANG_VERSION} \
    ADDITIONS_LOCK=/opt/empty-0.5.18.additions.lock \
    SKIP_DEVICE_CHECK=1 INSTALL_PLUGIN=0 \
    bash /opt/install-sglang-0.5.18-empty.sh && \
    pip cache purge

# Layer 2: test toolchain baked in so CI does not depend on PyPI at job time
# (.github/scripts/thead/setup.sh still re-installs the plugin under test).
# Record this layer's added distributions in the ledger.
RUN pip install --no-cache-dir pytest pytest-timeout pyyaml && \
    python - /opt/runtime-distributions.before /opt/added-distributions.txt <<'PY'
import importlib.metadata as m, sys
from pathlib import Path

def norm(name):
    return name.lower().replace("_", "-")

before = {norm(l.split("==", 1)[0]) for l in Path(sys.argv[1]).read_text().splitlines()
          if l and not l.startswith("#")}
ledger = Path(sys.argv[2])
known = {norm(l.split("==", 1)[0]) for l in ledger.read_text().splitlines()
         if l and not l.startswith("#")} if ledger.exists() else set()
with ledger.open("a") as fh:
    for d in m.distributions():
        name = d.metadata.get("Name")
        if name and norm(name) not in before and norm(name) not in known:
            fh.write(f"{name}=={d.version}\n")
            print("ledger +", name, d.version)
PY

# Layer 3 (most volatile): sglang-plugin-FL.  Only packaging metadata and the
# package tree enter the image; the plugin's dependencies are already present.
COPY pyproject.toml ./
COPY sglang_fl/ ./sglang_fl/
RUN pip install --no-deps --no-build-isolation -e . && \
    python -c "import importlib.metadata as m, sglang_fl; \
               print('sglang', m.version('sglang'), '| sglang_fl', m.version('sglang_fl'), \
                     'from', sglang_fl.__file__)" && \
    python - /opt/runtime-distributions.before /opt/added-distributions.txt <<'PY'
import importlib.metadata as m, sys
from pathlib import Path

def norm(name):
    return name.lower().replace("_", "-")

before = {norm(l.split("==", 1)[0]) for l in Path(sys.argv[1]).read_text().splitlines()
          if l and not l.startswith("#")}
ledger = Path(sys.argv[2])
known = {norm(l.split("==", 1)[0]) for l in ledger.read_text().splitlines()
         if l and not l.startswith("#")} if ledger.exists() else set()
with ledger.open("a") as fh:
    for d in m.distributions():
        name = d.metadata.get("Name")
        if name and norm(name) not in before and norm(name) not in known:
            fh.write(f"{name}=={d.version}\n")
            print("ledger +", name, d.version)
PY

# Layer 4: final closed-set verification.  The runtime snapshot must be
# unchanged and every other distribution must be covered by the committed
# additions lock plus the layer ledger; pip check must sit at its reviewed
# baseline (the single xgrammar/triton metadata exception).
RUN VERIFY_ONLY=1 SKIP_DEVICE_CHECK=1 \
    ADDITIONS_LOCK=/opt/empty-0.5.18.additions.lock \
    bash /opt/install-sglang-0.5.18-empty.sh

# Runtime defaults for the empty runtime (match the validated 0.5.11 empty env).
#   SGLANG_IS_FLASHINFER_AVAILABLE=false: FlashInfer is not installed; the plugin also
#     serves an import placeholder (vendor/thead/import_guard.py) for module-level imports.
#   Do not set SGLANG_FL_FLAGOS_BLACKLIST: it replaces the blacklist in
#     sglang_fl/dispatch/config/thead.yaml instead of extending it.
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false

# CI runs offline against the mounted model store (/data/models/Qwen, see
# .github/configs/thead.yml). HF_HOME is pinned because the runner overrides
# HOME=/github/home at job time. Same settings as docker/thead/containerfile.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /opt/sglang-plugin-fl
