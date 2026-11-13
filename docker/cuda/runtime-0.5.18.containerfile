# SGLang 0.5.18 CI image on the NVIDIA operations runtime (CUDA 13.3).
#
# Principles (do not weaken without review):
#   * The operations runtime is immutable: no runtime distribution may be
#     upgraded, downgraded, replaced, or removed. The resolver must only ADD
#     the SGLang dependency closure. Any planned change to an existing
#     distribution fails the build.
#   * NVIDIA installs the normal sglang==0.5.18 wheel and its complete,
#     version-locked standard dependency closure. Its CUDA compiler, CRT,
#     NVVM, and CCCL are pinned to CUDA 13.0 to match the Runtime's CUDA 13.0
#     headers and torch 2.13.0+cu130. Triton 3.7.1 stays in the image to
#     satisfy torch's exact metadata requirement; the ephemeral GPU job swaps
#     this application-added distribution to 3.6.0 because 3.7.1 mixed with
#     FlagTree's Triton 3.6 code crashed kernel launches. The Runtime's own
#     torch/CUDA/NCCL/FlagTree/FlagGems/FlagCX distributions are never touched.
#   * Raw-Runtime baseline finding, kept on record: torch's distribution
#     requires Triton while the Runtime provides the module out-of-band under
#     /opt/flagtree. The image preserves standard-closure Triton 3.7.1 for
#     metadata consistency; CI changes only that added distribution in its
#     disposable test container.
#   * No sglang-plugin-FL source in this image; CI installs the plugin from
#     the PR checkout via the standard setup flow.
#
# Build from the repository root:
#   docker build -f docker/cuda/runtime-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-nvidia-runtime-ci-<date> .

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:edf29b9a28badaf72fc75fd0e057325df9610694914cca3f571e587038bf17e8
FROM ${BASE_IMAGE}

ARG SGLANG_WHEEL_URL=https://files.pythonhosted.org/packages/48/02/8c15afa2bb3f4d060d4ff7a8cd66d35bfa233f94b41f02a2d59b439dbfde/sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl
ARG SGLANG_WHEEL_SHA256=fba7bb31a5de4c014945cbc50e9357c9253cc965a423736c324b2aaccd2270dc
ARG SGLANG_WHEEL_NAME=sglang-0.5.18-cp312-cp312-manylinux_2_34_x86_64.whl
# resource.flagos.net's pypi-proxy intermittently 504s large upstream fetches
# and pip does not retry HTTP 504; every pip call below loops instead, because
# pip's HTTP cache keeps whatever the previous attempt already fetched.
ARG PIP_MAIN_INDEX=https://resource.flagos.net/repository/flagos-pypi-nvidia/simple
# PyPI serves generic wheels at datacenter speed; the pypi-proxy mirrors
# PyPI byte-for-byte but its gateway times out delivering large wheels on
# transpacific links (sglang-kernel is 377 MB), so it stays the LAST index.
# The platform index keeps first priority for flagos-built wheels.
ARG PIP_FALLBACK_INDEX=https://pypi.org/simple
ARG PIP_EXTRA_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

WORKDIR /opt/sglang-ci

# Version lock for every SGLang closure addition (name==version), extracted
# from the reviewed resolver run. Passed as an extra --constraint so resolver
# drift across indexes cannot change the closure; the dry-run gate still
# verifies zero runtime replacements on top of it.
COPY docker/cuda/runtime-0.5.18.install.lock /opt/sglang-ci/sglang-install.lock

# --- Stable, slow: fetch and SHA256-verify the SGLang wheel ------------------
# Pure-Python downloader so the build depends only on the runtime's own Python.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && mkdir -p /opt/sglang-ci/wheels \
    && "$PYTHON_BIN" - <<'PY'
import hashlib
import os
import time
import urllib.request
from pathlib import Path

url = os.environ["SGLANG_WHEEL_URL"]
expected = os.environ["SGLANG_WHEEL_SHA256"]
dest = Path("/opt/sglang-ci/wheels") / os.environ["SGLANG_WHEEL_NAME"]

last_error = None
for attempt in range(1, 6):
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(dest, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        break
    except Exception as error:  # noqa: BLE001 - retry any transport failure
        last_error = error
        print(f"wheel fetch attempt {attempt} failed: {error!r}; retrying", flush=True)
        time.sleep(15)
else:
    raise SystemExit(f"could not fetch {url}: {last_error!r}")

digest = hashlib.sha256(dest.read_bytes()).hexdigest()
if digest != expected:
    raise SystemExit(
        f"SHA256 mismatch for {dest.name}: expected {expected}, got {digest}"
    )
print(f"verified {dest.name} sha256={digest}")
PY

# --- Stable: capture the pre-install runtime state ---------------------------
# Canonicalized distribution inventory, exact constraints for the resolver,
# the runtime's own pip check result (kept as the baseline so pre-existing
# runtime findings are not attributed to SGLang), and module-level provenance
# for the vendor stack. Provenance matters because the runtime provides the
# triton module out-of-band (pip cannot see it), so a pip-level zero-drift
# check alone would miss a shadowing install.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" -m pip freeze --all | sort > /opt/sglang-ci/runtime-freeze-before.txt \
    && { "$PYTHON_BIN" -m pip check > /opt/sglang-ci/runtime-pip-check-baseline.txt 2>&1; \
         printf '%s\n' "$?" > /opt/sglang-ci/runtime-pip-check-baseline.exitcode; true; } \
    && "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import importlib.util
import json
import os
import sys
from pathlib import Path
from packaging.utils import canonicalize_name

# Mirror the runtime's container-start module resolution: PYTHONPATH
# (/opt/flagtree) precedes site-packages there, while build RUN steps do not
# inherit that ENV. Same position, same precedence as at container start.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")

packages = {}
for dist in md.distributions():
    name = dist.metadata.get("Name")
    if name:
        packages[canonicalize_name(name)] = dist.version
with (evidence / "runtime-distributions-before.json").open("w") as f:
    json.dump(packages, f, indent=2, sort_keys=True)
with (evidence / "runtime-constraints.txt").open("w") as f:
    for name, version in sorted(packages.items()):
        f.write(f"{name}=={version}\n")

import hashlib
import sysconfig

# File-level baseline for the runtime's site-packages: every file's SHA256.
# The post-install gate requires every baseline file to still exist with the
# identical hash, so an in-place overwrite by any closure wheel is caught
# even when pip metadata (or an import origin) stays unchanged.
purelib = Path(sysconfig.get_paths()["purelib"])
manifest = {}
for path in sorted(purelib.rglob("*")):
    if path.is_file():
        manifest[str(path.relative_to(purelib))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
with (evidence / "site-files-before.sha256").open("w") as f:
    for relative, digest in manifest.items():
        f.write(f"{digest}  {relative}\n")
print(f"site-packages baseline: {len(manifest)} files hashed under {purelib}")

def probe_module(module):
    # find_spec resolves without executing the module: no device libraries
    # are touched, so the probe is meaningful on a GPU-less builder.
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}

flagtree = Path("/opt/flagtree")
provenance = {
    "_context": {
        "pythonpath": os.environ.get("PYTHONPATH"),
        "flagtree_dir_present": flagtree.is_dir(),
        "flagtree_entries": (
            sorted(entry.name for entry in flagtree.iterdir())
            if flagtree.is_dir()
            else None
        ),
    },
    **{
        module: probe_module(module)
        for module in ("torch", "triton", "flag_gems", "flagcx", "flagtree")
    },
}
with (evidence / "module-provenance-before.json").open("w") as f:
    json.dump(provenance, f, indent=2, sort_keys=True)
print(json.dumps(provenance, indent=2, sort_keys=True))
PY

# --- Lock gate: the install list must not touch any runtime distribution --
# Installing with --no-deps from the reviewed, version-locked standard closure
# (with the documented CUDA 13.0 toolchain constraints). This gate statically
# proves the lock cannot replace a Runtime distribution.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" - <<'PY'
import json
import sys
from pathlib import Path
from packaging.utils import canonicalize_name

evidence = Path("/opt/sglang-ci")
before = json.loads((evidence / "runtime-distributions-before.json").read_text())
lock = set()
for line in (evidence / "sglang-install.lock").read_text().splitlines():
    if line.strip():
        lock.add(canonicalize_name(line.split("==")[0]))
overlap = lock & set(before)
if overlap:
    print("FATAL: install lock overlaps runtime distributions: "
          + ", ".join(sorted(overlap)))
    sys.exit(1)
print(f"install lock: {len(lock)} packages, zero overlap with the "
      f"{len(before)} runtime distributions")
PY

# --- Stable, slow: the locked install ---------------------------------------
# --no-deps against exact pins: no resolution, deterministic regardless of
# index state. The sglang wheel comes from the SHA256-verified local file.
RUN --mount=type=cache,target=/tmp/pip-cache \
    export PYTHON_BIN="$(command -v python || command -v python3)" \
    && install_rc=1 \
    && for attempt in 1 2 3 4 5 6 7 8; do \
         if "$PYTHON_BIN" -m pip install \
             --no-deps \
             --timeout 120 \
             --retries 20 \
             --cache-dir /tmp/pip-cache \
             --index-url "$PIP_MAIN_INDEX" \
             --extra-index-url "$PIP_FALLBACK_INDEX" \
             --extra-index-url "$PIP_EXTRA_INDEX" \
             -r /opt/sglang-ci/sglang-install.lock \
             "/opt/sglang-ci/wheels/${SGLANG_WHEEL_NAME}"; then \
           install_rc=0; break; \
         else \
           install_rc="$?"; \
         fi; \
         echo "install attempt ${attempt} failed (rc=${install_rc}); retrying" >&2; \
         sleep 20; \
       done \
    && [ "$install_rc" -eq 0 ]

# The standard closure owns the CUDA development toolkit used by SGLang and
# FlashInfer JIT. Keep the operations Runtime's /usr/local/cuda tree immutable
# and activate the complete pip toolkit as one coherent root instead of
# linking a lone nvcc binary into Runtime-owned paths.
ENV CUDA_HOME=/flagos/lib/python3.12/site-packages/nvidia/cu13 \
    CUDA_PATH=/flagos/lib/python3.12/site-packages/nvidia/cu13
ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"

# Standard CUDA layout shim for the JIT link step: FlashInfer's ninja link
# line assumes $CUDA_HOME/lib64 plus an unversioned libcudart.so, while the
# pip cu13 tree uses lib/ and ships only the versioned libcudart.so.13
# ("ld: cannot find -lcudart", run 43389681469). Purely additive symlinks;
# no runtime-owned file is touched.
RUN ln -sfn lib "${CUDA_HOME}/lib64" \
    && ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so" \
    && test -e "${CUDA_HOME}/lib64/libcudart.so"

# A GPU-less build can still prove the full JIT contract: the CCCL version
# gate that aborted e2e compiles (cuda_toolkit.h compares the nvcc
# MAJOR.MINOR against the headers' CUDART_VERSION) and the -lcudart link
# resolution through the shim above.
RUN test -x "${CUDA_HOME}/bin/nvcc" \
    && test -f "${CUDA_HOME}/include/cuda_runtime.h" \
    && test -f "${CUDA_HOME}/include/cuda.h" \
    && FI_CCCL=/flagos/lib/python3.12/site-packages/flashinfer/data/cccl \
    && test -d "$FI_CCCL/libcudacxx" \
    && "${CUDA_HOME}/bin/nvcc" --version \
    && printf '%s\n' \
         '#include <cuda/std/__cccl/cuda_toolkit.h>' \
         '#include <cub/util_type.cuh>' \
         '__global__ void probe_kernel() {}' \
         'int main() { return 0; }' \
       | "${CUDA_HOME}/bin/nvcc" -x cu -std=c++17 \
           -I"$FI_CCCL/cub" -I"$FI_CCCL/libcudacxx/include" -I"$FI_CCCL/thrust" \
           -isystem "${CUDA_HOME}/include" \
           --compiler-options=-fPIC --expt-relaxed-constexpr \
           -gencode=arch=compute_80,code=sm_80 -c \
           -o /tmp/sglang-jit-toolchain-probe.o - \
    && test -s /tmp/sglang-jit-toolchain-probe.o \
    && "${CUDA_HOME}/bin/nvcc" -shared -o /tmp/sglang-jit-toolchain-probe.so \
           /tmp/sglang-jit-toolchain-probe.o \
           -L"${CUDA_HOME}/lib64" -lcudart \
    && test -s /tmp/sglang-jit-toolchain-probe.so \
    && rm -f /tmp/sglang-jit-toolchain-probe.o /tmp/sglang-jit-toolchain-probe.so

# --- Volatile verification (cheap layers, tuned without re-downloading) ------
# 1. Every runtime distribution survives with the exact same version and the
#    installed additions match the reviewed resolver plan.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
from packaging.utils import canonicalize_name

# Same container-start resolution mirroring as the baseline probe above:
# the runtime's flagtree dist-info lives under /opt/flagtree, which is on
# sys.path via PYTHONPATH at container start.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")
before = json.loads((evidence / "runtime-distributions-before.json").read_text())
after = {}
for dist in md.distributions():
    name = dist.metadata.get("Name")
    if name:
        after[canonicalize_name(name)] = dist.version
with (evidence / "runtime-distributions-after.json").open("w") as f:
    json.dump(after, f, indent=2, sort_keys=True)
with (evidence / "runtime-freeze-after.txt").open("w") as f:
    for name in sorted(after):
        f.write(f"{name}=={after[name]}\n")

drift = {n: (v, after.get(n)) for n, v in before.items() if after.get(n) != v}
removed = sorted(set(before) - set(after))
added = sorted(set(after) - set(before))
failed = False
(evidence / "sglang-additions-installed.txt").write_text(
    "\n".join(f"{name}=={after[name]}" for name in added) + "\n"
)
planned = {
    f"{canonicalize_name(line.split('==')[0])}=={line.split('==', 1)[1]}"
    for line in (evidence / "sglang-install.lock").read_text().splitlines()
    if line.strip()
} | {f"sglang=={after.get('sglang', '')}"}
installed = set(f"{name}=={after[name]}" for name in added)
print(f"{len(added)} distributions added; runtime baseline size {len(before)}")
if installed != planned:
    failed = True
    print("FATAL: installed additions differ from the reviewed resolver plan:")
    for line in sorted(installed - planned):
        print(f"  + {line}")
    for line in sorted(planned - installed):
        print(f"  - {line}")
if drift:
    failed = True
    print("FATAL: runtime distribution versions changed:")
    for name, (old, new) in sorted(drift.items()):
        print(f"  ! {name}: {old} -> {new}")
if removed:
    failed = True
    print("FATAL: runtime distributions removed: " + ", ".join(removed))
sys.exit(1 if failed else 0)
PY

# 1b. File-level zero-drift: every baseline site-packages file must still
#     exist with the identical SHA256 (catches in-place overwrites).
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" - <<'PY'
import hashlib
import sysconfig
import sys
from pathlib import Path

evidence = Path("/opt/sglang-ci")
purelib = Path(sysconfig.get_paths()["purelib"])
baseline = {}
for line in (evidence / "site-files-before.sha256").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    baseline[relative] = digest

missing = []
changed = []
for relative, expected in baseline.items():
    path = purelib / relative
    if not path.is_file():
        missing.append(relative)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        changed.append(relative)
if missing or changed:
    print(f"FATAL: site-packages baseline drifted: {len(missing)} missing, "
          f"{len(changed)} content-changed")
    for item in missing[:20]:
        print(f"  - missing: {item}")
    for item in changed[:20]:
        print(f"  ! changed: {item}")
    sys.exit(1)
print(f"site-packages file baseline unchanged "
      f"({len(baseline)} files re-hashed identical)")
PY

# 2. pip check: only violations NEW relative to the runtime baseline fail the
#    build; the runtime's own pre-existing findings stay on record.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && { "$PYTHON_BIN" -m pip check > /opt/sglang-ci/runtime-pip-check-after.txt 2>&1; \
         printf '%s\n' "$?" > /opt/sglang-ci/runtime-pip-check-after.exitcode; true; } \
    && grep -v '^No broken requirements found\.$' /opt/sglang-ci/runtime-pip-check-baseline.txt | sort -u > /tmp/pip-check-baseline.lines \
    && grep -v '^No broken requirements found\.$' /opt/sglang-ci/runtime-pip-check-after.txt | sort -u > /tmp/pip-check-after.lines \
    && comm -13 /tmp/pip-check-baseline.lines /tmp/pip-check-after.lines > /opt/sglang-ci/pip-check-new-violations.txt \
    && if [ -s /opt/sglang-ci/pip-check-new-violations.txt ]; then \
         echo 'FATAL: new pip check violations introduced by the SGLang install:'; \
         cat /opt/sglang-ci/pip-check-new-violations.txt; \
         exit 1; \
       fi \
    && echo 'pip check: no unexpected new violations (baseline preserved)'

# 3. Module provenance: every vendor module must still resolve to the exact
#    same origin. The SGLang closure adds standalone Triton 3.7.1 (xgrammar
#    and humming-kernels dependency) to site-packages,
#    while the runtime's authoritative triton module lives in the /opt/flagtree
#    volume, which Docker only populates at container start (VOLUME copy-up),
#    never during build RUN steps. At container start PYTHONPATH=/opt/flagtree
#    still precedes site-packages, so FlagTree's triton keeps winning; the
#    transition is allowed only under exactly those recorded conditions and
#    CI job-time checks assert the /opt/flagtree resolution.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

# Same container-start resolution mirroring as the baseline probe above.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")

def probe_module(module):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}

after = {
    module: probe_module(module)
    for module in ("torch", "triton", "flag_gems", "flagcx", "flagtree")
}
with (evidence / "module-provenance-after.json").open("w") as f:
    json.dump(after, f, indent=2, sort_keys=True)

recorded = json.loads((evidence / "module-provenance-before.json").read_text())
context = recorded.pop("_context")
before = recorded

changed = {
    module: (before.get(module), after.get(module))
    for module in sorted(set(before) | set(after))
    if before.get(module) != after.get(module)
}
if changed:
    print("FATAL: vendor module provenance changed across the install:")
    for module, (was, now) in changed.items():
        print(f"  ! {module}: {was} -> {now}")
    print("--- recorded build context ---")
    print(json.dumps(context, indent=2, sort_keys=True))
    sys.exit(1)
print("module provenance unchanged under container-start resolution order "
      "(triton resolves via /opt/flagtree before site-packages):")
print(json.dumps(after, indent=2, sort_keys=True))
PY

# 4. Import smoke check for sglang itself. A GPU-less builder cannot load
#    device libraries (libgalaxyhip / libcuda / torch device backends), so
#    those failures are recorded and tolerated; any other failure - notably
#    a missing pure-Python dependency of the closure - fails the build.
RUN export PYTHON_BIN="$(command -v python || command -v python3)" \
    && "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import traceback

try:
    import sglang
    print("sglang", md.version("sglang"), "imported from", sglang.__file__)
except BaseException:  # noqa: BLE001 - classify below
    text = traceback.format_exc()
    print(text)
    tolerated = (
        "libgalaxyhip" in text,
        "libcuda.so" in text,
        "Failed to load the backend extension" in text,
        "cannot open shared object file" in text,
    )
    if not any(tolerated):
        raise SystemExit(
            "sglang import failed for reasons other than missing device libraries"
        )
    print("sglang import blocked by device libraries only "
          "(expected on a GPU-less builder)")
PY
