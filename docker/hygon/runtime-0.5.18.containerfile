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
#
# Build from the repository root:
#   docker build -f docker/hygon/runtime-0.5.18.containerfile \
#     -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-runtime-ci-<date> .

# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04@sha256:03e598b348e58e7233628394bbefa0ccef26969a091c7e6f2f02835161757f8b
FROM ${BASE_IMAGE}

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

# Version lock for every SGLang closure addition (name==version), extracted
# from the reviewed resolver run. Passed as an extra --constraint so resolver
# drift across indexes cannot change the closure; the dry-run gate still
# verifies zero runtime replacements on top of it.
COPY docker/hygon/runtime-0.5.18.additions.lock /opt/sglang-ci/sglang-additions.lock

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
# for the vendor stack. Provenance matters because the runtime provides some
# modules out-of-band (pip cannot see FlagTree's triton), so a pip-level
# zero-drift check alone would miss a shadowing install.
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

# --- Resolver gate: dry-run with report before touching site-packages --------
# Any runtime distribution in the install plan fails the build: a different
# version is a hard conflict, an identical version is a reinstall that needs
# explicit review. New (absent) distributions are the SGLang closure; they are
# recorded for review but allowed.
RUN --mount=type=cache,target=/tmp/pip-cache \
    export PYTHON_BIN="$(command -v python || command -v python3)" \
    && dry_run_rc=1 \
    && for attempt in 1 2 3 4 5 6 7 8; do \
         if "$PYTHON_BIN" -m pip install \
             --dry-run \
             --timeout 120 \
             --retries 20 \
             --cache-dir /tmp/pip-cache \
             --report /opt/sglang-ci/sglang-resolver-plan.json \
             --constraint /opt/sglang-ci/runtime-constraints.txt \
             --constraint /opt/sglang-ci/sglang-additions.lock \
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
    && "$PYTHON_BIN" - <<'PY'
import json
import sys
from pathlib import Path
from packaging.utils import canonicalize_name

evidence = Path("/opt/sglang-ci")
before = json.loads((evidence / "runtime-distributions-before.json").read_text())
report = json.loads((evidence / "sglang-resolver-plan.json").read_text())

additions, same_version, conflicts = [], [], []
for item in report.get("install", []):
    metadata = item.get("metadata", {})
    name = canonicalize_name(metadata.get("name", ""))
    version = metadata.get("version", "")
    if name not in before:
        additions.append((name, version))
    elif version == before[name]:
        same_version.append((name, version))
    else:
        conflicts.append((name, before[name], version))

(evidence / "sglang-resolver-additions.txt").write_text(
    "\n".join(f"{name}=={version}" for name, version in sorted(additions)) + "\n"
)
print(f"resolver plans {len(additions)} new distributions "
      "(recorded in sglang-resolver-additions.txt)")
for name, version in sorted(additions):
    print(f"  + {name}=={version}")

failed = False
if same_version:
    failed = True
    print("FATAL: resolver plans to reinstall runtime distributions "
          "(same version) - explicit review required:", flush=True)
    for name, version in sorted(same_version):
        print(f"  = {name}=={version}")
if conflicts:
    failed = True
    print("FATAL: resolver plans to replace runtime distributions "
          "with different versions:", flush=True)
    for name, old, new in sorted(conflicts):
        print(f"  ! {name}: runtime={old} planned={new}")
if failed:
    sys.exit(1)
PY

# --- Stable, slow: the real install ------------------------------------------
# Same constraints and indexes as the reviewed dry-run plan.
RUN --mount=type=cache,target=/tmp/pip-cache \
    export PYTHON_BIN="$(command -v python || command -v python3)" \
    && install_rc=1 \
    && for attempt in 1 2 3 4 5 6 7 8; do \
         if "$PYTHON_BIN" -m pip install \
             --timeout 120 \
             --retries 20 \
             --cache-dir /tmp/pip-cache \
             --constraint /opt/sglang-ci/runtime-constraints.txt \
             --constraint /opt/sglang-ci/sglang-additions.lock \
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
    && [ "$install_rc" -eq 0 ]

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
planned = set(
    (evidence / "sglang-resolver-additions.txt").read_text().split()
)
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
    && echo 'pip check: no new violations (baseline preserved)'

# 3. Module provenance: every vendor module must still resolve to the exact
#    same origin. One narrow, loudly-reported exception: the SGLang closure
#    pulls a standalone triton wheel (xgrammar dependency) into site-packages,
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
