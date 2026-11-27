#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Post-build acceptance for the Hygon two-wheel CI image. Nothing here
# runs during image construction; every phase is a docker run against the
# finished image, and the reference data is collected dynamically from the
# base digest the image itself records in its org.flagos.ci.base-digest OCI
# label (no committed snapshots, nothing COPYed into the image - baselines
# live in this script's mktemp dir and die with it).
#
# A non-zero exit means the candidate image must NOT be pushed.
#
# Phases:
#   0. collect the Runtime reference (freeze, site-file hashes, pip-check
#      baseline, provenance) from the labeled base digest
#   1. runtime immutability: distributions exact, site files unchanged,
#      pip-check only the documented (xgrammar, triton) exception,
#      compiler slots + no standalone triton + Runtime Bash slot resolution,
#      two-wheel contract, HCU compat markers   [verify_runtime_immutable]
#   2. vendor module provenance vs the base reference
#   3. kernel-wheel stub import smoke
#   4. FlagTree's real HCU compile path (its own make_amdgcn/make_hsaco)
#      through the Runtime clang and DTK device libraries, GPU-less
#   5. SGLang import smoke (sentinel verdict from the log)
#   6. Triton slot contract for the slot the image declares + entrypoint
#      matrix (sh, bash, child process) + no resolver-installed triton dist
#   7. installed-wheel resolution contract from an empty working
#      directory (parent and child process; full import deferred to HCU CI)
#   8. Runtime Bash environment/label contract: no application overrides,
#      vendor.sh/zz-flagos.sh exports, the provenance labels, the
#      declared resolver mode and candidate contract, the Runtime's complete
#      parent layers, every installed wheel compared member by member with
#      the staged artifact (the whole closure for an offline-wheelhouse
#      candidate), the neutral WORKDIR, and the absence of any CI working
#      directory or build residue
#
# Usage: verify_image.sh <image>

set -euo pipefail

# Usage:
#   verify_image.sh <image>
#   verify_image.sh <image> --sglang-wheel-sha256 <hex> --sglang-provenance <json> \
#       --xgrammar-wheel-sha256 <hex> --xgrammar-provenance <json> \
#       --sglang-fl-wheel <whl> --sglang-fl-wheel-sha256 <hex> \
#       --sglang-fl-provenance <json> \
#       [--sglang-wheel <whl> --xgrammar-wheel <whl>] \
#       [--wheelhouse-manifest <json>]
#
# The strict form is what build_two_wheel.sh runs: the expected wheel
# digests and provenance files bind the acceptance to the exact artifact
# content (patched-file manifests for the derived wheels; the wheel itself
# for sglang_fl, whose installed files are compared member by member).
# --sglang-wheel / --xgrammar-wheel add the same member-by-member comparison
# for the two derived wheels: the whole installed payload, native libraries
# included, must be the staged wheel. --wheelhouse-manifest is required for
# an offline-wheelhouse candidate and rejected otherwise: its whole closure
# is verified against it. Without the strict arguments only the weaker
# identity gate (label present + installed version) runs, and the report
# says so.
image="${1:?usage: verify_image.sh <image> [strict artifact arguments]}"
shift || true
expected_sglang_sha256=""
sglang_provenance=""
sglang_wheel=""
expected_xgrammar_sha256=""
xgrammar_provenance=""
xgrammar_wheel=""
sglang_fl_wheel=""
expected_sglang_fl_sha256=""
sglang_fl_provenance=""
wheelhouse_manifest=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --sglang-wheel-sha256) expected_sglang_sha256="${2:?}"; shift 2 ;;
        --sglang-provenance) sglang_provenance="${2:?}"; shift 2 ;;
        --sglang-wheel) sglang_wheel="${2:?}"; shift 2 ;;
        --xgrammar-wheel-sha256) expected_xgrammar_sha256="${2:?}"; shift 2 ;;
        --xgrammar-provenance) xgrammar_provenance="${2:?}"; shift 2 ;;
        --xgrammar-wheel) xgrammar_wheel="${2:?}"; shift 2 ;;
        --sglang-fl-wheel) sglang_fl_wheel="${2:?}"; shift 2 ;;
        --sglang-fl-wheel-sha256) expected_sglang_fl_sha256="${2:?}"; shift 2 ;;
        --sglang-fl-provenance) sglang_fl_provenance="${2:?}"; shift 2 ;;
        --wheelhouse-manifest) wheelhouse_manifest="${2:?}"; shift 2 ;;
        *) echo "FATAL: unknown argument $1" >&2; exit 1 ;;
    esac
done
# A staged wheel is compared under the digest its label names.
if [ -n "$sglang_wheel" ] && [ -z "$expected_sglang_sha256" ] \
    || [ -n "$xgrammar_wheel" ] && [ -z "$expected_xgrammar_sha256" ]; then
    echo "FATAL: --sglang-wheel/--xgrammar-wheel must come with the expected wheel digest" >&2
    exit 1
fi
plugin_args=0
for value in "$sglang_fl_wheel" "$expected_sglang_fl_sha256" "$sglang_fl_provenance"; do
    [ -n "$value" ] && plugin_args=$((plugin_args + 1))
done
if [ "$plugin_args" -ne 0 ] && [ "$plugin_args" -ne 3 ]; then
    echo "FATAL: --sglang-fl-wheel, --sglang-fl-wheel-sha256 and --sglang-fl-provenance must be given together" >&2
    exit 1
fi
if [ -n "$expected_sglang_sha256" ] && [ -z "$sglang_provenance" ] \
    || [ -z "$expected_sglang_sha256" ] && [ -n "$sglang_provenance" ]; then
    echo "FATAL: --sglang-wheel-sha256 and --sglang-provenance must be given together" >&2
    exit 1
fi
if [ -n "$expected_xgrammar_sha256" ] && [ -z "$xgrammar_provenance" ] \
    || [ -z "$expected_xgrammar_sha256" ] && [ -n "$xgrammar_provenance" ]; then
    echo "FATAL: --xgrammar-wheel-sha256 and --xgrammar-provenance must be given together" >&2
    exit 1
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$here"
repo_root="$(cd "$here/../../.." && pwd)"
DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"
DERIVED_XGRAMMAR_VERSION="0.2.1+flagos.hcu1"

# The acceptance binds to the image itself: the base digest comes from the
# OCI label build_two_wheel.sh bakes in, never from the checkout.
BASE_IMAGE="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.base-digest"}}' "$image")"
[ -n "$BASE_IMAGE" ] && [ "$BASE_IMAGE" != "<no value>" ] || {
    echo "FATAL: $image lacks the org.flagos.ci.base-digest label" >&2
    exit 1
}

# The compiler slot the image selects (org.flagos.ci.triton-slot; the A/B
# experiment derives /opt/triton images from an accepted candidate). Every
# slot is verified against its own contract; the production default is
# /opt/flagtree.
TRITON_SLOT="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.triton-slot"}}' "$image")"
case "$TRITON_SLOT" in
    /opt/flagtree|/opt/triton) ;;
    *)
        echo "FATAL: $image lacks a valid org.flagos.ci.triton-slot label (got '$TRITON_SLOT')" >&2
        exit 1
        ;;
esac

# --- Resolver mode and candidate contract --------------------------------------
# Every image says how its closure was resolved (org.flagos.ci.resolver-mode)
# and which local Runtime object it was built from. The offline-wheelhouse
# mode is a DEVELOPMENT CANDIDATE by construction: it must declare itself
# release-ready=false and carry the identity of the resolver cache its
# wheelhouse was derived from; its closure is compared with the very
# manifest the build consumed (phase 8).
label() { docker image inspect --format "{{index .Config.Labels \"$1\"}}" "$image"; }
RESOLVER_MODE="$(label org.flagos.ci.resolver-mode)"
case "$RESOLVER_MODE" in
    index)
        [ -z "$wheelhouse_manifest" ] || {
            echo "FATAL: --wheelhouse-manifest was given for an image resolved from the indexes (mode $RESOLVER_MODE)" >&2
            exit 1
        }
        ;;
    offline-wheelhouse)
        [ -s "$wheelhouse_manifest" ] || {
            echo "FATAL: an offline-wheelhouse candidate needs --wheelhouse-manifest: its closure is verified against it" >&2
            exit 1
        }
        [ "$(label org.flagos.ci.release-ready)" = "false" ] || {
            echo "FATAL: an offline-wheelhouse candidate must carry org.flagos.ci.release-ready=false" >&2
            exit 1
        }
        # the labels must name the build inputs the verifier holds
        expected_cache_source="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cache_source"]["image"])' "$wheelhouse_manifest")"
        [ "$(label org.flagos.ci.resolver-cache-source)" = "$expected_cache_source" ] || {
            echo "FATAL: org.flagos.ci.resolver-cache-source is not the manifest's cache source ($expected_cache_source)" >&2
            exit 1
        }
        [ "$(label org.flagos.ci.wheelhouse-manifest-sha256)" = "$(sha256sum "$wheelhouse_manifest" | awk '{print $1}')" ] || {
            echo "FATAL: org.flagos.ci.wheelhouse-manifest-sha256 does not match $wheelhouse_manifest" >&2
            exit 1
        }
        ;;
    *)
        echo "FATAL: $image lacks a valid org.flagos.ci.resolver-mode label (got '$RESOLVER_MODE'; build it with build_two_wheel.sh)" >&2
        exit 1
        ;;
esac
BASE_LOCAL_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
[ "$(label org.flagos.ci.base-local-image-id)" = "$BASE_LOCAL_IMAGE_ID" ] || {
    echo "FATAL: org.flagos.ci.base-local-image-id ($(label org.flagos.ci.base-local-image-id)) is not the local object of $BASE_IMAGE ($BASE_LOCAL_IMAGE_ID)" >&2
    exit 1
}
# The image is built FROM the declared base, so its RootFS must start with
# every layer of that base: the candidate carries the complete Runtime, and
# a push ships all of it.
base_layers="$(docker image inspect --format '{{json .RootFS.Layers}}' "$BASE_IMAGE")"
image_layers="$(docker image inspect --format '{{json .RootFS.Layers}}' "$image")"
python3 - "$base_layers" "$image_layers" <<'PY'
import json
import sys

base, image = (json.loads(argument) for argument in sys.argv[1:3])
if image[: len(base)] != base:
    raise SystemExit(
        f"FATAL: the image does not carry the complete Runtime base: "
        f"{len(base)} base layers expected as the RootFS prefix, "
        f"{len(image)} layers found"
    )
print(f"parent layers: all {len(base)} Runtime base layers lead the "
      f"{len(image)} image layers")
PY

# The required (prepended) DTK loader prefix and whatever the base image
# itself carries; both feed the phase-8 probe.
BASE_WORKDIR="$(docker image inspect --format '{{.Config.WorkingDir}}' "$BASE_IMAGE")"

refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-verify.XXXXXX")"
trap 'rm -rf "$refdir"' EXIT

# -i: several phases feed their program to the container over stdin.
run() { docker run --rm -i --platform linux/amd64 "$@"; }

echo "=== [0/8] Collecting the Runtime reference from the labeled base digest ==="
echo "    base (org.flagos.ci.base-digest): $BASE_IMAGE"
run --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$BASE_IMAGE" -c 'exec python "$@"' bash \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$BASE_IMAGE" \
    --selected-slot "$TRITON_SLOT"

echo "=== [1/8] Runtime immutability (six gates) ==="
run --volume "$refdir":/evidence:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" -c 'exec python "$@"' bash \
    /scripts/verify_runtime_immutable.py --evidence-dir /evidence

echo "=== [2/8] Vendor module provenance ==="
run --volume "$refdir":/evidence:ro \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" -c 'exec python "$@"' bash \
    /scripts/verify_module_provenance.py \
    --reference /evidence/runtime-0.5.18.provenance.json

echo "=== [3/8] Plugin-owned kernel import-guard smoke ==="
run --entrypoint bash "$image" -c 'exec python -' <<'PY'
import importlib.metadata as md
import importlib.util
import pathlib

# Load the guard file without importing sglang_fl/__init__.py: the no-device
# host cannot initialize DTK torch, while the guard itself deliberately has no
# torch dependency. HCU activation timing remains a hardware-matrix gate.
dist = md.distribution("sglang-fl")
root = pathlib.Path(dist.locate_file("")).resolve()
guard_path = root / "sglang_fl/dispatch/backends/vendor/hygon/import_guard.py"
spec = importlib.util.spec_from_file_location("hygon_import_guard_acceptance", guard_path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
active = guard.install_import_guards()
assert active == ["sgl_kernel", "flashinfer"], active

from sgl_kernel.kvcacheio import save_kv_cache
from flashinfer.sampling import top_k_renorm_prob

import sgl_kernel
import sgl_kernel.utils
import flashinfer

assert sgl_kernel.utils.is_arch_support_pdl() is False
assert importlib.util.find_spec("sgl_kernel.scalar_type") is None
try:
    md.version("sglang-kernel")
except md.PackageNotFoundError:
    pass
else:
    raise AssertionError("placeholder fabricated sglang-kernel metadata")
try:
    save_kv_cache()
except RuntimeError as e:
    assert "sgl_kernel is not installed on HCU" in str(e)
try:
    top_k_renorm_prob()
except RuntimeError as e:
    assert "flashinfer is not installed on HCU" in str(e)
assert guard._SKIP_VERSION_CHECK in __import__("os").environ
print("plugin-owned kernel import guard semantics: OK")
PY

echo "=== [4/8] FlagTree HCU compile path (GPU-less, FlagTree's own stage functions) ==="
# The probe drives FlagTree's make_amdgcn/make_hsaco with FlagTree's default
# configuration: the real per-kernel code path, real clang argv, no wrapper.
run --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/flagtree_compile_probe.py \
    --probe /scripts/hcu-device-lib-check.ll

echo "=== [5/8] SGLang import smoke ==="
# The verdict is a log sentinel (scripts/sglang_import_smoke.py): a GPU-less
# builder cannot initialize the device libraries, so the exit code says
# nothing. With the DTK loader path baked into the image the import may die
# inside the C++ runtime ("terminate called ... no ROCm-capable device is
# detected"); torch's terminate handler then needs tens of minutes on an
# emulated builder to symbolize a stack trace nobody reads. The container is
# therefore started detached and stopped once its log holds a FINAL verdict:
# the Python verdict (SMOKE-*) or that abort signature, after which the
# process cannot continue. No verdict within the bound fails the phase.
smoke_log="$refdir/sglang-import-smoke.log"
smoke_verdict_in_log() {
    grep -qE "SMOKE-PASS|SMOKE-TOLERATED" "$smoke_log" \
        || { grep -q "^terminate called after throwing an instance of 'c10::AcceleratorError'$" "$smoke_log" \
            && grep -q "^  what():  HIP error: no ROCm-capable device is detected$" "$smoke_log"; }
}
smoke_container="$(docker run -d --platform linux/amd64 \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/sglang_import_smoke.py)"
smoke_verdict=""
for _ in $(seq 1 180); do
    docker logs "$smoke_container" > "$smoke_log" 2>&1 || true
    if smoke_verdict_in_log; then
        smoke_verdict="sentinel"
        break
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$smoke_container")" != "true" ]; then
        smoke_verdict="exited"
        break
    fi
    sleep 10
done
docker logs "$smoke_container" > "$smoke_log" 2>&1 || true
docker rm -f "$smoke_container" >/dev/null 2>&1 || true
cat "$smoke_log"
case "$smoke_verdict" in
    sentinel)
        echo "SGLang import smoke: final verdict read from the log (the C++ runtime"
        echo "aborted for want of a device, as expected on a GPU-less builder)"
        ;;
    exited)
        smoke_verdict_in_log \
            || { echo "FATAL: sglang import failed for reasons other than missing device libraries" >&2; exit 1; }
        ;;
    *)
        echo "FATAL: the SGLang import smoke produced no verdict within 30 minutes" >&2
        exit 1
        ;;
esac

echo "=== [6/8] Triton slot contract + entrypoint matrix (slot: $TRITON_SLOT) ==="
run -e TRITON_SLOT="$TRITON_SLOT" \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$image" <<'SLOT_EOF'
python -c 'import os, triton; p = triton.__file__; assert p and p.startswith(os.environ["TRITON_SLOT"] + "/"), p; print("triton (sh):", p)' \
&& bash -c 'python -c "import os, triton; assert triton.__file__.startswith(os.environ[\"TRITON_SLOT\"] + \"/\"), triton.__file__; print(\"triton (bash):\", triton.__file__)"' \
&& python - "$TRITON_SLOT" <<'PY'
import importlib.metadata as md
import pathlib
import sys
import sysconfig

sys.path.insert(0, "/scripts")
from triton_slot import assert_slot_root, contract_of, dist_info_version

slot = sys.argv[1]
observed = assert_slot_root(slot)
contract = contract_of(slot)
# the resolver must never install a standalone triton; the declared slot's
# own distribution metadata is the only triton a check may see
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
assert not list(purelib.glob("triton-*.dist-info")), (
    "standalone triton distribution installed into site-packages"
)
try:
    version = md.version("triton")
except md.PackageNotFoundError:
    version = None
if contract["pip_check_sees_triton"]:
    assert version == dist_info_version(contract["triton_dist_info"]), version
else:
    assert version is None, f"unexpected triton distribution {version!r}"
print(f"triton slot {slot}: module {observed['module_version']}, backends "
      f"{observed['backends']}, distribution metadata {version or 'absent'}; "
      "no resolver-installed triton")
PY
SLOT_EOF

echo "=== [7/8] Installed-wheel resolution contract (empty cwd, no shadowing) ==="
run --entrypoint bash "$image" <<'EOF'
mkdir -p /tmp/empty-cwd && cd /tmp/empty-cwd \
&& python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("sglang_fl")
assert spec is not None and spec.origin, "sglang_fl not resolvable"
assert "site-packages" in spec.origin, f"resolves outside site-packages: {spec.origin}"
print("parent process:", spec.origin)
PY
python - <<'PY'
import importlib.util
import subprocess
import sys

child = subprocess.run(
    [sys.executable, "-c",
     "import importlib.util; s = importlib.util.find_spec('sglang_fl');"
     "print(s.origin if s else None)"],
    capture_output=True, text=True,
)
origin = child.stdout.strip()
assert child.returncode == 0 and "site-packages" in origin, origin + child.stderr
print("child process:", origin)
PY
EOF

echo "=== [8/8] Runtime Bash environment/label contract ==="

# Provenance labels: the base digest is consumed above; the installed
# SGLang must be the DERIVED HCU wheel - the reviewed gating patch folded
# into the pinned flagos wheel BEFORE assembly (the former Dockerfile
# patch + patch-label gate).
labeled_source_commit="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.source-commit"}}' "$image")"
[ -n "$labeled_source_commit" ] && [ "$labeled_source_commit" != "<no value>" ] || {
    echo "FATAL: $image lacks the org.flagos.ci.source-commit label" >&2
    exit 1
}
labeled_wheel_sha256="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.sglang-wheel-sha256"}}' "$image")"
[ -n "$labeled_wheel_sha256" ] && [ "$labeled_wheel_sha256" != "<no value>" ] || {
    echo "FATAL: $image lacks the org.flagos.ci.sglang-wheel-sha256 label" >&2
    exit 1
}
if [ -n "$expected_sglang_sha256" ]; then
    # STRICT CONTENT BINDING (the form build_two_wheel.sh runs): the OCI
    # label must equal the digest of the exact staged derived wheel, the
    # provenance manifest must describe that same artifact, and every
    # patched file inside the image must match the manifest byte-for-byte.
    # A same-version, different-content wheel cannot pass this gate.
    [ "$labeled_wheel_sha256" = "$expected_sglang_sha256" ] || {
        echo "FATAL: org.flagos.ci.sglang-wheel-sha256=$labeled_wheel_sha256 != expected $expected_sglang_sha256" >&2
        exit 1
    }
    [ -s "$sglang_provenance" ] || {
        echo "FATAL: provenance manifest not found: $sglang_provenance" >&2
        exit 1
    }
    # The artifact is bound by its SHA256 above. It may have been derived -
    # or published - at an earlier commit than this image, so the commit its
    # provenance records must be part of the image's history, not equal to it.
    python3 "$scripts/provenance_lineage.py" "$sglang_provenance" \
        "$labeled_source_commit" --repo "$repo_root"
    python3 - "$sglang_provenance" "$expected_sglang_sha256" \
        "$DERIVED_SGLANG_VERSION" <<'PY'
import json
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert provenance["derived_sha256"] == sys.argv[2], (
    f"provenance derived_sha256 {provenance['derived_sha256']} != expected")
assert provenance["derived_version"] == sys.argv[3], provenance["derived_version"]
assert provenance["patched_files_manifest"], "provenance carries no manifest"
PY
    run -e DERIVED_SGLANG_VERSION="$DERIVED_SGLANG_VERSION" \
        --volume "$sglang_provenance":/provenance.json:ro \
        --entrypoint bash "$image" -c 'exec python -' <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import sysconfig

version = md.version("sglang")
expected = os.environ["DERIVED_SGLANG_VERSION"]
assert version == expected, f"installed sglang is {version!r}, expected {expected!r}"
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
dist_infos = sorted(p.name for p in purelib.glob("sglang-*.dist-info"))
assert dist_infos == [f"sglang-{version}.dist-info"], dist_infos
manifest = json.loads(pathlib.Path("/provenance.json").read_text())[
    "patched_files_manifest"]
# the manifest keys are site-packages relative (they start with "sglang/")
site = purelib
mismatches = []
for rel, expected_hash in sorted(manifest.items()):
    installed = hashlib.sha256((site / rel).read_bytes()).hexdigest()
    if installed != expected_hash:
        mismatches.append(rel)
assert not mismatches, f"installed patched files differ from the manifest: {mismatches}"
print(f"installed sglang distribution: {version} ({dist_infos[0]}); "
      f"{len(manifest)} patched files match the derived-wheel manifest")
PY
else
    echo "WARNING: no --sglang-wheel-sha256/--sglang-provenance given;"
    echo "running the weaker identity gate only (label present + installed"
    echo "version + dist-info directory); content binding was NOT checked"
    run -e DERIVED_SGLANG_VERSION="$DERIVED_SGLANG_VERSION" \
        --entrypoint bash "$image" -c 'exec python -' <<'PY'
import importlib.metadata as md
import os
import pathlib
import sysconfig

version = md.version("sglang")
expected = os.environ["DERIVED_SGLANG_VERSION"]
assert version == expected, f"installed sglang is {version!r}, expected {expected!r}"
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
dist_infos = sorted(p.name for p in purelib.glob("sglang-*.dist-info"))
assert dist_infos == [f"sglang-{version}.dist-info"], dist_infos
    print("installed sglang distribution:", version, "(", dist_infos[0], ")")
PY
fi
# Beyond the patched files: the WHOLE derived wheel must be installed as
# is - every other member, native payload included, byte for byte.
if [ -n "$sglang_wheel" ]; then
    [ "$(sha256sum "$sglang_wheel" | awk '{print $1}')" = "$labeled_wheel_sha256" ] || {
        echo "FATAL: staged sglang wheel does not hash to the labeled digest $labeled_wheel_sha256" >&2
        exit 1
    }
    run --volume "$sglang_wheel":/wheel.whl:ro \
        --volume "$scripts":/scripts:ro \
        --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/verify_installed_wheel.py \
        --wheel /wheel.whl --own-tree sglang
else
    echo "WARNING: no --sglang-wheel given; only the patched-file manifest was"
    echo "compared, the rest of the installed SGLang payload was NOT"
fi

# The corrected xgrammar artifact follows the same content-binding contract.
# RECORD itself is installer-owned (uv may add INSTALLER/direct_url entries),
# so every other wheel member, including METADATA/WHEEL and all native payload,
# must match the staged provenance byte-for-byte.
labeled_xgrammar_sha256="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.xgrammar-wheel-sha256"}}' "$image")"
[ -n "$labeled_xgrammar_sha256" ] && [ "$labeled_xgrammar_sha256" != "<no value>" ] || {
    echo "FATAL: $image lacks the org.flagos.ci.xgrammar-wheel-sha256 label" >&2
    exit 1
}
if [ -n "$expected_xgrammar_sha256" ]; then
    [ "$labeled_xgrammar_sha256" = "$expected_xgrammar_sha256" ] || {
        echo "FATAL: org.flagos.ci.xgrammar-wheel-sha256=$labeled_xgrammar_sha256 != expected $expected_xgrammar_sha256" >&2
        exit 1
    }
    [ -s "$xgrammar_provenance" ] || {
        echo "FATAL: xgrammar provenance not found: $xgrammar_provenance" >&2
        exit 1
    }
    # same lineage rule as the SGLang wheel: derived no later than the image
    python3 "$scripts/provenance_lineage.py" "$xgrammar_provenance" \
        "$labeled_source_commit" --repo "$repo_root"
    python3 - "$xgrammar_provenance" "$expected_xgrammar_sha256" \
        "$DERIVED_XGRAMMAR_VERSION" <<'PY'
import json
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert provenance["derived_sha256"] == sys.argv[2], provenance["derived_sha256"]
assert provenance["derived_version"] == sys.argv[3], provenance["derived_version"]
assert provenance["drop_distribution"] == "triton", provenance["drop_distribution"]
assert provenance["dropped_requires_dist"] == [
    'triton; platform_system == "Linux" and platform_machine == "x86_64"'
], provenance["dropped_requires_dist"]
assert provenance["bound_files_manifest"], "xgrammar provenance has no manifest"
PY
    run -e DERIVED_XGRAMMAR_VERSION="$DERIVED_XGRAMMAR_VERSION" \
        --volume "$xgrammar_provenance":/xgrammar-provenance.json:ro \
        --entrypoint bash "$image" -c 'exec python -' <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import sysconfig
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

version = md.version("xgrammar")
expected = os.environ["DERIVED_XGRAMMAR_VERSION"]
assert version == expected, f"installed xgrammar is {version!r}, expected {expected!r}"
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
dist_infos = sorted(p.name for p in purelib.glob("xgrammar-*.dist-info"))
assert dist_infos == [f"xgrammar-{version}.dist-info"], dist_infos
requirements = md.distribution("xgrammar").requires or ()
triton = [r for r in requirements if canonicalize_name(Requirement(r).name) == "triton"]
assert not triton, f"corrected xgrammar still declares triton: {triton}"
manifest = json.loads(pathlib.Path("/xgrammar-provenance.json").read_text())[
    "bound_files_manifest"
]
mismatches = []
checked = 0
for relative, expected_hash in sorted(manifest.items()):
    if relative.endswith(".dist-info/RECORD"):
        continue
    path = purelib / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual != expected_hash:
        mismatches.append(relative)
    checked += 1
assert not mismatches, f"installed xgrammar files differ from provenance: {mismatches}"
print(f"installed xgrammar distribution: {version} ({dist_infos[0]}); "
      f"{checked} wheel members match provenance; triton metadata absent")
PY
else
    echo "WARNING: no --xgrammar-wheel-sha256/--xgrammar-provenance given;"
    echo "running only the xgrammar identity/metadata gate; content binding was NOT checked"
    run -e DERIVED_XGRAMMAR_VERSION="$DERIVED_XGRAMMAR_VERSION" \
        --entrypoint bash "$image" -c 'exec python -' <<'PY'
import importlib.metadata as md
import os
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

version = md.version("xgrammar")
assert version == os.environ["DERIVED_XGRAMMAR_VERSION"], version
requirements = md.distribution("xgrammar").requires or ()
assert not [
    r for r in requirements if canonicalize_name(Requirement(r).name) == "triton"
]
print("installed xgrammar distribution:", version, "(triton metadata absent)")
PY
fi
# The whole corrected wheel, not only the provenance manifest: every other
# member, native payload included, must be installed byte for byte.
if [ -n "$xgrammar_wheel" ]; then
    [ "$(sha256sum "$xgrammar_wheel" | awk '{print $1}')" = "$labeled_xgrammar_sha256" ] || {
        echo "FATAL: staged xgrammar wheel does not hash to the labeled digest $labeled_xgrammar_sha256" >&2
        exit 1
    }
    run --volume "$xgrammar_wheel":/wheel.whl:ro \
        --volume "$scripts":/scripts:ro \
        --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/verify_installed_wheel.py \
        --wheel /wheel.whl --own-tree xgrammar
else
    echo "WARNING: no --xgrammar-wheel given; the installed xgrammar payload was"
    echo "NOT compared with the staged wheel beyond its provenance manifest"
fi

# The plugin wheel follows the same contract as the two platform wheels:
# the OCI label must equal the staged wheel's digest, its provenance must
# belong to the image's history, and every installed sglang_fl file must be
# the wheel's own content (RECORD is installer-owned).
labeled_plugin_sha256="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.sglang-fl-wheel-sha256"}}' "$image")"
if [ -n "$expected_sglang_fl_sha256" ]; then
    [ "$labeled_plugin_sha256" = "$expected_sglang_fl_sha256" ] || {
        echo "FATAL: org.flagos.ci.sglang-fl-wheel-sha256=$labeled_plugin_sha256 != expected $expected_sglang_fl_sha256" >&2
        exit 1
    }
    [ -s "$sglang_fl_provenance" ] || {
        echo "FATAL: plugin-wheel provenance not found: $sglang_fl_provenance" >&2
        exit 1
    }
    [ -s "$sglang_fl_wheel" ] || {
        echo "FATAL: staged sglang_fl wheel not found: $sglang_fl_wheel" >&2
        exit 1
    }
    [ "$(sha256sum "$sglang_fl_wheel" | awk '{print $1}')" = "$expected_sglang_fl_sha256" ] || {
        echo "FATAL: staged sglang_fl wheel does not hash to $expected_sglang_fl_sha256" >&2
        exit 1
    }
    python3 "$scripts/provenance_lineage.py" "$sglang_fl_provenance" \
        "$labeled_source_commit" --repo "$repo_root"
    python3 - "$sglang_fl_provenance" "$expected_sglang_fl_sha256" <<'PY'
import json
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert provenance["distribution"] == "sglang_fl", provenance["distribution"]
assert provenance["sha256"] == sys.argv[2], provenance["sha256"]
PY
    # RECORD is rewritten by the installer; every other member is the wheel's
    # own bytes, and nothing else may live under the installed package.
    run --volume "$sglang_fl_wheel":/wheel.whl:ro \
        --volume "$scripts":/scripts:ro \
        --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/verify_installed_wheel.py \
        --wheel /wheel.whl --own-tree sglang_fl --min-members 100
else
    echo "WARNING: no --sglang-fl-wheel/--sglang-fl-wheel-sha256/--sglang-fl-provenance given;"
    echo "the installed sglang_fl was NOT compared with a staged artifact"
fi

# The offline-wheelhouse candidate binds its WHOLE closure to the manifest
# the build consumed: every distribution added on top of the Runtime is one
# of the three explicit wheels or a wheelhouse wheel at its exact version,
# and every consumed wheelhouse wheel is installed with its own content.
if [ -n "$wheelhouse_manifest" ]; then
    # the manifest travels inside the wheelhouse directory mount, next to
    # the wheels it describes
    run --volume "$(dirname "$wheelhouse_manifest")":/wheelhouse:ro \
        --volume "$refdir":/evidence:ro \
        --volume "$scripts":/scripts:ro \
        --entrypoint bash "$image" -c 'exec python "$@"' bash /scripts/verify_installed_wheel.py \
        --wheelhouse-manifest /wheelhouse/wheelhouse-manifest.json \
        --runtime /evidence/runtime-distributions-before.json
fi

# The image keeps the Runtime's neutral WORKDIR and carries no CI working
# directory, test tooling or build residue: no wheelhouse, resolver cache,
# wheel archive, constraints, exclusion, resolver plan, provenance or
# checkout anywhere (the interpreter bundle's own ensurepip wheels are the
# only legitimate wheel files).
workdir="$(docker image inspect --format '{{.Config.WorkingDir}}' "$image")"
[ "$workdir" = "$BASE_WORKDIR" ] || {
    echo "FATAL: WORKDIR drifted from the Runtime base: '$workdir' != '$BASE_WORKDIR'" >&2
    exit 1
}
run --entrypoint bash "$image" -c '
    for path in /opt/sglang-ci /tmp/artifacts /tmp/wheels /tmp/stubs /wheelhouse /root/.cache/uv; do
        if [ -e "$path" ]; then echo "build residue: $path"; exit 1; fi
    done
    found="$(find / -xdev \( -path /proc -o -path /sys -o -path "*/ensurepip/_bundled" \) -prune -o \
        \( -name "*.whl" -o -name "runtime-constraints.txt" -o -name "runtime-0.5.18.excludes.txt" \
           -o -name "*.provenance.json" -o -name "publication-manifest.json" \
           -o -name "wheelhouse-manifest.json" -o -name "uv-dry-run.txt" \
           -o -name "sglang-resolver-additions.txt" -o -name ".git" \) -print 2>/dev/null | head -n 1)"
    if [ -n "$found" ]; then echo "build residue: $found"; exit 1; fi
    if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(\"pytest\") is None else 1)"
    then :; else echo "test tooling in the image: pytest"; exit 1; fi
' || { echo "FATAL: CI working directory, build residue or test tooling in the image" >&2; exit 1; }
echo "residue check: no CI working directory, wheelhouse, wheel archive, constraints,"
echo "  resolver plan, provenance, checkout or test runner in the image"

run -e TRITON_SLOT="$TRITON_SLOT" --entrypoint bash "$image" -c '
set -eu
test "${PYTHONPATH:-}" = "$TRITON_SLOT"
test "${ROCM_PATH:-}" = /opt/dtk
case ":${LD_LIBRARY_PATH:-}:" in *:/opt/dtk/hip/lib:*) ;; *) echo "Runtime did not export DTK loader paths" >&2; exit 1 ;; esac
test ! -e /opt/rocm
test -x "$ROCM_PATH/bin/hipcc"
"$ROCM_PATH/bin/hipcc" --version | head -n 1
python -c '\''import os, importlib.util; s=importlib.util.find_spec("triton"); slot=os.environ["TRITON_SLOT"]; assert s and s.origin.startswith(slot + "/"), s; print("Runtime Bash environment:", os.environ["PYTHONPATH"], os.environ["ROCM_PATH"], "DTK loader path present")'\''
'
env_json="$(docker image inspect --format '{{json .Config.Env}}' "$image")"
experiment="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.triton-slot-experiment"}}' "$image")"
for var in LD_LIBRARY_PATH= ROCM_PATH=; do
    if grep -q "\"$var" <<<"$env_json"; then
        echo "FATAL: application image overrides Runtime-owned $var" >&2
        exit 1
    fi
done
if [ -z "$experiment" ] || [ "$experiment" = "<no value>" ]; then
    if grep -q '"PYTHONPATH=' <<<"$env_json"; then
        echo "FATAL: production image overrides Runtime-owned PYTHONPATH" >&2
        exit 1
    fi
else
    grep -q "\"PYTHONPATH=$TRITON_SLOT\"" <<<"$env_json" || {
        echo "FATAL: A/B arm does not declare PYTHONPATH=$TRITON_SLOT" >&2
        exit 1
    }
fi

echo "Runtime Bash environment/label contract: FlagTree slot, DTK loader/compiler paths,"
echo "FlagTree default clang selection, plugin-owned import guard, provenance"
echo "labels, resolver mode $RESOLVER_MODE, neutral WORKDIR ('$workdir'),"
echo "no build residue"

echo "ACCEPTED: $image passed all post-build gates (Runtime Bash environment"
echo "was exercised; reference data was"
echo "collected dynamically from $BASE_IMAGE and disposed with $refdir)"
