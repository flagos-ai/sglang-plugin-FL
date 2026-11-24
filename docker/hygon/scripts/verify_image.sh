#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Post-build acceptance for the Hygon two-wheel CI image. Nothing here runs
# during image construction; every stage is a docker run against the
# finished image.
#
# The runtime reference is derived data, not source: it is generated fresh
# from the Runtime base digest inside that base image, and only then compared
# against the candidate. The candidate never supplies its own reference data,
# and nothing lands in the repository: the baseline lives in a mktemp
# directory removed on exit. A non-zero exit means the image must not be
# pushed.
#
#   stage 1  docker run <base>       collect_runtime_reference.py
#   stage 2  docker run <candidate>  verify_runtime_immutable.py (six gates)
#   stage 3  docker run <candidate>  verify_module_provenance.py
#   stage 4  image-level contract: the compatibility ENVs, the Runtime's
#            complete parent layers, the neutral WORKDIR, no build residue
#
# Usage: verify_image.sh <candidate-image> [base-image]
#        (base defaults to the tag in ARG BASE_IMAGE of the containerfile,
#         resolved to its repo digest; pass the digest explicitly for an
#         image built from an older revision of that moving tag)

set -euo pipefail

candidate="${1:?usage: verify_image.sh <candidate-image> [base-image]}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
containerfile="$(cd "$here/.." && pwd)/runtime-0.5.18.containerfile"

if [ "$#" -ge 2 ]; then
    base="$2"
else
    tag="$(sed -n 's/^ARG BASE_IMAGE=//p' "$containerfile" | head -n 1)"
    [ -n "$tag" ] || {
        echo "FATAL: cannot derive the base image from $containerfile" >&2
        exit 1
    }
    base="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$tag" \
        | grep -F "${tag%:*}@sha256:" | head -n 1)"
    [ -n "$base" ] || {
        echo "FATAL: could not resolve $tag to a repo digest" >&2
        exit 1
    }
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/hygon-runtime-baseline.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

run() { docker run --rm --platform linux/amd64 "$@"; }
env_of() { docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$1"; }

echo "=== Stage 1: collecting the runtime reference inside the base ==="
echo "    base:      $base"
echo "    candidate: $candidate"
run --volume "$here":/verifier:ro --volume "$tmp":/baseline \
    --entrypoint python "$base" \
    /verifier/collect_runtime_reference.py --out /baseline --base-ref "$base"

echo "=== Stage 2: runtime immutability (six gates) ==="
run --volume "$here":/verifier:ro --volume "$tmp":/baseline:ro \
    --entrypoint python "$candidate" \
    /verifier/verify_runtime_immutable.py --evidence-dir /baseline

echo "=== Stage 3: vendor module provenance ==="
run --volume "$here":/verifier:ro --volume "$tmp":/baseline:ro \
    --entrypoint python "$candidate" \
    /verifier/verify_module_provenance.py \
    --reference /baseline/runtime-0.5.18.provenance.json

echo "=== Stage 4: image-level contract ==="
image_env="$(env_of "$candidate")"
base_ld="$(env_of "$base" | sed -n 's/^LD_LIBRARY_PATH=//p' | head -n 1)"
image_ld="$(sed -n 's/^LD_LIBRARY_PATH=//p' <<<"$image_env" | head -n 1)"
for entry in PYTHONPATH=/opt/flagtree ROCM_PATH=/opt/dtk; do
    grep -qxF "$entry" <<<"$image_env" || {
        echo "FATAL: image ENV lacks $entry" >&2
        exit 1
    }
done
# The DTK loader paths are prepended to whatever the base carries.
case "$image_ld" in
    /opt/dtk/dcc/gcvm/lib:*) ;;
    *)
        echo "FATAL: image LD_LIBRARY_PATH does not start with the DTK loader paths: '$image_ld'" >&2
        exit 1
        ;;
esac
if [ -n "$base_ld" ]; then
    case "$image_ld" in
        *":$base_ld") ;;
        *)
            echo "FATAL: image LD_LIBRARY_PATH dropped the base value '$base_ld'" >&2
            exit 1
            ;;
    esac
fi

# The image is built FROM the base, so its RootFS must start with every
# layer of that base.
python3 - "$(docker image inspect --format '{{json .RootFS.Layers}}' "$base")" \
    "$(docker image inspect --format '{{json .RootFS.Layers}}' "$candidate")" <<'PY'
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

workdir="$(docker image inspect --format '{{.Config.WorkingDir}}' "$candidate")"
[ "$workdir" = "$(docker image inspect --format '{{.Config.WorkingDir}}' "$base")" ] || {
    echo "FATAL: WORKDIR drifted from the Runtime base: '$workdir'" >&2
    exit 1
}

# No CI working directory, build residue or test tooling in the image (the
# interpreter bundle's own ensurepip wheels are the only legitimate ones).
run --entrypoint bash "$candidate" -c '
    for path in /opt/sglang-ci /tmp/artifacts /root/.cache/uv; do
        if [ -e "$path" ]; then echo "build residue: $path"; exit 1; fi
    done
    found="$(find / -xdev \( -path /proc -o -path /sys -o -path "*/ensurepip/_bundled" \) -prune -o \
        \( -name "*.whl" -o -name "runtime-constraints.txt" -o -name "runtime-0.5.18.excludes.txt" \) -print 2>/dev/null | head -n 1)"
    if [ -n "$found" ]; then echo "build residue: $found"; exit 1; fi
    if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(\"pytest\") is None else 1)"
    then :; else echo "test tooling in the image: pytest"; exit 1; fi
' || { echo "FATAL: CI working directory, build residue or test tooling in the image" >&2; exit 1; }
echo "image-level contract: PYTHONPATH=/opt/flagtree, ROCM_PATH=/opt/dtk,"
echo "  LD_LIBRARY_PATH (DTK paths prepended), complete Runtime base layers,"
echo "  WORKDIR '$workdir', no build residue"

echo "ACCEPTED: $candidate passed the post-build gates (the reference was"
echo "collected from $base and disposed with $tmp)"
