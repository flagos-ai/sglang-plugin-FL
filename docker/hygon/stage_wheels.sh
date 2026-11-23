#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stage the reviewed wheel artifacts for the Hygon CI image build.
#
# This entry only prepares a build context; it never builds the full image
# and never touches the repository working tree:
#   1. stage the sglang-plugin-FL wheel (scripts/stage_plugin_wheel.py).
#      DEVELOPMENT CANDIDATE: built from the current checkout with zip
#      timestamps taken from the commit (SOURCE_DATE_EPOCH), so two builds
#      of one commit are identical for a given build backend. RELEASE INPUT:
#      an already published, immutable wheel given by SGLANG_FL_WHEEL and
#      pinned by SGLANG_FL_SHA256. Either way a provenance JSON (mode,
#      sha256, source_commit) travels with the wheel;
#   2. run the plugin-wheel acceptance checks (METADATA, hygon extra
#      anchors, both SGLang entry-point groups, Hygon attention backend
#      and dispatch config presence, no tests, no triton declaration);
#   3. download the fixed flagos-built SGLang 0.5.18 wheel and verify its
#      SHA256;
#   4. derive the HCU SGLang wheel from it with
#      tools/wheels/hygon/derive_sglang_wheel.py: the reviewed HCU gating
#      patch is folded in and the wheel is republished as
#      0.5.18+flagos.hcu1 BEFORE any image assembly (strict diff gates run
#      inside the derive; the image never patches site-packages);
#   5. fetch the SHA-pinned xgrammar 0.2.1 wheel and derive a local-version
#      wheel whose only metadata change removes its active HCU ``triton``
#      requirement (the payload, tags and every native file stay identical);
#      both derived wheels must hash to their PINNED digests - one version,
#      one content - and an already-derived wheel (published, or a secured
#      local copy) can be staged instead of deriving: SGLANG_DERIVED_WHEEL /
#      XGRAMMAR_DERIVED_WHEEL, verified against the same pins;
#   6. assemble a temporary build context under $TMPDIR with the two
#      application wheels, the corrected dependency wheel, their provenance,
#      the still-transitional resolver exclusion and the minimal containerfile;
#   7. write the publication manifest (disposable): filename, version,
#      SHA256 and suggested Nexus path per artifact;
#   8. print the reproducibility record (source commit, input/derived wheel
#      identities, patch identity, plugin wheel identity and context path).
#
# Wheels and provenance JSONs are never committed; the staging directory
# is disposable.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SGLANG_WHEEL_NAME="sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
SGLANG_WHEEL_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/packages/sglang/0.5.18+flagos/${SGLANG_WHEEL_NAME}"
SGLANG_WHEEL_SHA256="1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800"
DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"
DERIVED_SGLANG_WHEEL_NAME="sglang-${DERIVED_SGLANG_VERSION}-cp310-cp310-linux_x86_64.whl"
HCU_PATCH="$repo_root/tools/wheels/hygon/sglang-v0.5.18-hcu.patch"
XGRAMMAR_WHEEL_NAME="xgrammar-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
XGRAMMAR_WHEEL_URL="https://mirrors.cloud.tencent.com/pypi/packages/1a/62/4fba98477159f4f79b2baa33f891708df3585ff6a1ac77b4c64895ec462b/${XGRAMMAR_WHEEL_NAME}"
XGRAMMAR_WHEEL_SHA256="5b80b8060fcf6c94324f563e1d4778029b21d35c556dca5bbf49428d307eca37"
DERIVED_XGRAMMAR_VERSION="0.2.1+flagos.hcu1"
DERIVED_XGRAMMAR_WHEEL_NAME="xgrammar-${DERIVED_XGRAMMAR_VERSION}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
XGRAMMAR_TRITON_REQUIREMENT='triton; platform_system == "Linux" and platform_machine == "x86_64"'
# A derived version is bound to ONE content. These are the digests the hcu1
# artifacts hash to with this checkout's tools and pinned inputs; a derive
# that produces anything else fails here (different content needs the next
# local version, never a re-published hcu1). They are also what a release
# build pins once the wheels are published to resource.flagos.net.
SGLANG_DERIVED_SHA256="edcae8c0678290ec68b83d8ec38537d0ef7852b3f8f183984126bba9e8639b5f"
XGRAMMAR_DERIVED_SHA256="c2deb2853ac98cbaf63ecfa7f55dd00c54e25c9c43a34e9edb234f84e316d1a2"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_sha256() {  # <file> <expected sha256> <what>
    local actual
    actual="$(sha256_of "$1")"
    if [ "$actual" != "$2" ]; then
        echo "FATAL: SHA256 mismatch for $3: expected $2, got $actual" >&2
        exit 1
    fi
}
stage_pre_derived() {  # <provided wheel> <destination dir> <derived wheel name> <pinned sha256>
    # already-derived wheel (published, or a secured local copy): staged
    # with its provenance and verified against the same pin as a derive
    local provided="$1" dest="$2" name="$3" pinned="$4"
    test -s "$provided" || { echo "FATAL: $provided is not a non-empty file" >&2; exit 1; }
    test -s "${provided%.whl}.provenance.json" || {
        echo "FATAL: ${provided%.whl}.provenance.json (provenance of the pre-derived wheel) is missing" >&2
        exit 1
    }
    mkdir -p "$dest"
    cp "$provided" "$dest/$name"
    cp "${provided%.whl}.provenance.json" "$dest/${name%.whl}.provenance.json"
    echo "using pre-derived wheel: $provided"
    require_sha256 "$dest/$name" "$pinned" "$name (version content is pinned; different content needs the next local version)"
}

stage="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-stage.XXXXXX")"
# The final build context must SURVIVE a successful run of this script (the
# printed docker build command references it); a failed or interrupted run
# leaves nothing behind in $TMPDIR.
context="$(mktemp -d "${TMPDIR:-/tmp}/hygon-two-wheel-context.XXXXXX")"
cleanup() {
    local status=$?
    rm -rf "$stage"
    [ "$status" -eq 0 ] || rm -rf "$context"
}
trap cleanup EXIT

# Keep both areas strictly outside the repository.
for area in "$stage" "$context"; do
    case "$(cd "$area" && pwd -P)/" in
        "$(cd "$repo_root" && pwd -P)/"*)
            echo "FATAL: staging directory resolved inside the repository" >&2
            exit 1
            ;;
    esac
done

echo "=== Staging the sglang-plugin-FL wheel ==="
PLUGIN_WHEEL_NAME="sglang_fl-0.1.0-py3-none-any.whl"
if [ -n "${SGLANG_FL_WHEEL:-}" ]; then
    # RELEASE INPUT: the published, immutable sglang_fl wheel, pinned by its
    # SHA256 and accompanied by its provenance (scripts/stage_plugin_wheel.py
    # verifies both). Nothing is built in this mode.
    [ -n "${SGLANG_FL_SHA256:-}" ] || {
        echo "FATAL: SGLANG_FL_WHEEL requires SGLANG_FL_SHA256 (the pinned digest of the published wheel)" >&2
        exit 1
    }
    python3 "$repo_root/docker/hygon/scripts/stage_plugin_wheel.py" --dest "$stage" \
        --mode published --wheel "$SGLANG_FL_WHEEL" --sha256 "$SGLANG_FL_SHA256"
else
    # DEVELOPMENT CANDIDATE: built from this checkout with reproducible zip
    # timestamps. Not releasable: build_two_wheel.sh refuses to push it.
    [ -z "${SGLANG_FL_SHA256:-}" ] || {
        echo "FATAL: SGLANG_FL_SHA256 is given without SGLANG_FL_WHEEL" >&2
        exit 1
    }
    python3 "$repo_root/docker/hygon/scripts/stage_plugin_wheel.py" --dest "$stage" \
        --mode built-from-checkout --repo-root "$repo_root"
fi
plugin_wheel="$stage/$PLUGIN_WHEEL_NAME"
plugin_provenance="${PLUGIN_WHEEL_NAME%.whl}.provenance.json"
test -s "$plugin_wheel"
test -s "$stage/$plugin_provenance"

echo "=== Plugin wheel acceptance ==="
python3 "$repo_root/docker/hygon/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"
plugin_mode="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["mode"])' \
    "$stage/$plugin_provenance")"

echo "=== Fetching and verifying the original SGLang wheel ==="
sglang_wheel="$stage/$SGLANG_WHEEL_NAME"
if [ -n "${SGLANG_INPUT_WHEEL:-}" ]; then
    test -s "$SGLANG_INPUT_WHEEL" || {
        echo "FATAL: SGLANG_INPUT_WHEEL is not a non-empty file: $SGLANG_INPUT_WHEEL" >&2
        exit 1
    }
    # Offline/dev escape with the SAME identity gate: a pre-secured local
    # copy of the pinned wheel (same override as the derive tests).
    cp "$SGLANG_INPUT_WHEEL" "$sglang_wheel"
    echo "using pre-secured input wheel: $SGLANG_INPUT_WHEEL"
else
    for attempt in 1 2 3 4 5; do
        if curl -fL --retry 3 --retry-delay 5 "$SGLANG_WHEEL_URL" -o "$sglang_wheel"; then
            break
        fi
        echo "wheel fetch attempt ${attempt} failed; retrying" >&2
        sleep 10
    done
fi
actual="$(sha256sum "$sglang_wheel" | awk '{print $1}')"
if [ "$actual" != "$SGLANG_WHEEL_SHA256" ]; then
    echo "FATAL: SHA256 mismatch for $SGLANG_WHEEL_NAME: expected $SGLANG_WHEEL_SHA256, got $actual" >&2
    exit 1
fi
echo "verified $SGLANG_WHEEL_NAME sha256=$actual"

echo "=== Staging the HCU SGLang wheel ($DERIVED_SGLANG_VERSION) ==="
derive_out="$stage/derived"
derived_wheel="$derive_out/$DERIVED_SGLANG_WHEEL_NAME"
derived_provenance="${DERIVED_SGLANG_WHEEL_NAME%.whl}.provenance.json"
if [ -n "${SGLANG_DERIVED_WHEEL:-}" ]; then
    stage_pre_derived "$SGLANG_DERIVED_WHEEL" "$derive_out" \
        "$DERIVED_SGLANG_WHEEL_NAME" "$SGLANG_DERIVED_SHA256"
else
    python3 "$repo_root/tools/wheels/hygon/derive_sglang_wheel.py" \
        --input-wheel "$sglang_wheel" \
        --expected-input-sha256 "$SGLANG_WHEEL_SHA256" \
        --patch "$HCU_PATCH" \
        --output-dir "$derive_out" \
        --derived-version "$DERIVED_SGLANG_VERSION" \
        --source-commit "$(git -C "$repo_root" rev-parse HEAD)" \
        | tee "$stage/derive.log"
    require_sha256 "$derived_wheel" "$SGLANG_DERIVED_SHA256" \
        "$DERIVED_SGLANG_WHEEL_NAME (version content is pinned; different content needs the next local version)"
fi
test -s "$derive_out/$derived_provenance"
derived_sha256="$(sha256_of "$derived_wheel")"
patch_sha256="$(sha256_of "$HCU_PATCH")"

echo "=== Fetching and verifying the original xgrammar wheel ==="
xgrammar_wheel="$stage/$XGRAMMAR_WHEEL_NAME"
if [ -n "${XGRAMMAR_INPUT_WHEEL:-}" ]; then
    test -s "$XGRAMMAR_INPUT_WHEEL" || {
        echo "FATAL: XGRAMMAR_INPUT_WHEEL is not a non-empty file: $XGRAMMAR_INPUT_WHEEL" >&2
        exit 1
    }
    cp "$XGRAMMAR_INPUT_WHEEL" "$xgrammar_wheel"
    echo "using pre-secured input wheel: $XGRAMMAR_INPUT_WHEEL"
else
    for attempt in 1 2 3 4 5; do
        if curl -fL --retry 3 --retry-delay 5 "$XGRAMMAR_WHEEL_URL" -o "$xgrammar_wheel"; then
            break
        fi
        echo "xgrammar wheel fetch attempt ${attempt} failed; retrying" >&2
        sleep 10
    done
fi
test -s "$xgrammar_wheel" || {
    echo "FATAL: xgrammar wheel fetch produced no artifact" >&2
    exit 1
}
xgrammar_actual="$(sha256sum "$xgrammar_wheel" | awk '{print $1}')"
if [ "$xgrammar_actual" != "$XGRAMMAR_WHEEL_SHA256" ]; then
    echo "FATAL: SHA256 mismatch for $XGRAMMAR_WHEEL_NAME: expected $XGRAMMAR_WHEEL_SHA256, got $xgrammar_actual" >&2
    exit 1
fi
echo "verified $XGRAMMAR_WHEEL_NAME sha256=$xgrammar_actual"

echo "=== Staging the HCU xgrammar wheel ($DERIVED_XGRAMMAR_VERSION) ==="
xgrammar_derive_out="$stage/xgrammar-derived"
derived_xgrammar_wheel="$xgrammar_derive_out/$DERIVED_XGRAMMAR_WHEEL_NAME"
derived_xgrammar_provenance="${DERIVED_XGRAMMAR_WHEEL_NAME%.whl}.provenance.json"
if [ -n "${XGRAMMAR_DERIVED_WHEEL:-}" ]; then
    stage_pre_derived "$XGRAMMAR_DERIVED_WHEEL" "$xgrammar_derive_out" \
        "$DERIVED_XGRAMMAR_WHEEL_NAME" "$XGRAMMAR_DERIVED_SHA256"
else
    python3 "$repo_root/tools/wheels/hygon/derive_metadata_wheel.py" \
        --input-wheel "$xgrammar_wheel" \
        --expected-input-sha256 "$XGRAMMAR_WHEEL_SHA256" \
        --drop-requires-dist "$XGRAMMAR_TRITON_REQUIREMENT" \
        --drop-distribution triton \
        --derived-version "$DERIVED_XGRAMMAR_VERSION" \
        --output-dir "$xgrammar_derive_out" \
        --source-commit "$(git -C "$repo_root" rev-parse HEAD)" \
        | tee "$stage/xgrammar-derive.log"
    require_sha256 "$derived_xgrammar_wheel" "$XGRAMMAR_DERIVED_SHA256" \
        "$DERIVED_XGRAMMAR_WHEEL_NAME (version content is pinned; different content needs the next local version)"
fi
test -s "$xgrammar_derive_out/$derived_xgrammar_provenance"
derived_xgrammar_sha256="$(sha256_of "$derived_xgrammar_wheel")"

echo "=== Assembling the build context ==="
mkdir -p "$context/artifacts"
cp "$plugin_wheel" "$context/artifacts/"
# the plugin wheel's provenance travels with the context like the derived
# wheels': the post-build acceptance binds the OCI label to it
cp "$stage/$plugin_provenance" "$context/artifacts/$plugin_provenance"
cp "$derived_wheel" "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME"
# the provenance JSON travels with the context (NOT into the image): the
# post-build acceptance binds the OCI label and the installed patched
# files against it
cp "$derive_out/$derived_provenance" "$context/artifacts/$derived_provenance"
cp "$derived_xgrammar_wheel" \
   "$context/artifacts/$DERIVED_XGRAMMAR_WHEEL_NAME"
cp "$xgrammar_derive_out/$derived_xgrammar_provenance" \
   "$context/artifacts/$derived_xgrammar_provenance"
cp "$repo_root/docker/hygon/runtime-0.5.18.excludes.txt" \
   "$context/artifacts/runtime-0.5.18.excludes.txt"
# The staged containerfile references docker/hygon/ (external acceptance
# scripts), so the whole platform directory travels with
# the context. The HCU patch is NOT part of the image inputs anymore: it
# is consumed by the derive step above (its input lives in
# tools/wheels/hygon/).
cp -R "$repo_root/docker" "$context/docker"
cp "$repo_root/docker/hygon/runtime-0.5.18.containerfile" "$context/"
# defense in depth: no patch may enter the build context
if [ -e "$context/docker/hygon/sglang-v0.5.18-hcu.patch" ]; then
    echo "FATAL: HCU patch leaked into the build context" >&2
    exit 1
fi

echo "=== Publication manifest (disposable evidence; nothing is published here) ==="
python3 "$repo_root/tools/wheels/hygon/publication_manifest.py" --context "$context"

echo "=== Reproducibility record ==="
commit="$(git -C "$repo_root" rev-parse HEAD)"
echo "source-commit:  $commit"
echo "plugin-wheel:   $PLUGIN_WHEEL_NAME sha256=$(sha256_of "$plugin_wheel") mode=$plugin_mode (release input must be 'published')"
echo "sglang-input:   $SGLANG_WHEEL_NAME sha256=$SGLANG_WHEEL_SHA256"
echo "hcu-patch:      sha256=$patch_sha256"
echo "sglang-derived: $DERIVED_SGLANG_WHEEL_NAME sha256=$derived_sha256"
echo "xgrammar-input: $XGRAMMAR_WHEEL_NAME sha256=$XGRAMMAR_WHEEL_SHA256"
echo "xgrammar-derived: $DERIVED_XGRAMMAR_WHEEL_NAME sha256=$derived_xgrammar_sha256"
echo "pinned-content: both derived wheels match their pinned digests (one version, one content)"
echo "context:        $context (survives this script; remove manually when done)"
echo "build command:"
echo "  docker/hygon/build_two_wheel.sh <tag>"
echo "  (requires a clean worktree; resolves the runtime tag to one digest,"
echo "   collects runtime-constraints.txt from it, runs the dry-run plan"
echo "   gate, copies the constraints into this context, then builds it)"
