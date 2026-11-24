#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build entry for the Hygon two-wheel CI image. The containerfile is pure
# image assembly; everything that used to be build-time acceptance lives
# here (pre-build, against the resolved base digest) or in
# docker/hygon/scripts/verify_image.sh (post-build):
#
#   1. docker/hygon/stage_wheels.sh - stage the plugin wheel (built from
#      this checkout for a development candidate, or the published wheel
#      pinned by SGLANG_FL_WHEEL / SGLANG_FL_SHA256 for a release), fetch
#      and verify the fixed flagos SGLang wheel, derive the two platform
#      wheels, assemble the disposable build context;
#   2. bind the build to ONE Runtime digest - the containerfile default's
#      moving semver tag resolved once (index mode), or the explicit
#      --runtime-image digest (offline-wheelhouse mode); every later step
#      uses that digest;
#   3. collect the Runtime reference data (freeze, site-file hashes,
#      pip-check baseline, provenance, constraints) from that digest into a
#      mktemp dir - never committed, never in the image;
#   4. run the resolver dry-run plan gate in a container of the same digest:
#      the dry-run is GENERATED from the recipe's own install instruction
#      (uv_install_contract.py) plus --dry-run, with the recipe's own mounts
#      and network mode, and validated to touch no runtime distribution;
#   5. place the generated runtime-constraints.txt into the build context
#      and build the image FROM THE RESOLVED DIGEST (passed as the
#      BASE_IMAGE build arg), carrying its provenance as OCI labels
#      (org.flagos.ci.base-digest / base-local-image-id / source-commit /
#      sglang-wheel-sha256 / xgrammar-wheel-sha256 / sglang-fl-wheel-sha256 /
#      triton-slot / resolver-mode) so acceptance binds to the exact staged
#      artifacts rather than only their version strings or the checkout;
#   6. run the post-build acceptance (docker/hygon/scripts/verify_image.sh)
#      against the produced tag - this is the publication gate.
#
# A failed acceptance aborts this script; --push and --push-candidate are
# reachable only after acceptance passes.
#
# Resolver modes
#   index (the release path)   the recipe resolves the published indexes from
#                              the Runtime's moving semver tag, resolved once.
#                              --push additionally requires the PUBLISHED
#                              sglang_fl wheel (SGLANG_FL_WHEEL +
#                              SGLANG_FL_SHA256): a wheel built from the
#                              checkout is a development candidate, however
#                              reproducible it is.
#   offline-wheelhouse         DEVELOPMENT CANDIDATE for periods when the
#                              FlagOS indexes are unreachable:
#                              --offline-wheelhouse <dir> (built by
#                              docker/hygon/build_wheelhouse.sh from the
#                              resolver cache of a last-green image) and
#                              --runtime-image <name@sha256:digest> (an
#                              explicit, locally verified Runtime object;
#                              never a tag). The install RUN has no network.
#                              --push refuses this mode; only
#                              --push-candidate publishes it, to an
#                              immutable self-describing tag, as
#                              release-ready=false.
#
# Provenance: a publishable build requires a completely clean worktree - the
# build context carries docker/hygon, so any staged, unstaged or untracked
# file could silently enter the artifact while the labels name the recorded
# HEAD. --allow-dirty is a dev-only escape that cannot publish and whose
# output is not publishable evidence.
#
# Usage: build_two_wheel.sh [--push | --push-candidate] [--allow-dirty]
#                            [--offline-wheelhouse <dir> --runtime-image <name@sha256:digest>]
#                            <tag>
#   release input (read by stage_wheels.sh):
#     SGLANG_FL_WHEEL=<published sglang_fl wheel>  SGLANG_FL_SHA256=<its digest>

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform_dir="$repo_root/docker/hygon"
scripts="$platform_dir/scripts"
RELEASE_RECIPE="$platform_dir/runtime-0.5.18.containerfile"
OFFLINE_RECIPE="$platform_dir/runtime-0.5.18.offline-wheelhouse.containerfile"
DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"
DERIVED_SGLANG_WHEEL_NAME="sglang-${DERIVED_SGLANG_VERSION}-cp310-cp310-linux_x86_64.whl"
DERIVED_XGRAMMAR_VERSION="0.2.1+flagos.hcu1"
DERIVED_XGRAMMAR_WHEEL_NAME="xgrammar-${DERIVED_XGRAMMAR_VERSION}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PLUGIN_WHEEL_NAME="sglang_fl-0.1.0-py3-none-any.whl"
# The compiler slot the Runtime Bash environment selects. The only other value
# is the A/B experiment's /opt/triton (triton_slot_ab.sh), which is never built
# here: the production default stays /opt/flagtree.
TRITON_SLOT="/opt/flagtree"
INDEX_MODE="index"
OFFLINE_MODE="offline-wheelhouse"
# A candidate tag is immutable and self-describing: it names the commit the
# plugin wheel was built from and the UTC build time.
CANDIDATE_TAG_PATTERN='^harbor\.baai\.ac\.cn/flagos-dev/sglang-plugin-fl:0\.5\.18-hygon-2w-offline-([0-9a-f]{7,40})-[0-9]{8}T[0-9]{6}Z$'
RUNTIME_REFERENCE_PATTERN='^[a-z0-9._/-]+@sha256:[0-9a-f]{64}$'

usage() {
    cat >&2 <<'USAGE'
usage: build_two_wheel.sh [--push | --push-candidate] [--allow-dirty]
                          [--offline-wheelhouse <dir> --runtime-image <name@sha256:digest>]
                          <tag>
  --push             release publication: the published indexes and the
                     PUBLISHED sglang_fl wheel (SGLANG_FL_WHEEL + SGLANG_FL_SHA256)
  --push-candidate   TEMP publication of the offline-wheelhouse development
                     candidate (release-ready=false). The tag must be
                     harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-2w-offline-<source-short-sha>-<UTC timestamp>
                     and must not exist in the registry yet.
  --allow-dirty      dev build from a dirty worktree; never publishable
USAGE
}

push=0
push_candidate=0
allow_dirty=0
wheelhouse=""
runtime_image=""
tag=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --push) push=1; shift ;;
        --push-candidate) push_candidate=1; shift ;;
        --allow-dirty) allow_dirty=1; shift ;;
        --offline-wheelhouse) wheelhouse="${2:?--offline-wheelhouse needs a directory}"; shift 2 ;;
        --runtime-image) runtime_image="${2:?--runtime-image needs a reference}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "FATAL: unknown option $1" >&2; usage; exit 1 ;;
        *) [ -z "$tag" ] || { usage; exit 1; }; tag="$1"; shift ;;
    esac
done
[ -n "$tag" ] || { usage; exit 1; }

publish=$((push + push_candidate))
[ "$publish" -le 1 ] || {
    echo "FATAL: --push and --push-candidate are mutually exclusive" >&2
    exit 1
}
if [ "$allow_dirty" -eq 1 ] && [ "$publish" -gt 0 ]; then
    echo "FATAL: --allow-dirty cannot be combined with --push or --push-candidate (a dirty build is not publishable)" >&2
    exit 1
fi

# --- Resolver mode and publication gates (before anything is staged) --------
if [ -n "$wheelhouse" ]; then
    resolver_mode="$OFFLINE_MODE"
    [ -n "$runtime_image" ] || {
        echo "FATAL: --offline-wheelhouse requires --runtime-image <name@sha256:digest>: the candidate is bound to ONE explicit Runtime digest and never falls back to a moving tag" >&2
        exit 1
    }
    [ "$push" -eq 0 ] || {
        echo "FATAL: --push (release) cannot consume an offline wheelhouse: a release resolves the published indexes with the published sglang_fl wheel (a development candidate is published with --push-candidate)" >&2
        exit 1
    }
    [ -d "$wheelhouse" ] || {
        echo "FATAL: --offline-wheelhouse is not a directory: $wheelhouse" >&2
        exit 1
    }
    recipe="$OFFLINE_RECIPE"
else
    resolver_mode="$INDEX_MODE"
    [ -z "$runtime_image" ] || {
        echo "FATAL: --runtime-image goes with --offline-wheelhouse (the index mode resolves the recipe's own runtime tag once)" >&2
        exit 1
    }
    [ "$push_candidate" -eq 0 ] || {
        echo "FATAL: --push-candidate publishes the offline-wheelhouse development candidate only (the release publication is --push)" >&2
        exit 1
    }
    recipe="$RELEASE_RECIPE"
fi

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [ "$allow_dirty" -eq 0 ] && [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
    echo "FATAL: worktree is dirty; commit or stash first (dev builds: --allow-dirty)" >&2
    git -C "$repo_root" status --short >&2
    exit 1
fi
[ "$allow_dirty" -eq 0 ] || echo "WARNING: --allow-dirty dev build; NOT releasable"

if [ "$push_candidate" -eq 1 ]; then
    # The tag names the commit and the build time, so a rebuilt candidate can
    # never hide behind an older tag, and it is never overwritten.
    if ! [[ "$tag" =~ $CANDIDATE_TAG_PATTERN ]]; then
        echo "FATAL: a candidate tag must be harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-2w-offline-<source-short-sha>-<UTC timestamp> (got $tag)" >&2
        exit 1
    fi
    case "$source_commit" in
        "${BASH_REMATCH[1]}"*) ;;
        *)
            echo "FATAL: the candidate tag names commit ${BASH_REMATCH[1]} but the worktree is at $source_commit" >&2
            exit 1
            ;;
    esac
fi

# registry_ref_state <ref>: "present" or "free"; anything it cannot decide is
# an error (an unreachable registry is not a free tag).
registry_ref_state() {
    local message status=0
    message="$(docker manifest inspect "$1" 2>&1 >/dev/null)" || status=$?
    if [ "$status" -eq 0 ]; then
        echo present
        return 0
    fi
    case "$message" in
        *"not found"*|*"manifest unknown"*|*"MANIFEST_UNKNOWN"*) echo free ;;
        *)
            echo "FATAL: cannot tell whether $1 exists in the registry: $message" >&2
            return 1
            ;;
    esac
}

if [ "$push_candidate" -eq 1 ]; then
    state="$(registry_ref_state "$tag")"
    [ "$state" = "free" ] || {
        echo "FATAL: refusing to overwrite: $tag already exists in the registry" >&2
        exit 1
    }
    echo "candidate tag is free in the registry: $tag"
fi

echo "=== [1/6] Staging the two wheels (resolver mode: $resolver_mode) ==="
staging_log="$(mktemp "${TMPDIR:-/tmp}/hygon-2w-stage-log.XXXXXX")"
if [ -n "$wheelhouse" ]; then
    HYGON_WHEELHOUSE="$wheelhouse" bash "$platform_dir/stage_wheels.sh" | tee "$staging_log"
else
    bash "$platform_dir/stage_wheels.sh" | tee "$staging_log"
fi
context="$(awk '/^context:/{print $2}' "$staging_log" | tail -n 1)"
rm -f "$staging_log"
[ -n "$context" ] && [ -d "$context" ] || {
    echo "FATAL: could not locate the staged build context" >&2
    exit 1
}
resolver_cache_source=""
wheelhouse_manifest_sha256="n/a"
if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
    wheelhouse_manifest="$context/wheelhouse/wheelhouse-manifest.json"
    [ -s "$wheelhouse_manifest" ] || {
        echo "FATAL: the context carries no wheelhouse manifest: $wheelhouse_manifest" >&2
        exit 1
    }
    resolver_cache_source="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cache_source"]["image"])' \
        "$wheelhouse_manifest")"
    wheelhouse_manifest_sha256="$(sha256sum "$wheelhouse_manifest" | awk '{print $1}')"
    echo "resolver cache source: $resolver_cache_source"
    echo "wheelhouse manifest:   sha256=$wheelhouse_manifest_sha256"
fi
# The plugin wheel is either the PUBLISHED artifact (release input) or a
# development candidate built from the checkout. Only the published one may
# be pushed as a release, whatever its provenance says about the commit; the
# offline-wheelhouse candidate is a development candidate by definition.
plugin_provenance="$context/artifacts/${PLUGIN_WHEEL_NAME%.whl}.provenance.json"
[ -s "$plugin_provenance" ] || {
    echo "FATAL: plugin-wheel provenance missing from the context: $plugin_provenance" >&2
    exit 1
}
plugin_mode="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["mode"])' "$plugin_provenance")"
if [ "$push" -eq 1 ] && [ "$plugin_mode" != "published" ]; then
    echo "FATAL: --push requires the published sglang_fl wheel: stage with SGLANG_FL_WHEEL and SGLANG_FL_SHA256 (mode is '$plugin_mode'; a wheel built from the checkout is a development candidate)" >&2
    exit 1
fi
echo "plugin wheel mode: $plugin_mode"

# Same index priority as the release recipe's ARGs (asserted by the
# two-wheel contract unit tests); passed to the dry-run container with -e.
FLAGOS_INDEX=https://resource.flagos.net/repository/flagos-pypi-hygon/simple
TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple
PROXY_INDEX=https://resource.flagos.net/repository/pypi-proxy/simple

echo "=== [2/6] Binding the build to one Runtime digest ==="
if [ "$resolver_mode" = "$INDEX_MODE" ]; then
    # Single source of truth for the Runtime base in the release path: the
    # containerfile's ARG default (a moving semver tag per the Runtime
    # owner), resolved ONCE to a repo digest.
    BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' "$recipe" | head -n 1)"
    [ -n "$BASE_IMAGE" ] || { echo "FATAL: no ARG BASE_IMAGE in $recipe" >&2; exit 1; }
    docker pull --platform linux/amd64 "$BASE_IMAGE" >/dev/null
    BASE_REPO="${BASE_IMAGE%:*}"
    RESOLVED_BASE="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$BASE_IMAGE" \
        | grep -F "$BASE_REPO@sha256:" | head -n 1)"
    [ -n "$RESOLVED_BASE" ] || {
        echo "FATAL: could not resolve $BASE_IMAGE to a repo digest" >&2
        exit 1
    }
    echo "runtime base tag:     $BASE_IMAGE"
else
    # The candidate names its Runtime explicitly and must find it locally:
    # the digest may long be gone from the registry. Nothing is pulled and
    # no tag is consulted; the object's identity is verified before use and
    # the push later uploads every parent layer the registry lacks.
    RESOLVED_BASE="$runtime_image"
    if ! [[ "$RESOLVED_BASE" =~ $RUNTIME_REFERENCE_PATTERN ]]; then
        echo "FATAL: --runtime-image must be name@sha256:<64 hex digest>, never a tag (got '$RESOLVED_BASE')" >&2
        exit 1
    fi
    docker image inspect "$RESOLVED_BASE" >/dev/null 2>&1 || {
        echo "FATAL: $RESOLVED_BASE is not in the local Docker store; this entry never falls back to a tag" >&2
        exit 1
    }
fi
BASE_LOCAL_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$RESOLVED_BASE")"
BASE_REPO_DIGESTS="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$RESOLVED_BASE")"
if [ "$BASE_LOCAL_IMAGE_ID" != "${RESOLVED_BASE##*@}" ] \
    && ! grep -qxF "$RESOLVED_BASE" <<<"$BASE_REPO_DIGESTS"; then
    echo "FATAL: the local object named $RESOLVED_BASE is not that digest (image ID $BASE_LOCAL_IMAGE_ID; RepoDigests: $BASE_REPO_DIGESTS)" >&2
    exit 1
fi
echo "runtime base:         $RESOLVED_BASE"
echo "local image id:       $BASE_LOCAL_IMAGE_ID"
# Recorded, never a gate: the remote manifest of an older Runtime may have
# been pruned while the verified local object still carries every layer.
case "$(registry_ref_state "$RESOLVED_BASE" 2>/dev/null || echo unavailable)" in
    present)
        echo "remote manifest:      present"
        ;;
    free)
        echo "remote manifest:      absent - building from the verified local object;"
        echo "                      the push uploads every parent layer the registry lacks"
        ;;
    *)
        echo "remote manifest:      unavailable (the registry could not be asked) -"
        echo "                      building from the verified local object; the push"
        echo "                      uploads every parent layer the registry lacks"
        ;;
esac

refdir="$(mktemp -d "${TMPDIR:-/tmp}/hygon-2w-reference.XXXXXX")"
trap 'rm -rf "$refdir"' EXIT

echo "=== [3/6] Collecting the Runtime reference from the base digest ==="
docker run --rm --platform linux/amd64 \
    --volume "$refdir":/out \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$RESOLVED_BASE" -c 'exec python "$@"' bash \
    /scripts/collect_runtime_reference.py --out /out --base-ref "$RESOLVED_BASE" \
    --selected-slot "$TRITON_SLOT"

echo "=== [4/6] Resolver dry-run plan gate (same base digest, same mounts and network) ==="
# /plan inside this container is the disposable plan dir. The install
# arguments are read from the recipe by uv_install_contract.py - the very
# RUN instruction the build executes - so the dry-run is the real install
# plus --dry-run and nothing else. The same module turns the RUN's own build
# options into this container's volumes and network mode (read-only
# artifacts and wheelhouse mounts; no network in the offline-wheelhouse mode).
plan="$refdir/uvplan"
mkdir -p "$plan"
cp "$refdir/runtime-constraints.txt" "$plan/runtime-constraints.txt"
cp "$refdir/runtime-distributions-before.json" "$plan/"
# The same artifacts directory is bind-mounted by both the dry-run container
# and the Dockerfile install RUN. BuildKit does not copy it into a final layer.
cp "$refdir/runtime-constraints.txt" "$context/artifacts/runtime-constraints.txt"
python3 "$scripts/uv_install_contract.py" "$recipe" > "$plan/args"
run_options=()
while IFS= read -r option; do run_options+=("$option"); done \
    < <(python3 "$scripts/uv_install_contract.py" --run-options --context "$context" "$recipe")
cat > "$plan/dry-run.sh" <<'EOF'
set -euo pipefail
ARGS=()
while IFS= read -r argument; do ARGS+=("$argument"); done < /plan/args
if ! /usr/local/bin/uv pip install --dry-run "${ARGS[@]}" > /plan/uv-dry-run.txt 2>&1; then
    cat /plan/uv-dry-run.txt >&2
    exit 2
fi
python /scripts/validate_uv_plan.py /plan/uv-dry-run.txt /plan
EOF
if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
    # every closure package of the plan must be a wheelhouse wheel
    cat >> "$plan/dry-run.sh" <<'EOF'
python /scripts/wheelhouse.py bind-plan \
    --wheelhouse /wheelhouse \
    --plan /plan/sglang-resolver-additions.txt \
    --out /plan/wheelhouse-consumed.txt
EOF
fi
docker run --rm --platform linux/amd64 \
    -e FLAGOS_INDEX -e TENCENT_INDEX -e PROXY_INDEX \
    ${run_options[@]+"${run_options[@]}"} \
    --volume "$plan":/plan \
    --volume "$scripts":/scripts:ro \
    --entrypoint bash "$RESOLVED_BASE" /plan/dry-run.sh
echo "resolver plan validated: no runtime distribution replaced"
echo "  plan audit: $plan/sglang-resolver-additions.txt (disposable)"

echo "=== [5/6] Building the image ==="
# The derived HCU SGLang wheel's SHA256 (staged by stage_wheels.sh from the
# pinned input wheel + reviewed patch) replaces the former patch label: the
# image binds to the exact wheel artifact it installs.
derived_wheel_sha256="$(sha256sum "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME" | awk '{print $1}')"
xgrammar_wheel_sha256="$(sha256sum "$context/artifacts/$DERIVED_XGRAMMAR_WHEEL_NAME" | awk '{print $1}')"
plugin_wheel_sha256="$(sha256sum "$context/artifacts/$PLUGIN_WHEEL_NAME" | awk '{print $1}')"
labels=(
    --label "org.flagos.ci.base-digest=$RESOLVED_BASE"
    --label "org.flagos.ci.base-local-image-id=$BASE_LOCAL_IMAGE_ID"
    --label "org.flagos.ci.source-commit=$source_commit"
    --label "org.flagos.ci.sglang-wheel-sha256=$derived_wheel_sha256"
    --label "org.flagos.ci.xgrammar-wheel-sha256=$xgrammar_wheel_sha256"
    --label "org.flagos.ci.sglang-fl-wheel-sha256=$plugin_wheel_sha256"
    --label "org.flagos.ci.triton-slot=$TRITON_SLOT"
    --label "org.flagos.ci.resolver-mode=$resolver_mode"
)
if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
    # The candidate says what it is: not release-ready, resolved offline,
    # from the cache of exactly that green image.
    labels+=(
        --label "org.flagos.ci.release-ready=false"
        --label "org.flagos.ci.resolver-cache-source=$resolver_cache_source"
        --label "org.flagos.ci.wheelhouse-manifest-sha256=$wheelhouse_manifest_sha256"
    )
fi
docker build --platform linux/amd64 \
    --build-arg "BASE_IMAGE=$RESOLVED_BASE" \
    "${labels[@]}" \
    -f "$recipe" \
    -t "$tag" "$context"

# The build must have consumed exactly the recorded commit: the worktree
# may not have moved, and the image must carry that commit as its label.
[ "$(git -C "$repo_root" rev-parse HEAD)" = "$source_commit" ] || {
    echo "FATAL: HEAD moved during the build" >&2
    exit 1
}
if [ "$allow_dirty" -eq 0 ] && [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
    echo "FATAL: the worktree became dirty during the build" >&2
    git -C "$repo_root" status --short >&2
    exit 1
fi
labeled_commit="$(docker image inspect --format '{{index .Config.Labels "org.flagos.ci.source-commit"}}' "$tag")"
[ "$labeled_commit" = "$source_commit" ] || {
    echo "FATAL: source-commit label mismatch: image=$labeled_commit worktree=$source_commit" >&2
    exit 1
}

echo "=== [6/6] Post-build acceptance (publication gate) ==="
# Bind the acceptance to the EXACT artifacts: each wheel's SHA256 and its
# provenance go to the verifier, which asserts the OCI label equals that
# digest and the installed files are the wheel's own content - a
# same-version, different-content wheel cannot pass.
derived_provenance="$context/artifacts/${DERIVED_SGLANG_WHEEL_NAME%.whl}.provenance.json"
xgrammar_provenance="$context/artifacts/${DERIVED_XGRAMMAR_WHEEL_NAME%.whl}.provenance.json"
if [ ! -s "$derived_provenance" ]; then
    echo "FATAL: derived-wheel provenance missing from the context: $derived_provenance" >&2
    exit 1
fi
if [ ! -s "$xgrammar_provenance" ]; then
    echo "FATAL: xgrammar provenance missing from the context: $xgrammar_provenance" >&2
    exit 1
fi
verify_args=(
    --sglang-wheel-sha256 "$derived_wheel_sha256"
    --sglang-provenance "$derived_provenance"
    --sglang-wheel "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME"
    --xgrammar-wheel-sha256 "$xgrammar_wheel_sha256"
    --xgrammar-provenance "$xgrammar_provenance"
    --xgrammar-wheel "$context/artifacts/$DERIVED_XGRAMMAR_WHEEL_NAME"
    --sglang-fl-wheel "$context/artifacts/$PLUGIN_WHEEL_NAME"
    --sglang-fl-wheel-sha256 "$plugin_wheel_sha256"
    --sglang-fl-provenance "$plugin_provenance"
)
if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
    verify_args+=(--wheelhouse-manifest "$context/wheelhouse/wheelhouse-manifest.json")
fi
bash "$scripts/verify_image.sh" "$tag" "${verify_args[@]}"

echo "=== Reproducibility record ==="
echo "source-commit:  $source_commit"
echo "resolver mode:  $resolver_mode"
echo "runtime base:   $RESOLVED_BASE (org.flagos.ci.base-digest)"
echo "local image id: $BASE_LOCAL_IMAGE_ID (org.flagos.ci.base-local-image-id)"
echo "sglang wheel:   sha256=$derived_wheel_sha256 (org.flagos.ci.sglang-wheel-sha256; derived 0.5.18+flagos.hcu1)"
echo "xgrammar wheel: sha256=$xgrammar_wheel_sha256 (org.flagos.ci.xgrammar-wheel-sha256; derived 0.2.1+flagos.hcu1)"
echo "plugin wheel:   sha256=$plugin_wheel_sha256 (org.flagos.ci.sglang-fl-wheel-sha256; mode=$plugin_mode)"
echo "triton slot:    $TRITON_SLOT (org.flagos.ci.triton-slot; production default)"
if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
    echo "cache source:   $resolver_cache_source (org.flagos.ci.resolver-cache-source)"
    echo "wheelhouse:     manifest sha256=$wheelhouse_manifest_sha256 (org.flagos.ci.wheelhouse-manifest-sha256)"
    echo "release-ready:  false (org.flagos.ci.release-ready; TEMP development candidate)"
fi
echo "image-tag:      $tag"
echo "image-id:       $(docker image inspect --format '{{.Id}}' "$tag")"
echo "build-context:  $context (from stage_wheels.sh; remove manually when done)"

if [ "$push" -eq 1 ]; then
    echo "=== Pushing (acceptance passed) ==="
    docker push "$tag"
    echo "pushed: $tag"
elif [ "$push_candidate" -eq 1 ]; then
    echo "=== Pushing the TEMP candidate (acceptance passed) ==="
    docker push "$tag"
    bash "$scripts/verify_remote_candidate.sh" "$tag"
    echo "pushed candidate: $tag"
else
    if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then
        echo "accepted (not pushed); publish the TEMP candidate with --push-candidate"
    else
        echo "accepted (not pushed); publish later with --push"
    fi
fi
