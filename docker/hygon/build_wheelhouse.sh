#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build the offline wheelhouse of the Hygon offline-wheelhouse DEVELOPMENT
# CANDIDATE from the resolver cache of a last-green two-wheel image.
#
#   1. verify the LOCAL identity of --cache-source (name@sha256:<digest>):
#      the image must already be in the local Docker store and its image ID
#      or RepoDigests must carry that digest - nothing is pulled, no tag is
#      consulted;
#   2. export the image's uv resolver cache (/root/.cache/uv) to a mktemp
#      directory outside the repository;
#   3. docker/hygon/scripts/wheelhouse.py fetch: inventory the cache,
#      download every recorded wheel from the URL the green build recorded
#      (the Tencent mirror is the fallback for generic wheels), verify each
#      one and write the wheelhouse plus wheelhouse-manifest.json;
#   4. verify the finished wheelhouse against its manifest.
#
# The old image's site-packages are never an installation input. Only the
# cache's own record of what the green build downloaded is used - the wheel
# URL, the filename, the registry SHA256 and the unpacked tree - and every
# wheelhouse file is upstream bytes fetched again, not a repack.
#
# Usage: build_wheelhouse.sh --cache-source <name@sha256:digest> --out <dir>
#                             [--tencent-index URL]

set -euo pipefail

platform_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$platform_dir/scripts"
repo_root="$(cd "$platform_dir/../.." && pwd)"
TENCENT_INDEX="https://mirrors.cloud.tencent.com/pypi/simple"

usage() {
    echo "usage: build_wheelhouse.sh --cache-source <name@sha256:digest> --out <dir> [--tencent-index URL]" >&2
}

cache_source=""
out=""
tencent_index="$TENCENT_INDEX"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cache-source) cache_source="${2:?--cache-source needs a value}"; shift 2 ;;
        --out) out="${2:?--out needs a value}"; shift 2 ;;
        --tencent-index) tencent_index="${2:?--tencent-index needs a value}"; shift 2 ;;
        -h|--help)
            sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) usage; exit 1 ;;
    esac
done
if [ -z "$cache_source" ] || [ -z "$out" ]; then
    usage
    exit 1
fi

# The wheelhouse is a build input, never repository content.
parent="$(cd "$(dirname "$out")" && pwd -P)"
case "$parent/" in
    "$repo_root/"*)
        echo "FATAL: the wheelhouse must live outside the repository: $out" >&2
        exit 1
        ;;
esac
out="$parent/$(basename "$out")"

# The cache source is bound to ONE content digest that must already be local.
reference='^[a-z0-9._/-]+@sha256:[0-9a-f]{64}$'
if ! [[ "$cache_source" =~ $reference ]]; then
    echo "FATAL: --cache-source must be name@sha256:<64 hex digest>, got '$cache_source'" >&2
    exit 1
fi
digest="${cache_source##*@}"
if ! docker image inspect "$cache_source" >/dev/null 2>&1; then
    echo "FATAL: $cache_source is not in the local Docker store (this entry never pulls)" >&2
    exit 1
fi
local_id="$(docker image inspect --format '{{.Id}}' "$cache_source")"
repo_digests="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$cache_source")"
if [ "$local_id" != "$digest" ] && ! grep -qxF "$cache_source" <<<"$repo_digests"; then
    echo "FATAL: the local object named $cache_source is not that digest (image ID $local_id, RepoDigests: $repo_digests)" >&2
    exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/hygon-wheelhouse-work.XXXXXX")"
container=""
cleanup() {
    local status=$?
    [ -z "$container" ] || docker rm "$container" >/dev/null 2>&1 || true
    rm -rf "$work"
    [ "$status" -eq 0 ] || true
}
trap cleanup EXIT

echo "=== [1/4] Cache source identity ==="
echo "cache source:  $cache_source"
echo "local image id: $local_id"
container="$(docker create --platform linux/amd64 "$cache_source")"

echo "=== [2/4] Exporting the image's uv resolver cache ==="
docker cp -L "$container:/root/.cache/uv" "$work/uv-cache"
docker rm "$container" >/dev/null
container=""
echo "exported /root/.cache/uv ($(du -sh "$work/uv-cache" | cut -f1))"

echo "=== [3/4] Downloading and verifying every recorded wheel ==="
python3 "$scripts/wheelhouse.py" fetch \
    --cache "$work/uv-cache" --out "$out" \
    --source-image "$cache_source" --tencent-index "$tencent_index"

echo "=== [4/4] Verifying the wheelhouse against its manifest ==="
python3 "$scripts/wheelhouse.py" verify --wheelhouse "$out"

echo "=== Wheelhouse record ==="
echo "cache source:     $cache_source (local image id $local_id)"
echo "wheelhouse:       $out"
python3 - "$out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

wheelhouse = Path(sys.argv[1])
manifest_path = wheelhouse / "wheelhouse-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
total = sum(wheel["size"] for wheel in manifest["wheels"])
print(f"wheels:           {manifest['count']} ({total} bytes)")
print(f"manifest sha256:  {hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
print(f"resolver mode:    {manifest['resolver_mode']} (development candidate; never a release input)")
PY
echo "build command:"
echo "  docker/hygon/build_two_wheel.sh --offline-wheelhouse $out \\"
echo "      --runtime-image <runtime name@sha256:digest> <candidate tag>"
