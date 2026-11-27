#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Verify a pushed TEMP candidate against the REGISTRY object, never against
# the local tag alone (docker/hygon/build_two_wheel.sh --push-candidate runs
# this after the push):
#
#   1. the object the tag resolves to in the registry (docker buildx
#      imagetools): its digest is the one CI consumes;
#   2. every blob reachable from that manifest exists in the registry -
#      read anonymously through the registry v2 API, so the local Docker
#      store cannot stand in for a missing blob. This is the proof that a
#      runner only needs to pull the candidate: the Runtime parent layers
#      included, even when the base's own manifest was pruned;
#   3. a fresh pull BY DIGEST;
#   4. the pulled object is the accepted local build: RepoDigest, image
#      config, RootFS diff_ids and the org.flagos.ci labels.
#
# Usage: verify_remote_candidate.sh <candidate tag>

set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATE_TAG_PATTERN='^harbor\.baai\.ac\.cn/flagos-dev/sglang-plugin-fl:0\.5\.18-hygon-2w-offline-([0-9a-f]{7,40})-[0-9]{8}T[0-9]{6}Z$'

tag="${1:?usage: verify_remote_candidate.sh <candidate tag>}"
if ! [[ "$tag" =~ $CANDIDATE_TAG_PATTERN ]]; then
    echo "FATAL: not a TEMP candidate tag: $tag" >&2
    exit 1
fi
repository="${tag%:*}"
host="${repository%%/*}"
name="${repository#*/}"

echo "=== [1/4] Remote object of the tag ==="
remote="$(docker buildx imagetools inspect "$tag")"
digest="$(sed -n 's/^Digest:[[:space:]]*//p' <<<"$remote" | head -n 1)"
media_type="$(sed -n 's/^MediaType:[[:space:]]*//p' <<<"$remote" | head -n 1)"
if ! [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "FATAL: the registry did not answer with a digest for $tag: $remote" >&2
    exit 1
fi
echo "tag:          $tag"
echo "RepoDigest:   $repository@$digest"
echo "media type:   $media_type"

echo "=== [2/4] Every blob of the remote manifest exists in the registry ==="
record="$(python3 "$scripts/verify_remote_blobs.py" "$host" "$name" "$digest")"
python3 - "$record" <<'PY'
import json
import sys

record = json.loads(sys.argv[1])
amd64 = record["linux_amd64"]
print(f"blobs present in the registry: {len(record['blobs_checked'])} "
      f"(configs and layers of every platform manifest)")
for item in record["manifests"]:
    print(f"  manifest {item['digest']}  platform {item['platform']}  "
          f"config {item['config_digest']}  layers {item['layers']}")
print(f"image config ID (linux/amd64): {amd64['config_digest']}")
print(f"RootFS diff_ids (linux/amd64): {len(amd64['diff_ids'])}; first "
      f"{amd64['diff_ids'][0]}, last {amd64['diff_ids'][-1]}")
PY

echo "=== [3/4] Fresh pull by digest ==="
docker pull --platform linux/amd64 "$repository@$digest"

echo "=== [4/4] The pulled object is the accepted local build ==="
pulled="$(docker image inspect "$repository@$digest")"
local="$(docker image inspect "$tag")"
python3 - "$record" "$pulled" "$local" "$repository@$digest" <<'PY'
import json
import sys

record = json.loads(sys.argv[1])
pulled = json.loads(sys.argv[2])[0]
local = json.loads(sys.argv[3])[0]
reference = sys.argv[4]
amd64 = record["linux_amd64"]
failures = []
if reference not in pulled.get("RepoDigests", []):
    failures.append(f"pulled RepoDigests {pulled.get('RepoDigests')} lack {reference}")
if pulled["RootFS"]["Layers"] != amd64["diff_ids"]:
    failures.append("pulled RootFS layers differ from the registry's diff_ids")
if local["RootFS"]["Layers"] != amd64["diff_ids"]:
    failures.append("local RootFS layers differ from the registry's diff_ids")
remote_config = amd64["config"]["config"] or {}
for key in ("Env", "Entrypoint", "Cmd", "WorkingDir", "User", "Labels"):
    expected = remote_config.get(key) or None
    if (pulled["Config"].get(key) or None) != expected:
        failures.append(f"pulled {key} differs from the registry's image config")
    if (local["Config"].get(key) or None) != expected:
        failures.append(f"local {key} differs from the registry's image config")
if failures:
    print("FATAL: " + "; ".join(failures), file=sys.stderr)
    raise SystemExit(1)
print(f"RepoDigest:     {reference}")
print(f"config ID:      {amd64['config_digest']}")
print(f"image id:       {pulled['Id']}")
print(f"RootFS layers:  {len(pulled['RootFS']['Layers'])} == registry diff_ids == local build")
print("OCI labels:")
for key in sorted(pulled["Config"].get("Labels") or {}):
    if key.startswith("org.flagos.ci."):
        print(f"  {key}={pulled['Config']['Labels'][key]}")
PY

echo "remote candidate verified: $repository@$digest"
