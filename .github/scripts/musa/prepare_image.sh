#!/bin/bash
# Pull the pinned image in the job process so Harbor can bypass the runner proxy.
set -euo pipefail
: "${MUSA_CI_IMAGE:?The digest-pinned CI image is required}"
: "${GITHUB_ENV:?The Actions environment file is required}"

task_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/musa-image.XXXXXX")
trap 'rm -rf -- "$task_dir"' EXIT
crane_version=v0.22.1
crane_sha=0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0
curl -fsSL --retry 3 --connect-timeout 30 --max-time 300 \
  --speed-time 60 --speed-limit 1024 \
  "https://github.com/google/go-containerregistry/releases/download/${crane_version}/go-containerregistry_Linux_x86_64.tar.gz" \
  -o "$task_dir/crane.tar.gz"
printf '%s  %s\n' "$crane_sha" "$task_dir/crane.tar.gz" | sha256sum -c -
tar -xzf "$task_dir/crane.tar.gz" -C "$task_dir" crane
mkdir "$task_dir/auth"
printf '{}\n' > "$task_dir/auth/config.json"

# This only changes crane's environment. GitHub downloads and the shared Docker
# daemon keep their existing configuration. Public Harbor pulls need no login.
direct_crane() {
  /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    DOCKER_CONFIG="$task_dir/auth" timeout 60m "$task_dir/crane" "$@"
}
direct_crane manifest "$MUSA_CI_IMAGE" > "$task_dir/manifest.json"
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["digest"])' \
  "$task_dir/manifest.json")
if docker image inspect "$image_id" >/dev/null 2>&1; then
  echo "Verified image is already cached: $image_id"
else
  echo "Downloading the pinned image directly from Harbor: $MUSA_CI_IMAGE"
  # The compressed tar stream avoids a second 30 GiB archive on the runner disk.
  direct_crane pull "$MUSA_CI_IMAGE" /dev/stdout \
    | python3 .github/scripts/musa/stream_image.py \
    | timeout 75m docker load
fi
actual_id=$(docker image inspect "$image_id" --format '{{.Id}}')
test "$actual_id" = "$image_id"
echo "Verified loaded image ID: $image_id"
printf 'MUSA_CI_IMAGE_ID=%s\n' "$image_id" >> "$GITHUB_ENV"
