#!/bin/bash
# Pull the pinned image in bounded ranges and verify every original layer.
set -euo pipefail
: "${MUSA_CI_IMAGE:?The digest-pinned CI image is required}"
: "${GITHUB_ENV:?The Actions environment file is required}"

task_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/musa-image.XXXXXX")
trap 'rm -rf -- "$task_dir"' EXIT
python3 .github/scripts/musa/pull_image.py --manifest-only "$MUSA_CI_IMAGE" \
  > "$task_dir/manifest.json"
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["digest"])' \
  "$task_dir/manifest.json")
if docker image inspect "$image_id" >/dev/null 2>&1; then
  echo "Verified image is already cached: $image_id"
else
  echo "Downloading the pinned image directly from Harbor: $MUSA_CI_IMAGE"
  # The compressed tar stream avoids a second 30 GiB archive on the runner disk.
  timeout 60m python3 .github/scripts/musa/pull_image.py "$MUSA_CI_IMAGE" \
    | python3 .github/scripts/musa/stream_image.py \
    | /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy timeout 75m docker load
fi
actual_id=$(docker image inspect "$image_id" --format '{{.Id}}')
test "$actual_id" = "$image_id"
echo "Verified loaded image ID: $image_id"
printf 'MUSA_CI_IMAGE_ID=%s\n' "$image_id" >> "$GITHUB_ENV"
