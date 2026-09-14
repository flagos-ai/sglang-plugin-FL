#!/usr/bin/env bash
# Download the Qwen models required by the thead (zw810e) e2e tests into the
# runner's model store. THROWAWAY scaffolding for the do-not-merge PR
# chore/thead-model-download — "下完就把 PR 关掉".
#
# The thead CI container mounts the store at
#   /mnt/airs-business/cicd/models:/data/models/Qwen
# (see .github/configs/thead.yml -> container_volumes). The e2e cases
# reference these four models as /data/models/Qwen/<Model>.
#
# This job runs in the SAME container setup as the thead test jobs (image +
# volume mount, --user root, no device needed), so writes land on the real
# store, not an ephemeral overlay. Idempotent: completed model directories are
# detected by config.json (and marked with .cache_ready for future runs), so
# re-runs skip models that are already present even if an older run did not
# create the marker.
#
# All four are public (Apache-2.0) and identical on ModelScope and HF.
#
# Source defaults to ModelScope (fast inside CN). Set SOURCE=huggingface to use
# Hugging Face instead; if HF is unreachable set HF_ENDPOINT=https://hf-mirror.com.
#
# Usage:
#   ./download_models.sh                                # all 4 models, default path
#   MODELS="Qwen3-4B Qwen3-0.6B" ./download_models.sh   # only the small ones
#   DEST=/some/dir ./download_models.sh                 # custom destination

set -euo pipefail

DEST="${DEST:-/data/models/Qwen}"
SOURCE="${SOURCE:-modelscope}"   # modelscope | huggingface

# Local copy roots: if a model already exists under one of these, copy it
# instead of downloading. Searched as <root>/<Model> and <root>/Qwen/<Model>
# (must contain config.json to count as a real model). Space-separated.
LOCAL_ROOTS="${LOCAL_ROOTS:-/data/models/Qwen /data /data/models}"

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }

# local_dir_name -> repo_id (identical on ModelScope and Hugging Face)
ALL_MODELS=(
  "Qwen3-0.6B:Qwen/Qwen3-0.6B"
  "Qwen3-4B:Qwen/Qwen3-4B"
  "Qwen3.6-35B-A3B:Qwen/Qwen3.6-35B-A3B"
  "Qwen3.6-27B:Qwen/Qwen3.6-27B"
)

# Optional allow-list via MODELS="Name1 Name2" (matched on local dir name).
if [[ -n "${MODELS:-}" ]]; then
  SELECTED=()
  for entry in "${ALL_MODELS[@]}"; do
    local_name="${entry%%:*}"
    for want in $MODELS; do [[ "$local_name" == "$want" ]] && SELECTED+=("$entry"); done
  done
  [[ ${#SELECTED[@]} -eq 0 ]] && { echo "No models matched MODELS='$MODELS'"; exit 1; }
  ALL_MODELS=("${SELECTED[@]}")
fi

echo "==> Destination: $DEST"
echo "==> Source:      $SOURCE"
echo "==> Existing contents of $DEST:"
ls -1 "$DEST" 2>/dev/null || echo "    (dir does not exist yet — will be created)"
echo

mkdir -p "$DEST"

# ---- Non-blocking mount verification ---------------------------------------
# If $DEST is not a real mount in this container, writes land on the ephemeral
# overlay and vanish — warn loudly but never block.
echo "==> Mount check for $DEST"
if command -v df >/dev/null 2>&1; then
  df -h "$DEST" 2>/dev/null || echo "    (df reported an error for $DEST)"
fi
is_mount="unknown"
if command -v mountpoint >/dev/null 2>&1; then
  if mountpoint -q "$DEST" 2>/dev/null; then is_mount="yes"; else is_mount="no"; fi
elif command -v stat >/dev/null 2>&1; then
  d1=$(stat -c %d "$DEST" 2>/dev/null || true)
  d2=$(stat -c %d "$DEST/.." 2>/dev/null || true)
  if [[ -n "$d1" && -n "$d2" ]]; then
    if [[ "$d1" != "$d2" ]]; then is_mount="yes"; else is_mount="no"; fi
  fi
fi
case "$is_mount" in
  yes) echo "    -> $DEST IS a mountpoint in this container (bind mount present)." ;;
  no)  echo "    -> WARNING: $DEST is NOT a mountpoint in this container."
       echo "       Writes would land on the container's ephemeral overlay and be LOST"
       echo "       when it exits (and invisible to other nodes). Ctrl-C if unintended." ;;
  *)   echo "    -> mount status unknown; continuing" ;;
esac
echo

py_download() {
  # $1 = backend (modelscope|huggingface), $2 = repo_id, $3 = local_dir
  python3 - <<PY
import sys
backend, repo, out = "$1", "$2", "$3"
try:
    if backend == "modelscope":
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    pkg = "modelscope" if backend == "modelscope" else "huggingface_hub"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--user", pkg])
    if backend == "modelscope":
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download
snapshot_download(repo, local_dir=out)
print("downloaded", repo, "->", out)
PY
}

find_local_copy() {
  # Print the first existing model dir for $1 (a local dir name) found under
  # LOCAL_ROOTS (as <root>/<name> or <root>/Qwen/<name>) that looks like a real
  # model (has config.json). Empty output + return 1 if none.
  local name="$1" root cand
  for root in $LOCAL_ROOTS; do
    for cand in "$root/$name" "$root/Qwen/$name"; do
      [[ -f "$cand/config.json" ]] && { echo "$cand"; return 0; }
    done
  done
  return 1
}

backend_ready=false
ensure_backend() {
  [[ "$backend_ready" == true ]] && return

  # Install while the CI proxy is still active. ModelScope downloads bypass
  # the proxy below because the proxy's CONNECT tunnel returns HTTP 500 for
  # modelscope.cn, while pip can use the proxy to reach PyPI.
  case "$SOURCE" in
    modelscope)
      python3 -c "import modelscope" 2>/dev/null || pip install -q modelscope
      ;;
    huggingface)
      python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
      ;;
    *)
      echo "Unknown SOURCE='$SOURCE' (use modelscope|huggingface)"
      exit 1
      ;;
  esac
  backend_ready=true
}

for entry in "${ALL_MODELS[@]}"; do
  local_name="${entry%%:*}"
  repo="${entry#*:}"
  out="$DEST/$local_name"
  marker="$out/.cache_ready"

  # A model directory containing config.json is considered present. This also
  # handles stores populated by an earlier job/version that did not write the
  # marker. A marker without config.json is not trusted, so interrupted
  # downloads are resumed instead of being served as complete models.
  if [[ -f "$out/config.json" ]]; then
    [[ -f "$marker" ]] || touch "$marker"
    echo "==> [skip]    $local_name already exists ($(du -sh "$out" 2>/dev/null | cut -f1))"
    continue
  fi

  # Prefer copying an existing local copy over downloading (network is flaky /
  # proxied in CI). Only finds paths actually visible inside the container.
  local_src="$(find_local_copy "$local_name" || true)"
  if [[ -n "$local_src" ]]; then
    if [[ "$local_src" -ef "$out" ]]; then
      touch "$marker"
      echo "==> [skip]    $local_name already at target ($out)"
      continue
    fi
    echo "==> [copy]    $local_name  <-  $local_src"
    rm -rf "$out"
    mkdir -p "$out"
    cp -a "$local_src/." "$out/"
    touch "$marker"
    echo "==> [done]    $local_name -> $out (copied, $(du -sh "$out" 2>/dev/null | cut -f1))"
    echo
    continue
  fi

  echo "==> [get]     $local_name  <-  $repo"
  ensure_backend
  # ModelScope is a domestic (CN) source. Bypass the proxy only after the
  # optional dependency installation above; Hugging Face keeps the proxy.
  if [[ "$SOURCE" == "modelscope" ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true
    export NO_PROXY="*" no_proxy="*"
  fi
  case "$SOURCE" in
    modelscope)  py_download modelscope  "$repo" "$out" ;;
    huggingface) py_download huggingface "$repo" "$out" ;;
  esac
  # Mark complete only after the download above succeeded (set -e aborts on failure).
  touch "$marker"
  echo "==> [done]    $local_name -> $out ($(du -sh "$out" 2>/dev/null | cut -f1))"
  echo
done

echo "==> All requested models ready under $DEST"
echo "==> Verify inside a CI container:  ls /data/models/Qwen/"
