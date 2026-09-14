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
# store, not an ephemeral overlay. Idempotent: per-model .cache_ready markers
# make re-runs skip completed models.
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

# Pre-install the download backend WHILE THE CI PROXY IS STILL ACTIVE — pip
# needs the proxy to reach pypi, but the modelscope download itself must
# bypass it (see below). Installing first avoids a proxy-less pip fallback
# later inside py_download.
if [[ "$SOURCE" == "modelscope" ]]; then
  python3 -c "import modelscope" 2>/dev/null || \
    pip install -q modelscope
else
  python3 -c "import huggingface_hub" 2>/dev/null || \
    pip install -q huggingface_hub
fi

# ModelScope is a domestic (CN) source. The CI container routes egress through
# an HTTP proxy that returns 500 on the CONNECT tunnel to modelscope.cn,
# breaking the download ("Unable to connect to proxy ... 500"). Bypass the
# proxy for ModelScope (direct domestic egress). Hugging Face (foreign) keeps
# the proxy, since it usually needs it.
if [[ "$SOURCE" == "modelscope" ]]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true
  export NO_PROXY="*" no_proxy="*"
fi

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

for entry in "${ALL_MODELS[@]}"; do
  local_name="${entry%%:*}"
  repo="${entry#*:}"
  out="$DEST/$local_name"
  marker="$out/.cache_ready"

  # Skip only on a completion marker we write after a fully successful download.
  # (Checking for "any file present" would falsely skip a directory left behind
  # by an interrupted/timed-out run, serving a half-downloaded model.)
  if [[ -f "$marker" ]]; then
    echo "==> [skip]    $local_name already complete (.cache_ready present, $(du -sh "$out" 2>/dev/null | cut -f1))"
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
  case "$SOURCE" in
    modelscope)  py_download modelscope  "$repo" "$out" ;;
    huggingface) py_download huggingface "$repo" "$out" ;;
    *) echo "Unknown SOURCE='$SOURCE' (use modelscope|huggingface)"; exit 1 ;;
  esac
  # Mark complete only after the download above succeeded (set -e aborts on failure).
  touch "$marker"
  echo "==> [done]    $local_name -> $out ($(du -sh "$out" 2>/dev/null | cut -f1))"
  echo
done

echo "==> All requested models ready under $DEST"
echo "==> Verify inside a CI container:  ls /data/models/Qwen/"
