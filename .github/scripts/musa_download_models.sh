#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
#
# THROWAWAY helper for the musa model-store population PR (same lineage as
# chore/thead-model-download, PR #118, and the ensure_qwen_models.sh stopgap
# from PR #146). Ensures the four Qwen models the e2e/benchmark tests
# reference are present in the store, downloading the missing ones from
# ModelScope (set SOURCE=huggingface to override).
#
# The musa S5000 runners mount the host's whole /data (see
# .github/configs/musa.yml), so the store root is /data/models and models
# are ensured under /data/models/Qwen/<model> — the exact path the test
# configs read (tests/models/qwen*/*.yaml):
#   - real Qwen/ subtree: writes land inside it
#   - empty store: a real Qwen/ dir is created
#
# Idempotent and cheap when nothing is missing: the fast path only stats
# config.json and exits before any backend install or network work.
# Interrupted downloads are safe: snapshots land in <model>.partial and are
# mv-ed into place only on success, so a cancelled job never leaves a
# half-downloaded dir that looks complete. Safe on stores SHARED across
# runner machines (NFS/PVC): an atomic mkdir lock serializes ensures across
# hosts, and per-attempt unique partial dirs plus a guarded mv keep even a
# lock-bypassing race from corrupting a model dir.
set -euo pipefail

store="${STORE_ROOT:-/data/models}"
DEST="$store/Qwen"
SOURCE="${SOURCE:-modelscope}"   # modelscope | huggingface

# local_dir_name -> repo_id (identical on ModelScope and Hugging Face)
ALL_MODELS=(
  "Qwen3-0.6B:Qwen/Qwen3-0.6B"
  "Qwen3-4B:Qwen/Qwen3-4B"
  "Qwen3.6-35B-A3B:Qwen/Qwen3.6-35B-A3B"
  "Qwen3.6-27B:Qwen/Qwen3.6-27B"
)

# Local copy roots: if a missing model already exists under one of these,
# copy it instead of downloading. Searched as <root>/<Model> and
# <root>/Qwen/<Model> (must contain config.json to count).
LOCAL_ROOTS="${LOCAL_ROOTS:-/data/models/Qwen /data /data/models}"

command -v python3 >/dev/null || { echo "ensure_qwen_models: python3 is required" >&2; exit 1; }

# Optional allow-list via MODELS="Name1 Name2" (matched on local dir name).
if [[ -n "${MODELS:-}" ]]; then
  SELECTED=()
  for entry in "${ALL_MODELS[@]}"; do
    local_name="${entry%%:*}"
    for want in $MODELS; do [[ "$local_name" == "$want" ]] && SELECTED+=("$entry"); done
  done
  [[ ${#SELECTED[@]} -eq 0 ]] && { echo "ensure_qwen_models: no models matched MODELS='$MODELS'" >&2; exit 1; }
  ALL_MODELS=("${SELECTED[@]}")
fi

model_present() { [[ -f "$DEST/$1/config.json" ]]; }

missing=()
for entry in "${ALL_MODELS[@]}"; do
  name="${entry%%:*}"
  model_present "$name" || missing+=("$entry")
done
if [[ ${#missing[@]} -eq 0 ]]; then
  echo "ensure_qwen_models: all ${#ALL_MODELS[@]} required models present under $DEST, nothing to do"
  exit 0
fi
echo "ensure_qwen_models: missing in $DEST: ${missing[*]}"

mkdir -p "$DEST"

# Mount check: if $DEST is not a real mount in this container, downloads land
# on the ephemeral overlay and vanish when the job exits. Warn, don't block.
if command -v mountpoint >/dev/null 2>&1 \
   && ! mountpoint -q "$DEST" 2>/dev/null \
   && ! mountpoint -q "$store" 2>/dev/null \
   && ! mountpoint -q /data 2>/dev/null; then
  echo "ensure_qwen_models: WARNING: neither $DEST, $store nor /data is a mountpoint; downloads will be lost on container exit"
fi

# Serialize concurrent ensures on the same store. The store may be SHARED
# across runner machines (NFS/PVC), where flock does NOT coordinate across
# hosts — use mkdir, which is atomic on local disks and NFS alike. The fast
# path above already exited when nothing is missing, so the lock is only ever
# contended during real downloads.
LOCK_DIR="$DEST/.ensure_models.lock.d"
LOCK_STALE_MIN=10   # a live download holds the lock for minutes, not tens of
LOCK_WAIT_MIN=15    # then degrade to lockless-but-safe rather than fail

find_local_copy() {
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
  # the proxy below: the proxy's CONNECT tunnel returns HTTP 500 for
  # modelscope.cn, while pip can use the proxy to reach PyPI.
  case "$SOURCE" in
    modelscope)   python3 -c "import modelscope" 2>/dev/null || pip install -q modelscope ;;
    huggingface)  python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub ;;
    *) echo "ensure_qwen_models: unknown SOURCE='$SOURCE' (use modelscope|huggingface)" >&2; exit 1 ;;
  esac
  backend_ready=true
}

py_download() {
  # $1 = backend, $2 = repo_id, $3 = local_dir
  python3 - "$1" "$2" "$3" <<'PY'
import sys
backend, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
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

fetch_one() {
  local entry="$1"
  local name="${entry%%:*}"
  local repo="${entry#*:}"
  local out="$DEST/$name"
  # Unique per attempt: if the lock had to be bypassed, concurrent writers on
  # other machines never share a partial dir.
  local partial="$out.partial.$$.$RANDOM"

  # Re-check under the lock: the concurrent job may have just finished it.
  if model_present "$name"; then
    echo "ensure_qwen_models: [skip] $name (completed by a concurrent job)"
    return 0
  fi

  # Under the lock, leftovers from cancelled runs are orphans — sweep them.
  if [[ "$held_lock" == true ]]; then
    rm -rf "$out".partial.* 2>/dev/null || true
  fi

  local local_src
  local_src="$(find_local_copy "$name" || true)"
  if [[ -n "$local_src" && ! "$local_src" -ef "$out" ]]; then
    echo "ensure_qwen_models: [copy] $name <- $local_src"
    mkdir -p "$partial"
    cp -a "$local_src/." "$partial/"
    publish_partial "$partial" "$out" "$name" "copied"
    return 0
  fi

  echo "ensure_qwen_models: [get] $name <- $repo"
  ensure_backend
  # ModelScope is domestic (CN); bypass the proxy only after the optional
  # dependency install above. Hugging Face keeps the proxy.
  if [[ "$SOURCE" == "modelscope" ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true
    export NO_PROXY="*" no_proxy="*"
  fi
  py_download "$SOURCE" "$repo" "$partial"
  publish_partial "$partial" "$out" "$name" "downloaded"
}

publish_partial() {
  # Atomically move a finished partial into place, tolerating a concurrent
  # winner on shared storage (first mv wins; losers discard their partial).
  local partial="$1" out="$2" name="$3" how="$4"
  if [[ -e "$out" ]]; then
    echo "ensure_qwen_models: [skip] $name (raced: completed concurrently)"
    rm -rf "$partial"
    return 0
  fi
  mv "$partial" "$out"
  # Sweep any stray partial a racing mv may have nested inside the final dir.
  rm -rf "$out"/.partial.* 2>/dev/null || true
  touch "$out/.cache_ready"
  echo "ensure_qwen_models: [done] $name ($how, $(du -sh "$out" 2>/dev/null | cut -f1))"
}

held_lock=false
acquire_lock() {
  mkdir "$LOCK_DIR" 2>/dev/null && { held_lock=true; return 0; }
  # Stale lock (holder crashed or was cancelled)? Steal it.
  if [[ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +"$LOCK_STALE_MIN" 2>/dev/null)" ]]; then
    echo "ensure_qwen_models: stealing stale lock (older than ${LOCK_STALE_MIN}m)"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null && { held_lock=true; return 0; }
  fi
  # Someone else is ensuring; poll. If their work completes the missing set,
  # we are done without ever taking the lock.
  local waited=0
  while (( waited < LOCK_WAIT_MIN * 60 )); do
    sleep 10; waited=$((waited + 10))
    local all_present=true entry name
    for entry in "${missing[@]}"; do
      name="${entry%%:*}"
      model_present "$name" || { all_present=false; break; }
    done
    if [[ "$all_present" == true ]]; then
      echo "ensure_qwen_models: concurrent job completed the missing models while we waited"
      exit 0
    fi
    if [[ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +"$LOCK_STALE_MIN" 2>/dev/null)" ]]; then
      echo "ensure_qwen_models: stealing stale lock after waiting ${waited}s"
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR" 2>/dev/null && { held_lock=true; return 0; }
    fi
  done
  # Give up on the lock; proceed with per-attempt unique partial dirs and a
  # guarded mv, which keep concurrent writers from corrupting each other
  # (worst case: duplicate download bytes).
  echo "ensure_qwen_models: WARNING: lock not acquired after ${LOCK_WAIT_MIN}m; proceeding lockless (safe but may duplicate work)"
}

cleanup() { [[ "$held_lock" == true ]] && rm -rf "$LOCK_DIR"; }
trap cleanup EXIT

acquire_lock
for entry in "${missing[@]}"; do
  fetch_one "$entry"
done

echo "ensure_qwen_models: all required models present under $DEST"
