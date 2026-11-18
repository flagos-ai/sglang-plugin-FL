#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Prepare the NVIDIA two-wheel image for CI validation.

set -euo pipefail
git config --global --add safe.directory "$(pwd)"

echo "=== Selecting the Runtime's default Triton 3.6 slot ==="
# The operations Runtime keeps compiler implementations in side directories
# and its login-shell profile already selects the default slot:
# PYTHONPATH=/opt/flagtree (Triton 3.6.0, provided by FlagTree). No pip
# triton distribution exists in this image -- FlagGems 5.4.0 is only
# compatible with the 3.6 launcher path, so the /opt/triton 3.7.1 tree must
# stay untouched and unselected.
source /etc/profile.d/zz-flagos.sh
if [ "${PYTHONPATH:-}" != "/opt/flagtree" ]; then
    echo "FATAL: the Runtime profile must select /opt/flagtree, PYTHONPATH=${PYTHONPATH:-<unset>}" >&2
    exit 1
fi
echo "compiler slot active: PYTHONPATH=$PYTHONPATH"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "PYTHONPATH=$PYTHONPATH" >> "$GITHUB_ENV"
fi

echo "=== Enabling FlagGems dispatch ==="
# Matches the platform config's env_defaults (tests/platforms/nvidia.yaml):
# torch ops dispatch through the Runtime's own FlagGems 5.4.0 via the
# /opt/flagtree Triton module.
export USE_FLAGGEMS=1
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "USE_FLAGGEMS=1" >> "$GITHUB_ENV"
fi
echo "FlagGems enabled: USE_FLAGGEMS=$USE_FLAGGEMS"

echo "=== Installing test runner (ephemeral) ==="
python3 -m pip install --break-system-packages \
    pytest==9.1.1 pytest-timeout==2.4.0

echo "=== Selecting installed-wheel test mode ==="
# SGLang and sglang-plugin-FL are real wheels baked into the image. Export
# the installed-wheel test mode so every later pytest/engine subprocess
# resolves sglang_fl from site-packages instead of the checkout source
# tree that CI happens to be running in.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SGLANG_FL_TEST_SOURCE=wheel" >> "$GITHUB_ENV"
fi
export SGLANG_FL_TEST_SOURCE=wheel

# Strict post-setup verification: no standalone triton distribution with
# the module resolving from /opt/flagtree (3.6.0), FlagGems 5.4.0 still the
# Runtime-original distribution, the plugin loading from site-packages with
# discoverable entry points, runner versions, and (on the GPU runner) the
# plugin CUDA view that FlashInfer's JIT must consume. The plugin activates
# its cache-local CUDA compatibility view inside every Python process;
# setup.sh must not patch CUDA_HOME, create symlinks, or replace the
# installed plugin with an editable checkout.
python3 .github/scripts/nvidia/verify_environment.py \
    --phase post-setup --min-gpus 4
echo "=== Installation complete ==="
