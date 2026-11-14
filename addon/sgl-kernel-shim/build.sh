#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the sgl_kernel import-face shim wheel (py3-none-any, zero deps).
# Distribution name is sgl-kernel-shim — the -shim suffix signals it is the
# import-face stub, not the upstream sgl-kernel (whose pip name it would
# otherwise shadow); the module it ships stays sgl_kernel, the name sglang
# imports. Version = the sglang version it stubs (0.5.18), clean of any
# +local label.
# Usage: build.sh [OUTDIR]   (default: ./dist under the shim source dir)
#
# generate.py stamps the sgl_kernel/ package next to itself, so the build
# runs on a throwaway copy — never stamp into a checkout / working tree.
# Pure Python with no vendor toolchain, so this is a plain script: an
# isolated venv does the pip work, so the caller's interpreter state never
# matters.
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
out="${1:-${src}/dist}"
work="$(mktemp -d "${TMPDIR:-/tmp}/sgl-shim-build.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

cp -a "${src}/." "${work}/src"
python3 -m venv "${work}/venv"
py="${work}/venv/bin/python"

echo ">>> generating sgl_kernel stub package"
(cd "${work}/src" && "${py}" generate.py)

echo ">>> building pure-Python wheel"
mkdir -p "${out}"
"${py}" -m pip wheel "${work}/src" --no-deps -w "${out}"

wheel="$(ls -t "${out}"/sgl_kernel_shim-*.whl 2>/dev/null | head -1)"
if [ -z "$wheel" ]; then
  echo "ERROR: no sgl_kernel_shim wheel produced" >&2
  exit 1
fi
case "$wheel" in
  *-py3-none-any.whl) : ;;
  *) echo "ERROR: expected a py3-none-any wheel, got $(basename "$wheel")" >&2
     exit 1 ;;
esac

echo ">>> built: $(basename "$wheel")"
