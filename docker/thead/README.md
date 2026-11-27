# T-Head PPU: SGLang 0.5.18 two-wheel CI image

[`runtime-0.5.18.containerfile`](runtime-0.5.18.containerfile) assembles the
CI image on the T-Head operations Runtime: the Runtime base (moving tag
`2.3.0`, resolved to a digest at build time) plus one uv resolver install of
`sglang[srt_empty,runtime_common]` and `sglang_fl[thead]`. The image sets no
environment variable and ships no platform artifact.

## Build

```bash
docker/thead/build_two_wheel.sh <image-tag>   # stage + build, prints the record
docker/thead/kernel_fallback.sh <context>     # dormant, see the script header
```

The build resolves the moving Runtime tag to a digest and reads the
constraints of that exact base (`pip freeze --all`, every distribution Python
can see), so the resolver can never replace a Runtime distribution. Wheels and
constraints are build inputs and are never committed; `staging-record.txt` is
the evidence of what went into an image.

## What the image relies on

| Need | Owner | Where |
|---|---|---|
| `PYTHONPATH=/opt/flagtree` | Runtime | `BASH_ENV=/etc/bash_env.sh`, exported by every non-interactive bash; the test steps declare `shell: bash` because container steps default to `sh` |
| PPU collective library | Runtime | torch links the SDK's `libnccl.so.2`; nothing to add to `LD_LIBRARY_PATH` |
| plugin activation (`SGLANG_PLUGINS`, `USE_FLAGGEMS`, `SGLANG_FL_THEAD_MOE_FLAGGEMS`) | tests | `tests/platforms/thead.yaml`, `env_defaults` |
| `sgl_kernel` | plugin | semantic placeholder (`vendor/thead/import_guard.py`); `kernel_fallback.sh` is the documented exit |
| HF offline switches | CI | `.github/configs/thead.yml`, `container_options` |

At job time `.github/scripts/thead/setup.sh` installs the pinned test runner
(`test-requirements.txt`) and proves the installed-wheel contract
(`verify_environment.py`): the plugin under test is the installed wheel, never
the checkout, and the test tooling install left the Runtime inventory untouched.
