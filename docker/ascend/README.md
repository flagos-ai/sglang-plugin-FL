# Ascend 910C: SGLang 0.5.18 two-wheel CI image

[`runtime-0.5.18.containerfile`](runtime-0.5.18.containerfile) assembles the
CI image on the Ascend operations Runtime (`flagos-runtime-ascend-cann9.2.1-910c`,
moving tag `2.3.0`, resolved to a digest at build time): the Runtime base plus
one uv resolver install of `sglang[srt_empty,runtime_common]` and
`sglang_fl[ascend]`. The image sets no environment variable and ships no
platform artifact.

## The sgl_kernel_npu gap

The Runtime ships no `sgl_kernel_npu` wheel. SGLang 0.5.18 imports it
module-level while loading the NPU attention backends and the MoE top-k, so
the plugin wheel carries the gap (`vendor/ascend/`):

- `import_guard.py` — a placeholder finder (real wheel always wins) that lets
  the imports succeed; unknown operators fail loudly when called;
- `sgl_kernel_npu_fallbacks.py` — real implementations for the operators the
  CI matrix calls: SGLang's own Triton GDN kernels (chunk, gating, fused
  decode) and plain-PyTorch `l1_norm` / `npu::causal_conv1d`;
- `early.py` — installs both during platform discovery, before SGLang imports
  its layers.

Not implemented, by the same policy as Hygon: the MTP/speculative operators
(`fused_gdn_gating_kernel_without_sigmoid`, `causal_conv1d_verify`,
`mamba_state_update_triton`, the `kda_*` family, `solve_tril`). The Ascend CI
matrix (`tests/platforms/ascend.yaml`) is nograph eager without MTP and does
not reach them; the placeholder raises on first use instead of mis-answering.
`torchair` (graph mode only) is likewise absent on this Runtime; every CI
case runs `*_nograph` and the plugin's compile-backend hook raises loudly if
graph mode is requested. Both gaps are Runtime-owned.

## Build

```bash
docker/ascend/build_two_wheel.sh <image-tag>   # stage + build, prints the record
```

The build resolves the moving Runtime tag to a digest and reads the
constraints of that exact base (`pip freeze --all`, every distribution Python
can see), so the resolver can never replace a Runtime distribution. Wheels and
constraints are build inputs and are never committed; `staging-record.txt` is
the evidence of what went into an image. `SGLANG_INPUT_WHEEL=<path>` reuses a
SGLang wheel already built from the pinned commit.

## What the image relies on

| Need | Owner | Where |
|---|---|---|
| `/opt/flagtree` compiler slot (`BASH_ENV=/etc/bash_env.sh` sources it into `sys.path`) | Runtime | exported by every non-interactive bash; the test steps declare `shell: bash` because container steps default to `sh` |
| CANN / torch_npu / FlagGems / FlagCX / shmem | Runtime | printed by the job-time verifier, never pinned |
| plugin activation | tests | SGLang entry points, loaded once `torch_npu` reports devices |
| `sgl_kernel_npu` | plugin | placeholder + fallbacks (`vendor/ascend/`) |
| HF offline switches | CI | `.github/configs/ascend.yml`, `container_options` |

At job time `.github/scripts/ascend/setup.sh` installs the pinned test runner
(`test-requirements.txt`) and proves the installed-wheel contract
(`verify_environment.py`): the plugin under test is the installed wheel, never
the checkout, and the test tooling install left the Runtime inventory
untouched.
