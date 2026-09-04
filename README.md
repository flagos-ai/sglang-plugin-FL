# sglang-plugin-FL

sglang-plugin-FL is an out-of-tree (OOT) plugin for [SGLang](https://github.com/sgl-project/sglang), built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends SGLang's inference capabilities across diverse hardware platforms. Without changing SGLang's original interfaces or usage patterns, the same command can run model inference on different chips.

## Overview

SGLang's inference engine relies on NVIDIA-specific components: flashinfer for attention, sgl_kernel for fused CUDA kernels, and NCCL for distributed communication. Running on alternative hardware (Huawei Ascend, Cambricon MLU, Iluvatar, etc.) would otherwise require invasive source modifications.

This plugin provides a non-intrusive adaptation layer through three levels of replacement:

- **Layer 1 — ATen Operators**: Replaces PyTorch's low-level ops (matmul, softmax, embedding, etc.) with FlagGems Triton kernels via PyTorch's dispatch mechanism
- **Layer 2 — SGLang Fused Kernels**: Registers SGLang custom fused ops (SiluAndMul, RMSNorm, RotaryEmbedding) through the BaseFusedOp OOT registry, routing through a standardized dispatch system (aligned with vllm-plugin-FL) to FlagGems, vendor-native, or PyTorch reference implementations
- **Layer 3 — Distributed Communication**: Replaces NCCL-based collectives with CommunicatorFL (backed by FlagCX or torch.distributed), enabling multi-card inference on any hardware

```
┌──────────────────────────────────────────────────────────────┐
│                       SGLang Runtime                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: ATen Ops (flag_gems.enable → PyTorch dispatch)     │
│    torch.mm / torch.add / torch.softmax / ...                │
│      → FlagGems Triton kernels                               │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: SGLang Fused Ops (BaseFusedOp OOT registry)        │
│    SiluAndMul / RMSNorm / RotaryEmbedding                    │
│      → flagos (FlagGems Triton) | vendor (chip-native) | ref │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Communication (AROUND hooks on GroupCoordinator)   │
│    all_reduce / all_gather / reduce_scatter / send / recv    │
│      → CommunicatorFL (FlagCX / torch.distributed)           │
├──────────────────────────────────────────────────────────────┤
│  Triton JIT / Vendor Native → GPU / NPU Kernels              │
└──────────────────────────────────────────────────────────────┘
```

Chip vendors only need to implement a backend class + `register_ops.py`. The dispatch system's auto-discovery mechanism handles the rest. The same vendor implementations work across both sglang-plugin-FL and vllm-plugin-FL.

## Environment

| Package | Version |
|---------|---------|
| SGLang | 0.5.18 |
| sglang-kernel | 0.4.6.post1 |
| PyTorch | 2.13.0+cu130 |
| FlagTree | 0.6.2a1 (Triton 3.6 module) |
| Triton package | Not installed; provided by FlagTree |
| FlagGems | master @ `8ea592557659491930ebf24c24392f958b29ac21` |
| flashinfer | 0.6.17 |
| Python | 3.12 |
| CUDA runtime | 13.0 |

This table is the current **NVIDIA CUDA** target. The MUSA and Ascend images
remain pinned to SGLang v0.5.12 and v0.5.11 respectively until their dedicated
upgrade passes are completed.

The validated H100 environment removes the Triton package installed with
PyTorch and uses FlagTree 0.6.2a1 as its Triton 3.6-compatible compiler. The
verified FlagGems master snapshot handles `to_copy` on H100. CUDA container
integration for this dependency stack is intentionally deferred.

## Model Validation Status

| Model | TP | Status |
|-------|-----|--------|
| Qwen3.6-27B (Hybrid Attention + FLA) | tp=2 | Verified on H100 with v0.5.18 |
| Qwen3.6-35B-A3B (MoE, 256 experts) | tp=2 | Verified on H100 with v0.5.18 |
| Qwen2.5-14B-Instruct | tp=8 | v0.5.18 rerun pending |

The v0.5.18 H100 validation kept both FlagGems ATen replacement and fused-op
dispatch enabled. Qwen3.6-27B passed normal and streaming chat, forced 128-token
decode, and four concurrent requests. Qwen3.6-35B-A3B passed normal chat,
forced 128-token decode, and four concurrent requests. Decode CUDA Graph was
enabled in both TP=2 runs. The Qwen2.5 rerun remains pending because its weights
were not available in the shared model directory.

## Quick Start

### Option A: Standard Install (NVIDIA CUDA)

1. Prepare an SGLang v0.5.18 NVIDIA environment with the dependency versions
   listed above. The official `lmsysorg/sglang:v0.5.18-runtime` image is a
   suitable starting point; NVIDIA communication continues to use NCCL. The
   repository's CUDA containerfile has not yet been upgraded to this stack.

2. Install the plugin itself without replacing the validated dependencies:

```bash
git clone https://github.com/flagos-ai/sglang-plugin-FL
cd sglang-plugin-FL && pip install --no-deps -e .
```

3. (Optional) install [FlagCX](https://github.com/flagos-ai/FlagCX) only when
   FlagCX collectives or PD-disaggregation transfer are required:

```bash
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX && make USE_NVIDIA=1
export FLAGCX_PATH="$PWD"
```

### Option B: Empty Install (Multi-Chip / Non-CUDA)

For running on non-NVIDIA hardware (Ascend, MUSA, etc.), use SGLang's **empty install** mode. This installs SGLang's pure Python code without pulling in CUDA-specific dependencies (torch, torchao, flashinfer, etc.), avoiding conflicts with vendor-specific PyTorch builds.

> **Requires**: [sglang-FL](https://github.com/flagos-ai/sglang-FL) with `srt_empty` support, or upstream SGLang once the [empty PR](https://github.com/sgl-project/sglang/pull/31300) is merged.

```bash
# Step 1: Install the vendor runtime stack (varies by chip)
# Keep the vendor's PyTorch and any operator packages required by the current
# vendor backend; the SGLang empty install will not install them for you.
pip install torch torch_npu       # Ascend
# pip install torch torch_musa    # MUSA
# pip install torch               # NVIDIA (standard)

# Step 2: Install SGLang (empty mode — no torch dependency conflicts)
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_empty]"

# Step 3: Install this plugin
git clone https://github.com/flagos-ai/sglang-plugin-FL
cd sglang-plugin-FL && pip install -e .

# Step 4: Install FlagGems
pip install flag-gems

# Step 5: (Optional) Install FlagCX for distributed communication
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX && make USE_NVIDIA=1  # or USE_ASCEND=1, USE_MUSA=1
export FLAGCX_PATH="$PWD"
```

#### Running with Empty Install

Empty mode does not choose or install a platform's attention, fused-op, or communication packages. Prepare those packages as part of the vendor runtime stack and select an attention backend that has been verified on the target platform. Until FlagOS covers every required operator, keep the platform vendor backend enabled as the functional fallback:

```bash
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero  # FlagGems bug workaround
export ATTENTION_BACKEND=triton                  # NVIDIA or another verified Triton-compatible platform

python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --disable-piecewise-cuda-graph
```

> **Note on `ATTENTION_BACKEND`**:
> - `triton` — SGLang's built-in Triton attention path; use it only after verifying the target platform's Triton-compatible compiler and the complete downstream call path
> - `ascend` — Huawei Ascend optimized attention (use on NPU hardware)
> - If not set, defaults to `flashinfer` on CUDA-like devices — **will fail** in empty environments without flashinfer installed

`SGLANG_FL_DENY_VENDORS` is a coverage-audit switch, not a default Empty-mode setting. For example, `SGLANG_FL_DENY_VENDORS=cuda` intentionally removes `vendor.cuda`. Operators that are not yet implemented by FlagOS or a genuine torch reference are then expected to report `No available implementation`. FLA and Fused MoE currently use the platform vendor backend until their FlagOS implementations are available and validated.

**Why empty install?** In short, SGLang's default `pyproject.toml` hard-pins `torch==2.11.0`, `torchao`, `flashinfer`, etc., which conflicts with vendor-specific PyTorch. Empty install strips these out, letting each vendor bring their own torch.

### Download Models

```bash
# Small model for quick testing (single GPU)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# Larger model for multi-GPU (tp=8)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

If HuggingFace is not accessible, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

Models are cached in `~/.cache/huggingface/hub/` by default. You can also pass a local path to `--model-path`.

### Run a Task

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --disable-piecewise-cuda-graph
```

Multi-GPU with tensor parallelism:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --tp 8 --port 30000 \
    --disable-piecewise-cuda-graph
```

After the server is ready (`The server is fired up and ready to roll`), send a request:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List the first 5 prime numbers."}],
    "temperature": 0
  }' | python -m json.tool
```

### Using Native CUDA Operators

To disable the plugin and use SGLang's original CUDA path:

```bash
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

To disable only the ATen layer (keep fused op dispatch):

```bash
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

## Advanced Configuration

For most use cases, the defaults work out of the box. When you need to customize, use a YAML config file.

### YAML Config File

The plugin ships a [`config/sample.yaml`](sglang_fl/config/sample.yaml) with all available options. Copy it and customize:

```bash
# Copy the sample config
cp $(python -c "from sglang_fl.config import _CONFIG_DIR; print(_CONFIG_DIR / 'sample.yaml')") my_config.yaml

# Edit as needed, then launch with it
SGLANG_FL_CONFIG=./my_config.yaml python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

If `SGLANG_FL_CONFIG` is not set, the plugin uses sensible defaults (equivalent to `prefer: flagos` on CUDA). You only need a YAML file when you want to customize behavior.

#### Config fields

```yaml
# Global backend preference: flagos | vendor | reference
prefer: flagos

# Per-op backend priority (ordered list, first available wins)
op_backends:
  rms_norm: [vendor, flagos, reference]
  silu_and_mul: [flagos, vendor, reference]

# Layer 2 fused ops to skip (fall through to SGLang native CUDA)
# Available: SiluAndMul, RMSNorm, RotaryEmbedding
oot_blacklist:
  - RotaryEmbedding

# Layer 1 ATen ops to exclude from FlagGems Triton replacement
flagos_blacklist:
  - mul
  - sub
```

| Field | Description |
|-------|-------------|
| `prefer` | Global backend preference: `flagos`, `vendor`, `reference` |
| `op_backends` | Per-op ordered backend list (first available wins, can list 1–3 backends) |
| `oot_blacklist` | Layer 2 fused ops to skip from OOT dispatch (fall through to SGLang native CUDA) |
| `flagos_blacklist` | Layer 1 ATen ops to exclude from FlagGems replacement (fall through to PyTorch native) |

#### Common recipes

Each recipe shows a YAML config and expected dispatch result. Use [Dispatch Log](#dispatch-log) to verify.

**1. Skip RotaryEmbedding from OOT dispatch** (fall through to SGLang native CUDA):

```yaml
# my_config.yaml
prefer: flagos
oot_blacklist:
  - RotaryEmbedding
```

Expected dispatch log: only SiluAndMul and RMSNorm appear, no RotaryEmbedding.

**2. Force RMSNorm to use vendor backend, others use flagos:**

```yaml
# my_config.yaml
prefer: flagos
op_backends:
  rms_norm: [vendor, flagos, reference]
```

Expected dispatch log: `RMSNorm → vendor(vendor.nvidia)`, `SiluAndMul → flagos(flagos)`.

**3. Use pure PyTorch reference for all ops** (useful for precision debugging):

```yaml
# my_config.yaml
prefer: reference
```

Expected dispatch log: all ops → `reference(reference)`.

### Environment Variables — Complete Reference

All plugin behavior is controlled via `SGLANG_FL_*` environment variables. They take precedence over YAML config.

**Priority chain:**
```
SGLANG_FL_* env vars > YAML config (SGLANG_FL_CONFIG) > Platform auto-detect YAML > Code defaults
```

#### Layer 2 — Fused Op Dispatch

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_FL_OOT_ENABLED` | `1` | Master switch: `0` disables Layer 2 (keeps Layer 1 ATen active) |
| `SGLANG_FL_PREFER` | `flagos` | Global backend preference: `flagos`, `vendor`, `reference` |
| `SGLANG_FL_PER_OP` | — | Per-op backend priority, e.g. `rms_norm=vendor\|flagos;silu_and_mul=reference` |
| `SGLANG_FL_OOT_BLACKLIST` | — | Skip listed ops from OOT dispatch (comma-separated class names) |
| `SGLANG_FL_OOT_WHITELIST` | — | Only dispatch listed ops (mutually exclusive with BLACKLIST) |
| `SGLANG_FL_STRICT` | `0` | `1` = disable fallback (error if preferred backend unavailable) |
| `SGLANG_FL_DENY_VENDORS` | — | Deny specific vendors (comma-separated, e.g. `cuda,ascend`) |
| `SGLANG_FL_ALLOW_VENDORS` | — | Allow only listed vendors (comma-separated) |
| `SGLANG_FL_DISPATCH_LOG` | — | Path to dispatch log file (records which ops are intercepted) |

#### Layer 1 — ATen Replacement (FlagGems)

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FLAGGEMS` | `1` | Master switch: `0` disables all ATen replacement |
| `SGLANG_FL_FLAGOS_WHITELIST` | — | Only listed ATen ops use FlagGems (comma-separated) |
| `SGLANG_FL_FLAGOS_BLACKLIST` | — | Listed ATen ops don't use FlagGems (comma-separated) |
| `SGLANG_FLAGGEMS_RECORD` | `0` | `1` = record which ATen ops are replaced |
| `SGLANG_FLAGGEMS_LOG_PATH` | — | Path to ATen replacement log file |
| `SGLANG_FLAGGEMS_LOG_ONCE` | `1` | `1` = log each op only once, `0` = log every call |

> `FLAGOS_WHITELIST` and `FLAGOS_BLACKLIST` are mutually exclusive. `FLAGOS_WHITELIST` takes priority over YAML `flagos_blacklist`.

#### Layer 3 — Distributed Communication

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_FL_DIST_BACKEND` | `nccl` | Backend: `nccl` / `hccl` / `flagcx` |
| `FLAGCX_PATH` | — | FlagCX installation path (if set, defaults to `flagcx` backend) |

#### System / Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_FL_CONFIG` | — | Path to YAML config file (overrides platform auto-detect) |
| `SGLANG_FL_PLATFORM` | (auto) | Force platform: `cuda`, `ascend` (overrides auto-detection) |
| `SGLANG_FL_LOG_LEVEL` | `INFO` | Dispatch system log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `SGLANG_PLUGINS` | (all) | SGLang built-in: filter which plugins to load (comma-separated). Not needed — plugin auto-discovered after `pip install` |

#### Examples

```bash
# Force all ops to reference backend (pure PyTorch, useful for precision debugging)
SGLANG_FL_PREFER=reference python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Per-op: RMSNorm uses vendor, others use flagos
SGLANG_FL_PER_OP="rms_norm=vendor|flagos;silu_and_mul=flagos" \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Skip RotaryEmbedding from OOT dispatch (fall through to SGLang native CUDA)
SGLANG_FL_OOT_BLACKLIST=RotaryEmbedding python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Disable ATen layer, keep only fused op dispatch
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Use YAML config with env var override
SGLANG_FL_CONFIG=./my_config.yaml SGLANG_FL_PREFER=reference \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

## Debugging & Diagnostics

### Dispatch Log

See which backend each fused op resolved to (written at server startup):

```bash
rm -f /tmp/dispatch.log
SGLANG_FL_DISPATCH_LOG=/tmp/dispatch.log \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

sort -u /tmp/dispatch.log
# [OOT-DISPATCH] SiluAndMul → flagos(flagos)
# [OOT-DISPATCH] RMSNorm → flagos(flagos)
# [OOT-DISPATCH] RotaryEmbedding → flagos(flagos)
```

### ATen Replacement Log

Record which PyTorch ATen ops were replaced by FlagGems:

```bash
rm -f /tmp/gems_aten.txt
SGLANG_FLAGGEMS_RECORD=1 SGLANG_FLAGGEMS_LOG_PATH=/tmp/gems_aten.txt \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# After first inference request:
sort -u /tmp/gems_aten.txt
```

> The log uses `_AtenOnlyFilter` to record only `flag_gems.ops.*` namespace calls, excluding internal FlagGems calls triggered by Layer 2 implementations.

### Precision Bisection

When numerical differences appear, isolate the responsible layer:

```bash
# Step 1: Disable everything — confirm vanilla SGLang works
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 2: Enable only Layer 2 (fused ops), disable ATen replacement
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 3: Per-op isolation — only SiluAndMul uses flagos, RMSNorm uses reference
USE_FLAGGEMS=0 \
SGLANG_FL_PER_OP="silu_and_mul=flagos;rms_norm=reference" \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 4: Disable Layer 2, only ATen replacement active
SGLANG_FL_OOT_ENABLED=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 5: Gradually enable ATen ops with whitelist
SGLANG_FL_OOT_ENABLED=0 SGLANG_FL_FLAGOS_WHITELIST=rms_norm,silu \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

> If output diverges at Step N but not Step N-1, the responsible layer/op is isolated.

### Common Issues

| Symptom | Cause & Fix |
|---------|-------------|
| `dispatch.log` is empty | Plugin not loaded — check `pip show sglang_fl` |
| `gems_aten.txt` is empty | `USE_FLAGGEMS=0` is set, or `SGLANG_FL_FLAGOS_WHITELIST` excludes the op |
| `forward_cuda` error on non-NVIDIA | An op lacks OOT registration — register it or add to whitelist |
| `ImportError: sgl_kernel` | Normal on non-CUDA — the OOT dispatch bypasses `forward_cuda` |
| tp>1 hangs at startup | Check GPU count, NCCL env vars, model TP compatibility |
| OOM at engine startup | Reduce `--mem-fraction-static` (default 0.5) |

## Vendor Integration

Chip vendors integrate by adding a backend directory under `dispatch/backends/vendor/`:

```bash
cp -r sglang_fl/dispatch/backends/vendor/template/ \
      sglang_fl/dispatch/backends/vendor/my_chip/
```

You need to implement two files:

### 1. Backend class (`my_chip.py`)

```python
from sglang_fl.dispatch.backends import Backend


class MyChipBackend(Backend):
    _available = None

    @property
    def name(self) -> str:
        return "my_chip"

    def is_available(self) -> bool:
        if MyChipBackend._available is None:
            try:
                import my_chip_sdk
                MyChipBackend._available = my_chip_sdk.device_count() > 0
            except ImportError:
                MyChipBackend._available = False
        return MyChipBackend._available

    def silu_and_mul(self, obj, x):
        from .impl.activation import silu_and_mul_my_chip
        return silu_and_mul_my_chip(obj, x)

    def rms_norm(self, obj, x, residual=None):
        from .impl.normalization import rms_norm_my_chip
        return rms_norm_my_chip(obj, x, residual)

    def rotary_embedding(self, obj, query, key, cos, sin, position_ids,
                         rotary_interleaved=False, inplace=True):
        from .impl.rotary import rotary_embedding_my_chip
        return rotary_embedding_my_chip(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )
```

### 2. Registration (`register_ops.py`)

```python
import functools
from sglang_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    from .my_chip import MyChipBackend

    backend = MyChipBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
    ]
    registry.register_many(impls)
```

### 3. Operator implementations (`impl/`)

Each op function receives standardized arguments (same as vllm-plugin-FL):

| Op | Signature |
|----|-----------|
| `silu_and_mul` | `fn(obj, x: Tensor) -> Tensor` |
| `rms_norm` | `fn(obj, x: Tensor, residual: Optional[Tensor] = None) -> Tensor \| tuple[Tensor, Tensor]` |
| `rotary_embedding` | `fn(obj, query, key, cos, sin, position_ids, rotary_interleaved=False, inplace=True) -> tuple[Tensor, Tensor]` |

The `obj` parameter provides access to layer attributes (`obj.weight`, `obj.variance_epsilon`, etc.). These attribute names are identical between SGLang and vLLM, so the same impl works for both frameworks.

### Auto-discovery

The plugin scans `dispatch/backends/vendor/*/register_ops.py` at startup. If `is_available()` returns True, the vendor's ops are registered. No other files need modification.

### Existing Backends

| Vendor | Directory | Hardware Detection |
|--------|-----------|-------------------|
| NVIDIA CUDA | `vendor/cuda/` | `sgl_kernel` importable |
| Huawei Ascend | `vendor/ascend/` | `torch_npu` importable |
| Template | `vendor/template/` | Always False (reference only) |

## Project Structure

```
sglang_fl/
├── pyproject.toml                    # Package config + entry_points registration
└── sglang_fl/
    ├── __init__.py                   # Plugin entry: FlagGems + dispatch init + communicator hooks
    ├── platform.py                   # PlatformFL (device identity, memory, graph capture)
    ├── distributed/                  # Communication module (aligned with vllm-plugin-FL)
    │   ├── __init__.py
    │   ├── communicator.py           # CommunicatorFL (FlagCX / torch.distributed wrapper)
    │   └── device_communicators/
    │       └── flagcx.py             # FlagCX-specific communicator
    ├── config/
    │   ├── __init__.py               # YAML config loader with platform auto-detection
    │   ├── sample.yaml               # Full example config with all options documented
    │   ├── nvidia.yaml               # NVIDIA CUDA platform defaults
    │   └── ascend.yaml               # Ascend platform defaults (with blacklists)
    └── dispatch/                     # Op dispatch system (aligned with vllm-plugin-FL)
        ├── __init__.py               # Public API: call_op(), resolve_op()
        ├── types.py                  # OpImpl, BackendImplKind, BackendPriority
        ├── registry.py               # Thread-safe OpRegistry
        ├── policy.py                 # SelectionPolicy + env var / YAML config
        ├── manager.py                # OpManager: resolve, call, cache, fallback
        ├── builtin_ops.py            # Registration orchestrator
        ├── ops.py                    # FLBackendBase ABC (op signature definitions)
        ├── logger_manager.py         # Logging with SGLANG_FL_LOG_LEVEL
        ├── bridge/                   # SGLang ↔ dispatch parameter translation
        │   ├── __init__.py
        │   ├── silu_and_mul.py       # forward_cuda(self, x) → call_op("silu_and_mul", obj, x)
        │   ├── rms_norm.py           # Handles post_residual_addition
        │   └── rotary_embedding.py   # Extracts cos/sin from cos_sin_cache, handles offsets
        └── backends/
            ├── __init__.py           # Backend ABC
            ├── flaggems/             # DEFAULT backend (FlagGems Triton kernels)
            │   ├── flaggems.py
            │   ├── register_ops.py
            │   └── impl/             # activation.py, normalization.py, rotary.py
            ├── reference/            # REFERENCE backend (PyTorch native, always available)
            │   ├── reference.py
            │   ├── register_ops.py
            │   └── impl/             # activation.py, normalization.py, rotary.py
            └── vendor/               # VENDOR backends (auto-discovered)
                ├── ascend/           # Huawei Ascend NPU (torch_npu)
                ├── cuda/             # NVIDIA CUDA (sgl_kernel)
                └── template/         # Template for new vendors
```

## How It Works

### Plugin Loading

The plugin registers two entry_points in `pyproject.toml`:

```toml
[project.entry-points."sglang.srt.plugins"]
sglang_fl = "sglang_fl:load_plugin"

[project.entry-points."sglang.srt.platforms"]
sglang_fl = "sglang_fl:activate_platform"
```

SGLang discovers and loads the plugin automatically at startup via setuptools entry_points.

### Fused-op Dispatch

On SGLang v0.5.18, the core mechanism registers bridges with
`BaseFusedOp.register_oot_forward()` and combines them with the standardized
dispatch system. Older SGLang versions use the legacy
`MultiPlatformOp.dispatch_forward()` AROUND hook.

```
Plugin startup
  → Check OOT_WHITELIST/OOT_BLACKLIST
  → Register RMSNorm → rms_norm_bridge for the active platform key
BaseFusedOp.forward() called
  → OOT registry selects rms_norm_bridge
  → SGLang calls the bridge function with framework args:
      rms_norm_bridge(self, x, residual, post_residual_addition)
    → Bridge handles SGLang-specific params (post_residual_addition → merge into residual)
    → Bridge calls dispatch.call_op("rms_norm", obj, x, residual)
      → OpManager resolves best impl via policy (flagos > vendor > reference)
      → Calls the selected backend: rms_norm_flaggems(obj, x, residual)
```

The bridge layer decouples framework-specific parameters from the standardized op signatures. Vendor backends only need to implement the standard signatures — the same impl works for both sglang-plugin-FL and vllm-plugin-FL.

### Dispatch Architecture (shared with vllm-plugin-FL)

```
┌─────────────────────────────────────────────────────────────┐
│  SGLang BaseFusedOp OOT    │  vLLM forward_oot override     │
│  (bridge/rms_norm.py)      │  (vllm_fl/ops/layernorm.py)    │
└────────────┬───────────────┴────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│  dispatch.call_op("rms_norm", obj, x, residual)             │
│  OpManager → SelectionPolicy → OpRegistry → resolve impl    │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌───────────┐  ┌──────────────┐
   │ DEFAULT     │  │ VENDOR    │  │ REFERENCE    │
   │ (FlagGems)  │  │ (Ascend/  │  │ (PyTorch)    │
   │ priority=150│  │  CUDA)    │  │ priority=50  │
   │             │  │ priority= │  │              │
   │             │  │   100     │  │              │
   └─────────────┘  └───────────┘  └──────────────┘
```

Chip vendors implement the **same backend interface** for both frameworks. The only framework-specific code is the bridge layer, which is maintained by the plugin.

### ATen Replacement

```
Plugin loads → flag_gems.enable(record=True)
  → PyTorch dispatch table registers Triton kernels for ATen ops
  → On first inference call, each replaced op is logged
  → _AtenOnlyFilter ensures only flag_gems.ops.* calls are recorded
    (excludes internal FlagGems calls from Layer 2 flagos implementations)
```

## Known Issues

- **Piecewise CUDA Graph not supported**: FlagGems Triton kernels contain `logging.Logger` calls that are incompatible with `torch.compile` (used by SGLang's piecewise CUDA graph). Use `--disable-piecewise-cuda-graph` when launching the server. Regular CUDA graph capture works normally.

## License

Apache-2.0
