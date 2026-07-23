# Tests

Platform-agnostic test suite for the sglang-plugin-FL project. All tests are designed to run on NVIDIA CUDA (and future Ascend NPU) without modification.

## Architecture Overview

```
                          ┌──────────────┐
                          │  tests/run.py │  ← Unified entry point
                          └──────┬───────┘
                                 │
          ┌──────────────────┬───┴───────────────┬──────────────────┐
          │                  │                   │                  │
    ┌─────▼─────┐   ┌───────▼───────┐   ┌────────▼────────┐   ┌──────▼──────┐
    │ Unit Tests │   │  Functional   │   │   E2E Tests     │   │ Benchmarks  │
    │ (no GPU)   │   │  (GPU, no     │   │ (GPU + models)  │   │ (GPU + dummy│
    │            │   │   models)     │   │                 │   │  weights)   │
    └────────────┘   └───────────────┘   └────────┬────────┘   └─────────────┘
                                                           │
                                          ┌────────────────┼────────────────┐
                                          │                │                │
                                    ┌─────▼────┐    ┌──────▼─────┐   ┌──────▼─────┐
                                    │Inference │    │  Serving   │   │ Concurrent │
                                    │(Engine)  │    │ (HTTP API) │   │(async gen) │
                                    └──────────┘    └────────────┘   └────────────┘
```

**Key design principles:**

- **YAML-driven E2E tests**: Model configs in `tests/models/` drive inference,
  serving, and concurrent smoke tests — no per-model test files needed.
- **Platform-aware orchestration**: Platform YAML configs (`tests/platforms/`)
  control which tests run on which hardware, inject tolerance, and set env
  defaults.
- **Process isolation**: Each E2E case runs in its own subprocess via `run.py`,
  so device state is fresh between cases.
- **Benchmark smoke tests**: `tests/benchmarks/configs/smoke.yaml` drives tiny
  health checks over `sglang.bench_offline_throughput` / `bench_one_batch` /
  `bench_serving`, using `load_format: dummy` so no real weights are needed.

## Directory Structure

```
tests/
├── run.py                          # Unified test runner (wraps pytest)
├── conftest.py                     # Root: custom marker registration
├── validate.sh                     # Precision alignment: baseline vs plugin
├── test_precision_align.py         # baseline/plugin/compare subcommands
├── test_e2e_config.py              # Env-var & YAML config e2e validation
│
├── models/                         # Model YAML configs (drive E2E tests)
│   ├── qwen3/
│   │   ├── 4b_tp2.yaml
│   │   └── 06b_tp1.yaml
│   └── qwen3_6/
│       ├── 27b_tp4_nograph.yaml            (+ _vl / _old variants)
│       └── 35b_a3b_tp4_nograph.yaml        (+ _vl / _old variants)
│
├── platforms/                      # Platform-specific test configs
│   ├── cuda.yaml                   # NVIDIA GPU: device types, tolerance, matrix
│   └── template.yaml               # Template for adding new platforms
│
├── e2e_tests/                      # End-to-end tests (require GPU + model files)
│   ├── inference/
│   │   └── test_inference_smoke.py # Unified inference test (YAML-driven)
│   ├── serving/
│   │   ├── test_serving_smoke.py   # Unified serving test (YAML-driven)
│   │   └── server_helper.py        # SGLangServer lifecycle manager
│   └── concurrent/
│       └── test_concurrent_smoke.py# N-way async generation (text/vl/mixed)
│
├── functional_tests/               # Component-level GPU tests (no model files)
│   ├── conftest.py                 # `device` fixture (cuda, skip if absent)
│   ├── ops/
│   │   └── test_ops_correctness.py # dispatch.call_op vs reference backend
│   ├── compilation/
│   │   └── test_graph_capture.py   # Graph capture & replay
│   └── distributed/
│       └── test_collective_ops.py  # Collective ops (all_reduce, etc.)
│
├── unit_tests/                     # Fast isolated tests (no GPU required)
│   ├── conftest.py                 # (per-subpackage conftests)
│   ├── dispatch/                   # Op dispatch system
│   │   ├── test_call_op.py
│   │   ├── test_registry.py
│   │   ├── test_manager.py
│   │   ├── test_policy.py
│   │   ├── test_types.py
│   │   ├── test_env_policy.py
│   │   ├── test_cuda_compat_vendors.py
│   │   ├── test_fork_safety.py
│   │   ├── backends/test_reference_ops.py
│   │   └── bridge/                  # Operator bridge unit tests
│   │       ├── test_silu_and_mul.py
│   │       ├── test_rms_norm.py
│   │       ├── test_gemma_rms_norm.py
│   │       ├── test_rotary_embedding.py
│   │       ├── test_mrotary_embedding.py
│   │       ├── test_fla_ops.py
│   │       ├── test_fused_moe.py
│   │       └── test_topk.py
│   ├── distributed/                # Distributed communication
│   │   ├── test_communicator.py
│   │   └── test_flagcx.py
│   ├── platform/                    # Plugin platform/backend loading
│   │   ├── test_load_plugin.py
│   │   ├── test_init_backend.py
│   │   └── test_apply_vendor_patches.py
│   └── flaggems/                    # FlagGems integration
│       └── test_setup_flaggems.py
│
├── benchmarks/                     # Benchmark smoke tests (dummy weights)
│   ├── configs/smoke.yaml          # throughput / latency / serve cases
│   ├── test_benchmark_throughput.py
│   ├── test_benchmark_latency.py
│   ├── test_benchmark_serve.py
│   └── utils.py                    # CLI-arg builder, JSONL reader, env passthrough
│
└── utils/                          # Shared test utilities
    ├── model_config.py             # YAML model config loader (SGLang names)
    └── platform_config.py          # Platform YAML config loader
```

## Running Tests

### Prerequisites

```bash
# Editable install of the plugin (provides setuptools entry_points auto-discovery)
pip install -e .

# Optional but recommended
pip install pytest pyyaml requests openai
```

For E2E / benchmark tests that load real weights, point model YAMLs at local
model paths (the `llm.model` field in each `tests/models/<m>/<c>.yaml`). Cases
whose model path does not exist are auto-skipped with a clear message, so a
missing model never fails the run.

### Via `run.py` (recommended for CI and full runs)

`run.py` is the unified entry point: it resolves the platform config, applies
env defaults, injects tolerance, discovers test cases, and runs each in a
subprocess. `--platform` accepts either a platform name (`cuda`) or a device
alias (`a100`) — aliases are resolved against the `device_types` keys of each
platform YAML.

```bash
# Run everything for a platform/device
python tests/run.py --platform cuda --device a100

# Run only unit tests (no GPU required)
python tests/run.py --platform cuda --device a100 --scope unit

# Run only functional tests (ops, compilation, distributed)
python tests/run.py --platform cuda --device a100 --scope functional

# Run only E2E tests (inference + serving + concurrent)
python tests/run.py --platform cuda --device a100 --scope e2e

# Run only benchmark smoke tests
python tests/run.py --platform cuda --device a100 --scope benchmark

# Run a specific E2E task/model/case
python tests/run.py --platform cuda --device a100 \
    --scope e2e --task inference --model qwen3_6 --case 27b_tp4_nograph

# Run a specific benchmark type
python tests/run.py --platform cuda --device a100 \
    --scope benchmark --benchmark latency

# Pass extra pytest args after '--'
python tests/run.py --platform cuda --device a100 --scope unit -- -x -v

# Device alias instead of explicit platform/device
python tests/run.py --platform a100 --scope unit
```

### Via `pytest` directly (for development)

```bash
# Unit tests (no GPU required)
pytest tests/unit_tests/ -v

# Functional tests (requires GPU)
pytest tests/functional_tests/ -v -s

# Single E2E inference test (env vars injected by run.py, but settable by hand)
FL_TEST_MODEL=qwen3 FL_TEST_CASE=06b_tp1 \
    pytest tests/e2e_tests/inference/test_inference_smoke.py -v -s

# Single E2E serving test
FL_TEST_MODEL=qwen3_6 FL_TEST_CASE=27b_tp4_nograph \
    pytest tests/e2e_tests/serving/test_serving_smoke.py -v -s

# Single E2E concurrent test
FL_TEST_MODEL=qwen3_6 FL_TEST_CASE=27b_tp4_nograph \
    pytest tests/e2e_tests/concurrent/test_concurrent_smoke.py -v -s

# Benchmark smoke tests are env-driven too (FL_BENCHMARK_CASE is a JSON blob
# assembled by run.py — prefer going through run.py for these)
python tests/run.py --platform cuda --device a100 --scope benchmark --benchmark latency
```

### Filter by markers

```bash
pytest -m gpu            # Only GPU tests
pytest -m functional     # Only functional tests
pytest -m e2e            # Only E2E tests
pytest -m benchmark       # Only benchmark tests
pytest -m "not slow"
```

### Precision alignment via `validate.sh`

`tests/validate.sh` is a standalone acceptance script comparing original SGLang
(no plugin) against sglang-plugin-FL (FlagGems + OOT dispatch), with an optional
mock-vendor mode. It runs `test_precision_align.py` subcommands and writes logs
to `/tmp/` plus `precision_results[_tpN].json` next to the script.

```bash
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct bash tests/validate.sh all

# Multi-GPU
TP_SIZE=8 MODEL_PATH=/path/to/14B bash tests/validate.sh full   # incl. vendor mode

# Subcommands
MODEL_PATH=... bash tests/validate.sh clean     # remove previous logs
MODEL_PATH=... bash tests/validate.sh baseline  # original SGLang, no plugin
MODEL_PATH=... bash tests/validate.sh plugin   # sglang-plugin-FL (flagos)
MODEL_PATH=... bash tests/validate.sh vendor   # mock_npu vendor backend
MODEL_PATH=... bash tests/validate.sh compare   # precision + dispatch diff
```

### Config / env-var validation via `test_e2e_config.py`

A self-contained script that boots an SGLang server per env-var/YAML-config
combination and verifies dispatch logging, backend selection, and inference
output.

```bash
MODEL_PATH=/path/to/model python tests/test_e2e_config.py          # all cases
MODEL_PATH=/path/to/model python tests/test_e2e_config.py --list   # list cases
MODEL_PATH=/path/to/model python tests/test_e2e_config.py --case 3 # one case
```

## Model YAML Config Format

E2E tests are driven by YAML configs under `tests/models/<model>/<case>.yaml`.
A single case can drive multiple e2e tasks (inference, serving, concurrent)
simultaneously. Parameter names are SGLang-native.

### Text model example

```yaml
# tests/models/qwen3/06b_tp1.yaml
llm:
  model: "/data/models/Qwen/Qwen3-0.6B"
  context_length: 8192
  mem_fraction_static: 0.7
  trust_remote_code: true

generate:
  prompts:
    - text: "How many states are there in the United States?"
      expected: "50"                    # Optional: substring must appear in output
    - text: "The capital of France is"
      expected: "Paris"
  sampling:
    max_new_tokens: 16
    temperature: 0.0
  parametrize:                          # List of explicit engine-param overrides
    - disable_cuda_graph: true
      disable_piecewise_cuda_graph: true
      dtype: "bfloat16"
```

### Multimodal (image) model example

```yaml
# tests/models/qwen3_6/35b_a3b_tp4_nograph_vl.yaml
llm:
  model: "/data/models/Qwen/Qwen3.6-35B-A3B"
  tp_size: 2
  context_length: 8192
  mem_fraction_static: 0.95
  disable_cuda_graph: true
  disable_piecewise_cuda_graph: true
  trust_remote_code: true

generate:
  modality: image                       # text (default) | audio | image | video
  prompts:
    - image: "examples/test_images/red_square.jpg"
      question: "What color is shown in this image? Answer with one word."
      expected: ["red"]
  sampling:
    max_new_tokens: 64
    temperature: 0.0
```

### Concurrent / serving in the same case

A case YAML may also carry `serve:` and `concurrent:` sections so that one file
drives inference, serving, and concurrent tasks.

```yaml
serve:
  served_model_name: "qwen"
  endpoints: ["chat"]                   # completion | chat | embedding
  stream: false
  max_tokens: 256
  chat_messages:
    - role: "user"
      content: "Introduce yourself,please"
  sampling:
    temperature: 0.0

concurrent:
  modes: ["text", "vl", "mixed"]        # subsets of: text / vl / mixed
  concurrent_n: 4
  text:
    prompts:
      - text: "How many states are there in the United States?"
        expected: ["50"]
  vl:
    cases:
      - image: "examples/test_images/cat.jpg"
        question: "What animal is in this image? Answer with one word."
        expected: ["cat"]
  sampling:
    text: { max_new_tokens: 256, temperature: 0.0 }
    vl:   { max_new_tokens: 64,  temperature: 0.0 }
```

### Config fields reference

| Field | Type | Required | Description |
|---|---|---|---|
| `llm.model` | str | Yes | Model path (case auto-skips if not found) |
| `llm.*` | dict | Yes | SGLang Engine kwargs (`tp_size`, `context_length`, `mem_fraction_static`, `disable_cuda_graph`, `trust_remote_code`, …) |
| `generate.modality` | str | No | `text` (default), `audio`, `image`, `video` |
| `generate.prompts` | list | Yes | Strings or dicts with `text`/`expected` (text) or `image`/`question`/`expected` (VL) |
| `generate.sampling` | dict | Yes | SGLang sampling params (`max_new_tokens`, `temperature`, …) |
| `generate.parametrize` | list | No | Explicit engine-param override combos (one Engine per combo) |
| `serve.endpoints` | list | No | `completion`, `chat`, `embedding` |
| `serve.stream` | bool | No | Use OpenAI SDK streaming for chat (default `false`) |
| `serve.served_model_name` | str | No | Override `--served-model-name` |
| `serve.api_key` | str | No | API key for authenticated endpoints |
| `serve.max_tokens` | int | No | Max tokens for serving requests (default 50) |
| `serve.chat_messages` | list | No | Messages for `/v1/chat/completions` |
| `serve.completion_prompt` | str | No | Prompt for `/v1/completions` |
| `serve.extra_engine` | dict | No | Engine param overrides for serving only |
| `serve.extra_body` | dict | No | Extra body fields merged into chat payload |
| `concurrent.modes` | list | No | `text`, `vl`, `mixed` |
| `concurrent.concurrent_n` | int | No | Async request count (default 4) |
| `concurrent.text.prompts` | list | No | Text cases with `text`/`expected` |
| `concurrent.vl.cases` | list | No | VL cases with `image`/`question`/`expected` |

## Platform Config Format

Platform configs (`tests/platforms/<platform>.yaml`) define device types,
tolerance, env defaults, and the per-device test matrix.

```yaml
# tests/platforms/cuda.yaml
platform: cuda
vendor: nvidia

device_types:
  a100: { compute_capability: "8.0", memory_gb: 80, tags: [ampere, bf16] }
  a800: { compute_capability: "8.0", memory_gb: 80, tags: [ampere, bf16, china-variant] }
  h100: { compute_capability: "9.0", memory_gb: 80, tags: [hopper, fp8, bf16] }

# Tolerance resolution: device_overrides -> platform default -> built-in default
tolerance:
  inference:
    default: {exact: true}
device_overrides:
  h100:
    tolerance:
      inference:
        default: {rtol: 1e-4, atol: 1e-7}

# Model names containing any token here are skipped.
unsupported_features: []

# Applied via os.environ.setdefault before pytest (existing env wins).
env_defaults:
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  NCCL_ALGO: "Ring"

# Per-device test matrix. Top-level key must match a device_types entry.
a100:
  name: "a100"
  tests:
    e2e:
      concurrent:
        qwen3_6: ["35b_a3b_tp4_nograph", "27b_tp4_nograph"]
      inference:
        qwen3_6: ["35b_a3b_tp4_nograph", "27b_tp4_nograph"]
      serving:
        qwen3:   ["4b_tp2"]
        qwen3_6: ["35b_a3b_tp4_nograph", "27b_tp4_nograph"]
    functional: { include: "*", exclude: [] }
    unit:       { include: "*", exclude: [] }
    benchmark:
      enabled: true
      smoke: ["throughput", "latency", "serve"]
```

## Benchmark Smoke Config Format

`tests/benchmarks/configs/smoke.yaml` lists one or more cases per benchmark
type. Parameters use SGLang CLI flag names (snake_case → `--kebab-case`).

```yaml
throughput:
  - name: throughput_smoke_dummy
    model: qwen3            # refers to tests/models/qwen3/<case>.yaml
    case: 06b_tp1
    parameters:            # merged over the model's engine params
      dataset_name: random
      random_input_len: 32
      random_output_len: 1
      num_prompts: 4
      disable_cuda_graph: true
      disable_piecewise_cuda_graph: true
      load_format: dummy   # no real weights needed

latency:
  - name: latency_smoke_dummy
    model: qwen3
    case: 06b_tp1
    parameters:
      input_len: 32
      output_len: 1
      batch_size: 1
      disable_cuda_graph: true
      disable_piecewise_cuda_graph: true
      load_format: dummy

serve:
  - name: serve_smoke_dummy
    model: qwen3
    case: 06b_tp1
    server_parameters:      # merged over model engine params; pops host/port/tp_size/served_model_name
      served_model_name: smoke-model
      tp_size: 1
      context_length: 1024
      disable_cuda_graph: true
      load_format: dummy
    client_parameters:      # passed to sglang.bench_serving
      model: smoke-model
      backend: sglang-oai-chat
      dataset_name: random
      random_input_len: 32
      random_output_len: 4
      num_prompts: 5
```

## Writing New Tests

### Adding a new E2E model test

1. Create a YAML config: `tests/models/<model>/<case>.yaml` (add `generate`,
   `serve`, and/or `concurrent` sections as needed).
2. Register it in the platform config: `tests/platforms/cuda.yaml` under
   `tests.e2e.<task>.<model>`.
3. Done — no Python code needed. The unified smoke tests handle the rest.

### Adding a unit test

Unit tests should be fast, isolated, and not require GPU or model weights.

```python
# tests/unit_tests/dispatch/test_my_op.py
import pytest
from sglang_fl.dispatch import call_op

class TestMyOp:
    def test_basic(self):
        result = call_op("my.op", *args)
        assert result == expected
```

### Adding a functional test

Functional tests validate component correctness on real hardware. They require
GPU but not model files. Use the `device` fixture and mark `functional` + `gpu`.

```python
# tests/functional_tests/ops/test_my_kernel.py
import pytest
import torch

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

def test_my_kernel(device):
    x = torch.randn(16, 256, device=device)
    result = call_op("my.kernel", x)
    reference = reference_impl(x)
    assert torch.allclose(result.float(), reference.float(), rtol=1e-3, atol=1e-3)
```

### Adding a benchmark smoke case

1. Append a case under the relevant type in
   `tests/benchmarks/configs/smoke.yaml`.
2. Reference an existing `model`/`case` so `run.py` can merge engine params from
   the model YAML, or supply a full `model`/`case` of your own.

### Adding a new platform

1. Copy `tests/platforms/template.yaml` to `tests/platforms/<platform>.yaml`.
2. Fill in `device_types`, `tolerance`, `env_defaults`, and the per-device
   `tests` matrix.
3. Run: `python tests/run.py --platform <platform> --device <device>`.

## Test Scopes

| Scope | Directory | GPU | Models | Runs via |
|---|---|---|---|---|
| `unit` | `tests/unit_tests/` | No | No | `pytest` directly |
| `functional` | `tests/functional_tests/` | Yes | No | `pytest` directly |
| `e2e` | `tests/e2e_tests/` | Yes | Yes | Subprocess per case (via `run.py`) |
| `benchmark` | `tests/benchmarks/` | Yes | Dummy only | Subprocess per case (via `run.py`) |

## Available Markers

| Marker | Description |
|---|---|
| `@pytest.mark.gpu` | Requires a GPU/accelerator |
| `@pytest.mark.functional` | Functional correctness test |
| `@pytest.mark.e2e` | End-to-end smoke test |
| `@pytest.mark.benchmark` | Benchmark smoke test |

(Markers are auto-registered in `tests/conftest.py`.)

## Environment Variables

| Variable | Set by | Purpose |
|---|---|---|
| `FL_TEST_PLATFORM` / `FL_TEST_DEVICE` | `run.py` | Active platform/device for the subprocess |
| `FL_TEST_MODEL` / `FL_TEST_CASE` | `run.py` (e2e) | Selects `tests/models/<model>/<case>.yaml` |
| `FL_BENCHMARK_CASE` | `run.py` (benchmark) | JSON blob of merged benchmark case params |
| `FL_CONCURRENT_MODES` | user | Override concurrent modes: `text,vl,mixed` or `all` |
| `FL_CONCURRENT_N` | user | Override async request count |
| `FL_CONCURRENT_MAX_TOKENS` / `FL_CONCURRENT_VL_MAX_TOKENS` | user | Override text/vl `max_new_tokens` |
| `FL_CONCURRENT_STRICT_TEXT` | user | `1` to enforce `expected` checks in text mode |
| `IMAGE_DIR` | user | Directory for concurrent VL test images |
| `MODEL_PATH` / `TP_SIZE` | user | Used by `validate.sh` and `test_*_align.py` |
| `SGLANG_PLUGINS` | user/`validate.sh` | `__none__` disables plugin in baseline runs |
| `SGLANG_FL_DISPATCH_LOG` | user/`validate.sh` | File path for OOT dispatch op log |
| `SGLANG_FL_PER_OP` | user/`validate.sh` | Per-op backend overrides, e.g. `silu_and_mul=vendor:mock_npu` |
| `SGLANG_FLAGGEMS_*` | user/`validate.sh` | FlagGems recording/log controls |

## Shared Fixtures

| Fixture | Scope | Source | Description |
|---|---|---|---|
| `device` | session | `tests/functional_tests/conftest.py` | `torch.device("cuda")`, skips if CUDA unavailable |
| `registry` | function | `tests/unit_tests/dispatch/conftest.py` | Fresh `OpRegistry` |
| `make_impl` | function | `tests/unit_tests/dispatch/conftest.py` | Factory for `OpImpl` instances |
| `dummy_fn` | function | `tests/unit_tests/dispatch/conftest.py` | Simple callable for `OpImpl` |
| `sglang_fl_module` | function | `tests/unit_tests/flaggems/conftest.py` | Imported `sglang_fl` module |
| `fake_flag_gems` | function | `tests/unit_tests/flaggems/conftest.py` | Mock `flag_gems` module with call capture |
| `clean_flaggems_env` | autouse | `tests/unit_tests/flaggems/conftest.py` | Clears FlagGems env vars per test |
