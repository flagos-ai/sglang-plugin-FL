# Examples

Offline inference examples for smoke-testing sglang-plugin-FL with various models.

Each script is self-contained: it loads the model, runs inference, validates output,
and exits with code 0 on success or non-zero on failure. This makes them directly
usable in CI pipelines.

## Available Examples

| Script | Model | Architecture | TP |
|--------|-------|-------------|-----|
| `qwen3_6_35b_a3b_offline_inference.py` | Qwen3.6-35B-A3B | MoE (256 experts) | 1 |
| `qwen3_6_27b_offline_inference.py` | Qwen3.6-27B | Dense (hybrid attention) | 1 |

## Usage

```bash
# Run with plugin loaded (default — SGLANG_PLUGINS auto-discovers sglang_fl)
python examples/qwen3_6_35b_a3b_offline_inference.py

# Run with only ATen replacement (no fused op dispatch)
SGLANG_FL_OOT_ENABLED=0 python examples/qwen3_6_35b_a3b_offline_inference.py

# Run baseline (no plugin at all)
SGLANG_PLUGINS=__none__ python examples/qwen3_6_35b_a3b_offline_inference.py

# Custom model path / TP
MODEL_PATH=/data/models/Qwen3.6-27B TP_SIZE=2 python examples/qwen3_6_27b_offline_inference.py
```

## CI Integration

These scripts are designed for CI use:

- **Exit code**: 0 = pass, non-zero = fail
- **Model path**: Configurable via `MODEL_PATH` env var
- **Skip if model missing**: Prints message and exits with code 1
- **No interactive input**: Fully automated
- **Deterministic**: temperature=0, greedy decoding

A CI workflow can run them as:

```yaml
- name: Run inference examples
  run: |
    for f in examples/*_offline_inference.py; do
      echo "========== Running $f =========="
      python "$f" || exit 1
    done
```

## Test Modes

Control the plugin behavior via environment variables:

| Mode | Env Vars | What's tested |
|------|----------|---------------|
| Baseline | `SGLANG_PLUGINS=__none__` | Native SGLang, no plugin |
| ATen only | `SGLANG_FL_OOT_ENABLED=0` | Layer 1: FlagGems ATen replacement |
| Full | (default) | Layer 1 + Layer 2: ATen + fused op dispatch |
| Vendor | `USE_FLAGGEMS=0 SGLANG_FL_PREFER=vendor` | Layer 2 only: vendor fused ops |

For precision comparison across modes, use `tests/test_precision_align.py` or `tests/validate.sh`.
