# AGENTS.md — sglang-plugin-FL

## Setup & Install

```bash
pip install -e .          # Editable install (auto-registers entry_points)
```

The plugin is auto-discovered by SGLang via setuptools entry_points. No manual import or env var needed.

## Dev Commands

```bash
# Lint + format (ruff)
ruff check --ignore=E731 sglang_fl/ tests/
ruff format sglang_fl/ tests/

# Typo check
typos

# Run all pre-commit hooks
pre-commit run --all-files

# Unit tests (pytest, no GPU needed)
python -m pytest tests/unit_tests/ -v

# E2E config tests (requires GPU + MODEL_PATH)
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_e2e_config.py
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_e2e_config.py --list
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_e2e_config.py --case 3

# Precision alignment (baseline vs plugin)
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_precision_align.py baseline
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_precision_align.py plugin
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct python tests/test_precision_align.py compare

# Full validation script (orchestrates above)
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct bash tests/validate.sh all
TP_SIZE=8 MODEL_PATH=/path/to/14B bash tests/validate.sh full
```

## Architecture

Three-layer plugin for SGLang multi-chip inference:

| Layer | What | Mechanism |
|-------|------|-----------|
| 1 | ATen ops → FlagGems Triton | `flag_gems.enable()` via PyTorch dispatch |
| 2 | SGLang fused kernels (SiluAndMul, RMSNorm, RotaryEmbedding, TopK, MoE, FLA) | AROUND hook on `MultiPlatformOp.dispatch_forward` |
| 3 | Distributed communication (all_reduce, etc.) | AROUND hooks on `GroupCoordinator` → `CommunicatorFL` |

**Dispatch system** (`sglang_fl/dispatch/`) is shared with vllm-plugin-FL. Vendor backends implement the same interface for both frameworks. The bridge layer (`sglang_fl/dispatch/bridge/`) translates SGLang-specific parameters to standardized op signatures.

**Entry points** (in `pyproject.toml`):
- `sglang.srt.platforms` → `activate_platform()` — device identity, memory, dist backend
- `sglang.srt.plugins` → `load_plugin()` — FlagGems + dispatch + communicator hooks

## Critical Gotchas

### `SGLANG_FL_STRICT` naming is inverted
`SGLANG_FL_STRICT=1` (default) means **fallback ENABLED** — if the preferred backend fails, the next candidate is tried. `SGLANG_FL_STRICT=0` means **direct resolve only, no fallback**. This is the opposite of what the name suggests.

### `--disable-piecewise-cuda-graph` is always required
FlagGems Triton kernels contain `logging.Logger` calls incompatible with `torch.compile`. Always pass `--disable-piecewise-cuda-graph` when launching SGLang with this plugin. Regular CUDA graph capture works fine.

### Config priority chain
```
SGLANG_FL_* env vars > SGLANG_FL_CONFIG YAML > platform auto-detect YAML > code defaults
```

### Plugin disable/partial enable
- `SGLANG_PLUGINS="__none__"` — disable entire plugin (vanilla SGLang)
- `USE_FLAGGEMS=0` — disable Layer 1 only (ATen replacement), keep Layer 2
- `SGLANG_FL_OOT_ENABLED=0` — disable Layer 2 only (fused ops), keep Layer 1

### Config file location
Platform YAML configs live in `sglang_fl/dispatch/config/` (not `sglang_fl/config/`). The README references `sglang_fl/config/sample.yaml` but no sample.yaml is shipped — use the env vars or write your own YAML.

### MUSA RotaryEmbedding patch
On MUSA platform, `RotaryEmbedding.__init__` stomps the dispatch bridge. `rotary_patch.py` monkey-patches `__init__` to restore it. This runs automatically in `load_plugin()`.

### FLA function patching
`fla_patch.py` replaces SGLang's FLA module-level functions (`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, etc.) with dispatch bridges. Original functions are saved in `fla_patch._originals` for backends that need to call the unpatched version.

### Log suppression on non-rank-0
`_is_rank0()` in `__init__.py` checks `multiprocessing.parent_process()` to detect spawned subprocesses. Non-rank-0 processes get `WARNING` log level to avoid duplicate output.

### `_AtenOnlyFilter`
FlagGems ATen replacement log is filtered to only show `flag_gems.ops.*` calls (Layer 1), excluding internal FlagGems calls from Layer 2 flagos implementations.

## Testing

- **Unit tests** (`tests/unit_tests/dispatch/`): pytest, no GPU, test dispatch registry/policy/manager in isolation
- **E2E tests** (`tests/test_e2e_config.py`): requires GPU + `MODEL_PATH` env var, starts real SGLang server per test case
- **Precision tests** (`tests/test_precision_align.py`): baseline vs plugin output comparison, greedy decoding
- **`validate.sh`**: orchestrates precision tests with dispatch log collection

## Vendor Integration

New chip vendors add a directory under `sglang_fl/dispatch/backends/vendor/<name>/` with:
- `<name>.py` — Backend subclass with `is_available()` and op methods
- `register_ops.py` — `register_builtins(registry)` function
- `impl/` — operator implementations

Auto-discovered at startup via `builtin_ops._register_vendor_backends()`. No other files need modification. Same backend interface works for both sglang-plugin-FL and vllm-plugin-FL.

## Dependencies

- SGLang 0.5.11 (pinned — entry_point API is version-specific)
- FlagGems 4.2.1rc0+
- PyYAML (for config loading)
- Optional: FlagCX (for multi-chip distributed communication)
