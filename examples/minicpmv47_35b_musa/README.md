# MiniCPM-V 4.7 35B-A3B on MTT S5000

This is a BF16, tensor-parallel-2 SGLang-FL integration candidate for two MTT
S5000 GPUs. The checkpoint remains outside the container. The model support
table will be updated after this branch passes end-to-end hardware validation.

Install a MUSA-compatible PyTorch, SGLang 0.5.18, FlagGems, and this plugin.
Freeze their exact commits and the image digest for independent reproduction.
The plugin's existing FlagOS FLA implementation handles chunk GDN state layout
and cache writeback; no separate MUSA chunk implementation is enabled here.

```bash
export MODEL_PATH=/path/to/MiniCPM-V-4.7-35B-A3B
export PORT=30000
bash examples/minicpmv47_35b_musa/launch.sh
```

Check the served identity and a small text response in a separate shell:

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30000/v1/models
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniCPM-V-4.7-35B-A3B","messages":[{"role":"user","content":"17+12="}],"max_tokens":32}'
```

The MUSA hardware comparison test in
`tests/unit_tests/platform/test_musa_fla_chunk_native_comparison.py` compares
FlagOS and native chunk kernels with `K == V` and asymmetric state values.
The service's graph coverage, multimodal behavior, concurrency, accuracy, and
latency still need evaluation on the final built artifact before claiming
those gates have passed.
