# MiniCPM-V 4.7 35B-A3B on MTT S5000

The model table lists the BF16, tensor-parallel-2 SGLang-FL configuration on
two MTT S5000 GPUs as supported. The checkpoint remains outside the container.
The code in this PR head has not yet completed a fresh end-to-end hardware
evaluation; the PR records that validation boundary.

Install a MUSA-compatible PyTorch, SGLang with the FLA interfaces exercised by
this branch, FlagGems, and this plugin. The tested SGLang core commit and
reproducible dependency set must be frozen before independent reproduction.
In particular, changing SGLang's private FLA or graph-runner
interfaces requires revalidation.

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

The FlagGems chunk GDN route is opt-in with `SGLANG_MUSA_FLAGOS_CHUNK_GDN=1`.
It has not passed an end-to-end numerical and performance gate on this PR head.
Keep it disabled for the documented launch. The service's graph coverage,
multimodal behavior, concurrency, accuracy, and latency still need evaluation
on the final built artifact before claiming those gates have passed.
