# GLM-5.2 FP8 on two S5000 nodes

Use the 2026-10-27 team MUSA image with SGLang `9fa1740602`. The runtime needs
working MUSA and RDMA, the model weights, and three wheel artifacts:

- This `sglang-plugin-FL` change.
- The companion `FlagGems-sglang` change that exports MUSA
  `native_layer_norm`.
- The FlagGems MUSA wheel built from `98fae44c` (`5.3.0rc2`).

Build the two project wheels from the review commits with
`python -m build --wheel`. Install all three on both nodes, along with their
required Python dependencies. Preserve the image's Torch, Triton, MUSA and
SGLang versions. For example, `python3 -m pip install --no-deps /wheels/*.whl`
works when all required wheels are already staged and the image has the
remaining dependencies. Confirm the installed paths before starting the server.

Inside the image, set `GLM52_MODEL_PATH`, `GLM52_DIST_ADDR` (node 0 IP and
rendezvous port), `GLM52_SOCKET_IFNAME`, `GLM52_IB_DEVICES` (comma-separated
HCAs), `GLM52_IB_GID_INDEX`, and `GLM52_DEEPEP_CONFIG` (the mounted path to
`deepep.json`). Run `bash serve.sh 0` and `bash serve.sh 1` on the respective
nodes. The example uses PP2 / TP8 / EP8 / DP1 / Attention CP8, preserves the
40/38 layer split, and enables Decode Graph for 8, 16, 24, and 32.

In a separate process on an idle MUSA GPU, run
`source operator-profile.sh && python3 check_contracts.py` before serving.
Check known-answer requests and an actual batch-size-32 decode replay before
measuring performance. For the matched performance run, send 64 requests at
concurrency 64, each with exactly 4096 input and 1024 output tokens, seed 101.
Keep profiling out of the throughput run. Capture a separate short trace to
verify the operator routing against the 30-entry target.

The profile preserves the image's platform selection, communication and vendor
hooks. It uses a launch-specific ATen allowlist and RMSNorm dispatch policy.
