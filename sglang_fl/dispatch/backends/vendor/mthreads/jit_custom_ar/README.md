# MUSA JIT custom all-reduce sources

These Apache-2.0 sources are vendored from the Moore Threads SGLang 0.5.12
runtime image used for the MTT S5000 comparison.  The Python integration was
adapted to remain inside `sglang-plugin-FL`; upstream SGLang is monkeypatched at
runtime and is not modified.

The compatibility default uses the explicit-input CUDA/MUSA-graph launcher.
The newer registered-input recapture protocol is opt-in with
`SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT=1` only when the matching SGLang
graph-runner handshake is present.
