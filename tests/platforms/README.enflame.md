# Enflame v0.5.18 tests

Run CPU platform contracts from the plugin repository:

```bash
python3 -m pytest tests/unit_tests/platform/test_enflame_*.py -q
```

The platform unit tests cover detection/configuration, runtime defaults, graph
API compatibility, decode page metadata, logprob selection, KV fallback and MoE
orchestration. Native MoE kernels are replaced by CPU math; these tests do not
validate native GCU kernels or full model inference. Request-table integer
conversion tests and the TopK dtype-only test have been removed.

`tests/platforms/enflame.yaml` remains the test-runner profile. Its model YAMLs
select existing no-graph inference/concurrent cases for both models. Model paths
are used directly from those YAMLs. No Enflame speculative-decoding cases are
registered. Tests requiring real devices or model services belong in functional
or E2E suites, not the platform unit-test directory.

The unit runner collects `tests/unit_tests` recursively; the Enflame profile has
no unit exclusions. Moving platform unit tests does not by itself add an Enflame
GitHub Actions runner or guarantee that a remote CI job has been executed.
