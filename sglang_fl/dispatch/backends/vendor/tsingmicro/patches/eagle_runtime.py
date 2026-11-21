# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EAGLE/MTP runtime settings for txda that SGLang 0.5.18 only defines per device.

* Draft attention backend: EAGLE's ``DraftBackendFactory`` has its own backend
  table without ``txda_oot`` (the TXDA target backend), so the default
  ``--speculative-draft-attention-backend`` fails. Map ``txda_oot`` to SGLang's
  ``triton`` draft backend (the draft runner uses plain Triton attention).
* Draft graph capture: ``EagleDraftWorker._capture_cuda_graphs`` only knows
  cuda/xpu/npu/musa runners. TXDA has no graph support: run the draft eagerly.
* Conv-window layout: ``conv_window_dedup_enabled`` lets linear draft chains
  share one buffer viewed as overlapping per-step windows (``as_strided``).
  Writes into that overlapping view are not reliable on txda; allocate dense
  per-step windows instead (what SGLang already does for NPU/CPU/tree verify).
* Input logprobs: the fused row logsumexp/top-k kernel does not lower on txda;
  use SGLang's ``log_softmax`` path.

Each piece is applied right after SGLang imports the owning module.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("eagle_runtime")
_patched = False


def _draft_backend() -> None:
    import functools

    from sglang.srt.speculative.draft_utils import DraftBackendFactory

    original = DraftBackendFactory._create_backend
    if getattr(original, "_txda", False):
        return

    @functools.wraps(original)
    def _create_backend(self, backend_name, backend_map, error_template, stamps_children=False):
        if "txda_oot" not in backend_map and "triton" in backend_map:
            backend_map = {**backend_map, "txda_oot": backend_map["triton"]}
        return original(self, backend_name, backend_map, error_template, stamps_children)

    _create_backend._txda = True
    DraftBackendFactory._create_backend = _create_backend
    _log.applied("EAGLE draft attention txda_oot -> SGLang triton draft backend")


def _draft_graphs() -> None:
    import functools

    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    original = EagleDraftWorker._capture_cuda_graphs
    if getattr(original, "_txda", False):
        return

    @functools.wraps(original)
    def _capture_cuda_graphs(self):
        if str(getattr(self.target_worker, "device", "")).startswith("txda"):
            self.cuda_graph_runner = None
            self.cuda_graph_runner_for_draft_extend = None
            return
        return original(self)

    _capture_cuda_graphs._txda = True
    EagleDraftWorker._capture_cuda_graphs = _capture_cuda_graphs
    _log.applied("EAGLE draft graph capture skipped on txda (eager draft)")


def _dense_conv_windows() -> None:
    import sglang.srt.mem_cache.memory_pool as mp

    def conv_window_dedup_enabled(is_npu, is_cpu, speculative_eagle_topk, is_kda):
        return False

    mp.conv_window_dedup_enabled = conv_window_dedup_enabled
    _log.applied("MTP conv verification windows dense (no overlapping view) on txda")


def _input_logprobs() -> None:
    import functools

    from sglang.srt.layers.logprob_processor import InputLogprobProcessor

    original = InputLogprobProcessor.__init__
    if getattr(original, "_txda", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.enable_fast_input_logprobs = False

    __init__._txda = True
    InputLogprobProcessor.__init__ = __init__
    _log.applied("input logprobs -> SGLang log_softmax path on txda")


def _safe(fn):
    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _log.failed("%s failed: %s", fn.__name__, exc)

    return run


def patch() -> None:
    """Install the lazy patches. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap("sglang.srt.speculative.draft_utils", _safe(_draft_backend))
    _install_post_import_wrap("sglang.srt.speculative.eagle_worker_v2", _safe(_draft_graphs))
    _install_post_import_wrap("sglang.srt.mem_cache.memory_pool", _safe(_dense_conv_windows))
    _install_post_import_wrap("sglang.srt.layers.logprob_processor", _safe(_input_logprobs))
    _patched = True
