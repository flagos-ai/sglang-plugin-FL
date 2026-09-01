# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prometheus metrics for FlagCX KV cache transfers.

Counters carry the volume series (use ``rate()``/``increase()`` in PromQL);
histograms carry the per-transfer distributions:

  sglang:flagcx_total_bytes_transferred -> rate() = KV transfer bytes/s
  sglang:flagcx_total_xfer_time_seconds -> ratio with the above gives
                                           effective bytes/s while the link
                                           was actually busy
  sglang:flagcx_transfers               -> rate() = transfers/s
  sglang:flagcx_total_descriptors       -> descriptors/s
  sglang:flagcx_num_failed_transfers    -> P-side write failures
  sglang:flagcx_num_failed_recvs        -> D-side side-channel failures
  sglang:flagcx_num_kv_expired_reqs     -> requests timed out before transfer
  sglang:flagcx_xfer_time_seconds       -> per-transfer latency histogram
  sglang:flagcx_bytes_transferred       -> per-transfer size histogram
  sglang:flagcx_num_descriptors         -> per-transfer descriptor count
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Uniform 2KB to 16GB range, matching the vLLM connector.
_BYTES_BUCKETS = [2 ** (10 + i) for i in range(1, 25, 2)]
_TIME_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0]
_DESC_BUCKETS = [
    10, 20, 30, 50, 75, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 50000,
]

_lock = threading.Lock()
_metrics = None
_labelnames = None


class _FlagcxMetrics:
    """Holds the registered collectors. Built once, on first record_* call."""

    def __init__(self, labelnames):
        from prometheus_client import Counter, Histogram

        def counter(name, doc):
            return Counter(name, doc, labelnames)

        def histogram(name, doc, buckets):
            return Histogram(name, doc, labelnames, buckets=buckets)

        self.total_bytes = counter(
            "sglang:flagcx_total_bytes_transferred",
            "Total bytes transferred by FlagCX KV Cache transfers.",
        )
        self.total_time = counter(
            "sglang:flagcx_total_xfer_time_seconds",
            "Total time spent inside FlagCX KV Cache transfers. Divide "
            "sglang:flagcx_total_bytes_transferred_total by this to get "
            "effective link throughput.",
        )
        self.transfers = counter(
            "sglang:flagcx_transfers",
            "Number of successful FlagCX KV Cache transfers.",
        )
        self.total_descriptors = counter(
            "sglang:flagcx_total_descriptors",
            "Total number of descriptors written by FlagCX KV Cache transfers.",
        )
        self.failed_transfers = counter(
            "sglang:flagcx_num_failed_transfers",
            "Number of failed FlagCX KV Cache transfers. "
            "NOTE: This metric is tracked on the P instance.",
        )
        self.failed_recvs = counter(
            "sglang:flagcx_num_failed_recvs",
            "Number of failed FlagCX KV Cache receives (side-channel or "
            "transfer errors reported to D). "
            "NOTE: This metric is tracked on the D instance.",
        )
        self.kv_expired_reqs = counter(
            "sglang:flagcx_num_kv_expired_reqs",
            "Number of requests that timed out before their KV was transferred.",
        )
        self.xfer_time = histogram(
            "sglang:flagcx_xfer_time_seconds",
            "Histogram of transfer duration for FlagCX KV Cache transfers.",
            _TIME_BUCKETS,
        )
        self.bytes_transferred = histogram(
            "sglang:flagcx_bytes_transferred",
            "Histogram of bytes transferred per FlagCX KV Cache transfer.",
            _BYTES_BUCKETS,
        )
        self.num_descriptors = histogram(
            "sglang:flagcx_num_descriptors",
            "Histogram of number of descriptors per FlagCX KV Cache transfer.",
            _DESC_BUCKETS,
        )


def _get(labels):
    """Return the collectors bound to `labels`, registering them on first call.

    Returns None if prometheus_client is unavailable, registration failed, or a
    later caller passed a different label set -- a metrics problem must never
    take down a KV transfer.
    """
    global _metrics, _labelnames
    if _metrics is None:
        with _lock:
            if _metrics is None:
                try:
                    _labelnames = sorted(labels)
                    _metrics = _FlagcxMetrics(_labelnames)
                except Exception:
                    logger.warning(
                        "FlagCX metrics unavailable; transfers continue unmeasured.",
                        exc_info=True,
                    )
                    _metrics = False
    if not _metrics:
        return None
    if sorted(labels) != _labelnames:
        logger.warning(
            "FlagCX metrics label mismatch: registered %s, got %s; sample dropped.",
            _labelnames,
            sorted(labels),
        )
        return None
    return _metrics


def record_transfer(duration_s, total_bytes, num_descs, labels):
    """One successful batch transfer: advance counters and the histograms."""
    m = _get(labels)
    if m is None:
        return
    m.total_bytes.labels(**labels).inc(total_bytes)
    m.total_time.labels(**labels).inc(duration_s)
    m.transfers.labels(**labels).inc()
    m.total_descriptors.labels(**labels).inc(num_descs)
    m.xfer_time.labels(**labels).observe(duration_s)
    m.bytes_transferred.labels(**labels).observe(total_bytes)
    m.num_descriptors.labels(**labels).observe(num_descs)


def record_failed_transfer(labels):
    m = _get(labels)
    if m is not None:
        m.failed_transfers.labels(**labels).inc()


def record_failed_recv(labels):
    m = _get(labels)
    if m is not None:
        m.failed_recvs.labels(**labels).inc()


def record_kv_expired_req(labels):
    m = _get(labels)
    if m is not None:
        m.kv_expired_reqs.labels(**labels).inc()
