#!/usr/bin/env python3
"""Low-cost real-NPU probes for the Ascend SGLang 0.5.18 compatibility paths.

This intentionally avoids loading a model.  It compiles and executes the exact
MTP state-copy tile that overflows the 910C unified buffer without the plugin
workaround, then exercises SGLang's logprob top-k API and verifies that the
CANN 8.5 split fallback is active and numerically correct.
"""

from __future__ import annotations

import importlib
import os
from typing import Any


MTP_SOURCE_SHAPE = (1, 1, 2, 2, 128, 128)
MTP_DESTINATION_SHAPE = (1, 2, 2, 128, 128)
MTP_PATCH_MARKER = "_sglang_fl_cann85_multibuffer_disabled"
LOGSUMEXP_SHAPE = (2, 16384)
LOGSUMEXP_TOP_K = 5
LOGSUMEXP_PATCH_MARKER = "_sglang_fl_ascend_split_topk"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_plugin() -> None:
    os.environ.setdefault("SGLANG_PLUGINS", "sglang_fl")
    from sglang.srt.plugins import load_plugins

    load_plugins()
    import sglang_fl

    _require(
        sglang_fl.is_plugin_active(),
        "sglang_fl did not become active before the Ascend compatibility probe",
    )


def probe_mtp_state_update(torch: Any) -> None:
    kernel_module = importlib.import_module(
        "sgl_kernel_npu.mamba.mamba_state_update_triton"
    )
    kernel = kernel_module.move_cache_dynamic_last_kernel_h_block
    _require(
        getattr(kernel.run, MTP_PATCH_MARKER, False),
        "MTP state-update multibuffer workaround is not active",
    )

    selected_state = (
        torch.arange(2 * 128 * 128, dtype=torch.float32).reshape(2, 128, 128) / 1024.0
    )
    source_cpu = torch.zeros(MTP_SOURCE_SHAPE, dtype=torch.float32)
    source_cpu[0, 0, 1].copy_(selected_state)
    source = source_cpu.to("npu")
    destination = torch.full(
        MTP_DESTINATION_SHAPE,
        -1.0,
        dtype=torch.float32,
        device="npu",
    )
    dst_indices = torch.tensor([1], dtype=torch.int64, device="npu")
    src_indices = torch.tensor([0], dtype=torch.int64, device="npu")
    last_steps = torch.tensor([1], dtype=torch.int64, device="npu")

    kernel_module.move_intermediate_cache(
        destination,
        source,
        dst_indices,
        src_indices,
        last_steps,
        h_block_size=2,
    )
    torch.npu.synchronize()

    actual = destination[0, 1].cpu()
    untouched = destination[0, 0].cpu()
    _require(
        torch.equal(actual, selected_state),
        "MTP state-update probe copied an incorrect temporal state",
    )
    _require(
        torch.equal(untouched, torch.full_like(untouched, -1.0)),
        "MTP state-update probe modified an unselected destination slot",
    )
    print("[ascend-compat] MTP state-update tile passed")


def probe_logsumexp_topk(torch: Any) -> None:
    logsumexp = importlib.import_module("sglang.srt.layers.logsumexp")
    row_logsumexp_topk = logsumexp.row_logsumexp_topk
    _require(
        getattr(row_logsumexp_topk, LOGSUMEXP_PATCH_MARKER, False),
        "row_logsumexp_topk CANN 8.5 fallback is not active",
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(518)
    logits_cpu = torch.randn(
        LOGSUMEXP_SHAPE,
        generator=generator,
        dtype=torch.float32,
    )
    logits = logits_cpu.to("npu")
    row_max, row_log_sum, top_values, top_indices = row_logsumexp_topk(
        logits, LOGSUMEXP_TOP_K
    )
    torch.npu.synchronize()

    row_max_cpu = row_max.cpu()
    row_log_sum_cpu = row_log_sum.cpu()
    top_values_cpu = top_values.cpu()
    top_indices_cpu = top_indices.cpu()
    expected_max = logits_cpu.amax(dim=-1)
    expected_log_sum = torch.logsumexp(logits_cpu - expected_max[:, None], dim=-1)
    expected_values, expected_indices = torch.topk(
        logits_cpu,
        LOGSUMEXP_TOP_K,
        dim=-1,
        largest=True,
        sorted=True,
    )

    torch.testing.assert_close(row_max_cpu, expected_max, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        row_log_sum_cpu,
        expected_log_sum,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(top_values_cpu, expected_values, rtol=0.0, atol=0.0)
    _require(
        torch.equal(top_indices_cpu, expected_indices),
        "row_logsumexp_topk fallback returned incorrect token indices",
    )
    print("[ascend-compat] row_logsumexp_topk fallback passed")


def main() -> int:
    import torch
    import torch_npu  # noqa: F401

    _require(torch.npu.is_available(), "Ascend torch.npu is unavailable")
    _require(torch.npu.device_count() >= 1, "no Ascend NPU is visible")
    torch.npu.set_device(0)
    _load_plugin()
    probe_mtp_state_update(torch)
    probe_logsumexp_topk(torch)
    print("[ascend-compat] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
