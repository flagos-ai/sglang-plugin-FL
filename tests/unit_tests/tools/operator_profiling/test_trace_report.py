# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import gzip
import json
from pathlib import Path

from tools.operator_profiling.trace_report import (
    collect_traces,
    discover_trace_files,
    generate_reports,
    iter_trace_events,
    stable_kernel_name,
)


def _event(
    category,
    name,
    external_id=None,
    *,
    timestamp=0,
    duration=1,
    pid=7,
    tid=8,
    shapes=None,
    dtypes=None,
    extra_args=None,
):
    args = dict(extra_args or {})
    if external_id is not None:
        args["External id"] = external_id
    if shapes is not None:
        args["Input Dims"] = shapes
    if dtypes is not None:
        args["Input type"] = dtypes
    return {
        "ph": "X",
        "cat": category,
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": timestamp,
        "dur": duration,
        "args": args,
    }


def _write_trace(path: Path, events, *, compact=False):
    document = {"schemaVersion": 1, "traceEvents": events, "traceName": str(path)}
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as output:
        json.dump(document, output, indent=None if compact else 2)


def _operator_marker(payload):
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    return "sglang_fl.operator_profile.kernel:" + encoded


def _sample_events(rank=0):
    return [
        _event(
            "user_annotation", "step[EXTEND bs=2 toks=8]", timestamp=10, duration=50
        ),
        _event(
            "cpu_op",
            "aten::mm",
            1,
            timestamp=20,
            duration=4,
            shapes=[[2, 4], [4, 8]],
            dtypes=["c10::BFloat16", "c10::BFloat16"],
        ),
        _event(
            "cpu_op",
            "sgl_kernel::fused_rmsnorm",
            2,
            timestamp=30,
            duration=4,
            shapes=[[2, 8], [8]],
            dtypes=["c10::BFloat16", "c10::BFloat16"],
        ),
        _event("cpu_op", "record_param_comms", 3, timestamp=40, duration=4),
        _event("cpu_op", "sgl_kernel::all_reduce", 6, timestamp=45, duration=4),
        _event("kernel", "mm_kernel", 1, timestamp=100, duration=10, pid=rank),
        _event(
            "kernel", "fused_rmsnorm_kernel", 2, timestamp=120, duration=20, pid=rank
        ),
        _event(
            "kernel",
            "ncclDevKernel_AllReduce_RING_LL",
            3,
            timestamp=140,
            duration=30,
            pid=rank,
        ),
        _event(
            "kernel",
            "sglang::cross_device_reduce_1stage",
            6,
            timestamp=170,
            duration=7,
            pid=rank,
        ),
        _event(
            "kernel", "unmapped_compute_kernel", timestamp=180, duration=5, pid=rank
        ),
        _event("gpu_memcpy", "Memcpy HtoD", 4, timestamp=200, duration=2, pid=rank),
        _event("gpu_memset", "Memset", 5, timestamp=210, duration=3, pid=rank),
    ]


def test_stream_reader_supports_pretty_plain_and_compact_gzip(tmp_path):
    plain = tmp_path / "run-TP-0.trace.json"
    compressed = tmp_path / "run-TP-1.trace.json.gz"
    events = _sample_events()
    _write_trace(plain, events)
    _write_trace(compressed, events, compact=True)

    assert list(iter_trace_events(plain, chunk_size=17)) == events
    assert list(iter_trace_events(compressed, chunk_size=19)) == events
    assert discover_trace_files(tmp_path) == [plain, compressed]


def test_stable_kernel_name_preserves_unnamed_namespace_and_drops_specialization():
    raw = (
        "void at::native::<unnamed>::multi_tensor_apply_kernel<"
        "at::native::CopyFunctor>(int, float*) [clone .123]"
    )
    assert stable_kernel_name(raw) == "at::native::<unnamed>::multi_tensor_apply_kernel"
    assert (
        stable_kernel_name(
            "void at::native::(anonymous namespace)::multi_tensor_apply_kernel<"
            "at::native::CopyFunctor>(int, float*)"
        )
        == "at::native::<anonymous>::multi_tensor_apply_kernel"
    )
    assert (
        stable_kernel_name(
            "kernel_cutlass_flashinfer::RMSNormKernel_object_at__tensorptr_0123456789ab"
        )
        == "kernel_cutlass_flashinfer::RMSNormKernel"
    )


def test_collection_maps_shapes_and_excludes_non_compute_activity(tmp_path):
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(trace, _sample_events(), compact=True)
    aggregate = collect_traces([trace])

    assert aggregate.compute_kernel_count == 3
    assert aggregate.compute_kernel_ns == 35_000
    assert aggregate.runtime_kernel_event_count == 5
    assert aggregate.runtime_kernel_time_ns == 72_000
    assert aggregate.excluded_count == {
        "communication": 2,
        "gpu_memcpy": 1,
        "gpu_memset": 1,
    }
    assert aggregate.excluded_ns == {
        "communication": 37_000,
        "gpu_memcpy": 2_000,
        "gpu_memset": 3_000,
    }
    assert aggregate.phase_count[("prefill", "aten")] == 1
    assert aggregate.phase_count[("prefill", "fused")] == 1
    assert aggregate.phase_count[("other", "fused")] == 1


def test_direct_kernel_uses_innermost_operator_marker_via_correlation(tmp_path):
    outer_marker = _operator_marker(
        {
            "operator_name": "sglang.layers.RMSNorm.forward_cuda",
            "input_shapes": [[2, 8], [8]],
            "input_dtypes": ["torch.bfloat16", "torch.bfloat16"],
            "source": "sglang_multi_platform_op",
        },
    )
    inner_marker = _operator_marker(
        {
            "operator_name": "sglang.kernels.fused_rmsnorm_kernel",
            "input_shapes": [[2, 8], [8], []],
            "input_dtypes": [
                "torch.bfloat16",
                "torch.bfloat16",
                "Scalar[int]",
            ],
            "source": "triton_jit",
        },
    )
    events = [
        _event(
            "user_annotation",
            "step[DECODE bs=2]",
            timestamp=10,
            duration=60,
        ),
        _event(
            "user_annotation",
            outer_marker,
            timestamp=20,
            duration=30,
        ),
        _event(
            "user_annotation",
            inner_marker,
            timestamp=25,
            duration=15,
        ),
        _event(
            "cuda_driver",
            "cuLaunchKernelEx",
            timestamp=30,
            duration=1,
            extra_args={"correlation": 91},
        ),
        _event(
            "kernel",
            "fused_rmsnorm_kernel",
            timestamp=100,
            duration=12,
            pid=0,
            extra_args={"correlation": 91},
        ),
    ]
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(trace, events, compact=True)

    aggregate = collect_traces([trace])
    variants = aggregate.kernel_variants_count["fused_rmsnorm_kernel"]
    assert len(variants) == 1
    key = next(iter(variants))
    assert key.mapping_status == "native_marker_operator_shape_matched"
    assert key.operator_name == "sglang.kernels.fused_rmsnorm_kernel"
    assert json.loads(key.input_shapes) == [[2, 8], [8], []]
    assert key.source_category == "third_party"
    assert key.source_library == "sglang"
    assert aggregate.phase_count[("decode", "fused")] == 1


def test_marker_sweep_falls_back_to_active_outer_span(tmp_path):
    outer_marker = _operator_marker(
        {
            "operator_name": "outer_fused_op",
            "input_shapes": [[2, 8]],
            "input_dtypes": ["torch.bfloat16"],
        }
    )
    inner_marker = _operator_marker(
        {
            "operator_name": "inner_kernel",
            "input_shapes": [[2, 8], []],
            "input_dtypes": ["torch.bfloat16", "Scalar[int]"],
        }
    )
    events = [
        _event("user_annotation", outer_marker, timestamp=20, duration=100),
        _event("user_annotation", inner_marker, timestamp=25, duration=10),
        _event(
            "cuda_driver",
            "cuLaunchKernel",
            timestamp=30,
            extra_args={"correlation": 91},
        ),
        _event(
            "cuda_driver",
            "cuLaunchKernel",
            timestamp=50,
            extra_args={"correlation": 92},
        ),
        _event(
            "kernel",
            "inner_kernel",
            timestamp=200,
            pid=0,
            extra_args={"correlation": 91},
        ),
        _event(
            "kernel",
            "outer_kernel",
            timestamp=210,
            pid=0,
            extra_args={"correlation": 92},
        ),
    ]
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(trace, events, compact=True)

    aggregate = collect_traces([trace])
    inner_key = next(iter(aggregate.kernel_variants_count["inner_kernel"]))
    outer_key = next(iter(aggregate.kernel_variants_count["outer_kernel"]))
    assert inner_key.operator_name == "inner_kernel"
    assert outer_key.operator_name == "outer_fused_op"


def test_reports_preserve_counts_time_and_reference_columns(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    for rank in (0, 1):
        _write_trace(
            traces / f"profile-TP-{rank}.trace.json.gz",
            _sample_events(rank),
            compact=bool(rank),
        )

    output = tmp_path / "results"
    summary = generate_reports(traces, output, profile_id="profile", jobs=2)

    assert summary["ranks"] == [0, 1]
    assert summary["runtime_kernel_event_count"] == 10
    assert summary["runtime_kernel_time_us"] == 144.0
    assert summary["compute_kernel_event_count"] == 6
    assert summary["compute_kernel_time_us"] == 70.0
    assert set(summary["per_rank"]) == {"0", "1"}
    assert summary["per_rank"]["0"]["compute_kernel_event_count"] == 3
    assert all(summary["validation"].values())
    assert {path.name for path in output.iterdir()} == {
        "operator_list.csv",
        "kernel_shape_dtype.csv",
        "kernel_time.csv",
        "profile_summary.json",
    }

    with (output / "kernel_shape_dtype.csv").open(newline="") as source:
        shape_rows = list(csv.DictReader(source))
        assert source.seekable()
    assert list(shape_rows[0]) == [
        "operator_name",
        "kernel_name",
        "source_category",
        "variant_index",
        "mapping_status",
        "input_shapes",
        "input_dtypes",
        "candidate_operators",
        "kernel_event_count",
        "kernel_time_us",
    ]
    assert sum(int(row["kernel_event_count"]) for row in shape_rows) == 6
    assert sum(float(row["kernel_time_us"]) for row in shape_rows) == 70.0

    with (output / "kernel_time.csv").open(newline="") as source:
        time_rows = list(csv.DictReader(source))
    assert list(time_rows[0]) == [
        "operator_name",
        "kernel_name",
        "source_category",
        "kernel_call_count",
        "kernel_time_us",
        "percent",
        "percent_of_category",
    ]
    assert sum(int(row["kernel_call_count"]) for row in time_rows) == 6
    assert sum(float(row["kernel_time_us"]) for row in time_rows) == 70.0
    assert abs(sum(float(row["percent"].rstrip("%")) for row in time_rows) - 100) < 0.01

    with (output / "operator_list.csv").open(newline="") as source:
        operators = list(csv.DictReader(source))
    assert list(operators[0]) == [
        "operator_id",
        "operator_name",
        "operator_kind",
        "kernel_name",
        "source_category",
        "source_library",
        "classification_rule",
        "execution_origin",
        "dispatch_operator_name",
        "dispatch_impl_id",
        "dispatch_impl_kind",
        "dispatch_vendor",
        "runtime_delegate_category",
        "runtime_delegate_library",
    ]
    assert {row["operator_kind"] for row in operators} == {
        "aten",
        "custom",
        "unattributed",
    }
    assert not any("nccl" in row["kernel_name"].lower() for row in operators)
    mm = next(row for row in operators if row["kernel_name"] == "mm_kernel")
    assert mm["operator_name"] == "aten::mm"
    assert mm["operator_kind"] == "aten"
    assert mm["source_category"] == "torch_aten"
    assert mm["source_library"] == "pytorch"
    mm_shape = next(row for row in shape_rows if row["kernel_name"] == "mm_kernel")
    assert mm_shape["source_category"] == "torch_aten"
    assert json.loads(mm_shape["input_shapes"]) == [[2, 4], [4, 8]]
    assert summary["source_category_event_count"] == {
        "third_party": 4,
        "vendor": 0,
        "torch_aten": 2,
        "torch_fused": 0,
    }
    assert summary["files"]["operator_list"]["row_count"] == len(operators)
    assert len(summary["files"]["operator_list"]["sha256"]) == 64


def test_torch_compile_kernels_are_identified_from_profiler_metadata(tmp_path):
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    events = [
        _event(
            "cpu_op",
            "triton_poi_fused_add_mul_0",
            11,
            timestamp=20,
            duration=4,
            shapes=[[64], [64]],
            dtypes=["float", "float"],
            extra_args={
                "kernel_hash": "abc123",
                "kernel_backend": "triton",
                "kernel_file": "/tmp/torchinductor_root/ab/kernel.py",
            },
        ),
        _event(
            "kernel",
            "triton_poi_fused_add_mul_0",
            11,
            timestamp=100,
            duration=9,
            pid=0,
        ),
    ]
    _write_trace(trace, events, compact=True)
    output = tmp_path / "results"
    summary = generate_reports(trace, output)

    assert summary["execution_origin_event_count"] == {"torch_compile": 1}
    assert summary["source_category_event_count"] == {
        "third_party": 0,
        "vendor": 0,
        "torch_aten": 0,
        "torch_fused": 1,
    }
    assert summary["torch_compile"] == {
        "inventory_policy": "included_as_torch_fused",
        "detection": (
            "torch profiler cpu_op metadata: kernel_hash + kernel_backend + "
            "kernel_file under torchinductor"
        ),
        "kernel_event_count": 1,
        "kernel_time_us": 9.0,
        "percent_of_all_compute": "100.000%",
        "unique_stable_kernel_names": 1,
    }
    with (output / "operator_list.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["kernel_name"] == "triton_poi_fused_add_mul_0"
    assert row["source_category"] == "torch_fused"
    assert row["execution_origin"] == "torch_compile"


def test_reference_reports_keep_fused_communication_compute_and_stage_family(
    tmp_path,
):
    events = [
        _event(
            "cpu_op",
            "sglang::fused_allreduce_rmsnorm",
            20,
            timestamp=10,
            duration=5,
            shapes=[[2, 128]],
            dtypes=["c10::BFloat16"],
        ),
        _event(
            "kernel",
            "flashinfer_allreduce_fusion_norm_kernel",
            20,
            timestamp=100,
            duration=11,
            pid=0,
        ),
    ]
    for index, kernel_name in enumerate(
        (
            "moe_align_block_size_stage1",
            "moe_align_block_size_stage2_vec",
            "moe_align_block_size_stage3",
            "moe_align_block_size_stage4",
        ),
        start=1,
    ):
        events.extend(
            [
                _event(
                    "cpu_op",
                    "sglang::moe_align_block_size",
                    20 + index,
                    timestamp=20 + index,
                    duration=3,
                    shapes=[[64]],
                    dtypes=["c10::Int"],
                ),
                _event(
                    "kernel",
                    kernel_name,
                    20 + index,
                    timestamp=120 + index,
                    duration=index,
                    pid=0,
                ),
            ]
        )
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(trace, events, compact=True)
    output = tmp_path / "results"

    summary = generate_reports(trace, output)

    assert summary["runtime_kernel_event_count"] == 5
    assert summary["compute_kernel_event_count"] == 5
    assert summary["excluded_event_count_by_reason"] == {}
    with (output / "operator_list.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    fused = next(
        row for row in rows if row["operator_kind"] == "fused_communication_compute"
    )
    assert fused["operator_id"] != "null"
    stages = [row for row in rows if row["operator_name"] == "moe_align_block_size"]
    assert len(stages) == 4
    assert len({row["operator_id"] for row in stages}) == 1
    assert all(summary["validation"].values())


def test_cuda_library_names_are_not_misclassified_as_plugin_vendor(tmp_path):
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    events = [
        _event(
            "kernel",
            "cutlass::Kernel<GemmShape<128, 128, 64>>",
            timestamp=100,
            duration=7,
            pid=0,
        )
    ]
    _write_trace(trace, events, compact=True)
    output = tmp_path / "results"
    summary = generate_reports(trace, output)

    assert summary["source_category_event_count"] == {
        "third_party": 1,
        "vendor": 0,
        "torch_aten": 0,
        "torch_fused": 0,
    }
    with (output / "operator_list.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["source_category"] == "third_party"
    assert row["source_library"] == "cutlass"
    assert row["classification_rule"] == "runtime_kernel_evidence"


def test_vendor_category_requires_explicit_plugin_dispatch_evidence(tmp_path):
    marker = _operator_marker(
        {
            "operator_name": "sglang_fl.vendor.nvidia.fused_op",
            "input_shapes": [[4, 128]],
            "input_dtypes": ["torch.bfloat16"],
            "source_category": "vendor",
            "source_library": "nvidia",
            "classification_rule": "plugin_dispatch_op_impl_kind",
        }
    )
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    events = [
        _event(
            "cpu_op",
            "generic_extension_launch",
            9,
            timestamp=10,
            duration=40,
            shapes=[[4, 128]],
            dtypes=["c10::BFloat16"],
        ),
        _event("user_annotation", marker, timestamp=20, duration=20),
        _event(
            "cuda_driver",
            "cuLaunchKernelEx",
            timestamp=25,
            duration=1,
            extra_args={"correlation": 17},
        ),
        _event(
            "kernel",
            "cutlass_vendor_fused_kernel",
            9,
            timestamp=100,
            duration=7,
            pid=0,
            extra_args={"correlation": 17},
        ),
    ]
    _write_trace(trace, events, compact=True)
    output = tmp_path / "results"
    summary = generate_reports(trace, output)

    assert summary["source_category_event_count"] == {
        "third_party": 0,
        "vendor": 1,
        "torch_aten": 0,
        "torch_fused": 0,
    }
    with (output / "operator_list.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["source_category"] == "vendor"
    assert row["source_library"] == "nvidia"
    assert row["classification_rule"] == "plugin_dispatch_op_impl_kind"


def test_vendor_dispatch_owns_all_nested_compute_kernels(tmp_path):
    marker = _operator_marker(
        {
            "operator_name": "rms_norm",
            "input_shapes": [[], [4, 128]],
            "input_dtypes": ["Object[RMSNorm]", "torch.bfloat16"],
            "source": "sglang_fl_dispatch",
            "dispatch_operator_name": "rms_norm",
            "dispatch_impl_id": "vendor.cuda",
            "dispatch_impl_kind": "vendor",
            "dispatch_vendor": "cuda",
        }
    )
    events = [
        _event("user_annotation", marker, timestamp=10, duration=60),
        _event(
            "cpu_op",
            "aten::mul",
            1,
            timestamp=15,
            duration=5,
            shapes=[[4, 128], []],
            dtypes=["c10::BFloat16", "Scalar"],
        ),
        _event(
            "cpu_op",
            "sgl_kernel::rmsnorm",
            2,
            timestamp=25,
            duration=5,
            shapes=[[4, 128]],
            dtypes=["c10::BFloat16"],
        ),
        _event(
            "cuda_driver",
            "cuLaunchKernel",
            timestamp=20,
            duration=1,
            extra_args={"correlation": 101},
        ),
        _event(
            "cuda_driver",
            "cuLaunchKernel",
            timestamp=30,
            duration=1,
            extra_args={"correlation": 102},
        ),
        _event(
            "cuda_driver",
            "cuLaunchKernel",
            timestamp=40,
            duration=1,
            extra_args={"correlation": 103},
        ),
        _event(
            "kernel",
            "aten_mul_kernel",
            1,
            timestamp=100,
            duration=3,
            pid=0,
            extra_args={"correlation": 101},
        ),
        _event(
            "kernel",
            "fused_rmsnorm_kernel",
            2,
            timestamp=110,
            duration=5,
            pid=0,
            extra_args={"correlation": 102},
        ),
        _event(
            "kernel",
            "chip_native_rmsnorm_kernel",
            timestamp=120,
            duration=7,
            pid=0,
            extra_args={"correlation": 103},
        ),
    ]
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(trace, events, compact=True)
    output = tmp_path / "results"

    summary = generate_reports(trace, output)

    assert summary["source_category_event_count"] == {
        "third_party": 0,
        "vendor": 3,
        "torch_aten": 0,
        "torch_fused": 0,
    }
    assert summary["dispatch"]["logical_call_count"] == 1
    assert summary["dispatch"]["physical_kernel_event_count"] == 3
    assert summary["dispatch"]["selected_impl_ids"] == ["vendor.cuda"]
    assert summary["dispatch"]["selected_impl_kinds"] == ["vendor"]
    assert summary["dispatch"]["operators"][0]["dispatch_operator_name"] == "rms_norm"
    with (output / "operator_list.csv").open(newline="") as source:
        details = {row["kernel_name"]: row for row in csv.DictReader(source)}
    assert {row["source_category"] for row in details.values()} == {"vendor"}
    assert {row["source_library"] for row in details.values()} == {"cuda"}
    assert {row["classification_rule"] for row in details.values()} == {
        "plugin_vendor_dispatch"
    }
    assert {row["dispatch_impl_id"] for row in details.values()} == {"vendor.cuda"}


def test_vendor_dispatch_category_preserves_upstream_delegate(tmp_path):
    marker = _operator_marker(
        {
            "operator_name": "chunk_gated_delta_rule",
            "input_shapes": [[2, 16, 8, 64]],
            "input_dtypes": ["torch.bfloat16"],
            "source": "sglang_fl_dispatch",
            "dispatch_operator_name": "chunk_gated_delta_rule",
            "dispatch_impl_id": "vendor.cuda",
            "dispatch_impl_kind": "vendor",
            "dispatch_vendor": "cuda",
            "dispatch_runtime_source_category": "third_party",
            "dispatch_runtime_source_library": "sglang",
        }
    )
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(
        trace,
        [
            _event("user_annotation", marker, timestamp=10, duration=20),
            _event(
                "cuda_driver",
                "cuLaunchKernel",
                timestamp=15,
                duration=1,
                extra_args={"correlation": 71},
            ),
            _event(
                "kernel",
                "generic_fused_kernel",
                timestamp=100,
                duration=9,
                pid=0,
                extra_args={"correlation": 71},
            ),
        ],
        compact=True,
    )

    output = tmp_path / "results"
    summary = generate_reports(trace, output)

    assert summary["source_category_event_count"] == {
        "third_party": 0,
        "vendor": 1,
        "torch_aten": 0,
        "torch_fused": 0,
    }
    with (output / "operator_list.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["dispatch_impl_kind"] == "vendor"
    assert row["runtime_delegate_category"] == "third_party"
    assert row["runtime_delegate_library"] == "sglang"
    assert row["source_category"] == "vendor"
    assert row["source_library"] == "cuda"
    assert row["classification_rule"] == "plugin_vendor_dispatch"


def test_dispatch_runtime_source_preserves_vendor_library(tmp_path):
    marker = _operator_marker(
        {
            "operator_name": "rms_norm",
            "input_shapes": [[4, 128]],
            "input_dtypes": ["torch.bfloat16"],
            "source": "sglang_fl_dispatch",
            "dispatch_operator_name": "rms_norm",
            "dispatch_impl_id": "vendor.ascend",
            "dispatch_impl_kind": "vendor",
            "dispatch_vendor": "ascend",
            "dispatch_runtime_source_category": "vendor",
            "dispatch_runtime_source_library": "ascend_ops",
        }
    )
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(
        trace,
        [
            _event("user_annotation", marker, timestamp=10, duration=20),
            _event(
                "cuda_driver",
                "launchKernel",
                timestamp=15,
                duration=1,
                extra_args={"correlation": 81},
            ),
            _event(
                "kernel",
                "vendor_rmsnorm_kernel",
                timestamp=100,
                duration=9,
                pid=0,
                extra_args={"correlation": 81},
            ),
        ],
        compact=True,
    )

    output = tmp_path / "results"
    summary = generate_reports(trace, output)

    assert summary["source_category_event_count"] == {
        "third_party": 0,
        "vendor": 1,
        "torch_aten": 0,
        "torch_fused": 0,
    }
    with (output / "operator_list.csv").open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["source_category"] == "vendor"
    assert row["source_library"] == "ascend"
    assert row["classification_rule"] == "plugin_vendor_dispatch"
    assert row["runtime_delegate_category"] == "vendor"
    assert row["runtime_delegate_library"] == "ascend_ops"
