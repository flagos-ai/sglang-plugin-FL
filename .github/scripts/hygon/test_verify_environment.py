# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for vendor-stack validation and Hygon test selection."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import unittest
from importlib import metadata
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
spec = importlib.util.spec_from_file_location(
    "hygon_verify_environment", Path(__file__).with_name("verify_environment.py")
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

TestRunner = importlib.import_module("tests.run").TestRunner
ModelConfig = __import__(
    "tests.utils.model_config", fromlist=["ModelConfig"]
).ModelConfig


class EnvironmentVerifierTests(unittest.TestCase):
    def setUp(self):
        print_patch = patch("builtins.print")
        print_patch.start()
        self.addCleanup(print_patch.stop)
        self.versions = {
            "sglang": "0.5.18",
            "torch": "2.10.0+vendor.dtk.build",
            "sglang-kernel": "0.4.2.post2+vendor.torch2100",
            "flagtree": "0.6.1+hcu3.6",
            "flag-gems": "5.4.0.dev0",
        }
        self.version_patch = patch.object(
            verifier.metadata, "version", side_effect=self.package_version
        )
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    def package_version(self, name):
        if name not in self.versions:
            raise metadata.PackageNotFoundError(name)
        return self.versions[name]

    def test_flat_docker_copy_can_capture_base_versions(self):
        source_path = Path(__file__).with_name("verify_environment.py")
        namespace = {
            "__file__": "/opt/hygon-verify-environment.py",
            "__name__": "flat_hygon_verifier",
        }
        exec(
            compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"),
            namespace,
        )
        self.assertEqual(namespace["REPO_ROOT"], Path.cwd())
        namespace["_triton_module_snapshot"] = lambda: {"schema_version": 1}
        with patch.object(Path, "write_text", autospec=True) as writer:
            rc = namespace["main"](["--capture-vendor-stack", "/opt/baseline.json"])
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(writer.call_args_list[0].args[1])["torch"],
            self.versions["torch"],
        )

    def test_local_dtk_versions_are_preserved_without_guessing_tag_metadata(self):
        with (
            patch.object(
                verifier, "_triton_module_snapshot", return_value={"schema_version": 1}
            ),
            patch.object(Path, "write_text", autospec=True) as writer,
        ):
            verifier.capture_vendor_stack(Path("/opt/baseline.json"))
        serialized = writer.call_args_list[0].args[1]
        constraints = writer.call_args_list[1].args[1]
        self.assertEqual(json.loads(serialized)["torch"], self.versions["torch"])
        self.assertIn("torch==2.10.0+vendor.dtk.build", constraints)
        self.assertIn("sglang-kernel==0.4.2.post2+vendor.torch2100", constraints)

    def test_baseline_detects_vendor_wheel_replacement(self):
        baseline = {
            name: self.versions[name] for name in verifier.REQUIRED_VENDOR_DISTRIBUTIONS
        }
        self.versions["torch"] = "2.10.0+replaced.vendor.build"
        with patch.object(Path, "read_text", return_value=json.dumps(baseline)):
            with self.assertRaisesRegex(RuntimeError, "vendor stack changed.*torch"):
                verifier._verify_versions(Path("/opt/baseline.json"))

    def test_missing_baseline_entry_cannot_silently_pass(self):
        with patch.object(Path, "read_text", return_value='{"torch": "2.10.0"}'):
            with self.assertRaisesRegex(RuntimeError, "missing distributions"):
                verifier._verify_versions(Path("/opt/baseline.json"))

    def test_wrong_sglang_release_fails(self):
        self.versions["sglang"] = "0.5.11"
        with self.assertRaisesRegex(RuntimeError, "sglang must have release 0.5.18"):
            verifier._verify_versions(None)

    def test_missing_vendor_distribution_has_clear_diagnostic(self):
        del self.versions["sglang-kernel"]
        with self.assertRaisesRegex(RuntimeError, "missing: sglang-kernel"):
            verifier._verify_versions(None)

    def test_cuda_only_torch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Hygon HCU/HIP"):
            verifier._verify_hygon_torch(
                SimpleNamespace(version=SimpleNamespace(hip=None))
            )

    def test_hcu_and_hip_runtime_markers_are_accepted(self):
        verifier._verify_hygon_torch(
            SimpleNamespace(__hcu_version__="3.6", version=SimpleNamespace(hip=None))
        )
        verifier._verify_hygon_torch(
            SimpleNamespace(version=SimpleNamespace(hip="6.4"))
        )

    def test_shadowing_old_plugin_copy_is_rejected(self):
        with patch.object(
            verifier.importlib.util,
            "find_spec",
            return_value=SimpleNamespace(origin=str(REPO_ROOT / "old" / "__init__.py")),
        ):
            with self.assertRaisesRegex(RuntimeError, "older plugin copy"):
                verifier._verify_plugin_path(REPO_ROOT)

    def test_metadata_only_does_not_load_gpu_libraries(self):
        with (
            patch.object(verifier.sys, "version_info", (3, 10, 12)),
            patch.object(verifier, "_verify_plugin_path"),
            patch.object(verifier, "_require_import") as importer,
        ):
            verifier.verify(metadata_only=True)
        importer.assert_not_called()

    def test_missing_python_runtime_dependency_names_the_failed_import(self):
        with patch.object(
            verifier.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'sqlalchemy'"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "runtime import failed: sqlalchemy"
            ):
                verifier._require_import("sqlalchemy")

    def test_missing_flagcx_path_fails_without_native_fallback(self):
        with patch.dict(verifier.os.environ, {"FLAGCX_PATH": ""}):
            with self.assertRaisesRegex(RuntimeError, "FLAGCX_PATH"):
                verifier._verify_flagcx()

    def test_insufficient_gpus_fail_before_loading_kernels(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )
        with self.assertRaisesRegex(RuntimeError, "at least 2 Hygon GPUs"):
            verifier._verify_gpu(torch, 2)

    def test_metadata_only_and_gpu_check_cannot_be_combined(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            verifier.verify(metadata_only=True, require_gpu=True)


@contextmanager
def triton_module_files(contents):
    """Provide real byte streams to the hasher without importing a GPU package."""
    root = (REPO_ROOT / "virtual_vendor" / "triton").resolve()

    def relative(path):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return None

    def is_file(path):
        return relative(path) in contents

    def open_file(path, mode):
        if mode != "rb":
            raise AssertionError(f"unexpected module-file mode: {mode}")
        return io.BytesIO(contents[relative(path)])

    with (
        patch.object(
            verifier.importlib.util,
            "find_spec",
            return_value=SimpleNamespace(origin=str(root / "__init__.py")),
        ) as finder,
        patch.object(Path, "is_file", autospec=True, side_effect=is_file),
        patch.object(Path, "open", autospec=True, side_effect=open_file),
    ):
        yield finder


class VendorModuleProtectionTests(unittest.TestCase):
    def setUp(self):
        self.versions = {
            "sglang": "0.5.18",
            "torch": "2.10.0+vendor.dtk.build",
            "sglang-kernel": "0.4.2.post2+vendor.torch2100",
            "flagtree": "0.6.1+hcu3.6",
            "flag-gems": "5.4.0.dev0",
        }
        self.baseline_versions = {
            name: self.versions[name] for name in verifier.REQUIRED_VENDOR_DISTRIBUTIONS
        }
        self.files = {
            "__init__.py": b"FlagTree HCU triton package",
            "compiler/compiler.py": b"FlagTree HCU compiler implementation",
            "_C/libtriton.so": b"FlagTree HCU native library",
        }
        self.baseline = Path("/opt/vendor-stack.json")
        version_patch = patch.object(
            verifier.metadata, "version", side_effect=self.version
        )
        version_patch.start()
        self.addCleanup(version_patch.stop)
        print_patch = patch("builtins.print")
        self.print_mock = print_patch.start()
        self.addCleanup(print_patch.stop)

    def version(self, name):
        if name not in self.versions:
            raise metadata.PackageNotFoundError(name)
        return self.versions[name]

    def verify_files(self, *, current=None, captured_files=None):
        with triton_module_files(
            self.files if captured_files is None else captured_files
        ):
            captured = verifier._triton_module_snapshot()

        def read_text(path, **kwargs):
            if path == self.baseline:
                return json.dumps(self.baseline_versions)
            if path == self.baseline.with_suffix(".modules.json"):
                return json.dumps(captured)
            raise FileNotFoundError(path)

        with (
            patch.object(Path, "read_text", autospec=True, side_effect=read_text),
            triton_module_files(self.files if current is None else current),
        ):
            verifier._verify_versions(self.baseline)

    def test_capture_writes_module_hashes_without_importing_torch_or_triton(self):
        with (
            triton_module_files(self.files) as finder,
            patch.object(verifier.importlib, "import_module") as importer,
            patch.object(Path, "write_text", autospec=True) as writer,
        ):
            verifier.capture_vendor_stack(self.baseline)
        importer.assert_not_called()
        finder.assert_called_once_with("triton")
        writes = {call.args[0]: call.args[1] for call in writer.call_args_list}
        # Existing consumers still receive a plain distribution -> version map.
        self.assertEqual(json.loads(writes[self.baseline]), self.baseline_versions)
        snapshot = json.loads(writes[self.baseline.with_suffix(".modules.json")])
        self.assertEqual(
            snapshot["triton"]["files"],
            {
                name: hashlib.sha256(data).hexdigest()
                for name, data in self.files.items()
            },
        )
        self.assertIn(self.baseline.with_suffix(".constraints.txt"), writes)

    def test_added_triton_distribution_is_rejected_when_flagtree_version_is_unchanged(
        self,
    ):
        self.versions["triton"] = "3.8.0"
        with self.assertRaisesRegex(RuntimeError, "gained distribution.*triton==3.8.0"):
            self.verify_files()
        self.assertEqual(self.versions["flagtree"], self.baseline_versions["flagtree"])

    def test_other_new_optional_vendor_distribution_is_rejected(self):
        self.versions["torchvision"] = "0.99.0"
        with self.assertRaisesRegex(RuntimeError, "gained distribution.*torchvision"):
            self.verify_files()

    def test_unchanged_versions_and_module_bytes_pass(self):
        self.verify_files()
        messages = " ".join(
            str(call.args[0]) for call in self.print_mock.call_args_list
        )
        self.assertIn("module SHA256 checks passed (3 files)", messages)

    def test_overwritten_module_bytes_fail_despite_identical_distribution_versions(
        self,
    ):
        original_versions = dict(self.versions)
        for relative in self.files:
            with self.subTest(relative=relative):
                current = dict(self.files)
                current[relative] += b" replaced by upstream pip triton"
                with self.assertRaisesRegex(
                    RuntimeError, "module hash changed.*" + relative
                ):
                    self.verify_files(current=current)
        self.assertEqual(self.versions, original_versions)

    def test_optional_native_library_added_or_removed_is_detected(self):
        no_native = {
            name: data for name, data in self.files.items() if name != "_C/libtriton.so"
        }
        with self.assertRaisesRegex(RuntimeError, "module file set changed"):
            self.verify_files(current=no_native)
        with self.assertRaisesRegex(RuntimeError, "module file set changed"):
            self.verify_files(captured_files=no_native)

    def test_capture_rejects_missing_required_compiler_before_writing_baseline(self):
        with (
            triton_module_files({"__init__.py": b"missing compiler"}),
            patch.object(Path, "write_text", autospec=True) as writer,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "required.*triton file is missing"
            ):
                verifier.capture_vendor_stack(self.baseline)
        writer.assert_not_called()

    def test_old_baseline_without_module_sidecar_has_clear_diagnostic(self):
        def read_text(path, **kwargs):
            if path == self.baseline:
                return json.dumps(self.baseline_versions)
            raise FileNotFoundError(path)

        with patch.object(Path, "read_text", autospec=True, side_effect=read_text):
            with self.assertRaisesRegex(
                RuntimeError, "module baseline is missing.*not checked.*original"
            ):
                verifier._verify_versions(self.baseline)
        messages = " ".join(
            str(call.args[0]) for call in self.print_mock.call_args_list
        )
        self.assertNotIn("module SHA256 checks passed", messages)


class SelectedTensorPathTests(unittest.TestCase):
    """Exercise real CPU tensors; these tests do not claim GPU kernel coverage."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        path = (
            REPO_ROOT / "sglang_fl/dispatch/backends/reference/impl/gemma_rms_norm.py"
        )
        reference_spec = importlib.util.spec_from_file_location(
            "hygon_verifier_reference_gemma", path
        )
        reference = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference)
        cls.gemma_reference = staticmethod(reference.gemma_rms_norm_torch)

    def setUp(self):
        print_patch = patch("builtins.print")
        self.print_mock = print_patch.start()
        self.addCleanup(print_patch.stop)

    def cpu_silu_and_mul(self, _obj, x):
        left, right = x.float().chunk(2, dim=-1)
        # Independent CPU formula; this is not a pretend FlagGems GPU launch.
        return (left * self.torch.sigmoid(left) * right).to(x.dtype)

    def check_cpu_paths(self, *, activation=None, normalization=None):
        verifier._verify_selected_tensor_paths(
            self.torch,
            device="cpu",
            silu_and_mul=activation or self.cpu_silu_and_mul,
            gemma_rms_norm=normalization or self.gemma_reference,
        )

    def test_real_cpu_bf16_and_reference_paths_match_numerically(self):
        self.check_cpu_paths()
        messages = " ".join(
            str(call.args[0]) for call in self.print_mock.call_args_list
        )
        self.assertIn("device=cpu", messages)
        self.assertNotIn("GPU", messages)

    def test_wrong_activation_output_is_detected(self):
        def incorrect(_obj, x):
            return self.torch.zeros_like(x[..., : x.shape[-1] // 2])

        with self.assertRaisesRegex(RuntimeError, "silu_and_mul.*CPU reference"):
            self.check_cpu_paths(activation=incorrect)

    def test_wrong_gemma_output_is_detected(self):
        def incorrect(_obj, x, residual=None):
            return self.torch.zeros_like(x)

        with self.assertRaisesRegex(RuntimeError, "GemmaRMSNorm.*CPU reference"):
            self.check_cpu_paths(normalization=incorrect)

    def test_wrong_residual_sum_is_detected(self):
        def incorrect(obj, x, residual=None):
            output = self.gemma_reference(obj, x, residual)
            if residual is None:
                return output
            normalized, combined = output
            return normalized, combined + 1

        with self.assertRaisesRegex(RuntimeError, "residual sum.*CPU reference"):
            self.check_cpu_paths(normalization=incorrect)

    def test_wrong_selected_path_dtype_is_detected(self):
        def incorrect(_obj, x):
            return self.cpu_silu_and_mul(None, x).float()

        with self.assertRaisesRegex(RuntimeError, "expected bfloat16/cpu"):
            self.check_cpu_paths(activation=incorrect)

    def test_cpu_validation_cannot_satisfy_require_gpu(self):
        with (
            patch.object(self.torch.cuda, "is_available", return_value=False),
            patch.object(verifier, "_require_import") as importer,
        ):
            with self.assertRaisesRegex(RuntimeError, "is_available.*false"):
                verifier._verify_gpu(self.torch, 1)
        importer.assert_not_called()

    def test_empty_native_rmsnorm_is_informational_and_never_called(self):
        def wrapper(input, weight, eps, out=None, enable_pdl=False):
            raise AssertionError("native RMSNorm must not execute")

        kernel = SimpleNamespace(
            rmsnorm=wrapper, __file__="/vendor/sgl_kernel/__init__.py"
        )
        torch_metadata = SimpleNamespace(
            ops=SimpleNamespace(sgl_kernel=SimpleNamespace())
        )
        verifier._report_sgl_kernel(torch_metadata, kernel)
        messages = " ".join(
            str(call.args[0]) for call in self.print_mock.call_args_list
        )
        self.assertIn("native-rmsnorm-registered=False (informational)", messages)
        self.assertIn("input, weight, eps, out=None, enable_pdl=False", messages)
        self.assertIn("native RMSNorm ABI/execution was not validated", messages)

    def test_registered_native_rmsnorm_is_still_not_executed(self):
        def native(*args):
            raise AssertionError("an unselected native operator must not execute")

        torch_metadata = SimpleNamespace(
            ops=SimpleNamespace(sgl_kernel=SimpleNamespace(rmsnorm=native))
        )
        verifier._report_sgl_kernel(torch_metadata, SimpleNamespace())
        messages = " ".join(
            str(call.args[0]) for call in self.print_mock.call_args_list
        )
        self.assertIn("native-rmsnorm-registered=True (informational)", messages)
        self.assertIn("native RMSNorm ABI/execution was not validated", messages)


class HygonAcceptanceSelectionTests(unittest.TestCase):
    def test_eager_and_graph_select_both_models_and_all_three_tasks(self):
        expected_pairs = {
            (task, case)
            for task in ("inference", "serving", "concurrent")
            for case in ("27b_tp2_hygon", "35b_a3b_tp2_hygon")
        }
        for device, graph in (("bw1000", False), ("bw1000_graph", True)):
            with self.subTest(device=device):
                runner = TestRunner("hygon", device, "e2e")
                selected = runner.discover_tests()
                self.assertEqual(
                    {(item.name.split("/")[1], item.case) for item in selected},
                    expected_pairs,
                )
                for item in selected:
                    config = ModelConfig.load(item.model, item.case)
                    config.apply_engine_overrides(item.engine_overrides)
                    kwargs = config.engine_kwargs()
                    self.assertEqual(kwargs["tp_size"], 2)
                    self.assertEqual(kwargs["pp_size"], 1)
                    self.assertEqual(kwargs["attention_backend"], "hcu")
                    self.assertEqual(kwargs["dtype"], "bfloat16")
                    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
                    self.assertEqual(kwargs["page_size"], 64)
                    self.assertTrue(kwargs["disable_radix_cache"])
                    self.assertEqual(kwargs["disable_cuda_graph"], not graph)
                    self.assertEqual(kwargs["cuda_graph_backend_prefill"], "disabled")
                    self.assertNotIn("enable_breakable_cuda_graph", kwargs)
                    self.assertNotIn("cuda_graph_max_bs", kwargs)
                    if graph:
                        self.assertEqual(kwargs["cuda_graph_backend_decode"], "full")
                        self.assertEqual(kwargs["cuda_graph_max_bs_decode"], 16)

    def test_benchmark_acceptance_remains_opt_in(self):
        self.assertEqual(
            TestRunner("hygon", "bw1000", "benchmark").discover_tests(), []
        )


if __name__ == "__main__":
    unittest.main()
