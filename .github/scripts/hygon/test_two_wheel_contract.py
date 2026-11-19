"""Two-wheel staging/coupling tests (no GPU, no image build).

Covers the product-candidate wiring that the next image build will run:

* the staged containerfile references exactly the artifacts stage_wheels.sh
  assembles (two wheels + the excludes file), stays assembly-only (no
  bind-mounted build scripts), and its single uv install carries the full
  resolver contract, mirrored flag-for-flag by the pre-build dry-run gate
  in build_two_wheel.sh;
* every script the external build/acceptance wrappers mount into their
  docker runs exists in docker/hygon/scripts/;
* the static runtime reference snapshots stay out of the repository (the
  reference data is collected dynamically from the pinned base digest);
* the excludes file is the single deliberate resolver exclusion;
* the plugin wheel acceptance script accepts the real wheel built from
  this checkout and rejects tampered variants (wrong extra anchor, triton
  declaration, missing entry points, shipped tests).
"""

import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER = REPO_ROOT / "docker" / "hygon"
SCRIPTS = DOCKER / "scripts"


def uv_args(chunk: str) -> list[str]:
    """Collect the resolver arguments of one uv invocation block."""
    lines = []
    for line in chunk.splitlines():
        stripped = line.strip().rstrip("\\").strip()
        if stripped.startswith("--") or stripped.startswith("'/tmp/wheels/"):
            lines.append(stripped)
        elif stripped.startswith(">") or stripped.startswith("&&"):
            break
    return lines


class StagingCouplingTests(unittest.TestCase):
    def test_containerfile_and_staging_agree_on_artifacts(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        staging = (DOCKER / "stage_wheels.sh").read_text()
        for wheel in (
            "sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl",
            "sglang_fl-0.1.0-py3-none-any.whl",
            "runtime-0.5.18.excludes.txt",
        ):
            self.assertIn(wheel, containerfile, f"containerfile never stages {wheel}")
            self.assertIn(wheel, staging, f"stage_wheels.sh never assembles {wheel}")

    def test_containerfile_is_assembly_only(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn(
            "/ctx/", containerfile, "build-time bind-mounted logic crept back in"
        )
        for banned in (
            "verify_distributions",
            "verify_site_files",
            "verify_module_provenance",
            "verify_runtime_immutable",
            "validate_uv_plan",
            "capture_runtime_baseline",
            "pip check",
            "pip_check_exceptions",
            "sglang_import_smoke",
        ):
            self.assertNotIn(
                banned, containerfile, f"acceptance logic {banned!r} in the recipe"
            )

    def test_external_wrapper_scripts_exist(self):
        referenced = set()
        for wrapper in ("build_two_wheel.sh", "scripts/verify_image.sh"):
            text = (DOCKER / wrapper).read_text()
            referenced |= set(re.findall(r"/scripts/([\w.-]+)", text))
        self.assertTrue(referenced, "wrappers reference no scripts?")
        for script in referenced:
            path = SCRIPTS / script
            self.assertTrue(path.is_file(), f"referenced script missing: {path}")

    def test_uv_install_contract_and_dry_run_mirror_it(self):
        """The single real install and the pre-build dry-run must agree.

        --index-strategy first-index was measured insufficient on the real
        runtime (the Hygon index has xgrammar but not ==0.2.1, and
        first-index stops at the name instead of falling through to PyPI);
        the install must use unsafe-first-match with the full flag
        contract, and build_two_wheel.sh's dry-run.sh must carry exactly
        the same arguments plus --dry-run (it plan-gates the build against
        the same pinned base digest).
        """
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        marker = "/usr/local/bin/uv pip install"
        parts = containerfile.split(marker)
        self.assertEqual(len(parts), 2, "expected exactly one uv invocation")

        real = uv_args(parts[1])
        self.assertNotIn("--dry-run", real)
        self.assertEqual(real.count("--index-strategy unsafe-first-match"), 1)
        self.assertNotIn("first-index", " ".join(real))
        for required in (
            "--excludes /tmp/runtime-0.5.18.excludes.txt",
            "--constraints /opt/sglang-ci/runtime-constraints.txt",
            "--constraints /opt/sglang-ci/top-level-constraints.txt",
            "'/tmp/wheels/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64"
            ".whl[srt_empty,runtime_common]'",
            "'/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[hygon]'",
        ):
            self.assertIn(required, real, f"install lost contract flag {required!r}")

        build_wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        heredoc = re.search(
            r"cat > \"\$plan/dry-run\.sh\" <<'EOF'\n(.*?)\nEOF\n",
            build_wrapper,
            re.DOTALL,
        )
        self.assertIsNotNone(heredoc, "build_two_wheel.sh lost the dry-run heredoc")
        # Split the heredoc at the same marker so both extractions start
        # from the identical remainder-of-marker-line form.
        dry_run = uv_args(heredoc.group(1).split(marker)[1])
        self.assertIn("--dry-run", dry_run)
        self.assertEqual(
            [a for a in dry_run if a != "--dry-run"],
            real,
            "dry-run and real install arguments diverge",
        )

    def test_no_committed_runtime_snapshots(self):
        for name in (
            "runtime-0.5.18.runtime-freeze.txt",
            "runtime-0.5.18.site-files.sha256",
            "runtime-0.5.18.pip-check-baseline.txt",
            "runtime-0.5.18.provenance.json",
            "scripts/capture_runtime_baseline.py",
        ):
            self.assertFalse(
                (DOCKER / name).exists(),
                f"static runtime snapshot returned to the repo: {name}",
            )

    def test_excludes_file_is_single_item_triton(self):
        content = (DOCKER / "runtime-0.5.18.excludes.txt").read_text()
        requirements = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertEqual(requirements, ["triton"])


class PluginWheelAcceptanceTests(unittest.TestCase):
    """Drive check_plugin_wheel.py against real and tampered wheels."""

    @classmethod
    def setUpClass(cls):
        cls.stage = tempfile.mkdtemp(prefix="hygon-2w-acceptance.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                cls.stage,
                str(REPO_ROOT),
            ],
            check=True,
            capture_output=True,
        )
        cls.wheel = Path(cls.stage) / "sglang_fl-0.1.0-py3-none-any.whl"

    def run_checker(self, wheel: Path) -> int:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "check_plugin_wheel.py"),
                str(wheel),
            ],
            capture_output=True,
            text=True,
        ).returncode

    def test_real_wheel_is_accepted(self):
        self.assertEqual(self.run_checker(self.wheel), 0)

    def tamper(self, mutation) -> Path:
        target = Path(self.stage) / f"tampered-{mutation.__name__}.whl"
        with (
            zipfile.ZipFile(self.wheel) as archive,
            zipfile.ZipFile(target, "w") as out,
        ):
            for name in archive.namelist():
                data = archive.read(name)
                if name == "sglang_fl-0.1.0.dist-info/METADATA":
                    metadata = Parser().parsestr(data.decode("utf-8"))
                    mutation(metadata)
                    lines = []
                    for header in metadata.items():
                        lines.append(f"{header[0]}: {header[1]}")
                    body = metadata.get_payload()
                    data = ("\n".join(lines) + ("\n\n" + body if body else "")).encode()
                out.writestr(name, data)
        return target

    def test_wrong_extra_anchor_is_rejected(self):
        def mutate(metadata):
            del metadata["Requires-Dist"]

        wheel = self.tamper(mutate)
        self.assertNotEqual(self.run_checker(wheel), 0)

    def test_triton_declaration_is_rejected(self):
        def mutate(metadata):
            metadata["Requires-Dist"] = 'triton; extra == "hygon"'

        wheel = self.tamper(mutate)
        self.assertNotEqual(self.run_checker(wheel), 0)

    def test_missing_entry_points_are_rejected(self):
        target = Path(self.stage) / "tampered-entrypoints.whl"
        with (
            zipfile.ZipFile(self.wheel) as archive,
            zipfile.ZipFile(target, "w") as out,
        ):
            for name in archive.namelist():
                if name.endswith("entry_points.txt"):
                    out.writestr(name, "")
                else:
                    out.writestr(name, archive.read(name))
        self.assertNotEqual(self.run_checker(target), 0)

    def test_wrong_entry_point_value_is_rejected(self):
        target = Path(self.stage) / "tampered-ep-value.whl"
        with (
            zipfile.ZipFile(self.wheel) as archive,
            zipfile.ZipFile(target, "w") as out,
        ):
            for name in archive.namelist():
                data = archive.read(name)
                if name.endswith("entry_points.txt"):
                    data = data.replace(
                        b"sglang_fl = sglang_fl:activate_platform",
                        b"sglang_fl = sglang_fl:wrong_callable",
                    )
                out.writestr(name, data)
        self.assertNotEqual(self.run_checker(target), 0)

    def test_shipped_tests_are_rejected(self):
        target = Path(self.stage) / "tampered-tests.whl"
        with (
            zipfile.ZipFile(self.wheel) as archive,
            zipfile.ZipFile(target, "w") as out,
        ):
            for name in archive.namelist():
                out.writestr(name, archive.read(name))
            out.writestr("tests/test_smuggled.py", "def test_x(): pass")
        self.assertNotEqual(self.run_checker(target), 0)


class UvPlanParserTests(unittest.TestCase):
    """Parse the REAL uv 0.11.30 dry-run output (stderr, multi-line)."""

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        import tempfile
        import zipfile

        _sys.path.insert(0, str(SCRIPTS))
        import validate_uv_plan

        cls.validate_uv_plan = validate_uv_plan
        cls.tmp = tempfile.mkdtemp(prefix="uv-plan-test.")
        # Real (tiny) wheels so the file:// plan lines have METADATA to read.
        cls.sglang_wheel = (
            Path(cls.tmp) / "sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
        )
        with zipfile.ZipFile(cls.sglang_wheel, "w") as archive:
            archive.writestr(
                "sglang-0.5.18+flagos.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.18+flagos\n",
            )
        cls.plugin_wheel = Path(cls.tmp) / "sglang_fl-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(cls.plugin_wheel, "w") as archive:
            archive.writestr(
                "sglang_fl-0.1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: sglang_fl\nVersion: 0.1.0\n",
            )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    def real_output(self) -> str:
        return (
            "Using Python 3.10.20 environment at: /flagos\n"
            "Resolved 132 packages in 800ms\n"
            "Would download 130 packages\n"
            "Would install 132 packages\n"
            f" + sglang @ file://{self.sglang_wheel}\n"
            f" + sglang_fl @ file://{self.plugin_wheel}\n"
            " + datasets==5.2.0\n"
            " + XGrammar==0.2.1\n"
            " + torch==2.9.0\n"
        )

    def test_parses_multiline_stderr_plan(self):
        planned = self.validate_uv_plan.parse_plan(self.real_output())
        self.assertEqual(planned["datasets"], "5.2.0")
        self.assertEqual(planned["xgrammar"], "0.2.1")  # canonicalized
        self.assertEqual(planned["sglang"], "0.5.18+flagos")
        self.assertEqual(planned["sglang-fl"], "0.1.0")
        self.assertNotIn("triton", planned)

    def test_local_wheel_version_comes_from_metadata(self):
        # A file:// plan line contributes the version from the wheel's own
        # METADATA (here: the plugin wheel masquerading as 'sglang').
        planned = self.validate_uv_plan.parse_plan(
            f" + sglang @ file://{self.plugin_wheel}\n"
        )
        self.assertEqual(planned, {"sglang": "0.1.0"})

    def test_empty_plan_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.validate_uv_plan.parse_plan("Resolved 0 packages\n")


class StageContextSurvivalTests(unittest.TestCase):
    """The printed build context must outlive stage_wheels.sh itself."""

    @classmethod
    def setUpClass(cls):
        import re as _re
        import subprocess as _sp

        completed = _sp.run(
            ["bash", str(DOCKER / "stage_wheels.sh")],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        match = _re.search(r"^context:\s+(\S+)", completed.stdout, _re.MULTILINE)
        assert match, completed.stdout[-2000:]
        cls.context = Path(match.group(1))
        cls.output = completed.stdout

    def test_context_exists_after_script_exit(self):
        self.assertTrue(self.context.is_dir(), f"context vanished: {self.context}")

    def test_context_carries_artifacts_and_containerfile(self):
        for relative in (
            "artifacts/sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl",
            "artifacts/sglang_fl-0.1.0-py3-none-any.whl",
            "artifacts/runtime-0.5.18.excludes.txt",
            "runtime-0.5.18.containerfile",
            "docker/hygon/scripts/validate_uv_plan.py",
        ):
            self.assertTrue((self.context / relative).is_file(), f"missing {relative}")

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.context, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
