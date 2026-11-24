"""Two-wheel staging/coupling tests (no GPU, no image build).

Covers the product-candidate wiring that the next image build will run:

* the staged containerfile references exactly the artifacts stage_wheels.sh
  assembles (two application wheels, corrected xgrammar wheel and the
  transitional excludes file), stays assembly-only (no
  bind-mounted build scripts), and its single uv install carries the full
  resolver contract, mirrored flag-for-flag by the pre-build dry-run gate
  in build_two_wheel.sh;
* every script the external build/acceptance wrappers mount into their
  docker runs exists in docker/hygon/scripts/;
* the static runtime reference snapshots stay out of the repository (the
  reference data is collected dynamically from the digest each build
  resolves once from the moving tag);
* the excludes file is the single deliberate resolver exclusion;
* the plugin wheel acceptance script accepts the real wheel built from
  this checkout and rejects tampered variants (wrong extra anchor, triton
  declaration, missing entry points, shipped tests).
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
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
RELEASE_RECIPE = DOCKER / "runtime-0.5.18.containerfile"
OFFLINE_RECIPE = DOCKER / "runtime-0.5.18.offline-wheelhouse.containerfile"
DRY_RUN_HEREDOC = r"cat > \"\$plan/dry-run\.sh\" <<'EOF'\n(.*?)\nEOF\n"
CANDIDATE_TAG = (
    "harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-2w-offline-"
    "{commit}-20260101T000000Z"
)


def load_script(name: str):
    """Import one of docker/hygon/scripts/*.py under its own name."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def uv_args(chunk: str) -> list[str]:
    """Collect the resolver arguments of one uv invocation block."""
    lines = []
    for line in chunk.splitlines():
        stripped = line.strip().rstrip("\\").strip()
        if stripped.startswith(("--", "'/tmp/artifacts/")):
            lines.append(stripped)
        elif stripped.startswith(">") or stripped.startswith("&&"):
            break
    return lines


class StagingCouplingTests(unittest.TestCase):
    def test_containerfile_and_staging_agree_on_artifacts(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        staging = (DOCKER / "stage_wheels.sh").read_text()
        for wheel in (
            "sglang_fl-0.1.0-py3-none-any.whl",
            "runtime-0.5.18.excludes.txt",
        ):
            self.assertIn(wheel, containerfile, f"containerfile never stages {wheel}")
            self.assertIn(wheel, staging, f"stage_wheels.sh never assembles {wheel}")
        # the derived HCU SGLang wheel: literal in the containerfile,
        # composed from DERIVED_SGLANG_VERSION in staging (which also
        # derives it via tools/wheels/hygon/derive_sglang_wheel.py)
        derived = "sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl"
        self.assertIn(derived, containerfile)
        self.assertIn('DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"', staging)
        self.assertIn(
            'DERIVED_SGLANG_WHEEL_NAME="sglang-${DERIVED_SGLANG_VERSION}'
            '-cp310-cp310-linux_x86_64.whl"',
            staging,
        )
        self.assertIn("tools/wheels/hygon/derive_sglang_wheel.py", staging)
        xgrammar = (
            "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64."
            "manylinux_2_28_x86_64.whl"
        )
        self.assertIn(xgrammar, containerfile)
        self.assertIn('DERIVED_XGRAMMAR_VERSION="0.2.1+flagos.hcu1"', staging)
        self.assertIn("tools/wheels/hygon/derive_metadata_wheel.py", staging)
        self.assertIn("XGRAMMAR_INPUT_WHEEL", staging)
        self.assertIn(
            "5b80b8060fcf6c94324f563e1d4778029b21d35c556dca5bbf49428d307eca37",
            staging,
        )

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

    def test_containerfile_is_a_release_image_not_a_ci_workspace(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn(
            "WORKDIR", containerfile, "the recipe must inherit the Runtime WORKDIR"
        )
        self.assertNotIn(
            "/opt/sglang-ci",
            containerfile,
            "CI working-directory semantics crept back into the release image",
        )
        # Build inputs are read-only mounts and never enter an image layer.
        self.assertIn(
            "--mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly",
            containerfile,
        )
        self.assertNotIn("COPY artifacts/", containerfile)
        self.assertNotIn("/tmp/requirements.txt", containerfile)
        self.assertFalse((DOCKER / "requirements.txt").exists())
        self.assertNotIn("docker/hygon/stubs", containerfile)
        self.assertFalse(any(path.is_file() for path in (DOCKER / "stubs").rglob("*")))
        self.assertNotIn("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", containerfile)
        # The HCU gating patch is folded into the derived SGLang wheel
        # BEFORE staging: no patch may be referenced or applied here.
        self.assertNotIn("hcu.patch", containerfile)
        self.assertNotIn("git apply", containerfile)

    def test_containerfile_base_is_the_moving_semver_tag(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        arg = re.search(r"^ARG BASE_IMAGE=(\S+)", containerfile, re.MULTILINE)
        self.assertIsNotNone(arg, "no ARG BASE_IMAGE")
        self.assertEqual(
            arg.group(1),
            "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04:2.3.0",
        )
        self.assertNotIn(
            "@sha256:",
            arg.group(1),
            "the file must carry the moving tag; digest pinning is build-time",
        )

    def test_pypi_fallback_is_the_tencent_mirror(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertIn(
            "ARG TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple",
            containerfile,
        )
        self.assertNotRegex(containerfile, r"https://pypi\.org")
        build_wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        self.assertIn(
            "TENCENT_INDEX=https://mirrors.cloud.tencent.com/pypi/simple",
            build_wrapper,
        )
        self.assertNotIn("PYPI_INDEX", build_wrapper)

    def test_external_wrapper_scripts_exist(self):
        referenced = set()
        for wrapper in ("build_two_wheel.sh", "scripts/verify_image.sh"):
            text = (DOCKER / wrapper).read_text()
            referenced |= set(re.findall(r"/scripts/([\w.-]+)", text))
        self.assertTrue(referenced, "wrappers reference no scripts?")
        for script in referenced:
            path = SCRIPTS / script
            self.assertTrue(path.is_file(), f"referenced script missing: {path}")

    def test_the_release_install_keeps_the_full_flag_contract(self):
        """The release recipe's single uv install carries the index contract.

        --index-strategy first-index was measured insufficient on the real
        runtime (the Hygon index has xgrammar but not ==0.2.1, and
        first-index stops at the name instead of falling through to PyPI);
        the install must use unsafe-first-match with the full flag contract.
        The pre-build dry-run is generated from this very instruction
        (GeneratedDryRunTests).
        """
        containerfile = RELEASE_RECIPE.read_text()
        marker = "/usr/local/bin/uv pip install"
        parts = containerfile.split(marker)
        self.assertEqual(len(parts), 2, "expected exactly one uv invocation")

        real = uv_args(parts[1])
        self.assertNotIn("--dry-run", real)
        self.assertFalse(any(argument.startswith("-r ") for argument in real))
        self.assertEqual(real.count("--no-cache"), 1)
        self.assertEqual(real.count("--index-strategy unsafe-first-match"), 1)
        self.assertNotIn("first-index", " ".join(real))
        self.assertEqual(real.count('--index "$TENCENT_INDEX"'), 1)
        self.assertNotIn("--no-index", real)
        for required in (
            "--excludes /tmp/artifacts/runtime-0.5.18.excludes.txt",
            "--constraints /tmp/artifacts/runtime-constraints.txt",
            "'/tmp/artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'",
            "'/tmp/artifacts/sglang-0.5.18+flagos.hcu1-cp310-cp310-"
            "linux_x86_64.whl[srt_empty,runtime_common]'",
            "'/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[hygon]'",
        ):
            self.assertIn(required, real, f"install lost contract flag {required!r}")

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


class SetupScriptContractTests(unittest.TestCase):
    """The CI job's runner install must follow the no-pypi.org network rule."""

    def test_runner_install_uses_only_the_tencent_mirror(self):
        setup = (REPO_ROOT / ".github/scripts/hygon/setup.sh").read_text()
        # --default-index replaces uv's pypi.org default outright; a plain
        # --index extra would still leave pypi.org as the fallback.
        self.assertIn(
            "--default-index https://mirrors.cloud.tencent.com/pypi/simple",
            setup,
        )
        self.assertNotIn("pypi.org", setup)

    def test_runner_dependencies_have_a_single_source(self):
        setup = (REPO_ROOT / ".github/scripts/hygon/setup.sh").read_text()
        self.assertEqual(
            setup.count("uv pip install"),
            1,
            "setup.sh must install dependencies in exactly one uv call",
        )
        self.assertIn(
            '-r "$PLUGIN_ROOT/.github/scripts/hygon/test-requirements.txt"', setup
        )
        self.assertNotRegex(
            setup,
            r"pytest==",
            "test dependencies are pinned only in test-requirements.txt",
        )


class BuildProvenanceContractTests(unittest.TestCase):
    """A releasable build must come from a clean, unmoving worktree."""

    def test_release_build_requires_a_clean_worktree(self):
        wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        self.assertIn(
            "status --porcelain",
            wrapper,
            "the release gate must reject staged/unstaged/untracked files",
        )
        self.assertIn("--allow-dirty", wrapper)
        self.assertIn(
            "--allow-dirty cannot be combined with --push",
            wrapper,
            "dirty dev builds must never publish",
        )

    def test_push_refuses_a_plugin_wheel_built_from_the_checkout(self):
        """Run the release gate of build_two_wheel.sh as written."""
        wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        start = wrapper.index('plugin_provenance="$context/artifacts/')
        end = wrapper.index('echo "plugin wheel mode: $plugin_mode"')
        gate = wrapper[start:end]
        with tempfile.TemporaryDirectory(prefix="push-gate-test.") as tmp:
            context = Path(tmp) / "context"
            (context / "artifacts").mkdir(parents=True)
            provenance = (
                context / "artifacts/sglang_fl-0.1.0-py3-none-any.provenance.json"
            )
            script = Path(tmp) / "gate.sh"
            script.write_text(
                "\n".join(
                    [
                        "set -euo pipefail",
                        'context="$1"; push="$2"',
                        'PLUGIN_WHEEL_NAME="sglang_fl-0.1.0-py3-none-any.whl"',
                        gate,
                        'echo "GATE-PASSED mode=$plugin_mode"',
                    ]
                )
            )

            def run(push):
                return subprocess.run(
                    ["bash", str(script), str(context), push],
                    capture_output=True,
                    text=True,
                )

            # a release needs the published wheel; a dev build may use the
            # checkout build
            for mode, push, accepted in (
                ("built-from-checkout", "1", False),
                ("published", "1", True),
                ("built-from-checkout", "0", True),
            ):
                provenance.write_text(json.dumps({"mode": mode}))
                completed = run(push)
                if accepted:
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn(f"GATE-PASSED mode={mode}", completed.stdout)
                else:
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "--push requires the published sglang_fl wheel",
                        completed.stderr,
                    )
            # a context without the plugin provenance is rejected outright
            provenance.unlink()
            completed = run("0")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("plugin-wheel provenance missing", completed.stderr)

    def test_head_and_label_are_reasserted_after_the_build(self):
        wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        self.assertIn("HEAD moved during the build", wrapper)
        self.assertIn("source-commit label mismatch", wrapper)
        self.assertIn("org.flagos.ci.xgrammar-wheel-sha256", wrapper)
        self.assertIn('--xgrammar-wheel-sha256 "$xgrammar_wheel_sha256"', wrapper)
        self.assertIn('--xgrammar-provenance "$xgrammar_provenance"', wrapper)
        verifier = (SCRIPTS / "verify_image.sh").read_text()
        # The wheels are immutable artifacts: their provenance must belong to
        # the image's history, not be equal to the image's own commit (a
        # published wheel is always derived earlier).
        self.assertEqual(
            verifier.count('provenance_lineage.py" "$sglang_provenance"'), 1
        )
        self.assertEqual(
            verifier.count('provenance_lineage.py" "$xgrammar_provenance"'), 1
        )
        self.assertNotIn('provenance["source_commit"] == sys.argv[4]', verifier)
        self.assertIn('!= "<no value>"', verifier)

    def test_the_plugin_wheel_is_a_pinned_artifact_of_the_image(self):
        wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        for pin in (
            '--label "org.flagos.ci.sglang-fl-wheel-sha256=$plugin_wheel_sha256"',
            '--sglang-fl-wheel "$context/artifacts/$PLUGIN_WHEEL_NAME"',
            '--sglang-fl-wheel-sha256 "$plugin_wheel_sha256"',
            '--sglang-fl-provenance "$plugin_provenance"',
        ):
            self.assertIn(pin, wrapper)
        staging = (DOCKER / "stage_wheels.sh").read_text()
        # the release input is the published wheel, pinned by its digest ...
        self.assertIn("SGLANG_FL_WHEEL", staging)
        self.assertIn("SGLANG_FL_SHA256", staging)
        self.assertIn("--mode published --wheel", staging)
        # ... and the checkout build stays a development candidate
        self.assertIn("--mode built-from-checkout --repo-root", staging)


class PluginWheelAcceptanceTests(unittest.TestCase):
    """Drive check_plugin_wheel.py against real and tampered wheels."""

    @classmethod
    def setUpClass(cls):
        # removed again in tearDownClass: the acceptance stage must not
        # outlive the test run in $TMPDIR
        cls._stage = tempfile.TemporaryDirectory(prefix="hygon-2w-acceptance.")
        cls.stage = cls._stage.name
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

    @classmethod
    def tearDownClass(cls):
        cls._stage.cleanup()

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
            Path(cls.tmp) / "sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl"
        )
        with zipfile.ZipFile(cls.sglang_wheel, "w") as archive:
            archive.writestr(
                "sglang-0.5.18+flagos.hcu1.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.18+flagos.hcu1\n",
            )
        cls.plugin_wheel = Path(cls.tmp) / "sglang_fl-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(cls.plugin_wheel, "w") as archive:
            archive.writestr(
                "sglang_fl-0.1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: sglang_fl\nVersion: 0.1.0\n",
            )
        cls.xgrammar_wheel = (
            Path(cls.tmp)
            / "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        with zipfile.ZipFile(cls.xgrammar_wheel, "w") as archive:
            archive.writestr(
                "xgrammar-0.2.1+flagos.hcu1.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: xgrammar\nVersion: 0.2.1+flagos.hcu1\n",
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
            f" + xgrammar @ file://{self.xgrammar_wheel}\n"
            " + datasets==5.2.0\n"
            " + torch==2.9.0\n"
        )

    def test_parses_multiline_stderr_plan(self):
        planned = self.validate_uv_plan.parse_plan(self.real_output())
        self.assertEqual(planned["datasets"], "5.2.0")
        self.assertEqual(planned["xgrammar"], "0.2.1+flagos.hcu1")
        self.assertEqual(planned["sglang"], "0.5.18+flagos.hcu1")
        self.assertEqual(planned["sglang-fl"], "0.1.0")
        self.assertNotIn("triton", planned)
        self.validate_uv_plan.validate_platform_plan(planned)

    def test_original_xgrammar_is_rejected(self):
        planned = self.validate_uv_plan.parse_plan(self.real_output())
        planned["xgrammar"] = "0.2.1"
        with self.assertRaises(SystemExit):
            self.validate_uv_plan.validate_platform_plan(planned)

    def test_standalone_triton_is_rejected(self):
        planned = self.validate_uv_plan.parse_plan(self.real_output())
        planned["triton"] = "3.5.0"
        with self.assertRaises(SystemExit):
            self.validate_uv_plan.validate_platform_plan(planned)

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
            "artifacts/sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl",
            "artifacts/sglang_fl-0.1.0-py3-none-any.whl",
            "artifacts/sglang_fl-0.1.0-py3-none-any.provenance.json",
            "artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "artifacts/xgrammar-0.2.1+flagos.hcu1-cp310-cp310-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.provenance.json",
            "artifacts/runtime-0.5.18.excludes.txt",
            "runtime-0.5.18.containerfile",
            "docker/hygon/scripts/validate_uv_plan.py",
        ):
            self.assertTrue((self.context / relative).is_file(), f"missing {relative}")
        # the ORIGINAL wheel and the patch are derive inputs only; neither
        # may travel with the image build context
        self.assertFalse(
            (self.context / "docker/hygon/sglang-v0.5.18-hcu.patch").exists(),
            "HCU patch entered the build context",
        )
        self.assertFalse(
            any((self.context / "artifacts").glob("sglang-0.5.18+flagos-cp310*")),
            "the original (non-derived) SGLang wheel was staged for the image",
        )
        self.assertFalse(
            any((self.context / "artifacts").glob("xgrammar-0.2.1-cp310*")),
            "the original xgrammar wheel was staged for the image",
        )

    def test_plugin_wheel_provenance_binds_the_development_build(self):
        wheel = self.context / "artifacts/sglang_fl-0.1.0-py3-none-any.whl"
        provenance = json.loads(
            (
                self.context / "artifacts/sglang_fl-0.1.0-py3-none-any.provenance.json"
            ).read_text()
        )
        # a staging without SGLANG_FL_WHEEL is a development candidate
        self.assertEqual(provenance["mode"], "built-from-checkout")
        self.assertEqual(provenance["distribution"], "sglang_fl")
        self.assertEqual(
            provenance["sha256"], hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(provenance["source_commit"], head)
        self.assertIn("mode=built-from-checkout", self.output)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.context, ignore_errors=True)


class ReviewerCommentContractTests(unittest.TestCase):
    """One block per PR #180 review comment this branch answers.

    Each test pins the repository state that answers the comment, so a
    later change cannot silently bring the rejected design back.
    """

    def test_r4627164234_slot_default_stays_flagtree_and_the_ab_is_prepared(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn("ENV PYTHONPATH=", containerfile)
        self.assertNotIn("ENV PYTHONPATH=/opt/triton", containerfile)
        self.assertTrue((DOCKER / "triton_slot_ab.sh").is_file())
        ab = (DOCKER / "triton_slot_ab.sh").read_text()
        self.assertIn('SLOT_A="/opt/flagtree"', ab)
        self.assertIn('SLOT_B="/opt/triton"', ab)
        self.assertIn("verify_triton_slot_pair.py", ab)
        # the A/B never pushes and says the hardware matrix is still owed
        self.assertNotIn("docker push", ab)
        self.assertIn("NOT run here", ab)

    def test_r4627367002_clang_wrapper_is_gone_and_flagtrees_own_path_is_probed(self):
        for name in (
            "runtime-0.5.18.containerfile",
            "build_two_wheel.sh",
            "stage_wheels.sh",
        ):
            text = (DOCKER / name).read_text()
            self.assertNotIn("hcu-clang", text, name)
            self.assertNotIn("TRITON_HIP_CLANG_PATH", text, name)
        self.assertFalse((DOCKER / "hcu-clang").exists())
        # the acceptance drives FlagTree's own stage functions, not a
        # hand-written clang command line
        verify = (DOCKER / "scripts" / "verify_image.sh").read_text()
        self.assertIn("/scripts/flagtree_compile_probe.py", verify)
        probe = (DOCKER / "scripts" / "flagtree_compile_probe.py").read_text()
        self.assertIn("HIPBackend.make_amdgcn(", probe)
        self.assertIn("HIPBackend.make_hsaco(", probe)
        # the owner-side fix candidate is kept, and is never applied here
        upstream = (
            DOCKER
            / "upstream"
            / ("flagtree-0.7.0+hcu3.6-explicit-device-lib-dir.patch")
        )
        self.assertTrue(upstream.is_file())
        self.assertIn("never applied here", upstream.read_text()[:200])
        for name in (
            "runtime-0.5.18.containerfile",
            "build_two_wheel.sh",
            "stage_wheels.sh",
        ):
            self.assertNotIn("upstream/", (DOCKER / name).read_text())

    def test_r4627443492_wheels_are_pinned_artifacts_with_a_publication_manifest(self):
        staging = (DOCKER / "stage_wheels.sh").read_text()
        self.assertIn(
            'SGLANG_WHEEL_URL="https://resource.flagos.net/repository/flagos-pypi-hygon/',
            staging,
        )
        # one version, one content: the derived wheels must hash to pins
        self.assertIn(
            'SGLANG_DERIVED_SHA256="edcae8c0678290ec68b83d8ec38537d0ef7852b3f8f183984126bba9e8639b5f"',
            staging,
        )
        self.assertIn(
            'XGRAMMAR_DERIVED_SHA256="c2deb2853ac98cbaf63ecfa7f55dd00c54e25c9c43a34e9edb234f84e316d1a2"',
            staging,
        )
        self.assertEqual(staging.count('require_sha256 "$derived_wheel"'), 1)
        self.assertEqual(staging.count('require_sha256 "$derived_xgrammar_wheel"'), 1)
        # an already-published wheel is staged by download-then-verify
        self.assertIn("stage_pre_derived", staging)
        self.assertIn("publication_manifest.py", staging)
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn("COPY artifacts/", containerfile)

    def test_r4627462330_the_patch_is_folded_into_the_wheel_not_the_image(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        for banned in ("git apply", "hcu.patch", "patch -p"):
            self.assertNotIn(banned, containerfile)
        self.assertTrue(
            (REPO_ROOT / "tools/wheels/hygon/sglang-v0.5.18-hcu.patch").is_file()
        )
        self.assertFalse((DOCKER / "sglang-v0.5.18-hcu.patch").exists())

    def test_r4627477929_the_stubs_live_in_the_plugin_wheel(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        for banned in ("/tmp/stubs", "stubs/", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
            self.assertNotIn(banned, containerfile)
        self.assertFalse((DOCKER / "stubs").exists())
        vendor = REPO_ROOT / "sglang_fl/dispatch/backends/vendor/hygon"
        self.assertTrue((vendor / "early.py").is_file())
        guard = (vendor / "import_guard.py").read_text()
        self.assertIn("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", guard)
        # the version-check switch is set only while a placeholder is active
        self.assertIn('if root == "sgl_kernel":', guard)

    def test_r4627488390_runtime_owns_the_vendor_environment(self):
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn("ENV LD_LIBRARY_PATH=", containerfile)
        self.assertNotIn("ENV ROCM_PATH=", containerfile)
        self.assertNotIn("/opt/rocm", containerfile)

    def test_r4627512386_the_excludes_removal_is_gated(self):
        # transitional state: the exclusion and its exact exception remain
        # until the real-index gate and the built image prove them redundant
        self.assertTrue((DOCKER / "runtime-0.5.18.excludes.txt").is_file())
        exceptions = (SCRIPTS / "pip_check_exceptions.py").read_text()
        self.assertIn('("xgrammar", "triton")', exceptions)
        gate = (DOCKER / "no_excludes_gate.sh").read_text()
        self.assertIn("plan B", gate)
        self.assertIn("identical", gate)
        self.assertIn("validate_uv_plan.py", gate)
        self.assertNotIn("docker push", gate)
        self.assertNotIn("docker build", gate)

    def test_r4627527485_the_datasets_pin_is_wheel_metadata(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn('"datasets==5.2.0"', pyproject)
        self.assertFalse((DOCKER / "requirements.txt").exists())
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn("-r /tmp", containerfile)
        self.assertNotIn("requirements.txt", containerfile)


class TritonSlotContractTests(unittest.TestCase):
    """Each Runtime compiler slot is verified against its own contract."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import triton_slot

        cls.slot = triton_slot

    def test_two_known_slots_and_the_production_default(self):
        slot = self.slot
        self.assertEqual(sorted(slot.SLOT_CONTRACTS), ["/opt/flagtree", "/opt/triton"])
        self.assertEqual(slot.PRODUCTION_SLOT, "/opt/flagtree")
        self.assertFalse(slot.SLOT_CONTRACTS["/opt/flagtree"]["pip_check_sees_triton"])
        self.assertTrue(slot.SLOT_CONTRACTS["/opt/triton"]["pip_check_sees_triton"])
        self.assertEqual(
            slot.dist_info_version(
                slot.SLOT_CONTRACTS["/opt/triton"]["triton_dist_info"]
            ),
            "3.5.1+das.opt1.dtk2604.torch290",
        )
        self.assertIsNone(
            slot.dist_info_version(
                slot.SLOT_CONTRACTS["/opt/flagtree"]["triton_dist_info"]
            )
        )

    def _tree(self, root, version, dist_info, backends):
        triton = Path(root) / "triton"
        (triton / "backends").mkdir(parents=True)
        (triton / "__init__.py").write_text(f"__version__ = '{version}'\n")
        for name in backends:
            (triton / "backends" / name).mkdir()
        if dist_info:
            (Path(root) / dist_info).mkdir()
        return str(root)

    def test_inspect_slot_reads_the_facts_from_disk(self):
        import tempfile

        slot = self.slot
        with tempfile.TemporaryDirectory() as tmp:
            flagtree = self._tree(Path(tmp) / "flagtree", "3.6.0", None, ["hcu"])
            vendor = self._tree(
                Path(tmp) / "vendor",
                "3.5.1",
                "triton-3.5.1+das.opt1.dtk2604.torch290.dist-info",
                ["amd", "nvidia"],
            )
            self.assertEqual(
                slot.inspect_slot(flagtree),
                {
                    "module_version": "3.6.0",
                    "triton_dist_info": [],
                    "backends": ["hcu"],
                },
            )
            observed = slot.inspect_slot(vendor)
            self.assertEqual(observed["module_version"], "3.5.1")
            self.assertEqual(observed["backends"], ["amd", "nvidia"])
            self.assertEqual(
                observed["triton_dist_info"],
                ["triton-3.5.1+das.opt1.dtk2604.torch290.dist-info"],
            )

    def test_unknown_slot_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.slot.contract_of("/opt/elsewhere")
        with self.assertRaises(AssertionError):
            self.slot.dist_info_version("garbage")

    def test_contracts_match_the_contract_table(self):
        # the table is the single source: every contract is self-consistent
        for root, contract in self.slot.SLOT_CONTRACTS.items():
            sees = contract["pip_check_sees_triton"]
            self.assertEqual(sees, contract["triton_dist_info"] is not None, root)
            if sees:
                version = self.slot.dist_info_version(contract["triton_dist_info"])
                self.assertTrue(version.startswith(contract["module_version"]))

    def test_verifier_is_parametrized_by_the_declared_slot(self):
        build = (DOCKER / "build_two_wheel.sh").read_text()
        self.assertIn('--label "org.flagos.ci.triton-slot=$TRITON_SLOT"', build)
        self.assertIn('TRITON_SLOT="/opt/flagtree"', build)
        verify = (DOCKER / "scripts" / "verify_image.sh").read_text()
        self.assertIn("org.flagos.ci.triton-slot", verify)
        self.assertIn('--selected-slot "$TRITON_SLOT"', verify)
        self.assertIn('-e TRITON_SLOT="$TRITON_SLOT"', verify)
        self.assertIn('test "${PYTHONPATH:-}" = "$TRITON_SLOT"', verify)
        self.assertIn("production image overrides Runtime-owned PYTHONPATH", verify)
        # no phase may still hard-code the production slot
        phase6 = verify[verify.index("=== [6/8]") : verify.index("=== [7/8]")]
        self.assertNotIn("/opt/flagtree", phase6)
        # the slot contract helper must be reachable inside the phase-6 container
        self.assertIn('--volume "$scripts":/scripts:ro', phase6)
        collector = (SCRIPTS / "collect_runtime_reference.py").read_text()
        self.assertIn("--selected-slot", collector)
        self.assertIn("assert_slot_root", collector)

    def test_pair_verifier_checks_layers_env_labels_and_trees(self):
        pair = (SCRIPTS / "verify_triton_slot_pair.py").read_text()
        for check in (
            'a["RootFS"]["Layers"] == b["RootFS"]["Layers"]',
            'differing == ["PYTHONPATH"]',
            "label_diff == [SLOT_LABEL]",
            "rest(a) == rest(b)",
            'TREES = ("/opt/flagtree", "/opt/triton", "<site-packages>")',
        ):
            self.assertIn(check, pair)


class UvInstallContractTests(unittest.TestCase):
    """The containerfile's install arguments, extracted once for other gates."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import uv_install_contract

        cls.module = uv_install_contract
        cls.args = uv_install_contract.install_arguments(
            (DOCKER / "runtime-0.5.18.containerfile").read_text()
        )

    def test_matches_the_release_recipe(self):
        containerfile = RELEASE_RECIPE.read_text()
        import shlex

        # uv_args keeps continuation lines whole; the extractor emits tokens
        extracted = shlex.split(
            " ".join(uv_args(containerfile.split("/usr/local/bin/uv pip install")[1]))
        )
        arg_defaults = {
            match.group(1): match.group(2)
            for match in re.finditer(r"^ARG (\w+)=(\S+)$", containerfile, re.MULTILINE)
        }
        extracted = [
            arg_defaults.get(token.lstrip("$"), token)
            if token.startswith("$")
            else token
            for token in extracted
        ]
        self.assertEqual(self.args, extracted)

    def test_run_options_reproduce_the_recipe_mounts_and_network(self):
        contract = self.module
        self.assertEqual(
            contract.run_options(RELEASE_RECIPE.read_text(), Path("/ctx")),
            ["--volume", "/ctx/artifacts:/tmp/artifacts:ro"],
        )
        self.assertEqual(
            contract.run_options(OFFLINE_RECIPE.read_text(), Path("/ctx")),
            [
                "--network",
                "none",
                "--volume",
                "/ctx/artifacts:/tmp/artifacts:ro",
                "--volume",
                "/ctx/wheelhouse:/wheelhouse:ro",
            ],
        )
        # a bind mount needs the context it resolves against
        with self.assertRaises(SystemExit):
            contract.run_options(OFFLINE_RECIPE.read_text())
        with self.assertRaises(SystemExit):
            contract.run_options(
                "RUN --mount=type=cache,target=/x \\\n    /usr/local/bin/uv pip install\n",
                Path("/ctx"),
            )

    def test_stops_at_the_end_of_the_run_instruction(self):
        self.assertEqual(
            self.args[-1], "/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[hygon]"
        )
        self.assertEqual(self.args[0], "--python")
        self.assertNotIn("\\", self.args)
        self.assertFalse(any(a.startswith("#") or a == "ENV" for a in self.args))
        self.assertIn("--excludes", self.args)

    def test_index_args_are_substituted(self):
        self.assertIn(
            "https://resource.flagos.net/repository/flagos-pypi-hygon/simple",
            self.args,
        )
        self.assertNotIn("$FLAGOS_INDEX", " ".join(self.args))


class PublicationManifestTests(unittest.TestCase):
    """The publication manifest names every staged artifact with its pin."""

    @classmethod
    def setUpClass(cls):
        import json
        import tempfile
        import zipfile

        sys.path.insert(0, str(REPO_ROOT / "tools" / "wheels" / "hygon"))
        import publication_manifest

        cls.module = publication_manifest
        cls._tmp = tempfile.TemporaryDirectory(prefix="pub-manifest-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = cls._tmp.name
        artifacts = Path(cls.tmp) / "artifacts"
        artifacts.mkdir()
        cls.wheels = {}
        for name, dist, version in (
            (
                "sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl",
                "sglang",
                "0.5.18+flagos.hcu1",
            ),
            (
                "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64."
                "manylinux_2_28_x86_64.whl",
                "xgrammar",
                "0.2.1+flagos.hcu1",
            ),
            ("sglang_fl-0.1.0-py3-none-any.whl", "sglang_fl", "0.1.0"),
        ):
            path = artifacts / name
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    f"{dist}-{version}.dist-info/METADATA",
                    f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n",
                )
            cls.wheels[dist] = path
        (
            artifacts
            / ("sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.provenance.json")
        ).write_text(
            json.dumps(
                {
                    "input_wheel": "sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl",
                    "input_sha256": "1fe8a560" + "0" * 56,
                    "patch": "sglang-v0.5.18-hcu.patch",
                    "patch_sha256": "4327e6f7" + "0" * 56,
                    "source_commit": "0" * 40,
                }
            )
        )
        (
            artifacts
            / (
                "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64."
                "manylinux_2_28_x86_64.provenance.json"
            )
        ).write_text(
            json.dumps(
                {
                    "input_wheel": "xgrammar-0.2.1-cp310-cp310-manylinux_2_27_x86_64."
                    "manylinux_2_28_x86_64.whl",
                    "input_sha256": "5b80b806" + "0" * 56,
                    "dropped_requires_dist": [
                        'triton; platform_system == "Linux" and platform_machine == "x86_64"'
                    ],
                    "source_commit": "0" * 40,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_manifest_lists_every_artifact_with_its_nexus_target(self):
        manifest = self.module.build(Path(self.tmp))
        self.assertEqual(
            sorted(a["distribution"] for a in manifest["artifacts"]),
            ["sglang", "sglang_fl", "xgrammar"],
        )
        by_name = {a["distribution"]: a for a in manifest["artifacts"]}
        sglang = by_name["sglang"]
        self.assertEqual(
            sglang["sha256"], self.module.sha256_file(self.wheels["sglang"])
        )
        self.assertEqual(
            sglang["nexus"]["path"],
            "packages/sglang/0.5.18+flagos.hcu1/"
            "sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64.whl",
        )
        self.assertEqual(sglang["derived"]["kind"], "derived")
        self.assertEqual(sglang["derived"]["input_sha256"], "1fe8a560" + "0" * 56)
        self.assertEqual(by_name["xgrammar"]["derived"]["kind"], "metadata-corrected")
        self.assertIn("built_from", by_name["sglang_fl"])
        self.assertNotIn("derived", by_name["sglang_fl"])

    def test_nothing_is_called_published(self):
        manifest = self.module.build(Path(self.tmp))
        for artifact in manifest["artifacts"]:
            self.assertIn("LOCAL DEVELOPMENT CANDIDATE", artifact["status"])
        self.assertIn("Nothing here is published", manifest["status"])
        self.assertIn("hcu2", " ".join(manifest["policy"]))

    def test_staging_can_consume_an_already_derived_wheel_by_pin(self):
        staging = (DOCKER / "stage_wheels.sh").read_text()
        self.assertIn("SGLANG_DERIVED_WHEEL", staging)
        self.assertIn("XGRAMMAR_DERIVED_WHEEL", staging)
        # the pre-derived path verifies the same pin as a fresh derive
        self.assertEqual(staging.count("$SGLANG_DERIVED_SHA256"), 2)
        self.assertEqual(staging.count("$XGRAMMAR_DERIVED_SHA256"), 2)

    def test_the_plugin_wheel_reports_how_it_was_staged(self):
        provenance = self.wheels["sglang_fl"].with_suffix(".provenance.json")
        by_name = {
            entry["distribution"]: entry
            for entry in self.module.build(Path(self.tmp))["artifacts"]
        }
        # no provenance next to the wheel: nothing is known beyond the name
        self.assertIn("checkout", by_name["sglang_fl"]["built_from"])
        self.assertNotIn("provenance", by_name["sglang_fl"])
        for mode, status, origin in (
            (
                "built-from-checkout",
                "LOCAL DEVELOPMENT CANDIDATE",
                "SOURCE_DATE_EPOCH",
            ),
            ("published", "PUBLISHED sglang_fl ARTIFACT", "SGLANG_FL_WHEEL"),
        ):
            provenance.write_text(
                json.dumps(
                    {"mode": mode, "source_commit": "2" * 40, "sha256": "3" * 64}
                )
            )
            entry = {
                item["distribution"]: item
                for item in self.module.build(Path(self.tmp))["artifacts"]
            }["sglang_fl"]
            self.assertIn(status, entry["status"])
            self.assertIn(origin, entry["built_from"])
            self.assertEqual(
                entry["provenance"],
                {"mode": mode, "source_commit": "2" * 40, "sha256": "3" * 64},
            )


class ProvenanceLineageTests(unittest.TestCase):
    """A wheel may be derived (or published) before the image installing it.

    The content binding is the SHA256 label plus the per-file manifest; the
    commit recorded in the provenance only has to belong to the image's
    history, so a pre-derived or published wheel passes the strict gate.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "provenance_lineage_test", SCRIPTS / "provenance_lineage.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.tmp = tempfile.TemporaryDirectory(prefix="provenance-lineage-test.")
        cls.provenance_dir = Path(cls.tmp.name) / "provenance"
        cls.provenance_dir.mkdir()
        # a throwaway repository: first -> second on the main line, and one
        # commit on a side branch that is not part of second's history
        cls.repo = Path(cls.tmp.name) / "repo"
        cls.repo.mkdir()
        cls.git("init", "-q")
        for name, value in (
            ("user.name", "test"),
            ("user.email", "test@example.com"),
            ("commit.gpgsign", "false"),
        ):
            cls.git("config", name, value)
        cls.git("commit", "-q", "--allow-empty", "-m", "first")
        cls.first = cls.git("rev-parse", "HEAD")
        cls.git("commit", "-q", "--allow-empty", "-m", "second")
        cls.second = cls.git("rev-parse", "HEAD")
        cls.git("checkout", "-q", "-b", "side", cls.first)
        cls.git("commit", "-q", "--allow-empty", "-m", "side")
        cls.side = cls.git("rev-parse", "HEAD")
        cls.git("checkout", "-q", "-")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def git(cls, *arguments):
        return subprocess.run(
            ["git", "-C", str(cls.repo), *arguments],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def provenance(self, source_commit):
        path = self.provenance_dir / f"{source_commit}.json"
        path.write_text(json.dumps({"source_commit": source_commit}))
        return path

    def violation(self, source_commit, image_commit):
        return self.module.lineage_violation(
            self.provenance(source_commit), image_commit, self.repo
        )

    def test_the_image_commit_and_earlier_commits_are_accepted(self):
        self.assertIsNone(self.violation(self.second, self.second))
        self.assertIsNone(self.violation(self.first, self.second))

    def test_a_commit_outside_the_image_history_is_rejected(self):
        self.assertIn("not an ancestor", self.violation(self.side, self.second))

    def test_an_unknown_or_malformed_commit_is_rejected(self):
        self.assertIn("not a commit of", self.violation("0" * 40, self.second))
        self.assertIn("no full source_commit", self.violation("HEAD", self.second))
        self.assertIn("not a full commit id", self.violation(self.first, "main"))

    def test_an_unreadable_provenance_is_rejected(self):
        broken = self.provenance_dir / "broken.json"
        broken.write_text("{not json")
        self.assertIn(
            "cannot be read",
            self.module.lineage_violation(broken, self.second, self.repo),
        )
        broken.write_text("[]")  # readable, but not a provenance object
        self.assertIn(
            "no full source_commit",
            self.module.lineage_violation(broken, self.second, self.repo),
        )
        self.assertIn(
            "cannot be read",
            self.module.lineage_violation(
                self.provenance_dir / "absent.json", self.second, self.repo
            ),
        )

    def test_the_cli_fails_loudly(self):
        script = SCRIPTS / "provenance_lineage.py"
        command = [sys.executable, str(script)]
        accepted = subprocess.run(
            [
                *command,
                str(self.provenance(self.first)),
                self.second,
                "--repo",
                str(self.repo),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        rejected = subprocess.run(
            [
                *command,
                str(self.provenance(self.side)),
                self.second,
                "--repo",
                str(self.repo),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("FATAL: provenance was derived at", rejected.stderr)

    @unittest.skipUnless(
        (REPO_ROOT / ".git").exists(), "needs the git checkout this test runs in"
    )
    def test_this_checkout_satisfies_the_rule_for_its_own_head(self):
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertIsNone(
            self.module.lineage_violation(self.provenance(head), head, REPO_ROOT)
        )

    def run_strict_blocks(self, provenance_commit, image_commit):
        """Run verify_image.sh's two strict provenance blocks as written.

        The blocks are cut out of the real script (up to their in-image
        runs) and executed with a fake docker that answers the label
        queries, so the shell wiring around the lineage check is exercised,
        not just its text.
        """
        verifier = (SCRIPTS / "verify_image.sh").read_text()
        blocks = []
        for start_mark, end_mark in (
            (
                'labeled_source_commit="$(docker image inspect',
                "run -e DERIVED_SGLANG_VERSION=",
            ),
            (
                'labeled_xgrammar_sha256="$(docker image inspect',
                "run -e DERIVED_XGRAMMAR_VERSION=",
            ),
        ):
            start = verifier.index(start_mark)
            # each block opens an `if` whose body ends at the in-image run
            blocks.append(verifier[start : verifier.index(end_mark, start)] + "fi\n")
        sha = "ab" * 32
        common = {
            "source_commit": provenance_commit,
            "derived_sha256": sha,
        }
        sglang = self.provenance_dir / "sglang.json"
        xgrammar = self.provenance_dir / "xgrammar.json"
        sglang.write_text(
            json.dumps(
                {
                    **common,
                    "derived_version": "0.5.18+flagos.hcu1",
                    "patched_files_manifest": {"sglang/a.py": "00"},
                }
            )
        )
        xgrammar.write_text(
            json.dumps(
                {
                    **common,
                    "derived_version": "0.2.1+flagos.hcu1",
                    "drop_distribution": "triton",
                    "dropped_requires_dist": [
                        'triton; platform_system == "Linux" and platform_machine == "x86_64"'
                    ],
                    "bound_files_manifest": {"xgrammar/a.py": "00"},
                }
            )
        )
        script = self.provenance_dir / "strict-blocks.sh"
        script.write_text(
            "\n".join(
                [
                    "set -euo pipefail",
                    "image=fake-image",
                    'scripts="$1"; repo_root="$2"; sglang_provenance="$3"; xgrammar_provenance="$4"',
                    f'expected_sglang_sha256="{sha}"; expected_xgrammar_sha256="{sha}"',
                    'DERIVED_SGLANG_VERSION="0.5.18+flagos.hcu1"',
                    'DERIVED_XGRAMMAR_VERSION="0.2.1+flagos.hcu1"',
                    "docker() {",
                    '    case "$*" in',
                    '        *"org.flagos.ci.source-commit"*) printf \'%s\\n\' "$FAKE_IMAGE_COMMIT" ;;',
                    '        *"wheel-sha256"*) printf \'%s\\n\' "' + sha + '" ;;',
                    "        *) return 1 ;;",
                    "    esac",
                    "}",
                    *blocks,
                    "echo BLOCKS-DONE",
                ]
            )
        )
        return subprocess.run(
            [
                "bash",
                str(script),
                str(SCRIPTS),
                str(self.repo),
                str(sglang),
                str(xgrammar),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "FAKE_IMAGE_COMMIT": image_commit},
        )

    def test_the_strict_gate_accepts_a_wheel_derived_before_the_image(self):
        # both wheels' blocks run to their end for a same-commit provenance
        # and for one derived one commit earlier (the published-wheel case)
        for derived_at in (self.second, self.first):
            completed = self.run_strict_blocks(derived_at, self.second)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("BLOCKS-DONE", completed.stdout)

    def test_the_strict_gate_rejects_a_wheel_from_another_history(self):
        completed = self.run_strict_blocks(self.side, self.second)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("FATAL: provenance was derived at", completed.stderr)
        self.assertNotIn("BLOCKS-DONE", completed.stdout)

    def run_plugin_block(self, label, pin, provenance_sha, provenance_commit):
        """Run verify_image.sh's strict plugin-wheel block as written."""
        verifier = (SCRIPTS / "verify_image.sh").read_text()
        start = verifier.index('labeled_plugin_sha256="$(docker image inspect')
        end = verifier.index('run --volume "$sglang_fl_wheel"', start)
        wheel = self.provenance_dir / "plugin.whl"
        provenance = self.provenance_dir / "plugin.json"
        provenance.write_text(
            json.dumps(
                {
                    "distribution": "sglang_fl",
                    "sha256": provenance_sha,
                    "source_commit": provenance_commit,
                }
            )
        )
        script = self.provenance_dir / "plugin-block.sh"
        script.write_text(
            "\n".join(
                [
                    "set -euo pipefail",
                    "image=fake-image",
                    'scripts="$1"; repo_root="$2"; sglang_fl_wheel="$3"; '
                    'sglang_fl_provenance="$4"',
                    'expected_sglang_fl_sha256="$5"; labeled_source_commit="$6"',
                    "docker() {",
                    '    case "$*" in',
                    "        *org.flagos.ci.sglang-fl-wheel-sha256*) "
                    "printf '%s\\n' \"$FAKE_LABEL\" ;;",
                    "        *) return 1 ;;",
                    "    esac",
                    "}",
                    verifier[start:end] + "fi\n",
                    "echo PLUGIN-BLOCK-DONE",
                ]
            )
        )
        return subprocess.run(
            [
                "bash",
                str(script),
                str(SCRIPTS),
                str(self.repo),
                str(wheel),
                str(provenance),
                pin,
                self.second,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "FAKE_LABEL": label},
        )

    def staged_plugin_wheel_sha(self) -> str:
        """Write the staged wheel the block sees and return its digest."""
        wheel = self.provenance_dir / "plugin.whl"
        wheel.write_bytes(b"the staged sglang_fl wheel")
        return hashlib.sha256(wheel.read_bytes()).hexdigest()

    def test_the_plugin_wheel_block_accepts_the_pinned_artifact(self):
        wheel_sha = self.staged_plugin_wheel_sha()
        # label, pin, provenance and staged file agree; the wheel may have
        # been built at the image commit or earlier
        for commit in (self.second, self.first):
            completed = self.run_plugin_block(wheel_sha, wheel_sha, wheel_sha, commit)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PLUGIN-BLOCK-DONE", completed.stdout)

    def test_the_plugin_wheel_block_rejects_every_other_combination(self):
        wheel_sha = self.staged_plugin_wheel_sha()
        other = "0" * 64
        for name, arguments, message in (
            ("label", dict(label=other), "!= expected"),
            (
                "staged wheel",
                dict(label=other, pin=other, provenance_sha=other),
                "does not hash to",
            ),
            ("provenance", dict(provenance_sha=other), "AssertionError"),
            ("lineage", dict(provenance_commit=self.side), "FATAL: provenance"),
        ):
            with self.subTest(name):
                completed = self.run_plugin_block(
                    label=arguments.get("label", wheel_sha),
                    pin=arguments.get("pin", wheel_sha),
                    provenance_sha=arguments.get("provenance_sha", wheel_sha),
                    provenance_commit=arguments.get("provenance_commit", self.first),
                )
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn(message, completed.stderr)
                self.assertNotIn("PLUGIN-BLOCK-DONE", completed.stdout)


def synthetic_plugin_wheel(path: Path, files: int = 120) -> Path:
    """A minimal sglang_fl wheel with enough members for the content gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "sglang_fl-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: sglang_fl\nVersion: 0.1.0\n",
        )
        archive.writestr(
            "sglang_fl-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "sglang_fl-0.1.0.dist-info/RECORD", "sglang_fl-0.1.0.dist-info/METADATA,,\n"
        )
        archive.writestr("sglang_fl/__init__.py", '"""synthetic"""\n')
        for number in range(files):
            archive.writestr(f"sglang_fl/mod_{number:03d}.py", f"value = {number}\n")
    return path


class PluginWheelArtifactTests(unittest.TestCase):
    """sglang_fl is an artifact with the same contract as the platform wheels.

    A development candidate is built from the checkout with reproducible
    timestamps; a release input is the published wheel, pinned by its SHA256
    and accompanied by the provenance of the commit it was built from.
    """

    WHEEL = "sglang_fl-0.1.0-py3-none-any.whl"
    PROVENANCE = "sglang_fl-0.1.0-py3-none-any.provenance.json"

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import stage_plugin_wheel

        cls.module = stage_plugin_wheel
        cls._tmp = tempfile.TemporaryDirectory(prefix="plugin-artifact-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = Path(cls._tmp.name)
        cls.commit = "1" * 40
        cls.published = cls.tmp / "published"
        cls.wheel = synthetic_plugin_wheel(cls.published / cls.WHEEL)
        cls.sha = cls.module.sha256_file(cls.wheel)
        (cls.published / cls.PROVENANCE).write_text(
            json.dumps({"source_commit": cls.commit, "sha256": cls.sha})
        )

    def stage(self, sha=None, wheel=None) -> Path:
        dest = Path(tempfile.mkdtemp(prefix="stage.", dir=self.tmp))
        return self.module.stage_published(wheel or self.wheel, sha or self.sha, dest)

    def test_the_published_wheel_is_staged_with_its_pin_and_lineage(self):
        staged = self.stage()
        self.assertEqual(self.module.sha256_file(staged), self.sha)
        provenance = json.loads(staged.with_suffix(".provenance.json").read_text())
        self.assertEqual(
            provenance,
            {
                "distribution": "sglang_fl",
                "version": "0.1.0",
                "mode": "published",
                "sha256": self.sha,
                "source_commit": self.commit,
            },
        )

    def test_content_other_than_the_pin_is_rejected(self):
        with self.assertRaisesRegex(self.module.StageError, "SHA256 mismatch"):
            self.stage(sha="0" * 64)

    def test_a_pin_that_is_not_a_digest_is_rejected(self):
        with self.assertRaisesRegex(self.module.StageError, "not a SHA256 digest"):
            self.stage(sha="latest")

    def test_another_artifact_cannot_stand_in(self):
        other = synthetic_plugin_wheel(
            self.tmp / "other" / "sglang_fl-9.9.9-py3-none-any.whl"
        )
        with self.assertRaisesRegex(
            self.module.StageError, "not the expected artifact"
        ):
            self.module.stage_published(
                other, self.module.sha256_file(other), self.tmp / "x"
            )

    def test_the_published_wheel_needs_its_provenance(self):
        orphan = self.tmp / "orphan" / self.WHEEL
        orphan.parent.mkdir()
        shutil.copy(self.wheel, orphan)
        with self.assertRaisesRegex(
            self.module.StageError, "provenance of the published wheel"
        ):
            self.stage(wheel=orphan)

    def test_the_provenance_must_vouch_for_this_wheel_and_its_commit(self):
        untrusted = self.tmp / "untrusted"
        untrusted.mkdir()
        wheel = shutil.copy(self.wheel, untrusted / self.WHEEL)
        provenance = untrusted / self.PROVENANCE
        provenance.write_text(json.dumps({"source_commit": "HEAD"}))
        with self.assertRaisesRegex(self.module.StageError, "no full source_commit"):
            self.stage(wheel=wheel)
        provenance.write_text(
            json.dumps({"source_commit": self.commit, "sha256": "0" * 64})
        )
        with self.assertRaisesRegex(self.module.StageError, "describes another wheel"):
            self.stage(wheel=wheel)
        # a provenance without a digest is fine: the pin is the trust anchor
        provenance.write_text(json.dumps({"source_commit": self.commit}))
        self.assertEqual(self.module.sha256_file(self.stage(wheel=wheel)), self.sha)

    def test_the_cli_reports_the_mode_and_fails_loudly(self):
        script = SCRIPTS / "stage_plugin_wheel.py"
        command = [
            sys.executable,
            str(script),
            "--mode",
            "published",
            "--wheel",
            str(self.wheel),
        ]
        accepted = subprocess.run(
            [*command, "--dest", str(self.tmp / "cli-ok"), "--sha256", self.sha],
            capture_output=True,
            text=True,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn(
            f"plugin-wheel: sglang_fl==0.1.0 sha256={self.sha} mode=published",
            accepted.stdout,
        )
        rejected = subprocess.run(
            [*command, "--dest", str(self.tmp / "cli-bad"), "--sha256", "0" * 64],
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("FATAL: SHA256 mismatch", rejected.stderr)
        # the pin is mandatory
        unpinned = subprocess.run(
            [*command, "--dest", str(self.tmp / "cli-unpinned")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(unpinned.returncode, 1)
        self.assertIn(
            "FATAL: --mode published needs --wheel and --sha256", unpinned.stderr
        )

    @unittest.skipUnless(
        (REPO_ROOT / ".git").exists() and importlib.util.find_spec("build"),
        "needs the git checkout and the build module",
    )
    def test_development_builds_of_one_commit_are_identical(self):
        first = self.module.build_from_checkout(REPO_ROOT, self.tmp / "build-1")
        second = self.module.build_from_checkout(REPO_ROOT, self.tmp / "build-2")
        first_sha = self.module.sha256_file(first)
        self.assertEqual(first_sha, self.module.sha256_file(second))
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        epoch = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%ct", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        provenance = json.loads(first.with_suffix(".provenance.json").read_text())
        self.assertEqual(provenance["mode"], "built-from-checkout")
        self.assertEqual(provenance["sha256"], first_sha)
        self.assertEqual(provenance["source_commit"], head)
        self.assertEqual(provenance["source_date_epoch"], int(epoch))


FAKE_DOCKER = r"""#!/bin/bash
# A fake `docker` for block-level runs of the build and acceptance scripts.
# Every answer comes from FAKE_* environment variables; any other command
# fails loudly.
set -u
if [ "$1" = manifest ] && [ "$2" = inspect ]; then
    if [ "$FAKE_REMOTE" = present ]; then exit 0; fi
    echo "$FAKE_REMOTE_MESSAGE" >&2
    exit 1
fi
if [ "$1" = image ] && [ "$2" = inspect ]; then
    shift 2
    format=""
    if [ "${1:-}" = --format ]; then format="$2"; shift 2; fi
    reference="${1:-}"
    present=1
    for known in $FAKE_PRESENT; do
        if [ "$reference" = "$known" ]; then present=0; fi
    done
    [ "$present" -eq 0 ] || exit 1
    if [ "$reference" = "$FAKE_BASE" ]; then
        identifier="$FAKE_BASE_ID"; layers="$FAKE_BASE_LAYERS"
    else
        identifier="$FAKE_IMAGE_ID"; layers="$FAKE_IMAGE_LAYERS"
    fi
    case "$format" in
        "{{.Id}}") printf "%s\n" "$identifier" ;;
        "{{range .RepoDigests}}{{println .}}{{end}}")
            [ -z "$FAKE_REPO_DIGESTS" ] || printf "%s\n" "$FAKE_REPO_DIGESTS" ;;
        "{{json .RootFS.Layers}}") printf "%s\n" "$layers" ;;
        *".Config.Labels "*)
            key="$(printf "%s" "$format" | sed -n 's/.*"\(org[^"]*\)".*/\1/p')"
            value="$(printf "%s\n" "$FAKE_LABELS" | sed -n "s/^$key=//p")"
            if [ -n "$value" ]; then printf "%s\n" "$value"; else printf "<no value>\n"; fi ;;
        "") ;;
        *) echo "fake docker: unsupported format $format" >&2; exit 97 ;;
    esac
    exit 0
fi
echo "fake docker: unsupported command $*" >&2
exit 97
"""

FAKE_GIT = r"""#!/bin/bash
# A fake `git` for block-level runs of the build script.
case "$*" in
    *rev-parse*) printf "%s\n" "$FAKE_HEAD" ;;
    *status*) printf "%s" "$FAKE_DIRTY" ;;
    *) echo "fake git: unsupported command $*" >&2; exit 97 ;;
esac
"""


def block(text, start, end):
    """The region of a script between a unique start marker and the next
    end marker after it."""
    assert text.count(start) == 1, f"start marker not unique: {start!r}"
    first = text.index(start)
    return text[first : text.index(end, first)]


def run_block(lines, fake_env, *arguments):
    """Write and run a block-level script with the fake docker and git first
    on PATH; return the completed process."""
    with tempfile.TemporaryDirectory(prefix="block-test.") as tmp:
        root = Path(tmp)
        (root / "bin").mkdir()
        for name, source in (("docker", FAKE_DOCKER), ("git", FAKE_GIT)):
            program = root / "bin" / name
            program.write_text(source)
            program.chmod(0o755)
        script = root / "block.sh"
        script.write_text("\n".join(lines) + "\necho BLOCK-DONE\n")
        return subprocess.run(
            ["bash", str(script), *(str(argument) for argument in arguments)],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
                "FAKE_REMOTE": "free",
                "FAKE_REMOTE_MESSAGE": "unknown: artifact not found",
                "FAKE_PRESENT": "",
                "FAKE_LABELS": "",
                "FAKE_BASE": "",
                "FAKE_IMAGE_ID": "",
                "FAKE_IMAGE_LAYERS": "",
                **fake_env,
            },
        )


def tiny_wheel(path, name, version, extra=None):
    """A minimal wheel: dist-info (METADATA, WHEEL, RECORD), one package
    file and whatever extra members are given."""
    dist_info = f"{name}-{version}.dist-info"
    members = {
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{name}/__init__.py": '"""synthetic"""\n',
    }
    members.update(extra or {})
    members[f"{dist_info}/RECORD"] = "".join(
        f"{member},,\n" for member in sorted(members)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(
                member, content.encode() if isinstance(content, str) else content
            )
    return path


def write_synthetic_wheelhouse(module, root, wheels):
    """A wheelhouse with a manifest the module's verify() accepts."""
    records = []
    for name, version in wheels:
        wheel = tiny_wheel(root / f"{name}-{version}-py3-none-any.whl", name, version)
        url = f"https://files.example.invalid/{wheel.name}"
        records.append(
            {
                "filename": wheel.name,
                "name": name,
                "version": version,
                "sha256": module.sha256_file(wheel),
                "size": wheel.stat().st_size,
                "source_url": url,
                "download_url": url,
                "verified": "synthetic test wheel",
            }
        )
    source = "registry.example/dev/app@sha256:" + "7" * 64
    (root / module.MANIFEST_NAME).write_text(
        json.dumps(module.build_manifest(records, source, []), indent=2, sort_keys=True)
    )
    return root


class GeneratedDryRunTests(unittest.TestCase):
    """The pre-build dry-run is generated from the recipe that is built.

    build_two_wheel.sh carries no hand-written twin of the install command:
    uv_install_contract.py reads the recipe's RUN instruction into
    /plan/args, and the generated dry-run.sh passes exactly those arguments
    to uv plus --dry-run.
    """

    @classmethod
    def setUpClass(cls):
        cls.contract = load_script("uv_install_contract")
        cls.build = (DOCKER / "build_two_wheel.sh").read_text()

    def heredoc(self):
        match = re.search(DRY_RUN_HEREDOC, self.build, re.DOTALL)
        self.assertIsNotNone(match, "build_two_wheel.sh lost the dry-run heredoc")
        return match.group(1)

    def execute(self, recipe, uv_exit=0):
        """Run the generated script with a fake uv and a fake plan validator
        under the recipe's arguments; return the process and what each got."""
        with tempfile.TemporaryDirectory(prefix="dry-run-test.") as tmp:
            root = Path(tmp)
            for relative in ("usr/local/bin", "scripts", "plan", "bin"):
                (root / relative).mkdir(parents=True)
            uv = root / "usr/local/bin/uv"
            uv.write_text(
                "#!/bin/bash\n"
                f'printf "%s\\n" "$@" > {root}/plan/uv-argv.txt\n'
                'echo "fake uv output"\n'
                f"exit {uv_exit}\n"
            )
            (root / "scripts/validate_uv_plan.py").write_text(
                "import sys\n"
                f"open({str(root / 'plan/validator-argv.txt')!r}, 'w')"
                ".write('\\n'.join(sys.argv[1:]))\n"
            )
            shim = root / "bin/python"
            shim.write_text('#!/bin/bash\nexec python3 "$@"\n')
            for program in (uv, shim):
                program.chmod(0o755)
            (root / "plan/args").write_text(
                "\n".join(self.contract.install_arguments(recipe.read_text())) + "\n"
            )
            script = (
                self.heredoc()
                .replace("/usr/local/bin/uv", str(uv))
                .replace("/scripts/", f"{root}/scripts/")
                .replace("/plan", str(root / "plan"))
            )
            (root / "dry-run.sh").write_text(script + "\n")
            completed = subprocess.run(
                ["bash", str(root / "dry-run.sh")],
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}"},
            )

            def received(name):
                path = root / "plan" / name
                return path.read_text().splitlines() if path.exists() else None

            return completed, received("uv-argv.txt"), received("validator-argv.txt")

    def test_each_recipe_dry_runs_with_its_own_arguments_plus_dry_run(self):
        for recipe in (RELEASE_RECIPE, OFFLINE_RECIPE):
            with self.subTest(recipe=recipe.name):
                completed, argv, validated = self.execute(recipe)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                expected = self.contract.install_arguments(recipe.read_text())
                # the only semantic difference from the recipe's RUN
                self.assertEqual(argv, ["pip", "install", "--dry-run", *expected])
                # the uv output and the plan dir reach the plan validator
                self.assertTrue(validated[0].endswith("/uv-dry-run.txt"), validated)
                self.assertTrue(validated[1].endswith("/plan"), validated)

    def test_a_failing_dry_run_surfaces_the_uv_output(self):
        completed, _, validated = self.execute(OFFLINE_RECIPE, uv_exit=2)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("fake uv output", completed.stderr)
        self.assertIsNone(validated)

    def test_the_wrapper_generates_the_dry_run_from_the_recipe_it_builds(self):
        build = self.build
        self.assertIn(
            'python3 "$scripts/uv_install_contract.py" "$recipe" > "$plan/args"', build
        )
        self.assertIn('--run-options --context "$context" "$recipe"', build)
        heredoc = self.heredoc()
        self.assertEqual(heredoc.count("uv pip install"), 1)
        self.assertIn('uv pip install --dry-run "${ARGS[@]}"', heredoc)
        self.assertIn("< /plan/args", heredoc)
        self.assertIn("cat /plan/uv-dry-run.txt >&2", heredoc)
        for flag in (
            "--index",
            "--excludes",
            "--constraints",
            "--no-index",
            "/tmp/artifacts",
            "sglang",
        ):
            self.assertNotIn(flag, heredoc, "a hand-written copy crept back")
        # one recipe drives the dry-run and the docker build alike
        self.assertEqual(build.count('-f "$recipe"'), 1)
        self.assertEqual(build.count('--build-arg "BASE_IMAGE=$RESOLVED_BASE"'), 1)
        # the offline candidate additionally binds its plan to the wheelhouse
        self.assertIn("wheelhouse.py bind-plan", build)
        self.assertIn("--wheelhouse /wheelhouse", build)


class OfflineRecipeTests(unittest.TestCase):
    """The offline-wheelhouse recipe differs from the release recipe in the
    resolver source only."""

    @classmethod
    def setUpClass(cls):
        cls.contract = load_script("uv_install_contract")
        cls.release = RELEASE_RECIPE.read_text()
        cls.offline = OFFLINE_RECIPE.read_text()

    @staticmethod
    def instructions(text):
        """Tokenized non-comment instructions, continuations joined."""
        pending, result = [], []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                pending.append(stripped)
                if not stripped.endswith("\\"):
                    result.append(
                        " ".join(part.rstrip("\\").strip() for part in pending).split()
                    )
                    pending = []
        return result

    def test_the_recipes_are_identical_outside_the_resolver_source(self):
        def skeleton(text):
            kept = []
            for tokens in self.instructions(text):
                if tokens[0] == "ARG" and tokens[1].partition("=")[0] in (
                    "BASE_IMAGE",
                    "FLAGOS_INDEX",
                    "TENCENT_INDEX",
                    "PROXY_INDEX",
                ):
                    continue
                if "/usr/local/bin/uv" in tokens:
                    continue
                kept.append(tokens)
            return kept

        self.assertEqual(skeleton(self.offline), skeleton(self.release))
        # the release recipe defaults to the moving tag; the candidate has no
        # default at all, its base is always an explicit digest
        self.assertEqual(
            [
                tokens
                for tokens in self.instructions(self.offline)
                if tokens[0] == "ARG"
            ],
            [["ARG", "BASE_IMAGE"]],
        )
        self.assertNotIn("2.3.0", self.offline)

    def test_the_offline_run_is_the_release_run_with_a_wheelhouse_source(self):
        release = self.contract.install_arguments(self.release)
        offline = self.contract.install_arguments(self.offline)

        def without_index(arguments):
            kept, skip = [], False
            for argument in arguments:
                if skip:
                    skip = False
                elif argument in ("--index", "--default-index", "--index-strategy"):
                    skip = True
                else:
                    kept.append(argument)
            return kept

        source = offline[offline.index("--no-index") : offline.index("/wheelhouse") + 1]
        self.assertEqual(source, ["--no-index", "--find-links", "/wheelhouse"])
        rest = [argument for argument in offline if argument not in source]
        self.assertEqual(rest, without_index(release))
        for banned in ("--no-deps", "--index", "--default-index", "--index-strategy"):
            self.assertNotIn(banned, offline)

    def test_the_offline_run_has_no_network_and_mounts_every_input_read_only(self):
        self.assertEqual(
            self.contract.run_options(self.offline, "/ctx"),
            [
                "--network",
                "none",
                "--volume",
                "/ctx/artifacts:/tmp/artifacts:ro",
                "--volume",
                "/ctx/wheelhouse:/wheelhouse:ro",
            ],
        )
        self.assertIn("RUN --network=none", self.offline)

    def test_the_offline_recipe_depends_on_nothing_external(self):
        # the pinned frontend image lives on docker.io; the builder's own
        # Dockerfile frontend is enough and keeps the build offline
        self.assertIsNone(re.search(r"^# syntax=", self.offline, re.MULTILINE))
        for banned in (
            "COPY",
            "WORKDIR",
            "/opt/sglang-ci",
            "git apply",
            "hcu.patch",
            "pip check",
            "requirements.txt",
            "/tmp/stubs",
            "TRITON_HIP_CLANG_PATH",
            "hcu-clang",
            "upstream/",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
        ):
            self.assertNotIn(banned, self.offline, banned)
        self.assertNotIn("ENV PYTHONPATH=", self.offline)
        self.assertNotIn("ENV LD_LIBRARY_PATH=", self.offline)
        self.assertNotIn("ENV ROCM_PATH=", self.offline)
        self.assertNotIn("/opt/rocm", self.offline)
        # the candidate installs the very wheel files the release installs
        wheels = [
            argument
            for argument in self.contract.install_arguments(self.offline)
            if argument.startswith("/tmp/artifacts/")
        ]
        self.assertEqual(
            wheels,
            [
                argument
                for argument in self.contract.install_arguments(self.release)
                if argument.startswith("/tmp/artifacts/")
            ],
        )


class WheelhouseTests(unittest.TestCase):
    """The offline wheelhouse is rebuilt from a green image's uv cache and
    verified against it; nothing is repacked."""

    SOURCE_IMAGE = "registry.example/dev/app@sha256:" + "7" * 64
    ORJSON = {
        "orjson-3.13.0.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: orjson\nVersion: 3.13.0\n"
        ),
        "orjson/__init__.py": "x = 1\n",
    }
    CT = {
        "compressed_tensors-0.17.0+flagos.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: compressed-tensors\nVersion: 0.17.0+flagos\n"
        ),
        "compressed_tensors/__init__.py": "x = 2\n",
    }
    ORJSON_WHEEL = "orjson-3.13.0-py3-none-any.whl"
    CT_WHEEL = "compressed_tensors-0.17.0+flagos-py3-none-any.whl"
    ORJSON_ID = "tYIbTVCg0zyAL5w5"
    CT_ID = "_zQAbIxLNj_FqERs"

    @classmethod
    def setUpClass(cls):
        cls.module = load_script("wheelhouse")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="wheelhouse-test.")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cache = self.root / "cache"
        self.sources = self.root / "sources"
        self.out = self.root / "wheelhouse"
        self.sources.mkdir()

    def cache_entry(self, archive_id, filename, files, url=None, bucket="pypi"):
        """Write a wheel, its uv cache entry and its unpacked cache tree;
        return the manifest record the fetch should produce."""
        wheel = self.sources / filename
        with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
            for member, content in files.items():
                archive.writestr(member, content)
        stem = filename[: -len(".whl")]
        name, version = stem.rsplit("-", 3)[0].split("-", 1)
        sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
        url = url or f"https://files.example.invalid/{filename}"
        self.assertLess(len(url), 128)
        # uv 0.11.30 layout: 2-byte prefix, archive id, then length-prefixed
        # Sha256 header / filename / URL and the quoted etag
        data = (
            b"\x94\xb0"
            + archive_id.encode()
            + b"\x91\x92\xa6Sha256\xd9\x40"
            + sha.encode()
            + b"\xd9"
            + bytes([len(filename)])
            + filename.encode()
            + b" "
            + bytes([len(url)])
            + url.encode()
            + b'"'
            + b"0" * 32
        )
        entry = self.cache / "wheels-v6" / bucket / name
        entry.mkdir(parents=True, exist_ok=True)
        (entry / f"{version}-py3-none-any.http").write_bytes(data)
        for member, content in files.items():
            path = self.cache / "archive-v0" / archive_id / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
        return {
            "filename": filename,
            "name": name,
            "version": version,
            "sha256": sha,
            "size": wheel.stat().st_size,
            "source_url": url,
            "download_url": url,
        }

    def entry_path(self, name, version):
        return self.cache / f"wheels-v6/pypi/{name}/{version}-py3-none-any.http"

    def local_fetch(self, url, destination):
        """A download that serves the prepared wheel files instead of the
        network."""
        destination.write_bytes((self.sources / url.rsplit("/", 1)[-1]).read_bytes())

    def fetch(self, **overrides):
        return self.module.fetch(
            self.cache,
            self.out,
            self.SOURCE_IMAGE,
            "https://mirror.example.invalid/simple",
            2,
            **overrides,
        )

    def test_inventory_reads_every_recorded_wheel(self):
        orjson = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        compressed = self.cache_entry(
            self.CT_ID, self.CT_WHEEL, self.CT, bucket="index"
        )
        entries = self.module.inventory(self.cache)
        self.assertEqual(
            [
                (entry["filename"], entry["sha256"], entry["bucket"])
                for entry in entries
            ],
            [
                (compressed["filename"], compressed["sha256"], "index"),
                (orjson["filename"], orjson["sha256"], "pypi"),
            ],
        )
        self.assertEqual(entries[1]["archive_id"], self.ORJSON_ID)
        self.assertEqual(entries[1]["version"], "3.13.0")
        self.assertEqual(entries[0]["version"], "0.17.0+flagos")
        self.assertEqual(entries[0]["source_url"], compressed["source_url"])

    def test_direct_url_wheels_are_never_wheelhouse_entries(self):
        self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON, bucket="url")
        with self.assertRaisesRegex(self.module.WheelhouseError, "no registry wheel"):
            self.module.inventory(self.cache)

    def test_an_entry_whose_archive_disagrees_is_rejected(self):
        self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        tree = self.cache / "archive-v0" / self.ORJSON_ID
        (tree / "orjson-3.13.0.dist-info").rename(tree / "other-9.9.dist-info")
        with self.assertRaisesRegex(self.module.WheelhouseError, "dist-info of"):
            self.module.inventory(self.cache)

    def test_an_entry_without_a_wheel_url_is_rejected(self):
        self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        path = self.entry_path("orjson", "3.13.0")
        path.write_bytes(path.read_bytes().replace(b"https://", b"https-x//", 1))
        with self.assertRaisesRegex(self.module.WheelhouseError, "one wheel URL"):
            self.module.inventory(self.cache)

    def test_fetch_rebuilds_the_recorded_wheels_and_their_manifest(self):
        orjson = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        compressed = self.cache_entry(self.CT_ID, self.CT_WHEEL, self.CT)
        manifest = self.fetch(fetch_url=self.local_fetch)
        self.assertEqual(manifest["count"], 2)
        self.assertEqual(manifest["cache_source"]["image"], self.SOURCE_IMAGE)
        self.assertEqual(manifest["resolver_mode"], "offline-wheelhouse")
        self.assertEqual(manifest["excluded_explicit_requirements"], [])
        for record in (orjson, compressed):
            self.assertEqual(
                self.module.sha256_file(self.out / record["filename"]),
                record["sha256"],
            )
        by_name = {wheel["name"]: wheel for wheel in manifest["wheels"]}
        self.assertEqual(by_name["orjson"]["source_url"], orjson["source_url"])
        self.assertEqual(by_name["orjson"]["download_url"], orjson["source_url"])
        self.assertEqual(by_name["orjson"]["cache_archive_id"], self.ORJSON_ID)
        self.assertIn("registry SHA256", by_name["orjson"]["verified"])
        # the manifest on disk passes verification
        self.assertEqual(self.module.verify(self.out)["count"], 2)

    def test_the_originals_of_the_explicit_wheels_stay_out(self):
        original = "xgrammar-0.2.1-cp310-cp310-manylinux_2_28_x86_64.whl"
        xgrammar = self.cache_entry(
            self.CT_ID,
            original,
            {
                "xgrammar-0.2.1.dist-info/METADATA": (
                    "Metadata-Version: 2.1\nName: xgrammar\nVersion: 0.2.1\n"
                ),
                "xgrammar/__init__.py": "x = 3\n",
            },
        )
        orjson = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        manifest = self.fetch(fetch_url=self.local_fetch)
        self.assertEqual(
            [wheel["filename"] for wheel in manifest["wheels"]], [orjson["filename"]]
        )
        self.assertEqual(
            manifest["excluded_explicit_requirements"],
            [
                {
                    key: xgrammar[key]
                    for key in ("filename", "name", "sha256", "source_url", "version")
                }
            ],
        )

    def test_a_wheel_that_does_not_hash_to_the_registry_digest_is_rejected(self):
        record = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        (self.sources / record["filename"]).write_bytes(b"not the wheel")
        with self.assertRaisesRegex(self.module.WheelhouseError, "SHA256"):
            self.fetch(fetch_url=self.local_fetch)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_content_other_than_the_cached_tree_is_rejected(self):
        # the digest is honest for a wheel whose payload differs from what the
        # green build unpacked: the cache tree is the second witness
        record = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        wheel = self.sources / record["filename"]
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("orjson/__init__.py", "x = 99\n")
            archive.writestr(
                "orjson-3.13.0.dist-info/METADATA",
                self.ORJSON["orjson-3.13.0.dist-info/METADATA"],
            )
        path = self.entry_path("orjson", "3.13.0")
        data = bytearray(path.read_bytes())
        position = data.index(record["sha256"].encode())
        data[position : position + 64] = (
            hashlib.sha256(wheel.read_bytes()).hexdigest().encode()
        )
        path.write_bytes(data)
        with self.assertRaisesRegex(
            self.module.WheelhouseError, "differ from the cached"
        ):
            self.fetch(fetch_url=self.local_fetch)

    def test_a_wheel_that_lies_about_its_metadata_is_rejected(self):
        lying = dict(self.ORJSON)
        lying["orjson-3.13.0.dist-info/METADATA"] = (
            "Metadata-Version: 2.1\nName: orjson\nVersion: 3.12.0\n"
        )
        self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, lying)
        # name and version come from the filename; the wheel's METADATA
        # disagrees with it although the cache tree is identical
        with self.assertRaisesRegex(self.module.WheelhouseError, "METADATA says"):
            self.fetch(fetch_url=self.local_fetch)

    def test_the_mirror_completes_a_generic_wheel_when_the_recorded_url_is_dead(
        self,
    ):
        record = self.cache_entry(
            self.ORJSON_ID,
            self.ORJSON_WHEEL,
            self.ORJSON,
            url="https://dead.example.invalid/" + self.ORJSON_WHEEL,
        )
        mirror = f"file://{self.sources / self.ORJSON_WHEEL}"
        manifest = self.fetch(
            pause=lambda seconds: None, lookup=lambda index, entry: mirror
        )
        wheel = manifest["wheels"][0]
        self.assertEqual(wheel["download_url"], mirror)
        self.assertEqual(wheel["source_url"], record["source_url"])

    def test_a_wheel_that_cannot_be_established_leaves_no_wheelhouse(self):
        self.cache_entry(
            self.ORJSON_ID,
            self.ORJSON_WHEEL,
            self.ORJSON,
            url="https://dead.example.invalid/" + self.ORJSON_WHEEL,
        )
        with self.assertRaisesRegex(
            self.module.WheelhouseError, "could not be established"
        ):
            self.fetch(pause=lambda seconds: None, lookup=lambda index, entry: None)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_an_occupied_output_directory_is_refused(self):
        self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        self.out.mkdir()
        (self.out / "leftover.whl").write_bytes(b"x")
        with self.assertRaisesRegex(self.module.WheelhouseError, "not empty"):
            self.fetch(fetch_url=self.local_fetch)

    def test_verify_rejects_a_changed_extra_or_missing_file(self):
        orjson = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        compressed = self.cache_entry(self.CT_ID, self.CT_WHEEL, self.CT)
        self.fetch(fetch_url=self.local_fetch)
        self.assertEqual(self.module.verify(self.out)["count"], 2)
        # changed
        target = self.out / orjson["filename"]
        keep = target.read_bytes()
        target.write_bytes(b"tampered")
        with self.assertRaisesRegex(
            self.module.WheelhouseError, "changed since the manifest"
        ):
            self.module.verify(self.out)
        target.write_bytes(keep)
        # extra
        (self.out / "surprise.whl").write_bytes(b"x")
        with self.assertRaisesRegex(self.module.WheelhouseError, "unexpected"):
            self.module.verify(self.out)
        (self.out / "surprise.whl").unlink()
        # missing
        moved = self.out / compressed["filename"]
        keep = moved.read_bytes()
        moved.unlink()
        with self.assertRaisesRegex(self.module.WheelhouseError, "missing"):
            self.module.verify(self.out)
        moved.write_bytes(keep)
        self.assertEqual(self.module.verify(self.out)["count"], 2)

    def test_bind_plan_accepts_only_the_wheelhouse_closure(self):
        orjson = self.cache_entry(self.ORJSON_ID, self.ORJSON_WHEEL, self.ORJSON)
        self.cache_entry(self.CT_ID, self.CT_WHEEL, self.CT)
        self.fetch(fetch_url=self.local_fetch)
        plan, consumed = self.root / "plan.txt", self.root / "consumed.txt"

        def bound(lines):
            plan.write_text("\n".join(lines) + "\n")
            return self.module.bind_plan(self.out, plan, consumed)

        report = bound(
            [
                "sglang==0.5.18+flagos.hcu1",
                "sglang-fl==0.1.0",
                "xgrammar==0.2.1+flagos.hcu1",
                "orjson==3.13.0",
            ]
        )
        self.assertEqual(report, {"consumed": 1, "unused": ["compressed-tensors"]})
        self.assertIn(
            f"orjson==3.13.0 {orjson['sha256']} {orjson['filename']}",
            consumed.read_text(),
        )
        for lines in (
            ["orjson==3.12.0"],
            ["datasets==5.2.0"],
            ["xgrammar==0.2.1"],
            ["sglang==0.5.18+flagos"],
        ):
            with self.subTest(lines), self.assertRaises(self.module.WheelhouseError):
                bound(lines)

    def test_the_cli_fails_loudly(self):
        self.assertEqual(
            self.module.main(["verify", "--wheelhouse", str(self.root / "absent")]), 1
        )


class ImportSmokeVerdictTests(unittest.TestCase):
    """Phase 5 decides by the log and stops waiting once the verdict is final."""

    ABORT = (
        "PlatformFL initialized: vendor=hygon, device=cuda, dist_backend=nccl, count=0\n"
        "terminate called after throwing an instance of 'c10::AcceleratorError'\n"
        "  what():  HIP error: no ROCm-capable device is detected\n"
        "Search for `hipErrorNoDevice' in the HIP documentation.\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.verifier = (SCRIPTS / "verify_image.sh").read_text()

    def region(self, start, end):
        first = self.verifier.index(start)
        return self.verifier[first : self.verifier.index(end, first)]

    def verdict_in(self, log):
        """Run the phase's own verdict predicate against a log."""
        function = self.region("smoke_verdict_in_log() {", "\nsmoke_container=")
        with tempfile.TemporaryDirectory(prefix="smoke-verdict-test.") as tmp:
            path = Path(tmp) / "smoke.log"
            path.write_text(log)
            script = Path(tmp) / "verdict.sh"
            script.write_text(
                "\n".join(
                    [
                        "set -euo pipefail",
                        f'smoke_log="{path}"',
                        function,
                        "smoke_verdict_in_log",
                    ]
                )
                + "\n"
            )
            completed = subprocess.run(
                ["bash", str(script)], capture_output=True, text=True
            )
        self.assertNotIn("SyntaxError", completed.stderr)
        return completed.returncode == 0

    def test_the_python_verdicts_and_the_device_abort_are_final_verdicts(self):
        self.assertTrue(
            self.verdict_in("warnings\nSMOKE-PASS sglang 0.5.18+flagos.hcu1\n")
        )
        self.assertTrue(
            self.verdict_in(
                "traceback\nSMOKE-TOLERATED sglang import blocked by device libraries only\n"
            )
        )
        self.assertTrue(self.verdict_in(self.ABORT))

    def test_a_mention_of_the_device_error_inside_a_failed_import_is_no_verdict(
        self,
    ):
        # the smoke script classifies the exception itself: a traceback that
        # merely mentions the device error and then fails is not a verdict
        self.assertFalse(
            self.verdict_in(
                "Traceback\nImportError: HIP error: no ROCm-capable device is detected\n"
                "sglang import failed for reasons other than missing device libraries\n"
            )
        )
        self.assertFalse(self.verdict_in(""))
        self.assertFalse(
            self.verdict_in(
                self.ABORT.replace("c10::AcceleratorError", "std::bad_alloc")
            )
        )

    def test_the_phase_waits_for_a_verdict_within_a_bound(self):
        phase5 = self.region("=== [5/8] SGLang import smoke ===", "=== [6/8]")
        self.assertIn("docker run -d --platform linux/amd64", phase5)
        self.assertIn("for _ in $(seq 1 180); do", phase5)
        self.assertIn('docker rm -f "$smoke_container"', phase5)
        self.assertIn("produced no verdict within 30 minutes", phase5)
        self.assertIn("failed for reasons other than missing device libraries", phase5)


_STAGED = {}


def staged(**environment):
    """Run stage_wheels.sh once per environment (it builds the plugin wheel
    and derives both platform wheels); the context outlives the call."""
    key = tuple(sorted(environment.items()))
    if key not in _STAGED:
        completed = subprocess.run(
            ["bash", str(DOCKER / "stage_wheels.sh")],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, **environment},
        )
        match = re.search(r"^context:\s+(\S+)", completed.stdout, re.MULTILINE)
        _STAGED[key] = (completed, Path(match.group(1)) if match else None)
    return _STAGED[key]


class StagedWheelhouseTests(unittest.TestCase):
    """stage_wheels.sh carries an external wheelhouse into the build context."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_script("wheelhouse")
        cls._tmp = tempfile.TemporaryDirectory(prefix="staged-wheelhouse-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.wheelhouse = write_synthetic_wheelhouse(
            cls.module,
            Path(cls._tmp.name) / "wheelhouse",
            (("alpha", "1.0"), ("beta", "2.0+local.1")),
        )
        cls.completed, cls.context = staged(HYGON_WHEELHOUSE=str(cls.wheelhouse))
        if cls.context is not None:
            cls.addClassCleanup(shutil.rmtree, cls.context, ignore_errors=True)

    def test_the_context_carries_the_verified_wheelhouse(self):
        completed = self.completed
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        copied = self.context / "wheelhouse"
        self.assertEqual(self.module.verify(copied)["count"], 2)
        for wheel in self.wheelhouse.iterdir():
            self.assertEqual(wheel.read_bytes(), (copied / wheel.name).read_bytes())
        digest = hashlib.sha256(
            (copied / "wheelhouse-manifest.json").read_bytes()
        ).hexdigest()
        self.assertIn(
            f"resolver:       offline-wheelhouse (2 wheels, manifest sha256={digest}",
            completed.stdout,
        )
        self.assertIn(f"--offline-wheelhouse {self.wheelhouse}", completed.stdout)

    def test_a_tampered_wheelhouse_stops_the_staging(self):
        wheel = next(self.wheelhouse.glob("beta-*.whl"))
        keep = wheel.read_bytes()
        wheel.write_bytes(b"tampered")
        try:
            completed = subprocess.run(
                ["bash", str(DOCKER / "stage_wheels.sh")],
                capture_output=True,
                text=True,
                timeout=900,
                env={**os.environ, "HYGON_WHEELHOUSE": str(self.wheelhouse)},
            )
        finally:
            wheel.write_bytes(keep)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("changed since the manifest", completed.stderr)
        self.assertNotIn("context:", completed.stdout)

    def test_without_a_wheelhouse_staging_stays_on_the_index_path(self):
        completed, context = staged()
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.addClassCleanup(shutil.rmtree, context, ignore_errors=True)
        self.assertIn("resolver:       index", completed.stdout)
        self.assertFalse((context / "wheelhouse").exists())


class BuildModeGateTests(unittest.TestCase):
    """build_two_wheel.sh decides the resolver mode and the publication path
    before anything is staged."""

    COMMIT = "a" * 40
    RUNTIME = "registry.example/runtime@sha256:" + "b" * 64

    @classmethod
    def setUpClass(cls):
        cls.wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        cls._tmp = tempfile.TemporaryDirectory(prefix="mode-gate-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.directory = Path(cls._tmp.name) / "wheelhouse"
        cls.directory.mkdir()

    def gates(self, **options):
        wrapper = self.wrapper
        region = wrapper[wrapper.index("publish=$((push + push_candidate))") :]
        block_text = region[: region.index('echo "=== [1/6] Staging')]
        settings = {
            "push": 0,
            "push_candidate": 0,
            "allow_dirty": 0,
            "wheelhouse": "",
            "runtime_image": "",
            "tag": "dev/image:1",
            "head": self.COMMIT,
            "dirty": "",
            "remote": "free",
            "message": "unknown: artifact not found",
        }
        settings.update(options)
        settings["tag"] = settings["tag"].format(commit=self.COMMIT[:12])
        lines = [
            "set -euo pipefail",
            'repo_root="$1"',
            re.search(r"^CANDIDATE_TAG_PATTERN=.*$", wrapper, re.MULTILINE).group(0),
            re.search(r"^INDEX_MODE=.*$", wrapper, re.MULTILINE).group(0),
            re.search(r"^OFFLINE_MODE=.*$", wrapper, re.MULTILINE).group(0),
            "RELEASE_RECIPE=unused; OFFLINE_RECIPE=unused",
            "push={push}; push_candidate={push_candidate}; allow_dirty={allow_dirty}".format(
                **settings
            ),
            'wheelhouse="{wheelhouse}"; runtime_image="{runtime_image}"; tag="{tag}"'.format(
                **settings
            ),
            block_text,
        ]
        return run_block(
            lines,
            {
                "FAKE_HEAD": settings["head"],
                "FAKE_DIRTY": settings["dirty"],
                "FAKE_REMOTE": settings["remote"],
                "FAKE_REMOTE_MESSAGE": settings["message"],
            },
            str(self._tmp.name),
        )

    def candidate(self, **options):
        candidate = {
            "push_candidate": 1,
            "wheelhouse": str(self.directory),
            "runtime_image": self.RUNTIME,
            "tag": CANDIDATE_TAG,
        }
        candidate.update(options)
        return self.gates(**candidate)

    def test_the_release_path_is_the_index_mode(self):
        for push in (0, 1):
            with self.subTest(push=push):
                completed = self.gates(push=push)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("BLOCK-DONE", completed.stdout)

    def test_the_candidate_path_passes_its_gates(self):
        completed = self.candidate()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("candidate tag is free in the registry", completed.stdout)
        self.assertIn("BLOCK-DONE", completed.stdout)

    def test_every_inconsistent_mode_is_refused(self):
        directory, runtime = str(self.directory), self.RUNTIME
        for name, options, message in (
            (
                "offline without a runtime digest",
                dict(wheelhouse=directory),
                "requires --runtime-image",
            ),
            (
                "release push of an offline build",
                dict(push=1, wheelhouse=directory, runtime_image=runtime),
                "cannot consume an offline wheelhouse",
            ),
            (
                "runtime image without a wheelhouse",
                dict(runtime_image=runtime),
                "--runtime-image goes with --offline-wheelhouse",
            ),
            (
                "candidate push without a wheelhouse",
                dict(push_candidate=1, tag=CANDIDATE_TAG),
                "publishes the offline-wheelhouse development candidate only",
            ),
            ("both publications", dict(push=1, push_candidate=1), "mutually exclusive"),
            (
                "dirty candidate push",
                dict(
                    push_candidate=1,
                    allow_dirty=1,
                    wheelhouse=directory,
                    runtime_image=runtime,
                    tag=CANDIDATE_TAG,
                ),
                "--allow-dirty cannot be combined",
            ),
            (
                "dirty release push",
                dict(push=1, allow_dirty=1),
                "--allow-dirty cannot be combined",
            ),
            (
                "missing wheelhouse directory",
                dict(wheelhouse="/nonexistent/dir", runtime_image=runtime),
                "not a directory",
            ),
        ):
            with self.subTest(name):
                completed = self.gates(**options)
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn(message, completed.stderr)
                self.assertNotIn("BLOCK-DONE", completed.stdout)

    def test_a_publishable_build_needs_a_clean_worktree(self):
        for completed in (
            self.gates(push=1, dirty=" M docker/hygon/x\n"),
            self.candidate(dirty=" M docker/hygon/x\n"),
        ):
            self.assertEqual(completed.returncode, 1)
            self.assertIn("worktree is dirty", completed.stderr)
        # a dirty dev build that publishes nothing stays possible
        completed = self.gates(allow_dirty=1, dirty=" M x\n")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("NOT releasable", completed.stdout)

    def test_a_candidate_tag_names_the_commit_and_is_never_overwritten(self):
        for name, options, message in (
            ("free-form tag", dict(tag="dev/candidate:1"), "a candidate tag must be"),
            (
                "an existing release tag",
                dict(
                    tag=(
                        "harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:"
                        "0.5.18-hygon-two-wheel-20261118"
                    )
                ),
                "a candidate tag must be",
            ),
            (
                "another commit in the tag",
                dict(tag=CANDIDATE_TAG.format(commit="b" * 12)),
                "names commit",
            ),
            ("an existing remote tag", dict(remote="present"), "refusing to overwrite"),
            (
                "an unreachable registry",
                dict(remote="error", message="dial tcp: i/o timeout"),
                "cannot tell whether",
            ),
        ):
            with self.subTest(name):
                completed = self.candidate(**options)
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn(message, completed.stderr)
                self.assertNotIn("BLOCK-DONE", completed.stdout)


class RuntimeBindingTests(unittest.TestCase):
    """The candidate's Runtime base is an explicit digest, verified against
    the local Docker store; no tag is ever consulted."""

    DIGEST = "sha256:" + "c" * 64
    REFERENCE = "registry.example/runtime@" + DIGEST

    @classmethod
    def setUpClass(cls):
        cls.wrapper = (DOCKER / "build_two_wheel.sh").read_text()

    def bind(
        self,
        reference=None,
        present=True,
        identifier=None,
        repo_digests="",
        remote="free",
    ):
        wrapper = self.wrapper
        offline = block(
            wrapper,
            "    # The candidate names its Runtime explicitly",
            'BASE_LOCAL_IMAGE_ID="$(docker image inspect',
        )
        # the slice ends with the `fi` that closes the mode branch
        offline = offline[: offline.rindex("\nfi\n") + 1]
        lines = [
            "set -euo pipefail",
            re.search(r"^RUNTIME_REFERENCE_PATTERN=.*$", wrapper, re.MULTILINE).group(
                0
            ),
            f'runtime_image="{reference or self.REFERENCE}"',
            block(
                wrapper,
                "registry_ref_state() {",
                '\n\nif [ "$push_candidate" -eq 1 ]; then\n    state="$(registry_ref_state',
            ).rstrip("\n"),
            offline,
            block(
                wrapper,
                'BASE_LOCAL_IMAGE_ID="$(docker image inspect',
                'refdir="$(mktemp -d',
            ),
        ]
        identifier = identifier or self.DIGEST
        return run_block(
            lines,
            {
                "FAKE_PRESENT": (reference or self.REFERENCE) if present else "",
                "FAKE_IMAGE_ID": identifier,
                "FAKE_REPO_DIGESTS": repo_digests,
                "FAKE_REMOTE": remote,
                "FAKE_REMOTE_MESSAGE": (
                    "dial tcp: i/o timeout"
                    if remote == "error"
                    else "unknown: artifact not found"
                ),
            },
        )

    def test_a_digest_that_is_the_local_image_id_is_accepted(self):
        completed = self.bind()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(f"runtime base:         {self.REFERENCE}", completed.stdout)
        self.assertIn("remote manifest:      absent", completed.stdout)
        self.assertIn("the push uploads every parent layer", completed.stdout)

    def test_a_digest_among_the_repo_digests_is_accepted(self):
        other = "sha256:" + "d" * 64
        completed = self.bind(identifier=other, repo_digests=self.REFERENCE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(f"local image id:       {other}", completed.stdout)

    def test_the_registry_state_of_the_base_is_recorded_never_a_gate(self):
        present = self.bind(remote="present")
        self.assertEqual(present.returncode, 0, present.stderr)
        self.assertIn("remote manifest:      present", present.stdout)
        unreachable = self.bind(remote="error")
        self.assertEqual(unreachable.returncode, 0, unreachable.stderr)
        self.assertIn("remote manifest:      unavailable", unreachable.stdout)

    def test_a_local_object_that_is_not_the_digest_is_refused(self):
        completed = self.bind(identifier="sha256:" + "d" * 64)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("is not that digest", completed.stderr)
        self.assertNotIn("BLOCK-DONE", completed.stdout)

    def test_an_absent_object_is_never_pulled_or_replaced_by_a_tag(self):
        completed = self.bind(present=False)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("not in the local Docker store", completed.stderr)
        for tag in ("registry.example/runtime:2.3.0", "runtime:latest"):
            with self.subTest(tag):
                completed = self.bind(reference=tag, present=False)
                self.assertEqual(completed.returncode, 1)
                self.assertIn(
                    "must be name@sha256:<64 hex digest>, never a tag", completed.stderr
                )


class CandidateContractTests(unittest.TestCase):
    """verify_image.sh holds an image to the resolver mode it declares."""

    BASE = "registry.example/runtime@sha256:" + "e" * 64
    BASE_ID = "sha256:" + "e" * 64
    BASE_LAYERS = json.dumps(["sha256:" + "1" * 64, "sha256:" + "2" * 64])
    IMAGE_ID = "sha256:" + "f" * 64
    IMAGE_LAYERS = json.dumps(
        ["sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64]
    )
    SOURCE_IMAGE = "registry.example/dev/green@sha256:" + "9" * 64
    LABELS = {
        "mode": "org.flagos.ci.resolver-mode",
        "release_ready": "org.flagos.ci.release-ready",
        "cache_source": "org.flagos.ci.resolver-cache-source",
        "manifest_sha": "org.flagos.ci.wheelhouse-manifest-sha256",
        "base_id": "org.flagos.ci.base-local-image-id",
    }

    @classmethod
    def setUpClass(cls):
        cls.verifier = (SCRIPTS / "verify_image.sh").read_text()
        cls._tmp = tempfile.TemporaryDirectory(prefix="candidate-contract-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.manifest = Path(cls._tmp.name) / "wheelhouse-manifest.json"
        cls.manifest.write_text(
            json.dumps({"cache_source": {"image": cls.SOURCE_IMAGE}})
        )
        cls.manifest_sha = hashlib.sha256(cls.manifest.read_bytes()).hexdigest()

    def candidate_labels(self, **override):
        values = {
            "mode": "offline-wheelhouse",
            "release_ready": "false",
            "cache_source": self.SOURCE_IMAGE,
            "manifest_sha": self.manifest_sha,
            "base_id": self.BASE_ID,
        }
        values.update(override)
        return {
            self.LABELS[key]: value
            for key, value in values.items()
            if value is not None
        }

    def verify(self, labels, manifest=True, layers=IMAGE_LAYERS):
        block_text = block(
            self.verifier,
            "label() { docker image inspect",
            "\n# The required (prepended) DTK loader prefix",
        )
        lines = [
            "set -euo pipefail",
            "image=fake-image",
            f'BASE_IMAGE="{self.BASE}"',
            f'wheelhouse_manifest="{self.manifest if manifest else ""}"',
            block_text,
        ]
        return run_block(
            lines,
            {
                "FAKE_BASE": self.BASE,
                "FAKE_BASE_ID": self.BASE_ID,
                "FAKE_BASE_LAYERS": self.BASE_LAYERS,
                "FAKE_IMAGE_ID": self.IMAGE_ID,
                "FAKE_IMAGE_LAYERS": layers,
                "FAKE_PRESENT": f"{self.BASE} fake-image",
                "FAKE_LABELS": "\n".join(
                    f"{key}={value}" for key, value in labels.items()
                ),
            },
        )

    def test_an_offline_wheelhouse_candidate_with_its_manifest_is_accepted(self):
        completed = self.verify(self.candidate_labels())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "all 2 Runtime base layers lead the 3 image layers", completed.stdout
        )
        self.assertIn("BLOCK-DONE", completed.stdout)

    def test_an_index_image_needs_no_manifest_and_takes_none(self):
        labels = {
            "org.flagos.ci.resolver-mode": "index",
            self.LABELS["base_id"]: self.BASE_ID,
        }
        accepted = self.verify(labels, manifest=False)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        refused = self.verify(labels)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("resolved from the indexes", refused.stderr)

    def test_a_candidate_verified_without_its_manifest_is_refused(self):
        completed = self.verify(self.candidate_labels(), manifest=False)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("needs --wheelhouse-manifest", completed.stderr)

    def test_the_candidate_contract_rejects_every_deviation(self):
        for name, override, message in (
            (
                "not declared release-ready=false",
                dict(release_ready="true"),
                "release-ready=false",
            ),
            (
                "release-ready label missing",
                dict(release_ready=None),
                "release-ready=false",
            ),
            (
                "another cache source",
                dict(cache_source="registry.example/dev/other@sha256:" + "8" * 64),
                "resolver-cache-source is not the manifest",
            ),
            (
                "another wheelhouse manifest",
                dict(manifest_sha="0" * 64),
                "wheelhouse-manifest-sha256 does not match",
            ),
            (
                "another local base",
                dict(base_id="sha256:" + "0" * 64),
                "base-local-image-id",
            ),
            (
                "no resolver mode",
                dict(mode=None),
                "lacks a valid org.flagos.ci.resolver-mode",
            ),
            (
                "an unknown resolver mode",
                dict(mode="wheel"),
                "lacks a valid org.flagos.ci.resolver-mode",
            ),
        ):
            with self.subTest(name):
                completed = self.verify(self.candidate_labels(**override))
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn(message, completed.stderr)
                self.assertNotIn("BLOCK-DONE", completed.stdout)

    def test_an_image_without_the_complete_runtime_base_is_refused(self):
        for layers in (
            json.dumps(
                ["sha256:" + "9" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64]
            ),
            json.dumps(["sha256:" + "1" * 64]),
        ):
            with self.subTest(layers):
                completed = self.verify(self.candidate_labels(), layers=layers)
                self.assertEqual(completed.returncode, 1)
                self.assertIn(
                    "does not carry the complete Runtime base", completed.stderr
                )

    def test_the_verifier_passes_the_staged_wheels_and_the_manifest(self):
        build = (DOCKER / "build_two_wheel.sh").read_text()
        for pin in (
            '--sglang-wheel "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME"',
            '--xgrammar-wheel "$context/artifacts/$DERIVED_XGRAMMAR_WHEEL_NAME"',
            '--wheelhouse-manifest "$context/wheelhouse/wheelhouse-manifest.json"',
            '--label "org.flagos.ci.resolver-cache-source=$resolver_cache_source"',
            '--label "org.flagos.ci.resolver-mode=$resolver_mode"',
            '--label "org.flagos.ci.base-local-image-id=$BASE_LOCAL_IMAGE_ID"',
        ):
            self.assertIn(pin, build)
        # the candidate labels are added in the offline-wheelhouse mode only
        labels = build[build.index("labels=(") : build.index("docker build --platform")]
        common, candidate = labels.split(
            'if [ "$resolver_mode" = "$OFFLINE_MODE" ]; then\n    # The candidate says what it is'
        )
        self.assertNotIn("release-ready", common)
        self.assertNotIn("resolver-cache-source", common)
        self.assertIn('--label "org.flagos.ci.release-ready=false"', candidate)
        self.assertIn(
            '--label "org.flagos.ci.wheelhouse-manifest-sha256=$wheelhouse_manifest_sha256"',
            candidate,
        )

    def test_the_verifier_compares_every_staged_wheel_member_by_member(self):
        verifier = self.verifier
        self.assertEqual(verifier.count("/scripts/verify_installed_wheel.py"), 4)
        for option in (
            "--wheel /wheel.whl --own-tree sglang\n",
            "--wheel /wheel.whl --own-tree xgrammar\n",
            "--wheel /wheel.whl --own-tree sglang_fl --min-members 100",
            "--wheelhouse-manifest /wheelhouse/wheelhouse-manifest.json",
            "--runtime /evidence/runtime-distributions-before.json",
        ):
            self.assertIn(option, verifier)
        # the closure check is bound to the candidate mode only
        self.assertIn('if [ -n "$wheelhouse_manifest" ]; then', verifier)
        # and the residue scan knows every build input
        for residue in (
            "/wheelhouse",
            "wheelhouse-manifest.json",
            "runtime-constraints.txt",
        ):
            self.assertIn(residue, verifier[verifier.index("build residue") :])


class InstalledWheelComparisonTests(unittest.TestCase):
    """verify_installed_wheel.py compares an installed distribution with its
    wheel, and an offline-wheelhouse closure with its manifest."""

    @classmethod
    def setUpClass(cls):
        import venv

        cls._tmp = tempfile.TemporaryDirectory(prefix="wheel-installed-test.")
        cls.addClassCleanup(cls._tmp.cleanup)
        tmp = Path(cls._tmp.name)
        home = tmp / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(home)
        cls.python = home / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        cls.purelib = Path(
            cls.python_code("import sysconfig; print(sysconfig.get_paths()['purelib'])")
        )
        # the verifier imports packaging: the fresh environment gets the copy
        # this test interpreter uses
        import packaging

        shutil.copytree(Path(packaging.__file__).parent, cls.purelib / "packaging")
        cls.verifier = SCRIPTS / "verify_installed_wheel.py"
        cls.wheel = synthetic_plugin_wheel(tmp / "sglang_fl-0.1.0-py3-none-any.whl")
        cls.pip("install", "-q", "--no-deps", "--no-index", str(cls.wheel))

    @classmethod
    def python_code(cls, code):
        return subprocess.run(
            [str(cls.python), "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()

    @classmethod
    def pip(cls, *arguments):
        subprocess.run(
            [str(cls.python), "-m", "pip", *arguments], check=True, capture_output=True
        )

    def compare(self, wheel=None):
        return subprocess.run(
            [
                str(self.python),
                str(self.verifier),
                "--wheel",
                str(wheel or self.wheel),
                "--own-tree",
                "sglang_fl",
                "--min-members",
                "100",
            ],
            capture_output=True,
            text=True,
        )

    def test_an_untouched_installation_is_the_wheel(self):
        completed = self.compare()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "installed sglang-fl==0.1.0: 123 wheel members are byte-identical",
            completed.stdout,
        )

    def test_the_mounted_file_name_does_not_decide_the_identity(self):
        # the verifier mounts every wheel as /wheel.whl
        mounted = Path(self._tmp.name) / "mounted" / "wheel.whl"
        mounted.parent.mkdir(exist_ok=True)
        shutil.copy(self.wheel, mounted)
        completed = self.compare(mounted)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_a_wheel_that_is_not_installed_is_detected(self):
        absent = tiny_wheel(
            Path(self._tmp.name) / "absent" / "absent-1.0-py3-none-any.whl",
            "absent",
            "1.0",
        )
        completed = subprocess.run(
            [str(self.python), str(self.verifier), "--wheel", str(absent)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("absent: installed None, wheel 1.0", completed.stderr)

    def test_a_changed_installed_file_is_detected(self):
        target = self.purelib / "sglang_fl/mod_007.py"
        original = target.read_bytes()
        target.write_text("value = 'tampered'\n")
        try:
            completed = self.compare()
        finally:
            target.write_bytes(original)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("1 changed ['sglang_fl/mod_007.py']", completed.stderr)

    def test_a_missing_installed_file_is_detected(self):
        target = self.purelib / "sglang_fl/mod_007.py"
        original = target.read_bytes()
        target.unlink()
        try:
            completed = self.compare()
        finally:
            target.write_bytes(original)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("1 missing ['sglang_fl/mod_007.py']", completed.stderr)

    def test_an_extra_file_under_the_package_is_detected(self):
        planted = self.purelib / "sglang_fl" / "planted.py"
        planted.write_text("value = 1\n")
        try:
            completed = self.compare()
        finally:
            planted.unlink()
        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "not in sglang_fl-0.1.0-py3-none-any.whl: ['sglang_fl/planted.py']",
            completed.stderr,
        )

    def test_a_wheel_with_too_few_members_cannot_pass_the_plugin_gate(self):
        small = tiny_wheel(
            Path(self._tmp.name) / "small" / "sglang_fl-0.1.0-py3-none-any.whl",
            "sglang_fl",
            "0.1.0",
        )
        completed = self.compare(small)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("fewer than 100", completed.stderr)

    def test_another_version_of_the_distribution_is_detected(self):
        other = tiny_wheel(
            Path(self._tmp.name) / "other" / "sglang_fl-0.2.0-py3-none-any.whl",
            "sglang_fl",
            "0.2.0",
        )
        completed = self.compare(other)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("installed 0.1.0, wheel 0.2.0", completed.stderr)

    def test_installer_rewrites_of_console_scripts_are_tolerated_bodies_are_not(
        self,
    ):
        module = load_script("verify_installed_wheel")
        body = b"import demo\nprint('tool')\n"
        wheel_form = b"#!python\n" + body
        plain_shebang = b"#!/tmp/venv/bin/python3\n" + body
        trampoline = (
            b"#!/bin/sh\n"
            b'\'\'\'exec\' "$(dirname -- "$(realpath -- "$0")")"/\'python3\' "$0" "$@"\n'
            b"' '''\n" + body
        )
        self.assertEqual(
            module.strip_script_prologue(plain_shebang),
            module.strip_script_prologue(wheel_form),
        )
        self.assertEqual(
            module.strip_script_prologue(trampoline),
            module.strip_script_prologue(wheel_form),
        )
        self.assertNotEqual(
            module.strip_script_prologue(
                trampoline.replace(b"print('tool')", b"print('x')")
            ),
            module.strip_script_prologue(wheel_form),
        )
        # end to end: pip rewrites the shebang of the console script it
        # installs and the distribution still compares equal; the uv
        # trampoline does too, a changed body does not
        wheel = tiny_wheel(
            Path(self._tmp.name) / "tool" / "demo-1.0-py3-none-any.whl",
            "demo",
            "1.0",
            {"demo-1.0.data/scripts/demo-tool": wheel_form},
        )
        self.pip("install", "-q", "--no-deps", "--no-index", str(wheel))

        def compare():
            return subprocess.run(
                [str(self.python), str(self.verifier), "--wheel", str(wheel)],
                capture_output=True,
                text=True,
            )

        completed = compare()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        script = (
            Path(
                self.python_code(
                    "import sysconfig; print(sysconfig.get_path('scripts'))"
                )
            )
            / "demo-tool"
        )
        self.assertTrue(script.read_bytes().startswith(b"#!"))
        script.write_bytes(trampoline)
        completed = compare()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        script.write_bytes(trampoline.replace(b"print('tool')", b"print('x')"))
        completed = compare()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("1 changed ['demo-1.0.data/scripts/demo-tool']", completed.stderr)

    def wheelhouse(self):
        """Install two wheelhouse wheels next to the three explicit wheels;
        return (wheelhouse dir, the Runtime baseline)."""
        module = load_script("wheelhouse")
        tmp = Path(self._tmp.name) / "closure"
        root = write_synthetic_wheelhouse(
            module, tmp / "wheelhouse", (("alpha", "1.0"), ("beta", "2.0+local.1"))
        )
        for name, version in (("alpha", "1.0"), ("beta", "2.0+local.1")):
            self.pip(
                "install",
                "-q",
                "--no-deps",
                "--no-index",
                str(root / f"{name}-{version}-py3-none-any.whl"),
            )
        for name, version in (
            ("sglang", "0.5.18+flagos.hcu1"),
            ("xgrammar", "0.2.1+flagos.hcu1"),
        ):
            wheel = tiny_wheel(
                tmp / "explicit" / f"{name}-{version}-py3-none-any.whl", name, version
            )
            self.pip("install", "-q", "--no-deps", "--no-index", str(wheel))
        # everything that is not part of the closure is "the Runtime"
        closure = ["alpha", "beta", "sglang", "sglang-fl", "xgrammar", "gamma"]
        baseline = tmp / "runtime.json"
        baseline.write_text(
            self.python_code(
                "import importlib.metadata as md, json\n"
                "from packaging.utils import canonicalize_name\n"
                f"closure = {closure!r}\n"
                "print(json.dumps({canonicalize_name(d.metadata['Name']): d.version "
                "for d in md.distributions() if d.metadata['Name'] "
                "and canonicalize_name(d.metadata['Name']) not in closure}))\n"
            )
        )
        return root, baseline

    def closure(self, manifest, baseline):
        return subprocess.run(
            [
                str(self.python),
                str(self.verifier),
                "--wheelhouse-manifest",
                str(manifest),
                "--runtime",
                str(baseline),
            ],
            capture_output=True,
            text=True,
        )

    def test_the_closure_binds_to_the_wheelhouse_manifest(self):
        module = load_script("wheelhouse")
        root, baseline = self.wheelhouse()
        completed = self.closure(root / "wheelhouse-manifest.json", baseline)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "2 wheelhouse wheels consumed, 6 installed members", completed.stdout
        )
        # a manifest that lists another version than the one installed
        other = write_synthetic_wheelhouse(
            module,
            Path(self._tmp.name) / "other" / "wheelhouse",
            (("alpha", "1.1"), ("beta", "2.0+local.1")),
        )
        completed = self.closure(other / "wheelhouse-manifest.json", baseline)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("alpha==1.0: the wheelhouse holds 1.1", completed.stderr)
        # an added distribution that is neither explicit nor wheelhouse
        gamma = tiny_wheel(
            Path(self._tmp.name) / "extra" / "gamma-5.0-py3-none-any.whl",
            "gamma",
            "5.0",
        )
        self.pip("install", "-q", "--no-deps", "--no-index", str(gamma))
        completed = self.closure(root / "wheelhouse-manifest.json", baseline)
        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "gamma==5.0: neither an explicit wheel nor a wheelhouse wheel",
            completed.stderr,
        )
        self.pip("uninstall", "-q", "-y", "gamma")
        # a consumed wheel whose installed content was changed
        target = self.purelib / "beta" / "__init__.py"
        original = target.read_bytes()
        target.write_text("tampered\n")
        try:
            completed = self.closure(root / "wheelhouse-manifest.json", baseline)
        finally:
            target.write_bytes(original)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("beta is not the content of", completed.stderr)
        # a missing explicit wheel
        self.pip("uninstall", "-q", "-y", "xgrammar")
        completed = self.closure(root / "wheelhouse-manifest.json", baseline)
        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "xgrammar==0.2.1+flagos.hcu1: the explicit wheel is not installed",
            completed.stderr,
        )


class RemoteCandidateTests(unittest.TestCase):
    """The pushed candidate is verified against the registry object."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_script("verify_remote_blobs")
        cls.script = SCRIPTS / "verify_remote_candidate.sh"

    def registry(self):
        """A fake Harbor on localhost: token endpoint, manifests and blobs.
        Returns (host, state); state holds the manifests and blobs."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        state = {"manifests": {}, "blobs": {}}

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/service/token"):
                    self.reply(200, json.dumps({"token": "fake"}).encode())
                elif "/manifests/" in self.path or "/blobs/" in self.path:
                    digest = self.path.rsplit("/", 1)[-1]
                    table = (
                        state["manifests"]
                        if "/manifests/" in self.path
                        else state["blobs"]
                    )
                    if digest in table:
                        self.reply(200, table[digest])
                    else:
                        self.reply(404, b'{"errors":[]}')
                else:
                    self.reply(404, b"")

            def do_HEAD(self):
                self.do_GET()

            def log_message(self, *arguments):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"127.0.0.1:{server.server_address[1]}", state

    @staticmethod
    def image(state, layer_count=2):
        """Store an image (config + layers) and its index in the registry;
        return (index digest, config digest, layer digests, diff_ids)."""
        blobs = state["blobs"]
        layers, digests, diff_ids = [], [], []
        for number in range(layer_count):
            content = f"layer-{number}".encode()
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            blobs[digest] = content
            layers.append({"digest": digest})
            digests.append(digest)
            diff_ids.append("sha256:" + hashlib.sha256(content).hexdigest())
        config = {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"diff_ids": diff_ids, "type": "layers"},
            "config": {"Labels": {"org.flagos.ci.release-ready": "false"}},
        }
        config_bytes = json.dumps(config).encode()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        blobs[config_digest] = config_bytes
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": config_digest},
            "layers": layers,
        }
        manifest_digest = (
            "sha256:" + hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        )
        state["manifests"][manifest_digest] = json.dumps(manifest).encode()
        index = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": manifest_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
        index_digest = (
            "sha256:" + hashlib.sha256(json.dumps(index).encode()).hexdigest()
        )
        state["manifests"][index_digest] = json.dumps(index).encode()
        return index_digest, config_digest, manifest_digest, digests, diff_ids

    def test_every_blob_of_the_remote_manifest_is_confirmed(self):
        host, state = self.registry()
        index, config, manifest, layers, diff_ids = self.image(state)
        record = self.module.inspect(host, "proj/repo", index, scheme="http")
        self.assertEqual(record["digest"], index)
        self.assertEqual(
            record["media_type"], "application/vnd.oci.image.index.v1+json"
        )
        self.assertEqual(
            record["manifests"],
            [
                {
                    "digest": manifest,
                    "platform": "linux/amd64",
                    "config_digest": config,
                    "layers": 2,
                }
            ],
        )
        self.assertEqual(record["linux_amd64"]["manifest_digest"], manifest)
        self.assertEqual(record["linux_amd64"]["config_digest"], config)
        self.assertEqual(record["linux_amd64"]["diff_ids"], diff_ids)
        self.assertEqual(
            record["linux_amd64"]["config"]["config"]["Labels"],
            {"org.flagos.ci.release-ready": "false"},
        )
        self.assertEqual(sorted(record["blobs_checked"]), sorted([config, *layers]))
        self.assertEqual(record["blobs_missing"], [])

    def test_a_missing_parent_layer_is_a_hard_failure(self):
        host, state = self.registry()
        index, _, _, layers, _ = self.image(state)
        del state["blobs"][layers[0]]
        with self.assertRaisesRegex(self.module.RegistryError, "not in the registry"):
            self.module.inspect(host, "proj/repo", index, scheme="http")

    def test_an_unknown_object_is_a_hard_failure(self):
        host, state = self.registry()
        with self.assertRaisesRegex(self.module.RegistryError, "HTTP 404"):
            self.module.inspect(host, "proj/repo", "sha256:" + "0" * 64, scheme="http")

    def test_the_cli_refuses_anything_but_a_digest(self):
        self.assertEqual(self.module.main(["harbor.baai.ac.cn", "a/b", "latest"]), 1)
        self.assertEqual(self.module.main(["harbor.baai.ac.cn", "a/b"]), 2)

    def test_the_candidate_tag_is_the_only_accepted_reference(self):
        text = self.script.read_text()
        pattern = re.search(r"CANDIDATE_TAG_PATTERN='([^']+)'", text).group(1)
        accepted = (
            "harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-hygon-2w-offline-"
            "abc1234-20260101T000000Z"
        )
        self.assertIsNotNone(re.match(pattern, accepted))
        for refused in (
            accepted.replace("0.5.18-hygon-2w-offline-", "0.5.18-hygon-two-wheel-"),
            accepted.replace("flagos-dev/sglang-plugin-fl", "flagos-dev/other"),
            accepted.replace("abc1234", "zzzzzzz"),
            accepted + "x",
            "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.04:2.3.0",
        ):
            self.assertIsNone(re.match(pattern, refused), refused)
        # the script reads the registry object, proves every blob, pulls by
        # digest and compares with the accepted local build
        self.assertIn('docker buildx imagetools inspect "$tag"', text)
        self.assertIn('docker pull --platform linux/amd64 "$repository@$digest"', text)
        self.assertIn('verify_remote_blobs.py" "$host" "$name" "$digest"', text)
        self.assertIn("diff_ids", text)
        self.assertIn('"Labels"', text)
        self.assertNotIn("docker push", text)


if __name__ == "__main__":
    unittest.main()
