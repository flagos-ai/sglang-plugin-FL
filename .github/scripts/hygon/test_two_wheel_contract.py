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

    def test_uv_install_contract_and_dry_run_mirror_it(self):
        """The single real install and the pre-build dry-run must agree.

        --index-strategy first-index was measured insufficient on the real
        runtime (the Hygon index has xgrammar but not ==0.2.1, and
        first-index stops at the name instead of falling through to PyPI);
        the install must use unsafe-first-match with the full flag
        contract, and build_two_wheel.sh's dry-run.sh must carry exactly
        the same arguments plus --dry-run (it plan-gates the build against
        the same resolved base digest).
        """
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
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
        self.assertIn('--volume "$context/artifacts":/tmp/artifacts:ro', build_wrapper)
        self.assertIn("cat /plan/uv-dry-run.txt >&2", heredoc.group(1))

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

    def test_head_and_label_are_reasserted_after_the_build(self):
        wrapper = (DOCKER / "build_two_wheel.sh").read_text()
        self.assertIn("HEAD moved during the build", wrapper)
        self.assertIn("source-commit label mismatch", wrapper)
        self.assertIn("org.flagos.ci.xgrammar-wheel-sha256", wrapper)
        self.assertIn('--xgrammar-wheel-sha256 "$xgrammar_wheel_sha256"', wrapper)
        self.assertIn('--xgrammar-provenance "$xgrammar_provenance"', wrapper)
        verifier = (SCRIPTS / "verify_image.sh").read_text()
        self.assertIn('provenance["source_commit"] == sys.argv[4]', verifier)
        self.assertIn('!= "<no value>"', verifier)


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

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.context, ignore_errors=True)


class ReviewerCommentContractTests(unittest.TestCase):
    """One block per PR #180 review comment this branch answers.

    Each test pins the repository state that answers the comment, so a
    later change cannot silently bring the rejected design back.
    """

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

    def test_r4627527485_the_datasets_pin_is_wheel_metadata(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn('"datasets==5.2.0"', pyproject)
        self.assertFalse((DOCKER / "requirements.txt").exists())
        containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
        self.assertNotIn("-r /tmp", containerfile)
        self.assertNotIn("requirements.txt", containerfile)


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
        cls.tmp = tempfile.mkdtemp(prefix="pub-manifest-test.")
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


if __name__ == "__main__":
    unittest.main()
