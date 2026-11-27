"""Derived SGLang wheel: provenance and payload tests (no GPU, no image).

Coverage follows the derived-wheel prototype contract:

Error branches (minimal SYNTHETIC wheel + a synthetic git patch):
  1. wrong input SHA256 is rejected;
  2. wrong distribution name / version is rejected;
  3. derived version equal to the input (or not this round's exact value)
     is rejected;
  4. an inapplicable patch is rejected;
  5. an unexpected change to a non-allowlist file is rejected by the gates;
  6. any native payload byte change is rejected;
  7. METADATA/filename/dist-info version inconsistency is rejected;
  8. a wrong RECORD hash is rejected;
  9. Requires-Dist / entry point drift is rejected.

Success paths (the SHA-pinned REAL SGLang wheel):
  10. two independent derivations are byte-identical;
  11. the forward patch no longer applies, the reverse patch checks, and
      the patched files equal original+patch;
  12. staging consumes the derived wheel and carries no Docker patch;
  13. the containerfile has no git apply / patch COPY / patch label;
  14. the plugin [hygon] extra selects exactly the derived version;
  15. installing the derived wheel yields the derived distribution
      version.

The real input wheel is fetched once into a user cache directory (or taken
from $SGLANG_DERIVED_INPUT_WHEEL) and its SHA256 is verified; success-path
tests FAIL (not skip) when it cannot be secured.
"""

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_sglang_wheel as derive_mod
import verify_derived_sglang_wheel as verify_mod

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools" / "wheels" / "hygon"
DOCKER = REPO_ROOT / "docker" / "hygon"

INPUT_NAME = "sglang-0.5.18+flagos-cp310-cp310-linux_x86_64.whl"
INPUT_URL = (
    "https://resource.flagos.net/repository/flagos-pypi-hygon/packages/"
    "sglang/0.5.18+flagos/" + INPUT_NAME
)
INPUT_SHA256 = "1fe8a560353fd15aea7194869f1fd50e1fdbedd8fc1074db52b9a13028756800"
DERIVED_VERSION = "0.5.18+flagos.hcu1"
DERIVED_NAME = f"sglang-{DERIVED_VERSION}-cp310-cp310-linux_x86_64.whl"
SOURCE_COMMIT = "0" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# real input wheel


@pytest.fixture(scope="session")
def real_input_wheel() -> Path:
    override = Path(os.environ.get("SGLANG_DERIVED_INPUT_WHEEL", ""))
    candidates = [override] if str(override) else []
    cache = Path.home() / ".cache" / "sglang-fl-hygon-derived"
    candidates.append(cache / INPUT_NAME)
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == INPUT_SHA256:
            return candidate
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / INPUT_NAME
    completed = None
    for attempt in range(3):
        completed = subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                INPUT_URL,
                "-o",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            break
    assert completed is not None and completed.returncode == 0, (
        completed.stderr if completed else "curl never ran"
    )[-500:]
    actual = _sha256(target)
    assert actual == INPUT_SHA256, (
        f"downloaded wheel SHA256 {actual} != pinned {INPUT_SHA256}"
    )
    return target


@pytest.fixture(scope="session")
def real_derived(real_input_wheel, tmp_path_factory) -> Path:
    """Derive the REAL wheel once; all success-path tests reuse it."""
    out_dir = tmp_path_factory.mktemp("real-derived")
    provenance = derive_mod.derive(
        real_input_wheel,
        INPUT_SHA256,
        TOOLS_DIR / "sglang-v0.5.18-hcu.patch",
        out_dir,
        DERIVED_VERSION,
        SOURCE_COMMIT,
    )
    wheel = out_dir / DERIVED_NAME
    assert wheel.is_file()
    assert provenance["derived_sha256"] == _sha256(wheel)
    return wheel


# --------------------------------------------------------------------------
# synthetic wheel factory


SYNTH_FILES = {
    "sglang/__init__.py": b"__all__ = []\n",
    "sglang/_version.py": b"__version__ = version = '0.5.18'\n",
    "sglang/srt/layers/sampler.py": b"sampler line 1\nsampler line 2\nsampler line 3\n",
    "sglang/srt/layers/activation.py": b"activation A\nactivation B\n",
    "sglang/srt/bystander.py": b"bystander content\n",
    # a comma-in-path member, mirroring the real wheel's triton config
    # JSONs (lora_expand,K=1024,...json): RECORD must quote it
    "sglang/kernels/ops/config,K=64,S=1.json": b'{"K": 64}\n',
    "sglang/native/payload.so": b"\x7fELF-fake-native-payload",
}
SYNTH_METADATA = (
    "Metadata-Version: 2.1\n"
    "Name: sglang\n"
    "Version: 0.5.18+flagos\n"
    "Requires-Dist: packaging\n"
    "\n"
)
SYNTH_PATCH = """diff --git a/python/sglang/srt/layers/sampler.py b/python/sglang/srt/layers/sampler.py
--- a/python/sglang/srt/layers/sampler.py
+++ b/python/sglang/srt/layers/sampler.py
@@ -1,3 +1,3 @@
 sampler line 1
-sampler line 2
+sampler line 2 (HCU)
 sampler line 3
diff --git a/python/sglang/srt/layers/activation.py b/python/sglang/srt/layers/activation.py
--- a/python/sglang/srt/layers/activation.py
+++ b/python/sglang/srt/layers/activation.py
@@ -1,2 +1,2 @@
-activation A
+activation A (HCU)
 activation B
"""


def synth_record(root: Path) -> bytes:
    """Standard-CSV RECORD via csv.writer (comma paths get quoted)."""
    import csv
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for rel in sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()):
        if rel == "sglang-0.5.18+flagos.dist-info/RECORD":
            writer.writerow([rel, "", ""])
            continue
        data = (root / rel).read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        writer.writerow([rel, f"sha256={digest}", str(len(data))])
    return buffer.getvalue().encode("utf-8")


def build_synth_wheel(path: Path, mutations=None) -> Path:
    """A minimal sglang 0.5.18+flagos wheel the synthetic patch applies to.

    ``mutations`` may override ANY member, including dist-info files; the
    RECORD is always rebuilt last so it matches the final tree.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, data in SYNTH_FILES.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        dist = root / "sglang-0.5.18+flagos.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(SYNTH_METADATA, encoding="utf-8")
        (dist / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: synthetic\n"
            "Root-Is-Purelib: false\nTag: cp310-cp310-linux_x86_64\n",
            encoding="utf-8",
        )
        (dist / "entry_points.txt").write_text(
            "[console_scripts]\nsglang = sglang.cli.main:main\n", encoding="utf-8"
        )
        (dist / "top_level.txt").write_text("sglang\n", encoding="utf-8")
        for rel, data in (mutations or {}).items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        (dist / "RECORD").write_bytes(synth_record(root))
        with zipfile.ZipFile(path, "w") as archive:
            for rel in sorted(
                str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
            ):
                archive.write(root / rel, rel)
    return path


@pytest.fixture()
def synth_env(tmp_path):
    """Synthetic wheel + patch + output dir, derived successfully."""
    wheel = build_synth_wheel(tmp_path / INPUT_NAME)
    patch = tmp_path / "synth.patch"
    patch.write_text(SYNTH_PATCH, encoding="utf-8")
    out_dir = tmp_path / "derived"
    provenance = derive_mod.derive(
        wheel, _sha256(wheel), patch, out_dir, DERIVED_VERSION, SOURCE_COMMIT
    )
    derived = out_dir / DERIVED_NAME
    assert derived.is_file() and provenance["derived_sha256"]
    return {"input": wheel, "patch": patch, "derived": derived}


def repack_tampered(derived_wheel: Path, mutations: dict, rebuild_record: bool) -> Path:
    """Unpack a derived wheel, mutate files, optionally fix RECORD, repack."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(derived_wheel) as archive:
            archive.extractall(root)
        for rel, data in mutations.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        if rebuild_record:
            dist = root / f"sglang-{DERIVED_VERSION}.dist-info"
            (dist / "RECORD").write_bytes(synth_record(root))
        # keep the legal wheel filename; separate via a directory
        out_dir = derived_wheel.parent / "tampered"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / derived_wheel.name
        with zipfile.ZipFile(out, "w") as archive:
            for rel in sorted(
                str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
            ):
                archive.write(root / rel, rel)
        return out


# --------------------------------------------------------------------------
# 1-4: derive error branches (synthetic)


def test_wrong_input_sha256_is_rejected(tmp_path):
    wheel = build_synth_wheel(tmp_path / INPUT_NAME)
    patch = tmp_path / "synth.patch"
    patch.write_text(SYNTH_PATCH, encoding="utf-8")
    with pytest.raises(derive_mod.DeriveFailure, match="SHA256"):
        derive_mod.derive(
            wheel, "0" * 64, patch, tmp_path / "out", DERIVED_VERSION, SOURCE_COMMIT
        )


def test_wrong_distribution_name_or_version_is_rejected(tmp_path):
    patch = tmp_path / "synth.patch"
    patch.write_text(SYNTH_PATCH, encoding="utf-8")
    wrong_name = build_synth_wheel(
        tmp_path / f"notsglang-{INPUT_NAME.split('-', 1)[1]}",
        mutations={
            "sglang-0.5.18+flagos.dist-info/METADATA": SYNTH_METADATA.replace(
                "Name: sglang\n", "Name: notsglang\n"
            )
        },
    )
    with pytest.raises(derive_mod.DeriveFailure, match="distribution"):
        derive_mod.derive(
            wrong_name,
            _sha256(wrong_name),
            patch,
            tmp_path / "out",
            DERIVED_VERSION,
            SOURCE_COMMIT,
        )
    wrong_version = build_synth_wheel(
        tmp_path / "sglang-0.5.19+flagos-cp310-cp310-linux_x86_64.whl",
        mutations={
            "sglang-0.5.18+flagos.dist-info/METADATA": SYNTH_METADATA.replace(
                "Version: 0.5.18+flagos", "Version: 0.5.19+flagos"
            )
        },
    )
    with pytest.raises(derive_mod.DeriveFailure, match="version"):
        derive_mod.derive(
            wrong_version,
            _sha256(wrong_version),
            patch,
            tmp_path / "out",
            DERIVED_VERSION,
            SOURCE_COMMIT,
        )


def test_derived_version_must_differ_and_be_this_rounds_value(tmp_path):
    wheel = build_synth_wheel(tmp_path / INPUT_NAME)
    patch = tmp_path / "synth.patch"
    patch.write_text(SYNTH_PATCH, encoding="utf-8")
    sha = _sha256(wheel)
    with pytest.raises(derive_mod.DeriveFailure, match="differ"):
        derive_mod.derive(
            wheel, sha, patch, tmp_path / "o1", "0.5.18+flagos", SOURCE_COMMIT
        )
    with pytest.raises(derive_mod.DeriveFailure, match="exactly"):
        derive_mod.derive(
            wheel, sha, patch, tmp_path / "o2", "0.5.18+flagos.hcu2", SOURCE_COMMIT
        )


def test_inapplicable_patch_is_rejected(tmp_path):
    wheel = build_synth_wheel(
        tmp_path / INPUT_NAME,
        mutations={
            "sglang/srt/layers/sampler.py": b"totally different content\n",
        },
    )
    patch = tmp_path / "synth.patch"
    patch.write_text(SYNTH_PATCH, encoding="utf-8")
    with pytest.raises(derive_mod.DeriveFailure, match="--check failed"):
        derive_mod.derive(
            wheel,
            _sha256(wheel),
            patch,
            tmp_path / "out",
            DERIVED_VERSION,
            SOURCE_COMMIT,
        )


# --------------------------------------------------------------------------
# 5-9: strict-gate rejections on tampered derivations (synthetic)


def test_unexpected_file_change_is_rejected(synth_env):
    tampered = repack_tampered(
        synth_env["derived"],
        {"sglang/srt/bystander.py": b"smuggled change\n"},
        rebuild_record=True,
    )
    with pytest.raises(verify_mod.GateFailure, match="allowlist"):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


def test_native_payload_change_is_rejected(synth_env):
    tampered = repack_tampered(
        synth_env["derived"],
        {"sglang/native/payload.so": b"\x7fELF-tampered-payload"},
        rebuild_record=True,
    )
    with pytest.raises(verify_mod.GateFailure):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


def test_version_inconsistency_is_rejected(synth_env):
    tampered = repack_tampered(
        synth_env["derived"],
        {
            f"sglang-{DERIVED_VERSION}.dist-info/METADATA": b"Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.19\n"
            b"Requires-Dist: packaging\n\n"
        },
        rebuild_record=True,
    )
    with pytest.raises(verify_mod.GateFailure, match="version"):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


def test_record_hash_error_is_rejected(synth_env):
    # corrupt one RECORDED digest while the file content stays intact:
    # only the RECORD-validity gate can catch this (csv-aware surgery)
    import csv
    import io

    record_rel = f"sglang-{DERIVED_VERSION}.dist-info/RECORD"
    with zipfile.ZipFile(synth_env["derived"]) as archive:
        rows = [
            row
            for row in csv.reader(
                io.StringIO(archive.read(record_rel).decode("utf-8"), newline="")
            )
            if row
        ]
    for row in rows:
        if row[0] == "sglang/srt/bystander.py":
            row[1] = "sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    tampered = repack_tampered(
        synth_env["derived"],
        {record_rel: buffer.getvalue()},
        rebuild_record=False,
    )
    with pytest.raises(verify_mod.GateFailure, match="RECORD hash/size"):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


def test_unquoted_comma_record_row_is_rejected(synth_env):
    # REGRESSION (review round 1): a RECORD whose comma-in-path row is
    # written UNQUOTED is not standard CSV; the verifier must reject it
    # instead of parsing permissively from the right
    record_rel = f"sglang-{DERIVED_VERSION}.dist-info/RECORD"
    with zipfile.ZipFile(synth_env["derived"]) as archive:
        record = archive.read(record_rel).decode("utf-8")
    unquoted = []
    for line in record.splitlines():
        if line.startswith('"sglang/kernels/ops/config,K=64,S=1.json",'):
            # strip the quotes from the path field: 5 raw fields instead
            # of the required 3
            line = line[1:].split('",', 1)[0] + "," + line.split('",', 1)[1]
        unquoted.append(line)
    tampered = repack_tampered(
        synth_env["derived"],
        {record_rel: "\n".join(unquoted) + "\n"},
        rebuild_record=False,
    )
    with pytest.raises(verify_mod.GateFailure, match="three CSV fields"):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


def test_requires_dist_drift_is_rejected(synth_env):
    metadata = (
        f"Metadata-Version: 2.1\nName: sglang\nVersion: {DERIVED_VERSION}\n"
        "Requires-Dist: packaging\nRequires-Dist: smuggled-dependency\n\n"
    ).encode()
    tampered = repack_tampered(
        synth_env["derived"],
        {f"sglang-{DERIVED_VERSION}.dist-info/METADATA": metadata},
        rebuild_record=True,
    )
    with pytest.raises(verify_mod.GateFailure, match="headers drifted"):
        verify_mod.verify(
            synth_env["input"],
            tampered,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )
    # entry-point drift hits the same byte-identity gate
    tampered_ep = repack_tampered(
        synth_env["derived"],
        {
            f"sglang-{DERIVED_VERSION}.dist-info/entry_points.txt": b"[console_scripts]\nsglang = smuggled:main\n"
        },
        rebuild_record=True,
    )
    with pytest.raises(verify_mod.GateFailure):
        verify_mod.verify(
            synth_env["input"],
            tampered_ep,
            synth_env["patch"],
            _sha256(synth_env["input"]),
        )


# --------------------------------------------------------------------------
# 10-11: success paths on the REAL wheel


def test_two_derivations_are_byte_identical(real_input_wheel, tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    patch = TOOLS_DIR / "sglang-v0.5.18-hcu.patch"
    provenance_a = derive_mod.derive(
        real_input_wheel, INPUT_SHA256, patch, out_a, DERIVED_VERSION, SOURCE_COMMIT
    )
    provenance_b = derive_mod.derive(
        real_input_wheel, INPUT_SHA256, patch, out_b, DERIVED_VERSION, SOURCE_COMMIT
    )
    wheel_a, wheel_b = out_a / DERIVED_NAME, out_b / DERIVED_NAME
    assert wheel_a.read_bytes() == wheel_b.read_bytes(), (
        "two independent derivations differ"
    )
    assert provenance_a == provenance_b


def test_forward_patch_not_reapplicable_reverse_applies(real_input_wheel, real_derived):
    patch = (TOOLS_DIR / "sglang-v0.5.18-hcu.patch").resolve()
    report = verify_mod.verify(
        real_input_wheel, real_derived, patch, INPUT_SHA256, _sha256(real_derived)
    )
    assert report["forward_patch_rejected"] is True
    assert report["reverse_patch_applies"] is True
    assert report["patched_files_reproduced"] == report["patch_files"]
    assert len(report["patch_files"]) == 7
    # direct structural proof, independent of the verifier's own calls
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(real_derived) as archive:
            archive.extractall(root)
        forward = verify_mod.run_git_apply(root, patch, "--check")
        assert forward.returncode != 0, "forward patch applied twice"
        reverse = verify_mod.run_git_apply(root, patch, "--reverse", "--check")
        assert reverse.returncode == 0, reverse.stderr


def test_real_record_is_standard_quoted_csv(real_derived, real_input_wheel):
    """Every RECORD row parses as CSV to exactly three fields, and the
    real wheel's comma-in-path members (536 triton config JSONs) are
    properly quoted and round-trip."""
    import csv
    import io

    record_rel = f"sglang-{DERIVED_VERSION}.dist-info/RECORD"
    with zipfile.ZipFile(real_derived) as archive:
        record_bytes = archive.read(record_rel)
        record = record_bytes.decode("utf-8")
    assert b"\r" not in record_bytes
    rows = [row for row in csv.reader(io.StringIO(record, newline="")) if row]
    assert rows, "RECORD is empty"
    bad = [row for row in rows if len(row) != 3]
    assert not bad, f"{len(bad)} rows are not three CSV fields, e.g. {bad[:2]}"
    comma_paths = [row[0] for row in rows if "," in row[0]]
    assert len(comma_paths) >= 500, (
        f"expected the real wheel's comma-path members, got {len(comma_paths)}"
    )
    # quoted on disk and consistent with the actual member set
    with zipfile.ZipFile(real_derived) as archive:
        names = set(archive.namelist())
    assert {row[0] for row in rows} == names
    quoted_lines = [line for line in record.splitlines() if line.startswith('"')]
    assert len(quoted_lines) == len(comma_paths)
    serialized = io.StringIO(newline="")
    csv.writer(serialized, lineterminator="\n").writerows(rows)
    assert serialized.getvalue().encode("utf-8") == record_bytes


def test_compression_kinds_are_preserved(real_derived, real_input_wheel):
    """The repack preserves every member's compression kind: the input
    wheel ships 6 STORED dist-info entries and 4307 DEFLATED ones."""
    from collections import Counter

    with zipfile.ZipFile(real_input_wheel) as archive:
        input_kinds = Counter(i.compress_type for i in archive.infolist())
    with zipfile.ZipFile(real_derived) as archive:
        derived_kinds = Counter(i.compress_type for i in archive.infolist())
    assert derived_kinds == input_kinds, (derived_kinds, input_kinds)
    assert input_kinds[zipfile.ZIP_STORED] == 6

    with (
        zipfile.ZipFile(real_input_wheel) as original,
        zipfile.ZipFile(real_derived) as derived,
    ):
        original_dist = verify_mod.wheel_dist_info(original)
        derived_dist = verify_mod.wheel_dist_info(derived)
        original_by_normalized_name = {
            (
                derived_dist + info.filename[len(original_dist) :]
                if info.filename == original_dist
                or info.filename.startswith(original_dist + "/")
                else info.filename
            ): info.compress_type
            for info in original.infolist()
        }
        derived_by_name = {
            info.filename: info.compress_type for info in derived.infolist()
        }
    assert derived_by_name == original_by_normalized_name


def test_record_and_verifier_use_standard_csv_implementation():
    derive_source = (TOOLS_DIR / "derive_sglang_wheel.py").read_text(encoding="utf-8")
    verifier_source = (TOOLS_DIR / "verify_derived_sglang_wheel.py").read_text(
        encoding="utf-8"
    )
    assert "csv.writer(" in derive_source
    assert 'lineterminator="\\n"' in derive_source
    assert ".rsplit(" not in verifier_source


def test_provenance_manifest_binds_patched_files(real_derived):
    """The provenance JSON carries per-file hashes of every patched file,
    read back from the finished wheel - the data the image acceptance
    binds the installed files against."""
    provenance = json.loads(
        real_derived.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    manifest = provenance["patched_files_manifest"]
    assert sorted(manifest) == sorted(
        verify_mod.parse_patch_files(TOOLS_DIR / "sglang-v0.5.18-hcu.patch")
    )
    with zipfile.ZipFile(real_derived) as archive:
        for rel, expected_hash in manifest.items():
            actual = hashlib.sha256(archive.read(rel)).hexdigest()
            assert actual == expected_hash, rel


# --------------------------------------------------------------------------
# 12-13: staging and containerfile contracts (repository text)


def test_staging_consumes_the_derived_wheel_and_carries_no_patch():
    staging = (DOCKER / "stage_wheels.sh").read_text()
    assert "tools/wheels/hygon/derive_sglang_wheel.py" in staging
    assert f'DERIVED_SGLANG_VERSION="{DERIVED_VERSION}"' in staging
    assert (
        'DERIVED_SGLANG_WHEEL_NAME="sglang-${DERIVED_SGLANG_VERSION}'
        '-cp310-cp310-linux_x86_64.whl"' in staging
    )
    assert '--derived-version "$DERIVED_SGLANG_VERSION"' in staging
    assert (
        'cp "$derived_wheel" "$context/artifacts/$DERIVED_SGLANG_WHEEL_NAME"' in staging
    )
    assert "hcu-patch:      sha256=" in staging  # reproducibility record
    # the original wheel is a derive INPUT only: it must not be staged
    assert 'cp "$sglang_wheel" "$context' not in staging
    # defense in depth against the patch leaking into the context
    assert "HCU patch leaked into the build context" in staging


def test_containerfile_has_no_patch_machinery():
    containerfile = (DOCKER / "runtime-0.5.18.containerfile").read_text()
    assert "git apply" not in containerfile
    assert "hcu.patch" not in containerfile
    assert DERIVED_NAME in containerfile
    build_wrapper = (DOCKER / "build_two_wheel.sh").read_text()
    assert "hcu-patch-sha256" not in build_wrapper
    assert "sglang-wheel-sha256" in build_wrapper
    # content binding: the wrapper hands the expected digest and the
    # provenance manifest to the acceptance; the verifier asserts the OCI
    # label equals the digest and checks the installed patched files
    # against the manifest
    assert '--sglang-wheel-sha256 "$derived_wheel_sha256"' in build_wrapper
    assert '--sglang-provenance "$derived_provenance"' in build_wrapper
    verify = (DOCKER / "scripts/verify_image.sh").read_text()
    assert "hcu-patch-sha256" not in verify
    assert "--sglang-wheel-sha256" in verify
    assert "patched_files_manifest" in verify
    assert DERIVED_VERSION in verify
    # the patch input moved to the tools directory; the empty-image
    # source-build line consumes it from there
    empty = (DOCKER / "empty-0.5.18.containerfile").read_text()
    assert "tools/wheels/hygon/sglang-v0.5.18-hcu.patch" in empty
    assert not (DOCKER / "sglang-v0.5.18-hcu.patch").exists()


# --------------------------------------------------------------------------
# 14: plugin extra selects the derived version


def test_hygon_extra_selects_the_derived_version(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    wheel = tmp_path / "sglang_fl-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(
            archive.read("sglang_fl-0.1.0.dist-info/METADATA")
        )
        names = archive.namelist()
    hygon = sorted(
        line.split(";")[0].strip()
        for line in metadata.get_all("Requires-Dist", ())
        if 'extra == "hygon"' in line
    )
    assert hygon == [
        "datasets==5.2.0",
        "flag-gems==5.4.0",
        f"sglang=={DERIVED_VERSION}",
    ]
    cuda = [
        line.split(";")[0].strip()
        for line in metadata.get_all("Requires-Dist", ())
        if 'extra == "cuda"' in line
    ]
    assert cuda == ["sglang==0.5.18"], cuda
    assert not any(
        line.split()[0].rstrip(";").lower() == "triton"
        for line in metadata.get_all("Requires-Dist", ())
    )
    assert not [n for n in names if n.startswith(("tests/", "test_"))]


# --------------------------------------------------------------------------
# 15: installed distribution version


def test_installed_distribution_version_is_derived(real_derived, tmp_path):
    """pip must accept the derived wheel and install it as hcu1.

    The wheel is cp310/linux_x86_64, so the install runs in pip's
    documented cross-environment mode (--platform/--abi/--python-version
    with --target): pip still validates the filename, tags and metadata
    and lays out the real dist-info tree. The in-image install contract
    (uv, real interpreter) is covered by the image acceptance gates.
    """
    target = tmp_path / "site"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--only-binary=:all:",
            "--implementation",
            "cp",
            "--python-version",
            "310",
            "--platform",
            "linux_x86_64",
            "--abi",
            "cp310",
            "--target",
            str(target),
            str(real_derived),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    import importlib.metadata as md

    installed = {
        dist.metadata["Name"].lower(): dist
        for dist in md.distributions(path=[str(target)])
    }
    assert "sglang" in installed, sorted(installed)
    assert installed["sglang"].version == DERIVED_VERSION
    dist_info = target / f"sglang-{DERIVED_VERSION}.dist-info"
    assert dist_info.is_dir(), sorted(p.name for p in target.iterdir())
    metadata = BytesParser().parsebytes((dist_info / "METADATA").read_bytes())
    assert metadata.get("Version") == DERIVED_VERSION
    # the package version source keeps the ORIGINAL wheel's value: it never
    # carried the +flagos local segment, so the derivation leaves it alone
    assert "__version__ = version = '0.5.18'" in (
        target / "sglang" / "_version.py"
    ).read_text(encoding="utf-8")
