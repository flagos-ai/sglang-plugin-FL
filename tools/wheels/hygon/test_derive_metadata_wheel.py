"""Metadata-only dependency-wheel derivation tests (no GPU or image needed).

The synthetic cases exercise every rejection gate.  The real xgrammar case
binds the tool to the SHA-pinned artifact whose Linux/x86_64 Triton metadata
causes the Hygon resolver conflict.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_metadata_wheel as derive_mod
import verify_metadata_wheel as verify_mod

INPUT_VERSION = "0.2.1"
DERIVED_VERSION = "0.2.1+flagos.hcu1"
DROP_LINE = 'triton; platform_system == "Linux" and platform_machine == "x86_64"'
INPUT_NAME = "xgrammar-0.2.1-cp310-cp310-manylinux_2_28_x86_64.whl"
DERIVED_NAME = "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_28_x86_64.whl"
REAL_INPUT_NAME = (
    "xgrammar-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
REAL_DERIVED_NAME = (
    "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
REAL_INPUT_SHA256 = "5b80b8060fcf6c94324f563e1d4778029b21d35c556dca5bbf49428d307eca37"
REAL_INPUT_URL = (
    "https://mirrors.cloud.tencent.com/pypi/packages/1a/62/"
    "4fba98477159f4f79b2baa33f891708df3585ff6a1ac77b4c64895ec462b/" + REAL_INPUT_NAME
)
SOURCE_COMMIT = "0" * 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_bytes(files: dict[str, bytes], record_name: str) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name in sorted(files | {record_name: b""}):
        if name == record_name:
            writer.writerow([name, "", ""])
            continue
        data = files[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([name, f"sha256={digest.decode()}", str(len(data))])
    return buffer.getvalue().encode()


def metadata(
    *,
    name: str = "xgrammar",
    version: str = INPUT_VERSION,
    drop_line: str = DROP_LINE,
    drop_count: int = 1,
) -> bytes:
    lines = [
        "Metadata-Version: 2.2",
        f"Name: {name}",
        f"Version: {version}",
        "Requires-Dist: numpy",
    ]
    lines.extend([f"Requires-Dist: {drop_line}"] * drop_count)
    return ("\n".join(lines) + "\n\nSynthetic body.\n").encode()


def build_synthetic_wheel(
    path: Path,
    *,
    metadata_bytes: bytes | None = None,
    add_duplicate_drop: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dist = "xgrammar-0.2.1.dist-info"
    files = {
        "xgrammar/__init__.py": b'__version__ = "0.2.1"\n',
        "xgrammar/config,K=64.json": b'{"K": 64}\n',
        "xgrammar/native/libxgrammar.so": b"\x7fELF-synthetic-native",
        f"{dist}/METADATA": metadata_bytes
        or metadata(drop_count=2 if add_duplicate_drop else 1),
        f"{dist}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: false\n"
            b"Tag: cp310-cp310-manylinux_2_28_x86_64\n"
        ),
        f"{dist}/licenses/LICENSE": b"synthetic license\n",
    }
    record_name = f"{dist}/RECORD"
    files[record_name] = record_bytes(files, record_name)
    directories = [
        "xgrammar/",
        "xgrammar/native/",
        f"{dist}/",
        f"{dist}/licenses/",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for index, name in enumerate(directories + list(files)):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 1, 2, 4))
            info.create_system = 3
            info.external_attr = (0o40755 if name.endswith("/") else 0o100644) << 16
            info.compress_type = (
                zipfile.ZIP_STORED if index % 2 == 0 else zipfile.ZIP_DEFLATED
            )
            archive.writestr(
                info,
                b"" if name.endswith("/") else files[name],
                compress_type=info.compress_type,
                compresslevel=9 if info.compress_type == zipfile.ZIP_DEFLATED else None,
            )
    return path


def derive_synthetic(input_wheel: Path, output_dir: Path) -> tuple[dict, Path]:
    result = derive_mod.derive(
        input_wheel,
        sha256(input_wheel),
        [DROP_LINE],
        "triton",
        DERIVED_VERSION,
        output_dir,
        SOURCE_COMMIT,
    )
    return result, output_dir / DERIVED_NAME


@pytest.fixture()
def synthetic(tmp_path):
    input_wheel = build_synthetic_wheel(tmp_path / INPUT_NAME)
    result, derived = derive_synthetic(input_wheel, tmp_path / "derived")
    assert derived.is_file()
    return input_wheel, derived, result


def repack(
    wheel: Path,
    destination: Path,
    *,
    mutations: dict[str, bytes] | None = None,
    compression_mutation: str | None = None,
) -> Path:
    mutations = mutations or {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(destination, "w") as out:
        for old in source.infolist():
            info = zipfile.ZipInfo(old.filename, date_time=old.date_time)
            info.create_system = old.create_system
            info.external_attr = old.external_attr
            info.compress_type = old.compress_type
            if old.filename == compression_mutation:
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if old.compress_type != zipfile.ZIP_STORED
                    else zipfile.ZIP_DEFLATED
                )
            data = mutations.get(old.filename, source.read(old))
            out.writestr(
                info,
                data,
                compress_type=info.compress_type,
                compresslevel=9 if info.compress_type == zipfile.ZIP_DEFLATED else None,
            )
    return destination


def verify(input_wheel: Path, derived: Path, **kwargs):
    return verify_mod.verify(
        input_wheel,
        derived,
        kwargs.pop("drop_lines", [DROP_LINE]),
        kwargs.pop("drop_distribution", "triton"),
        kwargs.pop("expected_input_sha256", sha256(input_wheel)),
        **kwargs,
    )


def test_wrong_input_sha256_is_rejected(tmp_path):
    wheel = build_synthetic_wheel(tmp_path / INPUT_NAME)
    with pytest.raises(derive_mod.DeriveFailure, match="SHA256"):
        derive_mod.derive(
            wheel,
            "0" * 64,
            [DROP_LINE],
            "triton",
            DERIVED_VERSION,
            tmp_path / "out",
            SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    ("bad_metadata", "message"),
    [
        (metadata(name="other"), "METADATA names"),
        (metadata(version="0.2.2"), "METADATA"),
    ],
)
def test_filename_metadata_identity_is_enforced(tmp_path, bad_metadata, message):
    wheel = build_synthetic_wheel(tmp_path / INPUT_NAME, metadata_bytes=bad_metadata)
    with pytest.raises(derive_mod.DeriveFailure, match=message):
        derive_synthetic(wheel, tmp_path / "out")


@pytest.mark.parametrize("version", [INPUT_VERSION, "0.2.1+hcu1", "0.2.2+flagos.hcu1"])
def test_derived_version_shape_is_enforced(tmp_path, version):
    wheel = build_synthetic_wheel(tmp_path / INPUT_NAME)
    with pytest.raises(derive_mod.DeriveFailure, match="derived version"):
        derive_mod.derive(
            wheel,
            sha256(wheel),
            [DROP_LINE],
            "triton",
            version,
            tmp_path / "out",
            SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    ("drop_line", "drop_distribution", "message"),
    [
        ("not a valid requirement ; ;", "triton", "valid requirement"),
        ("numpy", "triton", "names"),
        ('triton; platform_system == "Windows"', "triton", "inactive"),
        (DROP_LINE, "standalone-triton", "names"),
    ],
)
def test_drop_contract_is_enforced(tmp_path, drop_line, drop_distribution, message):
    wheel = build_synthetic_wheel(tmp_path / INPUT_NAME)
    with pytest.raises(derive_mod.DeriveFailure, match=message):
        derive_mod.derive(
            wheel,
            sha256(wheel),
            [drop_line],
            drop_distribution,
            DERIVED_VERSION,
            tmp_path / "out",
            SOURCE_COMMIT,
        )


def test_drop_line_must_exist_exactly_once(tmp_path):
    missing = build_synthetic_wheel(
        tmp_path / "missing" / INPUT_NAME,
        metadata_bytes=metadata(drop_line="packaging"),
    )
    with pytest.raises(derive_mod.DeriveFailure, match="exactly one"):
        derive_synthetic(missing, tmp_path / "missing-out")

    duplicate = build_synthetic_wheel(
        tmp_path / "duplicate" / INPUT_NAME, add_duplicate_drop=True
    )
    with pytest.raises(derive_mod.DeriveFailure, match="exactly one"):
        derive_synthetic(duplicate, tmp_path / "duplicate-out")


def test_synthetic_success_contract(synthetic):
    input_wheel, derived, provenance = synthetic
    report = verify(input_wheel, derived)
    assert report["record_standard_csv_lf"] is True
    assert report["zip_attributes_identical"] is True
    assert report["native_payload"]["byte_identical"] is True
    assert report["remaining_requires_dist"] == []
    assert provenance["derived_sha256"] == sha256(derived)
    assert provenance["bound_files_manifest"]

    with zipfile.ZipFile(input_wheel) as original, zipfile.ZipFile(derived) as new:
        old_dist = verify_mod.wheel_dist_info(original)
        new_dist = verify_mod.wheel_dist_info(new)
        old_infos = original.infolist()
        new_infos = new.infolist()
        assert [
            verify_mod.rename_member(i.filename, old_dist, new_dist) for i in old_infos
        ] == [i.filename for i in new_infos]
        assert [i.compress_type for i in old_infos] == [
            i.compress_type for i in new_infos
        ]
        record = new.read(f"{new_dist}/RECORD")
        assert b"\r" not in record
        rows = verify_mod.read_record(record)
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        assert buffer.getvalue().encode() == record


def test_two_derivations_are_byte_identical(tmp_path):
    wheel = build_synthetic_wheel(tmp_path / INPUT_NAME)
    result_a, derived_a = derive_synthetic(wheel, tmp_path / "a")
    result_b, derived_b = derive_synthetic(wheel, tmp_path / "b")
    assert derived_a.read_bytes() == derived_b.read_bytes()
    assert result_a == result_b


def test_verifier_rejects_payload_and_native_drift(synthetic, tmp_path):
    input_wheel, derived, _ = synthetic
    package = repack(
        derived,
        tmp_path / "payload" / derived.name,
        mutations={"xgrammar/__init__.py": b"payload drift\n"},
    )
    with pytest.raises(verify_mod.GateFailure, match="allowlist"):
        verify(input_wheel, package)

    native = repack(
        derived,
        tmp_path / "native" / derived.name,
        mutations={"xgrammar/native/libxgrammar.so": b"native drift"},
    )
    with pytest.raises(verify_mod.GateFailure, match="allowlist|native"):
        verify(input_wheel, native)


def test_verifier_rejects_zip_attribute_drift(synthetic, tmp_path):
    input_wheel, derived, _ = synthetic
    tampered = repack(
        derived,
        tmp_path / "attrs" / derived.name,
        compression_mutation="xgrammar/__init__.py",
    )
    with pytest.raises(verify_mod.GateFailure, match="attributes drifted"):
        verify(input_wheel, tampered)


def test_verifier_rejects_metadata_drift(synthetic, tmp_path):
    input_wheel, derived, _ = synthetic
    dist = f"xgrammar-{DERIVED_VERSION}.dist-info"
    with zipfile.ZipFile(derived) as archive:
        changed = archive.read(f"{dist}/METADATA").replace(
            b"Requires-Dist: numpy", b"Requires-Dist: packaging"
        )
    tampered = repack(
        derived,
        tmp_path / "metadata" / derived.name,
        mutations={f"{dist}/METADATA": changed},
    )
    with pytest.raises(verify_mod.GateFailure, match="Requires-Dist"):
        verify(input_wheel, tampered)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(b"\r\n", "CR line endings"), (b"bad,row,with,too,many,fields\n", "three CSV")],
)
def test_verifier_rejects_noncanonical_record(
    synthetic, tmp_path, replacement, message
):
    input_wheel, derived, _ = synthetic
    dist = f"xgrammar-{DERIVED_VERSION}.dist-info"
    record_name = f"{dist}/RECORD"
    with zipfile.ZipFile(derived) as archive:
        record = archive.read(record_name)
    changed = (
        record.replace(b"\n", replacement, 1) if replacement == b"\r\n" else replacement
    )
    tampered = repack(
        derived,
        tmp_path / message.replace(" ", "-") / derived.name,
        mutations={record_name: changed},
    )
    with pytest.raises(verify_mod.GateFailure, match=message):
        verify(input_wheel, tampered)


def test_verifier_rejects_wrong_expected_hashes(synthetic):
    input_wheel, derived, _ = synthetic
    with pytest.raises(verify_mod.GateFailure, match="input wheel SHA256"):
        verify(input_wheel, derived, expected_input_sha256="0" * 64)
    with pytest.raises(verify_mod.GateFailure, match="derived wheel SHA256"):
        verify(input_wheel, derived, expected_derived_sha256="0" * 64)


@pytest.fixture(scope="session")
def real_xgrammar_input() -> Path:
    override = os.environ.get("XGRAMMAR_DERIVED_INPUT_WHEEL")
    candidates = [Path(override)] if override else []
    cache = Path.home() / ".cache" / "sglang-fl-hygon-derived"
    candidates.append(cache / REAL_INPUT_NAME)
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == REAL_INPUT_SHA256:
            return candidate
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / REAL_INPUT_NAME
    completed = subprocess.run(
        ["curl", "-fL", "--retry", "3", REAL_INPUT_URL, "-o", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    assert sha256(target) == REAL_INPUT_SHA256
    return target


def test_real_xgrammar_exact_delta_and_native_identity(real_xgrammar_input, tmp_path):
    first = derive_mod.derive(
        real_xgrammar_input,
        REAL_INPUT_SHA256,
        [DROP_LINE],
        "triton",
        DERIVED_VERSION,
        tmp_path / "first",
        SOURCE_COMMIT,
    )
    second = derive_mod.derive(
        real_xgrammar_input,
        REAL_INPUT_SHA256,
        [DROP_LINE],
        "triton",
        DERIVED_VERSION,
        tmp_path / "second",
        SOURCE_COMMIT,
    )
    derived = tmp_path / "first" / REAL_DERIVED_NAME
    duplicate = tmp_path / "second" / REAL_DERIVED_NAME
    assert derived.read_bytes() == duplicate.read_bytes()
    assert first == second

    report = verify(
        real_xgrammar_input,
        derived,
        expected_input_sha256=REAL_INPUT_SHA256,
        expected_derived_sha256=sha256(derived),
    )
    assert len(report["native_payload"]["files"]) == 11
    assert report["native_payload"]["byte_identical"] is True
    assert report["payload_byte_identical"] is True
    assert report["remaining_requires_dist"] == []

    with (
        zipfile.ZipFile(real_xgrammar_input) as original,
        zipfile.ZipFile(derived) as corrected,
    ):
        old_dist = verify_mod.wheel_dist_info(original)
        new_dist = verify_mod.wheel_dist_info(corrected)
        old_meta = original.read(f"{old_dist}/METADATA")
        new_meta = corrected.read(f"{new_dist}/METADATA")
        parsed = BytesParser().parsebytes(new_meta)
        assert parsed["Version"] == DERIVED_VERSION
        assert DROP_LINE not in parsed.get_all("Requires-Dist", ())
        expected = old_meta.replace(
            b"Version: 0.2.1", b"Version: 0.2.1+flagos.hcu1", 1
        ).replace(f"Requires-Dist: {DROP_LINE}\n".encode(), b"", 1)
        assert new_meta == expected


def test_static_csv_guard_uses_no_right_split():
    tools = Path(__file__).resolve().parent
    derive_source = (tools / "derive_metadata_wheel.py").read_text(encoding="utf-8")
    verifier_source = (tools / "verify_metadata_wheel.py").read_text(encoding="utf-8")
    assert "rebuild_record(" in derive_source
    assert "csv.writer(" in verifier_source
    assert 'lineterminator="\\n"' in verifier_source
    assert ".rsplit(" not in verifier_source
