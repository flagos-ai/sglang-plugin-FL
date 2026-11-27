#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare installed distributions with the wheels they were installed from.

Runs INSIDE the finished image (docker/hygon/scripts/verify_image.sh wraps
the docker run) with the wheels bind-mounted read-only. RECORD is
installer-owned, so it is not compared; every other wheel member must exist
at its installed location with exactly the wheel's bytes. That binds the
installed payload - native libraries included - to the artifact whose SHA256
the image label names; a same-version, different-content install cannot pass.

Wheels may relocate their members through a ``*.data`` directory; the
installed location follows the wheel scheme of the running interpreter:

  <name>-<version>.data/purelib|platlib   that site directory
  <name>-<version>.data/scripts          the scripts directory (the installer
                                         may replace the shebang line with a
                                         short shell trampoline; the body must
                                         be identical)
  <name>-<version>.data/data             sys.prefix
  <name>-<version>.data/headers          the include directory
  anything else                          purelib

Two modes:

  --wheel WHEEL [--own-tree PACKAGE]... [--min-members N]
      one staged wheel against its installed distribution. --own-tree
      additionally rejects any file under that top-level package that the
      wheel does not carry (``__pycache__`` excluded).

  --wheelhouse-manifest FILE --runtime FILE
      the whole closure of an offline-wheelhouse candidate: every
      distribution added on top of the Runtime baseline must be one of the
      three explicit wheels at its exact version or a wheelhouse wheel at
      its exact version, and every consumed wheelhouse wheel must be
      installed with its own content.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import sysconfig
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from packaging.utils import canonicalize_name


class WheelMismatch(AssertionError):
    """The installed distribution is not the wheel's content."""


def strip_script_prologue(data: bytes) -> bytes:
    """Drop what an installer may rewrite at the top of a console script: the
    shebang line, plus the short ``#!/bin/sh`` trampoline uv writes instead
    of a shebang that is too long for the environment. The body must follow
    unchanged."""
    body = data.split(b"\n", 1)[1] if data.startswith(b"#!") and b"\n" in data else b""
    lines = body.split(b"\n")
    if lines and lines[0].startswith(b"'''exec'"):
        closing = [i for i, line in enumerate(lines) if line == b"' '''"]
        if closing:
            del lines[: closing[0] + 1]
    return b"\n".join(lines)


def read_identity(archive: ZipFile, dist_info_stem: str) -> tuple[str, str]:
    metadata = archive.read(f"{dist_info_stem}.dist-info/METADATA").decode("utf-8")
    fields = {}
    for line in metadata.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep and key in ("Name", "Version"):
            fields[key] = value.strip()
    if "Name" not in fields or "Version" not in fields:
        raise WheelMismatch(f"{dist_info_stem}: METADATA lacks Name/Version")
    return canonicalize_name(fields["Name"]), fields["Version"]


def installed_path(member: str, dist_info_stem: str) -> Path:
    data_prefix = f"{dist_info_stem}.data/"
    if not member.startswith(data_prefix):
        return Path(sysconfig.get_path("purelib")) / member
    scheme, _, rest = member[len(data_prefix) :].partition("/")
    if scheme in ("purelib", "platlib"):
        return Path(sysconfig.get_path(scheme)) / rest
    if scheme == "scripts":
        return Path(sysconfig.get_path("scripts")) / rest
    if scheme == "data":
        return Path(sys.prefix) / rest
    if scheme == "headers":
        return Path(sysconfig.get_path("include")) / rest
    raise WheelMismatch(f"{member}: unsupported wheel data scheme {scheme!r}")


def compare_wheel(
    wheel: Path, own_tree: tuple[str, ...] = (), min_members: int = 0
) -> dict:
    """Every wheel member except RECORD must be installed byte-for-byte."""
    try:
        zip_file = ZipFile(wheel)
    except BadZipFile as failure:
        raise WheelMismatch(f"{wheel}: not a zip archive: {failure}")
    # the file may be mounted under any name (/wheel.whl): identity comes
    # from the archive's single dist-info directory and its METADATA
    dist_infos = sorted(
        {
            info.filename.split("/")[0]
            for info in zip_file.infolist()
            if info.filename.split("/")[0].endswith(".dist-info")
        }
    )
    if len(dist_infos) != 1:
        raise WheelMismatch(f"{wheel}: {len(dist_infos)} dist-info directories")
    dist_info_stem = dist_infos[0][: -len(".dist-info")]
    with zip_file:
        name, version = read_identity(zip_file, dist_info_stem)
        try:
            installed = md.distribution(name)
            installed_version = installed.version
        except md.PackageNotFoundError:
            installed_version = None
        if installed_version != version:
            raise WheelMismatch(
                f"{name}: installed {installed_version}, wheel {version}"
            )
        purelib = Path(sysconfig.get_path("purelib"))
        if not (purelib / f"{dist_info_stem}.dist-info").is_dir():
            raise WheelMismatch(
                f"{purelib / (dist_info_stem + '.dist-info')} is missing"
            )
        missing, changed, compared, expected = [], [], 0, set()
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            expected.add(info.filename)
            if info.filename.endswith(".dist-info/RECORD"):
                continue
            target = installed_path(info.filename, dist_info_stem)
            if not target.is_file():
                missing.append(info.filename)
            elif target.read_bytes() != zip_file.read(info) and not (
                info.filename.startswith(f"{dist_info_stem}.data/scripts/")
                and strip_script_prologue(target.read_bytes())
                == strip_script_prologue(zip_file.read(info))
            ):
                changed.append(info.filename)
            else:
                compared += 1
        if missing or changed:
            raise WheelMismatch(
                f"{name} is not the content of {wheel.name}: "
                f"{len(missing)} missing {missing[:3]}, {len(changed)} changed {changed[:3]}"
            )
    if compared < min_members:
        raise WheelMismatch(
            f"{name}: only {compared} members compared, fewer than {min_members}"
        )
    for tree in own_tree:
        extra = sorted(
            path.relative_to(purelib).as_posix()
            for path in (purelib / tree).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.relative_to(purelib).as_posix() not in expected
        )
        if extra:
            raise WheelMismatch(
                f"files under {tree}/ that are not in {wheel.name}: {extra[:5]}"
            )
    return {"compared": compared, "name": name, "version": version}


def verify_wheelhouse(manifest_path: Path, runtime_path: Path) -> list[dict]:
    """The candidate's additions are exactly the explicit wheels plus
    wheelhouse wheels, and each consumed wheelhouse wheel is installed as
    its own content."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    direct = {
        canonicalize_name(name): version
        for name, version in manifest["direct_requirements"].items()
    }
    wheelhouse = {wheel["name"]: wheel for wheel in manifest["wheels"]}
    added = {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
        and canonicalize_name(dist.metadata["Name"]) not in runtime
    }
    problems = []
    for name, version in sorted(added.items()):
        if name in direct:
            if direct[name] != version:
                problems.append(
                    f"{name}=={version}: the explicit wheel is {direct[name]}"
                )
        elif name not in wheelhouse:
            problems.append(
                f"{name}=={version}: neither an explicit wheel nor a wheelhouse wheel"
            )
        elif wheelhouse[name]["version"] != version:
            problems.append(
                f"{name}=={version}: the wheelhouse holds {wheelhouse[name]['version']}"
            )
    for name, version in sorted(direct.items()):
        if added.get(name) != version:
            problems.append(f"{name}=={version}: the explicit wheel is not installed")
    if problems:
        raise WheelMismatch(
            "the additions do not bind to the wheelhouse: " + "; ".join(problems)
        )
    reports = [
        compare_wheel(manifest_path.parent / wheelhouse[name]["filename"])
        for name in sorted(added)
        if name in wheelhouse
    ]
    unused = sorted(name for name in wheelhouse if name not in added)
    if unused:
        print(f"wheelhouse wheels not consumed by this image: {unused}")
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheel", type=Path, help="a staged wheel to compare")
    parser.add_argument(
        "--own-tree",
        action="append",
        default=[],
        help="top-level package that may contain only the wheel's files",
    )
    parser.add_argument("--min-members", type=int, default=0)
    parser.add_argument("--wheelhouse-manifest", type=Path)
    parser.add_argument(
        "--runtime", type=Path, help="runtime-distributions-before.json"
    )
    args = parser.parse_args(argv)
    if not args.wheel and not (args.wheelhouse_manifest and args.runtime):
        parser.error("give --wheel, or --wheelhouse-manifest with --runtime")
    try:
        if args.wheel:
            report = compare_wheel(args.wheel, tuple(args.own_tree), args.min_members)
            print(
                f"installed {report['name']}=={report['version']}: {report['compared']} "
                "wheel members are byte-identical to the staged wheel"
            )
        else:
            reports = verify_wheelhouse(args.wheelhouse_manifest, args.runtime)
            members = sum(report["compared"] for report in reports)
            print(
                f"offline-wheelhouse closure: {len(reports)} wheelhouse wheels consumed, "
                f"{members} installed members byte-identical to the wheels"
            )
    except (WheelMismatch, OSError, ValueError) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
