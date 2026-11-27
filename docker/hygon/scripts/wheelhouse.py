#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline wheelhouse of the Hygon offline-wheelhouse development candidate.

The FlagOS package indexes are not always reachable, so a DEVELOPMENT
candidate can resolve the SGLang closure from a wheelhouse instead: a flat
directory holding exactly the wheels a last-green two-wheel image already
downloaded, consumed by uv with ``--no-index --find-links``. Nothing is
rebuilt or repacked: every file is the upstream wheel, verified against the
digest the registry declared when the green build downloaded it.

The source is the uv resolver cache that image still carries
(``/root/.cache/uv``). For every registry wheel a build downloaded, the cache
records the archive id, the ORIGINAL wheel URL, the filename, the SHA256 and
the unpacked tree. This tool reads that cache (exported to a directory by
docker/hygon/build_wheelhouse.sh):

  fetch      inventory the exported cache, download every recorded wheel,
             verify it three ways (registry SHA256 from the cache;
             member-by-member equality with the unpacked cache tree;
             METADATA Name/Version equal to the filename) and write the
             wheelhouse plus wheelhouse-manifest.json
  verify     re-check a wheelhouse directory against its manifest: every
             wheel present and unchanged, no extra or missing file
  bind-plan  cross-check a uv dry-run plan (sglang-resolver-additions.txt)
             against the manifest: every planned closure package must be a
             wheelhouse wheel at the same version

A wheel whose identity cannot be established is an error, never a warning:
there is no repack-from-unpacked-tree fallback.

Usage:
  wheelhouse.py fetch --cache DIR --out DIR --source-image NAME@sha256:HEX
                      [--tencent-index URL]
  wheelhouse.py verify --wheelhouse DIR
  wheelhouse.py bind-plan --wheelhouse DIR --plan FILE --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from packaging.utils import canonicalize_name, parse_wheel_filename

SCHEMA = "hygon-offline-wheelhouse-v1"
RESOLVER_MODE = "offline-wheelhouse"
MANIFEST_NAME = "wheelhouse-manifest.json"
# uv cache buckets of REGISTRY wheels (index pages and flat indexes). The
# ``url`` bucket holds direct-URL requirements - the application wheels the
# staging context passes as explicit files - and is never a wheelhouse entry.
REGISTRY_BUCKETS = ("index", "pypi")
# The three requirements uv receives as explicit wheel files; everything else
# in the plan must come from the wheelhouse.
DIRECT_REQUIREMENTS = {
    "sglang": "0.5.18+flagos.hcu1",
    "sglang-fl": "0.1.0",
    "xgrammar": "0.2.1+flagos.hcu1",
}
TENCENT_INDEX = "https://mirrors.cloud.tencent.com/pypi/simple"
SOURCE_IMAGE_REFERENCE = re.compile(r"^[a-z0-9._/-]+@sha256:[0-9a-f]{64}$")

# uv 0.11.30 stores a registry wheel's response metadata as
# wheels-v<N>/<bucket>/.../<entry>.http: a 2-byte rkyv prefix, the 16-char
# archive id, then length-prefixed strings - the Sha256 header value, the
# wheel filename, the wheel URL - and the quoted etag.
_ARCHIVE_ID = re.compile(rb"^[A-Za-z0-9_-]{16}$")
_SHA256 = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_WHEEL_URL = re.compile(rb"https?://[^\x00-\x20\"<>]+?\.whl")
_SIMPLE_LINK = re.compile(r'<a\s+href="([^"]+)"[^>]*>([^<]*)</a>')
_HEX256 = re.compile(r"^[0-9a-f]{64}$")
# where a wheel came from: the registry, a mirror, or a local file share
_URL = re.compile(r"^(https?|file)://\S+$")
_USER_AGENT = "sglang-plugin-fl-hygon-wheelhouse/1"
_DOWNLOAD_ATTEMPTS = 3


class WheelhouseError(Exception):
    """A wheel identity could not be established or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_identity(filename: str) -> tuple[str, str]:
    """(canonical name, complete version incl. local segment) of a filename."""
    name, version, _, _ = parse_wheel_filename(filename)
    return canonicalize_name(name), str(version)


def read_metadata(zip_file: ZipFile, filename: str) -> tuple[str, str]:
    stem = filename[: -len(".whl")]
    dist_info = stem.rsplit("-", 3)[0]
    try:
        metadata = zip_file.read(f"{dist_info}.dist-info/METADATA").decode("utf-8")
    except (KeyError, BadZipFile) as failure:
        raise WheelhouseError(f"{filename}: unreadable METADATA: {failure}")
    fields = {}
    for line in metadata.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep and key in ("Name", "Version"):
            fields[key] = value.strip()
    if "Name" not in fields or "Version" not in fields:
        raise WheelhouseError(f"{filename}: METADATA lacks Name/Version")
    return fields["Name"], fields["Version"]


def read_cache_entry(path: Path, cache: Path) -> dict:
    """Parse one ``.http`` cache entry and check it against its archive."""
    data = path.read_bytes()
    archive_id = data[2:18]
    digests = _SHA256.findall(data)
    urls = _WHEEL_URL.findall(data)
    if not _ARCHIVE_ID.match(archive_id):
        raise WheelhouseError(f"{path}: not a uv registry-wheel cache entry")
    if len(digests) != 1 or len(urls) != 1:
        raise WheelhouseError(
            f"{path}: expected one SHA256 and one wheel URL, found "
            f"{len(digests)} and {len(urls)}"
        )
    url = urls[0].decode("utf-8")
    filename = url.rsplit("/", 1)[-1]
    archive = cache / "archive-v0" / archive_id.decode("ascii")
    dist_infos = sorted(p.name for p in archive.glob("*.dist-info"))
    dist_info_stem = dist_infos[0][: -len(".dist-info")] if dist_infos else None
    if len(dist_infos) != 1 or not filename[: -len(".whl")].startswith(
        dist_info_stem + "-"
    ):
        raise WheelhouseError(
            f"{path}: the cached archive {archive} does not carry exactly the "
            f"dist-info of {filename} (found {dist_infos})"
        )
    name, version = wheel_identity(filename)
    return {
        "archive_id": archive_id.decode("ascii"),
        "bucket": path.relative_to(cache).parts[1],
        "filename": filename,
        "name": name,
        "sha256": digests[0].decode("ascii"),
        "source_url": url,
        "version": version,
    }


def inventory(cache: Path) -> list[dict]:
    """Every registry wheel the exported uv cache holds, sorted by filename."""
    entries = []
    for bucket in REGISTRY_BUCKETS:
        root = cache / "wheels-v6" / bucket
        for path in sorted(root.rglob("*.http")) if root.is_dir() else ():
            entries.append(read_cache_entry(path, cache))
    if not entries:
        raise WheelhouseError(f"{cache}: no registry wheel entries under wheels-v6")
    return sorted(entries, key=lambda entry: entry["filename"])


def verify_wheel_content(wheel: Path, archive: Path, filename: str) -> None:
    """The wheel's members must be exactly the cache's unpacked tree;
    ``filename`` is the wheel's real name (``wheel`` may be a partial file)."""
    tree = {
        path.relative_to(archive).as_posix()
        for path in archive.rglob("*")
        if path.is_file()
    }
    try:
        zip_file = ZipFile(wheel)
    except BadZipFile as failure:
        raise WheelhouseError(f"{filename}: not a zip archive: {failure}")
    with zip_file:
        members = {info.filename for info in zip_file.infolist() if not info.is_dir()}
        if members != tree:
            raise WheelhouseError(
                f"{filename}: members differ from the cached tree "
                f"(only in wheel {sorted(members - tree)[:3]}, only in cache "
                f"{sorted(tree - members)[:3]})"
            )
        try:
            differing = [
                member
                for member in sorted(members)
                if hashlib.sha256(zip_file.read(member)).hexdigest()
                != sha256_file(archive / member)
            ]
        except (BadZipFile, OSError) as failure:
            raise WheelhouseError(f"{filename}: unreadable member: {failure}")
        if differing:
            raise WheelhouseError(
                f"{filename}: {len(differing)} members differ from the cached "
                f"tree, first {differing[:3]}"
            )
        metadata_name, metadata_version = read_metadata(zip_file, filename)
    name, version = wheel_identity(filename)
    if canonicalize_name(metadata_name) != name or metadata_version != version:
        raise WheelhouseError(
            f"{filename}: METADATA says {metadata_name} {metadata_version}, "
            f"the filename says {name} {version}"
        )


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as target:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                target.write(chunk)


def tencent_wheel_url(index: str, entry: dict) -> str | None:
    """Resolve the exact mirror link of one wheel from the simple page."""
    page_url = f"{index.rstrip('/')}/{entry['name']}/"
    request = urllib.request.Request(page_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8")
    for href, text in _SIMPLE_LINK.findall(page):
        if text.strip().split("#")[0] == entry["filename"]:
            return urllib.parse.urljoin(page_url, href)
    return None


def fetch_wheel(
    entry: dict,
    cache: Path,
    out: Path,
    tencent_index: str = TENCENT_INDEX,
    fetch=download,
    pause=time.sleep,
    lookup=tencent_wheel_url,
) -> dict:
    """Download one wheel from its recorded URL (the mirror is the fallback
    for generic wheels), verify it and return its manifest record."""
    archive = cache / "archive-v0" / entry["archive_id"]
    target = out / entry["filename"]
    partial = out / f".{entry['filename']}.part"
    candidates = [entry["source_url"]]
    failures = []
    index = 0
    while index < len(candidates):
        url = candidates[index]
        index += 1
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                fetch(url, partial)
            except OSError as failure:
                failures.append(f"{url}: {failure}")
                if attempt < _DOWNLOAD_ATTEMPTS:
                    pause(2 * attempt)
                    continue
                break
            try:
                actual = sha256_file(partial)
                if actual != entry["sha256"]:
                    raise WheelhouseError(
                        f"downloaded SHA256 {actual} != registry digest "
                        f"{entry['sha256']}"
                    )
                verify_wheel_content(partial, archive, entry["filename"])
            except WheelhouseError as failure:
                failures.append(f"{url}: {failure}")
                break
            os.replace(partial, target)
            return {
                "cache_archive_id": entry["archive_id"],
                "download_url": url,
                "filename": entry["filename"],
                "name": entry["name"],
                "sha256": entry["sha256"],
                "size": target.stat().st_size,
                "source_url": entry["source_url"],
                "verified": (
                    "registry SHA256 recorded by the green build + every member "
                    "equal to the cached unpacked tree + METADATA identity"
                ),
                "version": entry["version"],
            }
        partial.unlink(missing_ok=True)
        if index != 1:
            continue
        # the URL the green build used is unreachable: a GENERIC wheel may be
        # completed from the mirror, by its exact file link
        try:
            fallback = lookup(tencent_index, entry)
        except OSError as lookup_failure:
            fallback = None
            failures.append(f"{tencent_index}/{entry['name']}/: {lookup_failure}")
        if fallback is None:
            break
        candidates.append(fallback)
    raise WheelhouseError(f"{entry['filename']}: " + " | ".join(failures))


def build_manifest(
    records: list[dict], source_image: str, excluded: list[dict]
) -> dict:
    return {
        "schema": SCHEMA,
        "resolver_mode": RESOLVER_MODE,
        "purpose": (
            "offline resolver source (--no-index --find-links) of the Hygon "
            "DEVELOPMENT CANDIDATE image; never a release input"
        ),
        "cache_source": {
            "image": source_image,
            "cache_dir": "/root/.cache/uv",
            "role": (
                "uv resolver cache of a green two-wheel image: every wheel "
                "below is one that build downloaded"
            ),
        },
        "direct_requirements": DIRECT_REQUIREMENTS,
        # wheels of the green build that the candidate passes as explicit
        # (corrected) files instead; recorded so nothing silently disappears
        "excluded_explicit_requirements": [
            {
                "filename": entry["filename"],
                "name": entry["name"],
                "sha256": entry["sha256"],
                "source_url": entry["source_url"],
                "version": entry["version"],
            }
            for entry in excluded
        ],
        "count": len(records),
        "wheels": sorted(
            records, key=lambda record: (record["name"], record["version"])
        ),
    }


def verify(wheelhouse: Path) -> dict:
    """The manifest and the directory must describe exactly the same files,
    every one of them hashing to its recorded digest."""
    manifest_path = wheelhouse / MANIFEST_NAME
    if not manifest_path.is_file():
        raise WheelhouseError(f"{wheelhouse}: no {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise WheelhouseError(f"{manifest_path}: schema {manifest.get('schema')!r}")
    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise WheelhouseError(f"{manifest_path}: carries no wheels")
    names = []
    for wheel in wheels:
        try:
            identity = wheel_identity(wheel["filename"])
            valid = (
                identity == (wheel["name"], wheel["version"])
                and _HEX256.match(wheel["sha256"]) is not None
                and _URL.match(wheel["source_url"]) is not None
                and _URL.match(wheel["download_url"]) is not None
            )
        except (KeyError, ValueError) as failure:
            raise WheelhouseError(
                f"{manifest_path}: malformed entry {wheel}: {failure}"
            )
        if not valid:
            raise WheelhouseError(f"{manifest_path}: inconsistent entry {wheel}")
        names.append(wheel["name"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    direct = sorted(set(names) & set(DIRECT_REQUIREMENTS))
    if duplicates or direct:
        raise WheelhouseError(
            f"{manifest_path}: duplicate names {duplicates}, explicit wheel "
            f"requirements inside the wheelhouse {direct}"
        )
    if manifest.get("count") != len(wheels):
        raise WheelhouseError(
            f"{manifest_path}: count {manifest.get('count')} != {len(wheels)} wheels"
        )
    present = {path.name for path in wheelhouse.iterdir()}
    expected = {MANIFEST_NAME} | {wheel["filename"] for wheel in wheels}
    missing, extra = sorted(expected - present), sorted(present - expected)
    if missing or extra:
        raise WheelhouseError(f"{wheelhouse}: missing {missing}, unexpected {extra}")
    changed = [
        wheel["filename"]
        for wheel in wheels
        if sha256_file(wheelhouse / wheel["filename"]) != wheel["sha256"]
    ]
    if changed:
        raise WheelhouseError(
            f"{wheelhouse}: wheels changed since the manifest: {changed}"
        )
    return manifest


def bind_plan(wheelhouse: Path, plan_file: Path, out_file: Path) -> dict:
    """Every closure package of a dry-run plan must be a wheelhouse wheel at
    the same version; the three explicit wheels must keep their versions."""
    by_name = {wheel["name"]: wheel for wheel in verify(wheelhouse)["wheels"]}
    consumed, unused, problems = [], set(by_name), []
    for line in plan_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, version = line.partition("==")
        if name in DIRECT_REQUIREMENTS:
            if DIRECT_REQUIREMENTS[name] != version:
                problems.append(f"{name}=={version} != the explicit wheel")
            continue
        wheel = by_name.get(name)
        if wheel is None:
            problems.append(f"{line}: not a wheelhouse wheel")
        elif wheel["version"] != version:
            problems.append(f"{line}: the wheelhouse holds {wheel['version']}")
        else:
            consumed.append(wheel)
            unused.discard(name)
    if problems:
        raise WheelhouseError(
            "the plan does not bind to the wheelhouse: " + "; ".join(problems)
        )
    report = {
        "consumed": len(consumed),
        "unused": sorted(unused),
    }
    out_file.write_text(
        "# closure wheels the dry-run plan consumes from the wheelhouse\n"
        + "".join(
            f"{wheel['name']}=={wheel['version']} {wheel['sha256']} {wheel['filename']}\n"
            for wheel in consumed
        )
        + (
            "# unused wheelhouse wheels: " + ", ".join(sorted(unused)) + "\n"
            if unused
            else ""
        )
    )
    return report


def fetch(
    cache: Path,
    out: Path,
    source_image: str,
    tencent_index: str,
    workers: int,
    fetch_url=download,
    pause=time.sleep,
    lookup=tencent_wheel_url,
) -> dict:
    """Build the wheelhouse ``out`` from the exported cache ``cache``."""
    if SOURCE_IMAGE_REFERENCE.match(source_image) is None:
        raise WheelhouseError(
            f"--source-image must be NAME@sha256:<64 hex>, got {source_image!r}"
        )
    entries = inventory(cache)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise WheelhouseError(
            f"{out} is not empty; a wheelhouse is built into a fresh directory"
        )
    print(f"cache inventory: {len(entries)} registry wheels", flush=True)
    # the candidate hands uv the corrected xgrammar/SGLang and the current
    # sglang_fl as explicit files, so their green-build originals stay out
    closure = [entry for entry in entries if entry["name"] not in DIRECT_REQUIREMENTS]
    excluded = [entry for entry in entries if entry["name"] in DIRECT_REQUIREMENTS]
    print(
        f"closure wheels: {len(closure)}; explicit-requirement originals left out: "
        + ", ".join(
            f"{entry['filename']} ({entry['sha256'][:12]})" for entry in excluded
        ),
        flush=True,
    )

    def guarded(entry: dict):
        try:
            return fetch_wheel(
                entry, cache, out, tencent_index, fetch_url, pause, lookup
            )
        except (WheelhouseError, OSError, ValueError) as failure:
            return WheelhouseError(f"{entry['filename']}: {failure}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(guarded, closure))
    failures = [result for result in results if isinstance(result, WheelhouseError)]
    if failures:
        for path in out.iterdir():
            path.unlink()
        raise WheelhouseError(
            f"{len(failures)} of {len(closure)} wheels could not be established:\n  "
            + "\n  ".join(str(failure) for failure in failures)
        )
    manifest = build_manifest(results, source_image, excluded)
    manifest_path = out / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        verify(out)
    except WheelhouseError:
        # a failed build leaves no half wheelhouse behind
        for path in out.iterdir():
            path.unlink()
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser(
        "fetch", help="build a wheelhouse from an exported uv cache"
    )
    fetch_parser.add_argument(
        "--cache", type=Path, required=True, help="exported /root/.cache/uv"
    )
    fetch_parser.add_argument(
        "--out", type=Path, required=True, help="new wheelhouse directory"
    )
    fetch_parser.add_argument(
        "--source-image", required=True, help="NAME@sha256:<digest> of the cache source"
    )
    fetch_parser.add_argument("--tencent-index", default=TENCENT_INDEX)
    fetch_parser.add_argument("--workers", type=int, default=8)

    verify_parser = commands.add_parser(
        "verify", help="re-check a wheelhouse against its manifest"
    )
    verify_parser.add_argument("--wheelhouse", type=Path, required=True)

    bind_parser = commands.add_parser(
        "bind-plan", help="bind a dry-run plan to the wheelhouse"
    )
    bind_parser.add_argument("--wheelhouse", type=Path, required=True)
    bind_parser.add_argument(
        "--plan", type=Path, required=True, help="sglang-resolver-additions.txt"
    )
    bind_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            manifest = fetch(
                args.cache,
                args.out,
                args.source_image,
                args.tencent_index,
                args.workers,
            )
            total = sum(wheel["size"] for wheel in manifest["wheels"])
            print(
                f"wheelhouse: {manifest['count']} wheels, {total} bytes, under {args.out}"
            )
            print(f"manifest:   {args.out / MANIFEST_NAME}")
        elif args.command == "verify":
            manifest = verify(args.wheelhouse)
            print(
                f"wheelhouse verified: {manifest['count']} wheels match {MANIFEST_NAME}"
            )
        else:
            report = bind_plan(args.wheelhouse, args.plan, args.out)
            print(
                f"wheelhouse binding: {report['consumed']} wheels consumed by the plan"
                + (
                    f"; {len(report['unused'])} unused ({', '.join(report['unused'])})"
                    if report["unused"]
                    else "; none unused"
                )
            )
    except (WheelhouseError, OSError, ValueError) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
