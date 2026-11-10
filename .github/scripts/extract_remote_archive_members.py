#!/usr/bin/env python3
"""Download a digest-pinned remote archive in parallel and extract named files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--registry-repository")
    parser.add_argument("--registry", default="https://harbor.baai.ac.cn")
    parser.add_argument("--blob-digest")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--member",
        action="append",
        help="Archive member and output filename as MEMBER=OUTPUT",
    )
    parser.add_argument(
        "--archive-output",
        help="Keep the verified archive at this path instead of extracting members",
    )
    args = parser.parse_args()
    if not args.member and not args.archive_output:
        parser.error("one of --member or --archive-output is required")
    return args


def registry_url_and_headers(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    if not args.blob_digest:
        raise SystemExit("--blob-digest is required with --registry-repository")
    query = urllib.parse.urlencode(
        {
            "service": "harbor-registry",
            "scope": f"repository:{args.registry_repository}:pull",
        }
    )
    with urllib.request.urlopen(f"{args.registry}/service/token?{query}") as response:
        token = json.load(response)["token"]
    url = (
        f"{args.registry}/v2/{args.registry_repository}/blobs/"
        f"{args.blob_digest}"
    )
    return url, {"Authorization": f"Bearer {token}"}


def download(
    url: str,
    headers: dict[str, str],
    size: int,
    expected_sha256: str,
    workers: int,
    destination: Path,
) -> None:
    chunk_size = math.ceil(size / workers)

    def fetch(part: int) -> Path:
        first = part * chunk_size
        last = min(size, (part + 1) * chunk_size) - 1
        output = destination.with_suffix(f".part{part}")
        request_headers = {**headers, "Range": f"bytes={first}-{last}"}
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(url, headers=request_headers)
                with urllib.request.urlopen(request, timeout=300) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"range {first}-{last} returned HTTP {response.status}"
                        )
                    with output.open("wb") as stream:
                        while data := response.read(8 * 1024 * 1024):
                            stream.write(data)
                expected_size = last - first + 1
                if output.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"part {part} has {output.stat().st_size}, expected {expected_size}"
                    )
                print(f"part {part}: {expected_size / 1_000_000:.1f} MB", flush=True)
                return output
            except Exception as error:
                print(f"part {part} attempt {attempt} failed: {error}", flush=True)
        raise RuntimeError(f"failed to download part {part}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(fetch, range(workers)))

    digest = hashlib.sha256()
    with destination.open("wb") as combined:
        for part in parts:
            with part.open("rb") as stream:
                while data := stream.read(8 * 1024 * 1024):
                    combined.write(data)
                    digest.update(data)
            part.unlink()
    actual = digest.hexdigest()
    if destination.stat().st_size != size or actual != expected_sha256:
        raise RuntimeError(
            f"archive verification failed: size={destination.stat().st_size}, sha256={actual}"
        )
    print(f"verified sha256:{actual}", flush=True)


def main() -> None:
    args = parse_args()
    url, headers = (
        registry_url_and_headers(args)
        if args.registry_repository
        else (args.url, {})
    )
    if args.archive_output:
        archive = Path(args.archive_output)
        archive.parent.mkdir(parents=True, exist_ok=True)
        download(
            url,
            headers,
            args.size,
            args.sha256,
            args.workers,
            archive,
        )
        return

    members = dict(item.split("=", 1) for item in args.member)
    with tempfile.TemporaryDirectory(prefix="vendor-archive-") as directory:
        archive = Path(directory) / "archive.tar.gz"
        download(url, headers, args.size, args.sha256, args.workers, archive)
        with tarfile.open(archive, mode="r:gz") as tar:
            names = set(tar.getnames())
            for member, output_name in members.items():
                if member not in names:
                    raise RuntimeError(f"missing archive member: {member}")
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"archive member is not a file: {member}")
                output = Path(output_name)
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with output.open("wb") as destination:
                    while data := source.read(8 * 1024 * 1024):
                        destination.write(data)
                        digest.update(data)
                print(f"{output}: sha256:{digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
