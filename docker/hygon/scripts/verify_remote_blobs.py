#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Read a pushed image straight from the Harbor registry and prove that every
blob its manifest references exists there.

Used after a candidate push (docker/hygon/scripts/verify_remote_candidate.sh):
the runner must be able to pull the candidate from the registry alone, so
every parent layer of the Runtime base has to be present remotely even when
the base's own manifest was pruned long ago.

Reads anonymously (pull scope) through the registry v2 API and never touches
the local Docker store: a blob that exists only locally cannot pass.

Usage: verify_remote_blobs.py <registry host> <repository> <sha256:digest>

Prints a JSON record (index or manifest digest, per-platform manifests, the
linux/amd64 image config digest, its diff_ids and org.flagos.ci labels, and
the blobs checked) on success.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMEOUT = 120


class RegistryError(Exception):
    """The registry could not confirm the object."""


class Registry:
    def __init__(self, host: str, repository: str, scheme: str = "https") -> None:
        self.base = f"{scheme}://{host}/v2/{repository}"
        self.authorization = ""
        token_url = (
            f"{scheme}://{host}/service/token"
            f"?service=harbor-registry&scope=repository:{repository}:pull"
        )
        try:
            with urllib.request.urlopen(token_url, timeout=_TIMEOUT) as response:
                token = json.load(response).get("token")
            if token:
                self.authorization = f"Bearer {token}"
        except (OSError, ValueError):
            pass  # a registry that needs no token

    def _request(self, url: str, method: str = "GET", accept: str = "") -> tuple:
        request = urllib.request.Request(url, method=method)
        if self.authorization:
            request.add_header("Authorization", self.authorization)
        if accept:
            request.add_header("Accept", accept)
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as failure:
            return failure.code, b"", failure.headers
        except OSError as failure:
            raise RegistryError(f"{method} {url}: {failure}")

    def manifest(self, digest: str) -> dict:
        status, body, _ = self._request(
            f"{self.base}/manifests/{digest}", accept=MANIFEST_TYPES
        )
        if status != 200:
            raise RegistryError(f"GET manifests/{digest}: HTTP {status}")
        return json.loads(body)

    def has_blob(self, digest: str) -> bool:
        status, _, _ = self._request(f"{self.base}/blobs/{digest}", method="HEAD")
        return status == 200

    def blob(self, digest: str) -> dict:
        status, body, _ = self._request(f"{self.base}/blobs/{digest}")
        if status != 200:
            raise RegistryError(f"GET blobs/{digest}: HTTP {status}")
        return json.loads(body)


def inspect(host: str, repository: str, digest: str, scheme: str = "https") -> dict:
    registry = Registry(host, repository, scheme)
    root = registry.manifest(digest)
    record = {
        "registry": host,
        "repository": repository,
        "digest": digest,
        "media_type": root.get("mediaType", "unspecified"),
        "manifests": [],
        "blobs_checked": [],
        "blobs_missing": [],
    }
    is_index = "manifests" in root
    platforms = (
        [(entry["digest"], entry.get("platform", {})) for entry in root["manifests"]]
        if is_index
        else [(digest, {})]
    )
    blobs: dict[str, None] = {}
    amd64 = None
    for manifest_digest, platform in platforms:
        manifest = (
            root if manifest_digest == digest else registry.manifest(manifest_digest)
        )
        name = "{}/{}".format(
            platform.get("os", "?"), platform.get("architecture", "?")
        )
        record["manifests"].append(
            {
                "digest": manifest_digest,
                "platform": name,
                "config_digest": manifest["config"]["digest"],
                "layers": len(manifest["layers"]),
            }
        )
        blobs[manifest["config"]["digest"]] = None
        for layer in manifest["layers"]:
            blobs[layer["digest"]] = None
        if name == "linux/amd64":
            amd64 = manifest
    if amd64 is None:
        raise RegistryError(f"{digest}: no linux/amd64 image manifest")
    config = registry.blob(amd64["config"]["digest"])
    record["linux_amd64"] = {
        "manifest_digest": next(
            item["digest"]
            for item in record["manifests"]
            if item["platform"] == "linux/amd64"
        ),
        "config_digest": amd64["config"]["digest"],
        "layers": len(amd64["layers"]),
        "diff_ids": config["rootfs"]["diff_ids"],
        "config": {
            key: config.get(key) for key in ("architecture", "os", "config", "created")
        },
    }
    if record["linux_amd64"]["layers"] != len(config["rootfs"]["diff_ids"]):
        raise RegistryError("layer count differs from the diff_ids of the config")
    for blob in sorted(blobs):
        if registry.has_blob(blob):
            record["blobs_checked"].append(blob)
        else:
            record["blobs_missing"].append(blob)
    if record["blobs_missing"]:
        raise RegistryError(
            f"{len(record['blobs_missing'])} blobs are not in the registry, "
            f"first {record['blobs_missing'][:3]}"
        )
    return record


def main(argv: list[str] | None = None) -> int:
    if len(argv or sys.argv[1:]) != 3:
        print(
            "usage: verify_remote_blobs.py <registry host> <repository> <sha256:digest>",
            file=sys.stderr,
        )
        return 2
    host, repository, digest = argv or sys.argv[1:]
    if DIGEST.match(digest) is None:
        print(f"FATAL: not a sha256 digest: {digest}", file=sys.stderr)
        return 1
    try:
        record = inspect(host, repository, digest)
    except (RegistryError, KeyError, ValueError) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
