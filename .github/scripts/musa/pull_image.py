"""Stream the pinned MUSA image using bounded, verified HTTP range reads."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from http.client import IncompleteRead
import io
import json
import re
import socket
import sys
import tarfile
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


CHUNK_SIZE = 8 * 1024**2
WORKERS = 4
ARCHIVE_FORMAT = tarfile.PAX_FORMAT


@contextmanager
def cached_dns():
    # Thousands of range connections share a few hosts. Cache successful
    # resolutions for this pull only; TLS still verifies the original hostname.
    original = socket.getaddrinfo
    socket.getaddrinfo = lru_cache(maxsize=32)(original)
    try:
        yield
    finally:
        socket.getaddrinfo = original


def verify_digest(data, expected):
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"Digest mismatch: expected {expected}, received {actual}")


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def check_range(response, start, end, size):
    if response.status == 200 and start == 0 and end + 1 == size:
        return
    expected = f"bytes {start}-{end}/{size}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        raise ValueError(f"Registry did not honor the requested range: {expected}")


class Registry:
    def __init__(self, image):
        repository, self.digest = image.split("@", 1)
        self.host, self.repository = repository.split("/", 1)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest):
            raise ValueError("A SHA256-pinned image is required")
        self.base = f"https://{self.host}/v2/{self.repository}"
        self.token = None
        self.auth_lock = threading.Lock()

    def request(self, path, headers=None):
        # urllib uses HTTP/1.1; each bounded request has its own connection.
        # No registry credentials or runner-wide proxy changes are needed.
        opener = build_opener(ProxyHandler({}), SafeRedirect())
        for attempt in range(2):
            used_token = self.token
            request_headers = dict(headers or {})
            if used_token:
                request_headers["Authorization"] = "Bearer " + used_token
            try:
                return opener.open(
                    Request(self.base + path, headers=request_headers), timeout=30
                )
            except HTTPError as error:
                if error.code != 401 or attempt:
                    raise
                challenge = dict(
                    re.findall(r'(\w+)="([^"]+)"', error.headers["WWW-Authenticate"])
                )
                error.close()
                realm = challenge["realm"]
                parsed = urlsplit(realm)
                if parsed.scheme != "https" or parsed.netloc != self.host:
                    raise ValueError("Unexpected registry authentication origin")
                with self.auth_lock:
                    if self.token != used_token:
                        continue
                    query = urlencode(
                        {
                            "service": challenge["service"],
                            "scope": f"repository:{self.repository}:pull",
                        }
                    )
                    with opener.open(f"{realm}?{query}", timeout=30) as response:
                        auth = json.load(response)
                    self.token = auth.get("token") or auth["access_token"]
        raise RuntimeError("Registry authentication did not complete")

    def manifest(self):
        with self.request(
            f"/manifests/{self.digest}",
            {
                "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"
            },
        ) as response:
            raw = response.read()
        verify_digest(raw, self.digest)
        return raw

    def config(self, descriptor):
        with self.request(f"/blobs/{descriptor['digest']}") as response:
            raw = response.read(descriptor["size"] + 1)
        if len(raw) != descriptor["size"]:
            raise ValueError("Unexpected image configuration size")
        verify_digest(raw, descriptor["digest"])
        return raw

    def chunk(self, descriptor, start, end):
        for attempt in range(3):
            try:
                with self.request(
                    f"/blobs/{descriptor['digest']}",
                    {"Range": f"bytes={start}-{end}"},
                ) as response:
                    check_range(response, start, end, descriptor["size"])
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise OSError("Incomplete registry range response")
                return data
            except (OSError, IncompleteRead):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)


class ChunkReader:
    def __init__(self, registry, descriptor, executor, chunk_size=CHUNK_SIZE):
        self.registry = registry
        self.descriptor = descriptor
        self.executor = executor
        self.chunk_size = chunk_size
        self.offsets = iter(range(0, descriptor["size"], chunk_size))
        self.pending = deque()
        self.buffer = memoryview(b"")
        self.hasher = hashlib.sha256()
        self.written = 0
        for _ in range(WORKERS):
            self.submit()

    def submit(self):
        start = next(self.offsets, None)
        if start is not None:
            end = min(start + self.chunk_size, self.descriptor["size"]) - 1
            self.pending.append(
                self.executor.submit(self.registry.chunk, self.descriptor, start, end)
            )

    def read(self, size):
        if size < 0:
            raise ValueError("Image archive reads must be bounded")
        output = bytearray()
        while len(output) < size:
            if not self.buffer:
                if not self.pending:
                    break
                data = self.pending.popleft().result()
                self.hasher.update(data)
                self.buffer = memoryview(data)
                self.submit()
            count = min(size - len(output), len(self.buffer))
            output.extend(self.buffer[:count])
            self.buffer = self.buffer[count:]
        self.written += len(output)
        return bytes(output)

    def finish(self):
        if self.written != self.descriptor["size"]:
            raise ValueError("Incomplete image layer")
        actual = "sha256:" + self.hasher.hexdigest()
        if actual != self.descriptor["digest"]:
            raise ValueError(f"Layer digest mismatch: {self.descriptor['digest']}")


def layer_name(descriptor):
    media_type = descriptor["mediaType"]
    extension = ".tar.gz" if media_type.endswith("gzip") else ".tar"
    return descriptor["digest"].split(":", 1)[1] + extension


def save_image(registry, manifest, config, output):
    config_name = manifest["config"]["digest"].split(":", 1)[1] + ".json"
    docker_manifest = json.dumps(
        [
            {
                "Config": config_name,
                "RepoTags": [],
                "Layers": [layer_name(layer) for layer in manifest["layers"]],
            }
        ]
    ).encode()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        with tarfile.open(fileobj=output, mode="w|", format=ARCHIVE_FORMAT) as archive:
            for name, data in (
                (config_name, config),
                ("manifest.json", docker_manifest),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            seen = set()
            for index, layer in enumerate(manifest["layers"], 1):
                if layer["digest"] in seen:
                    continue
                reader = ChunkReader(registry, layer, executor)
                info = tarfile.TarInfo(layer_name(layer))
                info.size = layer["size"]
                archive.addfile(info, reader)
                reader.finish()
                seen.add(layer["digest"])
                print(
                    f"Verified layer {index}/{len(manifest['layers'])}: {layer['digest']}",
                    file=sys.stderr,
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    with cached_dns():
        registry = Registry(args.image)
        raw_manifest = registry.manifest()
        if args.manifest_only:
            sys.stdout.buffer.write(raw_manifest)
            return
        manifest = json.loads(raw_manifest)
        config = registry.config(manifest["config"])
        save_image(registry, manifest, config, sys.stdout.buffer)


if __name__ == "__main__":
    main()
