"""Compare real image bytes and high-entropy Docker uploads without using GPUs."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
import uuid
from urllib.parse import urlsplit

from pull_image import Registry, cached_dns


SIZE = 64 * 1024**2


def record(name, **values):
    print(json.dumps({"probe": name, **values}), flush=True)


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def archive_file(name, data):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def docker_probe(data, entropy, operation):
    nonce = uuid.uuid4().hex
    tag = f"musa-ci-entropy-probe:{nonce}"
    inner = archive_file("probe", data)
    target = tag
    if operation == "load":
        config = json.dumps(
            {
                "architecture": "amd64",
                "os": "linux",
                "config": {"Labels": {"musa-ci-transport-probe": nonce}},
                "rootfs": {"type": "layers", "diff_ids": [digest(inner)]},
                "history": [{}],
            }
        ).encode()
        target = digest(config)
        compressed = gzip.compress(inner, compresslevel=1, mtime=0)
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for name, contents in (
                ("config.json", config),
                (
                    "manifest.json",
                    json.dumps(
                        [
                            {
                                "Config": "config.json",
                                "RepoTags": [],
                                "Layers": ["layer.tar.gz"],
                            }
                        ]
                    ).encode(),
                ),
                ("layer.tar.gz", compressed),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        payload = output.getvalue()
        command = ["docker", "load"]
    else:
        payload = inner
        command = ["docker", "import", "-", tag]
    if (
        subprocess.run(
            ["docker", "image", "inspect", target], capture_output=True
        ).returncode
        == 0
    ):
        raise RuntimeError("Diagnostic image identity unexpectedly already exists")
    environment = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["timeout", "180", *command],
            input=payload,
            capture_output=True,
            env=environment,
            timeout=190,
        )
        elapsed = time.monotonic() - started
        record(
            f"docker-{operation}-{entropy}",
            bytes=len(payload),
            seconds=round(elapsed, 3),
            MiB_per_second=round(len(payload) / 1024**2 / elapsed, 3),
            returncode=result.returncode,
            error=result.stderr.decode(errors="replace")[-1000:],
        )
        result.check_returncode()
    finally:
        subprocess.run(
            ["timeout", "30", "docker", "image", "rm", target],
            capture_output=True,
            env=environment,
        )


def main():
    executable = Path(shutil.which("docker")).resolve()
    with executable.open("rb") as stream:
        prefix = stream.read(4)
    info = json.loads(
        subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True)
    )
    context = subprocess.run(
        ["docker", "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    endpoint = urlsplit(json.loads(context.stdout) if context.returncode == 0 else "")
    record(
        "environment",
        runner=platform.node(),
        docker_daemon=info.get("Name"),
        docker_executable=str(executable),
        docker_is_elf=prefix == b"\x7fELF",
        docker_root=info.get("DockerRootDir"),
        driver=info.get("Driver"),
        daemon_cpus=info.get("NCPU"),
        daemon_memory=info.get("MemTotal"),
        context_transport=endpoint.scheme,
        context_host=endpoint.hostname,
    )
    with cached_dns():
        registry = Registry(os.environ["MUSA_CI_IMAGE"])
        manifest = json.loads(registry.manifest())
        layer = next(x for x in manifest["layers"] if x["size"] >= 1024**3)
        started = time.monotonic()
        chunk_size = 8 * 1024**2
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(registry.chunk, layer, start, start + chunk_size - 1)
                for start in range(0, 2 * SIZE, chunk_size)
            ]
            total = sum(len(future.result()) for future in futures)
        elapsed = time.monotonic() - started
        record(
            "registry-ranges",
            bytes=total,
            seconds=round(elapsed, 3),
            MiB_per_second=round(total / 1024**2 / elapsed, 3),
        )
    # Unlike all-zero data, these bytes cannot benefit from compression in an
    # intermediate Docker transport. Reuse identical bytes for both APIs.
    random_data = os.urandom(SIZE)
    for operation in ("import", "load"):
        docker_probe(random_data, "random", operation)
    docker_probe(bytes(SIZE), "zero", "load")


if __name__ == "__main__":
    main()
