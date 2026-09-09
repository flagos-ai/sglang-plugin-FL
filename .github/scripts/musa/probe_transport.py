"""Measure registry reads and Docker API uploads without using any GPU."""

import io
import json
import os
import re
import subprocess
import tarfile
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def registry_read(direct):
    image = os.environ["MUSA_CI_IMAGE"]
    repository, digest = image.split("@", 1)
    registry, repository = repository.split("/", 1)
    opener = build_opener(
        ProxyHandler({}) if direct else ProxyHandler(), SafeRedirect()
    )
    base = f"https://{registry}/v2/{repository}"
    headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    request = Request(f"{base}/manifests/{digest}", headers=headers)
    try:
        response = opener.open(request, timeout=20)
    except HTTPError as error:
        if error.code != 401:
            raise
        challenge = dict(
            re.findall(r'(\w+)="([^"]+)"', error.headers["WWW-Authenticate"])
        )
        realm = challenge["realm"]
        assert urlsplit(realm).hostname == registry
        query = urlencode(
            {"service": challenge["service"], "scope": f"repository:{repository}:pull"}
        )
        with opener.open(f"{realm}?{query}", timeout=20) as auth_response:
            auth = json.load(auth_response)
        headers["Authorization"] = "Bearer " + (
            auth.get("token") or auth["access_token"]
        )
        response = opener.open(Request(request.full_url, headers=headers), timeout=20)
    with response:
        manifest = json.load(response)
    layer = next(layer for layer in manifest["layers"] if layer["size"] >= 1024**3)
    count = 8 * 1024**2
    request = Request(
        f"{base}/blobs/{layer['digest']}",
        headers={**headers, "Range": f"bytes=0-{count - 1}"},
    )
    started = time.monotonic()
    with opener.open(request, timeout=30) as response:
        status = response.status
        total = 0
        while total < count:
            block = response.read(min(64 * 1024, count - total))
            if not block:
                break
            total += len(block)
    elapsed = time.monotonic() - started
    return {
        "bytes": total,
        "seconds": round(elapsed, 3),
        "MiB_per_second": round(total / 1024**2 / elapsed, 3),
        "http_status": status,
    }


def docker_upload(direct):
    route = "direct" if direct else "inherited"
    tag = f"musa-ci-transfer-probe:{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{route}"
    environment = dict(os.environ)
    if direct:
        for key in PROXY_KEYS:
            environment.pop(key, None)
    started = time.monotonic()
    process = subprocess.Popen(
        ["timeout", "90", "docker", "import", "-", tag],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        size = 32 * 1024**2
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            info = tarfile.TarInfo("musa-transfer-probe")
            info.size = size
            archive.addfile(info, io.BytesIO(bytes(size)))
        process.stdin.close()
        process.stdin = None
        process.communicate(timeout=100)
        elapsed = time.monotonic() - started
        return {
            "returncode": process.returncode,
            "seconds": round(elapsed, 3),
            "MiB_per_second": round(size / 1024**2 / elapsed, 3),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        # Only the uniquely named diagnostic image from this attempt is removed.
        subprocess.run(
            ["timeout", "30", "docker", "image", "rm", tag],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def report(name, probe):
    try:
        result = probe()
    except Exception as error:
        result = {"error_type": type(error).__name__}
    print(json.dumps({"probe": name, **result}), flush=True)


def main():
    endpoint = urlsplit(os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"))
    print(
        json.dumps(
            {
                "docker_transport": endpoint.scheme,
                "docker_host": endpoint.hostname,
                "proxy_hosts": {
                    key: urlsplit(os.environ[key]).hostname
                    for key in PROXY_KEYS
                    if os.environ.get(key)
                },
            }
        ),
        flush=True,
    )
    info = json.loads(
        subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True)
    )
    print(
        json.dumps(
            {
                "docker_storage_driver": info.get("Driver"),
                "docker_root": info.get("DockerRootDir"),
                "daemon_proxy_hosts": {
                    key: urlsplit(info.get(key) or "").hostname
                    for key in ("HTTPProxy", "HTTPSProxy")
                },
            }
        ),
        flush=True,
    )
    for direct in (False, True):
        route = "direct" if direct else "inherited"
        report(f"registry-{route}", lambda: registry_read(direct))
        report(f"docker-upload-{route}", lambda: docker_upload(direct))


if __name__ == "__main__":
    main()
