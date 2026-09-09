"""Check range integrity, ordered streaming and exact image metadata retention."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import tarfile
import time
from types import SimpleNamespace
import unittest

from pull_image import ChunkReader, check_range, save_image, verify_digest


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


class FakeRegistry:
    def __init__(self, data):
        self.data = data

    def chunk(self, descriptor, start, end):
        if start == 0:
            time.sleep(0.01)  # Later ranges can finish before the first one.
        return self.data[start : end + 1]


class ImagePullTests(unittest.TestCase):
    def test_out_of_order_chunks_support_small_archive_reads(self):
        data = bytes(range(256)) * 3
        descriptor = {"size": len(data), "digest": digest(data)}
        with ThreadPoolExecutor(max_workers=4) as executor:
            reader = ChunkReader(
                FakeRegistry(data), descriptor, executor, chunk_size=31
            )
            received = b"".join(iter(lambda: reader.read(7), b""))
            reader.finish()
        self.assertEqual(received, data)

    def test_corrupt_layer_is_rejected(self):
        data = b"corrupt bytes"
        descriptor = {"size": len(data), "digest": digest(b"original data")}
        with ThreadPoolExecutor(max_workers=4) as executor:
            reader = ChunkReader(FakeRegistry(data), descriptor, executor, chunk_size=4)
            reader.read(len(data))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                reader.finish()

    def test_config_and_duplicate_layer_references_are_preserved(self):
        inner = io.BytesIO()
        with tarfile.open(fileobj=inner, mode="w") as archive:
            info = tarfile.TarInfo("hello")
            info.size = 5
            archive.addfile(info, io.BytesIO(b"hello"))
        data = gzip.compress(inner.getvalue(), mtime=0)
        config = json.dumps(
            {
                "architecture": "amd64",
                "os": "linux",
                "rootfs": {
                    "type": "layers",
                    "diff_ids": [digest(inner.getvalue())] * 2,
                },
            }
        ).encode()
        layer = {
            "size": len(data),
            "digest": digest(data),
            "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
        }
        manifest = {
            "config": {"digest": digest(config), "size": len(config)},
            "layers": [layer, layer],
        }
        output = io.BytesIO()
        save_image(FakeRegistry(data), manifest, config, output)
        output.seek(0)
        with tarfile.open(fileobj=output) as archive:
            saved = json.load(archive.extractfile("manifest.json"))[0]
            self.assertEqual(archive.extractfile(saved["Config"]).read(), config)
            self.assertEqual(saved["Layers"][0], saved["Layers"][1])
            self.assertEqual(archive.getnames().count(saved["Layers"][0]), 1)
            self.assertEqual(archive.extractfile(saved["Layers"][0]).read(), data)

    def test_ignored_partial_range_is_rejected(self):
        with self.assertRaises(ValueError):
            check_range(SimpleNamespace(status=200, headers={}), 0, 7, 100)

    def test_wrong_content_range_is_rejected(self):
        response = SimpleNamespace(
            status=206, headers={"Content-Range": "bytes 8-15/100"}
        )
        with self.assertRaises(ValueError):
            check_range(response, 0, 7, 100)

    def test_manifest_or_config_digest_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_digest(b"changed metadata", digest(b"pinned metadata"))


if __name__ == "__main__":
    unittest.main()
