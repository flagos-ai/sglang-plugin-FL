"""FlagCX API selection, stream binding and existing fallback contracts."""
import ctypes
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang_fl.distributed.device_communicators import flagcx

class FlagCXAPIContract(unittest.TestCase):
    def test_packaged_api_precedes_legacy(self):
        api = object()
        spec = SimpleNamespace(submodule_search_locations=["/fake"])
        with patch.object(flagcx, "find_spec", return_value=spec), patch.object(flagcx.PathFinder, "find_spec", return_value=object()), patch.object(flagcx, "import_module", return_value=api) as importer:
            self.assertIs(flagcx._import_flagcx_api(), api)
            importer.assert_called_once_with("flagcx.api")

    def test_packaged_dependency_error_is_not_hidden(self):
        spec = SimpleNamespace(submodule_search_locations=["/fake"])
        error = ModuleNotFoundError("missing dependency", name="flagcx._C")
        with patch.object(flagcx, "find_spec", return_value=spec), patch.object(flagcx.PathFinder, "find_spec", return_value=object()), patch.object(flagcx, "import_module", side_effect=error) as importer:
            with self.assertRaises(ModuleNotFoundError) as caught:
                flagcx._import_flagcx_api()
            self.assertIs(caught.exception, error)
            importer.assert_called_once_with("flagcx.api")

    def test_legacy_api_remains_supported(self):
        api = object()
        with patch.object(flagcx, "find_spec", return_value=None), patch.object(flagcx, "import_module", return_value=api) as importer:
            self.assertIs(flagcx._import_flagcx_api(), api)
            importer.assert_called_once_with("plugin.interservice.flagcx_wrapper")

    def test_package_without_new_api_uses_legacy_wrapper(self):
        legacy = object()
        paths = ["/legacy/flagcx"]
        spec = SimpleNamespace(submodule_search_locations=paths)
        with patch.object(flagcx, "find_spec", return_value=spec), patch.object(flagcx.PathFinder, "find_spec", return_value=None) as finder, patch.object(flagcx, "import_module", return_value=legacy) as importer:
            self.assertIs(flagcx._import_flagcx_api(), legacy)
            finder.assert_called_once_with("flagcx.api", paths)
            importer.assert_called_once_with("plugin.interservice.flagcx_wrapper")

    def test_missing_api_is_reported(self):
        error = ModuleNotFoundError("missing legacy API", name="plugin")
        with patch.object(flagcx, "find_spec", return_value=None), patch.object(flagcx, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                flagcx._import_flagcx_api()
            self.assertIs(caught.exception, error)

    def test_explicit_root_pairs_packaged_api_with_source_library(self):
        api = object()
        spec = SimpleNamespace(submodule_search_locations=["/fake"])
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FLAGCX_PATH": root}),
            patch.object(sys, "path", list(sys.path)),
            patch.object(flagcx, "find_spec", return_value=spec),
            patch.object(flagcx.PathFinder, "find_spec", return_value=object()),
            patch.object(flagcx, "import_module", return_value=api) as importer,
        ):
            selected_api, library_path = flagcx._load_collective_api()

        self.assertIs(selected_api, api)
        self.assertEqual(Path(library_path), Path(root) / "build/lib/libflagcx.so")
        importer.assert_called_once_with("flagcx.api")

    def test_explicit_root_keeps_its_legacy_wrapper(self):
        legacy = object()
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FLAGCX_PATH": root}),
            patch.object(sys, "path", list(sys.path)),
            patch.object(flagcx, "find_spec") as find_spec,
            patch.object(flagcx, "import_module", return_value=legacy) as importer,
        ):
            wrapper = Path(root) / "plugin/interservice/flagcx_wrapper.py"
            wrapper.parent.mkdir(parents=True)
            wrapper.touch()
            selected_api, library_path = flagcx._load_collective_api()

        self.assertIs(selected_api, legacy)
        self.assertEqual(Path(library_path), Path(root) / "build/lib/libflagcx.so")
        find_spec.assert_not_called()
        importer.assert_called_once_with("plugin.interservice.flagcx_wrapper")

    def test_initialization_failure_preserves_existing_fallback(self):
        from sglang_fl.distributed.communicator import CommunicatorFL
        platform = SimpleNamespace(PlatformFL=Mock(return_value=SimpleNamespace(_dist_backend="flagcx")))
        with (
            patch.dict(sys.modules, {"sglang_fl.platform": platform}),
            patch.object(
                flagcx, "_load_collective_api", side_effect=RuntimeError("intentional import failure")
            ),
        ):
            comm = CommunicatorFL("cpu-group", __import__("torch").device("cuda:0"), "device-group", 2, 0, [0, 1])
            self.assertFalse(comm.is_active)
            self.assertIsNone(comm._flagcx_comm)
        tensor = object()
        with patch("sglang_fl.distributed.communicator.dist.all_reduce") as native:
            self.assertIs(comm.all_reduce(tensor), tensor)
            native.assert_called_once_with(tensor, group="device-group")

    def test_stream_uses_selected_api_and_caches_copy(self):
        def copy(dst, handle):
            dst._obj.value = 123
            return 0
        comm = flagcx.FlagCXCommunicator.__new__(flagcx.FlagCXCommunicator)
        comm._stream_cache = {}
        comm._get_raw_stream_handle = lambda: 42
        comm.flagcx = SimpleNamespace(devHandle=SimpleNamespace(contents=SimpleNamespace(streamCopy=Mock(side_effect=copy))), FLAGCX_CHECK=Mock())
        comm._flagcx_api = SimpleNamespace(flagcxStream_t=ctypes.c_void_p)
        stream = comm._get_stream()
        self.assertEqual(stream.value, 123)
        self.assertIs(comm._get_stream(), stream)
        comm.flagcx.devHandle.contents.streamCopy.assert_called_once()
        comm._stream_cache.clear()
        comm.flagcx = None


if __name__ == "__main__":
    unittest.main()
