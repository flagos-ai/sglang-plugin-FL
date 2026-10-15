"""CPU metadata guards; these do not execute or claim GPU validation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "hygon_ci_devices", Path(__file__).with_name("check_devices.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metadata_torch(*, count=2, free=99, hcu=True, available=True):
    queried = []

    def memory(index):
        queried.append(index)
        return free, 100

    torch = SimpleNamespace(
        __version__="metadata-only",
        version=SimpleNamespace(hip="6.3"),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            mem_get_info=memory,
            get_device_name=lambda _: "metadata-only HCU",
        ),
    )
    if hcu:
        torch.__hcu_version__ = "metadata-only"
    return torch, queried


class DeviceAllocationGuards(unittest.TestCase):
    def test_only_first_two_visible_devices_are_queried(self):
        torch, queried = metadata_torch(count=8)
        result = module.check_devices(torch)
        self.assertEqual(queried, [0, 1])
        self.assertEqual(result["visible_gpu_count"], 8)

    def test_non_hcu_is_rejected_without_memory_query(self):
        torch, queried = metadata_torch(hcu=False)
        with self.assertRaisesRegex(RuntimeError, "Hygon DTK"):
            module.check_devices(torch)
        self.assertEqual(queried, [])

    def test_single_gpu_cannot_pass_tp2(self):
        torch, queried = metadata_torch(count=1)
        with self.assertRaisesRegex(RuntimeError, "TP2"):
            module.check_devices(torch)
        self.assertEqual(queried, [])

    def test_occupied_gpu_is_rejected(self):
        torch, queried = metadata_torch(free=94)
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            module.check_devices(torch)
        self.assertEqual(queried, [0])

    def test_unavailable_accelerator_is_rejected(self):
        torch, queried = metadata_torch(available=False)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            module.check_devices(torch)
        self.assertEqual(queried, [])

    def test_invalid_memory_accounting_cannot_pass(self):
        torch, _ = metadata_torch(free=101)
        with self.assertRaisesRegex(RuntimeError, "Invalid GPU memory"):
            module.check_devices(torch)

    def test_minimum_cannot_be_reduced_below_model_requirement(self):
        torch, queried = metadata_torch()
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            module.check_devices(torch, min_gpus=1)
        self.assertEqual(queried, [])


if __name__ == "__main__":
    unittest.main()
