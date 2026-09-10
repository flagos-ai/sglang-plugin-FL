"""Regression checks for CI GPU occupancy and explicit device allocations."""

import unittest

from select_gpus import allowed_gpu_ids, choose_gpu_ids, idle_gpu_ids


def snapshot(used, utilization=None):
    utilization = utilization or {}
    return {
        "GPU": [
            {
                "Index": str(index),
                "FB Memory Usage": {"Used": f"{memory}MiB"},
                "Utilization": {"Gpu": f"{utilization.get(index, 0)}%"},
            }
            for index, memory in enumerate(used)
        ]
    }


class SelectGpuTests(unittest.TestCase):
    def test_observed_ci_occupancy_avoids_gpu_zero(self):
        report = snapshot([66325, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(choose_gpu_ids(idle_gpu_ids(report)), [4, 5, 6, 7])

    def test_busy_upper_devices_use_free_lower_group(self):
        report = snapshot([8, 8, 8, 8, 79446, 78690, 8, 8])
        self.assertEqual(choose_gpu_ids(idle_gpu_ids(report)), [0, 1, 2, 3])

    def test_running_small_kernel_is_not_idle(self):
        report = snapshot([8, 8, 8, 8], {0: 50})
        self.assertIsNone(choose_gpu_ids(idle_gpu_ids(report)))

    def test_insufficient_idle_devices_do_not_reduce_tp(self):
        self.assertIsNone(choose_gpu_ids([1, 2, 3]))

    def test_runner_allocation_is_preserved(self):
        allowed = allowed_gpu_ids(
            {"MTHREADS_VISIBLE_DEVICES": "0,1,2,3", "MUSA_VISIBLE_DEVICES": "1,2,3"}
        )
        self.assertEqual(allowed, {1, 2, 3})
        self.assertIsNone(choose_gpu_ids(idle_gpu_ids(snapshot([2] * 8), allowed)))

    def test_noncontiguous_allocation_remains_usable(self):
        self.assertEqual(choose_gpu_ids([0, 2, 4, 6]), [0, 2, 4, 6])

    def test_uuid_allocation_maps_to_visible_indices_and_intersects_mask(self):
        report = snapshot([2] * 8)
        for index, gpu in enumerate(report["GPU"]):
            gpu["GPU UUID"] = f"c4c4e04c-7908-1fcc-7713-{index:012x}"
        allocation = ",".join(report["GPU"][i]["GPU UUID"] for i in (1, 2, 4, 6))
        environ = {"MTHREADS_VISIBLE_DEVICES": allocation}
        allowed = allowed_gpu_ids(environ, report)
        self.assertEqual(choose_gpu_ids(idle_gpu_ids(report, allowed)), [1, 2, 4, 6])
        environ["MUSA_VISIBLE_DEVICES"] = "0,2,4"
        self.assertEqual(allowed_gpu_ids(environ, report), {2, 4})

    def test_unreported_uuid_does_not_expand_runner_allocation(self):
        with self.assertRaises(ValueError):
            allowed_gpu_ids(
                {"MTHREADS_VISIBLE_DEVICES": "c4c4e04c-7908-1fcc-7713-4c2f35bdaabb"},
                snapshot([2] * 8),
            )

    def test_hidden_devices_remain_hidden(self):
        self.assertEqual(allowed_gpu_ids({"MUSA_VISIBLE_DEVICES": ""}), set())
        with self.assertRaises(ValueError):
            allowed_gpu_ids({"MUSA_VISIBLE_DEVICES": "unrecognized-allocation"})


if __name__ == "__main__":
    unittest.main()
