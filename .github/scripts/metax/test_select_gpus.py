"""Regression checks for MetaX CI GPU selection from mx-smi text reports."""

import unittest

from select_gpus import (
    allowed_gpu_ids,
    choose_gpu_ids,
    idle_gpu_ids,
    parse_utilization,
    parse_vram_used_mib,
)

_HEADER = """mx-smi  version: 2.3.4

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Tue Nov  3 18:59:05 2026

Attached GPUs                                     : 8
"""


def memory_report(used_kb):
    blocks = [
        f"""GPU#{index}  MXC550  0000:{index:02x}:00.0
    Memory
        vis_vram total                            : 67108864 KB
        vis_vram used                             : {used} KB
        vis_vram usage                            : 1.3 %
        vram total                                : 67108864 KB
        vram used                                 : {used} KB
        vram usage                                : 1.3 %
        xtt total                                 : 792357519 KB
        xtt used                                  : 137248 KB
        xtt usage                                 : 0 %
"""
        for index, used in enumerate(used_kb)
    ]
    return _HEADER + "\n".join(blocks) + "\nEnd of Log\n"


def usage_report(utilization):
    blocks = [
        f"""GPU#{index}  MXC550  0000:{index:02x}:00.0
    Utilization
        GPU                                       : {util} %
        VPUE                                      : 0 %
        VPUD                                      : 0 %
"""
        for index, util in enumerate(utilization)
    ]
    return _HEADER + "\n".join(blocks) + "\nEnd of Log\n"


IDLE_KB = 879036  # driver-reserved VRAM on an idle C550 (~858 MiB)
BUSY_KB = 59700000


class ParseTests(unittest.TestCase):
    def test_reads_vram_not_visible_vram_or_xtt(self):
        used = parse_vram_used_mib(memory_report([IDLE_KB, BUSY_KB]))
        self.assertEqual(used, {0: IDLE_KB // 1024, 1: BUSY_KB // 1024})

    def test_reads_gpu_utilization(self):
        self.assertEqual(parse_utilization(usage_report([0, 75])), {0: 0, 1: 75})


class SelectGpuTests(unittest.TestCase):
    def _idle(self, used_kb, utilization=None, allowed=None):
        utilization = utilization or [0] * len(used_kb)
        return idle_gpu_ids(
            parse_vram_used_mib(memory_report(used_kb)),
            parse_utilization(usage_report(utilization)),
            allowed,
        )

    def test_all_idle_prefers_upper_aligned_group(self):
        self.assertEqual(choose_gpu_ids(self._idle([IDLE_KB] * 8)), [4, 5, 6, 7])

    def test_busy_cards_from_other_jobs_are_avoided(self):
        used = [IDLE_KB, BUSY_KB, IDLE_KB, BUSY_KB] + [IDLE_KB] * 4
        self.assertEqual(choose_gpu_ids(self._idle(used)), [4, 5, 6, 7])
        used = [IDLE_KB] * 4 + [BUSY_KB] * 2 + [IDLE_KB] * 2
        self.assertEqual(choose_gpu_ids(self._idle(used)), [0, 1, 2, 3])

    def test_running_kernel_with_little_memory_is_not_idle(self):
        self.assertIsNone(choose_gpu_ids(self._idle([IDLE_KB] * 4, [0, 50, 0, 0])))

    def test_insufficient_idle_devices_do_not_reduce_tp(self):
        self.assertIsNone(choose_gpu_ids([1, 2, 3]))

    def test_runner_allocation_is_preserved(self):
        allowed = allowed_gpu_ids({"CUDA_VISIBLE_DEVICES": "1,2,3"})
        self.assertEqual(allowed, {1, 2, 3})
        self.assertIsNone(choose_gpu_ids(self._idle([IDLE_KB] * 8, allowed=allowed)))

    def test_noncontiguous_allocation_remains_usable(self):
        self.assertEqual(choose_gpu_ids([0, 2, 4, 6]), [0, 2, 4, 6])

    def test_unset_or_all_means_unrestricted(self):
        self.assertIsNone(allowed_gpu_ids({}))
        self.assertIsNone(allowed_gpu_ids({"CUDA_VISIBLE_DEVICES": "all"}))

    def test_empty_allocation_selects_nothing(self):
        self.assertEqual(allowed_gpu_ids({"CUDA_VISIBLE_DEVICES": ""}), set())

    def test_uuid_allocation_is_rejected(self):
        with self.assertRaises(ValueError):
            allowed_gpu_ids({"CUDA_VISIBLE_DEVICES": "GPU-0f1e"})


if __name__ == "__main__":
    unittest.main()
