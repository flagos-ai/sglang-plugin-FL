"""Regression checks for TXDA CI card selection (tsm_smi parsing, grouping)."""

import unittest

from select_cards import (
    bad_cards,
    candidate_groups,
    idle_cards,
    parse_card_list,
    parse_tsm_smi,
)


def smi(
    used: dict[int, int], procs: dict[int, list[int]] | None = None, n: int = 32
) -> str:
    """Render a tsm_smi V0.30279.6 table (4 chips per card row block)."""
    lines = [
        "+----------------------------------------------------------------------------------------------+",
        "| SMI V0.30279.6.01                          Chip count: 32                                    |",
        "|  Card  Name              | Chip    Bus-Id      Temp   Pwr-Usage      Mem-Usage      NPU-Util |",
        "|==========================+===================================================================|",
    ]
    for chip in range(n):
        mem = used.get(chip, 8224)
        body = f"|  {chip:<2}   0000:{chip + 16:02x}:00.0   45C   64W / 200W   {mem:5d}M / 65536M     0%    |"
        if chip % 4 == 0:
            lines.append(f"|  {chip // 4}    TX8110-256-00      " + body)
        else:
            lines.append("|                          " + body)
    lines += [
        "| Processes:                                                                  |",
        "|  Card   PID      Process name                            Memory Usage       |",
        "|=============================================================================|",
    ]
    for chip, pids in (procs or {}).items():
        for pid in pids:
            lines.append(
                f"|   {chip:<2}    {pid:<7}    python3                                 46781M           |"
            )
            lines.append(
                "+-----------------------------------------------------------------------------+"
            )
    if not procs:
        lines.append(
            "|  No running processes found                                                 |"
        )
    return "\n".join(lines)


class ParseTests(unittest.TestCase):
    def test_parses_all_chips_including_card_header_rows(self):
        chips, procs = parse_tsm_smi(smi({}))
        self.assertEqual(sorted(chips), list(range(32)))
        self.assertEqual(chips[0]["used"], 8224)
        self.assertEqual(chips[0]["bus"], "0000:10:00.0")
        self.assertEqual(procs, {})

    def test_parses_real_header_row(self):
        row = (
            "|  0    TX8110-256-00      |  0    0000:a3:00.0   47C   72W / 200W   "
            "55005M / 65536M     0%    |"
        )
        chips, _ = parse_tsm_smi(row)
        self.assertEqual(
            chips,
            {0: {"bus": "0000:a3:00.0", "used": 55005, "total": 65536, "util": 0}},
        )

    def test_parses_processes(self):
        _, procs = parse_tsm_smi(smi({0: 55000}, {0: [421157], 1: [421157, 421158]}))
        self.assertEqual(procs, {0: [421157], 1: [421157, 421158]})


class SelectTests(unittest.TestCase):
    def test_all_idle_prefers_highest_aligned_group(self):
        chips, procs = parse_tsm_smi(smi({}))
        groups = candidate_groups(idle_cards(chips, procs))
        self.assertEqual(groups[0], list(range(24, 32)))
        self.assertEqual([g[0] for g in groups[:4]], [24, 16, 8, 0])

    def test_observed_occupancy_skips_busy_group(self):
        # Cards 0-7 held by another user's TP8 job (seen on host .38).
        chips, procs = parse_tsm_smi(
            smi({c: 55000 for c in range(8)}, {c: [421157 + c] for c in range(8)})
        )
        idle = idle_cards(chips, procs)
        self.assertEqual(idle, list(range(8, 32)))
        self.assertNotIn(0, [g[0] for g in candidate_groups(idle)])

    def test_process_without_memory_still_busy(self):
        chips, procs = parse_tsm_smi(smi({}, {13: [999]}))
        self.assertNotIn(13, idle_cards(chips, procs))

    def test_leftover_memory_without_process_is_busy(self):
        chips, procs = parse_tsm_smi(smi({20: 30000}))
        idle = idle_cards(chips, procs)
        self.assertNotIn(20, idle)
        self.assertNotIn(16, [g[0] for g in candidate_groups(idle) if g[0] % 8 == 0])

    def test_never_returns_partial_or_noncontiguous_group(self):
        self.assertEqual(candidate_groups([0, 1, 2, 3, 4, 5, 6]), [])
        self.assertEqual(candidate_groups([0, 2, 4, 6, 8, 10, 12, 14]), [])

    def test_unaligned_contiguous_group_is_fallback(self):
        self.assertEqual(candidate_groups(list(range(3, 11))), [list(range(3, 11))])

    def test_allocation_and_bad_cards(self):
        chips, procs = parse_tsm_smi(smi({}))
        allowed = parse_card_list("8-23")
        idle = idle_cards(chips, procs, allowed, bad={9})
        self.assertEqual(idle, [8] + list(range(10, 24)))
        self.assertEqual([g[0] for g in candidate_groups(idle) if g[0] % 8 == 0], [16])


class CardListTests(unittest.TestCase):
    def test_ranges_and_all(self):
        self.assertEqual(parse_card_list("8-10,20"), {8, 9, 10, 20})
        self.assertIsNone(parse_card_list("all"))
        self.assertIsNone(parse_card_list(None))
        with self.assertRaises(ValueError):
            parse_card_list("8,x")

    def test_bad_cards_env_and_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("# from dmesg\n22\n28-29\n")
        env = {"TSINGMICRO_CI_BAD_CARDS": "7", "TSINGMICRO_CI_BAD_CARDS_FILE": f.name}
        self.assertEqual(bad_cards(env), {7, 22, 28, 29})


if __name__ == "__main__":
    unittest.main()
