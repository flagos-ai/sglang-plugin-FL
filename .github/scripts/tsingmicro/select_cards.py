#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select a healthy, idle 8-card TXDA group for the TP8 CI suite.

Card protection, in order:
  1. Occupancy (``tsm_smi``, no device context is created): a card is busy if
     any process holds it or its memory use is above the idle baseline. A
     group is only chosen if all 8 cards are idle.
  2. Exclusions: cards listed in TSINGMICRO_CI_BAD_CARDS (comma separated) or
     in the file named by TSINGMICRO_CI_BAD_CARDS_FILE (one index per line,
     ``#`` comments) are never used. The runner host can write that file from
     dmesg, which is not readable inside the CI container.
  3. Health probe: every card of the candidate group must allocate, compute and
     synchronize a small tensor within TSINGMICRO_CI_PROBE_TIMEOUT seconds in a
     subprocess. This catches cards left broken by a previous job ("ctx is
     busy", "failed to submit", "Kernel Run Timeout") before 8 TP ranks hang
     on them. A failed group is skipped and the next one is tried.

Allocation: TSINGMICRO_CI_CARDS (e.g. "8-31" or "8,9,10") limits the pool.

stdout: the chosen indices, comma separated (for TXDA_VISIBLE_DEVICES).
Diagnostics go to stderr.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

GROUP_SIZE = 8
# An idle TX8110 reports ~8.2 GB used (runtime reservation); a loaded TP rank
# reports tens of GB. Anything above this is treated as in use.
IDLE_MEM_LIMIT_MIB = int(os.environ.get("TSINGMICRO_CI_IDLE_MEM_MIB", "12000"))

_CHIP_RE = re.compile(
    r"^\|\s*(?:\d+\s+\S+\s+)?\|\s*(?P<chip>\d+)\s+(?P<bus>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d)"
    r"\s+\S+\s+\S+\s*/\s*\S+\s+(?P<used>\d+)M\s*/\s*(?P<total>\d+)M\s+(?P<util>\d+)%",
    re.IGNORECASE,
)
_PROC_RE = re.compile(
    r"^\|\s+(?P<chip>\d+)\s+(?P<pid>\d+)\s+(?P<name>\S+)\s+(?P<mem>\d+)M\s*\|"
)


def parse_tsm_smi(text: str) -> tuple[dict[int, dict], dict[int, list[int]]]:
    """Return ({chip: {bus, used, total, util}}, {chip: [pids]})."""
    chips: dict[int, dict] = {}
    procs: dict[int, list[int]] = {}
    in_procs = False
    for line in text.splitlines():
        if "Processes:" in line:
            in_procs = True
            continue
        if not in_procs:
            m = _CHIP_RE.match(line)
            if m:
                chips[int(m["chip"])] = {
                    "bus": m["bus"],
                    "used": int(m["used"]),
                    "total": int(m["total"]),
                    "util": int(m["util"]),
                }
        else:
            m = _PROC_RE.match(line)
            if m:
                procs.setdefault(int(m["chip"]), []).append(int(m["pid"]))
    return chips, procs


def parse_card_list(value: str | None) -> set[int] | None:
    """'8-15,20,22' -> {8..15, 20, 22}; None/''/'all' -> None (no limit)."""
    if value is None or value.strip().lower() in ("", "all"):
        return None
    cards: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cards.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cards.add(int(part))
        else:
            raise ValueError(f"invalid card list entry {part!r} in {value!r}")
    return cards


def bad_cards(environ) -> set[int]:
    bad = parse_card_list(environ.get("TSINGMICRO_CI_BAD_CARDS")) or set()
    path = environ.get("TSINGMICRO_CI_BAD_CARDS_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    bad |= parse_card_list(line) or set()
    return bad


def idle_cards(
    chips, procs, allowed=None, bad=frozenset(), mem_limit=IDLE_MEM_LIMIT_MIB
):
    return sorted(
        c
        for c, info in chips.items()
        if (allowed is None or c in allowed)
        and c not in bad
        and not procs.get(c)
        and info["used"] <= mem_limit
    )


def candidate_groups(idle, size=GROUP_SIZE):
    """Aligned contiguous groups first (best FlagCX ring locality), highest
    first (low cards are the usual default of interactive jobs), then other
    contiguous groups; never a non-contiguous group."""
    avail = set(idle)
    contiguous = [
        list(range(s, s + size))
        for s in sorted(avail, reverse=True)
        if set(range(s, s + size)) <= avail
    ]
    aligned = [g for g in contiguous if g[0] % size == 0]
    return aligned + [g for g in contiguous if g not in aligned]


_PROBE = r"""
import sys, torch, torch_txda  # noqa: F401
n = torch.txda.device_count()
if n < int(sys.argv[1]):
    sys.exit(f"only {n} TXDA devices visible")
for i in range(n):
    # The device must be current before D2H copies (.item()); otherwise the
    # copy is issued on the previous device's stream and the runtime rejects
    # it with "txMemcpyAsync ... Invalid parameters".
    torch.txda.set_device(i)
    x = torch.ones(256, 256, device=f"txda:{i}", dtype=torch.float32)
    y = (x @ x).sum().item()
    torch.txda.synchronize()
    # Free while this device is current: torch_txda's caching allocator
    # aborts ("invalid device pointer") when blocks of several devices are
    # released at interpreter exit with another device current.
    del x
    torch.txda.empty_cache()
    if y != 256.0 * 256 * 256:
        sys.exit(f"card index {i}: wrong result {y}")
print("probe ok", n)
"""


def probe_group(group, timeout):
    env = dict(
        os.environ, TXDA_VISIBLE_DEVICES=",".join(map(str, group)), TX_LOG_LEVEL="error"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE, str(len(group))],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout}s (card hang?)"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return False, " | ".join(tail)
    return True, r.stdout.strip()


def main() -> int:
    wait = int(os.environ.get("TSINGMICRO_CI_CARD_WAIT_SECONDS", "1800"))
    probe_timeout = int(os.environ.get("TSINGMICRO_CI_PROBE_TIMEOUT", "180"))
    allowed = parse_card_list(os.environ.get("TSINGMICRO_CI_CARDS"))
    bad = bad_cards(os.environ)
    failed_groups: set[tuple[int, ...]] = set()
    deadline = time.monotonic() + wait
    while True:
        out = subprocess.run(
            ["tsm_smi"], capture_output=True, text=True, timeout=60
        ).stdout
        chips, procs = parse_tsm_smi(out)
        if not chips:
            raise RuntimeError(
                "tsm_smi reported no TXDA cards (device nodes / driver?)"
            )
        idle = idle_cards(chips, procs, allowed, bad)
        busy = sorted(c for c in chips if procs.get(c))
        print(
            f"TXDA cards={len(chips)} idle={idle} busy={busy} excluded={sorted(bad)}",
            file=sys.stderr,
            flush=True,
        )
        for group in candidate_groups(idle):
            if tuple(group) in failed_groups:
                continue
            ok, msg = probe_group(group, probe_timeout)
            print(
                f"probe {group}: {'OK' if ok else 'FAILED'} {msg}",
                file=sys.stderr,
                flush=True,
            )
            if ok:
                print(",".join(map(str, group)))
                return 0
            failed_groups.add(tuple(group))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"::error::no healthy idle {GROUP_SIZE}-card TXDA group "
                f"(idle={idle}, failed probes={sorted(failed_groups)})",
                file=sys.stderr,
            )
            return 1
        print(
            f"waiting for a healthy idle {GROUP_SIZE}-card group...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(min(60, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
