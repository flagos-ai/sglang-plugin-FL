# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject empty, skipped, or failed JUnit results in explicit Iluvatar CI gates."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check_results(path: Path, min_tests: int = 1) -> int:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    )
    print(
        f"Executed contract results: {len(cases)} tests, {skipped} skipped, {failed} failed"
    )
    if len(cases) < min_tests or skipped or failed:
        raise RuntimeError(
            f"Require at least {min_tests} test cases, zero skips and zero failures"
        )
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-tests", type=int, default=1)
    args = parser.parse_args()
    check_results(args.report, args.min_tests)
