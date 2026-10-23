#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert an audited profiler report into the supported three-file package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.operator_profiling.inventory_report import (  # noqa: E402
    generate_inventory_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render operator_list.csv, kernel_details_report.csv, and summary.json"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="directory containing the audited trace_report CSVs and profile_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_inventory_report(args.source_dir, args.output_dir)
    print(
        "OPERATOR_INVENTORY_FORMAT_PASS "
        f"output={args.output_dir} "
        f"events={summary['compute_kernel_event_count']}"
    )


if __name__ == "__main__":
    main()
