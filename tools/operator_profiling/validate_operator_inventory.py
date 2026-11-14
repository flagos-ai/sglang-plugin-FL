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

"""Validate a serialized three-file operator inventory package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.operator_profiling.inventory_report import (  # noqa: E402
    validate_inventory_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an operator inventory results directory"
    )
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    validation = validate_inventory_package(args.results_dir)
    print(
        "OPERATOR_INVENTORY_VALIDATION_PASS "
        f"output={args.results_dir} checks={len(validation)}"
    )


if __name__ == "__main__":
    main()
