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

"""Check whether the active environment can run operator profiling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# Match the real collection process before importing SGLang or its plugin.
os.environ["SGLANG_FL_MODE"] = "platform_profile"
os.environ["SGLANG_PLATFORM"] = "sglang_fl"
os.environ["SGLANG_PLUGINS"] = "sglang_fl"
os.environ["USE_FLAGGEMS"] = "0"

from tools.operator_profiling.environment import (  # noqa: E402
    ProfilingEnvironmentError,
    validate_profiling_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate operator-profiling runtime prerequisites"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument(
        "--require-validated-versions",
        action="store_true",
        help="reject versions outside the H20 combination documented by this tool",
    )
    args = parser.parse_args()
    try:
        report = validate_profiling_environment(
            model_path=args.model_path,
            tp_size=args.tp_size,
            repository_root=_REPOSITORY_ROOT,
            require_validated_versions=args.require_validated_versions,
        )
    except ProfilingEnvironmentError as error:
        print(f"OPERATOR_PROFILING_PREFLIGHT_FAIL\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("OPERATOR_PROFILING_PREFLIGHT_PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
