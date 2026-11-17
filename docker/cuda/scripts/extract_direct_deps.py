#!/usr/bin/env python3
"""Extract the sglang wheel's direct dependencies as the top-level basis.

Reads the pinned sglang wheel's METADATA, keeps the runtime-relevant
(non-extra) Requires-Dist entries, and drops every name the operations
runtime already provides (from `pip freeze`): the runtime's torch /
numpy / CUDA libs must never re-enter the resolution graph, because a
resolver backtrack re-considers requirement edges and would replace the
runtime's +cu130 builds with public-index variants.

The output is a requirements list for the resolver: the wheel's declared
closure minus the runtime-provided roots, with markers evaluated for this
environment.

Usage: extract_direct_deps.py <wheel> <runtime-freeze> <output>
"""

import sys
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def runtime_relevant(entry: str) -> bool:
    req = Requirement(entry)
    if req.marker is None:
        return True
    return req.marker.evaluate({"extra": ""})


def main() -> int:
    wheel, freeze_path, output = (Path(a) for a in sys.argv[1:4])

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))

    runtime_names = {
        canonicalize_name(line.split("==", 1)[0])
        for line in freeze_path.read_text().splitlines()
        if "==" in line
    }

    lines = []
    for entry in metadata.get_all("Requires-Dist") or []:
        if not runtime_relevant(entry):
            continue
        requirement = Requirement(entry)
        if canonicalize_name(requirement.name) in runtime_names:
            continue
        lines.append(str(requirement))

    output.write_text("\n".join(lines) + "\n")
    print(
        f"direct additions: {len(lines)} requirements "
        f"({len(runtime_names)} runtime-provided names excluded)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
