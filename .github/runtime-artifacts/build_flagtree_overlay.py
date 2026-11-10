#!/usr/bin/env python3
"""Build a file-level overlay between two installed FlagTree distributions."""

import argparse
import csv
from pathlib import Path, PurePosixPath


def read_record(path: Path) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    with path.open(newline="") as stream:
        for name, digest, size in csv.reader(stream):
            records[name] = (digest, size)
    return records


def source_and_artifact_path(record_path: str) -> tuple[str, str]:
    path = PurePosixPath(record_path)
    if record_path.startswith("../../../bin/"):
        name = path.name
        return f"/opt/kunlun/venv/bin/{name}", f"bin/{name}"
    return (
        f"/opt/kunlun/venv/lib/python3.10/site-packages/{record_path}",
        f"site-packages/{record_path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--runtime-record", type=Path, required=True)
    parser.add_argument("--copy-list", type=Path, required=True)
    parser.add_argument("--remove-list", type=Path, required=True)
    args = parser.parse_args()

    source = read_record(args.source_record)
    runtime = read_record(args.runtime_record)

    changed = sorted(
        path
        for path, metadata in source.items()
        if (metadata[0] and runtime.get(path) != metadata)
        or path.endswith(".dist-info/RECORD")
    )
    removed = sorted(path for path in runtime if path not in source)

    with args.copy_list.open("w") as stream:
        for path in changed:
            source_path, artifact_path = source_and_artifact_path(path)
            stream.write(f"{source_path}\t{artifact_path}\n")
    args.remove_list.write_text("\n".join(removed) + "\n")

    changed_bytes = sum(int(source[path][1]) for path in changed if source[path][1])
    print(f"changed/new files: {len(changed)} ({changed_bytes / 2**30:.2f} GiB)")
    print(f"removed files: {len(removed)}")


if __name__ == "__main__":
    main()
