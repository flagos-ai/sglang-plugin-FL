# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build and activate the opt-in native capture status/id query bridge."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def prepare(artifact_root, activation_file=None, github_env=None):
    enabled = (
        os.environ.get("GEMS_VENDOR") == "iluvatar"
        and os.environ.get("SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT") == "1"
    )
    source = Path(__file__).with_name("native") / "capture_info_compat.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact_root = Path(artifact_root).resolve()
    output = artifact_root / digest
    output.mkdir(parents=True, exist_ok=True)
    library = output / "libiluvatar_capture_info_compat.so"
    if not library.is_file():
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler:
            raise ValueError("CC must name a local C compiler")
        descriptor, temporary_name = tempfile.mkstemp(suffix=".so", dir=output)
        os.close(descriptor)
        temporary = Path(temporary_name)
        command = [
            *compiler,
            "-shared",
            "-fPIC",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(temporary),
            "-ldl",
        ]
        log = temporary.with_suffix(".build.log")
        try:
            with log.open("w", encoding="utf-8") as stream:
                stream.write(shlex.join(command) + "\n")
                stream.flush()
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise RuntimeError(f"Native capture-info build failed; see {log}")
            os.replace(temporary, library)
            (output / "build.json").write_text(
                json.dumps(
                    {
                        "source_sha256": digest,
                        "command": command,
                        "library": str(library),
                        "build_log": str(log),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            temporary.unlink(missing_ok=True)

    activation = Path(activation_file or artifact_root / "activation.env").resolve()
    activation.parent.mkdir(parents=True, exist_ok=True)
    activation_text = (
        "# Source in Bash before starting Python; no accelerator code runs here.\n"
    )
    if enabled:
        activation_text += (
            'export GEMS_VENDOR="${GEMS_VENDOR:-iluvatar}"\n'
            'export SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT="${SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT:-1}"\n'
            'if [[ "${GEMS_VENDOR:-}" == iluvatar && '
            '"${SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT:-}" == 1 ]]; then\n'
            "    _sglang_iluvatar_capture_info_library="
            + shlex.quote(str(library))
            + "\n"
            '    _sglang_iluvatar_capture_info_existing="${LD_PRELOAD:-}"\n'
            '    case ":${_sglang_iluvatar_capture_info_existing//[[:space:]]/:}:" in\n'
            '        *":${_sglang_iluvatar_capture_info_library}:"*) ;;\n'
            '        *) export LD_PRELOAD="${_sglang_iluvatar_capture_info_library}'
            '${LD_PRELOAD:+:$LD_PRELOAD}" ;;\n'
            "    esac\n"
            "    unset _sglang_iluvatar_capture_info_library _sglang_iluvatar_capture_info_existing\n"
            "fi\n"
        )
    else:
        activation_text += "# Inactive: rerun preparation with both Iluvatar opt-in variables enabled.\n"
    activation.write_text(activation_text, encoding="utf-8")

    if github_env and enabled:
        previous = os.environ.get("LD_PRELOAD", "")
        preload = (
            previous
            if str(library) in re.split(r"[:\s]+", previous)
            else (str(library) + (":" + previous if previous else ""))
        )
        if "\n" in preload or "\r" in preload:
            raise ValueError("Cannot write a multiline LD_PRELOAD to GITHUB_ENV")
        with Path(github_env).open("a", encoding="utf-8") as stream:
            stream.write("GEMS_VENDOR=iluvatar\n")
            stream.write("SGLANG_ILUVATAR_CAPTURE_INFO_COMPAT=1\n")
            stream.write("LD_PRELOAD=" + preload + "\n")
    print(f"Native capture-info library: {library}")
    print(f"Source activation before starting Python: {activation}")
    return activation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/iluvatar/native")
    )
    parser.add_argument("--activation-file", type=Path)
    parser.add_argument("--github-env", default=os.environ.get("GITHUB_ENV"))
    args = parser.parse_args()
    try:
        prepare(args.artifact_root, args.activation_file, args.github_env)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Capture-info compatibility activation failed: {error}\n")


if __name__ == "__main__":
    main()
