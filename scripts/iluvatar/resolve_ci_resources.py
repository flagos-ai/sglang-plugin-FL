# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit Iluvatar CI resources from the shared platform configuration."""

import json
import os
from pathlib import PurePosixPath
import re
import shlex


def resolve_resources(config, *, scope="full", overrides=None):
    overrides = overrides or {}
    if config.get("platform") != "iluvatar" or scope not in ("full", "operators"):
        raise ValueError("Require an Iluvatar config and full or operators scope")
    labels = overrides.get("runner_labels") or config.get("runner_labels")
    if isinstance(labels, str):
        labels = json.loads(labels)
    if (
        not isinstance(labels, list)
        or not labels
        or not all(
            isinstance(label, str)
            and label.strip()
            and "\n" not in label
            and "\r" not in label
            for label in labels
        )
    ):
        raise ValueError(
            "runner_labels must be a non-empty list of explicit label names"
        )
    image = str(overrides.get("ci_image") or config.get("ci_image", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]+", image):
        raise ValueError("Require an accessible published container reference")
    devices = str(
        overrides.get("cuda_visible_devices") or config.get("cuda_visible_devices", "")
    ).strip()
    if not re.fullmatch(r"\d+(,\d+)*", devices):
        raise ValueError("Explicit numeric GPU allocation is required")
    ids = devices.split(",")
    if len(set(ids)) != len(ids) or (scope == "full" and len(ids) < 4):
        raise ValueError("Full acceptance needs at least four unique allocated GPUs")
    volumes = list(config.get("container_volumes", []))
    # The standard config already mounts /models. A manual override replaces
    # that target, and operator-only checks do not require any model mount.
    volumes = [
        volume
        for volume in volumes
        if not isinstance(volume, str)
        or len(volume.split(":")) < 2
        or volume.split(":")[1] != "/models"
    ]
    if scope == "full":
        models = str(
            overrides.get("models_path") or config.get("models_path", "")
        ).strip()
        if not PurePosixPath(models).is_absolute() or not re.fullmatch(
            r"/[A-Za-z0-9_./-]+", models
        ):
            raise ValueError(
                "Full acceptance needs an absolute host models_path without shell characters"
            )
        volumes.append(f"{models}:/models:ro")
    if not all(
        isinstance(volume, str) and re.fullmatch(r"/[A-Za-z0-9_./:-]+", volume)
        for volume in volumes
    ):
        raise ValueError("Require absolute, literal Linux container mounts")
    options = str(config.get("container_options", "")).strip()
    if "\n" in options or "\r" in options:
        raise ValueError("Container options must be a single line")
    # Keep the explicit allocation consistent with the standard container env,
    # including when a manual dispatch changes the configured GPU indices.
    tokens = shlex.split(options)
    kept = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("--env", "-e") and index + 1 < len(tokens):
            if tokens[index + 1].startswith("CUDA_VISIBLE_DEVICES="):
                index += 2
                continue
        if token.startswith("--env=CUDA_VISIBLE_DEVICES="):
            index += 1
            continue
        kept.append(token)
        index += 1
    kept.extend(["--env", f"CUDA_VISIBLE_DEVICES={devices}"])
    kept.extend(f"--volume={volume}" for volume in volumes)
    options = shlex.join(kept)
    return {
        "runner_labels": json.dumps(sorted(set(labels))),
        "ci_image": image,
        "cuda_visible_devices": devices,
        "container_options": options,
    }


if __name__ == "__main__":
    resolved = resolve_resources(
        json.loads(os.environ["PLATFORM_CONFIG"]),
        scope=os.environ.get("ACCEPTANCE_SCOPE", "full"),
        overrides={
            key: os.environ.get(key.upper(), "")
            for key in (
                "runner_labels",
                "ci_image",
                "cuda_visible_devices",
                "models_path",
            )
        },
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf8") as output:
        for key, value in resolved.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(resolved, indent=2))
