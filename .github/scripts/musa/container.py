"""Run MUSA phases in one verified image, with cleanup on every job outcome."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def container_name():
    return f"musa-ci-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"


def forwarded_env(names):
    args = []
    for name in names:
        if name in os.environ:
            args.extend(["--env", f"{name}={os.environ[name]}"])
    return args


def start():
    config = json.loads(os.environ["MUSA_CI_CONFIG"])
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    # check.sh/setup.sh write Actions environment files in this shared mount.
    Path(os.environ["GITHUB_ENV"]).resolve().relative_to(runner_temp)
    container_home = runner_temp / container_name() / "home"
    container_home.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "--detach", "--name", container_name()]
    args.extend(shlex.split(config["container_options"]))
    for volume in config["container_volumes"]:
        args.extend(["--volume", volume])
    args.extend(
        [
            "--volume",
            f"{workspace}:/workspace",
            "--volume",
            f"{runner_temp}:{runner_temp}",
            "--volume",
            f"{container_home}:/github/home",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/github/home",
        ]
    )
    args.extend(forwarded_env(["MTHREADS_VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES"]))
    args.extend(
        [
            "--entrypoint",
            "/bin/bash",
            os.environ["MUSA_CI_IMAGE_ID"],
            "-c",
            "exec sleep infinity",
        ]
    )
    subprocess.run(args, check=True)


def execute(command):
    if not command:
        raise ValueError("A container command is required")
    args = ["docker", "exec"]
    args.extend(forwarded_env(["GITHUB_ENV", "MUSA_VISIBLE_DEVICES", "PYTHONPATH"]))
    return subprocess.call([*args, container_name(), *command])


def stop():
    found = subprocess.run(
        ["docker", "container", "inspect", container_name()], capture_output=True
    )
    if found.returncode == 0:
        subprocess.run(["docker", "rm", "--force", container_name()], check=True)


if __name__ == "__main__":
    operation = sys.argv[1]
    if operation == "start":
        start()
    elif operation == "exec":
        sys.exit(execute(sys.argv[2:]))
    elif operation == "stop":
        stop()
    else:
        raise ValueError(f"Unknown operation: {operation}")
