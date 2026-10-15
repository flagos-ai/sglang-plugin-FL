# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load real MUSA patch helpers without activating a foreign vendor runtime."""

import ast
import importlib.util
from pathlib import Path

PATCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "sglang_fl/dispatch/backends/vendor/mthreads/patch.py"
)
PACKAGE = "sglang_fl.dispatch.backends.vendor.mthreads"


def load_musa_patch():
    """Compile the unchanged definitions, omitting one explicit activation call.

    Production imports still execute that call. CPU tests invoke individual
    patch functions explicitly and must not activate every MUSA patch while
    pytest collects tests on another hardware backend.
    """
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"), filename=str(PATCH_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_musa_patches"
    ]
    if (
        len(calls) != 1
        or not isinstance(tree.body[-1], ast.Expr)
        or tree.body[-1].value is not calls[0]
        or calls[0].args
        or calls[0].keywords
    ):
        raise RuntimeError("Expected one final no-argument apply_musa_patches() call")
    tree.body.pop()
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}._unit_patch", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    exec(compile(tree, str(PATCH_PATH), "exec"), module.__dict__)
    return module
