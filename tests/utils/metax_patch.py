# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load the MetaX patch module without activating its patches."""

import ast
import importlib.util
from pathlib import Path

PATCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "sglang_fl/dispatch/backends/vendor/metax/patch.py"
)
PACKAGE = "sglang_fl.dispatch.backends.vendor.metax"


def load_metax_patch():
    """Compile the unchanged definitions, omitting the final activation call.

    Production imports still run ``apply_metax_patches()``. Unit tests exercise
    the loader logic with mocks and must not patch SGLang for real: on hosts
    with flashinfer installed (e.g. the CUDA runner) that would try to apply
    every MetaX patch during collection.
    """
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"), filename=str(PATCH_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_metax_patches"
    ]
    if (
        len(calls) != 1
        or not isinstance(tree.body[-1], ast.Expr)
        or tree.body[-1].value is not calls[0]
        or calls[0].args
        or calls[0].keywords
    ):
        raise RuntimeError("Expected one final no-argument apply_metax_patches() call")
    tree.body.pop()
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}._unit_patch", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    exec(compile(tree, str(PATCH_PATH), "exec"), module.__dict__)
    return module
