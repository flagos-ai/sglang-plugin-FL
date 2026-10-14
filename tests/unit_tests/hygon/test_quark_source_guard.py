# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise HCU gates in real patched SGLang source without GPU dependencies.

Set SGLANG_HCU_UPSTREAM_SOURCE to the patched v0.5.18 source checkout. Without
that override, installed SGLang is inspected only on HCU. Other platforms skip
these HCU-specific patch checks. These tests
execute only import gates and constructor bodies from the actual source AST;
they neither inject AITER modules nor claim GPU kernel or full-import coverage.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULES = (
    (
        "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
        "QuarkW4A4MXFP4",
        ({}, {}),
    ),
    (
        "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py",
        "QuarkW4A4MXFp4MoE",
        ({}, {}),
    ),
    (
        "srt/layers/quantization/quark_int4fp8_moe.py",
        "QuarkInt4Fp8Config",
        (),
    ),
)


@pytest.fixture(params=MODULES, ids=("mxfp4_dense", "mxfp4_moe", "int4fp8_moe"))
def upstream_module(request):
    override = os.getenv("SGLANG_HCU_UPSTREAM_SOURCE")
    if override:
        root = Path(override)
        candidates = (root / "python/sglang", root / "sglang", root)
    else:
        torch = pytest.importorskip(
            "torch", reason="automatic HCU source checks require HCU PyTorch"
        )
        if not hasattr(torch, "__hcu_version__"):
            pytest.skip(
                "HCU source checks auto-detect installed SGLang only on HCU; "
                "set SGLANG_HCU_UPSTREAM_SOURCE for an explicit source check"
            )
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            pytest.skip("set SGLANG_HCU_UPSTREAM_SOURCE to patched v0.5.18 source")
        candidates = (Path(spec.origin).resolve().parent,)
    relative, class_name, constructor_args = request.param
    path = next(
        (root / relative for root in candidates if (root / relative).is_file()), None
    )
    if path is None:
        pytest.fail(f"could not find SGLang source for {relative} under {candidates}")
    return ast.parse(path.read_text(encoding="utf-8")), class_name, constructor_args


class AiterImportReached(Exception):
    pass


def import_gates(tree, *, hcu, hip):
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in ("_is_hip", "_is_hcu", "_use_aiter")
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.If) and any(
            isinstance(child, ast.ImportFrom)
            and (child.module == "aiter" or child.module.startswith("aiter."))
            for child in ast.walk(node)
        ):
            selected.append(node)
    assert selected, "upstream AITER import gates were not found"
    torch = SimpleNamespace(version=SimpleNamespace(hip="6.3.26113" if hip else None))
    if hcu:
        torch.__hcu_version__ = "3.6"

    def reject_aiter_import(name, *args, **kwargs):
        if name == "aiter" or name.startswith("aiter."):
            raise AiterImportReached(name)
        return builtins.__import__(name, *args, **kwargs)

    namespace = {
        "__builtins__": {**vars(builtins), "__import__": reject_aiter_import},
        "torch": torch,
        "is_hip": lambda: hip,
        # Enabling this flag must still not enable AMD kernels on HCU.
        "get_bool_env_var": lambda _name: True,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<upstream-gates>", "exec"),
        namespace,
    )
    return namespace


def test_hcu_hip_identity_is_preserved_without_amd_imports(upstream_module):
    tree, _, _ = upstream_module
    namespace = import_gates(tree, hcu=True, hip=True)
    assert namespace["_is_hip"] is True
    assert namespace["is_hip"]() is True
    assert namespace["torch"].version.hip == "6.3.26113"


def test_amd_still_reaches_its_original_aiter_dependency(upstream_module):
    tree, _, _ = upstream_module
    with pytest.raises(AiterImportReached, match="aiter"):
        import_gates(tree, hcu=False, hip=True)


def test_non_hip_does_not_import_amd_kernels(upstream_module):
    tree, _, _ = upstream_module
    namespace = import_gates(tree, hcu=False, hip=False)
    assert namespace["_is_hip"] is False


def test_selecting_unsupported_quark_on_hcu_fails_before_loading_weights(
    upstream_module,
):
    tree, class_name, constructor_args = upstream_module
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    constructor = copy.deepcopy(
        next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
    )
    # Evaluate the real constructor body, with annotations postponed exactly as
    # in upstream. No SGLang dependencies or AMD library stubs are imported.
    module = ast.parse("from __future__ import annotations\n")
    module.body.append(constructor)
    namespace = {"_is_hcu": True}
    exec(compile(module, "<upstream-constructor>", "exec"), namespace)
    with pytest.raises(NotImplementedError, match="AMD AITER.*not supported on HCU"):
        namespace["__init__"](object(), *constructor_args)
