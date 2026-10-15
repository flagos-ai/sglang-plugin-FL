# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""HCU CLI registration must precede parsing and leave kernels lazy."""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
HCU_DIR = ROOT / "sglang_fl/dispatch/backends/vendor/hygon"


class KernelImportRequested(ImportError):
    pass


@pytest.fixture
def registration_env(monkeypatch):
    choices = ["triton", "torch_native", "aiter"]
    factories = {}
    events = []
    imported_kernels = []
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ATTENTION_BACKEND_CHOICES = choices

    def add_choices(values):
        events.append(("choices", tuple(values)))
        choices.extend(values)

    server_args.add_attention_backend_choices = add_choices
    registry = ModuleType("sglang.srt.layers.attention.attention_registry")

    def register_backend(name):
        def decorator(factory):
            factories[name] = factory
            return factory

        return decorator

    registry.register_attention_backend = register_backend
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    package_name = "_hcu_cli_registration"
    package = ModuleType(package_name)
    package.__path__ = [str(HCU_DIR)]
    monkeypatch.setitem(sys.modules, package_name, package)

    def stub(relative_name, **attributes):
        module = ModuleType(package_name + "." + relative_name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    stub(
        "patches.fla_dispatch_filter",
        patch_fla_dispatch_filter=lambda: events.append(("fla",)),
    )
    stub(
        "patches.pp_output_launch_dependency",
        setup_pp_output_launch_dependency=lambda: events.append(("pp",)),
    )
    stub("lifecycle", apply_hcu_lifecycle_patches=lambda: events.append(("lifecycle",)))
    original_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if name.startswith(
            (
                "sglang_fl.dispatch.backends.vendor.hygon.impl",
                "sglang.srt.layers.attention.triton_backend",
                "flash_attn",
            )
        ):
            imported_kernels.append(name)
            raise KernelImportRequested(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard_import)

    def load(name):
        qualified = package_name + "." + name
        if qualified in sys.modules:
            return sys.modules[qualified]
        spec = importlib.util.spec_from_file_location(
            qualified, HCU_DIR / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(
        choices=choices,
        factories=factories,
        events=events,
        imported_kernels=imported_kernels,
        load=load,
    )


@pytest.mark.parametrize(
    "option",
    ["attention-backend", "prefill-attention-backend", "decode-attention-backend"],
)
def test_three_cli_choices_accept_hcu_even_if_parser_already_references_list(
    registration_env, option
):
    env = registration_env
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + option, choices=env.choices)
    # v0.5.18 ServerArgs Arg metadata shares ATTENTION_BACKEND_CHOICES by reference.
    # Extending via its public function must update an existing parser too.
    env.load("register_platform")
    parsed = parser.parse_args(["--" + option, "hcu"])
    assert getattr(parsed, option.replace("-", "_")) == "hcu"
    assert env.imported_kernels == []
    assert callable(env.factories["hcu"])


def test_registration_is_idempotent_and_preserves_other_choices(registration_env):
    env = registration_env
    module = env.load("register_platform")
    factory = env.factories["hcu"]
    for _ in range(5):
        module.register_hcu_attention_backend()
    assert env.choices == ["triton", "torch_native", "aiter", "hcu"]
    assert env.factories["hcu"] is factory
    assert [event for event in env.events if event[0] == "choices"] == [
        ("choices", ("hcu",))
    ]
    assert env.imported_kernels == []


def test_existing_cli_choice_still_registers_factory_without_duplicates(
    registration_env,
):
    env = registration_env
    env.choices.append("hcu")
    env.load("register_platform")
    assert env.choices.count("hcu") == 1
    assert callable(env.factories["hcu"])
    assert env.events == []


def test_kernel_import_happens_only_when_factory_is_invoked(registration_env):
    env = registration_env
    env.load("register_platform")
    assert env.imported_kernels == []
    runner = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
    with pytest.raises(KernelImportRequested, match="hygon.impl.attention_backend"):
        env.factories["hcu"](runner)
    assert len(env.imported_kernels) == 1


def test_cross_attention_rejection_does_not_import_kernels(registration_env):
    env = registration_env
    env.load("register_platform")
    runner = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True))
    with pytest.raises(AssertionError, match="Cross attention"):
        env.factories["hcu"](runner)
    assert env.imported_kernels == []


def test_vendor_patch_explicitly_registers_before_other_patch_work(registration_env):
    env = registration_env
    env.load("register_platform")
    # Prove apply_hcu_patches itself calls registration, independently of the
    # register_platform module's existing import-time compatibility behavior.
    env.choices.remove("hcu")
    env.factories.clear()
    env.events.clear()
    patch = env.load("patch")
    assert env.choices.count("hcu") == 1
    assert callable(env.factories["hcu"])
    assert env.events == [("choices", ("hcu",)), ("fla",), ("pp",), ("lifecycle",)]
    patch.apply_hcu_patches()
    assert env.choices.count("hcu") == 1
    assert env.events == [("choices", ("hcu",)), ("fla",), ("pp",), ("lifecycle",)]
    assert env.imported_kernels == []


@pytest.mark.parametrize("vendor", ["amd", "nvidia", "mthreads"])
def test_other_vendor_dispatch_does_not_load_hcu_registration(
    monkeypatch, registration_env, vendor
):
    env = registration_env
    tree = ast.parse((ROOT / "sglang_fl/__init__.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_vendor_patches"
    )
    utils = ModuleType("sglang_fl.utils")
    utils.get_device_info = lambda: SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    imported = []

    def import_vendor(name):
        imported.append(name)
        return ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_vendor)
    namespace = {"logger": logging.getLogger("hcu_cli_vendor_test")}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<vendor-dispatch>", "exec"
        ),
        namespace,
    )
    namespace["_apply_vendor_patches"]()
    assert imported == [f"sglang_fl.dispatch.backends.vendor.{vendor}.patch"]
    assert "hcu" not in env.choices
    assert env.factories == {}
