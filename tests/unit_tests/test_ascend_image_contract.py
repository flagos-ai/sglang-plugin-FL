# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Static and executable contracts for the Ascend two-wheel image recipe.

The CI image is the operations Runtime plus exactly the two application
wheels; every assertion here pins a decision a reviewer cannot see from the
diff alone: the containerfile stays assembly-only, the resolver cannot
replace a Runtime distribution, the wheel acceptance covers the ascend
payload, and the job-time verifier asserts plugin-owned facts only.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml

_ROOT = Path(__file__).parents[2]
_CONTAINERFILE = _ROOT / "docker" / "ascend" / "runtime-0.5.18.containerfile"
_DOCKERIGNORE = Path(f"{_CONTAINERFILE}.dockerignore")
_BUILD_SCRIPT = _ROOT / "docker" / "ascend" / "build_two_wheel.sh"
_WHEEL_CHECK = _ROOT / "docker" / "ascend" / "scripts" / "check_plugin_wheel.py"
_SETUP = _ROOT / ".github" / "scripts" / "ascend" / "setup.sh"
_VERIFIER = _ROOT / ".github" / "scripts" / "ascend" / "verify_environment.py"
_ASCEND_CONFIG = _ROOT / ".github" / "configs" / "ascend.yml"
_PLATFORM_CONFIG = _ROOT / "tests" / "platforms" / "ascend.yaml"
_PYPROJECT = _ROOT / "pyproject.toml"
_RUNTIME_TAG = (
    "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1-910c:2.3.0"
)


def test_containerfile_is_assembly_only() -> None:
    containerfile = _CONTAINERFILE.read_text(encoding="utf-8")

    # The Runtime tag is a moving tag owned by the runtime team; the build
    # script records the resolved digest.
    assert f"ARG BASE_IMAGE={_RUNTIME_TAG}" in containerfile
    # One resolver install of the two application wheels, with the Runtime's
    # own freeze as the only constraints input. No version constraint is
    # written in the containerfile; dependency bounds live in wheel metadata.
    assert "uv pip install" in containerfile
    assert "sglang-0.5.18-py3-none-any.whl[srt_empty,runtime_common]" in containerfile
    assert "sglang_fl-0.1.0-py3-none-any.whl[ascend]" in containerfile
    assert "runtime-constraints.txt" in containerfile
    assert re.search(r":=.*==|==\d", containerfile) is None
    # Nothing the Runtime owns may be replaced, moved or re-exposed
    # (instructions only; comments may document why the slot stays as is).
    for instruction in containerfile.splitlines():
        if re.match(r"\s*(RUN|COPY|ADD|ENV|WORKDIR)\b", instruction):
            assert "/opt/flagtree" not in instruction, instruction
            assert "uninstall" not in instruction, instruction
    assert "pip install --force-reinstall" not in containerfile
    assert "WORKDIR" not in containerfile
    # The plugin is never installed from the checkout.
    assert "COPY pyproject.toml" not in containerfile
    assert "COPY sglang_fl" not in containerfile
    # No platform-artifact wheel (sgl-kernel / FlagGems) is installed here;
    # the sgl_kernel_npu gap is carried by the plugin's placeholder+fallbacks.
    for instruction in containerfile.splitlines():
        if re.match(r"\s*(RUN|COPY|ADD)\b", instruction):
            assert "sgl-kernel" not in instruction
            assert "flag_gems" not in instruction


def test_containerfile_dockerignore_ships_only_the_staging_contract() -> None:
    dockerignore = _DOCKERIGNORE.read_text(encoding="utf-8")

    assert "**" in dockerignore
    assert "runtime-0.5.18.containerfile" in dockerignore
    assert "artifacts/" in dockerignore


def test_build_script_pins_its_inputs() -> None:
    build = _BUILD_SCRIPT.read_text(encoding="utf-8")

    # The SGLang input wheel is built from (or verified against) the pinned
    # upstream commit; the builder is digest-pinned so two builds of one
    # commit produce identical wheels.
    assert "SGLANG_COMMIT=" in build
    assert "python:3.12-slim@sha256:" in build
    assert "SOURCE_DATE_EPOCH" in build
    assert "staging-record.txt" in build
    # The Runtime's constraints are read from the exact base the build uses.
    assert "pip freeze --all" in build
    assert "imagetools inspect" in build


def test_wheel_check_covers_the_ascend_contract() -> None:
    check = _WHEEL_CHECK.read_text(encoding="utf-8")

    assert 'EXPECTED_EXTRA = ("sglang==0.5.18", "pyarrow<26")' in check
    assert '"numpy", "triton"' in check
    assert "sglang_fl:activate_platform" in check
    assert "sglang_fl:load_plugin" in check
    # The gap-filling payload the wheel must ship.
    for payload in (
        "vendor/ascend/import_guard.py",
        "vendor/ascend/sgl_kernel_npu_fallbacks.py",
        "vendor/ascend/early.py",
        "config/ascend.yaml",
        "config/ascend-cann921.yaml",
    ):
        assert payload in check


def test_pyproject_declares_the_ascend_extra() -> None:
    pyproject = _PYPROJECT.read_text(encoding="utf-8")

    assert 'ascend = ["sglang==0.5.18", "pyarrow<26"]' in pyproject
    # pyarrow 26 imports only on NumPy 2 without declaring that requirement;
    # the Runtime provides NumPy 1.x. The bound lives in wheel metadata, not
    # in the containerfile.
    assert "pyarrow<26" not in _CONTAINERFILE.read_text(encoding="utf-8")


def test_setup_installs_only_the_pinned_test_runner() -> None:
    setup = _SETUP.read_text(encoding="utf-8")

    assert "test-requirements.txt" in setup
    assert "SGLANG_FL_TEST_SOURCE=wheel" in setup
    assert "verify_environment.py" in setup
    # The checkout is never installed over the image's wheel.
    assert "pip install -e" not in setup
    assert "--no-deps --no-build-isolation -e" not in setup


def test_verifier_asserts_plugin_owned_facts_only() -> None:
    verifier = _VERIFIER.read_text(encoding="utf-8")

    assert 'EXPECTED_WHEELS = {"sglang": "0.5.18", "sglang-fl": "0.1.0"}' in verifier
    # Runtime-owned versions are printed, never pinned as expectations.
    assert "EXPECTED_RUNTIME_DISTRIBUTIONS" not in verifier
    assert "EXPECTED_FLAGTREE_VERSION" not in verifier
    assert "torchair" not in verifier
    assert "runtime CANN=" in verifier
    # The placeholder contract is asserted together with the op fallback.
    assert "npu::causal_conv1d" in verifier
    assert "fused_rope_qk_mqa" in verifier


def test_ci_config_keeps_the_validated_container_contract() -> None:
    config = yaml.safe_load(_ASCEND_CONFIG.read_text(encoding="utf-8"))
    options = _ASCEND_CONFIG.read_text(encoding="utf-8").split(
        "container_options:", maxsplit=1
    )[1]

    assert config["ci_image"].startswith(
        "harbor.baai.ac.cn/plugin/sglang-plugin-fl@sha256:"
    )
    assert "flagcicd-910c" in config["runner_labels"]
    assert "/data/models/Qwen" in str(config["container_volumes"])
    for required in (
        "--runtime=ascend",
        "ASCEND_VISIBLE_DEVICES=0,1,2,3",
        "GLOO_SOCKET_IFNAME=lo",
        "HCCL_HOST_SOCKET_PORT_RANGE=auto",
    ):
        assert required in options
    assert "--privileged" not in options
    # The old image's expected-CANN knob died with the baked verifier.
    assert "SGLANG_FL_ASCEND_EXPECTED_CANN" not in options


def test_platform_matrix_stays_nograph_for_the_two_wheel_image() -> None:
    platform = yaml.safe_load(_PLATFORM_CONFIG.read_text(encoding="utf-8"))
    pr_cases = platform["910c"]["tests"]["e2e"]["pr"]

    assert set(pr_cases) == {"inference", "serving"}
    # The GDN (qwen3_6) cases pin the eager mode this image validates; the
    # dense qwen3 cases have no graph variant in their names.
    for model, cases in pr_cases["inference"].items():
        if model == "qwen3_6":
            assert all(case.endswith("_nograph") for case in cases), cases


def test_gdn_workarounds_stay_gated_on_the_affected_flagtree_versions() -> None:
    runtime = (
        _ROOT
        / "sglang_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "runtime.py"
    ).read_text(encoding="utf-8")

    assert '"0.7.0+ascend3.5"' in runtime
    assert '"0.7.1a1+ascend3.5"' in runtime


def test_legacy_recipes_are_untouched() -> None:
    # The CANN 8.5 and 9.2.1 source-build recipes remain for their own image
    # lines; this PR only adds the two-wheel runtime recipe and retargets CI.
    assert (_ROOT / "docker" / "ascend" / "empty-0.5.18.containerfile").exists()
    assert (_ROOT / "docker" / "ascend" / "cann921-0.5.18.containerfile").exists()
