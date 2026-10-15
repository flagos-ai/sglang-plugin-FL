# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep CPU helper loading separate from production import-time activation."""

import pytest

from tests.utils import musa_patch


def test_real_definitions_load_without_activating_musa():
    module = musa_patch.load_musa_patch()
    assert module._patches_applied is False
    assert module.__package__ == musa_patch.PACKAGE
    assert module.__file__ == str(musa_patch.PATCH_PATH)
    for name in (
        "apply_musa_patches",
        "_patch_flagcx_fp32_all_reduce",
        "_patch_multimodal_mask",
        "_patch_pp_send_recv_order",
    ):
        assert callable(getattr(module, name))


def test_helper_function_bodies_and_relative_import_context_are_preserved(
    monkeypatch, tmp_path
):
    path = tmp_path / "patch.py"
    path.write_text(
        "def helper():\n"
        "    from .runtime_config import needs_gdn_packed_decode_backport\n"
        "    return __file__, __package__, needs_gdn_packed_decode_backport('0.5.18')\n"
        "def apply_musa_patches():\n    raise RuntimeError('activation executed')\n"
        "apply_musa_patches()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(musa_patch, "PATCH_PATH", path)
    module = musa_patch.load_musa_patch()
    assert module.helper() == (str(path), musa_patch.PACKAGE, True)
    with pytest.raises(RuntimeError, match="activation executed"):
        module.apply_musa_patches()


@pytest.mark.parametrize(
    "ending",
    [
        "",
        "apply_musa_patches(1)\n",
        "apply_musa_patches(force=True)\n",
        "apply_musa_patches()\napply_musa_patches()\n",
        "apply_musa_patches()\nvalue = 1\n",
        "if True:\n    apply_musa_patches()\n",
        "if True:\n    apply_musa_patches()\napply_musa_patches()\n",
    ],
)
def test_changed_activation_layout_fails_explicitly(monkeypatch, tmp_path, ending):
    path = tmp_path / "patch.py"
    path.write_text("def apply_musa_patches():\n    pass\n" + ending, encoding="utf-8")
    monkeypatch.setattr(musa_patch, "PATCH_PATH", path)
    with pytest.raises(RuntimeError, match="one final no-argument"):
        musa_patch.load_musa_patch()
