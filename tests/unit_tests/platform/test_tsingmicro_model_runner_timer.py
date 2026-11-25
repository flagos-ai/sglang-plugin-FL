# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA per-iteration timer: token count for every forward mode, incl. MTP target verify."""

from types import SimpleNamespace

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import model_runner as mr


class _Mode:
    def __init__(self, decode):
        self._decode = decode

    def is_decode(self):
        return self._decode


def test_decode_counts_batch_size() -> None:
    fb = SimpleNamespace(forward_mode=_Mode(True), batch_size=3, extend_num_tokens=None)
    assert mr._extract_num_tokens(fb) == 3


def test_extend_counts_new_tokens() -> None:
    fb = SimpleNamespace(forward_mode=_Mode(False), extend_num_tokens=17)
    assert mr._extract_num_tokens(fb) == 17


def test_target_verify_with_seq_lens_sum_none() -> None:
    # SGLang 0.5.18 EAGLE/MTP target verify: no extend_num_tokens, seq_lens_sum None.
    fb = SimpleNamespace(
        forward_mode=_Mode(False),
        extend_num_tokens=None,
        seq_lens_sum=None,
        input_ids=torch.zeros(8, dtype=torch.int64),
    )
    assert mr._extract_num_tokens(fb) == 8


def test_no_token_info_returns_zero() -> None:
    fb = SimpleNamespace(forward_mode=None, extend_num_tokens=None, seq_lens_sum=None)
    assert mr._extract_num_tokens(fb) == 0
