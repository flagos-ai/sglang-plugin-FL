# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Candidate parity and score lifetime through the model's replacement method."""

import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location(
    "isolated_prefill", ROOT / "sglang_fl/dsv41_runtime/prefill.py"
)
prefill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefill)


def mock_native(name, value):
    modules = {}
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        path = ".".join(parts[:index])
        module = types.ModuleType(path)
        module.__path__ = []
        modules[path] = module
        if index > 1:
            setattr(modules[".".join(parts[: index - 1])], parts[index - 1], module)
    modules[name].__dict__.update(vars(value))
    return patch.dict(sys.modules, modules)


def reference(logits, lens, count, block):
    width = logits.shape[-1]
    scores = F.pad(logits, (0, -width % block), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block)).amax(-1)
    blocks = scores.shape[-1]
    scores = scores.masked_fill(torch.arange(blocks) == (lens - 1) // block, torch.inf)
    top = scores.topk(min(count, blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    ids = (
        torch.where(top.values > -torch.inf, top.indices, blocks).int().sort(-1).values
    )
    return keep.repeat_interleave(block, dim=-1)[..., :width], ids


class PrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_selector_parity_with_strides_ties_nonfinite_and_partial_tails(self):
        sizes = (
            1,
            3,
            4,
            7,
            8,
            9,
            15,
            16,
            17,
            127,
            128,
            129,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            4095,
            4096,
            4097,
        )
        generator = torch.Generator().manual_seed(739101)
        cases = 0
        for dtype in (torch.float32, torch.bfloat16):
            for rows in (1, 3, 8):
                for width in sizes:
                    for block in (4, 8, 16):
                        for pattern in ("random", "ties", "sentinels", "strided"):
                            x = torch.randn(
                                rows,
                                width * (2 if pattern == "strided" else 1),
                                dtype=dtype,
                                generator=generator,
                            )
                            if pattern == "strided":
                                x = x[:, ::2]
                            elif pattern == "ties":
                                x.zero_()
                                x[:, ::3] = -0.0
                            elif pattern == "sentinels":
                                x[:, ::3] = -torch.inf
                                x[:, ::7] = torch.inf
                                x[:, ::11] = torch.nan
                            lens = torch.tensor(
                                [(i * 73) % (width + 1) for i in range(rows)],
                                dtype=torch.int32,
                            )[:, None]
                            for count in (1, 3, 16):
                                mask, ids = reference(x, lens, count, block)
                                self.assertTrue(
                                    torch.equal(
                                        prefill.candidate_blocks(x, lens, count, block),
                                        mask,
                                    )
                                )
                                self.assertTrue(
                                    torch.equal(
                                        prefill.candidate_block_ids(
                                            x, lens, count, block
                                        ),
                                        ids,
                                    )
                                )
                                cases += 1
        self.assertEqual(cases, 4536)

    def test_aligned_width_never_calls_pad(self):
        with patch.object(prefill.F, "pad", wraps=F.pad) as padding:
            prefill._candidate_block_amax(torch.randn(2, 64), 8)
            padding.assert_not_called()

    def test_only_partial_tail_is_padded(self):
        with patch.object(prefill.F, "pad", wraps=F.pad) as padding:
            prefill._candidate_block_amax(torch.randn(2, 65), 8)
            self.assertEqual(padding.call_count, 1)
            self.assertEqual(padding.call_args.args[0].shape, (2, 1))

    def test_extend_releases_score_before_allocating_next_chunk(self):
        page_indices = torch.empty(4, 2, dtype=torch.int32)
        raw_indices = torch.empty_like(page_indices)
        core = types.SimpleNamespace(
            sparse_page_indices=lambda ratio: page_indices,
            sparse_raw_indices=lambda ratio: raw_indices,
        )
        backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(
                core_metadata=core, late_layer_tail=None
            ),
            token_to_kv_pool=types.SimpleNamespace(
                get_index_k_with_scale_buffer=lambda layer: torch.zeros(1, 68)
            ),
            req_to_token=torch.arange(8)[None],
        )
        indexer = types.SimpleNamespace(
            queries=lambda q, freqs: q,
            head_weights=lambda x: x,
            index_topk=2,
            candidate_block_size=8,
            is_candidate_source=False,
            uses_candidates=False,
        )
        layer = types.SimpleNamespace(
            compress_ratio=1, indexer=indexer, freqs_cis=torch.zeros(4), layer_id=0
        )
        batch = types.SimpleNamespace(
            extend_seq_lens_cpu=[4],
            seq_lens_cpu=[8],
            req_pool_indices=torch.tensor([0]),
        )
        prior_scores = []

        def score(q, weights, keys, lens, **kwargs):
            self.assertTrue(all(ref() is None for ref in prior_scores))
            out = torch.arange(8, dtype=torch.float32).expand(q.shape[0], -1).clone()
            out.masked_fill_(torch.arange(8)[None] >= lens[:, None], -torch.inf)
            prior_scores.append(weakref.ref(out))
            return out

        def topk(scores, lens, *, out_offsets, out_indices):
            out_indices.fill_(-1)
            for row, length in enumerate(lens):
                k = min(int(length), out_indices.shape[1])
                out_indices[row, :k] = scores[row, : int(length)].topk(k).indices.int()

        native = types.SimpleNamespace(
            torch=torch,
            _as_int_list=list,
            _TORCH_INDEXER_SCORE_BUDGET_BYTES=64,
            unpack_fp4_index_keys_to_fp8=lambda slots, table, page: torch.zeros(
                len(slots), 128
            ),
            quantize_bf16_index_queries_fp8=lambda q: q,
            fp8_index_logits_prefill=score,
            topk_transform_ragged_v2=topk,
        )
        with mock_native("sglang.srt.layers.attention.deepseek_v4_backend", native):
            prefill.extend_indexer(
                backend,
                layer,
                torch.ones(4, 1),
                torch.ones(4, 1),
                torch.arange(4),
                batch,
            )
        self.assertEqual(len(prior_scores), 2)
        self.assertTrue(all(ref() is None for ref in prior_scores))
        expected = torch.tensor([[0, -1], [0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        self.assertTrue(torch.equal(page_indices, expected))
        self.assertTrue(torch.equal(raw_indices, expected))


if __name__ == "__main__":
    unittest.main()
