# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend BF16 decode-convolution contracts; skip without NPU dependencies.

Run with CANN initialized and the SGLang/FlagGems runtime on PYTHONPATH:
  python -m unittest discover -s tests/unit_tests/platform \
      -p test_ascend_causal_conv_decode_fused.py -v
"""

import importlib.util
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    import torch_npu
    import flag_gems

    _AVAILABLE = torch.npu.is_available()
except (ImportError, AttributeError, RuntimeError):
    _AVAILABLE = False


@unittest.skipUnless(_AVAILABLE, "Ascend NPU, torch_npu and FlagGems are required")
class TestAscendCausalConvDecodeFused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        path = (
            root
            / "sglang_fl/dispatch/backends/vendor/ascend/patches/qwen36_causal_conv_decode_fused.py"
        )
        spec = importlib.util.spec_from_file_location("conv_decode_under_test", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

        cls.native_update = staticmethod(causal_conv1d_update_npu)
        torch.npu.set_device(0)
        cls.gems_context = flag_gems.use_gems(
            exclude=["sum_dim", "mul", "index", "index_put_", "_index_put_impl_"]
        )
        cls.gems_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.gems_context.__exit__(None, None, None)

    @staticmethod
    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cpu").to("npu")

    @staticmethod
    def metadata(batch):
        return SimpleNamespace(
            batch_size=batch,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

    def control(self, x, states, weight, indices):
        snapshot = states.transpose(1, 2).clone()
        result = self.native_update(
            x, snapshot, weight, activation="silu", conv_state_indices=indices
        )
        states[:] = snapshot.transpose(1, 2)
        return result

    def candidate(self, x, states, weight, indices):
        transposed = self.mod._transposed_weight(SimpleNamespace(conv_weights=weight))
        return self.mod.fused_decode_conv(
            x,
            states,
            transposed,
            indices,
        )

    def candidate_coalesced(self, x, states, weight, indices):
        return self.mod.fused_decode_conv(
            x, states, weight, indices, coalesced_weight=True
        )

    def assertBitsEqual(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.detach().cpu().contiguous().view(torch.int16),
                expected.detach().cpu().contiguous().view(torch.int16),
            )
        )

    def test_guards_and_real_blocked_format(self):
        x, weight = self.random((8, 5120)), self.random((5120, 4))
        pool = torch.zeros((2, 65, 3, 5120), dtype=x.dtype, device="npu")
        states = pool[1]
        self.assertEqual(torch_npu.get_npu_format(states), 0)
        indices = torch.arange(8, dtype=torch.int32, device="npu")
        layer = SimpleNamespace(conv_weights=weight, bias=None, activation="silu")
        batch = self.metadata(8)
        with torch.inference_mode():
            self.assertTrue(self.mod._supported(x, states, layer, batch, indices))
            batch.batch_size = 7
            self.assertFalse(self.mod._supported(x, states, layer, batch, indices))
            batch.batch_size = 8
            self.assertFalse(
                self.mod._supported(x[:, ::2], states, layer, batch, indices)
            )
            layer.bias = torch.zeros(5120, dtype=x.dtype, device="npu")
            self.assertFalse(self.mod._supported(x, states, layer, batch, indices))
            layer.bias = None
            batch.spec_algorithm = SimpleNamespace(is_none=lambda: False)
            self.assertFalse(self.mod._supported(x, states, layer, batch, indices))
            batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
            previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
            try:
                torch.npu.config.allow_internal_format = True
                layer.conv_weights = torch_npu.npu_format_cast(weight, 29)
                self.assertEqual(torch_npu.get_npu_format(layer.conv_weights), 29)
                self.assertFalse(self.mod._supported(x, states, layer, batch, indices))
            finally:
                torch.npu.config.allow_internal_format = previous
            layer.conv_weights = self.random((5120, 4))  # InferenceTensor: no _version.
            self.assertTrue(self.mod._supported(x, states, layer, batch, indices))
        self.assertFalse(self.mod._supported(x, states, layer, batch, indices))

    def test_installer_contract_and_custom_dispatch(self):
        # Test the installer ABI without importing SGLang's full attention stack
        # (and registering unrelated FLA autotuners in this unit test).
        class Backend:
            def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
                conv_states = self.req_to_token_pool.mamba2_layer_cache(
                    layer.layer_id
                ).conv[0]
                conv_states_tmp = conv_states.transpose(1, 2).clone()
                conv_states[:] = conv_states_tmp.transpose(1, 2)

        torch_causal_conv1d_update_npu = mock.Mock()

        def causal_conv1d_update(conv_state, conv_state_indices):
            conv_state_update = conv_state[conv_state_indices]
            return torch_causal_conv1d_update_npu(conv_state_update)

        class Replay:
            def _replay_metadata(self, bs, num_padding):
                req_pool_indices, mamba_indices = self.indices
                req_pool_indices[bs - num_padding :] = 0
                mamba_indices[bs - num_padding :] = 0

        backend = SimpleNamespace(
            AscendGDNAttnBackend=Backend,
            causal_conv1d_update=causal_conv1d_update,
            AscendMambaAttnBackendBase=Replay,
        )
        original = backend.AscendGDNAttnBackend.forward_decode
        env = {
            "SGLANG_FL_FUSED_CONV_DECODE": "1",
            "USE_FLAGGEMS": "1",
            "SGLANG_FL_FLAGOS_WHITELIST": "",
            "SGLANG_FL_FLAGOS_BLACKLIST": "sum_dim,mul",
        }
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.hardware_backend.npu.attention": SimpleNamespace(
                        ascend_gdn_backend=backend
                    )
                },
            ),
            mock.patch.dict(os.environ, env),
            mock.patch.object(backend.AscendGDNAttnBackend, "forward_decode", original),
        ):
            with mock.patch.object(
                self.mod.inspect, "signature", side_effect=ValueError("unknown")
            ):
                self.assertFalse(self.mod.patch_qwen36_causal_conv_decode_fused())
            with mock.patch.object(
                self.mod.inspect, "getsource", return_value="upstream changed"
            ):
                self.assertFalse(self.mod.patch_qwen36_causal_conv_decode_fused())
            for extra in (
                {"SGLANG_FL_FUSED_CONV_DECODE": "0"},
                {"USE_FLAGGEMS": "0"},
                {"SGLANG_FL_FLAGOS_WHITELIST": "silu"},
                {"SGLANG_FL_FLAGOS_BLACKLIST": "mul"},
                {"SGLANG_FL_FLAGOS_BLACKLIST": "sum_dim,silu"},
            ):
                with mock.patch.dict(os.environ, extra):
                    self.assertFalse(self.mod.patch_qwen36_causal_conv_decode_fused())
            self.assertTrue(self.mod.patch_qwen36_causal_conv_decode_fused())
            self.assertFalse(self.mod.patch_qwen36_causal_conv_decode_fused())

    def test_graph_real_pool_padding_and_inplace_weight_updates(self):
        # Ascend _replay_metadata reserves duplicate slot 0 for padding.
        # Active slots are unique 1..64; dummy slot 0 has no user state.
        for channels in (4096, 5120):
            for batch, padding in ((1, 0), (24, 2), (64, 0), (64, 23)):
                with self.subTest(channels=channels, batch=batch, padding=padding):
                    valid = batch - padding
                    pools = [
                        torch.zeros(
                            (2, 65, 3, channels), dtype=torch.bfloat16, device="npu"
                        )
                        for _ in range(3)
                    ]
                    states = [pool[1] for pool in pools]
                    initial = self.random((65, 3, channels))
                    for state in states:
                        state.copy_(initial)
                    self.assertEqual(torch_npu.get_npu_format(states[0]), 0)
                    self.assertGreater(states[0].storage_offset(), 0)
                    x, weight = (
                        self.random((batch, channels)),
                        self.random((channels, 4)),
                    )
                    indices = torch.cat(
                        (
                            torch.arange(1, valid + 1),
                            torch.zeros(padding, dtype=torch.int64),
                        )
                    ).to("npu")
                    graphs = [torch.npu.NPUGraph() for _ in range(3)]
                    outputs = []
                    for graph, state, fn in zip(
                        graphs,
                        states,
                        (self.control, self.candidate, self.candidate_coalesced),
                    ):
                        with torch.npu.graph(graph):
                            outputs.append(fn(x, state, weight, indices))
                    for _ in range(8):
                        x.copy_(self.random((batch, channels)))
                        weight.add_(
                            0.03125
                        )  # No graph recapture: transpose must replay.
                        new_indices = torch.randperm(64, device="cpu")[:valid] + 1
                        indices.copy_(
                            torch.cat(
                                (new_indices, torch.zeros(padding, dtype=torch.int64))
                            ).to("npu")
                        )
                        for graph in graphs:
                            graph.replay()
                        torch.npu.synchronize()
                        for candidate in (1, 2):
                            self.assertBitsEqual(
                                outputs[0][:valid], outputs[candidate][:valid]
                            )
                            self.assertBitsEqual(states[0][1:], states[candidate][1:])
                        for pool in pools:
                            self.assertFalse(bool(torch.any(pool[0].cpu())))

    def test_fixed_cancellation_patterns(self):
        patterns = set()
        for values in (
            (2.0**30, 1.0, -(2.0**30), 1.0),
            (2.0**25, 1.0, -(2.0**25), -1.0),
            (2.0**16, 2.0**-8, -(2.0**16), 2.0**-8),
            (2.0**120, 2.0**96, -(2.0**120), 2.0**96),
        ):
            patterns.update(itertools.permutations(values))
        base = torch.tensor(sorted(patterns), dtype=torch.bfloat16)
        for channels in (4096, 5120):
            data = base.repeat((channels + len(base) - 1) // len(base), 1)[
                :channels
            ].to("npu")
            states = torch.zeros((65, 3, channels), dtype=torch.bfloat16, device="npu")
            states[1:9] = data[:, :3].T.unsqueeze(0)
            other = states.clone()
            x = data[:, 3].repeat(8, 1)
            weight = torch.ones((channels, 4), dtype=x.dtype, device="npu")
            indices = torch.arange(1, 9, dtype=torch.int64, device="npu")
            self.assertBitsEqual(
                self.control(x, states, weight, indices),
                self.candidate(x, other, weight, indices),
            )
            self.assertBitsEqual(states, other)

    def test_special_value_semantics(self):
        values = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0, 2.0**-30],
            dtype=torch.bfloat16,
        )
        for channels in (4096, 5120):
            x = values.repeat(channels // 8).repeat(8, 1).to("npu")
            states = torch.zeros((65, 3, channels), dtype=x.dtype, device="npu")
            other = states.clone()
            weight = torch.ones((channels, 4), dtype=x.dtype, device="npu")
            indices = torch.arange(1, 9, dtype=torch.int64, device="npu")
            left = self.control(x, states, weight, indices).cpu()
            right = self.candidate(x, other, weight, indices).cpu()
            self.assertTrue(torch.equal(torch.isnan(left), torch.isnan(right)))
            mask = ~torch.isnan(left)
            self.assertBitsEqual(left[mask], right[mask])
            # Native canonicalizes NaN payload; payload bits are not the contract.
            left, right = states.cpu(), other.cpu()
            self.assertTrue(torch.equal(torch.isnan(left), torch.isnan(right)))
            mask = ~torch.isnan(left)
            self.assertBitsEqual(left[mask], right[mask])


if __name__ == "__main__":
    unittest.main()
