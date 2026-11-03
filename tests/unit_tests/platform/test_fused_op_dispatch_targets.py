# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import ast
from pathlib import Path


class TestFusedOpDispatchTargets(unittest.TestCase):
    def test_current_and_legacy_resolvers_are_hooked(self):
        """Prevent OOT fused ops from silently using the native CPU fallback."""
        source = Path(__file__).parents[3] / "sglang_fl" / "__init__.py"
        module = ast.parse(source.read_text())
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FUSED_OP_DISPATCH_TARGETS"
                for target in node.targets
            )
        )
        targets = ast.literal_eval(assignment.value)

        self.assertEqual(
            targets,
            (
                "sglang.srt.layers.utils.multi_platform.MultiPlatformOp._resolve_forward_method",
                "sglang.srt.layers.utils.multi_platform.MultiPlatformOp.dispatch_forward",
            ),
        )
