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

# sglang_fl.patches.apply_common_patches: entrypoint for vendor-independent patches.

import sglang_fl.patches as common_patches


def test_apply_common_patches_runs_each_patch_once(monkeypatch):
    calls = []
    monkeypatch.setattr(common_patches, "_patches_applied", False)
    monkeypatch.setattr(
        common_patches, "patch_mamba_pool_copy_from", lambda: calls.append("mamba")
    )

    common_patches.apply_common_patches()
    common_patches.apply_common_patches()

    assert calls == ["mamba"]
