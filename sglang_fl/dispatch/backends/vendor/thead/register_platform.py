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

"""Apply the T-Head PPU patches from ``PlatformFL.init_backend``.

The patches normally run from ``load_plugin()`` (``_apply_vendor_patches``).
Some SGLang 0.5.18 entry points never call ``load_plugins()`` --
``sglang.bench_one_batch`` builds a ModelRunner directly -- but every worker
still calls ``PlatformFL.init_backend``, which imports this module. Without it
those paths compile JIT kernels for plain ``sm_80`` (failing the arch assert on
PPU) and miss the sgl_kernel compat symbols.

``patch.py`` applies its patches once per process, so importing it here is a
no-op when ``load_plugin()`` already ran.
"""

from . import patch  # noqa: F401
