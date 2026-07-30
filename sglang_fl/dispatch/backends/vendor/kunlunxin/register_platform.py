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

"""OOT attention-backend registration for the kunlunxin vendor.

Auto-imported by ``PlatformFL.init_backend()`` when FlagGems reports
vendor_name == "kunlunxin". Importing this module registers the
``kunlunxin`` attention backend into sglang's ATTENTION_BACKENDS dict.

It is also the default backend on this vendor via ``_ATTN_BACKEND_MAP`` in
``sglang_fl/platform.py``, so ``--attention-backend`` need not be passed.
"""

from sglang.srt.layers.attention.attention_registry import register_attention_backend


@register_attention_backend("kunlunxin")
def _create_kunlunxin_backend(runner):
    from sglang_fl.dispatch.backends.vendor.kunlunxin.impl.attention_backend import (
        KunlunxinBackend,
    )

    return KunlunxinBackend(runner)
