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

"""FlagCX PD-disaggregation KV transfer backend.

Importing this package pulls in ``conn.py``, which transitively loads the
FlagCX shared library. Keep it out of the plugin's import path at load time --
``patch.py`` only imports it lazily, once the user actually selects
``--disaggregation-transfer-backend flagcx``.
"""

from sglang_fl.disaggregation.conn import (
    FlagcxKVBootstrapServer,
    FlagcxKVManager,
    FlagcxKVReceiver,
    FlagcxKVSender,
)

__all__ = [
    "FlagcxKVBootstrapServer",
    "FlagcxKVManager",
    "FlagcxKVReceiver",
    "FlagcxKVSender",
]
