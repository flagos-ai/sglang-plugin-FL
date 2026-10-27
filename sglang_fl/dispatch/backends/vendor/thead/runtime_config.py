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

"""Server-arg defaults validated with SGLang 0.5.18 on T-Head PPU.

Pipeline parallelism: SGLang 0.5.18 sizes the mamba state pool per PP stage
(``pp_layer_scale``), which gives each stage about twice the mamba pool 0.5.11
did. With SGLang's automatic ``mem_fraction_static`` (or 0.85) the PP1 stage
of Qwen3.6-35B-A3B runs out of activation memory on 4096-token prompts at
concurrency 64. 0.75 passed the full 5-round TP4 x PP2 stress run.

The value comes from ``pp_mem_fraction_static`` in dispatch/config/thead.yaml
(or ``SGLANG_FL_CONFIG``), overridable with ``SGLANG_FL_THEAD_PP_MEM_FRACTION``.
It applies only when PP is enabled and the user did not set
``--mem-fraction-static``. Set the env var to ``auto`` to keep SGLang's default.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENV = "SGLANG_FL_THEAD_PP_MEM_FRACTION"
_CONFIG_KEY = "pp_mem_fraction_static"


def _pp_mem_fraction(config: dict):
    raw = os.environ.get(_ENV, "").strip()
    if not raw:
        raw = config.get(_CONFIG_KEY)
    if raw is None or str(raw).strip().lower() in ("", "auto", "none"):
        return None
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{_ENV}/{_CONFIG_KEY} must be in (0, 1), got {raw!r}")
    return value


def apply_thead_runtime_defaults(server_args, config: dict = None) -> None:
    if config is None:
        from sglang_fl.dispatch.config import get_effective_config

        config = get_effective_config()

    if getattr(server_args, "pp_size", 1) <= 1:
        return
    # apply_server_args_defaults runs before SGLang resolves its automatic
    # value, so None here means the user did not pass --mem-fraction-static.
    if getattr(server_args, "mem_fraction_static", None) is not None:
        return
    value = _pp_mem_fraction(config)
    if value is None:
        return
    server_args.mem_fraction_static = value
    logger.info(
        "thead: PP=%d, mem_fraction_static defaulted to %.2f "
        "(SGLang 0.5.18 per-stage mamba pool sizing)",
        server_args.pp_size,
        value,
    )
