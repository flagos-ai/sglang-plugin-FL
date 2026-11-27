# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select upstream's backend-neutral rounding branch before MUSA launch.

SGLang v0.5.18 commit 71de97b264b04dcd514cf904003028aefe9775c8 exposes no
platform argument in per_token_quant_int8. Its private IS_HIP constexpr only
selects generic libdevice.round versus CUDA libdevice.round. MUSA requires
the former. This proxy changes that launch argument, not the kernel or any
rounding arithmetic. Remove it when the public wrapper selects generic
libdevice for MUSA. Upstream upgrades require fresh signature/numerical tests.
"""

import inspect
import logging

logger = logging.getLogger(__name__)
MARKER = "_sglang_fl_musa_int8_rounding"


class _GenericRoundOnMusa:
    _sglang_fl_musa_int8_rounding = True

    def __init__(self, original, selector_position):
        self.original = original
        self.selector_position = selector_position

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def call(*args, **kwargs):
            if args and args[0].device.type == "musa":
                if len(args) > self.selector_position:
                    args = list(args)
                    args[self.selector_position] = True
                else:
                    kwargs = dict(kwargs, IS_HIP=True)
            return launch(*args, **kwargs)

        return call


def install_int8_rounding_patch(upstream=None):
    if upstream is None:
        from sglang.kernels.ops.quantization import int8_kernel

        upstream = int8_kernel
    original = upstream._per_token_quant_int8
    if getattr(original, MARKER, False):
        return False
    parameters = list(inspect.signature(original.fn).parameters)
    if "IS_HIP" not in parameters or "CAL_SUM" not in parameters:
        raise RuntimeError(
            "MUSA INT8 quantization requires IS_HIP and CAL_SUM selectors"
        )
    upstream._per_token_quant_int8 = _GenericRoundOnMusa(
        original, parameters.index("IS_HIP")
    )
    logger.info("MUSA INT8 quantization selects upstream generic rounding")
    return True
