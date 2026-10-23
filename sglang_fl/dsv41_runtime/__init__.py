# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 runtime hooks installed by SGLang's plugin loader."""


def load_plugin():
    from .patch import apply_dsv41_patches

    apply_dsv41_patches()
