# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prune invalid upstream sparse-topk launch candidates, not numerical code.

SGLang 0.5.18 (71de97b) includes BLOCK_SIZE_K <= BLOCK_SIZE_T candidates
although its kernel statically forbids them. The final selected launch can
therefore fail compilation. Remove this adapter when upstream prunes invalid
candidates before compiling them. No executed kernel is retried or replaced.
"""

from copy import copy

MARKER = "_sglang_fl_musa_sparse_topk"


def install_sparse_topk_patch(upstream):
    original = upstream._topk_index_kernel
    if getattr(original, MARKER, False):
        return False
    tuner = getattr(original, "fn", None)
    if (
        tuner is None
        or not hasattr(tuner, "early_config_prune")
        or not hasattr(tuner, "configs")
    ):
        raise RuntimeError("MUSA sparse topk requires upstream autotune pruning API")
    prior = tuner.early_config_prune

    def prune(configs, named_args, **kwargs):
        if prior is not None:
            configs = prior(configs, named_args, **kwargs)
        args = dict(named_args, **kwargs)
        score = args.get("s_ptr")
        if score is None or score.device.type != "musa":
            return configs
        topk = int(args["topk"])
        tile = 1 << (max(1, topk) - 1).bit_length()
        valid = [c for c in configs if c.kwargs["BLOCK_SIZE_K"] > tile]
        if not valid:
            raise NotImplementedError("No legal upstream MUSA sparse topk tile")
        return valid

    replacement = copy(original)
    replacement.fn = copy(tuner)
    # Keep installation independent of an original tuner's previous choices.
    replacement.fn.cache = {}
    replacement.fn.early_config_prune = prune
    setattr(replacement, MARKER, True)
    upstream._topk_index_kernel = replacement
    return True
