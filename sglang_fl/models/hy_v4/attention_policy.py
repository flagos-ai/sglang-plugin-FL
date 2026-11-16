# SPDX-License-Identifier: Apache-2.0
"""Per-runner role and forward-mode qualification for sparse MLA dispatch."""


def freeze_role_policy(model_runner):
    """Only an actual bool identifies the target or draft runner."""
    flag = getattr(model_runner, "is_draft_worker", None)
    return "draft" if flag is True else "target" if flag is False else "unknown"


def eligible_forward(backend):
    return getattr(backend, "_hy4_mla_role", "unknown") in (
        "target",
        "draft",
    ) and getattr(getattr(backend, "_hy4_forward_mode", None), "name", None) in (
        "TARGET_VERIFY",
        "DECODE",
    )
