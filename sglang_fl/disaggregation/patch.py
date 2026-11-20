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

import logging
import sys

logger = logging.getLogger(__name__)

# NOTE: if SGLang grows a new caller, add it here.
_ALIAS_MODULES = (
    "sglang.srt.disaggregation.utils",
    "sglang.srt.disaggregation.prefill",
    "sglang.srt.disaggregation.decode",
    "sglang.srt.managers.disagg_service",
)

_BACKEND_NAME = "flagcx"

# Marker attribute set on our wrapper so re-applying is a no-op instead of
# stacking wrappers on top of each other.
_PATCH_MARKER = "_sglang_fl_flagcx_patched"


def _inject_enum_member(cls, name: str, value):
    """Add a new member to an already-created Enum class at runtime.

    ``TransferBackend(server_args.disaggregation_transfer_backend)`` is called
    in scheduler.py / disagg_service.py / multi_tokenizer_mixin.py and raises
    ValueError for unknown values, so the member has to really exist.

    Mutates the private lookup tables that ``EnumMeta`` uses. ``setattr`` on an
    Enum class is blocked ("Cannot reassign members"), hence
    ``type.__setattr__``. Verified to keep identity, ``isinstance`` and
    dict-key semantics intact on CPython 3.8-3.11.
    """
    member = object.__new__(cls)
    member._name_ = name
    member._value_ = value
    cls._value2member_map_[value] = member
    cls._member_map_[name] = member
    cls._member_names_.append(name)
    type.__setattr__(cls, name, member)
    return member


def _add_backend_choice() -> None:
    """Make ``--disaggregation-transfer-backend flagcx`` a valid CLI value."""
    from sglang.srt.server_args import (
        DISAGG_TRANSFER_BACKEND_CHOICES,
        add_disagg_transfer_backend_choices,
    )

    if _BACKEND_NAME not in DISAGG_TRANSFER_BACKEND_CHOICES:
        add_disagg_transfer_backend_choices([_BACKEND_NAME])


def _add_enum_member() -> None:
    """Make ``TransferBackend("flagcx")`` resolve instead of raising."""
    from sglang.srt.disaggregation.utils import TransferBackend

    if _BACKEND_NAME not in TransferBackend._value2member_map_:
        # Member *name* is upper-case (TransferBackend.FLAGCX), matching the
        # other members; the *value* is the CLI string.
        _inject_enum_member(TransferBackend, _BACKEND_NAME.upper(), _BACKEND_NAME)


def _flagcx_class_mapping(class_type):
    """Resolve a KVClassType to the FlagCX implementation.

    Imports ``sglang_fl.disaggregation`` lazily: that module loads the FlagCX
    shared library, which must not be a hard requirement for users who never
    select this backend.
    """
    from sglang.srt.disaggregation.base import KVArgs
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
    from sglang.srt.disaggregation.utils import KVClassType

    if hasattr(MooncakeKVManager, "_registerable_regions"):
        from .native import class_mapping

        return class_mapping(class_type)

    from sglang_fl.disaggregation import (
        FlagcxKVBootstrapServer,
        FlagcxKVManager,
        FlagcxKVReceiver,
        FlagcxKVSender,
    )

    return {
        KVClassType.KVARGS: KVArgs,
        KVClassType.MANAGER: FlagcxKVManager,
        KVClassType.SENDER: FlagcxKVSender,
        KVClassType.RECEIVER: FlagcxKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: FlagcxKVBootstrapServer,
    }.get(class_type)


def _patch_get_kv_class() -> None:
    """Wrap ``get_kv_class`` on the defining module and every alias site."""
    from sglang.srt.disaggregation import utils as _utils

    original = _utils.get_kv_class
    if getattr(original, _PATCH_MARKER, False):
        return  # already wrapped

    def get_kv_class(transfer_backend, class_type):
        if getattr(transfer_backend, "value", None) == _BACKEND_NAME:
            return _flagcx_class_mapping(class_type)
        return original(transfer_backend, class_type)

    get_kv_class.__doc__ = original.__doc__
    get_kv_class.__wrapped__ = original
    setattr(get_kv_class, _PATCH_MARKER, True)

    patched = []
    for module_name in _ALIAS_MODULES:
        # Only touch modules that are already imported -- importing them here
        # just to patch would perturb SGLang's import order. The defining
        # module is patched unconditionally below, so any module imported
        # *later* naturally binds the wrapper.
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_kv_class"):
            module.get_kv_class = get_kv_class
            patched.append(module_name)

    logger.debug("flagcx get_kv_class patched on: %s", patched)


_http_worker_original = None


def _flagcx_http_worker_started(config, target, sockets, stdin_fileno=None):
    """Spawn entry: register the backend before a tokenizer worker starts."""
    original = _http_worker_original
    if original is None:
        # A spawned interpreter imports this module afresh, so uvicorn still
        # owns its original entry point here.
        from uvicorn._subprocess import subprocess_started

        original = subprocess_started
    if getattr(config, "app", None) == "sglang.srt.entrypoints.http_server:app":
        apply_disaggregation_patch()
    return original(
        config=config, target=target, sockets=sockets, stdin_fileno=stdin_fileno
    )


def _patch_http_worker_bootstrap():
    global _http_worker_original
    from uvicorn import _subprocess

    if _subprocess.subprocess_started is _flagcx_http_worker_started:
        return
    _http_worker_original = _subprocess.subprocess_started
    _subprocess.subprocess_started = _flagcx_http_worker_started


def apply_disaggregation_patch() -> None:
    """Register the FlagCX transfer backend with SGLang. Idempotent."""
    _add_backend_choice()
    _add_enum_member()
    _patch_get_kv_class()
    _patch_http_worker_bootstrap()
    logger.info(
        "FlagCX PD disaggregation backend registered "
        "(--disaggregation-transfer-backend flagcx)"
    )
