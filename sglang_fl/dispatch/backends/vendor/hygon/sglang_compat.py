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

"""Plugin-owned HCU replacement for the SGLang 0.5.18 source patch.

The runtime image currently applies ``docker/hygon/sglang-v0.5.18-hcu.patch``
to the installed SGLang wheel (reference commit a5d492a). This module
reproduces the behavioral contract of that patch from plugin activation,
without touching any installed SGLang file:

* SGLang resolves the ``sglang.srt.platforms`` entry point during
  ``sglang.srt.server_args`` import (``current_platform`` is a lazy module
  attribute), which invokes ``sglang_fl.activate_platform()`` and therefore
  this package's ``early.py`` BEFORE any ``sglang.srt.layers`` module is
  imported in the natural engine flow.

* A targeted finder wraps the REAL loading of exactly the patched SGLang
  modules and adjusts their module-level bindings as the module finishes
  executing - before any consumer import can take unadjusted references
  (the wrapping loader runs the adjuster inside ``exec_module``).

* The patch deletes the ``if _is_hip:`` AITER blocks of the three quark
  modules outright: their ``from aiter...`` imports, the four
  ``sglang::aiter_*`` custom-op registrations, and the ``gcnArchName``
  probe. A plugin cannot edit files, so those three modules instead
  execute inside a scoped "aiter-absent window" that reproduces exactly
  that deletion and nothing more:

  - ``aiter`` imports issued from the TARGET module's own body are served
    an inert placeholder (raise-on-call) so module execution completes.
    Imports issued by any other module - including modules imported by
    the target's own prelude, e.g. ``quark/utils.py``'s guarded import,
    which runs before the target's own AITER block and therefore before
    any placeholder exists - are NOT served: they see aiter absent and
    take their pristine fallbacks, exactly like the patched wheel on an
    aiter-less runtime. The placeholder is visible in ``sys.modules``
    only for the remainder of the target's body: the three quark modules
    define only classes after their AITER blocks, so nothing else can
    observe it, and the window removes it before returning.
  - ``direct_register_custom_op`` calls for ``aiter_*`` op names are
    suppressed (recorded, never registered), so no ``sglang::aiter_*``
    operators enter the process-global torch ops registry. Every other
    registration delegates to the real function unchanged, and on window
    exit the target module's own ``from ... import`` binding is restored
    to the real function (a later registration issued from inside the
    module behaves exactly like the patched wheel).
  - the AITER block's device-arch probe (``ON_GFX950``) is shielded for
    ``quark_int4fp8_moe`` only: ``torch.cuda.get_device_properties``
    calls issued from that module's own body never reach the driver,
    matching the patched wheel, while every other caller - including
    cold prelude imports such as fp8.py's module-level
    ``is_gfx95_supported()`` - keeps the real (window-entry) function.
  - after execution the placeholder modules are removed from
    ``sys.modules`` and the block's module-level bindings are excised,
    leaving the same names absent that the patch leaves absent.

  The window exists ONLY while one of the three quark module bodies is
  executing. Everywhere else in the process ``import aiter`` fails
  naturally (the HCU runtime does not ship aiter), which is precisely
  the patched wheel's world: guarded aiter probes elsewhere in SGLang
  keep reporting aiter absent instead of finding a fake.

  Thread isolation: the placeholder modules live in ``sys.modules`` only
  while the target module body executes, and Python's import state is
  shared across threads, so a concurrent thread importing ``aiter`` in
  that interval could hit the module cache without consulting the
  finder (the frame check only applies at ``find_spec`` time). The
  deployment model makes this unreachable: SGLang resolves the platform
  entry point and imports its layer stack on one thread during engine
  startup, and the HCU runtime ships no aiter for any other thread to
  legitimately import (under the patched wheel such an import would
  simply fail). If the assumption is ever violated, the observer
  receives the raise-on-call placeholder objects and fails loudly at
  use instead of executing AMD kernels.

Adjustments per module (patch parity):

1. ``quark.schemes.quark_w4a4_mxfp4``  - AITER block excised (four op
   registrations suppressed, block names absent), ``QuarkW4A4MXFP4.__init__``
   raises ``NotImplementedError`` (AMD AITER dependency, unsupported on HCU).
2. ``quark.schemes.quark_w4a4_mxfp4_moe`` - same excision plus
   ``_use_aiter`` forced False and ``dynamic_mxfp4_quant`` set to None
   (the patch's else-branch value); ``QuarkW4A4MXFp4MoE.__init__`` raises
   ``NotImplementedError``.
3. ``quark_int4fp8_moe`` - AITER block excised (``shuffle_weight`` and the
   ``ON_GFX950`` ``gcnArchName`` probe leave no residue);
   ``QuarkInt4Fp8Config.__init__`` raises ``NotImplementedError``.
4. ``attention/triton_backend`` - if the CUDA-gated
   ``is_arch_support_pdl`` import fired, it is rebound to ``False`` so
   ``use_pdl`` resolves False (the patch's ``and not _is_hcu`` outcome).
5. ``activation`` - if the CUDA-gated ``sgl_kernel`` imports fired, the
   ``_sgl_*`` names are re-aliased to SGLang's own Triton/JIT activation
   kernels (identical ``(input, out)`` signatures; the patch's else
   branch). On HCU today ``is_cuda()`` is False, so both the patch and
   this rebind are defensive parity.
6. ``layernorm`` - if the CUDA-gated ``sgl_kernel`` layernorm imports
   fired, the four names are set to ``None`` so accidental calls fail
   loudly (``TypeError``), exactly like the patched wheel.
7. ``sampler`` - needs no adjustment: the flashinfer/sgl_kernel sampling
   imports are gated by ``is_cuda()``, which is False on HCU, and an
   explicitly selected FlashInfer path then fails on the missing names
   (clear ``NameError``) instead of faking success - the patched wheel's
   behavior.

Aiter-present runtimes: the window is aiter-agnostic. When the runtime
ships aiter (the hygon CI image does), the blocks' imports resolve to
the REAL package through the normal finders - the appended placeholder
finder is never consulted - and the aiter_* registration suppression,
the arch-probe shield and the post-execution excision apply unchanged.
The patch likewise skips the AITER blocks regardless of aiter's
presence; the remaining difference - aiter entering ``sys.modules``
during the window rather than at the first guarded probe elsewhere -
is state such an image reaches either way. When aiter is absent (the
operations runtime), the inert placeholder serves exactly the target's
own imports as described above.

Everything here activates only on HCU (``torch.__hcu_version__``), is
idempotent, installs nothing on other platforms, and never modifies,
rewrites, or re-imports-from-source any SGLang file. Opt out entirely
with ``SGLANG_FL_HYGON_SGLANG_COMPAT=0``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

COMPAT_OPT_OUT = "SGLANG_FL_HYGON_SGLANG_COMPAT"

_GUARD_ATTR = "__sglang_fl_hcu_guarded__"
_ADJUST_MARK = "__sglang_fl_hcu_adjusted__"
_PLACEHOLDER_MARK = "__sglang_fl_hcu_window_aiter__"

_active = False
_adjusted: dict[str, bool] = {}
_window_reports: dict[str, dict] = {}

# The modules whose patch hunks delete an `if _is_hip:` AITER block; only
# these execute inside the aiter-absent window. Every other patched module
# (and everything else in the process) must see the real import state.
AITER_WINDOW_MODULES = frozenset(
    {
        "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4",
        "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe",
        "sglang.srt.layers.quantization.quark_int4fp8_moe",
    }
)


def is_hcu() -> bool:
    """HCU torch builds carry ``__hcu_version__``; HIP alone does not."""
    return hasattr(torch, "__hcu_version__")


def is_active() -> bool:
    return _active


def adjusted_modules() -> dict[str, bool]:
    """Modules whose import this process routed through the adjusters."""
    return dict(_adjusted)


def window_reports() -> dict[str, dict]:
    """Per-module evidence from the aiter-absent windows (read-only)."""
    return {name: dict(report) for name, report in _window_reports.items()}


# ── inert placeholder objects (window-scoped) ───────────────────────────────


class _Unavailable:
    """Stand-in for an object of the missing package; fails when used."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._name} is unavailable: aiter is not shipped on HCU "
            "(sglang_fl HCU import window)"
        )

    def __mro_entries__(self, bases):
        # Allow `class X(pkg.Foo)` at import time.
        return (object,)

    def __repr__(self) -> str:
        return f"<unavailable {self._name}>"


class _PlaceholderModule(types.ModuleType):
    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self.__name__}.{attr}")


def _innermost_module_body_name(start: int) -> str | None:
    """Fullname of the innermost module-body frame ``start`` levels up.

    Walks outwards to the closest frame that is a module body
    (``co_name == "<module>"``) executing a loaded module's namespace;
    intermediate function frames (import machinery, helpers such as
    ``is_gfx95_supported``) are skipped. ``None`` when no such frame
    exists on the stack.
    """
    frame = sys._getframe(start)
    while frame is not None:
        if frame.f_code.co_name == "<module>":
            name = frame.f_globals.get("__name__")
            module = sys.modules.get(name) if isinstance(name, str) else None
            if (
                module is not None
                and isinstance(module, types.ModuleType)
                and not name.startswith("importlib")
                and frame.f_globals is vars(module)
            ):
                return name
        frame = frame.f_back
    return None


def _import_requesting_module() -> str | None:
    """Fullname of the module whose body issued the current import."""
    return _innermost_module_body_name(2)


# ── the aiter-absent window ─────────────────────────────────────────────────


class _WindowState:
    def __init__(self, target: str):
        self.target = target
        self.served: list[str] = []
        self.suppressed_ops: list[str] = []
        self.restored_module_binding = False


# The AITER block in quark_int4fp8_moe probes the device arch
# (``ON_GFX950 = ... get_device_properties("cuda").gcnArchName``), which
# the patch skips entirely. For exactly that module the window also
# shields ``torch.cuda.get_device_properties``: a call issued from the
# target's OWN module body receives a benign stand-in (the probe's only
# consumer, ``ON_GFX950``, is excised after execution) and never reaches
# the driver - the patched wheel issues no such query. Every other caller
# is delegated to the function installed when the window opened: cold
# prelude imports inside the window (e.g. fp8.py's module-level
# ``is_gfx95_supported()``) keep querying for real, exactly like the
# patched wheel. Sites 1/2 install no shield: their AITER blocks contain
# no device query, and site-2's module-level ``is_gfx95_supported()``
# sits OUTSIDE the patched block and must keep running for real.
DEVICE_QUERY_SHIELD_MODULES = frozenset(
    {"sglang.srt.layers.quantization.quark_int4fp8_moe"}
)


_window_state: _WindowState | None = None


class _WindowAiterFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve ``aiter`` imports issued from the window target's own body.

    Appended to ``sys.meta_path`` only while the target module executes.
    Any other importer is declined and sees aiter absent, exactly like
    the patched wheel's aiter-less runtime.
    """

    def find_spec(self, fullname, path, target=None):
        state = _window_state
        if state is None or (fullname != "aiter" and not fullname.startswith("aiter.")):
            return None
        if _import_requesting_module() != state.target:
            return None
        state.served.append(fullname)
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        setattr(module, _PLACEHOLDER_MARK, True)
        return module

    def exec_module(self, module):
        pass


@contextlib.contextmanager
def _aiter_absent_window(target: str):
    """Reproduce the patch's deletion of ``target``'s AITER import block.

    Aiter-agnostic: with the real package installed (the hygon CI image)
    the blocks' imports resolve to it through the normal finders and the
    appended placeholder finder stays idle; the suppression, the shield
    and the excision below apply identically in both modes.
    """
    global _window_state
    state = _WindowState(target)
    common = sys.modules.get("sglang.srt.utils.common")
    if common is None and target.startswith("sglang."):
        # The swap must be in place before the target's own prelude binds
        # `direct_register_custom_op` via `from ... import`.
        try:
            import sglang.srt.utils.common as common  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - synthetic-target tests have no sglang
            common = None
    real_register = getattr(common, "direct_register_custom_op", None)

    def filtered_register(
        op_name, op_func, mutates_args, fake_impl=None, target_lib=None
    ):
        if op_name.startswith("aiter_"):
            state.suppressed_ops.append(op_name)
            return None
        return real_register(op_name, op_func, mutates_args, fake_impl, target_lib)

    finder = _WindowAiterFinder()
    real_properties = (
        torch.cuda.get_device_properties
        if target in DEVICE_QUERY_SHIELD_MODULES
        else None
    )

    def shielded_get_device_properties(*args, **kwargs):
        if _innermost_module_body_name(1) == state.target:
            return SimpleNamespace(gcnArchName="")
        return real_properties(*args, **kwargs)

    try:
        if common is not None and real_register is not None:
            common.direct_register_custom_op = filtered_register
        if real_properties is not None:
            torch.cuda.get_device_properties = shielded_get_device_properties
        sys.meta_path.append(finder)
        _window_state = state
        yield state
    finally:
        _window_state = None
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
        if real_properties is not None:
            torch.cuda.get_device_properties = real_properties
        if common is not None and real_register is not None:
            common.direct_register_custom_op = real_register
            # The target module bound the filtered wrapper into its own
            # namespace via `from ... import direct_register_custom_op`
            # during execution; restore the real function there too, so a
            # later call from inside the module registers like the patched
            # wheel instead of staying suppressed forever.
            module = sys.modules.get(target)
            if module is not None and (
                getattr(module, "direct_register_custom_op", None) is filtered_register
            ):
                module.direct_register_custom_op = real_register
                state.restored_module_binding = True
        popped = sorted(
            name
            for name, module in sys.modules.items()
            if (name == "aiter" or name.startswith("aiter."))
            and getattr(module, _PLACEHOLDER_MARK, False)
        )
        for name in popped:
            del sys.modules[name]
        _window_reports[target] = {
            "served": list(state.served),
            "suppressed_ops": list(state.suppressed_ops),
            "popped": popped,
            "registration_swap": common is not None and real_register is not None,
            "restored_module_binding": state.restored_module_binding,
            "device_query_shielded": real_properties is not None,
        }
        if state.suppressed_ops:
            logger.debug(
                "HCU compat suppressed sglang::%s registrations in %s",
                ", ".join(state.suppressed_ops),
                target,
            )


# ── per-module adjustment (patch parity) ────────────────────────────────────


def _guard_init_raises(cls: type, message: str) -> None:
    """Replace ``cls.__init__`` with an HCU-only NotImplementedError guard."""
    if getattr(cls, _GUARD_ATTR, False):
        return

    @functools.wraps(cls.__init__)
    def guarded_init(self, *args, **kwargs):
        raise NotImplementedError(message)

    guarded_init.__sglang_fl_hcu_guard__ = message
    cls.__init__ = guarded_init
    setattr(cls, _GUARD_ATTR, True)


def _excise(module, *names: str) -> None:
    """Remove bindings the patch never creates (skipped AITER blocks)."""
    for name in names:
        if hasattr(module, name):
            delattr(module, name)


# Names created inside quark_w4a4_mxfp4's `if _is_hip:` AITER block
# (imports, op wrappers + fakes, public entry points). Under the patch
# none of them exist on HCU.
_QUARK_W4A4_MXFp4_BLOCK_NAMES = (
    "_fused_gemm_afp4wfp4_split_cat_orig",
    "_gemm_afp4wfp4_orig",
    "_gemm_afp4wfp4_pre_quant_orig",
    "_dynamic_mxfp4_quant_orig",
    "_aiter_gemm_afp4wfp4",
    "_aiter_gemm_afp4wfp4_fake",
    "_aiter_gemm_afp4wfp4_pre_quant",
    "_aiter_gemm_afp4wfp4_pre_quant_fake",
    "_aiter_dynamic_mxfp4_quant",
    "_aiter_dynamic_mxfp4_quant_fake",
    "_aiter_fused_gemm_split_cat",
    "_aiter_fused_gemm_split_cat_fake",
    "gemm_afp4wfp4",
    "gemm_afp4wfp4_pre_quant",
    "dynamic_mxfp4_quant",
    "fused_gemm_afp4wfp4_split_cat",
)


def _adjust_quark_w4a4_mxfp4(module) -> None:
    module._is_hcu = is_hcu()
    _excise(module, *_QUARK_W4A4_MXFp4_BLOCK_NAMES)
    _guard_init_raises(
        module.QuarkW4A4MXFP4,
        "Quark W4A4 MXFP4 requires AMD AITER kernels and is not supported on HCU.",
    )


def _adjust_quark_w4a4_mxfp4_moe(module) -> None:
    module._is_hcu = is_hcu()
    _excise(module, "shuffle_weight", "e8m0_shuffle")
    module._use_aiter = False
    module.dynamic_mxfp4_quant = None
    _guard_init_raises(
        module.QuarkW4A4MXFp4MoE,
        "Quark W4A4 MXFP4 MoE requires AMD AITER kernels and is not supported on HCU.",
    )


def _adjust_quark_int4fp8_moe(module) -> None:
    module._is_hcu = is_hcu()
    _excise(module, "shuffle_weight", "ON_GFX950")
    _guard_init_raises(
        module.QuarkInt4Fp8Config,
        "Quark Int4FP8 MoE requires AMD AITER kernels and is not supported on HCU.",
    )


def _adjust_triton_backend(module) -> None:
    if hasattr(module, "is_arch_support_pdl"):
        module.is_arch_support_pdl = lambda: False


def _adjust_activation(module) -> None:
    if not hasattr(module, "_sgl_gelu_and_mul"):
        # The CUDA gate did not fire (is_cuda() is False on HCU today) -
        # exactly the patched wheel's state: names simply absent.
        return
    from sglang.kernels.ops.activation.activation import (
        gelu_and_mul,
        gelu_tanh_and_mul,
        silu_and_mul,
    )

    module._sgl_gelu_and_mul = gelu_and_mul
    module._sgl_gelu_tanh_and_mul = gelu_tanh_and_mul
    module._sgl_silu_and_mul = silu_and_mul


def _adjust_layernorm(module) -> None:
    for name in (
        "fused_add_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "rmsnorm",
    ):
        if hasattr(module, name):
            setattr(module, name, None)


# Module fullname -> adjuster. ``sampler`` needs no adjuster (see module
# docstring, item 7); it is listed for documentation only via tests.
ADJUSTERS = {
    "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4": (
        _adjust_quark_w4a4_mxfp4
    ),
    "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe": (
        _adjust_quark_w4a4_mxfp4_moe
    ),
    "sglang.srt.layers.quantization.quark_int4fp8_moe": _adjust_quark_int4fp8_moe,
    "sglang.srt.layers.attention.triton_backend": _adjust_triton_backend,
    "sglang.srt.layers.activation": _adjust_activation,
    "sglang.srt.layers.layernorm": _adjust_layernorm,
}


# ── targeted post-exec adjustment finder ────────────────────────────────────


class _AdjustingLoader:
    """Delegate real module execution, then adjust bindings immediately."""

    def __init__(self, loader, adjuster):
        self._loader = loader
        self._adjuster = adjuster

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        if module.__name__ in AITER_WINDOW_MODULES:
            with _aiter_absent_window(module.__name__):
                self._loader.exec_module(module)
        else:
            self._loader.exec_module(module)
        # A failure here fails the import loudly: compatibility that cannot
        # be established must not degrade into silently unpatched behavior.
        self._adjuster(module)
        setattr(module, _ADJUST_MARK, True)
        _adjusted[module.__name__] = True
        logger.debug("HCU sglang compat adjusted %s", module.__name__)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _CompatFinder(importlib.abc.MetaPathFinder):
    """Wrap the real import of exactly the adjusted SGLang modules.

    Sits at the front of ``sys.meta_path`` ONLY while HCU compat is active,
    answers for nothing else, and delegates all real loading (source
    finder, loader, spec) to the normal machinery - this is a targeted
    binding-adjustment hook, not an unconditional package fabricator.
    """

    _resolving = False

    def find_spec(self, fullname, path=None, target=None):
        adjuster = ADJUSTERS.get(fullname)
        if adjuster is None or _CompatFinder._resolving:
            return None
        _CompatFinder._resolving = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _CompatFinder._resolving = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AdjustingLoader(spec.loader, adjuster)
        return spec


def _compat_finder_active() -> bool:
    return any(isinstance(f, _CompatFinder) for f in sys.meta_path)


def activate() -> bool:
    """Install HCU SGLang import compatibility. Idempotent; HCU only."""
    global _active
    if os.environ.get(COMPAT_OPT_OUT, "1") == "0":
        logger.info("HCU sglang compat disabled via %s=0", COMPAT_OPT_OUT)
        return False
    if not is_hcu():
        return False
    if _active:
        return True
    if not _compat_finder_active():
        sys.meta_path.insert(0, _CompatFinder())
    _active = True
    logger.info(
        "HCU sglang import compatibility active (%d adjusted-module hooks, "
        "aiter-absent window on %d quark modules)",
        len(ADJUSTERS),
        len(AITER_WINDOW_MODULES),
    )
    return True
