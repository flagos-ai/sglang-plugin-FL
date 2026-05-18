# Built-in operator implementations registration.
#
# Registers DEFAULT (FlagOS), REFERENCE (PyTorch), and VENDOR implementations.

from __future__ import annotations

import importlib
import os

from .registry import OpRegistry
from .logger_manager import get_logger

logger = get_logger()

_VENDOR_BACKENDS_DIR = os.path.join(os.path.dirname(__file__), "backends", "vendor")


def _register_vendor_backends(registry: OpRegistry) -> None:
    """Auto-discover and register all vendor backends."""
    if not os.path.isdir(_VENDOR_BACKENDS_DIR):
        return

    for vendor_name in sorted(os.listdir(_VENDOR_BACKENDS_DIR)):
        vendor_path = os.path.join(_VENDOR_BACKENDS_DIR, vendor_name)
        if not os.path.isdir(vendor_path) or vendor_name.startswith("_"):
            continue
        if vendor_name == "template":
            continue

        register_ops_path = os.path.join(vendor_path, "register_ops.py")
        if not os.path.isfile(register_ops_path):
            continue

        module_name = f"sglang_fl.dispatch.backends.vendor.{vendor_name}.register_ops"
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "register_builtins"):
                mod.register_builtins(registry)
                logger.debug(f"Registered {vendor_name} vendor operators")
        except Exception as e:
            logger.debug(f"{vendor_name} operators not available: {e}")


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all built-in operator implementations.

    Order: FlagOS (DEFAULT) → Reference (REFERENCE) → Vendors (VENDOR)
    """
    # Register FlagOS (DEFAULT)
    try:
        from .backends.flagos.register_ops import (
            register_builtins as register_flagos,
        )

        register_flagos(registry)
        logger.debug("Registered FlagOS operators")
    except Exception as e:
        logger.warning(f"Failed to register FlagOS operators: {e}")

    # Register Reference (PyTorch)
    try:
        from .backends.reference.register_ops import (
            register_builtins as register_reference,
        )

        register_reference(registry)
        logger.debug("Registered Reference operators")
    except Exception as e:
        logger.warning(f"Failed to register Reference operators: {e}")

    # Auto-discover vendor backends
    _register_vendor_backends(registry)
