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

# Selection policy management for operator dispatch.
#
# Environment variable prefix: SGLANG_FL_* (semantically aligned with VLLM_FL_*).

from __future__ import annotations

import contextvars
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .logger_manager import get_logger

logger = get_logger()

# Valid preference values
PREFER_DEFAULT = "flagos"
PREFER_VENDOR = "vendor"
PREFER_REFERENCE = "reference"

VALID_PREFER_VALUES = frozenset({PREFER_DEFAULT, PREFER_VENDOR, PREFER_REFERENCE})


@dataclass(frozen=True)
class SelectionPolicy:
    """
    Policy for selecting operator implementations.

    Attributes:
        prefer: Which implementation kind to prefer ("flagos", "vendor", "reference")
        strict: If True, enable fallback when primary fails
        per_op_order: Per-operator custom selection order
        deny_vendors: Set of vendor names to deny
        allow_vendors: Set of vendor names to allow (whitelist)
    """

    prefer: str = PREFER_DEFAULT
    strict: bool = False
    per_op_order: Tuple[Tuple[str, Tuple[str, ...]], ...] = field(default_factory=tuple)
    deny_vendors: FrozenSet[str] = field(default_factory=frozenset)
    allow_vendors: Optional[FrozenSet[str]] = None

    def __post_init__(self):
        if self.prefer not in VALID_PREFER_VALUES:
            raise ValueError(
                f"Invalid prefer value: '{self.prefer}'. "
                f"Must be one of: {', '.join(sorted(VALID_PREFER_VALUES))}"
            )

    @classmethod
    def from_dict(
        cls,
        prefer: str = PREFER_DEFAULT,
        strict: bool = False,
        per_op_order: Optional[Dict[str, List[str]]] = None,
        deny_vendors: Optional[Set[str]] = None,
        allow_vendors: Optional[Set[str]] = None,
    ) -> "SelectionPolicy":
        """Create a SelectionPolicy from dictionary-like arguments."""
        per_op_tuple = tuple()
        if per_op_order:
            per_op_tuple = tuple((k, tuple(v)) for k, v in sorted(per_op_order.items()))

        return cls(
            prefer=prefer.lower(),
            strict=strict,
            per_op_order=per_op_tuple,
            deny_vendors=frozenset(deny_vendors) if deny_vendors else frozenset(),
            allow_vendors=frozenset(allow_vendors) if allow_vendors else None,
        )

    @property
    def per_op_order_dict(self) -> Dict[str, List[str]]:
        """Get per_op_order as a mutable dict."""
        return {k: list(v) for k, v in self.per_op_order}

    def get_default_order(self) -> List[str]:
        """Get the default selection order based on preference."""
        if self.prefer == PREFER_REFERENCE:
            return ["reference", "flagos", "vendor"]
        elif self.prefer == PREFER_VENDOR:
            return ["vendor", "flagos", "reference"]
        else:  # PREFER_DEFAULT
            return ["flagos", "vendor", "reference"]

    def is_vendor_allowed(self, vendor_name: str) -> bool:
        """Check if a vendor is allowed by this policy."""
        if vendor_name in self.deny_vendors:
            return False
        if self.allow_vendors is not None and vendor_name not in self.allow_vendors:
            return False
        return True

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for caching."""
        parts = [
            f"prefer={self.prefer}",
            f"st={int(self.strict)}",
        ]
        if self.allow_vendors:
            parts.append(f"allow={','.join(sorted(self.allow_vendors))}")
        if self.deny_vendors:
            parts.append(f"deny={','.join(sorted(self.deny_vendors))}")
        if self.per_op_order:
            per_op_str = ";".join(f"{k}={'|'.join(v)}" for k, v in self.per_op_order)
            parts.append(f"per={per_op_str}")
        return ";".join(parts)

    def __hash__(self) -> int:
        return hash(
            (
                self.prefer,
                self.strict,
                self.per_op_order,
                self.deny_vendors,
                self.allow_vendors,
            )
        )


class PolicyManager:
    """
    Singleton manager for selection policies.

    Supports global policy (from env) and context-local policy (context managers).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, "_policy_epoch"):
            return
        self._policy_epoch = 0
        self._policy_epoch_lock = threading.Lock()
        self._global_policy = None
        self._global_policy_lock = threading.Lock()
        self._policy_var = contextvars.ContextVar(
            "sglang_fl_selection_policy",
            default=None,
        )

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def get_policy_epoch(self) -> int:
        return self._policy_epoch

    def bump_policy_epoch(self) -> int:
        with self._policy_epoch_lock:
            self._policy_epoch += 1
            return self._policy_epoch

    def get_policy(self) -> SelectionPolicy:
        """Get the current effective policy (context or global)."""
        ctx_policy = self._policy_var.get()
        if ctx_policy is not None:
            return ctx_policy

        if self._global_policy is None:
            with self._global_policy_lock:
                if self._global_policy is None:
                    self._global_policy = self._policy_from_env()
        return self._global_policy

    def set_global_policy(self, policy: SelectionPolicy) -> SelectionPolicy:
        """Set the global policy and return the old one."""
        with self._global_policy_lock:
            old_policy = self._global_policy
            self._global_policy = policy
            self.bump_policy_epoch()
            return old_policy if old_policy else self._policy_from_env()

    def reset_global_policy(self) -> None:
        """Reset global policy to environment defaults."""
        with self._global_policy_lock:
            self._global_policy = None
            self.bump_policy_epoch()

    def create_policy_context(self, policy: SelectionPolicy):
        """Create a context manager for temporary policy override."""
        return _PolicyContext(self, policy)

    def _get_policy_var(self):
        return self._policy_var

    @staticmethod
    def _parse_csv_set(value: str) -> Set[str]:
        if not value:
            return set()
        return {x.strip() for x in value.split(",") if x.strip()}

    @staticmethod
    def _parse_per_op(value: str) -> Dict[str, List[str]]:
        """Parse per-op order string (format: op1=a|b|c;op2=x|y)."""
        if not value:
            return {}
        result: Dict[str, List[str]] = {}
        parts = [p.strip() for p in value.split(";") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            op_name, order_str = part.split("=", 1)
            op_name = op_name.strip()
            order = [x.strip() for x in order_str.split("|") if x.strip()]
            if op_name and order:
                result[op_name] = order
        return result

    def _policy_from_config(self, config_path: str) -> SelectionPolicy:
        """Create a SelectionPolicy from a YAML configuration file."""
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml not installed, cannot load config file")
            return SelectionPolicy()

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file '{config_path}' not found.")

        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}

        prefer_str = str(config.get("prefer", PREFER_DEFAULT)).strip().lower()
        if prefer_str not in VALID_PREFER_VALUES:
            prefer_str = PREFER_DEFAULT

        strict = bool(config.get("strict", False))

        deny_vendors_raw = config.get("deny_vendors")
        deny_vendors: Optional[Set[str]] = None
        if deny_vendors_raw:
            if isinstance(deny_vendors_raw, list):
                deny_vendors = {str(v).strip() for v in deny_vendors_raw if v}
            elif isinstance(deny_vendors_raw, str):
                deny_vendors = self._parse_csv_set(deny_vendors_raw)

        allow_vendors_raw = config.get("allow_vendors")
        allow_vendors: Optional[Set[str]] = None
        if allow_vendors_raw:
            if isinstance(allow_vendors_raw, list):
                allow_vendors = {str(v).strip() for v in allow_vendors_raw if v}
            elif isinstance(allow_vendors_raw, str):
                allow_vendors = self._parse_csv_set(allow_vendors_raw)

        per_op_raw = config.get("op_backends")
        per_op_order: Optional[Dict[str, List[str]]] = None
        if per_op_raw and isinstance(per_op_raw, dict):
            per_op_order = {}
            for op_name, order in per_op_raw.items():
                if isinstance(order, list):
                    per_op_order[str(op_name)] = [str(o).strip() for o in order if o]
                elif isinstance(order, str):
                    per_op_order[str(op_name)] = [
                        o.strip() for o in order.split("|") if o.strip()
                    ]

        return SelectionPolicy.from_dict(
            prefer=prefer_str,
            strict=strict,
            per_op_order=per_op_order,
            deny_vendors=deny_vendors,
            allow_vendors=allow_vendors,
        )

    def _policy_from_env(self) -> SelectionPolicy:
        """
        Create a SelectionPolicy from config file or environment variables.

        Priority (highest to lowest):
        1. SGLANG_FL_CONFIG: Path to YAML config file (complete override)
        2. Environment variables: Override specific items
        3. Platform-specific config file: Auto-detected defaults
        4. Built-in default values
        """
        # Priority 1: User-specified config file
        config_path = os.environ.get("SGLANG_FL_CONFIG", "").strip()
        if config_path and os.path.isfile(config_path):
            return self._policy_from_config(config_path)

        # Priority 3: Platform-specific config as base
        from .config import get_config_path

        platform_config_path = get_config_path()
        platform_policy = None
        if platform_config_path:
            try:
                platform_policy = self._policy_from_config(str(platform_config_path))
            except Exception as e:
                logger.warning("Failed to load platform config: %s", e)

        # Priority 2: Environment variables override platform config
        env_prefer_str = os.environ.get("SGLANG_FL_PREFER", "").strip().lower()
        env_strict_str = os.environ.get("SGLANG_FL_STRICT", "").strip()
        env_deny_str = os.environ.get("SGLANG_FL_DENY_VENDORS", "").strip()
        env_allow_str = os.environ.get("SGLANG_FL_ALLOW_VENDORS", "").strip()
        env_per_op_str = os.environ.get("SGLANG_FL_PER_OP", "").strip()

        # Resolve: env > platform > default
        if env_prefer_str and env_prefer_str in VALID_PREFER_VALUES:
            prefer_str = env_prefer_str
        elif platform_policy:
            prefer_str = platform_policy.prefer
        else:
            prefer_str = PREFER_DEFAULT

        if env_strict_str:
            strict = env_strict_str == "1"
        elif platform_policy:
            strict = platform_policy.strict
        else:
            strict = True

        if env_deny_str:
            deny_vendors = self._parse_csv_set(env_deny_str)
        elif platform_policy and platform_policy.deny_vendors:
            deny_vendors = set(platform_policy.deny_vendors)
        else:
            deny_vendors = None

        if env_allow_str:
            allow_vendors = self._parse_csv_set(env_allow_str)
        elif platform_policy and platform_policy.allow_vendors:
            allow_vendors = set(platform_policy.allow_vendors)
        else:
            allow_vendors = None

        if env_per_op_str:
            per_op_order = self._parse_per_op(env_per_op_str)
        elif platform_policy and platform_policy.per_op_order:
            per_op_order = platform_policy.per_op_order_dict
        else:
            per_op_order = None

        return SelectionPolicy.from_dict(
            prefer=prefer_str,
            strict=strict,
            per_op_order=per_op_order,
            deny_vendors=deny_vendors,
            allow_vendors=allow_vendors,
        )


class _PolicyContext:
    """Context manager for temporary policy override."""

    def __init__(self, manager: PolicyManager, policy: SelectionPolicy):
        self._manager = manager
        self._policy = policy
        self._token = None

    def __enter__(self) -> "_PolicyContext":
        policy_var = self._manager._get_policy_var()
        self._token = policy_var.set(self._policy)
        self._manager.bump_policy_epoch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            policy_var = self._manager._get_policy_var()
            policy_var.reset(self._token)
            self._manager.bump_policy_epoch()


# Convenience functions
def get_policy() -> SelectionPolicy:
    return PolicyManager.get_instance().get_policy()


def set_global_policy(policy: SelectionPolicy) -> SelectionPolicy:
    return PolicyManager.get_instance().set_global_policy(policy)


def reset_global_policy() -> None:
    PolicyManager.get_instance().reset_global_policy()


def policy_context(policy: SelectionPolicy) -> _PolicyContext:
    return _PolicyContext(PolicyManager.get_instance(), policy)


def policy_from_config(config_path: str) -> SelectionPolicy:
    return PolicyManager.get_instance()._policy_from_config(config_path)


def with_strict_mode() -> _PolicyContext:
    current = get_policy()
    strict_policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=True,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(strict_policy)


def with_preference(prefer: str) -> _PolicyContext:
    current = get_policy()
    policy = SelectionPolicy.from_dict(
        prefer=prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(policy)


def with_allowed_vendors(*vendors: str) -> _PolicyContext:
    current = get_policy()
    policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=set(current.deny_vendors),
        allow_vendors=set(vendors),
    )
    return policy_context(policy)


def with_denied_vendors(*vendors: str) -> _PolicyContext:
    current = get_policy()
    denied = set(current.deny_vendors)
    denied.update(vendors)
    policy = SelectionPolicy.from_dict(
        prefer=current.prefer,
        strict=current.strict,
        per_op_order={k: list(v) for k, v in current.per_op_order},
        deny_vendors=denied,
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    )
    return policy_context(policy)
