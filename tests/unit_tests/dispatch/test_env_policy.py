# Tests for environment variable driven policy loading.

import os
from unittest.mock import patch

import pytest

from sglang_fl.dispatch.policy import (
    PREFER_DEFAULT,
    PREFER_REFERENCE,
    PREFER_VENDOR,
    PolicyManager,
    SelectionPolicy,
    reset_global_policy,
)


class TestEnvVarPrefer:
    """Test SGLANG_FL_PREFER env var."""

    def test_prefer_vendor(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "vendor"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_VENDOR

    def test_prefer_reference(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "reference"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_REFERENCE

    def test_prefer_flagos(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "flagos"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_DEFAULT

    def test_prefer_invalid_falls_back_to_default(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "invalid_value"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_DEFAULT

    def test_prefer_empty_string(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": ""}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_DEFAULT

    def test_prefer_with_whitespace(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "  vendor  "}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_VENDOR

    def test_prefer_case_insensitive(self):
        with patch.dict(os.environ, {"SGLANG_FL_PREFER": "VENDOR"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_VENDOR


class TestEnvVarStrict:
    """Test SGLANG_FL_STRICT env var."""

    def test_strict_enabled(self):
        with patch.dict(os.environ, {"SGLANG_FL_STRICT": "1"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.strict is True

    def test_strict_disabled(self):
        with patch.dict(os.environ, {"SGLANG_FL_STRICT": "0"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.strict is False

    def test_strict_empty_uses_platform_default(self):
        """When SGLANG_FL_STRICT is not set, strict comes from platform config or code default."""
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_FL_STRICT"}
        with patch.dict(os.environ, env, clear=True):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            # strict value depends on platform config (nvidia.yaml has no strict → False)
            # Just verify it's a bool and doesn't crash
            assert isinstance(policy.strict, bool)


class TestEnvVarDenyVendors:
    """Test SGLANG_FL_DENY_VENDORS env var."""

    def test_single_vendor(self):
        with patch.dict(os.environ, {"SGLANG_FL_DENY_VENDORS": "ascend"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert "ascend" in policy.deny_vendors

    def test_multiple_vendors(self):
        with patch.dict(os.environ, {"SGLANG_FL_DENY_VENDORS": "ascend,metax,iluvatar"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.deny_vendors == frozenset({"ascend", "metax", "iluvatar"})

    def test_whitespace_handling(self):
        with patch.dict(os.environ, {"SGLANG_FL_DENY_VENDORS": " ascend , metax "}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert "ascend" in policy.deny_vendors
            assert "metax" in policy.deny_vendors

    def test_empty_string(self):
        with patch.dict(os.environ, {"SGLANG_FL_DENY_VENDORS": ""}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.deny_vendors == frozenset()

    def test_trailing_comma(self):
        with patch.dict(os.environ, {"SGLANG_FL_DENY_VENDORS": "ascend,"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.deny_vendors == frozenset({"ascend"})


class TestEnvVarAllowVendors:
    """Test SGLANG_FL_ALLOW_VENDORS env var."""

    def test_single_vendor(self):
        with patch.dict(os.environ, {"SGLANG_FL_ALLOW_VENDORS": "cuda"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.allow_vendors == frozenset({"cuda"})

    def test_multiple_vendors(self):
        with patch.dict(os.environ, {"SGLANG_FL_ALLOW_VENDORS": "cuda,ascend"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.allow_vendors == frozenset({"cuda", "ascend"})

    def test_empty_means_no_filter(self):
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_FL_ALLOW_VENDORS"}
        with patch.dict(os.environ, env, clear=True):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.allow_vendors is None


class TestEnvVarPerOp:
    """Test SGLANG_FL_PER_OP env var."""

    def test_single_op(self):
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": "silu_and_mul=vendor|flagos|reference"}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            d = policy.per_op_order_dict
            assert "silu_and_mul" in d
            assert d["silu_and_mul"] == ["vendor", "flagos", "reference"]

    def test_multiple_ops(self):
        val = "silu_and_mul=vendor|flagos;rms_norm=flagos|reference"
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": val}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            d = policy.per_op_order_dict
            assert d["silu_and_mul"] == ["vendor", "flagos"]
            assert d["rms_norm"] == ["flagos", "reference"]

    def test_whitespace_in_per_op(self):
        val = " silu_and_mul = vendor | flagos "
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": val}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            d = policy.per_op_order_dict
            assert "silu_and_mul" in d
            assert "vendor" in d["silu_and_mul"]
            assert "flagos" in d["silu_and_mul"]

    def test_empty_string(self):
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": ""}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.per_op_order == ()

    def test_trailing_semicolon(self):
        val = "silu_and_mul=vendor|flagos;"
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": val}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            d = policy.per_op_order_dict
            assert "silu_and_mul" in d

    def test_malformed_entry_skipped(self):
        # Entry without '=' should be skipped
        val = "silu_and_mul=vendor|flagos;bad_entry;rms_norm=reference"
        with patch.dict(os.environ, {"SGLANG_FL_PER_OP": val}, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            d = policy.per_op_order_dict
            assert "silu_and_mul" in d
            assert "rms_norm" in d
            assert "bad_entry" not in d


class TestEnvVarCombined:
    """Test multiple env vars working together."""

    def test_prefer_and_deny(self):
        env = {
            "SGLANG_FL_PREFER": "vendor",
            "SGLANG_FL_DENY_VENDORS": "ascend",
        }
        with patch.dict(os.environ, env, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_VENDOR
            assert "ascend" in policy.deny_vendors

    def test_all_env_vars(self):
        env = {
            "SGLANG_FL_PREFER": "vendor",
            "SGLANG_FL_STRICT": "1",
            "SGLANG_FL_DENY_VENDORS": "metax",
            "SGLANG_FL_ALLOW_VENDORS": "cuda,ascend",
            "SGLANG_FL_PER_OP": "silu_and_mul=vendor|reference",
        }
        with patch.dict(os.environ, env, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            assert policy.prefer == PREFER_VENDOR
            assert policy.strict is True
            assert "metax" in policy.deny_vendors
            assert policy.allow_vendors == frozenset({"cuda", "ascend"})
            assert policy.per_op_order_dict["silu_and_mul"] == ["vendor", "reference"]


class TestEnvVarConfigFile:
    """Test SGLANG_FL_CONFIG env var (YAML file override)."""

    def test_config_file_overrides_env(self, tmp_path):
        config_file = tmp_path / "dispatch.yaml"
        config_file.write_text(
            "prefer: reference\n"
            "strict: false\n"
            "deny_vendors:\n"
            "  - cuda\n"
        )
        env = {
            "SGLANG_FL_CONFIG": str(config_file),
            "SGLANG_FL_PREFER": "vendor",  # Should be overridden by config file
        }
        with patch.dict(os.environ, env, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            # Config file takes priority over env vars
            assert policy.prefer == PREFER_REFERENCE
            assert policy.strict is False
            assert "cuda" in policy.deny_vendors

    def test_nonexistent_config_file_falls_through(self):
        env = {
            "SGLANG_FL_CONFIG": "/nonexistent/path/config.yaml",
            "SGLANG_FL_PREFER": "vendor",
        }
        with patch.dict(os.environ, env, clear=False):
            pm = PolicyManager.get_instance()
            pm.reset_global_policy()
            policy = pm.get_policy()
            # Config file doesn't exist, falls through to env vars
            assert policy.prefer == PREFER_VENDOR
