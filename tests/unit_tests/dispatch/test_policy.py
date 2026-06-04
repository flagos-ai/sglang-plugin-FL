# Tests for SelectionPolicy and PolicyManager.

import pytest

from sglang_fl.dispatch.policy import (
    PREFER_DEFAULT,
    PREFER_REFERENCE,
    PREFER_VENDOR,
    PolicyManager,
    SelectionPolicy,
    get_policy,
    policy_context,
    policy_from_config,
    reset_global_policy,
    set_global_policy,
    with_allowed_vendors,
    with_denied_vendors,
    with_preference,
    with_strict_mode,
)


class TestSelectionPolicy:
    def test_default_values(self):
        policy = SelectionPolicy()
        assert policy.prefer == PREFER_DEFAULT
        assert policy.strict is False
        assert policy.per_op_order == ()
        assert policy.deny_vendors == frozenset()
        assert policy.allow_vendors is None

    def test_invalid_prefer_raises(self):
        with pytest.raises(ValueError, match="Invalid prefer value"):
            SelectionPolicy(prefer="invalid")

    def test_from_dict(self):
        policy = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"silu_and_mul": ["vendor", "flagos"]},
            deny_vendors={"ascend"},
            allow_vendors={"cuda"},
        )
        assert policy.prefer == "vendor"
        assert policy.strict is True
        assert policy.deny_vendors == frozenset({"ascend"})
        assert policy.allow_vendors == frozenset({"cuda"})

    def test_get_default_order_flagos(self):
        policy = SelectionPolicy(prefer=PREFER_DEFAULT)
        order = policy.get_default_order()
        assert order[0] == "flagos"

    def test_get_default_order_vendor(self):
        policy = SelectionPolicy(prefer=PREFER_VENDOR)
        order = policy.get_default_order()
        assert order[0] == "vendor"

    def test_get_default_order_reference(self):
        policy = SelectionPolicy(prefer=PREFER_REFERENCE)
        order = policy.get_default_order()
        assert order[0] == "reference"

    def test_per_op_order_dict(self):
        policy = SelectionPolicy.from_dict(
            per_op_order={"silu_and_mul": ["vendor", "flagos", "reference"]}
        )
        d = policy.per_op_order_dict
        assert d["silu_and_mul"] == ["vendor", "flagos", "reference"]

    def test_fingerprint_changes_with_policy(self):
        p1 = SelectionPolicy(prefer=PREFER_DEFAULT)
        p2 = SelectionPolicy(prefer=PREFER_VENDOR)
        assert p1.fingerprint() != p2.fingerprint()

    def test_fingerprint_same_for_equal_policies(self):
        p1 = SelectionPolicy(prefer=PREFER_DEFAULT, strict=True)
        p2 = SelectionPolicy(prefer=PREFER_DEFAULT, strict=True)
        assert p1.fingerprint() == p2.fingerprint()


class TestPolicyManager:
    def test_default_policy(self):
        pm = PolicyManager.get_instance()
        pm.reset_global_policy()
        policy = pm.get_policy()
        assert policy.prefer == PREFER_DEFAULT

    def test_set_and_get(self):
        pm = PolicyManager.get_instance()
        new_policy = SelectionPolicy(prefer=PREFER_VENDOR)
        pm.set_global_policy(new_policy)
        assert pm.get_policy().prefer == PREFER_VENDOR

    def test_epoch_increments_on_set(self):
        pm = PolicyManager.get_instance()
        e0 = pm.get_policy_epoch()
        pm.set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        assert pm.get_policy_epoch() > e0


class TestGlobalPolicyFunctions:
    def test_get_default(self):
        reset_global_policy()
        policy = get_policy()
        assert policy.prefer == PREFER_DEFAULT

    def test_set_global(self):
        set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        assert get_policy().prefer == PREFER_VENDOR

    def test_policy_context(self):
        reset_global_policy()
        assert get_policy().prefer == PREFER_DEFAULT
        with policy_context(SelectionPolicy(prefer=PREFER_REFERENCE)):
            assert get_policy().prefer == PREFER_REFERENCE
        assert get_policy().prefer == PREFER_DEFAULT

    def test_with_preference(self):
        reset_global_policy()
        with with_preference(PREFER_VENDOR):
            assert get_policy().prefer == PREFER_VENDOR
        assert get_policy().prefer == PREFER_DEFAULT

    def test_with_strict_mode(self):
        reset_global_policy()
        with with_strict_mode():
            assert get_policy().strict is True
        assert get_policy().strict is False

    def test_with_allowed_vendors(self):
        reset_global_policy()
        with with_allowed_vendors("cuda"):
            policy = get_policy()
            assert policy.allow_vendors == frozenset({"cuda"})
        assert get_policy().allow_vendors is None

    def test_with_denied_vendors(self):
        reset_global_policy()
        with with_denied_vendors("ascend"):
            policy = get_policy()
            assert "ascend" in policy.deny_vendors
        assert get_policy().deny_vendors == frozenset()

    def test_nested_contexts(self):
        reset_global_policy()
        with with_preference(PREFER_VENDOR):
            assert get_policy().prefer == PREFER_VENDOR
            with with_denied_vendors("ascend"):
                p = get_policy()
                assert p.prefer == PREFER_VENDOR
                assert "ascend" in p.deny_vendors
            assert get_policy().deny_vendors == frozenset()


class TestPolicyFromConfig:
    def test_yaml_config(self, tmp_path):
        config_file = tmp_path / "dispatch.yaml"
        config_file.write_text(
            "prefer: vendor\n"
            "strict: true\n"
            "deny_vendors:\n"
            "  - ascend\n"
            "allow_vendors:\n"
            "  - cuda\n"
            "op_backends:\n"
            "  silu_and_mul:\n"
            "    - vendor\n"
            "    - reference\n"
        )
        policy = policy_from_config(str(config_file))
        assert policy.prefer == "vendor"
        assert policy.strict is True
        assert "ascend" in policy.deny_vendors
        assert policy.allow_vendors == frozenset({"cuda"})

    def test_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            policy_from_config("/nonexistent/path.yaml")
