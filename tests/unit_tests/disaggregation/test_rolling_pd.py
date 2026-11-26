"""Protocol isolation and failure semantics for rolling PD transport."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

spec = spec_from_file_location(
    "rolling",
    Path(__file__).parents[3] / "sglang_fl/disaggregation/rolling.py",
)
rolling = module_from_spec(spec)
spec.loader.exec_module(rolling)


class Engine:
    def __init__(self):
        self.calls = []
        self.result = 0

    def get_session_id(self):
        return "10.0.0.1:12345"

    def get_ib_device(self):
        return "mlx5_100"

    def batch_transfer_sync(self, *args):
        self.calls.append(("transfer", args))
        return self.result

    def batch_register(self, *args):
        self.calls.append(("register", args))
        return self.result

    def batch_register_host(self, *args):
        self.calls.append(("host", args))
        return self.result

    def batch_deregister(self, *args):
        self.calls.append(("deregister", args))
        return self.result

    def close(self):
        self.calls.append(("close", ()))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.flagcx, self.legacy = Engine(), Engine()
        self.observed = []
        self.engine = rolling.RollingPDEngine(
            self.flagcx,
            self.legacy,
            lambda code, size: self.observed.append((code, size)),
        )

    def test_flagcx_receiver_uses_only_flagcx(self):
        self.engine.batch_transfer_sync("flagcx://10.0.0.2:9000", [1], [2], [8])
        self.assertEqual(
            self.flagcx.calls, [("transfer", ("10.0.0.2:9000", [1], [2], [8]))]
        )
        self.assertEqual(self.legacy.calls, [])
        self.assertEqual(self.observed, [])

    def test_legacy_receiver_uses_only_explicit_migration_engine(self):
        self.engine.batch_transfer_sync("10.0.0.2:9000", [1], [2], [8])
        self.assertEqual(self.flagcx.calls, [])
        self.assertEqual(len(self.legacy.calls), 1)
        self.assertEqual(self.observed, [(0, 8)])

    def test_flagcx_failure_never_falls_back(self):
        self.flagcx.result = -1
        self.assertEqual(
            self.engine.batch_transfer_sync("flagcx://10.0.0.2:9000", [1], [2], [8]), -1
        )
        self.assertEqual(self.legacy.calls, [])

    def test_legacy_denied_without_explicit_admission(self):
        engine = rolling.RollingPDEngine(self.flagcx)
        with self.assertRaises(RuntimeError):
            engine.batch_transfer_sync("10.0.0.2:9000", [1], [2], [8])
        self.assertEqual(self.flagcx.calls, [])

    def test_unknown_or_malformed_protocol_is_not_retried(self):
        for session in [
            "tcp://10.0.0.2:9000",
            "flagcx://10.0.0.2:no",
            "10.0.0.2:0",
            "10.0.0.2:99999",
        ]:
            with self.assertRaises(ValueError):
                self.engine.batch_transfer_sync(session, [1], [2], [8])
        self.assertEqual(self.flagcx.calls, [])
        self.assertEqual(self.legacy.calls, [])

    def test_registration_failure_is_fatal_before_serving(self):
        self.legacy.result = -1
        with self.assertRaises(RuntimeError):
            self.engine.batch_register([1], [8])

    def test_host_registration_uses_host_api_and_native_legacy_api(self):
        self.engine.batch_register_host([1], [8])
        self.assertEqual(self.flagcx.calls[0][0], "host")
        self.assertEqual(self.legacy.calls[0][0], "register")

    def test_descriptor_mismatch_does_not_transfer(self):
        with self.assertRaises(ValueError):
            self.engine.batch_transfer_sync("flagcx://10.0.0.2:9000", [1], [2], [])
        self.assertEqual(self.flagcx.calls, [])

    def test_receiver_marker_and_process_owned_legacy_engine(self):
        self.assertEqual(self.engine.get_session_id(), "flagcx://10.0.0.1:12345")
        self.engine.close()
        self.assertEqual(self.flagcx.calls, [("close", ())])
        self.assertEqual(self.legacy.calls, [])


if __name__ == "__main__":
    unittest.main()
