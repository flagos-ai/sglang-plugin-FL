"""Explicit peer-protocol selection during a Mooncake-to-FlagCX rolling upgrade.

The session string remains an opaque native protocol field. New receivers
advertise flagcx://host:port. Untagged sessions belong to existing Mooncake
receivers and are admitted only by an explicitly enabled migration sender.
No transfer failure ever selects a different engine.
"""

FLAGCX_PREFIX = "flagcx://"


def endpoint(session):
    assert isinstance(session, str) and session
    value = (
        session[len(FLAGCX_PREFIX) :] if session.startswith(FLAGCX_PREFIX) else session
    )
    if "://" in value:
        raise ValueError("Unknown PD session protocol")
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("Invalid PD receiver session")
    return value


class RollingPDEngine:
    def __init__(self, flagcx, legacy=None, on_legacy=None):
        self.flagcx = flagcx
        self.legacy = legacy
        self.on_legacy = on_legacy

    def get_session_id(self):
        return FLAGCX_PREFIX + endpoint(self.flagcx.get_session_id())

    def get_ib_device(self):
        return self.flagcx.get_ib_device()

    def batch_register(self, ptrs, lengths):
        if len(ptrs) != len(lengths):
            raise ValueError("PD registration descriptor count mismatch")
        result = self.flagcx.batch_register(ptrs, lengths)
        if result != 0:
            raise RuntimeError("FlagCX PD memory registration failed")
        if self.legacy is not None and self.legacy.batch_register(ptrs, lengths) != 0:
            raise RuntimeError("Migration Mooncake PD memory registration failed")
        return 0

    def batch_register_host(self, ptrs, lengths):
        if len(ptrs) != len(lengths):
            raise ValueError("PD registration descriptor count mismatch")
        result = self.flagcx.batch_register_host(ptrs, lengths)
        if result != 0:
            raise RuntimeError("FlagCX PD host memory registration failed")
        if self.legacy is not None and self.legacy.batch_register(ptrs, lengths) != 0:
            raise RuntimeError("Migration Mooncake PD host memory registration failed")
        return 0

    def batch_deregister(self, ptrs):
        result = self.flagcx.batch_deregister(ptrs)
        if self.legacy is not None and self.legacy.batch_deregister(ptrs) != 0:
            raise RuntimeError("Migration Mooncake PD deregistration failed")
        return result

    def batch_transfer_sync(self, session, buffers, peer_addresses, lengths):
        if not len(buffers) == len(peer_addresses) == len(lengths):
            raise ValueError("PD transfer descriptor count mismatch")
        address = endpoint(session)
        if session.startswith(FLAGCX_PREFIX):
            return self.flagcx.batch_transfer_sync(
                address, buffers, peer_addresses, lengths
            )
        if self.legacy is None:
            raise RuntimeError("Untagged legacy PD receiver is not admitted")
        result = self.legacy.batch_transfer_sync(
            address, buffers, peer_addresses, lengths
        )
        if self.on_legacy is not None:
            self.on_legacy(result, sum(lengths))
        return result

    def transfer_sync(self, session, buffer, peer_address, length):
        return self.batch_transfer_sync(session, [buffer], [peer_address], [length])

    def close(self):
        # The native singleton is process-owned; do not close an engine shared
        # with other native components. The historical FlagCX engine owns its
        # own registrations and is closed exactly once here.
        return self.flagcx.close()
