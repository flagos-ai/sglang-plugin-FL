import struct

import pytest

pytest.importorskip("sglang")

from sglang_fl.disaggregation.conn import KVArgsRegisterInfo


def _registration_message(staging_base_ptr=b"", staging_total_size=b""):
    return [
        b"room",
        b"127.0.0.1",
        b"12345",
        b"session",
        struct.pack("2Q", 100, 200),
        struct.pack("Q", 300),
        b"",
        b"0",
        b"1",
        b"128",
        b"",
        b"",
        b"0",
        staging_base_ptr,
        staging_total_size,
    ]


def test_registration_parses_sglang_0518_staging_frames():
    info = KVArgsRegisterInfo.from_zmq(
        _registration_message(struct.pack("Q", 0x1234), b"4096")
    )

    assert info.staging is not None
    assert info.staging.base_ptr == 0x1234
    assert info.staging.total_size == 4096


def test_registration_without_staging_frames_uses_none():
    info = KVArgsRegisterInfo.from_zmq(_registration_message())

    assert info.staging is None
