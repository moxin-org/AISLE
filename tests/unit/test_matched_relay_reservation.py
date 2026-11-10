"""The session controller holds the relay port for a loopback session (ADR-69; TRT-5)."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from aisle.harness.matched_session import AdmissionError
from aisle.harness.relay_reservation import HostAddress, ReservationError

pytestmark = pytest.mark.unit
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

LOOPBACK = (HostAddress("v4", "127.0.0.1"), HostAddress("v6", "::1"))


class _Policy:
    def __init__(self, network_policy, port=None):
        self.network_policy = network_policy
        self.loopback_port = port


def _free_port() -> int:
    with socket.socket(socket.AF_INET6) as sock:
        sock.bind(("::", 0))
        return sock.getsockname()[1]


def test_loopback_session_holds_its_pinned_port_and_retains_the_record(tmp_path: Path):
    """ADR-69 §2, §5 (TRT-5): the controller reserves the pinned port for the
    whole session and retains the address-free reservation record."""
    from matched_campaign import session_relay_reservation

    port = _free_port()
    with session_relay_reservation(
        _Policy("loopback", port), tmp_path, addresses=lambda: LOOPBACK
    ) as reservation:
        assert reservation.port == port
        with socket.socket() as intruder, pytest.raises(OSError):
            intruder.bind(("127.0.0.1", port))
    record = json.loads((tmp_path / "relay-reservation.json").read_text())
    assert record["port"] == port and "127.0.0.1" not in json.dumps(record)


def test_deny_external_session_takes_no_reservation(tmp_path: Path):
    """TRT-5: only a loopback grant needs a held port."""
    from matched_campaign import session_relay_reservation

    with session_relay_reservation(_Policy("deny-external"), tmp_path) as reservation:
        assert reservation is None
    assert not (tmp_path / "relay-reservation.json").exists()


def test_unreservable_port_refuses_the_session(tmp_path: Path):
    """ADR-69 §2 (TRT-5): a port the controller cannot hold on every address
    refuses the launch instead of running unconfined."""
    from matched_campaign import session_relay_reservation

    def refused():
        raise ReservationError("an inet6 address on en0 is tentative")

    with pytest.raises(AdmissionError, match="relay port"):
        with session_relay_reservation(
            _Policy("loopback", _free_port()), tmp_path, addresses=refused
        ):
            pass


def test_reservation_sockets_are_not_inheritable(tmp_path: Path):
    """ADR-69 PR B (TRT-5): no reservation descriptor may reach a confined
    child, where it could listen on, or release, the relay port."""
    from matched_campaign import session_relay_reservation

    with session_relay_reservation(
        _Policy("loopback", _free_port()), tmp_path, addresses=lambda: LOOPBACK
    ) as reservation:
        assert not any(sock.get_inheritable() for sock in reservation.sockets())
