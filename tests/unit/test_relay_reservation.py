"""Exclusive relay port reservation (ADR-69; TRT-5, CON-5)."""

from __future__ import annotations

import errno
import socket
import subprocess
import sys
from itertools import product

import pytest

from aisle.harness import relay_reservation
from aisle.harness.relay_reservation import (
    RELAY_HOST,
    HostAddress,
    ReservationError,
    address_drift,
    drifted,
    host_addresses,
    parse_ifconfig,
    reserve_relay_port,
)

pytestmark = pytest.mark.unit

LOOPBACK = (
    HostAddress("v4", "127.0.0.1"),
    HostAddress("v6", "::1"),
)
WILDCARDS = (HostAddress("v4", "0.0.0.0"), HostAddress("v6", "::"))
FLAG_COMBINATIONS = tuple(product((False, True), repeat=2))
#: TEST-NET-3: never assigned to the host, so binding it fails mid-sequence
UNASSIGNED = HostAddress("v4", "203.0.113.9")

IFCONFIG_FIXTURE = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\toptions=1203<RXCSUM,TXCSUM,TXSTATUS,SW_TIMESTAMP>
\tinet 127.0.0.1 netmask 0xff000000
\tinet6 ::1 prefixlen 128
\tinet6 fe80::1%lo0 prefixlen 64 scopeid 0x1
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether aa:bb:cc:dd:ee:ff
\tinet6 fe80::10fd:5c01:3131:2094%en0 prefixlen 64 secured scopeid 0xb
\tinet 192.168.1.94 netmask 0xffffff00 broadcast 192.168.1.255
utun0: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1500
\tinet6 fe80::32bb:eb57:d92c:6fa5%utun0 prefixlen 64 scopeid 0x10
"""
LOOPBACK_ONLY = "lo0: flags=8049<UP,LOOPBACK>\n\tinet 127.0.0.1 netmask 0xff000000\n"


def _intruder_binds(
    address: HostAddress, port: int, reuseaddr: bool, reuseport: bool, *, v6only: bool = False
) -> bool:
    """True if an independent socket binds; only an address-conflict errno
    counts as blocked, so an unrelated failure cannot pass a test vacuously."""
    family = socket.AF_INET if address.family == "v4" else socket.AF_INET6
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if reuseaddr:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if reuseport:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, int(v6only))
        sock.bind(socket.getaddrinfo(address.host, port, family, socket.SOCK_STREAM)[0][4])
    except OSError as exc:
        assert exc.errno in (errno.EADDRINUSE, errno.EACCES), exc
        return False
    else:
        return True
    finally:
        sock.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
        sock.bind(("::", 0))
        return sock.getsockname()[1]


def _bound_intruders(targets: tuple[HostAddress, ...], port: int) -> list:
    return [
        (target.host, flags)
        for target in targets
        for flags in FLAG_COMBINATIONS
        if _intruder_binds(target, port, *flags)
    ]


def test_intruder_helper_binds_a_free_port():
    """Positive control for the intruder helper (TRT-5)."""
    assert _bound_intruders((*LOOPBACK, *WILDCARDS), _free_port()) != []


def test_parse_ifconfig_returns_every_assigned_address_with_link_local_scope():
    """ADR-69 §1 (TRT-5): the reservation must cover every address SBPL
    `localhost` admits, including scoped link-local and LAN addresses."""
    assert parse_ifconfig(IFCONFIG_FIXTURE) == (
        HostAddress("v4", "127.0.0.1"),
        HostAddress("v6", "::1"),
        HostAddress("v6", "fe80::1%lo0"),
        HostAddress("v6", "fe80::10fd:5c01:3131:2094%en0"),
        HostAddress("v4", "192.168.1.94"),
        HostAddress("v6", "fe80::32bb:eb57:d92c:6fa5%utun0"),
    )


def test_parse_ifconfig_scopes_unscoped_link_local_to_dotted_interface_names():
    """ADR-69 §1 (TRT-5): the scope comes from the enclosing interface block."""
    text = LOOPBACK_ONLY + "vlan0.5: flags=8843<UP>\n\tinet6 fe80::2 prefixlen 64\n"
    assert parse_ifconfig(text)[-1] == HostAddress("v6", "fe80::2%vlan0.5")


def test_parse_ifconfig_drops_duplicates_and_reads_point_to_point_lines():
    """ADR-69 §1 (TRT-5): one entry per address, in first-seen order."""
    text = LOOPBACK_ONLY + "utun4: flags=8051<UP>\n\tinet 10.8.0.2 --> 10.8.0.1 netmask 0xff\n"
    text += "\tinet 10.8.0.2 --> 10.8.0.1 netmask 0xff\n"
    assert parse_ifconfig(text) == (HostAddress("v4", RELAY_HOST), HostAddress("v4", "10.8.0.2"))


@pytest.mark.parametrize("flag", ["tentative", "duplicated", "detached"])
def test_parse_ifconfig_refuses_addresses_that_cannot_be_bound_yet(flag):
    """ADR-69 §1 (TRT-5): an address that may become bindable later cannot be
    covered now, so it refuses rather than being skipped."""
    text = LOOPBACK_ONLY + f"en0: flags=8863<UP>\n\tinet6 fe80::3%en0 prefixlen 64 {flag} \n"
    with pytest.raises(ReservationError, match=flag):
        parse_ifconfig(text)


@pytest.mark.parametrize("text", ["", "en0: flags=8863<UP>\n\tinet 10.0.0.2 netmask 0xff\n"])
def test_parse_ifconfig_refuses_output_without_loopback(text):
    """ADR-69 §1 (TRT-5): an enumeration that misses lo0 cannot be complete."""
    with pytest.raises(ReservationError, match="loopback"):
        parse_ifconfig(text)


def test_host_addresses_reports_ifconfig_failure(monkeypatch):
    """ADR-69 §1 (TRT-5): an enumeration failure refuses with a reason."""
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="boom")
    monkeypatch.setattr(relay_reservation.subprocess, "run", lambda *a, **k: failed)
    with pytest.raises(ReservationError, match="exited 1: boom"):
        host_addresses()


def test_host_addresses_reports_a_missing_ifconfig(monkeypatch):
    """ADR-69 §1 (TRT-5): a missing enumerator refuses rather than raising raw."""

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ifconfig")

    monkeypatch.setattr(relay_reservation.subprocess, "run", missing)
    with pytest.raises(ReservationError, match="cannot enumerate"):
        host_addresses()


def test_reservation_blocks_every_intruder_bind_on_covered_addresses():
    """ADR-69 §1 (TRT-5): with the wildcard and every address held, no
    reuse-flag combination lets another socket bind the relay port."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        assert _bound_intruders((*LOOPBACK, *WILDCARDS), reservation.port) == []


def test_reservation_refuses_a_port_another_socket_already_holds():
    """ADR-69 §1 (TRT-5): any bind failure refuses; no partial reservation."""
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind((RELAY_HOST, 0))
    try:
        with pytest.raises(ReservationError, match="cannot reserve relay port"):
            reserve_relay_port(port=squatter.getsockname()[1], addresses=lambda: LOOPBACK)
    finally:
        squatter.close()


def test_failure_after_partial_hold_releases_every_socket():
    """ADR-69 §1, §4 (TRT-5): a mid-sequence bind failure leaves the port free,
    and the refusal names the address's family and position, not the address."""
    port = _free_port()
    with pytest.raises(ReservationError, match="v4 address 2") as refused:
        reserve_relay_port(port=port, addresses=lambda: (*LOOPBACK, UNASSIGNED))
    assert UNASSIGNED.host not in str(refused.value)
    assert _intruder_binds(HostAddress("v6", "::"), port, False, False)


def test_reservation_blocks_ipv4_mapped_and_v6only_wildcard_intruders():
    """ADR-69 §1 (TRT-5): IPv6 sockets reaching v4 through a mapped address,
    or holding a v6-only wildcard, cannot share the relay port either."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        mapped = [
            flags
            for flags in FLAG_COMBINATIONS
            if _intruder_binds(HostAddress("v6", "::ffff:127.0.0.1"), reservation.port, *flags)
        ]
        v6only = [
            flags
            for flags in FLAG_COMBINATIONS
            if _intruder_binds(HostAddress("v6", "::"), reservation.port, *flags, v6only=True)
        ]
    assert mapped == [] and v6only == []


@pytest.mark.parametrize("port", [-1, 65536, 70000, True])
def test_reservation_refuses_out_of_range_ports(port):
    """ADR-69 §1 (TRT-5): an invalid port refuses before any socket opens."""
    with pytest.raises(ReservationError, match="out of range"):
        reserve_relay_port(port=port, addresses=lambda: LOOPBACK)


def test_reservation_refuses_when_addresses_change_while_binding():
    """ADR-69 §3 (TRT-5): an address that appears mid-reservation is uncovered."""
    calls = iter([LOOPBACK, (*LOOPBACK, HostAddress("v4", "10.8.0.2"))])
    with pytest.raises(ReservationError, match="changed"):
        reserve_relay_port(addresses=lambda: next(calls))


def test_reservation_sockets_set_reuseaddr_and_never_reuseport():
    """ADR-69 §2 (TRT-5): SO_REUSEPORT would let a cooperating socket share P;
    SO_REUSEADDR is what lets the port be re-reserved through TIME_WAIT."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        sockets = reservation.sockets()
        assert all(s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) for s in sockets)
        assert not any(s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) for s in sockets)


def test_port_can_be_reserved_again_after_served_connections():
    """ADR-69 §2 (TRT-5): the audit serves on P and closes first, leaving
    TIME_WAIT; the session that follows must still be able to reserve P."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as first:
        port = first.port
        relay = first.relay_socket()
        relay.listen(1)
        with socket.create_connection((RELAY_HOST, port), timeout=2) as client:
            served, _ = relay.accept()
            served.close()
            client.recv(1)
    with reserve_relay_port(port=port, addresses=lambda: LOOPBACK) as second:
        assert second.port == port


def test_only_the_relay_socket_answers():
    """ADR-69 §2 (TRT-5): a non-relay reservation socket never completes a
    connection (macOS drops the SYN; Linux resets it)."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        reservation.relay_socket().listen(1)
        with socket.create_connection((RELAY_HOST, reservation.port), timeout=2):
            pass
        with pytest.raises(OSError) as unanswered:
            socket.create_connection(("::1", reservation.port), timeout=0.3)
    assert isinstance(unanswered.value, TimeoutError) or unanswered.value.errno in (
        errno.ECONNREFUSED,
        errno.ECONNRESET,
    )


def test_close_releases_the_port():
    """ADR-69 §1 (TRT-5): the reservation's lifetime is the session's."""
    reservation = reserve_relay_port(addresses=lambda: LOOPBACK)
    port = reservation.port
    reservation.close()
    assert _intruder_binds(HostAddress("v4", RELAY_HOST), port, False, False)


def test_record_names_no_addresses_and_digests_the_set_order_independently():
    """ADR-69 §4 (TRT-5): retained evidence carries counts and a digest only."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as first:
        record = first.record()
    with reserve_relay_port(addresses=lambda: tuple(reversed(LOOPBACK))) as second:
        reordered = second.record()
    with reserve_relay_port(addresses=lambda: LOOPBACK[:1]) as third:
        smaller = third.record()
    assert record["address_counts"] == {"v4": 1, "v6": 1}
    assert record["address_set_sha256"] == reordered["address_set_sha256"]
    assert record["address_set_sha256"] != smaller["address_set_sha256"]
    assert RELAY_HOST not in str(record)


def test_address_drift_is_empty_when_the_set_is_unchanged():
    """ADR-69 §3 (TRT-5, CON-5): drift is computed from an injected source."""
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        assert address_drift(reservation, addresses=lambda: tuple(reversed(LOOPBACK))) == {
            "added": 0,
            "removed": 0,
        }
        assert not drifted(reservation, addresses=lambda: LOOPBACK)


def test_address_drift_counts_added_and_removed_addresses():
    """ADR-69 §3 (TRT-5): a new address is one no reservation covers."""
    current = (HostAddress("v4", RELAY_HOST), HostAddress("v4", "10.8.0.2"))
    with reserve_relay_port(addresses=lambda: LOOPBACK) as reservation:
        assert address_drift(reservation, addresses=lambda: current) == {"added": 1, "removed": 1}
        assert drifted(reservation, addresses=lambda: current)


@pytest.mark.skipif(sys.platform != "darwin", reason="ifconfig enumeration is macOS-specific")
def test_live_host_reservation_blocks_every_intruder():
    """ADR-69 §1 (TRT-5): the reservation on this host's real address set."""
    with reserve_relay_port() as reservation:
        assert _bound_intruders((*reservation.addresses, *WILDCARDS), reservation.port) == []
