"""Exclusive provider-relay port reservation (ADR-69; TRT-5).

macOS SBPL `(remote tcp "localhost:P")` admits TCP port P on every address
assigned to the host, not only loopback. (`remote ip` would also admit UDP,
which this TCP reservation does not cover; ADR-69 requires `remote tcp`.)
The grant becomes exclusive when the
controller holds P on the dual-stack wildcard and on every assigned address,
none with SO_REUSEPORT, so no other socket can bind P anywhere the sandbox
can reach. Only the 127.0.0.1 socket is handed to the relay to listen on; the
others stay bound and never answer (measured on macOS 27.2: the kernel drops
the SYN to a bound, non-listening socket).
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

IFCONFIG = "/sbin/ifconfig"
RELAY_HOST = "127.0.0.1"
SCHEMA_VERSION = "aisle.relay-reservation.v1"
#: inet6 states in which the address cannot be bound yet, but may become
#: bindable later, so the reservation could not cover it: refuse, never skip
_UNUSABLE_INET6_FLAGS = frozenset({"tentative", "duplicated", "detached"})
_IFCONFIG_ENV = {"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}

_INTERFACE = re.compile(r"^(\S+?): flags=")
_INET = re.compile(r"^\s+inet (\S+)")
_INET6 = re.compile(r"^\s+inet6 (\S+)(.*)$")


class ReservationError(RuntimeError):
    """The relay port cannot be held exclusively; the session must not start."""


@dataclass(frozen=True)
class HostAddress:
    family: Literal["v4", "v6"]
    host: str  # link-local IPv6 carries its %interface scope


AddressSource = Callable[[], tuple[HostAddress, ...]]


def _inet6_address(host: str, flags: str, interface: str | None) -> HostAddress:
    unusable = _UNUSABLE_INET6_FLAGS.intersection(flags.split())
    if unusable:
        raise ReservationError(
            f"an inet6 address on {interface} is {'/'.join(sorted(unusable))}; "
            "retry once it settles"
        )
    if "%" not in host and ipaddress.IPv6Address(host).is_link_local:
        if interface is None:
            raise ReservationError("link-local address listed before any interface")
        host = f"{host}%{interface}"
    return HostAddress("v6", host)


def parse_ifconfig(text: str) -> tuple[HostAddress, ...]:
    """Every address in macOS `ifconfig` output, in order, without duplicates."""
    found: list[HostAddress] = []
    interface = None
    for line in text.splitlines():
        if match := _INTERFACE.match(line):
            interface = match.group(1)
        elif match := _INET.match(line):
            found.append(HostAddress("v4", match.group(1)))
        elif match := _INET6.match(line):
            found.append(_inet6_address(match.group(1), match.group(2), interface))
    if HostAddress("v4", RELAY_HOST) not in found:
        raise ReservationError("address enumeration is missing the loopback relay address")
    return tuple(dict.fromkeys(found))


def host_addresses() -> tuple[HostAddress, ...]:
    """Enumerate this host's assigned addresses from `ifconfig`."""
    try:
        result = subprocess.run(
            [IFCONFIG], capture_output=True, text=True, timeout=10, env=_IFCONFIG_ENV
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReservationError(f"cannot enumerate host addresses: {exc}") from exc
    if result.returncode != 0:
        raise ReservationError(f"ifconfig exited {result.returncode}: {result.stderr[-200:]}")
    return parse_ifconfig(result.stdout)


def _bind(family: str, host: str, port: int, *, reuseaddr: bool) -> socket.socket:
    af = socket.AF_INET if family == "v4" else socket.AF_INET6
    sock = socket.socket(af, socket.SOCK_STREAM)
    try:
        if reuseaddr:
            # BSD needs SO_REUSEADDR to bind a specific address beneath a held
            # wildcard; SO_REUSEPORT is never set, so P stays unshareable.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if host == "::":
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        flags = socket.AI_NUMERICHOST | socket.AI_PASSIVE
        sock.bind(socket.getaddrinfo(host, port, af, socket.SOCK_STREAM, 0, flags)[0][4])
    except BaseException:
        sock.close()
        raise
    return sock


def _address_set_sha256(addresses: tuple[HostAddress, ...]) -> str:
    canonical = "\n".join(sorted(f"{a.family} {a.host}" for a in addresses))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PortReservation:
    """Holds the relay port on the wildcard and every covered address.
    Construct through `reserve_relay_port`, which validates the hold."""

    def __init__(
        self, port: int, addresses: tuple[HostAddress, ...], held: dict[str, socket.socket]
    ):
        self.port = port
        self.addresses = addresses
        self._held = held

    def __enter__(self) -> PortReservation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sockets(self) -> tuple[socket.socket, ...]:
        return tuple(self._held.values())

    def relay_socket(self) -> socket.socket:
        """The only socket the relay may listen on."""
        return self._held[RELAY_HOST]

    def record(self) -> dict[str, Any]:
        """Retained evidence: counts and a digest, never the addresses."""
        counts = Counter(a.family for a in self.addresses)
        return {
            "schema_version": SCHEMA_VERSION,
            "port": self.port,
            "address_counts": {"v4": counts["v4"], "v6": counts["v6"]},
            "address_set_sha256": _address_set_sha256(self.addresses),
        }

    def close(self) -> None:
        for sock in self._held.values():
            sock.close()


def _hold_all(port: int, covered: tuple[HostAddress, ...]) -> dict[str, socket.socket]:
    held: dict[str, socket.socket] = {}
    # The message names the family and position, never the address (ADR-69 §4).
    target = "the wildcard"
    try:
        held["::"] = _bind("v6", "::", port, reuseaddr=False)
        port = held["::"].getsockname()[1]
        for index, address in enumerate(covered):
            target = f"{address.family} address {index}"
            held[address.host] = _bind(address.family, address.host, port, reuseaddr=True)
    except OSError as exc:
        _close_all(held)
        raise ReservationError(
            f"cannot reserve relay port {port} on {target}: {exc.strerror or exc}"
        ) from exc
    except BaseException:
        _close_all(held)
        raise
    return held


def _close_all(held: dict[str, socket.socket]) -> None:
    for sock in held.values():
        sock.close()


def reserve_relay_port(port: int = 0, addresses: AddressSource = host_addresses) -> PortReservation:
    """Bind `port` (0 = ephemeral) on `::` and every address, or refuse entirely.
    The address set is enumerated again after binding; any change refuses,
    because an address that appeared mid-reservation is not covered."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ReservationError(f"relay port out of range: {port!r}")
    covered = addresses()
    if HostAddress("v4", RELAY_HOST) not in covered:
        raise ReservationError("covered addresses do not include the loopback relay address")
    held = _hold_all(port, covered)
    reservation = PortReservation(held["::"].getsockname()[1], covered, held)
    try:
        if drifted(reservation, addresses):
            raise ReservationError("host addresses changed while the relay port was reserved")
    except BaseException:
        reservation.close()
        raise
    return reservation


def address_drift(
    reservation: PortReservation, addresses: AddressSource = host_addresses
) -> dict[str, int]:
    """Addresses added or removed since the reservation; nonzero is a stop condition."""
    before, now = set(reservation.addresses), set(addresses())
    return {"added": len(now - before), "removed": len(before - now)}


def drifted(reservation: PortReservation, addresses: AddressSource = host_addresses) -> bool:
    """True when the host's address set no longer matches the reservation."""
    return any(address_drift(reservation, addresses).values())
