"""Network capability cases for a `loopback` session policy (ADR-69; TRT-5/TRT-6).

Under `loopback` the profile admits TCP to one pinned port. The audit holds
that port exactly as a session will (relay_reservation: the wildcard plus every
host address) and proves, under the session's own profile:

- the relay is reachable and serves its canary (`loopback_tcp_read`, allow);
- another local port is refused (`foreign_loopback_tcp_read`);
- a never-routable external address is refused (`external_tcp_read`);
- UDP to the pinned port is refused, while the same send outside the sandbox
  reaches a listener (`udp_send` against `unrestricted_udp_baseline`);
- no unsandboxed process can bind the pinned port on any covered address or
  wildcard, under any reuse-flag combination, every attempt failing with
  EADDRINUSE (`reservation_intruder_bind`), while the same matrix binds a free
  port (`unrestricted_intruder_baseline`).

Denials are proven by the sandbox's immediate "Operation not permitted", never
by a timeout. For the UDP and intruder cases the controller records what
reached it as the case's observed output, so `sentinel_exposed` means the
payload got through.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

from aisle.harness.relay_reservation import ReservationError, reserve_relay_port
from aisle.harness.treatment_confinement import (
    ConfinementError,
    _case_result,
    _run,
    _serve_once,
    _wrapped,
)

#: TEST-NET-1 (RFC 5737): never routable, so a permitted connect would hang
#: while a sandbox denial fails at once with "Operation not permitted".
EXTERNAL_PROBE = "192.0.2.1"
#: denial probes match the sandbox's English error text
PROBE_ENV = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
RELAY_CANARY = b"LOOPBACK-SYNTHETIC-CANARY\n"
UDP_CANARY = b"UDP-SYNTHETIC-CANARY"
INTRUDER_MARK = b"INTRUDER-BOUND"
#: bash's report of a sandbox-refused connect; a failed sandbox_apply or execvp
#: says "Operation not permitted" too, but never after "connect: "
_DENIED = b"connect: Operation not permitted"

_INTRUDER = r"""
import errno, json, socket, sys
port, targets = int(sys.argv[1]), json.loads(sys.argv[2])
bound = other = 0
for family, host in targets + [["v4", "0.0.0.0"], ["v6", "::"]]:
    for reuseaddr in (0, 1):
        for reuseport in (0, 1):
            af = socket.AF_INET if family == "v4" else socket.AF_INET6
            s = socket.socket(af, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuseaddr)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, reuseport)
                flags = socket.AI_NUMERICHOST | socket.AI_PASSIVE
                s.bind(socket.getaddrinfo(host, port, af, socket.SOCK_STREAM, 0, flags)[0][4])
                bound += 1
            except OSError as exc:
                other += exc.errno != errno.EADDRINUSE
            finally:
                s.close()
if bound:
    sys.stdout.write("INTRUDER-BOUND %d" % bound)
    sys.exit(0)
sys.exit(3 if other == 0 else 4)
"""


def _tcp_read(port: int, host: str = "127.0.0.1") -> list[str]:
    return ["/bin/bash", "-c", f"exec 3<>/dev/tcp/{host}/{port} && /bin/cat <&3"]


def _udp_send(port: int) -> list[str]:
    payload = UDP_CANARY.decode()
    return ["/bin/bash", "-c", f"exec 3<>/dev/udp/127.0.0.1/{port} && printf {payload} >&3"]


def run_denial_probe(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    """A probe that must be refused outright: a hang (the sandbox let the
    connect through to an address that never answers) is a failed case, not
    an aborted audit."""
    try:
        return subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, env=PROBE_ENV, timeout=15
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, -1, b"", f"probe hung: {exc}".encode())
    except OSError as exc:
        raise ConfinementError(
            f"capability command failed to execute: {command[0]}: {exc}"
        ) from exc


def _observed(
    result: subprocess.CompletedProcess[bytes], received: bytes
) -> subprocess.CompletedProcess[bytes]:
    """The probe's exit status with what reached the controller as its output."""
    return subprocess.CompletedProcess(result.args, result.returncode, received, result.stderr)


def _udp_cases(profile_path: Path, cwd: Path, port: int) -> list[dict]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.settimeout(0.5)

        def received() -> bytes:
            try:
                return listener.recvfrom(256)[0]
            except TimeoutError:
                return b""

        baseline = _run(_udp_send(port), cwd=cwd)
        baseline_bytes = received()
        confined = run_denial_probe(_wrapped(profile_path, _udp_send(port)), cwd=cwd)
        confined_bytes = received()
    return [
        _case_result(
            "unrestricted_udp_baseline",
            _observed(baseline, baseline_bytes),
            UDP_CANARY,
            expected="baseline-exposure",
        ),
        _case_result(
            "udp_send",
            _observed(confined, confined_bytes),
            UDP_CANARY,
            expected="deny",
            extra_pass=_DENIED in confined.stderr,
        ),
    ]


def _intruder(port: int, targets: list, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return _run([sys.executable, "-I", "-c", _INTRUDER, str(port), json.dumps(targets)], cwd=cwd)


def _intruder_cases(reservation, cwd: Path) -> list[dict]:
    """The intruder matrix against a free port (it must bind: positive
    control) and against the reservation (every bind must fail with
    EADDRINUSE, exit 3; a crash or another errno is not a pass)."""
    targets = [[address.family, address.host] for address in reservation.addresses]
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
        probe.bind(("::", 0))
        free = probe.getsockname()[1]
    baseline = _intruder(free, targets, cwd)
    held = _intruder(reservation.port, targets, cwd)
    return [
        _case_result(
            "unrestricted_intruder_baseline", baseline, INTRUDER_MARK, expected="baseline-exposure"
        ),
        _case_result(
            "reservation_intruder_bind",
            held,
            INTRUDER_MARK,
            expected="deny",
            extra_pass=held.returncode == 3,
        ),
    ]


def loopback_capability_cases(profile_path: Path, cwd: Path, port: int) -> list[dict]:
    """Run the loopback network matrix while holding `port` as a session would."""
    try:
        reservation = reserve_relay_port(port=port)
    except ReservationError as exc:
        raise ConfinementError(f"cannot hold the pinned relay port for the audit: {exc}") from exc
    with reservation, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as foreign:
        relay = reservation.relay_socket()
        foreign.bind(("127.0.0.1", 0))
        stopped = threading.Event()
        servers = []
        for listener in (relay, foreign):
            listener.listen()
            listener.settimeout(0.1)
            server = threading.Thread(
                target=_serve_once, args=(listener, RELAY_CANARY, stopped), daemon=True
            )
            server.start()
            servers.append(server)
        try:
            baseline = _run(_tcp_read(port), cwd=cwd)
            confined = _run(_wrapped(profile_path, _tcp_read(port)), cwd=cwd)
            foreign_read = run_denial_probe(
                _wrapped(profile_path, _tcp_read(foreign.getsockname()[1])), cwd=cwd
            )
            external = run_denial_probe(
                _wrapped(profile_path, _tcp_read(9, host=EXTERNAL_PROBE)), cwd=cwd
            )
        finally:
            stopped.set()
            for server in servers:
                server.join(timeout=2)
        cases = [
            _case_result(
                "unrestricted_tcp_baseline", baseline, RELAY_CANARY, expected="baseline-exposure"
            ),
            _case_result(
                "loopback_tcp_read",
                confined,
                b"HIDDEN-NEVER-SERVED",
                expected="allow",
                extra_pass=confined.stdout == RELAY_CANARY,
            ),
            _case_result(
                "foreign_loopback_tcp_read",
                foreign_read,
                RELAY_CANARY,
                expected="deny",
                extra_pass=_DENIED in foreign_read.stderr,
            ),
            _case_result(
                "external_tcp_read",
                external,
                RELAY_CANARY,
                expected="deny",
                extra_pass=_DENIED in external.stderr,
            ),
        ]
        cases.extend(_intruder_cases(reservation, cwd))
        cases.extend(_udp_cases(profile_path, cwd, port))
    return cases
