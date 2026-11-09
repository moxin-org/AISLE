"""Unscored macOS confinement capability audit for SPEC 420.

The audit is controller-owned and runs synthetic sentinels outside the visible
tree.  It deliberately does not authorize confirmatory collection: Linux,
vendor-network access, credential handling, and Claude/Codex parity remain
separate treatment-integrity gates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "aisle.macos-confinement-capability.v4"
EVIDENCE_CLASS = "synthetic_unscored_capability"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SYSTEM_PROFILE = Path("/System/Library/Sandbox/Profiles/system.sb")
_HASH_LENGTH = 64
#: `deny-external` allows no network at all. `loopback` allows TCP to one
#: pinned port only, the arm's provider relay, which the controller holds on
#: every host address for the session (ADR-69, relay_reservation).
NETWORK_POLICIES = ("deny-external", "loopback")
_LOOPBACK_CASE_IDS = frozenset(
    {
        "loopback_tcp_read",
        "foreign_loopback_tcp_read",
        "external_tcp_read",
        "unrestricted_udp_baseline",
        "udp_send",
        "reservation_intruder_bind",
        "unrestricted_intruder_baseline",
    }
)
#: the preferred Unix-socket probe; an admitted Python interpreter is the fallback
_NETCAT = "/usr/bin/nc"
#: the executables the audit's own probes need (plus the Apple developer Git);
#: a probe-widened audit may add these, and only these, to a session policy
PROBE_EXECUTABLES = ("/bin/bash", "/bin/cat", _NETCAT)
#: the read roots those probes load from (plus the Apple developer tree for Git)
PROBE_READ_ROOTS = ("/bin", "/usr/bin", "/usr/lib", "/usr/share", "/System", "/Library/Apple")
_PYTHON_SOCKET_CLIENT = (
    "import socket,sys\n"
    "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)\n"
    "s.settimeout(2)\n"
    "s.connect(sys.argv[1])\n"
    "sys.stdout.buffer.write(s.recv(4096))\n"
)
_SENTINEL_DIR = ".aisle-capability"
_BASE_CASE_IDS = frozenset(
    {
        "absolute_hidden_read",
        "alternate_worktree_hidden_read",
        "declared_output_write",
        "git_object_hidden_read",
        "hidden_write",
        "parent_traversal_hidden_read",
        "subprocess_hidden_read",
        "subprocess_visible_read",
        "symlink_hidden_read",
        "unrestricted_alternate_worktree_baseline",
        "unrestricted_git_object_baseline",
        "unrestricted_hidden_baseline",
        "visible_git_object_read",
        "visible_read",
        "unrestricted_tcp_baseline",
        "unrestricted_unix_socket_baseline",
        "unix_socket_network_control",
        "unix_socket_read",
        "unrestricted_exec_baseline",
        "unlisted_executable",
    }
)


def _require_network_policy(network_policy: str) -> None:
    if network_policy not in NETWORK_POLICIES:
        raise ConfinementError(f"unknown network policy: {network_policy}")


def required_case_ids(network_policy: str) -> frozenset[str]:
    """The case matrix an attestation must carry for a compiled network policy."""
    _require_network_policy(network_policy)
    if network_policy == "loopback":
        return _BASE_CASE_IDS | _LOOPBACK_CASE_IDS
    return _BASE_CASE_IDS | {"tcp_read"}


def verify_relay_port(policy: Any, provider: Any) -> None:
    """A `loopback` policy and its provider binding must name the same relay
    port (TRT-5/MON-8): the sandbox admits exactly that port and the relay
    serves on it from the controller's reservation."""
    declared = provider.get("relay_port", 0) if type(provider) is dict else 0
    if type(policy) is not dict or policy.get("network_policy") != "loopback":
        # A pinned port is served only from a loopback session's reservation;
        # anywhere else it would fail late, after launch evidence was written.
        if declared != 0:
            raise ValueError("only a loopback confinement policy may pin a provider relay_port")
        return
    pinned = policy.get("loopback_port")
    declared = provider.get("relay_port") if type(provider) is dict else None
    if type(declared) is not int or isinstance(declared, bool) or declared != pinned:
        raise ValueError("provider relay_port must equal the confinement policy's loopback_port")


class ConfinementError(RuntimeError):
    """The external adapter or its retained capability evidence is unusable."""


@dataclass(frozen=True)
class MacOSPolicy:
    """Controller-owned roots and process policy compiled to macOS SBPL."""

    visible_roots: tuple[Path, ...]
    output_roots: tuple[Path, ...]
    runtime_read_roots: tuple[Path, ...]
    allowed_executables: tuple[Path, ...]
    hidden_roots: tuple[Path, ...]
    network_policy: str
    #: the one local TCP port a `loopback` policy may reach (its provider relay)
    loopback_port: int | None = None

    @property
    def readable_roots(self) -> tuple[Path, ...]:
        """Everything the confined process may read: visible, output and runtime roots."""
        return (*self.visible_roots, *self.output_roots, *self.runtime_read_roots)

    def as_dict(self) -> dict[str, Any]:
        """Return constructor-compatible fields for tests and policy transforms.
        The port appears only when pinned, so every existing consumer of the
        root/network fields sees exactly the shape it always did."""
        fields = {
            "visible_roots": self.visible_roots,
            "output_roots": self.output_roots,
            "runtime_read_roots": self.runtime_read_roots,
            "allowed_executables": self.allowed_executables,
            "hidden_roots": self.hidden_roots,
            "network_policy": self.network_policy,
        }
        if self.loopback_port is not None:
            fields["loopback_port"] = self.loopback_port
        return fields

    def canonical_dict(self) -> dict[str, Any]:
        """The identity form; the port appears only when a policy pins one, so a
        deny-external policy keeps the identity it always had."""
        canonical = {
            "allowed_executables": [str(path) for path in self.allowed_executables],
            "hidden_roots": [str(path) for path in self.hidden_roots],
            "network_policy": self.network_policy,
            "output_roots": [str(path) for path in self.output_roots],
            "runtime_read_roots": [str(path) for path in self.runtime_read_roots],
            "visible_roots": [str(path) for path in self.visible_roots],
        }
        if self.loopback_port is not None:
            canonical["loopback_port"] = self.loopback_port
        return canonical

    @classmethod
    def from_canonical(cls, declared: Any) -> MacOSPolicy:
        """Rebuild a policy from its canonical form; anything mis-shaped refuses."""
        fields = set(_CANONICAL_ROOT_FIELDS) | {"network_policy"}
        if (
            type(declared) is not dict
            or not fields <= set(declared)
            or not set(declared) <= fields | {"loopback_port"}
        ):
            raise ConfinementError("policy must carry exactly the MacOSPolicy fields")
        port = declared.get("loopback_port")
        if (
            type(declared["network_policy"]) is not str
            or (port is not None and (type(port) is not int or isinstance(port, bool)))
            or any(
                not isinstance(declared[k], list) or any(type(p) is not str for p in declared[k])
                for k in _CANONICAL_ROOT_FIELDS
            )
        ):
            raise ConfinementError("policy fields must be lists of paths and a network policy")
        return cls(
            **{key: tuple(Path(p) for p in declared[key]) for key in _CANONICAL_ROOT_FIELDS},
            network_policy=declared["network_policy"],
            loopback_port=port,
        )


_CANONICAL_ROOT_FIELDS = (
    "visible_roots",
    "output_roots",
    "runtime_read_roots",
    "allowed_executables",
    "hidden_roots",
)


@dataclass(frozen=True)
class CompiledProfile:
    text: str
    sha256: str
    policy_id: str
    network_policy: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ConfinementError(f"cannot hash required adapter input {path}: {exc}") from exc


def _validate_roots(name: str, roots: tuple[Path, ...]) -> None:
    if not roots:
        raise ConfinementError(f"{name} must not be empty")
    rendered = [str(path) for path in roots]
    if len(rendered) != len(set(rendered)):
        raise ConfinementError(f"{name} contains a duplicate")
    for path in roots:
        if not path.is_absolute():
            raise ConfinementError(f"{name} contains a relative path: {path}")
        if not str(path).isascii() or not str(path).isprintable():
            # SBPL literals are JSON-escaped; a non-ASCII name would compile
            # to a different string than the directory it must guard.
            raise ConfinementError(f"{name} contains a non-ASCII or unprintable path: {path}")
        if path == Path("/"):
            raise ConfinementError(f"{name} cannot authorize the filesystem root")
        if not path.exists():
            raise ConfinementError(f"{name} contains an unresolved path: {path}")
        if path.resolve() != path:
            raise ConfinementError(f"{name} contains a non-canonical path: {path}")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_policy(policy: MacOSPolicy) -> None:
    _require_network_policy(policy.network_policy)
    port = policy.loopback_port
    if policy.network_policy == "loopback":
        if type(port) is not int or isinstance(port, bool) or not 0 < port < 65536:
            raise ConfinementError("loopback network policy must pin one relay port")
    elif port is not None:
        raise ConfinementError("only a loopback network policy may carry a relay port")
    for name in (
        "visible_roots",
        "output_roots",
        "runtime_read_roots",
        "allowed_executables",
        "hidden_roots",
    ):
        _validate_roots(name, getattr(policy, name))
    for executable in policy.allowed_executables:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ConfinementError(f"allowed executable is not runnable: {executable}")

    readable = policy.readable_roots
    for hidden in policy.hidden_roots:
        for allowed in readable:
            if _contains(allowed, hidden) or _contains(hidden, allowed):
                raise ConfinementError(
                    f"hidden root overlaps an agent-readable root: {hidden} vs {allowed}"
                )


def _sbpl_path(path: Path) -> str:
    return json.dumps(str(path))


def compile_macos_profile(policy: MacOSPolicy) -> CompiledProfile:
    """Compile a default-deny profile without disclosing hidden-root names."""
    _validate_policy(policy)
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(allow process-fork)",
        "(allow process-exec",
    ]
    lines.extend(f"  (literal {_sbpl_path(path)})" for path in policy.allowed_executables)
    lines.extend([")", "(allow file-read*"])
    lines.extend(
        f"  (subpath {_sbpl_path(path)})"
        for path in (*policy.visible_roots, *policy.output_roots, *policy.runtime_read_roots)
    )
    lines.extend([")", "(allow file-read-metadata file-test-existence"])
    lines.extend(
        f"  (path-ancestors {_sbpl_path(path)})"
        for path in (*policy.visible_roots, *policy.output_roots, *policy.runtime_read_roots)
    )
    lines.extend([")", "(allow file-write*"])
    lines.extend(f"  (subpath {_sbpl_path(path)})" for path in policy.output_roots)
    lines.extend([")", "(allow signal (target self))"])
    if policy.network_policy == "loopback":
        # TCP only: `remote ip` would also admit UDP to any local listener
        # on the port, which the TCP reservation does not cover (ADR-69).
        lines.append(f'(allow network-outbound (remote tcp "localhost:{policy.loopback_port}"))')
    lines.append("")
    text = "\n".join(lines)
    return CompiledProfile(
        text=text,
        sha256=_sha256_bytes(text.encode()),
        policy_id=_sha256_bytes(_canonical_bytes(policy.canonical_dict())),
        network_policy=policy.network_policy,
    )


def _case_result(
    case_id: str,
    result: subprocess.CompletedProcess[bytes],
    sentinel: bytes,
    *,
    expected: str,
    extra_pass: bool = True,
) -> dict[str, Any]:
    exposed = sentinel in result.stdout or sentinel in result.stderr
    if expected == "allow":
        denied = False
        passed = result.returncode == 0 and not exposed and extra_pass
    elif expected == "deny":
        denied = result.returncode != 0 and not exposed
        passed = denied and extra_pass
    elif expected == "baseline-exposure":
        denied = False
        passed = result.returncode == 0 and exposed and extra_pass
    else:  # pragma: no cover - controller constant
        raise AssertionError(f"unknown expectation {expected}")
    return {
        "denied": denied,
        "expected": expected,
        "id": case_id,
        "passed": passed,
        "returncode": result.returncode,
        "sentinel_exposed": exposed,
        "stderr_sha256": _sha256_bytes(result.stderr),
        "stdout_sha256": _sha256_bytes(result.stdout),
    }


def _run(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            env=env,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfinementError(
            f"capability command failed to execute: {command[0]}: {exc}"
        ) from exc


def _wrapped(profile_path: Path, command: list[str]) -> list[str]:
    return [str(SANDBOX_EXEC), "-f", str(profile_path), *command]


def _controller_command(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run synthetic-fixture setup outside the subject sandbox and fail loudly."""
    result = _run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise ConfinementError(
            "controller fixture command failed: "
            f"{command[0]} rc={result.returncode} "
            f"stdout_sha256={_sha256_bytes(result.stdout)} "
            f"stderr_sha256={_sha256_bytes(result.stderr)}"
        )
    return result


def _apple_git_runtime(*, cwd: Path) -> tuple[Path, Path]:
    """Resolve the selected Apple developer tree and its real Git executable."""
    developer_result = _controller_command(["/usr/bin/xcode-select", "-p"], cwd=cwd)
    developer_root = Path(developer_result.stdout.decode().strip()).resolve()
    git_result = _controller_command(["/usr/bin/xcrun", "--find", "git"], cwd=cwd)
    git = Path(git_result.stdout.decode().strip()).resolve()
    if not developer_root.is_dir() or not git.is_file() or not _contains(developer_root, git):
        raise ConfinementError("Apple developer Git runtime is unresolved or inconsistent")
    return git, developer_root


def _initialize_git_fixture(
    repository: Path,
    filename: str,
    content: bytes,
    *,
    git: Path,
    env: dict[str, str],
) -> str:
    """Create one isolated synthetic commit and return the committed blob id."""
    repository.mkdir()
    (repository / filename).write_bytes(content)
    git_command = str(git)
    _controller_command([git_command, "init", "--quiet"], cwd=repository, env=env)
    _controller_command([git_command, "add", "--", filename], cwd=repository, env=env)
    _controller_command(
        [
            git_command,
            "-c",
            "user.name=AISLE synthetic controller",
            "-c",
            "user.email=synthetic-controller@invalid",
            "commit",
            "--quiet",
            "-m",
            "synthetic confinement fixture",
        ],
        cwd=repository,
        env=env,
    )
    result = _controller_command(
        [git_command, "rev-parse", f"HEAD:{filename}"], cwd=repository, env=env
    )
    object_id = result.stdout.decode("ascii").strip()
    if len(object_id) not in (40, 64) or any(char not in "0123456789abcdef" for char in object_id):
        raise ConfinementError("controller fixture returned an invalid git object id")
    return object_id


def _platform_record() -> dict[str, str]:
    mac_version, _, machine = platform.mac_ver()
    try:
        build = subprocess.run(
            ["/usr/bin/sw_vers", "-buildVersion"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        build = "unresolved"
    return {
        "build": build,
        "machine": machine or platform.machine(),
        "macos_version": mac_version,
        "python": platform.python_version(),
    }


def _serve_once(listener: socket.socket, payload: bytes, stopped: threading.Event) -> None:
    while not stopped.is_set():
        try:
            connection, _ = listener.accept()
        except TimeoutError:
            continue
        with connection:
            connection.settimeout(1)
            connection.sendall(payload)


def _socket_capability_cases(profile_path: Path, cwd: Path, sentinel: bytes) -> list[dict]:
    """Exercise the same loopback read outside and inside the external profile."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.1)
        stopped = threading.Event()

        def serve():
            while not stopped.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(1)
                    connection.sendall(sentinel)

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        command = [
            "/bin/bash",
            "-c",
            f"exec 3<>/dev/tcp/127.0.0.1/{listener.getsockname()[1]} && /bin/cat <&3",
        ]
        try:
            baseline = _run(command, cwd=cwd)
            confined = _run(_wrapped(profile_path, command), cwd=cwd)
        finally:
            stopped.set()
            server.join(timeout=2)
        return [
            _case_result(
                "unrestricted_tcp_baseline", baseline, sentinel, expected="baseline-exposure"
            ),
            _case_result("tcp_read", confined, sentinel, expected="deny"),
        ]


def _unix_socket_capability_cases(
    profile_path: Path, cwd: Path, sentinel: bytes, client: list[str]
) -> list[dict]:
    """Distinguish local socket denial from executable or filesystem failure.

    `client` is the admitted argv prefix that connects to a Unix socket path
    and prints what it reads (nc, or an admitted Python interpreter)."""
    # A session root may exceed macOS sockaddr_un.sun_path; the endpoint lives
    # in a short controller-private directory outside every agent-readable root,
    # which changes nothing about the deny (network) or control (allow) cases.
    control_path = profile_path.with_name("unix-network-control.sb")
    control_text = profile_path.read_text() + (
        "\n(allow network* (local unix-socket) (remote unix-socket))\n"
    )
    control_path.write_text(control_text)
    with (
        tempfile.TemporaryDirectory(prefix="aisle-sock-", dir="/private/tmp") as socket_root,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
    ):
        endpoint = Path(socket_root) / "fixture.sock"
        listener.bind(str(endpoint))
        listener.listen()
        listener.settimeout(0.1)
        stopped = threading.Event()
        server = threading.Thread(
            target=_serve_once, args=(listener, sentinel, stopped), daemon=True
        )
        server.start()
        command = [*client, str(endpoint)]
        try:
            baseline = _run(command, cwd=cwd)
            confined = _run(_wrapped(profile_path, command), cwd=cwd)
            control = _run(_wrapped(control_path, command), cwd=cwd)
        finally:
            stopped.set()
            server.join(timeout=2)
        control_case = _case_result(
            "unix_socket_network_control", control, sentinel, expected="baseline-exposure"
        )
        control_case["compiled_profile_sha256"] = _sha256_bytes(control_text.encode())
        return [
            _case_result(
                "unrestricted_unix_socket_baseline",
                baseline,
                sentinel,
                expected="baseline-exposure",
            ),
            _case_result("unix_socket_read", confined, sentinel, expected="deny"),
            control_case,
        ]


def _synthetic_policy(
    visible: Path, output: Path, hidden: Path, git: Path, developer_root: Path
) -> MacOSPolicy:
    """The fixed policy of the synthetic (session-less) audit."""
    return MacOSPolicy(
        visible_roots=(visible,),
        output_roots=(output,),
        runtime_read_roots=(
            Path("/Library/Apple").resolve(),
            Path("/System").resolve(),
            Path("/bin").resolve(),
            Path("/usr/bin").resolve(),
            Path("/usr/lib").resolve(),
            Path("/usr/share").resolve(),
            developer_root,
        ),
        allowed_executables=(
            Path("/bin/bash").resolve(),
            Path("/bin/cat").resolve(),
            Path("/usr/bin/nc").resolve(),
            git,
        ),
        hidden_roots=(hidden,),
        network_policy="deny-external",
    )


def _socket_client(policy: MacOSPolicy) -> list[str]:
    """The Unix-socket probe the policy already admits: `nc` when present,
    otherwise an admitted Python interpreter. The audit never widens the arm's
    executable grants for its own probes."""
    netcat = Path(_NETCAT).resolve()
    if netcat in policy.allowed_executables:
        return [str(netcat), "-w", "2", "-U"]
    for path in policy.allowed_executables:
        if path.name.startswith("python"):
            return [str(path), "-I", "-c", _PYTHON_SOCKET_CLIENT]
    raise ConfinementError(
        "session policy must admit a Unix-socket probe (/usr/bin/nc or a python interpreter)"
    )


def widen_for_probes(
    policy: MacOSPolicy, *, git: Path, developer_root: Path
) -> tuple[MacOSPolicy, dict[str, list[str]]]:
    """The session policy plus exactly what the audit's probes need: the shell,
    cat, netcat and the Apple Git as executables, and the read roots they load
    from. The visible, output and hidden roots and the network policy remain
    the session's.
    Returns the widened policy and the additions, which the attestation records
    so the launch wrapper can recompute the audited profile from the session
    policy alone."""
    admitted = set(policy.allowed_executables)
    wanted = [Path(p).resolve() for p in PROBE_EXECUTABLES] + [git]
    executables = sorted({p for p in wanted if p not in admitted}, key=str)
    readable = policy.readable_roots
    roots = []
    for candidate in (*(Path(p).resolve() for p in PROBE_READ_ROOTS), developer_root):
        if not candidate.is_dir():
            continue
        if any(_contains(root, candidate) for root in readable) or candidate in roots:
            continue
        if any(
            _contains(candidate, hidden) or _contains(hidden, candidate)
            for hidden in policy.hidden_roots
        ):
            raise ConfinementError(f"probe read root {candidate} would overlap a hidden root")
        roots.append(candidate)
    roots.sort(key=str)
    widened = replace(
        policy,
        allowed_executables=(*policy.allowed_executables, *executables),
        runtime_read_roots=(*policy.runtime_read_roots, *roots),
    )
    return widened, {
        "allowed_executables": [str(p) for p in executables],
        "runtime_read_roots": [str(p) for p in roots],
    }


def _apply_declared_widening(
    policy: MacOSPolicy, widening: Any, *, git: Path, developer_root: Path
) -> MacOSPolicy:
    """Rebuild the audited policy the only way an honest audit could have: by
    widening the session policy with THIS host's own developer Git and probe
    roots, then requiring the attestation's declaration to equal that result.
    A declaration that names any other executable or root cannot match."""
    if (
        type(widening) is not dict
        or set(widening) != {"allowed_executables", "runtime_read_roots"}
        or any(
            not isinstance(widening[k], list) or any(type(p) is not str for p in widening[k])
            for k in widening
        )
    ):
        raise ConfinementError("probe widening declaration is mis-shaped")
    widened, expected = widen_for_probes(policy, git=git, developer_root=developer_root)
    if expected != widening:
        raise ConfinementError("probe widening differs from what this host's audit would add")
    return widened


def _check_session_policy(policy: MacOSPolicy, git: Path) -> list[str]:
    """A session policy can only be attested if the audit's own probes are
    admitted executables and its unlisted-executable control is not; returns
    the admitted Unix-socket probe argv."""
    _validate_policy(policy)
    probes = {Path(p).resolve() for p in ("/bin/bash", "/bin/cat")} | {git}
    missing = sorted(str(p) for p in probes - set(policy.allowed_executables))
    if missing:
        raise ConfinementError("session policy must admit the audit probes: " + ", ".join(missing))
    control = Path("/usr/bin/printf").resolve()
    if control in policy.allowed_executables:
        raise ConfinementError(f"session policy admits the unlisted-executable control {control}")
    return _socket_client(policy)


def _last_directory(name: str, roots: tuple[Path, ...]) -> Path:
    for root in reversed(roots):
        if root.is_dir() and not root.is_symlink():
            return root
    raise ConfinementError(f"{name} holds no directory for a sentinel")


def _sentinel_homes(
    policy: MacOSPolicy, requested: tuple[Path, Path, Path] | None
) -> tuple[Path, Path, Path]:
    """Where the three sentinels go: the caller's directories (each inside a
    root of the matching kind) or, by default, the last directory of each root
    list. A live session's caller names controller-private homes so the audit
    never writes into another arm's readable view."""
    kinds = (
        ("visible_roots", policy.visible_roots),
        ("output_roots", policy.output_roots),
        ("hidden_roots", policy.hidden_roots),
    )
    if requested is None:
        return tuple(_last_directory(name, roots) for name, roots in kinds)
    homes = []
    for (name, roots), home in zip(kinds, requested, strict=True):
        home = Path(home)
        if (
            not home.is_absolute()
            or home.resolve() != home
            or not home.is_dir()
            or not any(_contains(root, home) for root in roots if root.is_dir())
        ):
            raise ConfinementError(f"sentinel home must be a canonical directory inside {name}")
        homes.append(home)
    return tuple(homes)


def _place_sentinels(
    policy: MacOSPolicy,
    cleanup: contextlib.ExitStack,
    homes: tuple[Path, Path, Path] | None = None,
) -> tuple[Path, ...]:
    """Fresh sentinel directories inside the session's own visible, output and
    hidden roots. Each directory is registered for removal the moment it
    exists, so a later failure never leaves one behind; anything already at a
    sentinel path (file, directory or symlink) refuses."""
    homes = _sentinel_homes(policy, homes)
    sentinels = []
    for home in homes:
        path = home / _SENTINEL_DIR
        # Concurrent audits that share a home are serialized by their caller
        # (attest_many); across processes the owner marker refuses a live one.
        if path.is_dir() and not path.is_symlink() and _reclaimable(path):
            _remove_sentinel(path)
        try:
            path.mkdir()
        except FileExistsError as exc:
            raise ConfinementError(f"sentinel path already exists: {path}") from exc
        except OSError as exc:
            raise ConfinementError(f"sentinel path could not be created: {path}: {exc}") from exc
        cleanup.callback(_remove_sentinel, path)
        sentinels.append(path)
        _owner_marker(path).write_text(
            json.dumps({"schema_version": "aisle.sentinel-owner.v1", "pid": os.getpid()})
        )
    return tuple(sentinels)


def _redacted_policy(policy: MacOSPolicy) -> dict[str, Any]:
    """The attested policy with hidden roots replaced by their digests: the
    report is retained beside session evidence and must not name the private
    roots it protects. `policy_id` still binds the full canonical form."""
    canonical = policy.canonical_dict()
    canonical["hidden_roots"] = [
        "sha256:" + _sha256_bytes(path.encode()) for path in canonical["hidden_roots"]
    ]
    canonical["hidden_roots_redacted"] = True
    return canonical


def _owner_marker(path: Path) -> Path:
    return path / ".owner.json"


def _reclaimable(path: Path) -> bool:
    """A leftover sentinel from an audit process that no longer exists (a
    SIGKILL mid-run) may be reclaimed; anything else refuses."""
    marker_path = _owner_marker(path)
    if marker_path.is_symlink() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_bytes())
        pid = marker["pid"]
    except (OSError, ValueError, KeyError, TypeError):
        return False  # a corrupt or unreadable marker never authorizes removal
    if (
        type(pid) is not int
        or isinstance(pid, bool)
        or marker.get("schema_version") != "aisle.sentinel-owner.v1"
    ):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True  # the owner is gone
    except OSError:
        return False  # alive but not ours (EPERM) or undecidable
    return False


def _remove_sentinel(path: Path) -> None:
    failures: list[str] = []

    def record(_function, failed_path, exc_info):
        failures.append(f"{failed_path}: {exc_info[1]}")

    # onerror also works on the project's supported Python 3.11 runtime.
    shutil.rmtree(path, onerror=record)
    if path.exists() or path.is_symlink():
        detail = f" ({failures[0]})" if failures else ""
        raise ConfinementError(f"sentinel path could not be removed: {path}{detail}")


def _fixture_roots(
    policy: MacOSPolicy | None,
    sentinel_homes: tuple[Path, Path, Path] | None,
    root: Path,
    git: Path,
    developer_root: Path,
    cleanup: contextlib.ExitStack,
) -> tuple[MacOSPolicy, str, list[str], Path, Path, Path]:
    """The audited policy and the three fixture directories: a synthetic policy
    over fresh temporary roots, or the session policy with sentinels placed in
    its own roots and registered for removal."""
    if policy is None:
        visible, output, hidden = root / "visible", root / "output", root / "hidden"
        for path in (visible, output, hidden):
            path.mkdir()
        policy = _synthetic_policy(visible, output, hidden, git, developer_root)
        return policy, "synthetic", _socket_client(policy), visible, output, hidden
    client = _check_session_policy(policy, git)
    visible, output, hidden = _place_sentinels(policy, cleanup, sentinel_homes)
    return policy, "session", client, visible, output, hidden


def _place_hidden_link(
    visible: Path, hidden: Path, root: Path, source: str, hidden_sentinel: bytes
) -> None:
    """The symlink case proves following a link out of the visible root is
    refused. Under a session policy the link targets a hidden directory of the
    controller's private temporary tree, never the session's own hidden root: a
    link is readable text, and its target would otherwise disclose that root's
    path to the arm (the arm's real hidden root is still covered by the
    absolute and parent-traversal cases)."""
    link_target = hidden if source == "synthetic" else root / "hidden-target"
    if source == "session":
        link_target.mkdir()
        (link_target / "secret.txt").write_bytes(hidden_sentinel)
    (visible / "hidden-link").symlink_to(link_target, target_is_directory=True)


def run_macos_capability_audit(
    policy: MacOSPolicy | None = None,
    *,
    sentinel_homes: tuple[Path, Path, Path] | None = None,
    widen_probes: bool = False,
) -> dict[str, Any]:
    """Run the synthetic deny/allow matrix through the external adapter.

    Without a policy the audit runs its fixed synthetic policy in a private
    temporary tree. With a session policy it runs the same matrix under that
    exact policy, placing its sentinels inside the session's own roots (the
    caller's `sentinel_homes`, or the last directory of each root list) and
    removing them afterwards, so the retained attestation carries the profile
    hash and policy id the launch wrapper will demand (TRT-5/TRT-7). A policy
    that admits only an interpreter (a worker or validator) cannot run the
    probes; with `widen_probes` the matrix runs under that policy plus exactly
    the audit's own probes, the attestation still binds the SESSION profile,
    and it records the widening so the wrapper can recompute the audited
    profile from the session policy alone.
    """
    if sys.platform != "darwin":
        raise ConfinementError(
            "macOS confinement capability requires macOS; no simulation accepted"
        )
    if not SANDBOX_EXEC.is_file() or not os.access(SANDBOX_EXEC, os.X_OK):
        raise ConfinementError(f"required adapter is unavailable: {SANDBOX_EXEC}")
    if not SYSTEM_PROFILE.is_file():
        raise ConfinementError(f"required imported profile is unavailable: {SYSTEM_PROFILE}")

    # Cheap shape checks first: no subprocess is spawned for a malformed call.
    if policy is None and (sentinel_homes is not None or widen_probes):
        raise ConfinementError("sentinel homes and probe widening require a session policy")
    if policy is not None:
        _validate_policy(policy)
    session = policy

    # Keep the Unix socket pathname below macOS sockaddr_un.sun_path limits.
    with (
        tempfile.TemporaryDirectory(
            prefix="aisle-confinement-capability-", dir="/private/tmp"
        ) as temporary,
        contextlib.ExitStack() as cleanup,
    ):
        root = Path(temporary).resolve()
        git, developer_root = _apple_git_runtime(cwd=root)
        widening = None
        if session is not None and widen_probes:
            policy, widening = widen_for_probes(session, git=git, developer_root=developer_root)
        policy, source, client, visible, output, hidden = _fixture_roots(
            policy, sentinel_homes, root, git, developer_root, cleanup
        )
        if session is None:
            session = policy
        visible_sentinel = b"VISIBLE-SYNTHETIC-CAPABILITY\n"
        hidden_sentinel = b"HIDDEN-SYNTHETIC-CAPABILITY-7E4E\n"
        visible_file = visible / "allowed.txt"
        hidden_file = hidden / "secret.txt"
        visible_file.write_bytes(visible_sentinel)
        hidden_file.write_bytes(hidden_sentinel)
        _place_hidden_link(visible, hidden, root, source, hidden_sentinel)

        isolated_home = visible / "isolated-home"
        isolated_home.mkdir()
        git_environment = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(isolated_home),
            "PATH": str(git.parent),
        }
        visible_repository = visible / "repository"
        _initialize_git_fixture(
            visible_repository,
            "allowed.txt",
            visible_sentinel,
            git=git,
            env=git_environment,
        )
        hidden_repository = hidden / "evaluator-repository"
        hidden_object_id = _initialize_git_fixture(
            hidden_repository,
            "secret.txt",
            hidden_sentinel,
            git=git,
            env=git_environment,
        )
        hidden_worktree = hidden / "evaluator-alternate-worktree"
        _controller_command(
            [
                str(git),
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(hidden_worktree),
                "HEAD",
            ],
            cwd=hidden_repository,
            env=git_environment,
        )

        # The matrix runs under the audited (possibly widened) profile; the
        # attestation binds the SESSION profile the launch wrapper will compile.
        audited = compile_macos_profile(policy)
        compiled = audited if widening is None else compile_macos_profile(session)
        profile_path = root / "controller-profile.sb"
        profile_path.write_text(audited.text, encoding="utf-8")

        cases: list[dict[str, Any]] = []
        baseline = _run(["/bin/cat", str(hidden_file)], cwd=visible)
        cases.append(
            _case_result(
                "unrestricted_hidden_baseline",
                baseline,
                hidden_sentinel,
                expected="baseline-exposure",
            )
        )
        unrestricted_worktree = _run(
            [str(git), "-C", str(hidden_worktree), "show", "HEAD:secret.txt"],
            cwd=visible,
            env=git_environment,
        )
        cases.append(
            _case_result(
                "unrestricted_alternate_worktree_baseline",
                unrestricted_worktree,
                hidden_sentinel,
                expected="baseline-exposure",
            )
        )
        unrestricted_object = _run(
            [
                str(git),
                "--git-dir",
                str(hidden_repository / ".git"),
                "cat-file",
                "blob",
                hidden_object_id,
            ],
            cwd=visible,
            env=git_environment,
        )
        cases.append(
            _case_result(
                "unrestricted_git_object_baseline",
                unrestricted_object,
                hidden_sentinel,
                expected="baseline-exposure",
            )
        )
        visible_read = _run(_wrapped(profile_path, ["/bin/cat", str(visible_file)]), cwd=visible)
        cases.append(
            _case_result(
                "visible_read",
                visible_read,
                hidden_sentinel,
                expected="allow",
                extra_pass=visible_read.stdout == visible_sentinel,
            )
        )
        visible_shell = _run(
            _wrapped(
                profile_path, ["/bin/bash", "-c", f"/bin/cat {shlex.quote(str(visible_file))}"]
            ),
            cwd=visible,
        )
        cases.append(
            _case_result(
                "subprocess_visible_read",
                visible_shell,
                hidden_sentinel,
                expected="allow",
                extra_pass=visible_shell.stdout == visible_sentinel,
            )
        )
        visible_git_object = _run(
            _wrapped(
                profile_path,
                [
                    str(git),
                    "-C",
                    str(visible_repository),
                    "show",
                    "HEAD:allowed.txt",
                ],
            ),
            cwd=visible,
            env=git_environment,
        )
        cases.append(
            _case_result(
                "visible_git_object_read",
                visible_git_object,
                hidden_sentinel,
                expected="allow",
                extra_pass=visible_git_object.stdout == visible_sentinel,
            )
        )

        denied_reads = {
            "absolute_hidden_read": str(hidden_file),
            "parent_traversal_hidden_read": os.path.relpath(hidden_file, visible),
            "symlink_hidden_read": str(visible / "hidden-link" / "secret.txt"),
        }
        for case_id, path in denied_reads.items():
            result = _run(_wrapped(profile_path, ["/bin/cat", path]), cwd=visible)
            cases.append(_case_result(case_id, result, hidden_sentinel, expected="deny"))

        shell_hidden = _run(
            _wrapped(
                profile_path, ["/bin/bash", "-c", f"/bin/cat {shlex.quote(str(hidden_file))}"]
            ),
            cwd=visible,
        )
        cases.append(
            _case_result("subprocess_hidden_read", shell_hidden, hidden_sentinel, expected="deny")
        )

        alternate_worktree_hidden = _run(
            _wrapped(
                profile_path,
                [
                    str(git),
                    "-C",
                    str(hidden_worktree),
                    "show",
                    "HEAD:secret.txt",
                ],
            ),
            cwd=visible,
            env=git_environment,
        )
        cases.append(
            _case_result(
                "alternate_worktree_hidden_read",
                alternate_worktree_hidden,
                hidden_sentinel,
                expected="deny",
            )
        )
        git_object_hidden = _run(
            _wrapped(
                profile_path,
                [
                    str(git),
                    "--git-dir",
                    str(hidden_repository / ".git"),
                    "cat-file",
                    "blob",
                    hidden_object_id,
                ],
            ),
            cwd=visible,
            env=git_environment,
        )
        cases.append(
            _case_result(
                "git_object_hidden_read",
                git_object_hidden,
                hidden_sentinel,
                expected="deny",
            )
        )

        output_file = output / "declared.txt"
        output_write = _run(
            _wrapped(
                profile_path,
                ["/bin/bash", "-c", f"printf declared-output > {shlex.quote(str(output_file))}"],
            ),
            cwd=visible,
        )
        cases.append(
            _case_result(
                "declared_output_write",
                output_write,
                hidden_sentinel,
                expected="allow",
                extra_pass=output_file.read_text(encoding="utf-8") == "declared-output"
                if output_file.exists()
                else False,
            )
        )
        forbidden_write = _run(
            _wrapped(
                profile_path,
                [
                    "/bin/bash",
                    "-c",
                    f"printf forbidden-output > {shlex.quote(str(hidden / 'forbidden.txt'))}",
                ],
            ),
            cwd=visible,
        )
        cases.append(
            _case_result("hidden_write", forbidden_write, hidden_sentinel, expected="deny")
        )

        if policy.network_policy == "loopback":
            from aisle.harness.loopback_capability import loopback_capability_cases

            cases.extend(loopback_capability_cases(profile_path, visible, policy.loopback_port))
        else:
            cases.extend(_socket_capability_cases(profile_path, visible, hidden_sentinel))
        cases.extend(_unix_socket_capability_cases(profile_path, visible, hidden_sentinel, client))
        executable_command = ["/usr/bin/printf", "%s", hidden_sentinel.decode()]
        cases.append(
            _case_result(
                "unrestricted_exec_baseline",
                _run(executable_command, cwd=visible),
                hidden_sentinel,
                expected="baseline-exposure",
            )
        )
        cases.append(
            _case_result(
                "unlisted_executable",
                _run(_wrapped(profile_path, executable_command), cwd=visible),
                hidden_sentinel,
                expected="deny",
            )
        )

        if not visible_file.is_file() or not hidden_file.is_file():
            # a file-deny case would pass vacuously on a missing target
            raise ConfinementError("sentinel files vanished during the audit")
        denial_cases = [row for row in cases if row["expected"] == "deny"]
        allow_cases = [row for row in cases if row["expected"] == "allow"]
        baseline_cases = [row for row in cases if row["expected"] == "baseline-exposure"]
        denial_passes = sum(bool(row["passed"]) for row in denial_cases)
        false_alarms = sum(not bool(row["passed"]) for row in allow_cases)
        capability_pass = all(bool(row["passed"]) for row in cases)
        recorded_at = datetime.now(UTC)
        return {
            "adapter": {
                **(
                    {"audit_policy_id": audited.policy_id, "audit_profile_sha256": audited.sha256}
                    if widening is not None
                    else {}
                ),
                "compiled_profile_sha256": compiled.sha256,
                "imported_system_profile": str(SYSTEM_PROFILE),
                "imported_system_profile_sha256": _sha256_file(SYSTEM_PROFILE),
                "path": str(SANDBOX_EXEC),
                "policy_id": compiled.policy_id,
                "sha256": _sha256_file(SANDBOX_EXEC),
            },
            **({"probe_widening": widening} if widening is not None else {}),
            "capability_pass": capability_pass,
            "cases": cases,
            "confirmatory_ready": False,
            "evidence_class": EVIDENCE_CLASS,
            "limitations": [
                "macOS-only capability; no Linux adapter evaluated",
                "synthetic filesystem, TCP and Unix socket sentinels; "
                "no benchmark fault identities",
                "TCP/Unix socket reads and one unlisted executable tested; "
                "not exhaustive IPC/process coverage",
                "no vendor network or credential path evaluated",
                "no Claude/Codex end-to-end parity evaluated",
                "Git surfaces cover the system Git CLI only, not every future allowed tool",
                "Apple system.sb is a private interface and is hashed per audit",
            ],
            "platform": _platform_record(),
            "policy": _redacted_policy(session),
            "policy_source": source,
            "ephemeral_profile_path": str(profile_path),
            "recorded_at": recorded_at.isoformat(),
            "schema_version": SCHEMA_VERSION,
            "session_id": (
                f"macos-capability-{recorded_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
                f"{compiled.sha256[:12]}"
            ),
            "summary": {
                "baseline_tests": len(baseline_cases),
                "capability_pass": capability_pass,
                "declared_allow_tests": len(allow_cases),
                "denial_detection_rate": denial_passes / len(denial_cases),
                "denial_tests": len(denial_cases),
                "false_alarm_rate": false_alarms / len(allow_cases),
            },
        }


def write_macos_capability_audit(
    output: Path,
    policy: MacOSPolicy | None = None,
    *,
    sentinel_homes: tuple[Path, Path, Path] | None = None,
    widen_probes: bool = False,
) -> dict[str, Any]:
    """Retain one unscored audit without overwriting earlier evidence."""
    output = Path(output)
    if output.exists():
        raise ConfinementError(f"capability audit already exists: {output}")
    report = run_macos_capability_audit(
        policy, sentinel_homes=sentinel_homes, widen_probes=widen_probes
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    except FileExistsError as exc:
        raise ConfinementError(f"capability audit already exists: {output}") from exc
    return report


_HOST_DEVELOPER_GIT: tuple[Path, Path] | None = None


def _host_developer_git() -> tuple[Path, Path]:
    """This host's Apple developer Git and tree, resolved once per process: the
    verifier recomputes a widening from it on every launch."""
    global _HOST_DEVELOPER_GIT
    if _HOST_DEVELOPER_GIT is None:
        with tempfile.TemporaryDirectory(prefix="aisle-developer-git-") as temporary:
            _HOST_DEVELOPER_GIT = _apple_git_runtime(cwd=Path(temporary))
    return _HOST_DEVELOPER_GIT


def wrap_verified_command(
    command: list[str],
    compiled: CompiledProfile,
    profile_path: Path,
    attestation: dict[str, Any],
    *,
    purpose: str = "unscored_capability",
    policy: MacOSPolicy | None = None,
) -> list[str]:
    """Bind an unscored command to the exact externally verified adapter.

    An attestation that declares a `probe_widening` was audited under the
    session policy plus the audit's own probes; it is accepted only when the
    caller supplies that session `policy` and the wrapper recomputes the exact
    audited profile from the policy and the declared additions (TRT-5)."""
    _require_network_policy(compiled.network_policy)
    if purpose != "unscored_capability":
        raise ConfinementError(
            "this macOS capability attestation cannot authorize confirmatory work"
        )
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ConfinementError("command must be a non-empty argv list")
    if attestation.get("schema_version") != SCHEMA_VERSION:
        raise ConfinementError("adapter attestation schema is unsupported")
    if attestation.get("evidence_class") != EVIDENCE_CLASS:
        raise ConfinementError("adapter attestation evidence class is unsupported")
    if attestation.get("confirmatory_ready") is not False:
        raise ConfinementError("capability attestation has an ambiguous confirmatory marker")
    if not attestation.get("capability_pass"):
        raise ConfinementError("adapter capability did not pass")
    adapter = attestation.get("adapter")
    if not isinstance(adapter, dict):
        raise ConfinementError("adapter attestation is absent")
    if adapter.get("compiled_profile_sha256") != compiled.sha256:
        raise ConfinementError("adapter profile hash does not match compiled profile")
    if adapter.get("policy_id") != compiled.policy_id:
        raise ConfinementError("adapter policy id does not match compiled policy")
    widening = attestation.get("probe_widening")
    if widening is not None:
        if policy is None:
            raise ConfinementError("probe widening: the session policy is required to verify it")
        if compile_macos_profile(policy).sha256 != compiled.sha256:
            raise ConfinementError(
                "probe widening: session policy does not compile to this profile"
            )
        git, developer_root = _host_developer_git()
        audited = compile_macos_profile(
            _apply_declared_widening(policy, widening, git=git, developer_root=developer_root)
        )
        if (
            adapter.get("audit_profile_sha256") != audited.sha256
            or adapter.get("audit_policy_id") != audited.policy_id
        ):
            raise ConfinementError("probe widening: audited profile differs from the declaration")
    attested_policy = attestation.get("policy")
    if (
        not isinstance(attested_policy, dict)
        or attested_policy.get("network_policy") != compiled.network_policy
    ):
        raise ConfinementError("adapter attestation network policy does not match compiled profile")
    cases = attestation.get("cases")
    if (
        not isinstance(cases, list)
        or any(not isinstance(row, dict) for row in cases)
        or {row.get("id") for row in cases} != required_case_ids(compiled.network_policy)
        or any(not row.get("passed") for row in cases)
    ):
        raise ConfinementError("adapter attestation contains a failed case")

    profile_path = Path(profile_path)
    if _sha256_file(profile_path) != compiled.sha256:
        raise ConfinementError("retained profile hash does not match compiled profile")
    adapter_path = Path(str(adapter.get("path", "")))
    if not adapter_path.is_file() or not os.access(adapter_path, os.X_OK):
        raise ConfinementError("attested adapter executable is unavailable")
    if adapter.get("sha256") != _sha256_file(adapter_path):
        raise ConfinementError("attested adapter binary hash has drifted")
    system_profile = Path(str(adapter.get("imported_system_profile", "")))
    if adapter.get("imported_system_profile_sha256") != _sha256_file(system_profile):
        raise ConfinementError("attested imported system profile hash has drifted")
    return [str(adapter_path), "-f", str(profile_path), *command]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unscored SPEC 420 confinement audit")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-macos")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="session policy (MacOSPolicy.canonical_dict JSON) to attest under",
    )
    for role in ("visible", "output", "hidden"):
        audit.add_argument(
            f"--sentinel-{role}",
            type=Path,
            default=None,
            help=f"directory inside the policy's {role} roots that receives that sentinel",
        )
    audit.add_argument("--widen-probes", action="store_true", help=_WIDEN_HELP)
    many = commands.add_parser("attest-many")
    many.add_argument(
        "--policies",
        type=Path,
        required=True,
        help="directory of <name>.policy.json files; writes <name>.attestation.json beside each",
    )
    many.add_argument("--widen-probes", action="store_true", help=_WIDEN_HELP)
    many.add_argument("--jobs", type=int, default=4, help="concurrent audits")
    return parser


_WIDEN_HELP = (
    "audit under the policy plus exactly the audit's own probes (for interpreter-only "
    "worker and validator policies); the attestation still binds the session profile"
)


def _beside(policy_file: Path, suffix: str) -> Path:
    """`<name>.policy.json` -> `<name><suffix>` in the same directory."""
    return policy_file.with_name(policy_file.name.removesuffix(".policy.json") + suffix)


def _sentinel_sidecar(path: Path) -> tuple[Path, Path, Path] | None:
    """Optional `<name>.sentinels.json` beside a policy: the controller-private
    directories (visible, output, hidden) that receive that policy's sentinels."""
    sidecar = _beside(path, ".sentinels.json")
    if not sidecar.is_file():
        return None
    try:
        declared = json.loads(sidecar.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfinementError(f"sentinel sidecar unreadable: {sidecar}: {exc}") from exc
    roles = ("visible", "output", "hidden")
    if (
        type(declared) is not dict
        or set(declared) != set(roles)
        or any(type(declared[role]) is not str for role in roles)
    ):
        raise ConfinementError(f"sentinel sidecar must name visible, output and hidden: {sidecar}")
    return tuple(Path(declared[role]) for role in roles)


def attest_many(policies: Path, *, widen_probes: bool, jobs: int) -> dict[str, Any]:
    """Audit every `<name>.policy.json` in a directory, `jobs` at a time, writing
    `<name>.attestation.json` beside each. A valid attestation already bound to
    the policy and requested probe mode is skipped, so a batch can be resumed;
    one that cannot be loaded or attested is reported by name while the others
    still land. An optional
    `<name>.sentinels.json` names controller-private sentinel homes."""
    from concurrent.futures import ThreadPoolExecutor

    policies = Path(policies)
    if type(jobs) is not int or jobs < 1:
        raise ConfinementError("jobs must be a positive integer")
    files = sorted(policies.glob("*.policy.json"))
    if not files:
        raise ConfinementError(f"no *.policy.json files under {policies}")

    def attestation_path(path: Path) -> Path:
        return _beside(path, ".attestation.json")

    refused: dict[str, str] = {}
    loaded: dict[Path, tuple[MacOSPolicy, tuple[Path, Path, Path] | None]] = {}
    skipped = 0
    for path in files:
        try:
            policy, homes = load_policy(path), _sentinel_sidecar(path)
            existing = attestation_path(path)
            if existing.exists():
                report = json.loads(existing.read_bytes())
                if not isinstance(report, dict):
                    raise ConfinementError("existing attestation is not an object")
                if ("probe_widening" in report) != widen_probes:
                    raise ConfinementError("existing attestation probe mode differs")
                compiled = compile_macos_profile(policy)
                with tempfile.TemporaryDirectory(prefix="aisle-attestation-check-") as temp:
                    profile_path = Path(temp) / "profile.sb"
                    profile_path.write_text(compiled.text)
                    wrap_verified_command(
                        ["/usr/bin/true"], compiled, profile_path, report, policy=policy
                    )
                skipped += 1
                continue
            loaded[path] = (policy, homes)
        except (ConfinementError, OSError, ValueError, TypeError) as exc:
            refused[path.name] = str(exc)

    # Policies that would place a sentinel in the same directory (worker
    # policies share their last hidden root) must not audit at the same time:
    # each audit holds the locks of its three sentinel homes, in a fixed order.
    home_locks: dict[Path, threading.Lock] = {}

    def homes_of(policy: MacOSPolicy, homes) -> tuple[Path, ...]:
        try:
            return _sentinel_homes(policy, homes)
        except ConfinementError:
            return ()  # an invalid sentinel home fails again, and is reported, inside attest()

    for policy, homes in loaded.values():
        for home in homes_of(policy, homes):
            home_locks.setdefault(home, threading.Lock())

    def attest(path: Path) -> tuple[Path, dict[str, Any] | None, str | None]:
        policy, homes = loaded[path]
        with contextlib.ExitStack() as held:
            for home in sorted(set(homes_of(policy, homes)), key=str):
                held.enter_context(home_locks[home])
            try:
                report = run_macos_capability_audit(
                    policy, sentinel_homes=homes, widen_probes=widen_probes
                )
            except (ConfinementError, OSError) as exc:
                return path, None, str(exc)
        if not report["capability_pass"]:
            return path, report, "confinement capability cases failed"
        return path, report, None

    attested = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for path, report, error in pool.map(attest, loaded):
            if error is not None:
                refused[path.name] = error
                continue
            with attestation_path(path).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            attested += 1
    return {
        "ok": not refused,
        "attested": attested,
        "skipped": skipped,
        "refused": dict(sorted(refused.items())),
    }


def load_policy(path: Path) -> MacOSPolicy:
    """A MacOSPolicy from its canonical JSON form (lists of absolute paths)."""
    try:
        declared = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfinementError(f"policy file unreadable: {exc}") from exc
    return MacOSPolicy.from_canonical(declared)


def _sentinel_homes_from_args(args) -> tuple[Path, Path, Path] | None:
    homes = (args.sentinel_visible, args.sentinel_output, args.sentinel_hidden)
    if all(home is None for home in homes):
        return None
    if any(home is None for home in homes):
        raise ConfinementError("sentinel homes must be given for visible, output and hidden")
    return tuple(Path(home) for home in homes)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "attest-many":
        try:
            result = attest_many(args.policies, widen_probes=args.widen_probes, jobs=args.jobs)
        except (ConfinementError, OSError) as exc:
            print(json.dumps({"error": str(exc), "ok": False}, sort_keys=True), file=sys.stderr)
            return 2
        if not result["ok"]:
            names = ", ".join(f"{name}: {why}" for name, why in result["refused"].items())
            print(
                json.dumps({"error": "refused: " + names, "ok": False, **result}, sort_keys=True),
                file=sys.stderr,
            )
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        policy = load_policy(args.policy) if args.policy is not None else None
        report = write_macos_capability_audit(
            args.output,
            policy,
            sentinel_homes=_sentinel_homes_from_args(args),
            widen_probes=args.widen_probes,
        )
    except (ConfinementError, OSError) as exc:
        print(json.dumps({"error": str(exc), "ok": False}, sort_keys=True), file=sys.stderr)
        return 2
    if not report["capability_pass"]:
        print(
            json.dumps(
                {"error": "confinement capability cases failed", "ok": False}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "capability_pass": report["capability_pass"],
                "confirmatory_ready": report["confirmatory_ready"],
                "evidence_class": report["evidence_class"],
                "ok": True,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess test
    raise SystemExit(main())
