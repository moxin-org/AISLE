"""Private, deadline-bounded IPC between the P3 controller and trusted Dora host.

The controller retains ownership of its separately confined worker. A blocked
Node callback therefore cannot prevent its caller from timing out and reaping
that worker. The caller must also own and clean up the Dora graph process.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import socket
import time
from pathlib import Path

from aisle.harness.p3_task_boundary import _identifier, _require

MAX_FRAME = 32768


def _object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate IPC field")
        value[key] = item
    return value


def _decode(raw):
    value = json.loads(raw, object_pairs_hook=_object)
    _require(type(value) is dict, "IPC frame must be an object")
    json.dumps(value, allow_nan=False)
    return value


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _timeout(value):
    _require(
        type(value) in (int, float) and math.isfinite(value) and 0 < value <= 300,
        "host timeout must be in (0, 300]",
    )


class Channel:
    """One absolute controller deadline covers all socket reads and writes."""

    def __init__(self, sock, *, deadline, clock=time.monotonic):
        self.sock, self.deadline, self.clock = sock, deadline, clock

    def _remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("P3 host channel deadline exhausted")
        self.sock.settimeout(remaining)

    def send(self, value):
        _require(type(value) is dict, "IPC frame must be an object")
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        _require(0 < len(raw) <= MAX_FRAME, "IPC frame exceeds budget")
        self._remaining()
        self.sock.sendall(len(raw).to_bytes(4, "big") + raw)

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            self._remaining()
            chunk = self.sock.recv(size - len(result))
            _require(bool(chunk), "partial host IPC frame")
            result.extend(chunk)
        return bytes(result)

    def receive(self):
        size = int.from_bytes(self._read(4), "big")
        _require(0 < size <= MAX_FRAME, "IPC frame exceeds budget")
        return _decode(self._read(size))


class TaskLink:
    """Controller-side TaskSession transport; neither endpoint nor token reaches the worker."""

    def __init__(self, path, *, session_id, source_sha256, token, timeout_s):
        _timeout(timeout_s)
        _require(
            _identifier(session_id) and _digest(source_sha256) and _digest(token),
            "invalid host binding",
        )
        self.path = Path(path).absolute()
        _require(
            self.path.resolve() == self.path and not self.path.exists(),
            "host socket requires a fresh canonical path",
        )
        _require(len(str(self.path).encode()) <= 100, "host socket path is too long")
        self.binding = dict(
            version=1, session_id=session_id, source_sha256=source_sha256, token=token
        )
        self.deadline = time.monotonic() + timeout_s
        self.closed, self.channel, self.peer, self.calls = False, None, None, 0
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.server.bind(str(self.path))
            self.identity = (self.path.stat().st_dev, self.path.stat().st_ino)
            self.path.chmod(0o600)
            self.server.listen(1)
        except BaseException:
            self.server.close()
            if hasattr(self, "identity"):
                self.close()
            raise

    def accept(self):
        _require(
            not self.closed and self.channel is None, "host link is closed or already attached"
        )
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("P3 host attachment deadline exhausted")
            self.server.settimeout(remaining)
            self.peer, _ = self.server.accept()
            self.channel = Channel(self.peer, deadline=self.deadline)
            hello = self.channel.receive()
            _require(
                set(hello) == set(self.binding)
                and all(
                    type(hello[k]) is type(v)
                    and (hmac.compare_digest(hello[k], v) if k == "token" else hello[k] == v)
                    for k, v in self.binding.items()
                ),
                "host binding differs",
            )
        except BaseException:
            self.close()
            raise

    def _call(self, request):
        _require(not self.closed and self.channel is not None, "host link is closed or unattached")
        try:
            _require(self.calls < 2048, "host IPC call budget exhausted")
            self.calls += 1
            self.channel.send(request)
            response = self.channel.receive()
            _require(response.get("ok") is True, "host refused task operation")
            if request["operation"] == "next":
                _require(set(response) == {"ok", "event"}, "invalid host event reply")
                event = response["event"]
                _require(
                    event is None or isinstance(event, list) and len(event) == 3,
                    "invalid host task event",
                )
            else:
                _require(response == {"ok": True}, "invalid host send reply")
            return response
        except BaseException:
            self.close()
            raise

    def next_event(self):
        reply = self._call({"operation": "next"})
        value = reply["event"]
        if value is None:
            self.close()
            raise StopIteration
        return tuple(value)

    def send(self, topic, payload, metadata):
        self._call(dict(operation="send", topic=topic, payload=payload, metadata=metadata))

    def close(self):
        self.closed = True
        if self.peer is not None:
            self.peer.close()
        self.server.close()
        try:
            current = self.path.lstat()
            if (current.st_dev, current.st_ino) == self.identity:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def serve(channel, transport, *, session_id, source_sha256, token):
    """Trusted host dispatch only; a blocked callback is bounded by the controller's channel."""
    channel.send(dict(version=1, session_id=session_id, source_sha256=source_sha256, token=token))
    for index in range(2048):
        request = channel.receive()
        if request == {"operation": "next"}:
            try:
                event = transport.next_event()
            except StopIteration:
                channel.send(dict(ok=True, event=None))
                return dict(ok=True, calls=index + 1, input_exhausted=True)
            channel.send(dict(ok=True, event=event))
        elif (
            set(request) == {"operation", "topic", "payload", "metadata"}
            and request["operation"] == "send"
        ):
            transport.send(request["topic"], request["payload"], request["metadata"])
            channel.send(dict(ok=True))
        else:
            raise ValueError("undeclared host operation")
    raise ValueError("host IPC call budget exhausted")


def source_hashes():
    """Fixed trusted adapter closure; this is separate from runtime/graph admission."""
    root = Path(__file__).resolve().parent
    names = [
        "p3_dora_host.py",
        "p3_dora_transport.py",
        "p3_task_boundary.py",
        "../turn_node.py",
        "../turns.py",
        "../topics.py",
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def load_config(path, digest):
    path = Path(path).absolute()
    _require(
        path.resolve() == path and path.is_file() and _digest(digest),
        "invalid host config path/hash",
    )
    with path.open("rb") as stream:
        raw = stream.read(65537)
    _require(
        len(raw) <= 65536 and hashlib.sha256(raw).hexdigest() == digest, "host config hash differs"
    )
    config = _decode(raw)
    _require(
        set(config)
        == {
            "version",
            "session_id",
            "source_sha256",
            "token",
            "socket",
            "timeout_s",
            "turn_epoch",
            "source_hashes",
            "output",
        },
        "invalid host config fields",
    )
    _require(
        type(config["version"]) is int
        and config["version"] == 1
        and _identifier(config["session_id"])
        and _digest(config["source_sha256"])
        and _digest(config["token"]),
        "invalid host config binding",
    )
    _timeout(config["timeout_s"])
    _require(type(config["turn_epoch"]) is int and config["turn_epoch"] >= 0, "invalid host epoch")
    _require(config["source_hashes"] == source_hashes(), "host source drift")
    for name in ("socket", "output"):
        value = config[name]
        _require(
            type(value) is str
            and Path(value).is_absolute()
            and Path(value).resolve() == Path(value),
            "host paths must be canonical and absolute",
        )
    _require(not Path(config["output"]).exists(), "host output already exists")
    return config


def run_host(path, digest, *, raw_node_factory=None):
    config = load_config(path, digest)
    result = dict(
        ok=False,
        config_sha256=digest,
        source_sha256=config["source_sha256"],
        session_id=config["session_id"],
        provider_calls=0,
        measured_sessions=0,
        guard_traversal_verified=False,
    )
    try:
        from aisle.harness.p3_dora_transport import DoraTaskTransport
        from aisle.turn_node import Node

        if raw_node_factory is None:
            from dora import Node as DoraNode

            raw_node_factory = DoraNode
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(config["timeout_s"])
            sock.connect(config["socket"])
            channel = Channel(sock, deadline=time.monotonic() + config["timeout_s"])
            transport = DoraTaskTransport(Node(raw_node_factory()), turn_epoch=config["turn_epoch"])
            receipt = serve(
                channel,
                transport,
                **{k: config[k] for k in ("session_id", "source_sha256", "token")},
            )
            _require(load_config(path, digest) == config, "host config drift")
            result.update(receipt)
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        with Path(config["output"]).open("x") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_host(args.config, args.config_sha256)
    except Exception as exc:
        result = dict(ok=False, error=str(exc), provider_calls=0, measured_sessions=0)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
