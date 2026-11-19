"""Bounded controller/host channel, independent of Dora and authored code."""

import hashlib
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def short_path():
    with tempfile.TemporaryDirectory(prefix="p3h-", dir="/tmp") as directory:
        yield Path(directory).resolve()


def test_channel_round_trip_and_absolute_deadline():
    """HAR-4; TRT-10: a blocked host cannot strand the controller beyond its deadline."""
    from aisle.harness.p3_dora_host import Channel

    left, right = socket.socketpair()
    try:
        channel = Channel(left, deadline=time.monotonic() + 0.08)
        peer = Channel(right, deadline=time.monotonic() + 0.08)
        channel.send({"next": True})
        assert peer.receive() == {"next": True}
        with pytest.raises(TimeoutError):
            channel.receive()
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b'"text"'])
def test_invalid_frames_are_refused(payload):
    """CON-8; TRT-9: ambiguous or non-object host frames fail closed."""
    from aisle.harness.p3_dora_host import Channel

    left, right = socket.socketpair()
    try:
        right.sendall(len(payload).to_bytes(4, "big") + payload)
        with pytest.raises(ValueError):
            Channel(left, deadline=time.monotonic() + 0.2).receive()
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("data", [b"\0\0\0\0", (100000).to_bytes(4, "big"), b"\0\0"])
def test_partial_and_oversized_frames_are_refused(data):
    """HAR-4; TRT-10: truncated or oversized IPC cannot be mistaken for clean completion."""
    from aisle.harness.p3_dora_host import Channel

    left, right = socket.socketpair()
    try:
        right.sendall(data)
        right.shutdown(socket.SHUT_WR)
        with pytest.raises(ValueError):
            Channel(left, deadline=time.monotonic() + 0.2).receive()
    finally:
        left.close()
        right.close()


def test_controller_link_authenticates_session_and_delegates_fixed_calls(short_path):
    """TRT-5/6/9: the private host channel binds its token, session and candidate source."""
    from aisle.harness.p3_dora_host import Channel, TaskLink, serve

    class Transport:
        def __init__(self):
            self.events = iter([("reset_done", [1], {"env_id": 0, "seq": 1, "sim_time_ns": 0})])
            self.sent = []

        def next_event(self):
            return next(self.events)

        def send(self, *args):
            self.sent.append(args)

    transport = Transport()
    with TaskLink(
        short_path / "s",
        session_id="fixture",
        source_sha256="a" * 64,
        token="b" * 64,
        timeout_s=0.8,
    ) as link:
        errors = []

        def host():
            try:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.connect(str(link.path))
                    channel = Channel(sock, deadline=time.monotonic() + 0.8)
                    serve(
                        channel,
                        transport,
                        session_id="fixture",
                        source_sha256="a" * 64,
                        token="b" * 64,
                    )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=host)
        thread.start()
        try:
            link.accept()
            assert link.next_event() == (
                "reset_done",
                [1],
                {"env_id": 0, "seq": 1, "sim_time_ns": 0},
            )
            link.send("target_request", {"target_med": "a"}, {"goal_id": "g", "request_id": "r"})
            with pytest.raises(StopIteration):
                link.next_event()
        finally:
            thread.join(timeout=1)
        assert not thread.is_alive() and not errors
        assert transport.sent == [
            ("target_request", {"target_med": "a"}, {"goal_id": "g", "request_id": "r"})
        ]
    assert not (short_path / "s").exists()


@pytest.mark.parametrize(
    "field,value",
    [("token", "c" * 64), ("session_id", "other"), ("source_sha256", "c" * 64), ("version", 2)],
)
def test_wrong_host_binding_is_fatal(short_path, field, value):
    """TRT-9: a peer from another session cannot acquire this worker's task channel."""
    from aisle.harness.p3_dora_host import Channel, TaskLink

    with (
        TaskLink(
            short_path / "s",
            session_id="fixture",
            source_sha256="a" * 64,
            token="b" * 64,
            timeout_s=0.4,
        ) as link,
        socket.socket(socket.AF_UNIX) as peer,
    ):
        peer.connect(str(link.path))
        Channel(peer, deadline=time.monotonic() + 0.4).send(
            {
                "version": 1,
                "token": "b" * 64,
                "session_id": "fixture",
                "source_sha256": "a" * 64,
                field: value,
            }
        )
        with pytest.raises(ValueError, match="binding"):
            link.accept()
        with pytest.raises(ValueError, match="closed"):
            link.next_event()


def test_host_refuses_undeclared_operation_without_dispatch():
    """VAL-2; TRT-5: private IPC still refuses undeclared task operations."""
    from aisle.harness.p3_dora_host import serve

    class Channel:
        def send(self, value):
            pass

        def receive(self):
            return {"operation": "reset", "payload": [300, 0]}

    with pytest.raises(ValueError, match="operation"):
        serve(Channel(), object(), session_id="fixture", source_sha256="a" * 64, token="b" * 64)


def test_host_config_hash_and_source_drift_prevent_attachment(tmp_path, monkeypatch):
    """CON-5; TRT-9: exact config/source checks precede Dora attachment."""
    from aisle.harness.p3_dora_host import load_config, run_host, source_hashes

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"not": "valid"}))
    with pytest.raises(ValueError, match="hash"):
        load_config(path, "a" * 64)
    config = dict(
        version=1,
        session_id="fixture",
        source_sha256="a" * 64,
        token="b" * 64,
        socket=str(tmp_path / "s"),
        output=str(tmp_path / "output.json"),
        timeout_s=1,
        turn_epoch=7,
        source_hashes=source_hashes(),
    )
    config["source_hashes"]["p3_dora_host.py"] = "c" * 64
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="source drift"):
        run_host(
            path,
            hashlib.sha256(raw).hexdigest(),
            raw_node_factory=lambda: pytest.fail("Dora attached before source admission"),
        )


@pytest.mark.parametrize(
    "reply",
    [
        {"ok": True, "event": 7},
        {"ok": True, "event": None, "extra": 1},
        {"ok": False, "error": "failed"},
    ],
)
def test_malformed_host_reply_permanently_closes_link(short_path, reply):
    """TRT-9: malformed host results terminate authority even if the caller catches the error."""
    from aisle.harness.p3_dora_host import TaskLink

    with TaskLink(
        short_path / "s",
        session_id="fixture",
        source_sha256="a" * 64,
        token="b" * 64,
        timeout_s=0.5,
    ) as link:
        link.channel = SimpleNamespace(send=lambda value: None, receive=lambda: reply)
        with pytest.raises(ValueError):
            link.next_event()
        with pytest.raises(ValueError, match="closed"):
            link.next_event()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("session_id", "../bad"),
        ("source_sha256", "alias"),
        ("token", ""),
        ("timeout_s", 301),
        ("timeout_s", True),
        ("turn_epoch", True),
        ("socket", "relative"),
        ("output", "relative"),
    ],
)
def test_config_component_mutation_refuses_before_node_attachment(tmp_path, field, value):
    """TRT-5/9: malformed host identities, paths and budgets are refused before Dora starts."""
    from aisle.harness.p3_dora_host import load_config, run_host, source_hashes

    config = dict(
        version=1,
        session_id="fixture",
        source_sha256="a" * 64,
        token="b" * 64,
        timeout_s=1,
        turn_epoch=7,
        socket=str(tmp_path / "s"),
        output=str(tmp_path / "out"),
        source_hashes=source_hashes(),
    )
    path = tmp_path / "config.json"
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    assert load_config(path, hashlib.sha256(raw).hexdigest()) == config
    config[field] = value
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        run_host(
            path,
            hashlib.sha256(raw).hexdigest(),
            raw_node_factory=lambda: pytest.fail("Dora attached before admission"),
        )


def test_host_cli_refusal_is_one_json_object(tmp_path, capsys):
    """CON-8: failed host admission returns JSON with a nonzero exit and no provider calls."""
    from aisle.harness.p3_dora_host import main

    assert main(["--config", str(tmp_path / "missing"), "--config-sha256", "a" * 64]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and result["provider_calls"] == result["measured_sessions"] == 0
