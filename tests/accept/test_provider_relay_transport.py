"""MON-8/MON-12/MON-13: HTTP delivery uses the owned shared admission boundary."""

import hashlib
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aisle.harness.frontend_dispatch import DispatchBudget
from aisle.harness.provider_relay import ProviderRelay

pytestmark = pytest.mark.accept


@pytest.mark.parametrize("exhausted", [False, True])
def test_hosted_request_is_limited_before_upstream_work_and_replays(tmp_path, exhausted):
    """MON-8/MON-12/MON-13: real HTTP hosted work shares the same pre-execution ceiling."""
    from aisle.harness.provider_source_audit import verify_provider_sources

    requests = []
    effects = []
    source = response_frame(
        0,
        items=[
            {
                "type": "web_search_call",
                "id": "search-1",
                "status": "completed",
                "action": {"type": "search", "query": "fixture"},
            }
        ],
    )

    def call(index):
        return {"turn_id": "local", "call_id": str(index), "tool_name": "exec_command"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            # This operator-owned fixture implements the explicitly bound contract.
            assert body["max_tool_calls"] == 2
            effects.append("hosted-search")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(source)))
            self.end_headers()
            self.wfile.write(source)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    worker.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(
            tmp_path / "budget", session_id="session", ceiling=1 if exhausted else 3
        ) as budget:
            budget.dispatch(call(1), b"local-before", lambda _: None)
            with ProviderRelay(
                {
                    "base_url": f"http://127.0.0.1:{upstream.server_port}/v1",
                    "hosted_tool_contract": "aisle.fixture.responses.max_tool_calls.v1",
                },
                dispatch=budget,
                output=root,
                timeout_s=5,
            ) as relay:
                client = http.client.HTTPConnection(*relay.address, timeout=5)
                body = json.dumps(
                    {
                        "model": "fixture",
                        "stream": True,
                        "tools": [{"type": "web_search"}],
                        "max_tool_calls": 999,
                    }
                ).encode()
                client.request("POST", "/v1/responses", body, {"Content-Type": "application/json"})
                response = client.getresponse()
                delivered = response.read()
                client.close()
            if not exhausted:
                budget.dispatch(call(2), b"local-after", lambda _: None)
        assert len(requests) == len(effects) == (0 if exhausted else 1)
        assert response.status == (502 if exhausted else 200)
        if not exhausted:
            assert delivered == source
            audited = verify_provider_sources(
                {path.name: path.read_bytes() for path in root.iterdir()},
                expected=relay.reference(),
                byte_limit=100000,
                delegated_tools=(),
                dispatch={
                    "artifacts": {
                        path.name: path.read_bytes() for path in (tmp_path / "budget").iterdir()
                    },
                    "expected": budget.reference(),
                    "byte_limit": 100000,
                },
            )
            assert audited["ok"], audited
            assert len(audited["hosted_calls"]) == 1
            assert audited["native_attempts"] == []
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def response_frame(count, *, items=None):
    items = (
        items
        if items is not None
        else [
            {
                "id": f"item-{i}",
                "type": "function_call",
                "call_id": f"call-{i}",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": f"touch marker-{i}"}),
            }
            for i in range(count)
        ]
    )
    events = [{"type": "response.created", "response": {"id": "response"}}]
    events.extend(
        {"type": "response.output_item.done", "output_index": i, "item": item}
        for i, item in enumerate(items)
    )
    events.append(
        {
            "type": "response.completed",
            "response": {
                "id": "response",
                "status": "completed",
                "output": items,
            },
        }
    )
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


@pytest.mark.parametrize("case", ["success", "ceiling", "malformed"])
def test_http_relay_keeps_exact_source_and_admitted_prefix(tmp_path, case):
    """MON-13: real sockets cannot deliver a second executable item on one reservation."""
    frame = response_frame(2)
    if case == "malformed":
        frame = frame.replace(b'"status": "completed"', b'"status": "failed"')
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    worker.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(
            tmp_path / "budget", session_id="session", ceiling=1 if case == "ceiling" else 2
        ) as budget:
            with ProviderRelay(
                {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1"},
                dispatch=budget,
                output=root,
                timeout_s=5,
            ) as relay:
                listener = json.loads((root / "listener.json").read_bytes())
                assert listener == {
                    "schema_version": "aisle.provider-relay-listener.v1",
                    "session_id": "session",
                    "host": relay.address[0],
                    "port": relay.address[1],
                }
                client = http.client.HTTPConnection(*relay.address, timeout=5)
                body = b'{"model":"fixture","stream":true,"input":[],"tools":[]}'
                client.request("POST", "/v1/responses", body, {"Content-Type": "application/json"})
                reply = client.getresponse()
                delivered = reply.read()
                client.close()
            reference = relay.reference()
        assert requests == [body]
        assert reference["artifacts"] == {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
        }
        if case == "success":
            assert reply.status == 200 and delivered == frame
            assert reference["failure"] is None
            assert budget.reference()["reserved"] == 2
            from aisle.harness.provider_source_audit import verify_provider_sources

            source = {path.name: path.read_bytes() for path in root.iterdir()}
            dispatch = {
                "artifacts": {
                    path.name: path.read_bytes() for path in (tmp_path / "budget").iterdir()
                },
                "expected": budget.reference(),
                "byte_limit": 4 * 1024 * 1024,
            }
            options = dict(
                expected=reference,
                dispatch=dispatch,
                byte_limit=4 * 1024 * 1024,
                delegated_tools=set(),
            )
            audited = verify_provider_sources(source, **options)
            assert audited["ok"] and audited["native_attempts"] == [1, 2], audited
            altered = dict(source)
            altered["00000001-response.sse"] = frame.replace(b"touch marker-0", b"touch changed0")
            assert not verify_provider_sources(altered, **options)["ok"]
            # Even a separately authenticated but inconsistent source reference
            # cannot link different call bytes to the existing reservations.
            replacement = {
                **reference,
                "bytes": sum(map(len, altered.values())),
                "artifacts": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in altered.items()
                },
            }
            assert not verify_provider_sources(altered, **{**options, "expected": replacement})[
                "ok"
            ]
        elif case == "ceiling":
            assert b'"call-0"' in delivered and b'"call-1"' not in delivered
            assert b"response.completed" not in delivered
            assert reference["failure"] == "DispatchRefused"
        else:
            assert b"response.output_item.done" not in delivered
            assert reference["failure"] == "ValueError"
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join()


def test_shutdown_reaps_a_client_that_never_finishes_headers(tmp_path):
    """MON-13: cancellation owns accepted sockets before HTTP request parsing finishes."""
    import socket

    entered = threading.Event()
    with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=1) as budget:
        relay = ProviderRelay(
            {"base_url": "http://127.0.0.1:1/v1"},
            dispatch=budget,
            output=tmp_path / "provider",
            timeout_s=5,
        ).__enter__()
        original = relay._server.finish_request

        def observed(*args):
            entered.set()
            return original(*args)

        relay._server.finish_request = observed
        client = socket.create_connection(relay.address, timeout=2)
        stopped = threading.Thread(target=lambda: relay.__exit__(None, None, None))
        try:
            client.sendall(b"POST /v1/responses HTTP/1.1\r\n")
            assert entered.wait(2)
            stopped.start()
            stopped.join(0.5)
            still_running = stopped.is_alive()
        finally:
            client.close()
            stopped.join(2)
        assert not stopped.is_alive(), "relay shutdown left a handler running"
        assert not still_running, "shutdown waited for client input instead of closing its socket"
        assert relay.reference()["failure"] is not None


def test_upstream_timeout_retains_received_prefix(tmp_path):
    """MON-12/MON-13: a timed-out provider response keeps its acquired failure evidence."""
    release = threading.Event()
    prefix = (
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"slow"}}\n\n'
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(prefix)
            self.wfile.flush()
            release.wait(3)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    thread.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=1) as budget:
            with ProviderRelay(
                {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1"},
                dispatch=budget,
                output=root,
                timeout_s=0.25,
            ) as relay:
                client = http.client.HTTPConnection(*relay.address, timeout=2)
                client.request("POST", "/v1/responses", b"{}")
                response = client.getresponse()
                response.read()
                client.close()
            assert relay.reference()["failure"] == "TimeoutError"
        assert (root / "00000001-response-partial.sse").read_bytes() == prefix
        assert budget.reference()["reserved"] == 0
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        thread.join()


@pytest.mark.parametrize("identity", ["response", "call"])
def test_http_identity_replay_rejects_before_delivery_and_replays_closed_evidence(
    tmp_path, identity
):
    """MON-13: repeated identities release no SSE and consume no second reservation."""
    from aisle.harness.provider_source_audit import verify_provider_replay_sources

    original = response_frame(1)
    repeated = (
        original
        if identity == "response"
        else original.replace(b'"id": "response"', b'"id": "response-2"')
    )
    sources = iter([original, repeated])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            source = next(sources)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(source)))
            self.end_headers()
            self.wfile.write(source)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    worker.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=3) as budget:
            with ProviderRelay(
                {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1"},
                dispatch=budget,
                output=root,
                timeout_s=5,
            ) as relay:
                replies = []
                for _ in range(2):
                    client = http.client.HTTPConnection(*relay.address, timeout=5)
                    try:
                        client.request(
                            "POST", "/v1/responses", b"{}", {"Content-Type": "application/json"}
                        )
                        response = client.getresponse()
                        replies.append((response.status, response.read()))
                    finally:
                        client.close()
        assert replies[0] == (200, original)
        assert replies[1][0] == 502 and b"response.output_item.done" not in replies[1][1]
        assert budget.reference()["attempts"] == budget.reference()["reserved"] == 1
        artifacts = {path.name: path.read_bytes() for path in root.iterdir()}
        assert json.loads(artifacts["00000002-delivery.json"]) == {
            "sequence": 2,
            "events_returned": 0,
            "error_type": "ValueError",
        }
        report = verify_provider_replay_sources(
            artifacts,
            expected=relay.reference(),
            dispatch={
                "artifacts": {
                    path.name: path.read_bytes() for path in (tmp_path / "budget").iterdir()
                },
                "expected": budget.reference(),
                "byte_limit": 100000,
            },
            byte_limit=100000,
            delegated_tools=(),
        )
        assert report["ok"] and report["native_attempts"] == [1], report
        assert report["replayed_requests"] == ["00000002"]
        assert not report["complete_coverage"] and not report["confinement_verified"]
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def test_relay_forwards_the_account_header_verbatim_and_retains_only_presence(tmp_path):
    """MON-13 / CON-3: a ChatGPT-login frontend sends a bearer token and an account id;
    the relay forwards both unchanged, drops any header outside its allowlist, and
    retains only presence flags, never values."""
    frame = response_frame(1)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    worker.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=2) as budget:
            with ProviderRelay(
                {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1"},
                dispatch=budget,
                output=root,
                timeout_s=5,
            ) as relay:
                client = http.client.HTTPConnection(*relay.address, timeout=5)
                body = b'{"model":"fixture","stream":true,"input":[],"tools":[]}'
                client.request(
                    "POST",
                    "/v1/responses",
                    body,
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer secret-token-9c1f",
                        "chatgpt-account-id": "acct-7b2e",
                        "X-Unlisted": "dropped",
                    },
                )
                reply = client.getresponse()
                reply.read()
                client.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    assert reply.status == 200 and len(seen) == 1
    assert seen[0]["authorization"] == "Bearer secret-token-9c1f"
    assert seen[0]["chatgpt-account-id"] == "acct-7b2e"
    assert "x-unlisted" not in seen[0]
    retained = b"".join(path.read_bytes() for path in root.iterdir())
    assert b"secret-token-9c1f" not in retained and b"acct-7b2e" not in retained
    request = json.loads((root / "00000001-request.json").read_bytes())
    assert request["authorization_present"] is True
    assert request["account_header_present"] is True


def test_pinned_relay_serves_on_the_controller_reservation(tmp_path):
    """TRT-5/MON-8 (ADR-69 §3): a pinned `relay_port` is served from the
    controller's reservation socket, never a socket the relay binds itself, and
    the port stays held after the relay closes."""
    import socket

    from aisle.harness.relay_reservation import HostAddress, reserve_relay_port

    loopback = (HostAddress("v4", "127.0.0.1"), HostAddress("v6", "::1"))
    with reserve_relay_port(addresses=lambda: loopback) as reservation:
        pinned = reservation.port
        with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=1) as budget:
            with ProviderRelay(
                {"base_url": "http://127.0.0.1:1/v1", "relay_port": pinned},
                dispatch=budget,
                output=tmp_path / "provider",
                timeout_s=5,
                reservation=reservation,
                session_token="k" * 43,
            ) as relay:
                assert relay.address == ("127.0.0.1", pinned)
                listener = json.loads((tmp_path / "provider" / "listener.json").read_text())
                assert listener["port"] == pinned
                assert set(listener) == {"schema_version", "session_id", "host", "port"}
        with socket.socket() as intruder:
            intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                intruder.bind(("127.0.0.1", pinned))


def test_pinned_relay_refuses_without_a_matching_reservation(tmp_path):
    """TRT-5 (ADR-69 §3): a pinned port without the controller's reservation
    for that exact port is not exclusive, so the relay refuses to start."""
    from aisle.harness.relay_reservation import HostAddress, reserve_relay_port

    loopback = (HostAddress("v4", "127.0.0.1"),)
    with reserve_relay_port(addresses=lambda: loopback) as reservation:
        for index, (port, held) in enumerate(
            ((reservation.port, None), (reservation.port + 1, reservation))
        ):
            with DispatchBudget(tmp_path / f"b{index}", session_id="s", ceiling=1) as budget:
                with pytest.raises(ValueError, match="reservation"):
                    ProviderRelay(
                        {"base_url": "http://127.0.0.1:1/v1", "relay_port": port},
                        dispatch=budget,
                        output=tmp_path / f"provider-{index}",
                        timeout_s=5,
                        reservation=held,
                    ).__enter__()


def test_relay_port_binding_values_are_validated():
    """MON-8: a malformed relay port refuses at binding validation."""
    from aisle.harness.provider_relay import verify_provider

    for bad in (-1, 65536, True, "4321"):
        with pytest.raises(ValueError, match="relay port"):
            verify_provider({"base_url": "http://127.0.0.1:1/v1", "relay_port": bad})
    with pytest.raises(ValueError, match="invalid provider binding"):
        verify_provider({"base_url": "http://127.0.0.1:1/v1", "port": 1})


def _token_relay_exchange(tmp_path, *, token, sent):
    """One request through a token-guarded relay; returns (status, upstream
    headers seen, retained bytes)."""
    seen = []
    frame = response_frame(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: upstream.serve_forever(poll_interval=0.05))
    worker.start()
    root = tmp_path / "provider"
    try:
        with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=2) as budget:
            with ProviderRelay(
                {"base_url": f"http://127.0.0.1:{upstream.server_port}/v1"},
                dispatch=budget,
                output=root,
                timeout_s=5,
                session_token=token,
            ) as relay:
                client = http.client.HTTPConnection(*relay.address, timeout=5)
                headers = {"Content-Type": "application/json"}
                if sent is not None:
                    headers["X-Aisle-Relay-Token"] = sent
                body = b'{"model":"fixture","stream":true,"input":[],"tools":[]}'
                client.request("POST", "/v1/responses", body, headers)
                status = client.getresponse().status
                client.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    retained = b"".join(path.read_bytes() for path in root.iterdir())
    return status, seen, retained


def test_session_token_admits_only_the_session_frontend(tmp_path):
    """MON-8/TRT-5 (ADR-69 §6): with a session token, a request without it or
    with a wrong one never reaches upstream, dispatch or the request journal;
    the right one is accepted, never forwarded and never retained."""
    token = "t" * 43
    for index, sent in enumerate((None, "wrong", token[:-1] + "u")):
        status, seen, retained = _token_relay_exchange(
            tmp_path / f"refused-{index}", token=token, sent=sent
        )
        assert seen == [] and b"-request.body" not in retained and status != 200
    status, seen, retained = _token_relay_exchange(tmp_path / "admitted", token=token, sent=token)
    assert status == 200 and len(seen) == 1
    assert "x-aisle-relay-token" not in seen[0]
    assert token.encode() not in retained


def test_session_token_must_be_a_strong_secret(tmp_path):
    """TRT-5: a short or non-string session token refuses at construction."""
    for bad in ("short", 12345, ""):
        with DispatchBudget(tmp_path / f"b-{bad!s}", session_id="s", ceiling=1) as budget:
            with pytest.raises(ValueError, match="session token"):
                ProviderRelay(
                    {"base_url": "http://127.0.0.1:1/v1"},
                    dispatch=budget,
                    output=tmp_path / f"p-{bad!s}",
                    timeout_s=5,
                    session_token=bad,
                )


def test_session_token_mode_survives_foreign_probes(tmp_path):
    """TRT-5 (ADR-69 §6): with a session token, a bare connect, a GET and an
    unauthenticated POST are refused without failing the session, since any
    local process can reach a loopback listener."""
    import socket

    token = "k" * 43
    with DispatchBudget(tmp_path / "budget", session_id="session", ceiling=1) as budget:
        with ProviderRelay(
            {"base_url": "http://127.0.0.1:1/v1"},
            dispatch=budget,
            output=tmp_path / "provider",
            timeout_s=5,
            session_token=token,
        ) as relay:
            with socket.create_connection(relay.address, timeout=2):
                pass
            for method, body in (("GET", None), ("POST", b"{}")):
                client = http.client.HTTPConnection(*relay.address, timeout=5)
                client.request(method, "/v1/responses", body)
                assert client.getresponse().status == 403
                client.close()
            threading.Event().wait(0.2)
            assert not relay.failed.is_set(), relay._failure


def _token_relay(tmp_path, reservation=None, port=0):
    budget = DispatchBudget(tmp_path / "budget", session_id="session", ceiling=1)
    binding = {"base_url": "http://127.0.0.1:1/v1"}
    if port:
        binding["relay_port"] = port
    relay = ProviderRelay(
        binding,
        dispatch=budget.__enter__(),
        output=tmp_path / "provider",
        timeout_s=10,
        reservation=reservation,
        session_token="k" * 43,
    )
    return budget, relay


@pytest.mark.parametrize("probe", ["stalled_headers", "reset_after_send", "slot_flood"])
def test_token_mode_foreign_clients_cannot_fail_the_session(tmp_path, probe):
    """TRT-5 (ADR-69 §6): a stalled, reset or flooding foreign client is
    dropped without failing a token-guarded session."""
    import socket
    import struct
    import time

    budget, relay = _token_relay(tmp_path)
    held = []
    with relay:
        if probe == "stalled_headers":
            client = socket.create_connection(relay.address, timeout=5)
            client.sendall(b"POST /v1/responses HTTP/1.1\r\nHost:")
            held.append(client)
        elif probe == "reset_after_send":
            client = socket.create_connection(relay.address, timeout=5)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            client.close()
        else:
            held.extend(socket.create_connection(relay.address, timeout=5) for _ in range(12))
        time.sleep(1.6)
        assert not relay.failed.is_set(), relay._failure
        for client in held:
            client.close()
    budget.__exit__(None, None, None)


def test_pinned_relay_joins_in_flight_handlers_before_closing_evidence(tmp_path):
    """MON-13 (ADR-69 §3): serving from the reservation socket must not skip
    the handler join, or evidence closes under a live request."""
    import socket
    import time

    from aisle.harness.relay_reservation import HostAddress, reserve_relay_port

    loopback = (HostAddress("v4", "127.0.0.1"),)
    with reserve_relay_port(addresses=lambda: loopback) as reservation:
        budget, relay = _token_relay(tmp_path, reservation, reservation.port)
        with relay:
            client = socket.create_connection(relay.address, timeout=5)
            client.sendall(
                b"POST /v1/responses HTTP/1.1\r\nHost: x\r\nX-Aisle-Relay-Token: "
                + b"k" * 43
                + b'\r\nContent-Length: 100\r\n\r\n{"input":'
            )
            time.sleep(0.2)
        client.close()
        budget.__exit__(None, None, None)
        assert not relay._retention_failed
        assert not [t for t in __import__("threading").enumerate() if "process_request" in t.name]


def test_pinned_relay_refuses_to_run_without_a_session_token(tmp_path):
    """TRT-5 (ADR-69 §6): a reachable pinned port without a token would admit
    any local process into the session's journal and dispatch."""
    from aisle.harness.relay_reservation import HostAddress, reserve_relay_port

    with reserve_relay_port(addresses=lambda: (HostAddress("v4", "127.0.0.1"),)) as reservation:
        with DispatchBudget(tmp_path / "budget", session_id="s", ceiling=1) as budget:
            with pytest.raises(ValueError, match="requires a session token"):
                ProviderRelay(
                    {"base_url": "http://127.0.0.1:1/v1", "relay_port": reservation.port},
                    dispatch=budget,
                    output=tmp_path / "provider",
                    timeout_s=5,
                    reservation=reservation,
                )
