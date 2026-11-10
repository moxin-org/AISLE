"""MON-8/MON-13: the provider relay and frontend share a controller lifetime."""

import asyncio
import threading

import pytest

from aisle.harness import provider_runner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failed", [False, True, "closing"])
def test_provider_failure_reaps_frontend_and_retains_reference(tmp_path, monkeypatch, failed):
    """MON-13: failed admission cannot leave an unmediated frontend running."""
    finished = []
    captured = {}

    class Relay:
        address = ("127.0.0.1", 12345)

        def __init__(self, binding, **kwargs):
            captured.update(kwargs)
            self.failed = threading.Event()
            self._failure = None
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def reference(self):
            assert self.closed
            return {"failure": "DispatchRefused" if failed else None}

    relay_instance = None

    def factory(*args, **kwargs):
        nonlocal relay_instance
        relay_instance = Relay(*args, **kwargs)
        return relay_instance

    async def frontend(argv, **kwargs):
        captured["argv"] = argv
        try:
            if failed is True:
                relay_instance._failure = "DispatchRefused"
                relay_instance.failed.set()
                await asyncio.Event().wait()
            return {"ok": True}
        finally:
            finished.append(True)

    monkeypatch.setattr(provider_runner, "ProviderRelay", factory)
    monkeypatch.setattr(provider_runner, "run_app_server_async", frontend)
    references = {}
    options = dict(
        binding={"base_url": "http://127.0.0.1:1234/v1", "requires_openai_auth": False},
        dispatch=object(),
        delegated_tools={("harness", "check", "function_call")},
        provider_output=tmp_path,
        references=references,
        argv=["codex", "app-server"],
        timeout_s=1,
        output=tmp_path / "protocol",
        handle_call=lambda *_: None,
    )
    if failed:
        with pytest.raises(ValueError, match="provider.*DispatchRefused"):
            provider_runner.run_provider_app_server(**options)
    else:
        assert provider_runner.run_provider_app_server(**options) == {"ok": True}
    assert finished == [True]
    assert references["provider"]["failure"] == ("DispatchRefused" if failed else None)
    assert 'model_provider="aisle_authorized"' in captured["argv"]
    assert "model_providers.aisle_authorized.supports_websockets=false" in captured["argv"]
    assert "model_providers.aisle_authorized.requires_openai_auth=false" in captured["argv"]


def test_mcp_closure_failure_still_retains_provider_reference(tmp_path, monkeypatch):
    """MON-12/MON-13: one adapter's cleanup failure must not discard another's evidence."""
    from types import SimpleNamespace

    from aisle.harness import mcp_harness_authority, mcp_harness_transport

    class Resource:
        address = ("127.0.0.1", 12345)

        def __init__(self, *args, **kwargs):
            self.failed = threading.Event()
            self._failure = None
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def observe(self, *args, **kwargs):
            pass

        def reference(self):
            assert self.closed
            return {"failure": None}

    class FailedMCP(Resource):
        def reference(self):
            assert self.closed
            return {"failure": "OSError"}

    async def frontend(*args, **kwargs):
        return {"ok": True}

    monkeypatch.setattr(provider_runner, "ProviderRelay", Resource)
    monkeypatch.setattr(provider_runner, "run_app_server_async", frontend)
    monkeypatch.setattr(mcp_harness_authority, "MCPHarnessAuthority", FailedMCP)
    monkeypatch.setattr(mcp_harness_transport, "MCPHarnessServer", Resource)
    references = {}
    with pytest.raises(ValueError, match="MCP harness failed"):
        provider_runner.run_provider_app_server(
            binding={"base_url": "http://127.0.0.1:1234/v1"},
            dispatch=SimpleNamespace(session_id="session"),
            delegated_tools={("mcp__aisle_harness", "check", "function_call")},
            provider_output=tmp_path / "provider",
            references=references,
            argv=["codex", "app-server"],
            timeout_s=1,
            mcp_harness=True,
            mcp_output=tmp_path / "mcp",
            output=tmp_path / "protocol",
            handle_call=lambda *_: None,
        )
    assert references["mcp_harness"]["listener"] == {
        "schema_version": "aisle.mcp-harness-listener.v1",
        "session_id": "session",
        "host": "127.0.0.1",
        "port": 12345,
    }
    assert references["mcp_harness"]["failure"] == "OSError"
    assert references["provider"]["failure"] is None


class _PinnedRelay:
    address = ("127.0.0.1", 4321)
    seen = {}

    def __init__(self, binding, **kwargs):
        _PinnedRelay.seen = kwargs
        self.failed = threading.Event()
        self._failure = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def reference(self):
        return {"failure": None}


class _Reservation:
    port = 4321
    addresses = ("lo0",)


def _pinned_options(tmp_path, **extra):
    return dict(
        binding={"base_url": "https://api.openai.com/v1", "relay_port": 4321},
        dispatch=object(),
        delegated_tools=set(),
        provider_output=tmp_path / "provider",
        references={},
        argv=["codex", "app-server"],
        timeout_s=5,
        output=tmp_path / "protocol",
        handle_call=lambda *_: None,
        reservation=_Reservation(),
        **extra,
    )


def test_pinned_relay_session_gets_a_token_by_environment_only(tmp_path, monkeypatch):
    """TRT-5/MON-8 (ADR-69 §6): a pinned relay runs on the controller
    reservation with a fresh session token, which Codex reads from a
    controller-only variable named in argv; the value never appears in argv."""
    captured = {}

    async def frontend(argv, **kwargs):
        captured.update(kwargs, argv=argv)
        return {"ok": True}

    monkeypatch.setattr(provider_runner, "ProviderRelay", _PinnedRelay)
    monkeypatch.setattr(provider_runner, "run_app_server_async", frontend)
    monkeypatch.setattr(provider_runner, "drifted", lambda reservation, addresses: False)
    assert provider_runner.run_provider_app_server(**_pinned_options(tmp_path)) == {"ok": True}
    token = _PinnedRelay.seen["session_token"]
    assert isinstance(token, str) and len(token) >= 32
    assert _PinnedRelay.seen["reservation"].port == 4321
    assert captured["controller_env"] == {"AISLE_RELAY_KEY": token}
    assert (
        'model_providers.aisle_authorized.env_http_headers.X-Aisle-Relay-Token="AISLE_RELAY_KEY"'
        in captured["argv"]
    )
    assert all(token not in part for part in captured["argv"])


@pytest.mark.parametrize("extra", [{"mcp_harness": True}, {"code_mode_host": {"x": 1}}])
def test_pinned_relay_refuses_features_with_their_own_listeners(tmp_path, monkeypatch, extra):
    """TRT-5 (ADR-69 §6): MCP harness and Code Mode listen on ephemeral ports
    a one-port loopback grant cannot reach, so a pinned session refuses them."""
    monkeypatch.setattr(provider_runner, "ProviderRelay", _PinnedRelay)
    with pytest.raises(ValueError, match="single relay port"):
        provider_runner.run_provider_app_server(**_pinned_options(tmp_path, **extra))


def test_address_drift_during_a_pinned_session_stops_it(tmp_path, monkeypatch):
    """TRT-5 (ADR-69 §4): a host address change mid-session fails the session
    as an unattested environment and reaps the frontend."""
    finished = []

    async def frontend(argv, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    monkeypatch.setattr(provider_runner, "ProviderRelay", _PinnedRelay)
    monkeypatch.setattr(provider_runner, "run_app_server_async", frontend)
    monkeypatch.setattr(provider_runner, "DRIFT_INTERVAL_S", 0.01)
    monkeypatch.setattr(provider_runner, "drifted", lambda reservation, addresses: True)
    with pytest.raises(ValueError, match="host addresses changed"):
        provider_runner.run_provider_app_server(**_pinned_options(tmp_path))
    assert finished == [True]


def test_address_drift_found_at_teardown_fails_a_finished_session(tmp_path, monkeypatch):
    """TRT-5 (ADR-69 §4): drift is also checked once the frontend finishes."""

    async def frontend(argv, **kwargs):
        return {"ok": True}

    monkeypatch.setattr(provider_runner, "ProviderRelay", _PinnedRelay)
    monkeypatch.setattr(provider_runner, "run_app_server_async", frontend)
    monkeypatch.setattr(provider_runner, "drifted", lambda reservation, addresses: True)
    with pytest.raises(ValueError, match="by teardown"):
        provider_runner.run_provider_app_server(**_pinned_options(tmp_path))


def test_provider_settings_name_the_relay_token_variable_only_when_pinned():
    """MON-8 (ADR-69 §6): qualification rebuilds argv from these settings, so
    the token header binding is a pure function of the provider binding."""
    pinned = provider_runner.provider_settings(
        {"base_url": "https://api.openai.com/v1", "relay_port": 4321}, ("127.0.0.1", 4321), 5
    )
    assert pinned[provider_runner.RELAY_HEADER_SETTING] == "AISLE_RELAY_KEY"
    plain = provider_runner.provider_settings(
        {"base_url": "https://api.openai.com/v1"}, ("127.0.0.1", 4321), 5
    )
    assert provider_runner.RELAY_HEADER_SETTING not in plain
