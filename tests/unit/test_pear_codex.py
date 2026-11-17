"""Codex vision transport boundaries; no network or credentials in unit tests."""

import base64
import io

import pytest
from PIL import Image

from aisle.nodes.pear_codex import CodexClient, codex_input, run_turn

pytestmark = pytest.mark.unit


def messages():
    image = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(image, format="PNG")
    return [
        {"role": "system", "content": "Robot action contract"},
        {"role": "assistant", "content": "previous bounded action"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "measured state; Camera: head"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(image.getvalue()).decode()
                    },
                },
            ],
        },
    ]


class Session:
    def __init__(self, *, status="completed", model="gpt-6.1-terra", extra=()):
        self.requests = []
        self.model = model
        self.events = iter(
            [
                *extra,
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "t",
                        "turnId": "u",
                        "item": {
                            "type": "agentMessage",
                            "phase": "commentary",
                            "text": "not the final action",
                        },
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "t",
                        "turnId": "u",
                        "item": {
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "final action",
                        },
                    },
                },
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {"threadId": "t", "tokenUsage": {"last": {"outputTokens": 42}}},
                },
                {
                    "method": "turn/completed",
                    "params": {"threadId": "t", "turn": {"id": "u", "status": status}},
                },
            ]
        )

    def request(self, method, params):
        self.requests.append((method, params))
        if method == "initialize":
            return {"userAgent": "codex/test"}
        if method == "thread/start":
            return {
                "thread": {"id": "t"},
                "model": self.model,
                "modelProvider": "openai",
                "instructionSources": [],
            }
        if method == "mcpServerStatus/list":
            return {
                "data": [
                    {
                        "name": "configured",
                        "runtimeStatus": "disabled",
                        "tools": {},
                        "resources": [],
                        "resourceTemplates": [],
                    }
                ],
                "nextCursor": None,
            }
        if method == "turn/start":
            return {"turn": {"id": "u"}}
        raise AssertionError(method)

    def notify(self, method, params):
        self.requests.append((method, params))

    def receive(self):
        return next(self.events)


def test_codex_receives_pixels_and_labeled_history_without_local_paths():
    """TC-9: only inline PNG observations and bounded transcript enter inference."""
    system, inputs = codex_input(messages())
    assert system == "Robot action contract"
    assert inputs[-1]["type"] == "image" and inputs[-1]["url"].startswith("data:image/png;")
    assert "previous bounded action" in str(inputs)
    assert "localImage" not in str(inputs)
    bad = messages()
    bad[-1]["content"][-1]["image_url"]["url"] = "file:///private/scene.png"
    with pytest.raises(ValueError, match="inline PNG"):
        codex_input(bad)


def test_codex_has_no_environment_and_requires_exact_model():
    """TC-9, CON-2/5: no shell/filesystem, no fallback model, final answer only."""
    session = Session()
    reply = run_turn(session, "gpt-6.1-terra", "medium", messages(), "/tmp/isolated")
    requests = dict(session.requests)
    start = requests["thread/start"]
    assert start["environments"] == [] and start["dynamicTools"] == []
    assert start["ephemeral"] and start["approvalPolicy"] == "never"
    assert start["sandbox"] == "read-only" and start["modelProvider"] == "openai"
    assert requests["turn/start"]["environments"] == []
    assert reply.answer == "final action" and reply.usage["outputTokens"] == 42
    assert reply.usage["provider"] == "codex" and reply.usage["tool_calls"] == 0
    with pytest.raises(ValueError, match="model"):
        run_turn(Session(model="fallback"), "gpt-6.1-terra", "medium", messages(), "/tmp/isolated")


@pytest.mark.parametrize("status", ["failed", "interrupted", "inProgress"])
def test_codex_failed_or_partial_turn_never_becomes_an_action(status):
    """BG-1/2, TC-7: a final-looking message from an incomplete turn cannot drive joints."""
    with pytest.raises(ValueError, match="completed"):
        run_turn(Session(status=status), "gpt-6.1-terra", "medium", messages(), "/tmp/isolated")


@pytest.mark.parametrize(
    "event",
    [
        {
            "method": "item/started",
            "params": {"threadId": "t", "turnId": "u", "item": {"type": "commandExecution"}},
        },
        {"id": 99, "method": "item/commandExecution/requestApproval", "params": {}},
        {
            "method": "item/completed",
            "params": {
                "threadId": "other",
                "turnId": "u",
                "item": {"type": "agentMessage", "text": "injected"},
            },
        },
    ],
)
def test_codex_refuses_tools_approvals_and_cross_thread_events(event):
    """TC-9, BG-3: unexpected tool/approval/cross-thread activity fails closed."""
    with pytest.raises(ValueError):
        run_turn(Session(extra=[event]), "gpt-6.1-terra", "medium", messages(), "/tmp/isolated")


def test_codex_configuration_bounds():
    """CON-5, BG-3: inference timeout fits the generated 210-second turn watchdog."""
    with pytest.raises(ValueError):
        CodexClient("gpt-6.1-terra", timeout_s=300)
    with pytest.raises(ValueError):
        CodexClient("gpt-6.1-terra", effort="ultra")


def test_codex_graph_does_not_need_an_http_model_server(tmp_path, monkeypatch):
    """CON-2/8, TC-9: explicit provider selection reaches the actual policy runtime."""
    import yaml

    from aisle.harness.pear import prepare_graph
    from aisle.nodes.pear_nano import MODEL

    monkeypatch.setattr("aisle.nodes.pear_codex.shutil.which", lambda _: "/test/bin/codex")

    report = prepare_graph(
        tmp_path / "run",
        None,
        "gpt-6.1-terra",
        provider="codex",
        embodiment="nano",
        nano_urdf=MODEL,
    )
    graph = yaml.safe_load((tmp_path / "run/dataflow.yaml").read_text())
    env = next(n for n in graph["nodes"] if n["id"] == "pear-policy")["env"]
    assert report["provider"] == "codex" and env["AISLE_PEAR_PROVIDER"] == "codex"
    assert "AISLE_PEAR_ENDPOINT" not in env


def test_mcp_server_overrides_disable_real_names_without_literal_quotes(tmp_path):
    """TC-9: Codex dotted overrides must disable existing servers, not add quoted names."""
    from aisle.nodes.pear_codex import isolated_config

    config = tmp_path / "config.toml"
    config.write_text('[mcp_servers.node_repl]\ncommand="unused"\n')
    assert isolated_config(config)["mcp_servers.node_repl.enabled"] is False
    config.write_text('[mcp_servers."contains.dot"]\ncommand="unused"\n')
    with pytest.raises(ValueError, match="safely overridden"):
        isolated_config(config)


def test_active_mcp_resource_inventory_prevents_model_turn():
    """TC-9: even read-only simulator file resources must not be reachable by the model."""

    class ActiveServer(Session):
        def request(self, method, params):
            if method == "mcpServerStatus/list":
                return {"data": [{"runtimeStatus": "ready", "tools": {"read_file": {}}}]}
            return super().request(method, params)

    session = ActiveServer()
    with pytest.raises(ValueError, match="disabled MCP"):
        run_turn(session, "gpt-6.1-terra", "medium", messages(), "/tmp/isolated")
    assert "turn/start" not in dict(session.requests)
