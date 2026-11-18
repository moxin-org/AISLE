"""Authenticated Codex app-server vision backend; no robot code is model-executed.

Each inference uses a fresh ephemeral thread with no execution environments,
MCP, apps, plugins, memories, or project instructions. Only the supplied RGB /
proprioception transcript is passed in. Requires Codex's environments API.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
from collections import deque
from pathlib import Path

from aisle.nodes.pear_backend import ModelReply, integer

MAX_WIRE = 32 * 1024 * 1024
DISABLED_FEATURES = (
    "shell_tool",
    "hooks",
    "apps",
    "enable_mcp_apps",
    "plugins",
    "tool_suggest",
    "multi_agent",
    "multi_agent_v2",
    "spawn_csv",
    "image_generation",
    "imagegenext",
    "standalone_web_search",
    "request_permissions_tool",
    "code_mode",
    "code_mode_only",
    "token_budget",
    "current_time_reminder",
)


def codex_input(messages):
    """Convert the existing model envelope, preserving RGB and history order."""
    if (
        not isinstance(messages, list)
        or not 2 <= len(messages) <= 10
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("expected bounded robot system/history/current observation")
    if len(json.dumps(messages, allow_nan=False).encode()) > MAX_WIRE:
        raise ValueError("Codex observation transcript too large")
    inputs, images = [], 0
    for index, message in enumerate(messages[1:]):
        role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError("unexpected robot transcript role")
        label = "CURRENT OBSERVATION" if index == len(messages) - 2 else f"PREVIOUS {role}"
        inputs.append({"type": "text", "text": label, "text_elements": []})
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError("invalid observation content")
        for part in content:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                inputs.append({"type": "text", "text": part["text"], "text_elements": []})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
                    raise ValueError("Codex accepts inline PNG observations only")
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("invalid inline PNG observation")
                inputs.append({"type": "image", "url": url})
                images += 1
            else:
                raise ValueError("unexpected observation content type")
    if not 1 <= images <= 15:
        raise ValueError("Codex requires RGB observations (at most five three-camera snapshots)")
    return messages[0]["content"], inputs


def isolated_config(config=None):
    """Per-process overrides only; keep the CLI's supported login/auth handling."""
    result = {
        **{f"features.{feature}": False for feature in DISABLED_FEATURES},
        "web_search": "disabled",
        "skills.include_instructions": False,
        "memories.use_memories": False,
        "memories.generate_memories": False,
        "project_doc_max_bytes": 0,
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "notify": [],
    }
    # Empty TOML maps merge rather than delete: explicitly disable named servers.
    # Read names only from configuration, never auth files or secret values.
    if config is None:
        home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        config = home / "config.toml"
    if config.is_file():
        for name in tomllib.loads(config.read_text()).get("mcp_servers", {}):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("MCP server name cannot be safely overridden")
            result[f"mcp_servers.{name}.enabled"] = False
    return result


class AppServer:
    """One owned stdio subprocess, bounded I/O/deadline, no shared daemon."""

    def __init__(self, executable, cwd, timeout_s):
        command = [executable, "app-server", "--stdio"]
        for key, value in isolated_config().items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            start_new_session=True,
        )
        self.deadline = time.monotonic() + timeout_s
        self.buffer, self.pending, self.serial, self.received = b"", deque(), 0, 0
        os.set_blocking(self.process.stdin.fileno(), False)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
        self.process.stdin.close()
        self.process.stdout.close()
        self.errors.close()

    def _send(self, value):
        data = memoryview((json.dumps(value, allow_nan=False) + "\n").encode())
        if len(data) > MAX_WIRE:
            raise ValueError("Codex request exceeds transport budget")
        while data:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                raise TimeoutError("Codex request deadline exceeded")
            try:
                count = os.write(self.process.stdin.fileno(), data[:65536])
                data = data[count:]
            except BlockingIOError:
                continue

    def _read(self):
        while b"\n" not in self.buffer:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("Codex inference deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.errors.seek(0)
                detail = self.errors.read(2048).decode(errors="replace")
                raise ValueError(f"Codex app-server closed before completing inference: {detail}")
            self.received += len(chunk)
            if self.received > MAX_WIRE:
                raise ValueError("Codex response exceeds transport budget")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("invalid Codex event")
        if "id" in event and "method" in event:
            raise ValueError("Codex requested an unexpected tool/approval")
        return event

    def request(self, method, params):
        self.serial += 1
        self._send({"id": self.serial, "method": method, "params": params})
        while True:
            event = self._read()
            if event.get("id") == self.serial:
                if "error" in event:
                    raise ValueError(f"Codex {method}: {str(event['error'])[:512]}")
                return event["result"]
            if "id" in event:
                raise ValueError("unexpected Codex RPC response")
            self.pending.append(event)

    def notify(self, method, params):
        self._send({"method": method, "params": params})

    def receive(self):
        return self.pending.popleft() if self.pending else self._read()


def run_turn(session, model, effort, messages, cwd):
    system, inputs = codex_input(messages)
    identity = session.request(
        "initialize",
        {
            "clientInfo": {"name": "aisle_rgb_controller", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        },
    )
    session.notify("initialized", {})
    started = session.request(
        "thread/start",
        {
            "model": model,
            "modelProvider": "openai",
            "cwd": cwd,
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
            "environments": [],
            "dynamicTools": [],
            "selectedCapabilityRoots": [],
            "baseInstructions": system,
            "developerInstructions": (
                "You are a robot observation-to-action model. Use only supplied RGB images, "
                "measured robot state and execution feedback. You have no environment access. "
                "Do not use tools, files, shell, web, memories or other agents. "
                "Give a brief visible-scene observation and action intent before the final JSON. "
                "Do not claim contact or task success without visual evidence."
            ),
        },
    )
    if started.get("model") != model or started.get("modelProvider") != "openai":
        raise ValueError("Codex model/provider differs from the requested model")
    if started.get("instructionSources"):
        raise ValueError("unexpected Codex project instruction sources")
    thread = started["thread"]["id"]
    inventory = session.request("mcpServerStatus/list", {"threadId": thread})
    if inventory.get("nextCursor") or any(
        server.get("runtimeStatus") != "disabled"
        or any(server.get(key) for key in ("tools", "resources", "resourceTemplates"))
        for server in inventory.get("data", [])
    ):
        raise ValueError("Codex robot inference requires disabled MCP servers without resources")
    turn = session.request(
        "turn/start",
        {
            "threadId": thread,
            "input": inputs,
            "model": model,
            "effort": effort,
            "environments": [],
            "summary": "none",
        },
    )["turn"]["id"]
    finals, usage = [], {}
    while True:
        event = session.receive()
        if "id" in event:
            raise ValueError("unexpected Codex tool/approval/RPC")
        method, params = event.get("method", ""), event.get("params", {})
        if params.get("threadId", thread) != thread or params.get("turnId", turn) != turn:
            raise ValueError("cross-thread/turn Codex response")
        if method in {"item/started", "item/completed"}:
            item = params["item"]
            if item.get("type") not in {"agentMessage", "userMessage", "reasoning"}:
                raise ValueError("Codex attempted non-observation tool activity")
            if (
                method == "item/completed"
                and item["type"] == "agentMessage"
                and item.get("phase") in {None, "final_answer"}
            ):
                finals.append(item["text"])
        elif method == "thread/tokenUsage/updated":
            usage = params["tokenUsage"]["last"]
        elif method == "turn/completed":
            result = params["turn"]
            if result.get("id") != turn or result.get("status") != "completed":
                raise ValueError(f"Codex turn not completed: {result.get('status')}")
            break
    if len(finals) != 1 or not finals[0].strip() or len(finals[0]) > 65536:
        raise ValueError("Codex must return exactly one bounded final answer")
    return ModelReply(
        finals[0],
        {
            **usage,
            "provider": "codex",
            "model": model,
            "effort": effort,
            "codex_user_agent": identity.get("userAgent"),
            "thread_id": thread,
            "turn_id": turn,
            "tool_calls": 0,
            "environments": [],
            "seed_supported": False,
        },
    )


class CodexClient:
    def __init__(self, model, *, executable="codex", timeout_s=180, effort="medium"):
        if not isinstance(model, str) or not model.strip() or len(model) > 128:
            raise ValueError("explicit Codex model required")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 180:
            raise ValueError("Codex timeout must be within 180 seconds")
        if effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported robot inference effort")
        self.model, self.timeout_s, self.effort = model, timeout_s, effort
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ValueError("Codex CLI executable not found")

    def infer(self, messages, *, seed):
        integer(seed, "seed")  # retained for common interface; Codex has no seed parameter
        started = time.monotonic()
        # Outside repositories: no project discovery, no scene paths in the request.
        with tempfile.TemporaryDirectory(prefix="aisle-codex-rgb-") as cwd:
            with AppServer(self.executable, cwd, self.timeout_s) as session:
                result = run_turn(session, self.model, self.effort, messages, cwd)
        return ModelReply(
            result.answer, {**result.usage, "wall_time_s": time.monotonic() - started}
        )
