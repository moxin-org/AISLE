"""Controller-owned Codex app-server stdio transport for dynamic harness tools.

Only messages read from the owned process pipe reach the dispatch callback.
The caller binds argv, environment, thread configuration, frontend revision and
outer confinement to its admitted plan. This does not cover built-in tools or
attest external confinement; unsupported server requests fail the session.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import time
from pathlib import Path

MAX_MESSAGE_BYTES = 65536


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate app-server field")
        result[key] = value
    return result


def parse_dynamic_call(message, *, thread_id, turn_id):
    """Validate one server request after acquisition from the owned stdio pipe."""
    if (
        type(message) is not dict
        or set(message) != {"id", "method", "params"}
        or message["method"] != "item/tool/call"
        or type(message["id"]) not in (str, int)
    ):
        raise ValueError("invalid app-server tool request")
    params = message["params"]
    if (
        type(params) is not dict
        or set(params) != {"threadId", "turnId", "callId", "namespace", "tool", "arguments"}
        or params["threadId"] != thread_id
        or params["turnId"] != turn_id
        or params["namespace"] != "harness"
        or params["tool"] not in ("check", "run")
        or type(params["arguments"]) is not dict
        or params["arguments"]
        or any(
            type(params[key]) is not str or not 0 < len(params[key]) <= 256
            for key in ("threadId", "turnId", "callId")
        )
    ):
        raise ValueError("unbound or unsupported app-server tool call")
    return {
        "turn_id": turn_id,
        "call_id": params["callId"],
        "tool_name": "harness." + params["tool"],
    }


class AppServerUsage:
    """Meter cumulative new-input plus output tokens using the existing Codex budget unit."""

    def __init__(self, thread_id, turn_id):
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.total = None

    def feed(self, params):
        if (
            type(params) is not dict
            or params.get("threadId") != self.thread_id
            or params.get("turnId") != self.turn_id
        ):
            raise ValueError("usage belongs to another thread or turn")
        usage = params.get("tokenUsage")
        if type(usage) is not dict or not {"total", "last"}.issubset(usage):
            raise ValueError("missing app-server usage")
        fields = {
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        }
        for name in ("total", "last"):
            value = usage[name]
            if (
                type(value) is not dict
                or not fields.issubset(value)
                or set(value) - fields - {"cacheWriteInputTokens"}
                or any(type(v) is not int or v < 0 for v in value.values())
                or value["cachedInputTokens"] > value["inputTokens"]
                or value["reasoningOutputTokens"] > value["outputTokens"]
                or value["totalTokens"] != value["inputTokens"] + value["outputTokens"]
            ):
                raise ValueError("invalid app-server usage breakdown")
        total = usage["total"]
        if any(usage["last"][key] > total[key] for key in fields):
            raise ValueError("last usage exceeds cumulative usage")
        if self.total is not None and any(total[key] < self.total[key] for key in fields):
            raise ValueError("app-server cumulative usage moved backwards")
        spend = total["inputTokens"] - total["cachedInputTokens"] + total["outputTokens"]
        if self.total is not None and spend < self.report()["tokens"]:
            raise ValueError("app-server usage reduced previously observed spend")
        self.total = dict(total)

    def report(self):
        if self.total is None:
            raise ValueError("app-server usage is missing")
        return {
            "tokens": self.total["inputTokens"]
            - self.total["cachedInputTokens"]
            + self.total["outputTokens"],
            "tokens_generated": self.total["outputTokens"],
        }


class AppServerScope:
    """Bind child lifetimes from the owned stream and sum per-thread token totals."""

    def __init__(self, thread_id, turn_id):
        self.root = thread_id
        self.turns = {thread_id: turn_id}
        self.paths = {thread_id: "/root"}
        self.meters = {thread_id: AppServerUsage(thread_id, turn_id)}
        self.active = {thread_id}
        self.started = set()
        self.used_turns = {turn_id}
        self.accounted_turns = set()
        self.root_done = False
        self.complete = False

    def _identity(self, thread, turn):
        if (
            type(thread) is not str
            or type(turn) is not str
            or thread not in self.active
            or self.turns.get(thread) != turn
        ):
            raise ValueError("unbound or inactive app-server thread or turn")

    def parse_call(self, message):
        params = message["params"]
        self._identity(params.get("threadId"), params.get("turnId"))
        return parse_dynamic_call(message, thread_id=params["threadId"], turn_id=params["turnId"])

    def feed(self, message):
        if self.complete:
            raise ValueError("app-server protocol continues after root completion")
        method = message.get("method")
        params = message.get("params", {})
        if type(params) is not dict:
            raise ValueError("invalid app-server notification parameters")
        if method in {"item/started", "item/completed"} and type(params.get("item")) is not dict:
            raise ValueError("invalid app-server item notification")
        if method in {"turn/started", "turn/completed"} and type(params.get("turn")) is not dict:
            raise ValueError("invalid app-server turn notification")
        if method == "item/completed" and params.get("item", {}).get("type") == "subAgentActivity":
            item = params["item"]
            if item.get("kind") == "started":
                parent = params.get("threadId")
                self._identity(parent, params.get("turnId"))
                child, path = item.get("agentThreadId"), item.get("agentPath")
                if (
                    type(child) is not str
                    or not 0 < len(child) <= 256
                    or child in self.turns
                    or len(self.turns) >= 1024
                    or type(path) is not str
                    or not path.startswith(self.paths[parent] + "/")
                    or "/" in path[len(self.paths[parent]) + 1 :]
                    or not path[len(self.paths[parent]) + 1 :]
                    or path in self.paths.values()
                ):
                    raise ValueError("unbound or duplicate child-agent identity")
                self.turns[child] = None
                self.paths[child] = path
        elif method == "turn/started":
            thread, turn = params.get("threadId"), params.get("turn", {}).get("id")
            if (
                type(thread) is not str
                or thread not in self.turns
                or type(turn) is not str
                or not 0 < len(turn) <= 256
            ):
                raise ValueError("unbound app-server turn start")
            if thread == self.root and turn == self.turns[thread] and thread not in self.started:
                self.started.add(thread)
                return False
            if thread in self.active or turn in self.used_turns or len(self.used_turns) >= 10000:
                raise ValueError("overlapping or replayed app-server turn")
            self.used_turns.add(turn)
            self.turns[thread] = turn
            self.active.add(thread)
            self.started.add(thread)
            if thread not in self.meters:
                self.meters[thread] = AppServerUsage(thread, turn)
            else:
                self.meters[thread].turn_id = turn
        elif method == "thread/tokenUsage/updated":
            thread = params.get("threadId")
            self._identity(thread, params.get("turnId"))
            self.meters[thread].feed(params)
            self.accounted_turns.add(params["turnId"])
        elif method == "turn/completed":
            thread, turn = params.get("threadId"), params.get("turn", {})
            self._identity(thread, turn.get("id"))
            if turn.get("status") != "completed":
                raise ValueError("app-server turn failed")
            if thread == self.root:
                self.root_done = True
            self.active.remove(thread)
            if (
                self.root_done
                and not self.active
                and all(t is not None for t in self.turns.values())
            ):
                if (
                    len(self.turns) > 1 or self.has_usage
                ) and self.used_turns != self.accounted_turns:
                    raise ValueError("app-server usage is missing for an owned turn")
                self.complete = True
        return self.complete

    @property
    def has_usage(self):
        return any(meter.total is not None for meter in self.meters.values())

    def report(self):
        reports = [meter.report() for meter in self.meters.values() if meter.total is not None]
        if not reports:
            raise ValueError("app-server usage is missing")
        return {key: sum(row[key] for row in reports) for key in ("tokens", "tokens_generated")}


def dynamic_tools(operations=("check", "run")):
    """Declare the fixed check/run API without removing other frontend tools."""
    return [
        {
            "type": "namespace",
            "name": "harness",
            "description": "AISLE harness tools",
            "tools": [
                {
                    "type": "function",
                    "name": operation,
                    "description": f"Request the admitted harness {operation}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
                for operation in operations
            ],
        }
    ]


def run_app_server(*args, **kwargs):
    """Run the owned transport from a synchronous session controller."""
    return asyncio.run(run_app_server_async(*args, **kwargs))


async def run_app_server_async(
    argv,
    *,
    cwd,
    env,
    output,
    thread_params,
    input_items,
    handle_call,
    timeout_s,
    token_ceiling=None,
    on_message=None,
    on_mcp_source=None,
    on_reference=None,
    controller_env=None,
):
    """Run one fresh thread/turn over a private pipe and retain protocol bytes.

    ``controller_env`` adds controller-only variables (such as the relay
    session secret) to the child's environment. They may not replace a
    declared variable; the retained invocation hashes the declared
    environment and names the added keys, never their values.

    ``handle_call(call, raw_message)`` may return an awaitable. It must return the
    dynamic tool result only after controller authorization/execution. Callbacks
    doing blocking work must enforce their own finite deadline. The write-time
    reference returned here must be retained by the trusted controller.
    """
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 86400:
        raise ValueError("invalid app-server deadline")
    if token_ceiling is not None and (type(token_ceiling) is not int or token_ceiling <= 0):
        raise ValueError("invalid token ceiling")
    controller_env = dict(controller_env or {})
    if set(controller_env) & set(env) or any(
        type(k) is not str or type(v) is not str for k, v in controller_env.items()
    ):
        raise ValueError("controller environment may only add string variables")
    output = Path(output).absolute()
    if output.resolve() != output:
        raise ValueError("redirected app-server output")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    started = time.monotonic()

    def retain(name, raw):
        with (output / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        hashes[name] = hashlib.sha256(raw).hexdigest()

    def encode(value):
        return json.dumps(value, allow_nan=False).encode() + b"\n"

    retain(
        "invocation.json",
        encode(
            {
                "argv": argv,
                "cwd": str(cwd),
                "environment_sha256": hashlib.sha256(
                    json.dumps(env, sort_keys=True).encode()
                ).hexdigest(),
                **(
                    {"controller_environment_keys": sorted(controller_env)}
                    if controller_env
                    else {}
                ),
                "thread_params": thread_params,
                "input_items": input_items,
                "timeout_s": timeout_s,
                "transport": "owned_stdio",
                "complete_coverage": False,
                "confinement_verified": False,
            }
        ),
    )

    calls = set()
    mcp_calls = set()
    thread_id = turn_id = None

    def reference(*, failure=None):
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "dynamic_calls": len(calls),
            **({"mcp_calls": len(mcp_calls)} if on_mcp_source is not None else {}),
            "artifacts": dict(hashes),
            **({"stream_complete": False, "failure": failure} if failure is not None else {}),
        }

    async def run():
        nonlocal thread_id, turn_id
        with (output / "stderr.log").open("xb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env={**env, **controller_env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
                limit=MAX_MESSAGE_BYTES,
            )
            incoming = outgoing = 0
            request_ids = set()
            pending = []
            completed = None

            async def send(value):
                nonlocal outgoing
                data = encode(value)
                if len(data) > MAX_MESSAGE_BYTES:
                    raise ValueError("app-server outgoing message exceeds limit")
                outgoing += 1
                retain(f"{outgoing:08d}-sent.json", data)
                process.stdin.write(data)
                await process.stdin.drain()

            async def receive():
                nonlocal incoming
                raw = await process.stdout.readline()
                if not raw or not raw.endswith(b"\n") or len(raw) > MAX_MESSAGE_BYTES:
                    raise ValueError("app-server ended or exceeded its message limit")
                incoming += 1
                retain(f"{incoming:08d}-received.json", raw)
                if on_message is not None:
                    reason = on_message(raw.decode())
                    if reason is not None:
                        raise ValueError(f"frontend observation stopped session: {reason}")
                value = json.loads(raw, object_pairs_hook=_object)
                if type(value) is not dict:
                    raise ValueError("invalid app-server message")
                if "method" in value:
                    if type(value["method"]) is not str or not value["method"]:
                        raise ValueError("invalid app-server method")
                elif set(value) not in ({"id", "result"}, {"id", "error"}):
                    raise ValueError("invalid app-server response fields")
                if "id" in value and type(value["id"]) not in (str, int):
                    raise ValueError("invalid app-server message identity")
                return value, raw

            async def reply_for(identity):
                while True:
                    value, _ = await receive()
                    if "method" in value:
                        if "id" in value:
                            raise ValueError("unexpected server request before active turn")
                        pending.append(value)
                        continue
                    if value.get("id") != identity or "error" in value or "result" not in value:
                        raise ValueError("app-server setup failed or response identity differs")
                    return value["result"]

            try:
                async with asyncio.timeout(timeout_s):
                    await send(
                        {
                            "id": "initialize",
                            "method": "initialize",
                            "params": {
                                "clientInfo": {"name": "aisle", "version": "1"},
                                "capabilities": {"experimentalApi": True},
                            },
                        }
                    )
                    await reply_for("initialize")
                    await send({"method": "initialized"})
                    await send({"id": "thread", "method": "thread/start", "params": thread_params})
                    thread_id = (await reply_for("thread"))["thread"]["id"]
                    await send(
                        {
                            "id": "turn",
                            "method": "turn/start",
                            "params": {"threadId": thread_id, "input": input_items},
                        }
                    )
                    turn_id = (await reply_for("turn"))["turn"]["id"]
                    scope = AppServerScope(thread_id, turn_id)

                    def usage(value):
                        done = scope.feed(value)
                        if value.get("method") == "thread/tokenUsage/updated":
                            if (
                                token_ceiling is not None
                                and scope.report()["tokens"] >= token_ceiling
                            ):
                                raise ValueError("app-server token budget exhausted")
                        return done

                    for value in pending:
                        usage(value)
                    while True:
                        value, raw = await receive()
                        done = usage(value)
                        if (
                            on_mcp_source is not None
                            and value.get("method") == "item/started"
                            and value.get("params", {}).get("item", {}).get("type") == "mcpToolCall"
                            and value["params"]["item"].get("server") == "aisle_harness"
                        ):
                            params = value["params"]
                            scope._identity(params.get("threadId"), params.get("turnId"))
                            identity = (params["turnId"], params["item"]["id"])
                            if identity in calls or identity in mcp_calls:
                                raise ValueError("app-server MCP source identity replay")
                            on_mcp_source(
                                raw, thread_id=params["threadId"], turn_id=params["turnId"]
                            )
                            mcp_calls.add(identity)
                        if value.get("method") == "item/tool/call":
                            call = scope.parse_call(value)
                            identity = (call["turn_id"], call["call_id"])
                            wire_id = (type(value["id"]), value["id"])
                            if identity in calls or identity in mcp_calls or wire_id in request_ids:
                                raise ValueError("app-server request identity replay")
                            calls.add(identity)
                            request_ids.add(wire_id)
                            result = handle_call(call, raw)
                            if inspect.isawaitable(result):
                                result = await result
                            await send({"id": value["id"], "result": result})
                        elif "id" in value:
                            raise ValueError("unsupported app-server request or response")
                        elif done:
                            completed = {
                                "thread_id": thread_id,
                                "turn_id": turn_id,
                                "dynamic_calls": len(calls),
                                **(
                                    {"mcp_calls": len(mcp_calls)}
                                    if on_mcp_source is not None
                                    else {}
                                ),
                                **(
                                    scope.report()
                                    if token_ceiling is not None or scope.has_usage
                                    else {}
                                ),
                            }
                            return completed
            finally:

                async def reap():
                    if completed is not None:
                        process.stdin.close()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=1)
                        except TimeoutError:
                            pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()

                cleanup = asyncio.create_task(reap())
                cancellation = None
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                cleanup.result()
                if cancellation is not None:
                    raise cancellation
                if completed is not None:
                    completed["rc"] = process.returncode
                    if process.returncode != 0:
                        raise ValueError("app-server did not exit cleanly after turn completion")

    try:
        result = await run()
    except BaseException as exc:
        try:
            retain("failure.json", encode({"error_type": type(exc).__name__, "error": str(exc)}))
        except BaseException as retention_error:
            exc.add_note(f"failure evidence retention failed: {retention_error}")
        if on_reference is not None:
            try:
                on_reference(
                    reference(failure={"error_type": type(exc).__name__, "error": str(exc)})
                )
            except BaseException as retention_error:
                exc.add_note(f"protocol reference retention failed: {retention_error}")
        raise
    if on_reference is not None:
        on_reference(reference())
    return {
        **result,
        "artifacts": dict(hashes),
        "complete_coverage": False,
        "confinement_verified": False,
        "wall_s": time.monotonic() - started,
        "stopped": "agent_done",
        "stream_complete": True,
    }
