"""Executor-observed lifecycle and command provenance (HAR-4, TC-6/7).

No task verdict is accepted here. A successful transport send establishes only
command emission; an executor returning establishes no physical completion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def stamp(metadata):
    for key in ("sim_time_ns", "env_id", "seq"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"capture requires explicit nonnegative {key}")
    return {key: metadata[key] for key in ("sim_time_ns", "env_id", "seq")}


def reference(endpoint, value, metadata):
    row = stamp(metadata)
    textual = pa.types.is_string(value.type) or pa.types.is_large_string(value.type)
    row.update(
        data=None if textual else [float(x) for x in value.to_pylist()],
        text=value[0].as_py() if textual else None,
    )
    # Same canonical payload as the recorder's five Arrow columns, no newline.
    digest = hashlib.sha256(encoded(row).rstrip(b"\n")).hexdigest()
    return dict(endpoint=endpoint, **stamp(metadata), trace_row_sha256=digest)


class ExecutionCapture:
    def __init__(self, path, *, capture_id, node_id, inputs, skill):
        if not all(
            isinstance(x, str) and x
            for x in (capture_id, node_id, skill.get("name"), skill.get("version"))
        ):
            raise ValueError("capture requires explicit run/node/skill identity")
        if not isinstance(inputs, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and "/" in v for k, v in inputs.items()
        ):
            raise ValueError("capture requires declared input bindings")
        self.path = Path(path)
        self.capture_id, self.node_id, self.inputs, self.skill = capture_id, node_id, inputs, skill
        self.episode, self.env_id = -1, None
        self.reset_source = None
        self.active = None
        self.proposals = {}
        self.observation, self.stage = None, None
        self.event_seq, self.invocations = 0, 0
        self.digest = hashlib.sha256()
        self.stream = self.path.open("xb")
        self._write(
            dict(
                format="aisle_execution_capture",
                version=1,
                capture_id=capture_id,
                node_id=node_id,
                inputs=inputs,
                skill=skill,
            )
        )

    def _write(self, record):
        raw = encoded(record)
        self.stream.write(raw)
        self.stream.flush()
        self.digest.update(raw)

    def _event(self, kind, metadata, invocation=None, **payload):
        clock = stamp(metadata)
        if self.env_id is not None and clock["env_id"] != self.env_id:
            raise ValueError("capture does not mix environments")
        proposal = self.proposals.get(invocation, {})
        self._write(
            dict(
                kind=kind,
                event_seq=self.event_seq,
                sim_time_ns=clock["sim_time_ns"],
                env_id=clock["env_id"],
                episode=self.episode,
                invocation_id=invocation,
                goal_revision=proposal.get("goal_revision"),
                **payload,
            )
        )
        self.event_seq += 1

    def _source(self, event):
        if event["id"] not in self.inputs:
            raise ValueError("capture input has no source binding")
        source = reference(self.inputs[event["id"]], event["value"], event["metadata"])
        if self.env_id is not None and source["env_id"] != self.env_id:
            raise ValueError("capture does not mix environments")
        return source

    def reset(self, event):
        if event["value"].to_pylist() != [1]:
            raise ValueError("capture reset must be a successful boundary")
        source = self._source(event)
        if self.env_id is not None and source["env_id"] != self.env_id:
            raise ValueError("capture does not mix environments")
        if source == self.reset_source:
            return
        if self.reset_source is None and source["sim_time_ns"] != 0:
            raise ValueError("capture missed initial reset")
        if self.reset_source and source["sim_time_ns"] <= self.reset_source["sim_time_ns"]:
            raise ValueError("capture received stale reset")
        if self.active:
            self._event(
                "interrupted" if self.proposals[self.active]["started"] else "cancelled",
                event["metadata"],
                self.active,
                reason="reset",
                source=source,
                delivery_completion="unknown",
            )
        for invocation, proposal in self.proposals.items():
            if invocation == self.active:
                proposal["state"] = "interrupted" if proposal["started"] else "cancelled"
            elif proposal["state"] == "proposed":
                self._event(
                    "cancelled",
                    event["metadata"],
                    invocation,
                    reason="reset",
                    source=source,
                    delivery_completion="unknown",
                )
                proposal["state"] = "cancelled"
        self.active = None
        self.episode += 1
        self.env_id = source["env_id"]
        self.reset_source = source
        self.observation, self.stage = None, None
        self._event("reset", event["metadata"], source=source)

    def propose(self, event, parameters):
        if self.episode < 0:
            raise ValueError("capture requires initial reset before proposals")
        source = self._source(event)
        if source["sim_time_ns"] < self.reset_source["sim_time_ns"]:
            raise ValueError("capture request predates reset")
        invocation = f"{self.capture_id}:{self.node_id}:{self.episode}:{self.invocations}"
        self.invocations += 1
        goal = event["metadata"].get("goal_id")
        goal = goal if isinstance(goal, str) and goal.strip() else None
        self.proposals[invocation] = dict(
            goal_revision=goal,
            source=source,
            started=False,
            metadata=dict(event["metadata"]),
            state="proposed",
        )
        self._event(
            "proposed",
            event["metadata"],
            invocation,
            source=source,
            skill=self.skill,
            parameters=parameters,
            delivery_completion="unknown",
        )
        return invocation

    def accept(self, invocation):
        proposal = self.proposals[invocation]
        if self.active is not None or proposal["state"] != "proposed":
            raise ValueError("capture cannot accept overlapping or reused invocation")
        self.active = invocation
        proposal["state"] = "accepted"
        self._event("accepted", proposal["metadata"], invocation)

    def reject(self, invocation, reason):
        proposal = self.proposals[invocation]
        if proposal["state"] != "proposed":
            raise ValueError("capture can only reject an unaccepted proposal")
        proposal["state"] = "rejected"
        self._event("rejected", proposal["metadata"], invocation, reason=reason)

    def observe(self, event, *, stage=None):
        self.observation = self._source(event)
        self.stage = stage

    def command_metadata(self, topic, metadata):
        if self.active is None or topic not in {"joint_cmd", "gripper_cmd"}:
            return dict(metadata)
        stamp(metadata)
        proposal = self.proposals[self.active]
        extra = dict(
            invocation_id=self.active,
            skill_name=self.skill["name"],
            skill_version=self.skill["version"],
            command_origin_seq=metadata["seq"],
            command_origin_endpoint=f"{self.node_id}/{topic}",
        )
        if proposal["goal_revision"] is not None:
            extra["goal_id"] = proposal["goal_revision"]
        clean = dict(metadata)
        clean.pop("goal_id", None)
        return clean | extra

    def emitted(self, topic, value, metadata):
        if self.active is None or topic not in {"joint_cmd", "gripper_cmd"}:
            return
        proposal = self.proposals[self.active]
        source = reference(f"{self.node_id}/{topic}", value, metadata)
        if not proposal["started"]:
            self._event(
                "started",
                metadata,
                self.active,
                source=source,
                observation=self.observation,
                delivery_completion="unknown",
            )
            proposal["started"] = True
        self._event(
            "command_emitted",
            metadata,
            self.active,
            source=source,
            observation=self.observation,
            stage=self.stage,
        )

    def finish(self, metadata):
        if self.active is None:
            return
        self._event(
            "returned",
            metadata,
            self.active,
            observation=self.observation,
            delivery_completion="unknown",
        )
        self.proposals[self.active]["state"] = "returned"
        self.active = None

    def close(self):
        if self.stream.closed:
            return
        footer = dict(
            kind="capture_closed",
            events=self.event_seq,
            open_invocation=self.active,
            open_proposals=[k for k, p in self.proposals.items() if p["state"] == "proposed"],
            sha256=self.digest.hexdigest(),
        )
        self.stream.write(encoded(footer))
        self.stream.close()


class RecordedTransport:
    """Observe the actual raw send AFTER the lockstep wrapper stamps its output."""

    def __init__(self, raw, capture):
        self.raw, self.capture = raw, capture

    def send_output(self, topic, value, metadata=None):
        metadata = self.capture.command_metadata(topic, metadata or {})
        result = self.raw.send_output(topic, value, metadata)
        self.capture.emitted(topic, value, metadata)
        return result

    def __iter__(self):
        try:
            yield from self.raw
        finally:
            self.capture.close()


def read_capture(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("capture exceeds 512 MiB")
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < 2 or not all(line.endswith(b"\n") for line in lines):
        raise ValueError("capture is not closed")
    header, footer = json.loads(lines[0]), json.loads(lines[-1])
    if not isinstance(header, dict) or not isinstance(footer, dict):
        raise ValueError("capture records must be objects")
    if footer.get("kind") != "capture_closed":
        raise ValueError("capture is not closed")
    if hashlib.sha256(b"".join(lines[:-1])).hexdigest() != footer.get("sha256"):
        raise ValueError("capture digest mismatch")
    if header.get("format") != "aisle_execution_capture" or header.get("version") != 1:
        raise ValueError("unsupported execution capture")
    events = [json.loads(line) for line in lines[1:-1]]
    if not all(isinstance(e, dict) for e in events):
        raise ValueError("capture events must be objects")
    if footer.get("events") != len(events) or [e.get("event_seq") for e in events] != list(
        range(len(events))
    ):
        raise ValueError("capture event sequence mismatch")
    validate_lifecycle(header, events, footer)
    return dict(
        header=header,
        events=events,
        open_invocation=footer["open_invocation"],
        open_proposals=footer["open_proposals"],
        source_sha256=hashlib.sha256(b"".join(lines)).hexdigest(),
    )


def captured_node(environ):
    """Opt-in transport observation; the harness supplies the sealed graph binding."""
    from aisle.turn_node import Node

    config = environ.get("AISLE_EXECUTION_CAPTURE_CONFIG")
    if config is None:
        return Node(), None
    from dora import Node as DoraNode

    raw = DoraNode()
    capture = ExecutionCapture(**json.loads(config))
    return Node(raw_node=RecordedTransport(raw, capture)), capture


def bind_executors(doc, root, trace_dir, capture_id):
    """Bind only the two known executor entry points; unknown controllers stay uncaptured."""
    root, trace_dir = Path(root).resolve(), Path(trace_dir).resolve()
    base = "src/aisle/nodes/ik_trajectory.py"
    entries = {base: "ik-trajectory", "skills/ik-transfer-v2/ik_transfer_v2.py": "ik-transfer-v2"}
    for node in doc["nodes"]:
        entry = next((p for p in entries if Path(node["path"]).resolve() == root / p), None)
        if entry is None:
            continue
        if Path(node["id"]).name != node["id"] or node["id"] in {".", ".."}:
            raise RuntimeError("invalid execution capture node identity")
        sources = {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted({base, entry})
        }
        skill = dict(
            name=entries[entry],
            version="sha256:" + hashlib.sha256(encoded(sources)).hexdigest(),
            sources=sources,
        )
        inputs = {
            port: binding["source"] if isinstance(binding, dict) else binding
            for port, binding in node.get("inputs", {}).items()
        }
        directory = trace_dir / "executions"
        directory.mkdir(parents=True, exist_ok=True)
        config = dict(
            path=str(directory / f"{node['id']}.jsonl"),
            capture_id=capture_id,
            node_id=node["id"],
            inputs=inputs,
            skill=skill,
        )
        node["env"] = {
            **node.get("env", {}),
            "AISLE_EXECUTION_CAPTURE_CONFIG": encoded(config).decode().strip(),
        }


def validate_lifecycle(header, events, footer):
    """Check executor-local order; this is not authentication or verification against traces."""
    states, goals = {}, {}
    active, env, episode = None, None, -1
    transitions = {
        "accepted": {"proposed"},
        "rejected": {"proposed"},
        "started": {"accepted"},
        "command_emitted": {"started"},
        "returned": {"accepted", "started"},
        "interrupted": {"started"},
        "cancelled": {"proposed", "accepted"},
    }
    for seq, event in enumerate(events):
        if (
            type(event.get("event_seq")) is not int
            or event["event_seq"] != seq
            or type(event.get("env_id")) is not int
            or event["env_id"] < 0
            or type(event.get("sim_time_ns")) is not int
            or event["sim_time_ns"] < 0
            or type(event.get("episode")) is not int
        ):
            raise ValueError("invalid capture lifecycle clock")
        if env is not None and event["env_id"] != env:
            raise ValueError("capture lifecycle mixes environments")
        env = event["env_id"]
        kind, invocation = event.get("kind"), event.get("invocation_id")
        if kind == "reset":
            if (
                active is not None
                or "proposed" in states.values()
                or invocation is not None
                or event["episode"] != episode + 1
                or event.get("goal_revision") is not None
            ):
                raise ValueError("invalid reset lifecycle")
            episode += 1
        elif episode < 0 or event["episode"] != episode:
            raise ValueError("capture lifecycle lacks reset")
        elif kind == "proposed":
            if (
                not isinstance(invocation, str)
                or not invocation
                or invocation in states
                or event.get("skill") != header.get("skill")
                or not isinstance(event.get("parameters"), dict)
            ):
                raise ValueError("invalid proposal lifecycle")
            states[invocation] = "proposed"
            goals[invocation] = event.get("goal_revision")
        else:
            if kind not in transitions or states.get(invocation) not in transitions[kind]:
                raise ValueError("invalid invocation lifecycle transition")
            if event.get("goal_revision") != goals[invocation]:
                raise ValueError("capture lifecycle changes invocation goal")
            if kind == "accepted":
                if active is not None:
                    raise ValueError("overlapping invocation lifecycle")
                active = invocation
            elif kind in {"started", "command_emitted", "returned", "interrupted"}:
                if active != invocation:
                    raise ValueError("inactive invocation lifecycle")
            if kind in {"returned", "interrupted", "cancelled"} and active == invocation:
                active = None
            if kind != "command_emitted":
                states[invocation] = kind
        if kind in {"reset", "proposed", "started", "command_emitted"}:
            source = event.get("source")
            if not isinstance(source, dict) or not source.get("trace_row_sha256"):
                raise ValueError("capture lifecycle lacks source reference")
        if "delivery_completion" in event and event["delivery_completion"] != "unknown":
            raise ValueError("executor lifecycle cannot establish delivery")
    if footer.get("open_invocation") != active or footer.get("open_proposals") != [
        k for k, state in states.items() if state == "proposed"
    ]:
        raise ValueError("capture lifecycle footer mismatch")


def verify_capture(path, trace_dir, config, *, guard_endpoints):
    """Join executor claims to raw rows and explicit guard ancestry.

    ``config`` and guard endpoints must come from the controller's retained
    instrumented graph. This checks consistency, not authenticity against an
    adversary able to rewrite both the graph and its evidence. No verdicts or
    scorer goals are read, and forwarding does not establish physical delivery.
    """
    import re

    result = read_capture(path)
    if any(
        result["header"].get(k) != config.get(k)
        for k in ("capture_id", "node_id", "inputs", "skill")
    ):
        raise ValueError("capture does not match instrument binding")
    trace_dir = Path(trace_dir)
    allowed = (
        set(config["inputs"].values())
        | {f"{config['node_id']}/joint_cmd", f"{config['node_id']}/gripper_cmd"}
        | set(guard_endpoints)
    )
    rows, indexes = {}, {}

    def load(endpoint):
        if endpoint not in allowed or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", endpoint
        ):
            raise ValueError("unbound trace source endpoint")
        if endpoint not in rows:
            source = trace_dir / (endpoint.replace("/", "__") + ".arrow")
            if not source.is_file() or source.stat().st_size > 512 * 1024 * 1024:
                raise ValueError("missing or oversized trace source")
            with pa.memory_map(str(source), "r") as stream:
                table = pa.ipc.open_stream(stream).read_all()
            if table.column_names != ["sim_time_ns", "env_id", "seq", "data", "text"]:
                raise ValueError("unexpected trace source schema")
            rows[endpoint] = table.to_pylist()
            indexes[endpoint] = {}
            for index, row in enumerate(rows[endpoint]):
                key = (row["sim_time_ns"], row["env_id"], row["seq"])
                if key in indexes[endpoint]:
                    raise ValueError("ambiguous trace source identity")
                indexes[endpoint][key] = index
        return rows[endpoint]

    def resolve(ref):
        if not isinstance(ref, dict):
            raise ValueError("missing trace source reference")
        endpoint = ref.get("endpoint")
        data = load(endpoint)
        index = indexes[endpoint].get(tuple(ref.get(k) for k in ("sim_time_ns", "env_id", "seq")))
        if index is None or hashlib.sha256(
            encoded(data[index]).rstrip(b"\n")
        ).hexdigest() != ref.get("trace_row_sha256"):
            raise ValueError("trace source reference mismatch")
        return endpoint, index

    commands = {}
    command_rows = {}
    proposal_rows = {}
    for event in result["events"]:
        for field in ("source", "observation"):
            if event.get(field) is not None:
                resolve(event[field])
        if event.get("source") is not None and any(
            event["source"].get(k) != event[k] for k in ("sim_time_ns", "env_id")
        ):
            raise ValueError("lifecycle clock does not match source clock")
        observation = event.get("observation")
        if observation is not None and (
            observation["endpoint"] != config["inputs"].get("joint_state")
            or observation["env_id"] != event["env_id"]
            or observation["sim_time_ns"] > event["sim_time_ns"]
        ):
            raise ValueError("observation source clock or binding mismatch")
        if event["kind"] == "proposed":
            proposal_rows[resolve(event["source"])] = event
        if event["kind"] == "command_emitted":
            ref = event["source"]
            if ref["endpoint"] not in {
                f"{config['node_id']}/joint_cmd",
                f"{config['node_id']}/gripper_cmd",
            }:
                raise ValueError("command source is not bound executor")
            key = (event["invocation_id"], ref["endpoint"], ref["seq"])
            if key in commands:
                raise ValueError("duplicate emitted command source")
            commands[key] = event
            command_rows[resolve(ref)] = event
    sidecar = trace_dir / "identities.jsonl"
    if not sidecar.is_file() or sidecar.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("missing or oversized trace identity source")
    seen, witnessed, forwarded = set(), set(), []
    proposal_witnesses = set()
    endpoints = {e for e, _ in command_rows} | {e for e, _ in proposal_rows} | set(guard_endpoints)
    invocation_ids = {e["invocation_id"] for e in commands.values()}
    for line in sidecar.read_text().splitlines():
        identity = json.loads(line)
        endpoint = identity["endpoint"].replace("__", "/")
        if endpoint not in endpoints:
            continue
        data = load(endpoint)
        index = identity["row"]
        if type(index) is not int or not 0 <= index < len(data) or (endpoint, index) in seen:
            raise ValueError("invalid trace identity row")
        seen.add((endpoint, index))
        row = data[index]
        row_hash = hashlib.sha256(encoded(row).rstrip(b"\n")).hexdigest()
        if identity.get("trace_row_sha256") != row_hash:
            raise ValueError("trace identity source hash mismatch")
        meta = identity["metadata"]
        if (endpoint, index) in proposal_rows:
            goal = meta.get("goal_id")
            goal = goal if isinstance(goal, str) and goal.strip() else None
            if goal != proposal_rows[endpoint, index]["goal_revision"]:
                raise ValueError("proposal source goal mismatch")
            proposal_witnesses.add((endpoint, index))
        key = (
            meta.get("invocation_id"),
            meta.get("command_origin_endpoint"),
            meta.get("command_origin_seq"),
        )
        if (endpoint, index) in command_rows:
            event = command_rows[endpoint, index]
            if commands.get(key) != event:
                raise ValueError("emitted command identity mismatch")
            witnessed.add((endpoint, index))
        elif endpoint in guard_endpoints and key[0] in invocation_ids:
            event = commands.get(key)
            if (
                event is None
                or row["env_id"] != event["env_id"]
                or row["sim_time_ns"] < event["sim_time_ns"]
            ):
                raise ValueError("guard ancestry source mismatch")
            forwarded.append(
                dict(
                    invocation_id=key[0],
                    origin=event["source"],
                    source=dict(
                        endpoint=endpoint,
                        trace_row_sha256=row_hash,
                        **{k: row[k] for k in ("sim_time_ns", "env_id", "seq")},
                    ),
                )
            )
    if proposal_witnesses != set(proposal_rows):
        raise ValueError("proposal lacks captured goal identity")
    if witnessed != set(command_rows):
        raise ValueError("emitted command lacks captured identity")
    return {**result, "forwarded": forwarded}
