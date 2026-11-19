"""Controller-owned T4 dialogue and fixed-library request boundary.

This module does not launch a robot or attest graph/guard wiring. A live adapter
must bind its transport to reviewed public inputs and guarded fixed skills, own
the outer deadline, and keep private scoring separate. Authored code gets only
the framed call protocol, never these Python objects or the controller runtime.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


class TaskRefusal(ValueError):
    """An authored request exceeds its current task capability."""


def _require(condition, message, error=ValueError):
    if not condition:
        raise error(message)


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) is not None


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


class TaskBoundary:
    """Public lifecycle and one outstanding motion, independent of private verdicts.

    The controller supplies a fresh binding_id for each worker incarnation. The
    fixed library owns motion; requests grant no actuator or reset authority.
    Correctness of choosing a medicine or returning the old one is left to the
    worker and independent scorer, not silently supplied by this boundary.
    """

    def __init__(self, *, binding_id, medicines, max_motion_requests):
        _require(_identifier(binding_id), "invalid worker binding")
        _require(
            isinstance(medicines, (list, tuple))
            and bool(medicines)
            and all(_identifier(m) for m in medicines)
            and len(set(medicines)) == len(medicines),
            "invalid fixed medicine catalog",
        )
        _require(
            type(max_motion_requests) is int and 1 <= max_motion_requests <= 100,
            "invalid motion request budget",
        )
        self.binding_id, self.medicines = binding_id, frozenset(medicines)
        self.max_motion_requests, self.motion_requests = max_motion_requests, 0
        self.epoch = self.revision_serial = 0
        self.last_reset_seq = self.last_human_seq = -1
        self.reset_sim_ns = -1
        self.seen_goals = set()
        self.goal_id = self.revision = self.request = None
        self.active = self.active_metadata = None
        self.confirmed = False
        self.confirm_requested = None

    def snapshot(self):
        return copy.deepcopy(
            dict(
                revision=self.revision,
                request=self.request,
                confirmed=self.confirmed,
                active=self.active,
            )
        )

    def observe(self, topic, payload, metadata):
        """Project only explicit public fields; unknown wiring fails closed."""
        _require(
            topic in {"reset_done", "human_msg", "plan_done", "violation", "tick"},
            "undeclared task input",
        )
        _require(
            isinstance(metadata, dict)
            and type(metadata.get("env_id")) is int
            and metadata["env_id"] == 0
            and all(
                type(metadata.get(k)) is int and metadata[k] >= 0 for k in ("seq", "sim_time_ns")
            ),
            "invalid task input metadata",
        )
        seq = metadata["seq"]
        if topic == "reset_done":
            if seq <= self.last_reset_seq:
                return None
            _require(
                isinstance(payload, list)
                and len(payload) == 1
                and type(payload[0]) is int
                and payload[0] == 1,
                "invalid reset completion",
            )
            self.last_reset_seq = seq
            self.reset_sim_ns = metadata["sim_time_ns"]
            self.epoch += 1
            self.goal_id = self.revision = self.request = None
            self.active = self.active_metadata = None
            self.confirmed, self.confirm_requested = False, None
            self.seen_goals.clear()
            return dict(kind="reset", generation=self.epoch, state=self.snapshot())
        _require(self.epoch > 0, "task input precedes reset")
        if metadata["sim_time_ns"] < self.reset_sim_ns:
            return None
        if topic == "human_msg":
            if seq <= self.last_human_seq:
                return None
            self.last_human_seq = seq
            return self._human(payload, metadata.get("goal_id"))
        if topic == "plan_done":
            if self.active is None or any(
                metadata.get(k) != v for k, v in self.active_metadata.items()
            ):
                return None
            _require(
                isinstance(payload, dict) and set(payload) == {"ok"} and payload["ok"] is True,
                "invalid motion completion",
            )
            invocation = self.active["motion_id"]
            self.active = self.active_metadata = None
            return dict(
                kind="motion_returned",
                motion_id=invocation,
                physical_completion="unknown",
                state=self.snapshot(),
            )
        if topic == "violation":
            _require(
                isinstance(payload, dict) and _identifier(payload.get("reason")),
                "invalid guard violation",
            )
            return dict(kind="guard_violation", reason=payload["reason"], state=self.snapshot())
        return dict(kind="tick", sim_time_ns=metadata["sim_time_ns"], state=self.snapshot())

    def _human(self, payload, goal):
        _require(isinstance(goal, str) and 0 < len(goal) <= 256, "invalid dialogue goal identity")
        _require(
            isinstance(payload, dict)
            and payload.get("kind") in {"request", "correction", "confirm_reply"}
            and isinstance(payload.get("med"), str)
            and payload["med"] in self.medicines
            and isinstance(payload.get("text"), str)
            and len(payload["text"]) <= 2048,
            "invalid public dialogue",
        )
        kind, med = payload["kind"], payload["med"]
        if kind == "request":
            if goal in self.seen_goals:
                return None
            return_med = payload.get("return_med")
            _require(
                return_med is None or isinstance(return_med, str) and return_med in self.medicines,
                "invalid public return medicine",
            )
            self.seen_goals.add(goal)
            self.goal_id = goal
            self.request = dict(med=med, text=payload["text"], return_med=return_med)
        elif goal != self.goal_id:
            return None
        elif kind == "correction":
            _require(not self.confirmed, "correction follows confirmed motion task")
            self.request.update(med=med, text=payload["text"])
        else:
            if self.confirmed or self.confirm_requested != med or self.request["med"] != med:
                return None
            self.confirmed = True
            return dict(kind=kind, med=med, text=payload["text"], state=self.snapshot())
        self.revision_serial += 1
        self.revision = f"revision-{self.epoch}-{self.revision_serial}"
        self.confirmed, self.confirm_requested = False, None
        return dict(kind=kind, med=med, text=payload["text"], state=self.snapshot())

    def action(self, arguments):
        _require(
            isinstance(arguments, dict) and set(arguments) == {"revision", "kind", "med"},
            "invalid action fields",
            TaskRefusal,
        )
        _require(
            self.revision is not None and arguments["revision"] == self.revision,
            "stale task revision",
            TaskRefusal,
        )
        kind, med = arguments["kind"], arguments["med"]
        _require(
            isinstance(kind, str) and kind in {"confirm", "deliver", "return"},
            "undeclared action",
            TaskRefusal,
        )
        _require(isinstance(med, str) and med in self.medicines, "unknown medicine", TaskRefusal)
        if kind == "confirm":
            _require(not self.confirmed, "task is already confirmed", TaskRefusal)
            self.confirm_requested = med
            return dict(
                topic="robot_msg",
                payload=dict(kind="confirm", med=med, text=f"you want the {med}, correct?"),
                metadata=dict(goal_id=self.goal_id),
                motion_id=None,
            )
        _require(self.confirmed, "task is not confirmed", TaskRefusal)
        _require(self.active is None, "fixed library is busy", TaskRefusal)
        _require(
            self.motion_requests < self.max_motion_requests,
            "motion request budget exhausted",
            TaskRefusal,
        )
        self.motion_requests += 1
        invocation = f"motion-{self.epoch}-{self.motion_requests}"
        self.active = dict(motion_id=invocation, revision=self.revision, kind=kind, med=med)
        self.active_metadata = dict(
            goal_id=self.goal_id,
            request_id=f"{self.binding_id}/{self.epoch}/{self.motion_requests}",
        )
        return dict(
            topic="target_request" if kind == "deliver" else "return_request",
            payload={"target_med" if kind == "deliver" else "return_med": med},
            metadata=dict(self.active_metadata),
            motion_id=invocation,
        )


class TaskSession:
    """Retained task RPC joined to the existing arm-bound evidence dispatcher.

    transport.next_event() supplies (topic, payload, metadata), or StopIteration;
    transport.send() receives only controller-produced fixed-library requests.
    Both methods are trusted host operations and need an outer deadline in any
    live adapter. The supplied transport must never be visible to authored code.
    """

    def __init__(
        self,
        *,
        evidence,
        transport,
        output,
        source_sha256,
        medicines,
        max_calls=64,
        max_motion_requests=16,
    ):
        _require(
            isinstance(source_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha256),
            "invalid bound source hash",
        )
        _require(type(max_calls) is int and 1 <= max_calls <= 1000, "invalid task call budget")
        self.boundary = TaskBoundary(
            binding_id=evidence.session_id,
            medicines=medicines,
            max_motion_requests=max_motion_requests,
        )
        self.evidence, self.transport = evidence, transport
        self.output, self.source_sha256 = Path(output), source_sha256
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_calls, self.calls, self.refused = max_calls, 0, 0
        self.exhausted = self.closed = False
        self.failure = None
        (self.output / "binding.json").write_text(
            _json(
                dict(
                    session_id=evidence.session_id,
                    arm=evidence.arm,
                    source_sha256=source_sha256,
                    medicines=list(medicines),
                    max_calls=max_calls,
                    max_motion_requests=max_motion_requests,
                )
            )
        )
        self.log = (self.output / "events.jsonl").open("x")

    def _retain(self, value):
        self.log.write(_json(value))
        self.log.flush()

    def call(self, request):
        _require(not self.closed and not self.exhausted, "task session is closed or exhausted")
        try:
            return self._call(request)
        except Exception as exc:
            self.failure, self.closed = str(exc), True
            raise

    def _call(self, request):
        _require(self.calls < self.max_calls, "task call budget exhausted")
        self.calls += 1
        try:
            encoded = _json(request)
            retained = json.loads(encoded) if len(encoded.encode()) <= 4096 else {"invalid": True}
        except (ValueError, TypeError, RecursionError):
            retained = {"invalid": True}
        self._retain(dict(kind="request", call=self.calls, request=retained))
        self.evidence._verify_bundle()
        try:
            _require(
                isinstance(request, dict)
                and retained != {"invalid": True}
                and set(request) == {"operation", "arguments"},
                "invalid request",
                TaskRefusal,
            )
            operation, arguments = request["operation"], request["arguments"]
            _require(
                isinstance(operation, str) and isinstance(arguments, dict),
                "invalid request types",
                TaskRefusal,
            )
            if operation == "next":
                _require(not arguments, "next takes no arguments", TaskRefusal)
                event = None
                # Bound ignored stale traffic as well as authored calls.
                for _ in range(1024):
                    try:
                        raw = self.transport.next_event()
                    except StopIteration:
                        self.exhausted = True
                        break
                    self._retain(dict(kind="input", event=raw))
                    event = self.boundary.observe(*raw)
                    if event is not None:
                        break
                else:
                    raise ValueError("task input budget exhausted")
                response = dict(ok=True, event=event)
            elif operation == "act":
                command = self.boundary.action(arguments)
                self._retain(dict(kind="proposal", command=command))
                self.transport.send(command["topic"], command["payload"], command["metadata"])
                self._retain(dict(kind="dispatched", command=command))
                response = dict(
                    ok=True, motion_id=command["motion_id"], state=self.boundary.snapshot()
                )
            else:
                response = self.evidence.call(request)
        except TaskRefusal:
            self.refused += 1
            response = dict(ok=False, error="task_request_refused")
        self._retain(dict(kind="response", call=self.calls, response=response))
        return response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        error = str(exc) if exc is not None else self.failure
        if error is None and not self.exhausted:
            error = "worker ended before task input exhaustion"
        self.log.close()
        (self.output / "summary.json").write_text(
            _json(
                dict(
                    ok=error is None,
                    error=error,
                    source_sha256=self.source_sha256,
                    calls=self.calls,
                    refused_calls=self.refused,
                    input_exhausted=self.exhausted,
                    motion_requests=self.boundary.motion_requests,
                    active=self.boundary.snapshot()["active"],
                    measured_sessions=0,
                    provider_calls=0,
                    guard_traversal_verified=False,
                )
            )
        )
        if exc is None and error:
            raise ValueError(error)
