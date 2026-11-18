"""Experimental PEAR-style Dora policy; every command requires budget-guard (ADR-74)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aisle.kinematics import _rpy_rotation
from aisle.nodes.budget_guard import (
    fingers_to_gripper,
    fk_flange,
    gripper_to_fingers,
    load_limits,
)
from aisle.nodes.ik_trajectory import fk_tcp, ik_continuation
from aisle.nodes.pear_backend import (
    ActionDelta,
    Frame,
    Observation,
    build_messages,
    integer,
    joint_vector,
    parse_action,
    rotation_to_rpy,
)


def trace_metadata(metadata) -> dict:
    """Project contract fields; Dora also supplies non-JSON native timestamps."""
    fields = {"sim_time_ns", "env_id", "seq", "turn_epoch", "turn_id", "goal_id", "request_id"}
    return {
        key: value for key, value in metadata.items() if key in fields and type(value) in (int, str)
    }


class Motion:
    """Small joint steps, measured settling, then gripper motion; no task verdict."""

    def __init__(self, path: list[np.ndarray], grip: float, start_ns: int):
        if not path:
            raise ValueError("empty motion path")
        self.path, self.grip, self.start_ns = path, float(grip), start_ns
        self.limits = load_limits("franka")
        self.index, self.settled, self.ticks = 0, 0, 0
        self.command = None
        self.current_grip = None
        self.phase, self.feedback, self.done = "arm", "executing", False

    def step(self, q, now_ns: int):
        q = joint_vector(q)
        if self.done:
            return None, None
        if now_ns < self.start_ns or now_ns - self.start_ns > 10_000_000_000:
            self.done, self.feedback = True, "motion_timeout"
            return None, None
        if self.command is None:
            self.command = q[:7].copy()
            self.current_grip = float(np.clip(fingers_to_gripper(q, self.limits), 0, 1))
        self.ticks += 1
        grip_output = None
        if self.phase == "arm":
            target = self.path[self.index]
            self.command += np.clip(target - self.command, -0.002, 0.002)
            self.command = np.clip(self.command, q[:7] - 0.03, q[:7] + 0.03)
            self.settled = self.settled + 1 if np.max(np.abs(target - q[:7])) < 0.005 else 0
            if self.settled >= 3 and np.max(np.abs(target - self.command)) < 1e-6:
                self.index += 1
                self.settled = 0
                if self.index == len(self.path):
                    self.phase = "gripper"
        else:
            if self.ticks % 4 == 0:  # 25 Hz at the joint_state contract rate
                self.current_grip += float(np.clip(self.grip - self.current_grip, -0.01, 0.01))
                grip_output = self.current_grip
            measured = fingers_to_gripper(q, self.limits)
            reached = abs(self.current_grip - self.grip) < 1e-6 and abs(measured - self.grip) < 0.04
            self.settled = self.settled + 1 if reached else 0
            if self.settled >= 3:
                self.done, self.feedback = True, "reached"
        fingers = gripper_to_fingers(self.current_grip, self.limits)
        return np.concatenate([self.command, fingers]).astype(np.float32), grip_output


def plan_motion(q, action: ActionDelta, now_ns: int) -> Motion:
    """Only known robot kinematics and measured state; no scene/object queries."""
    q = joint_vector(q)
    values = (*action.position_cm, *action.rpy_rad, action.gripper_cm)
    if len(values) != 7 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("malformed Cartesian delta")
    if (
        math.hypot(*action.position_cm) > 5
        or math.hypot(*action.rpy_rad) > 0.35
        or abs(action.gripper_cm) > 2
    ):
        raise ValueError("Cartesian delta exceeds bounds")
    limits = load_limits("franka")
    if np.any(q < limits.q_min_arr - 1e-6) or np.any(q > limits.q_max_arr + 1e-6):
        raise ValueError("measured joints are outside the robot limits")
    start = fk_tcp(q[:7])
    angles = rotation_to_rpy(fk_flange(q[:7])[1])
    delta = np.asarray(action.position_cm) / 100
    delta_angles = np.asarray(action.rpy_rad)
    path, previous, arm = [], start, q[:7]
    count = max(
        1, math.ceil(np.linalg.norm(delta) / 0.01), math.ceil(np.linalg.norm(delta_angles) / 0.07)
    )
    if not np.any(delta) and not np.any(delta_angles):
        path = [arm.copy()]
    else:
        for index in range(1, count + 1):
            position = start + delta * index / count
            rotation = _rpy_rotation(angles + delta_angles * index / count)
            segment = ik_continuation(previous, position, rotation, arm, step_m=0.01)
            if not segment:
                raise ValueError("IK rejected the requested pose")
            for waypoint in segment:
                flange = fk_flange(waypoint)[0]
                if (
                    not np.isfinite(waypoint).all()
                    or np.any(waypoint < limits.q_min_arr[:7])
                    or np.any(waypoint > limits.q_max_arr[:7])
                    or np.any(flange < limits.workspace_min)
                    or np.any(flange > limits.workspace_max)
                ):
                    raise ValueError("IK path exceeds joint/workspace bounds")
            path.extend(segment)
            previous, arm = position, segment[-1]
    width = float(np.clip(sum(q[7:]) + action.gripper_cm / 100, 0, 0.08))
    return Motion(path, 1 - width / 0.08, now_ns)


@dataclass(frozen=True)
class Ticket:
    epoch: int
    number: int
    goal_id: str
    observation: Observation
    messages: list


class FrankaAdapter:
    cameras = ("overhead", "wrist")
    joint_vector = staticmethod(joint_vector)
    build_messages = staticmethod(build_messages)
    plan_motion = staticmethod(plan_motion)
    parse_action = staticmethod(parse_action)
    description = {"embodiment": "franka"}

    def validate_metadata(self, event_id, metadata):
        pass

    def reset(self):
        pass


class Controller:
    """Pure lifecycle core. Slow model work lives outside it; SIM time is injected."""

    def __init__(self, env_id: int = 0, cameras=None, adapter=None):
        self.adapter = adapter or FrankaAdapter()
        self.env_id = integer(env_id, "env_id")
        self.cameras = tuple(cameras or self.adapter.cameras)
        if not 1 <= len(self.cameras) <= 3 or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("one to three distinct camera roles required")
        self.epoch = 0
        self.reset()

    def reset(self):
        self.adapter.reset()
        self.epoch += 1
        self.frames, self.joints, self.joint_ns = {}, None, 0
        self.instruction, self.goal_id = None, None
        self.history, self.motion, self.pending = [], None, None
        self.number, self.last_obs_ns, self.last_request_ns = 0, -1, -(10**18)
        self.feedback = "initial observation"

    def set_goal(self, instruction: str, goal_id: str):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 8192:
            raise ValueError("instruction must be nonempty bounded text")
        if not isinstance(goal_id, str) or not goal_id:
            raise ValueError("goal_id required")
        if (instruction, goal_id) == (self.instruction, self.goal_id):
            return
        self.epoch += 1
        self.instruction, self.goal_id = instruction, goal_id
        self.history, self.motion, self.pending = [], None, None
        self.number, self.last_obs_ns, self.last_request_ns = 0, -1, -(10**18)
        self.feedback = "new instruction"

    def finish(self):
        self.epoch += 1
        self.instruction, self.motion, self.pending = None, None, None
        self.history = []

    def observe_frame(self, name, value, metadata):
        if name not in self.cameras:
            raise ValueError("unexpected camera role")
        frame = Frame.from_arrow(name, value, metadata)
        if frame.env_id != self.env_id:
            raise ValueError("cross-environment frame")
        old = self.frames.get(name)
        if old is None or frame.stamp_ns > old.stamp_ns:
            self.frames[name] = frame

    def observe_joints(self, values, stamp_ns, env_id):
        stamp_ns = integer(stamp_ns, "sim_time_ns")
        if integer(env_id, "env_id") != self.env_id:
            raise ValueError("cross-environment joint state")
        q = self.adapter.joint_vector(values)
        if stamp_ns >= self.joint_ns:
            self.joints, self.joint_ns = q, stamp_ns

    def prepare(self, now_ns: int) -> Ticket | None:
        if (
            self.instruction is None
            or self.joints is None
            or self.pending is not None
            or self.motion is not None
            or self.number >= 64
            or now_ns - self.last_request_ns < 1_000_000_000
            or any(name not in self.frames for name in self.cameras)
        ):
            return None
        frames = tuple(self.frames[name] for name in self.cameras)
        stamp = frames[0].stamp_ns
        if (
            any(frame.stamp_ns != stamp for frame in frames)
            or stamp <= self.last_obs_ns
            or not 0 <= now_ns - stamp <= 100_000_000
            or abs(self.joint_ns - stamp) > 100_000_000
        ):
            return None
        observation = Observation(frames, self.joints, stamp, self.env_id)
        self.number += 1
        self.last_obs_ns, self.last_request_ns = stamp, now_ns
        self.pending = Ticket(
            self.epoch,
            self.number,
            self.goal_id,
            observation,
            self.adapter.build_messages(self.instruction, observation, self.history, self.feedback),
        )
        return self.pending

    def reject(self, ticket: Ticket, reason: str):
        if self.pending is ticket and ticket.epoch == self.epoch:
            self.pending, self.feedback = None, str(reason)[:512]

    def accept(self, ticket: Ticket, action: ActionDelta, answer: str, now_ns: int) -> bool:
        if self.pending is not ticket or ticket.epoch != self.epoch or self.instruction is None:
            return False
        if not 0 <= now_ns - ticket.observation.stamp_ns <= 500_000_000:
            self.reject(ticket, "stale_observation")
            return False
        # A response belongs to its observation. A changed robot invalidates
        # the absolute IK target, even when the nominal time budget still holds.
        if self.joints is None or np.max(np.abs(self.joints - ticket.observation.joints)) > 0.01:
            self.reject(ticket, "robot_moved_during_inference")
            return False
        self.history = [
            *self.history,
            ticket.messages[-1],
            {"role": "assistant", "content": answer},
        ][-8:]
        try:
            self.motion = self.adapter.plan_motion(ticket.observation.joints, action, now_ns)
        except ValueError as error:
            self.reject(ticket, f"action_refused: {error}")
            return False
        self.pending, self.feedback = None, "executing"
        return True

    def answer(self, ticket: Ticket, text: str, now_ns: int) -> bool:
        """Parse a model reply and accept it; a malformed reply becomes feedback."""
        try:
            action = self.adapter.parse_action(text)
        except ValueError as error:
            self.reject(ticket, f"answer_refused: {error}")
            return False
        return self.accept(ticket, action, text, now_ns)

    def advance(self, now_ns: int):
        if self.motion is None or self.joints is None:
            return None, None
        if not 0 <= now_ns - self.joint_ns <= 100_000_000:
            self.motion, self.feedback = None, "stale_joint_state"
            return None, None
        command, grip = self.motion.step(self.joints, now_ns)
        if self.motion.done:
            self.feedback, self.motion = self.motion.feedback, None
        return command, grip


def main() -> None:  # pragma: no cover — exercised by the runtime acceptance test
    import hashlib
    import json
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import pyarrow as pa
    from PIL import Image

    from aisle.nodes.pear_backend import ChatClient
    from aisle.nodes.vla_smolvla import instruction_from_request
    from aisle.topics import make_sender
    from aisle.turn_node import Node

    embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
    if embodiment == "nano":
        from aisle.nodes.pear_nano import NanoAdapter

        adapter = NanoAdapter(Path(os.environ["AISLE_NANO_URDF"]))
        if adapter.sha256 != os.environ["AISLE_NANO_MODEL_SHA256"]:
            raise ValueError("Nano URDF changed after graph preparation")
    elif embodiment == "franka":
        adapter = FrankaAdapter()
    else:
        raise ValueError("PEAR supports the Franka and Nano embodiment adapters")
    node = Node()
    send = make_sender(node)
    control = Controller(int(os.environ.get("AISLE_ENV_PIN", "0")), adapter=adapter)
    synchronous = os.environ.get("AISLE_PEAR_LOCKSTEP") == "1"
    if synchronous and not node.lockstep:
        raise ValueError("PEAR lockstep inference requires the actual AISLE turn barrier")
    provider = os.environ.get("AISLE_PEAR_PROVIDER", "chat-completions")
    timeout_s = float(os.environ.get("AISLE_PEAR_TIMEOUT_S", "180"))
    if provider == "codex":
        from aisle.nodes.pear_codex import CodexClient

        client = CodexClient(
            os.environ["AISLE_PEAR_MODEL"],
            executable=os.environ.get("AISLE_CODEX_EXECUTABLE", "codex"),
            effort=os.environ.get("AISLE_CODEX_EFFORT", "medium"),
            timeout_s=timeout_s,
        )
    elif provider == "chat-completions":
        thinking = os.environ.get("AISLE_PEAR_THINKING", "1")
        if thinking not in {"0", "1"}:
            raise ValueError("AISLE_PEAR_THINKING must be 0 or 1")
        client = ChatClient(
            os.environ["AISLE_PEAR_ENDPOINT"],
            os.environ["AISLE_PEAR_MODEL"],
            timeout_s=timeout_s,
            enable_thinking=thinking == "1",
            max_tokens=int(os.environ.get("AISLE_PEAR_MAX_TOKENS", "4096")),
        )
    else:
        raise ValueError("unsupported inference provider")
    trace = (
        Path(os.environ["AISLE_PEAR_TRACE_DIR"]) if os.environ.get("AISLE_PEAR_TRACE_DIR") else None
    )
    if trace:
        trace.mkdir(parents=True, exist_ok=True)
    future, ticket = None, None
    pool = ThreadPoolExecutor(max_workers=1)

    def report(item: dict, metadata):
        item = {
            "model": client.model,
            "provider": provider,
            "model_revision": os.environ.get("AISLE_PEAR_MODEL_REVISION", "unverified"),
            "condition": "lockstep-eval" if synchronous else "async-live",
            "attested": False,
            **adapter.description,
            **item,
        }
        send("policy_result", pa.array([json.dumps(item, allow_nan=False)]), metadata)
        if trace:
            with (trace / "events.jsonl").open("a") as file:
                file.write(
                    json.dumps({**item, "metadata": trace_metadata(metadata)}, allow_nan=False)
                    + "\n"
                )

    def land(active, reply, metadata):
        now = integer(metadata.get("sim_time_ns"), "sim_time_ns")
        accepted = control.answer(active, reply.answer, now)
        report(
            {
                "event": "model_reply",
                "epoch": active.epoch,
                "request": active.number,
                "goal_id": active.goal_id,
                "accepted": accepted,
                "feedback": control.feedback,
                "answer": reply.answer,
                "usage": reply.usage,
            },
            metadata,
        )

    try:
        for event in node:
            if event.get("type") != "INPUT":
                continue
            metadata, eid = event.get("metadata") or {}, event["id"]
            if eid in {"oracle_state", "poses", "seg_overhead", "depth_overhead", "bridge_info"}:
                raise ValueError(f"PEAR policy refuses privileged/unneeded input {eid}")
            if metadata.get("env_id", control.env_id) != control.env_id:
                continue
            try:
                adapter.validate_metadata(eid, metadata)
                if eid == "reset_done":
                    control.reset()
                elif eid == "episode_result":
                    result = json.loads(event["value"][0].as_py())
                    if result.get("goal_id", metadata.get("goal_id")) == control.goal_id:
                        control.finish()
                        if node.lockstep and os.environ.get("AISLE_PEAR_STOP_ON_RESULT") == "1":
                            node.stop_after_turn()
                elif eid == "target_request":
                    request = json.loads(event["value"][0].as_py())
                    control.set_goal(
                        request.get("instruction") or instruction_from_request(request),
                        str(metadata.get("goal_id") or request.get("goal_id") or ""),
                    )
                elif eid.startswith("rgb_"):
                    control.observe_frame(eid.removeprefix("rgb_"), event["value"], metadata)
                elif eid == "joint_state":
                    control.observe_joints(
                        event["value"].to_numpy(zero_copy_only=False),
                        metadata.get("sim_time_ns"),
                        metadata.get("env_id"),
                    )
                cadence = eid == "turn" if node.lockstep else eid == "joint_state"
                if not cadence:
                    continue
                now = integer(metadata.get("sim_time_ns"), "sim_time_ns")
                if future is not None and future.done():
                    completed, active = future, ticket
                    future, ticket = None, None
                    try:
                        land(active, completed.result(), metadata)
                    except Exception as error:
                        control.reject(active, f"inference_error: {type(error).__name__}")
                        report({"event": "inference_error", "error": str(error)[:512]}, metadata)
                command, grip = control.advance(now)
                if command is not None:
                    send("joint_cmd", pa.array(command, type=pa.float32()), metadata)
                if grip is not None:
                    send("gripper_cmd", pa.array([grip], type=pa.float32()), metadata)
                active = control.prepare(now) if future is None else None
                if active is None:
                    continue
                if trace:
                    folder = trace / f"{active.epoch:04d}-{active.number:04d}"
                    folder.mkdir(exist_ok=True)
                    for frame in active.observation.frames:
                        Image.fromarray(frame.rgb).save(folder / f"{frame.name}.png")
                    (folder / "request.json").write_text(
                        json.dumps(
                            {
                                "messages": active.messages,
                                "observation_ns": active.observation.stamp_ns,
                                "goal_id": active.goal_id,
                            }
                        )
                    )
                report(
                    {
                        "event": "request",
                        "epoch": active.epoch,
                        "request": active.number,
                        "goal_id": active.goal_id,
                        "observation_ns": active.observation.stamp_ns,
                        "messages_sha256": hashlib.sha256(
                            json.dumps(active.messages).encode()
                        ).hexdigest(),
                    },
                    metadata,
                )
                if synchronous:
                    try:
                        land(
                            active,
                            client.infer(active.messages, seed=active.observation.stamp_ns),
                            metadata,
                        )
                    except Exception as error:
                        control.reject(active, f"inference_error: {type(error).__name__}")
                        report({"event": "inference_error", "error": str(error)[:512]}, metadata)
                else:
                    ticket = active
                    future = pool.submit(
                        client.infer, active.messages, seed=active.observation.stamp_ns
                    )
            except (ValueError, TypeError, KeyError, IndexError) as error:
                control.finish()
                control.frames, control.joints = {}, None
                print(f"PEAR input refused: {error}", file=sys.stderr)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
