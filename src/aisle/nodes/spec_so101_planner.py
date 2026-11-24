"""Pick-and-place planner that drives an SO-101 through SPEC verbs (ADR-30 participant).

The expert's planning, a SPEC vendor node's execution: on each ``grasp_pose`` it
solves the same ``StagedPlan`` as ik-trajectory (rise, staging, pregrasp, preclose,
advance, close, lift, retract, transfer, lower, release, clear, home) and sends the
waypoints as ``vendor.lerobot.so101.move_to_joint_state`` requests, one at a time,
to the vendor node (spec-so101). The vendor node does the interpolation, safety
checks and settling; this node never commands joints itself.

Waypoints are thinned to one per ``AISLE_PLANNER_STRIDE_DEG`` of joint travel (each
stage's last waypoint always kept). ``NOT_REACHED`` answers are logged and the plan
continues, as the expert's stage bail does; ``TRACKING_ERROR`` and any refusal abort
the plan.

    in   grasp_pose    pose7d_f32 (+ approach_m, place_tcp_z, place_xy metadata)
    in   cmd_response  json_utf8, episodic back-edge from the vendor node
    in   reset_done    episode boundary: drop any plan
    in   turn          clock
    out  cmd_request   json_utf8 SPEC envelopes
    out  turn_done
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

NS = "vendor.lerobot.so101"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
RAD_S = math.degrees(1.0)  # the expert's 1 rad/s joint speed, in deg/s
# The vendor node eases each move with a cosine profile whose PEAK is the requested
# speed, so its mean is 2/pi of it; the expert marches at a constant ``vel`` rad/s.
# Asking for pi/2 times the expert's speed gives each hop the expert's mean speed
# (seeds 1 and 9 missed the 150 s per-episode wall clamp by 1-2 s without it).
EASE_PEAK = math.pi / 2
ABORT_CODES = {
    "TRACKING_ERROR",
    "LIMIT_VIOLATION",
    "ESTOPPED",
    "NOT_ENABLED",
    "HARDWARE_ERROR",
    "INVALID_PARAMS",
    "CONTROLLER_BUSY",
}


@dataclass
class Step:
    stage: str
    verb: str
    params: dict


def gripper_percent(aisle_gripper: float) -> float:
    """AISLE 0 open .. 1 closed -> SPEC percent open."""
    return round((1.0 - min(1.0, max(0.0, aisle_gripper))) * 100.0, 2)


def plan_steps(stages, stride_deg: float, max_speed_deg_s: float) -> list[Step]:
    """StagedPlan stages -> SPEC move requests (joint degrees + gripper percent)."""
    steps: list[Step] = []
    last: np.ndarray | None = None
    for stage in stages:
        path = [np.degrees(np.asarray(q, dtype=np.float64)) for q in stage.path]
        speed = round(min(max_speed_deg_s, EASE_PEAK * RAD_S * float(stage.vel)), 2)
        for index, q in enumerate(path):
            final = index == len(path) - 1
            if not final and last is not None and float(np.max(np.abs(q - last))) < stride_deg:
                continue
            joints = {name: round(float(v), 3) for name, v in zip(ARM_JOINTS, q, strict=True)}
            joints["gripper"] = gripper_percent(stage.gripper)
            steps.append(
                Step(
                    stage.name,
                    f"{NS}.move_to_joint_state",
                    {"joints": joints, "speed_deg_s": speed},
                )
            )
            last = q
    return steps


class Planner:
    def __init__(
        self, send, *, stride_deg: float, max_speed_deg_s: float, embodiment: str = "so101"
    ):
        from aisle.scenes.pharmacy import load_physics, resolve_layout

        self.send = send
        self.stride_deg = stride_deg
        self.max_speed = max_speed_deg_s
        self.embodiment = embodiment
        physics = load_physics()
        tray = resolve_layout(physics, embodiment)["tray"]["pos"]
        self.tray_xy = (float(tray[0]), float(tray[1]))
        self.home = np.asarray(physics["embodiment"][embodiment]["home_qpos"], dtype=np.float32)
        self.queue: list[Step] = []
        self.pending: str | None = None
        self.counter = 0
        self.enabled = False
        self.log: list[dict] = []

    @property
    def busy(self) -> bool:
        return bool(self.queue) or self.pending is not None

    def on_reset(self) -> None:
        self.queue, self.pending, self.enabled = [], None, False

    def on_grasp_pose(self, pose: np.ndarray, metadata: dict) -> str:
        from aisle.nodes.ik_trajectory import PLACE_TCP_Z, StagedPlan

        if self.busy:
            return "ignored: a plan is executing"
        dest = metadata.get("place_xy") or self.tray_xy
        plan = StagedPlan(
            pose,
            (float(dest[0]), float(dest[1])),
            float(metadata.get("approach_m", 0.15)),
            self.home,
            place_z=float(metadata.get("place_tcp_z", PLACE_TCP_Z)),
            embodiment=self.embodiment,
        )
        if not plan.ok:
            return f"plan failed: {plan.error}"
        steps = plan_steps(plan.stages, self.stride_deg, self.max_speed)
        if not self.enabled:
            steps.insert(0, Step("enable", f"{NS}.enable_torque", {}))
        self.queue = steps
        return f"plan ready: {len(plan.stages)} stages, {len(steps)} requests"

    def on_turn(self) -> None:
        if self.pending is not None or not self.queue:
            return
        step = self.queue[0]
        self.counter += 1
        self.pending = f"plan-{self.counter}"
        envelope = {
            "request_id": self.pending,
            "verb": step.verb,
            "params": step.params,
            "spec_version": "1.0.0",
        }
        self.send("cmd_request", pa.array([json.dumps(envelope)]))

    def on_response(self, response: dict, sim_s: float) -> str | None:
        if self.pending is None or response.get("request_id") != self.pending:
            return None
        self.pending = None
        step = self.queue.pop(0)
        code = str(response.get("code"))
        entry = {"stage": step.stage, "code": code, "sim_s": round(sim_s, 2)}
        data = response.get("data") or {}
        if data.get("gripper_blocked"):
            entry["gripper_blocked"] = True
        self.log.append(entry)
        if step.verb.endswith("enable_torque") and response.get("ok"):
            self.enabled = True
        stage_done = not self.queue or self.queue[0].stage != step.stage
        if code in ABORT_CODES:
            self.queue = []
            return f"plan aborted at {step.stage}: {code} {(response.get('msg') or '')[:120]}"
        if code == "NOT_REACHED":
            return f"{step.stage}: NOT_REACHED, continuing ({(response.get('msg') or '')[:100]})"
        if stage_done:
            return f"stage done: {step.stage} at {sim_s:.2f}s" + (
                " (gripper blocked)" if data.get("gripper_blocked") else ""
            )
        return None


def main() -> None:
    from aisle.topics import make_sender
    from aisle.turn_node import Node

    node = Node()
    sender = make_sender(node)
    planner = Planner(
        lambda topic, value: sender(topic, value, {}),
        stride_deg=float(os.environ.get("AISLE_PLANNER_STRIDE_DEG", "8")),
        max_speed_deg_s=float(os.environ.get("AISLE_PLANNER_MAX_SPEED_DEG_S", "57.3")),
        embodiment=os.environ.get("AISLE_EMBODIMENT", "so101"),
    )
    sim_s = 0.0
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        if metadata.get("sim_time_ns") is not None:
            sim_s = int(metadata["sim_time_ns"]) / 1e9
        input_id = event["id"]
        if input_id == "reset_done":
            planner.on_reset()
        elif input_id == "grasp_pose":
            pose = np.asarray(event["value"].to_numpy(zero_copy_only=False), dtype=np.float32)
            print(
                f"planner: {planner.on_grasp_pose(pose.reshape(-1)[:7], metadata)}",
                file=sys.stderr,
                flush=True,
            )
        elif input_id == "cmd_response":
            for raw in event["value"].to_pylist():
                line = planner.on_response(json.loads(raw), sim_s)
                if line:
                    print(f"planner: {line}", file=sys.stderr, flush=True)
        elif input_id == "turn":
            planner.on_turn()


if __name__ == "__main__":
    main()
