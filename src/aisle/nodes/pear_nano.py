"""Experimental dual-arm Nano PEAR adapter; model geometry is read from its URDF."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from aisle.kinematics import load_urdf_chain
from aisle.nodes.pear_backend import (
    final_action_json,
    finite_number,
    observation_messages,
    rotation_to_rpy,
)

MODEL = Path(__file__).resolve().parents[3] / "assets/adora_nano/adora_nano.urdf"
SIDES = ("left", "right")
SUFFIXES = ("j1_yaw", "j2_pitch", "j3_pitch", "j4_pitch", "j5_roll")
PROMPT = """Control a simulated Adora Nano with two five-axis arms and hinged grippers.
Use only the three supplied RGB cameras and measured robot state. Base and head are held.
Positions are the URDF left_tool_mount and right_tool_mount origins, in the base_link
frame (+x forward, +y left, +z up), centimeters. These are tool mounting frames, not
calibrated finger-contact centers. Observe the fingers before attempting contact.
Each arm has FIVE independently controlled coordinates: xyz, pitch, roll; yaw is coupled
to the position and cannot be independently commanded. In this adapter pitch is the
sum of joints j2+j3+j4 and roll is j5, in radians; these are robot coordinates, not world
Euler angles. The observed tool_rpy_rad gives the actual FK orientation separately.
Actions are increments from this observation: xyz in centimeters, pitch/roll in radians.
Gripper angle is a joint-angle increment in radians, NOT jaw width or a normalized
open/close value. Infer how the hinged fingers move from images and execution feedback.
Zero means hold. Arm movement completes before the gripper moves. Read rejection feedback.
Per arm/turn: translation norm <=5 cm, pitch/roll delta norm <=0.2 rad, gripper delta
<=0.1 rad. Use small moves and reobserve. No object coordinates or depth are provided.
End with exactly one fenced json object with ALL fields (zero for unchanged quantities):
```json
{"left_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},
 "right_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},
 "left_gripper":{"angle":0},"right_gripper":{"angle":0}}
```
Do not produce executable code or declare task success. Joint limits and IK may refuse
an action. Reaching a tool pose does not establish contact or a successful grasp.
"""


class NanoAdapter:
    cameras = ("head", "left_wrist", "right_wrist")

    def __init__(self, path: Path = MODEL):
        self.path = Path(path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        try:
            root = ET.parse(self.path).getroot()
        except ET.ParseError as error:
            raise ValueError(f"Nano URDF is not well-formed XML: {error}") from error
        joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
        self.names = tuple(j.get("name") for j in joints)
        expected = {
            "wheel_1_joint",
            "wheel_2_joint",
            "wheel_3_joint",
            "head_pan_joint",
            "head_tilt_joint",
        }
        expected |= {side + "_" + suffix for side in SIDES for suffix in (*SUFFIXES, "j6_gripper")}
        if len(self.names) != 17 or set(self.names) != expected:
            raise ValueError("expected the named 17-channel Nano URDF")
        bounds, speeds, efforts = [], [], []
        for joint in joints:
            limit = joint.find("limit")
            kind = joint.get("type")
            if limit is None or kind not in {"revolute", "continuous"}:
                raise ValueError("Nano joints must have explicit angular limits/velocities")
            required = {"velocity", "effort"}
            if kind == "revolute":
                required |= {"lower", "upper"}
            if not required <= limit.attrib.keys():
                raise ValueError("Nano joint limit is missing a required bound")
            if any(not math.isfinite(float(limit.attrib[key])) for key in required):
                raise ValueError("Nano joint limits must be finite")
            speeds.append(float(limit.attrib["velocity"]))
            efforts.append(float(limit.attrib["effort"]))
            bounds.append(
                (-np.inf, np.inf)
                if kind == "continuous"
                else (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            )
        self.lower, self.upper = np.asarray(bounds).T
        self.speed = np.minimum(speeds, 0.3)
        self.effort = np.asarray(efforts)
        if np.any(self.lower >= self.upper) or np.any(self.speed <= 0) or np.any(self.effort <= 0):
            raise ValueError("invalid Nano limits")
        self.arms = {side: [self.names.index(side + "_" + s) for s in SUFFIXES] for side in SIDES}
        self.grippers = {side: self.names.index(side + "_j6_gripper") for side in SIDES}
        active = {*self.arms["left"], *self.arms["right"], *self.grippers.values()}
        self.held = [i for i in range(17) if i not in active]
        self.chains = {
            side: load_urdf_chain(self.path, "base_link", side + "_tool_mount") for side in SIDES
        }
        for side in SIDES:
            if self.chains[side].joint_names != tuple(self.names[i] for i in self.arms[side]):
                raise ValueError("unexpected Nano arm chain")
        self.description = {"embodiment": "nano", "robot_model_sha256": self.sha256}
        self.reset()

    def reset(self):
        self.held_reference = None

    def joint_vector(self, values):
        q = np.asarray(values)
        if q.shape != (17,) or q.dtype.kind not in "fi" or not np.isfinite(q).all():
            raise ValueError("Nano joint state requires 17 finite angular positions in URDF order")
        q = q.astype(float, copy=True)
        q.setflags(write=False)
        return q

    def validate_metadata(self, event_id, metadata):
        if event_id == "joint_state" or event_id.startswith("rgb_"):
            if metadata.get("robot_model_sha256") != self.sha256:
                raise ValueError("Nano observation/model hash mismatch")
        if event_id == "joint_state" and tuple(metadata.get("joint_names", ())) != self.names:
            raise ValueError("Nano joint order mismatch")

    def pose(self, q, side):
        return self.chains[side].forward(np.asarray(q)[self.arms[side]])

    def coordinates(self, q, side):
        arm = np.asarray(q)[self.arms[side]]
        return np.r_[self.pose(q, side)[0], sum(arm[1:4]), arm[4]]

    def build_messages(self, instruction, observation, history, feedback="initial observation"):
        q = self.joint_vector(observation.joints)
        state = {"joint_positions_rad": dict(zip(self.names, q.tolist(), strict=True))}
        for side in SIDES:
            coordinates = self.coordinates(q, side)
            state[side + "_arm"] = {
                "xyz_cm": (coordinates[:3] * 100).tolist(),
                "pitch_coordinate_rad": float(coordinates[3]),
                "roll_coordinate_rad": float(coordinates[4]),
                "tool_rpy_rad": rotation_to_rpy(self.pose(q, side)[1]).tolist(),
            }
            i = self.grippers[side]
            state[side + "_gripper"] = {
                "position_rad": float(q[i]),
                "limits_rad": [self.lower[i], self.upper[i]],
            }
        return observation_messages(instruction, observation, history, feedback, state, PROMPT)

    @staticmethod
    def parse_action(answer):
        action = final_action_json(answer)
        keys = {s + k for s in SIDES for k in ("_arm", "_gripper")}
        if not isinstance(action, dict) or set(action) != keys:
            raise ValueError("expected both Nano arms and both angular grippers")
        for side in SIDES:
            arm, grip = action[side + "_arm"], action[side + "_gripper"]
            if not isinstance(arm, dict) or set(arm) != {"x", "y", "z", "pitch", "roll"}:
                raise ValueError("Nano arm has xyz, pitch and roll; no independent yaw")
            if not isinstance(grip, dict) or set(grip) != {"angle"}:
                raise ValueError("Nano gripper uses angle, not jaw width")
            if not all(finite_number(v) for v in (*arm.values(), *grip.values())):
                raise ValueError("Nano action values must be finite numbers")
            if (
                math.hypot(*(arm[k] for k in ("x", "y", "z"))) > 5
                or math.hypot(arm["pitch"], arm["roll"]) > 0.2
                or abs(grip["angle"]) > 0.1
            ):
                raise ValueError("Nano action exceeds per-turn bounds")
        return action

    def solve(self, q, side, target):
        """Deterministic damped numerical IK for the five actual Nano coordinates."""
        q = np.array(q, dtype=float, copy=True)
        ids = self.arms[side]
        weights = np.array([5, 5, 5, 0.2, 0.2])
        for _ in range(100):
            current = self.coordinates(q, side)
            error = target - current
            if np.linalg.norm(error[:3]) < 0.00015 and np.max(np.abs(error[3:])) < 0.001:
                return q
            jacobian = np.empty((5, 5))
            for column, i in enumerate(ids):
                probe = q.copy()
                probe[i] += 1e-5
                jacobian[:, column] = (self.coordinates(probe, side) - current) / 1e-5
            jacobian *= weights[:, None]
            delta = np.linalg.solve(
                jacobian.T @ jacobian + 1e-5 * np.eye(5), jacobian.T @ (error * weights)
            )
            delta *= min(1.0, 0.1 / max(np.linalg.norm(delta), 1e-12))
            score = np.linalg.norm(error * weights)
            for scale in (1, 0.5, 0.25, 0.1):
                candidate = q.copy()
                candidate[ids] = np.clip(q[ids] + scale * delta, self.lower[ids], self.upper[ids])
                if np.linalg.norm((target - self.coordinates(candidate, side)) * weights) < score:
                    q = candidate
                    break
            else:
                break
        raise ValueError(f"{side} Nano target is unreachable within its five-axis limits")

    def workspace_ok(self, q):
        positions = [self.pose(q, side)[0] for side in SIDES]
        return (
            all(p[2] >= 0 and np.linalg.norm(p) <= 0.8 for p in positions)
            and np.linalg.norm(positions[0] - positions[1]) >= 0.08
        )

    def plan_motion(self, q, action, now_ns):
        q = self.joint_vector(q)
        # Revalidate direct callers too; no unchecked alternate action entry point.
        import json

        self.parse_action("```json\n" + json.dumps(action, allow_nan=False) + "\n```")
        if np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6):
            raise ValueError("measured Nano joints exceed URDF limits")
        path, current = [], q.copy()
        if self.held_reference is None:
            self.held_reference = q[self.held].copy()
        current[self.held] = self.held_reference
        starts = {s: self.coordinates(q, s) for s in SIDES}
        for step in range(1, 6):
            for side in SIDES:
                arm = action[side + "_arm"]
                delta = np.array([arm[k] for k in ("x", "y", "z", "pitch", "roll")], dtype=float)
                delta[:3] /= 100
                if np.any(delta):
                    current = self.solve(current, side, starts[side] + delta * step / 5)
            if not self.workspace_ok(current):
                raise ValueError("Nano tool workspace/separation bound exceeded")
            path.append(current.copy())
        for side, i in self.grippers.items():
            current[i] = np.clip(
                q[i] + action[side + "_gripper"]["angle"], self.lower[i], self.upper[i]
            )
        path.append(current.copy())
        return NanoMotion(self, path, now_ns)


class NanoMotion:
    def __init__(self, robot, path, now_ns):
        self.robot, self.path, self.start_ns = robot, path, now_ns
        self.index, self.settled = 0, 0
        self.active = [i for i in range(17) if i not in robot.held]
        self.command = None
        self.done, self.feedback = False, "executing"

    def step(self, q, now_ns):
        q = self.robot.joint_vector(q)
        if self.done or not 0 <= now_ns - self.start_ns <= 10_000_000_000:
            self.done, self.feedback = True, "motion_timeout"
            return None, None
        if self.command is None:
            self.command = q.copy()
        target = self.path[self.index]
        self.command += np.clip(0.5 * (target - q), -0.002, 0.002)
        self.command = np.clip(self.command, q - 0.03, q + 0.03)
        self.command[self.robot.held] = self.path[0][self.robot.held]
        self.settled = (
            self.settled + 1 if np.max(np.abs(target[self.active] - q[self.active])) < 0.001 else 0
        )
        if self.settled >= 3:
            self.index += 1
            self.settled = 0
            if self.index == len(self.path):
                self.done, self.feedback = True, "reached"
        return self.command.astype(np.float32), None


class NanoGuard:
    """Independent simulation guard: joint limits, per-tick velocity, held base/head."""

    def __init__(self, robot):
        self.robot = robot
        self.reset()

    def reset(self):
        self.q, self.last, self.anchor, self.stamp = None, None, None, -1
        self.last_command_ns = None
        self.corrected_joints = []

    def observe(self, q, stamp):
        self.q = self.robot.joint_vector(q)
        self.stamp = stamp
        if self.anchor is None:
            # Measurements arrive as float32: a joint resting on a limit may read 1e-7 past it.
            if np.any(self.q < self.robot.lower - 1e-6) or np.any(self.q > self.robot.upper + 1e-6):
                raise ValueError("cannot anchor Nano guard outside URDF limits")
            self.anchor = np.clip(self.q, self.robot.lower, self.robot.upper)
            self.last = self.anchor.copy()

    def apply(self, request, now_ns):
        if self.last is None:
            raise ValueError("Nano guard needs measured joints first")
        violations = []
        self.corrected_joints = []
        if self.last_command_ns is not None and now_ns <= self.last_command_ns:
            return self.last.astype(np.float32), ["duplicate_or_out_of_order_command"]
        dt = (
            0.01
            if self.last_command_ns is None
            else min(0.01, (now_ns - self.last_command_ns) / 1e9)
        )
        self.last_command_ns = now_ns
        try:
            command = self.robot.joint_vector(request).copy()
        except ValueError:
            return self.last.copy(), ["malformed_command"]
        if not 0 <= now_ns - self.stamp <= 100_000_000:
            return self.last.copy(), ["stale_joint_state"]
        # Joint and velocity limits always hold. The tracking window around the
        # measurement is applied first, so when it has drifted out of reach of the
        # last command the output returns toward it at the velocity limit instead
        # of holding for the rest of the run.
        lower = np.maximum(self.robot.lower, self.last - self.robot.speed * dt)
        upper = np.minimum(self.robot.upper, self.last + self.robot.speed * dt)
        if np.any(self.q - 0.03 > upper) or np.any(self.q + 0.03 < lower):
            violations.append("tracking_interval_conflict")
        bounded = np.clip(np.clip(command, self.q - 0.03, self.q + 0.03), lower, upper)
        bounded[self.robot.held] = self.anchor[self.robot.held]
        self.corrected_joints = [self.robot.names[i] for i in np.flatnonzero(bounded != command)]
        if not np.array_equal(bounded, command):
            violations.append("joint_velocity_or_held_joint_limit")
        if not self.robot.workspace_ok(bounded):
            return self.last.copy(), [*violations, "tool_workspace_or_separation"]
        self.last = bounded.copy()
        return bounded.astype(np.float32), violations
