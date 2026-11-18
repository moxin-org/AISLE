"""PEAR-style RGB/proprioception -> Cartesian-delta inference (ADR-74).

This is an inference adapter, not PEAR training or a bundled model. The server
must provide vision-capable Chat Completions and final answers separate from
reasoning. No simulator objects or executable model-generated code enter here.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
from PIL import Image

from aisle.nodes.budget_guard import fk_flange
from aisle.nodes.ik_trajectory import fk_tcp

SYSTEM_PROMPT = """Control a simulated Franka Panda using the supplied RGB views and measured
joint/tool state. Interpret the instruction and observe the result of each action before
choosing the next. No object positions are provided. Robot base frame: +x forward,
+y left, +z up. Camera image axes must be inferred from observations; do not assume
camera calibration. Tool xyz is the center between fingertips, in centimeters.
Orientation is fixed-axis XYZ roll/pitch/yaw in radians. Gripper width is in centimeters,
0 closed to 8 open; position or reaching a pose does not prove object contact or success.
Actions are increments from the OBSERVED tool pose, with additive XYZ Euler angles.
Positive gripper width opens, negative closes, zero holds. The arm moves before the
gripper. The controller may reject unreachable moves; read previous execution feedback.
Each step permits translation norm <=5 cm, rotation-delta norm <=0.35 rad, and width
change <=2 cm. Use small moves and reobserve. End with exactly one fenced json action
including every field, using zero for unchanged quantities:
```json
{"arm":{"x":0,"y":0,"z":0,"roll":0,"pitch":0,"yaw":0},"gripper":{"width":0}}
```
Do not output code or claim a task-success verdict; the independent verifier judges it.
"""


def integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def joint_vector(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (9,) or raw.dtype.kind not in "fi" or not np.isfinite(raw).all():
        raise ValueError("Franka joint_state must contain nine finite numeric positions")
    result = raw.astype(np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Frame:
    name: str
    rgb: np.ndarray
    stamp_ns: int
    env_id: int

    @classmethod
    def from_arrow(cls, name: str, value, metadata: dict) -> Frame:
        h, w = integer(metadata.get("h"), "h"), integer(metadata.get("w"), "w")
        if not h or not w or h * w > 4_194_304 or metadata.get("enc") != "rgb8":
            raise ValueError("expected a bounded, nonempty rgb8 image")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
            raise ValueError("invalid camera role")
        if value.type != pa.uint8() or value.null_count or len(value) != h * w * 3:
            raise ValueError("RGB payload must be UInt8[h*w*3] without nulls")
        rgb = value.to_numpy(zero_copy_only=False).reshape(h, w, 3).copy()
        rgb.setflags(write=False)
        return cls(
            name,
            rgb,
            integer(metadata.get("sim_time_ns"), "sim_time_ns"),
            integer(metadata.get("env_id"), "env_id"),
        )


@dataclass(frozen=True)
class Observation:
    frames: tuple[Frame, ...]
    joints: np.ndarray
    stamp_ns: int
    env_id: int


@dataclass(frozen=True)
class ActionDelta:
    position_cm: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    gripper_cm: float


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate action field: {name}")
        result[name] = value
    return result


def final_action_json(answer: str) -> dict:
    if not isinstance(answer, str) or len(answer) > 65_536 or "<think>" in answer:
        raise ValueError("expected a bounded final answer separate from reasoning")
    blocks = list(re.finditer(r"```json\s*\n?(.*?)```", answer, re.DOTALL))
    if len(blocks) != 1 or answer[blocks[0].end() :].strip():
        raise ValueError("answer must end with exactly one fenced json action")
    return json.loads(blocks[0].group(1), object_pairs_hook=_unique)


def parse_action(answer: str) -> ActionDelta:
    action = final_action_json(answer)
    names = ("x", "y", "z", "roll", "pitch", "yaw")
    if not isinstance(action, dict) or set(action) != {"arm", "gripper"}:
        raise ValueError("expected exactly arm and gripper")
    for part, fields in (("arm", set(names)), ("gripper", {"width"})):
        if not isinstance(action[part], dict) or set(action[part]) != fields:
            raise ValueError(f"invalid {part} fields")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in action[part].values()):
            raise ValueError("action quantities must be finite numbers, not booleans")
    position = tuple(float(action["arm"][key]) for key in names[:3])
    rotation = tuple(float(action["arm"][key]) for key in names[3:])
    width = float(action["gripper"]["width"])
    if math.hypot(*position) > 5 or math.hypot(*rotation) > 0.35 or abs(width) > 2:
        raise ValueError("action exceeds the per-turn translation/rotation/gripper bound")
    return ActionDelta(position, rotation, width)


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    if abs(math.cos(pitch)) < 1e-7:
        roll, yaw = math.atan2(-rotation[1, 2], rotation[1, 1]), 0.0
    else:
        roll, yaw = (
            math.atan2(rotation[2, 1], rotation[2, 2]),
            math.atan2(rotation[1, 0], rotation[0, 0]),
        )
    return np.array([roll, pitch, yaw])


def build_messages(
    instruction: str, observation: Observation, history: list, feedback: str = "initial observation"
) -> list:
    q = joint_vector(observation.joints)
    position = fk_tcp(q[:7]) * 100
    angles = rotation_to_rpy(fk_flange(q[:7])[1])
    state = {
        "arm": dict(
            zip(
                ("x", "y", "z", "roll", "pitch", "yaw"),
                [*position.tolist(), *angles.tolist()],
                strict=True,
            )
        ),
        "gripper": {"width": float(sum(q[7:]) * 100)},
        "arm_joint_positions_rad": q[:7].tolist(),
        "finger_positions_m": q[7:].tolist(),
    }
    return observation_messages(instruction, observation, history, feedback, state, SYSTEM_PROMPT)


def observation_messages(instruction, observation, history, feedback, state, prompt) -> list:
    """Shared RGB/history envelope; the embodiment supplies only measured robot state."""
    if not 1 <= len(observation.frames) <= 3:
        raise ValueError("one to three real RGB views are required")
    if len({frame.name for frame in observation.frames}) != len(observation.frames):
        raise ValueError("camera roles must be unique")
    if any(
        frame.stamp_ns != observation.stamp_ns or frame.env_id != observation.env_id
        for frame in observation.frames
    ):
        raise ValueError("all cameras must share an observation stamp and environment")
    content = [
        {
            "type": "text",
            "text": json.dumps(state, allow_nan=False)
            + f"\nInstruction: {instruction}\nPrevious execution: {feedback}",
        }
    ]
    for frame in observation.frames:
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(buffer, format="PNG")
        content.extend(
            [
                {"type": "text", "text": f"Camera: {frame.name}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode("ascii")
                    },
                },
            ]
        )
    return [
        {"role": "system", "content": prompt},
        *history[-8:],
        {"role": "user", "content": content},
    ]


@dataclass(frozen=True)
class ModelReply:
    answer: str
    usage: dict


def http_transport(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(4_194_305)
    if len(raw) > 4_194_304:
        raise ValueError("model response exceeds 4 MiB")
    return json.loads(raw)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 180,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        transport=http_transport,
        clock=time.monotonic,
    ):
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "expected an http(s) model base URL without credentials/query/fragment"
            )
        if (
            not model
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 32768
            or type(enable_thinking) is not bool
        ):
            raise ValueError("invalid model, timeout or token limit")
        self.url, self.model = base_url.rstrip("/") + "/chat/completions", model
        self.timeout_s, self.max_tokens, self.transport = timeout_s, max_tokens, transport
        self.enable_thinking = enable_thinking
        self.clock = clock

    def infer(self, messages: list, *, seed: int) -> ModelReply:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": integer(seed, "seed") % 2**31,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
                "preserve_thinking": False,
                "reasoning_effort": "xhigh",
            },
        }
        started = self.clock()
        result = self.transport(self.url, body, self.timeout_s)
        client_wall_time_s = self.clock() - started
        choices = result.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("model response incomplete or ambiguous")
        message = choices[0].get("message", {})
        if message.get("refusal") or not isinstance(message.get("content"), str):
            raise ValueError("model refused or supplied no final text")
        usage = {**(result.get("usage") or {}), "client_wall_time_s": client_wall_time_s}
        return ModelReply(message["content"], usage)
