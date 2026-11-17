"""PEAR-style inference contract, independent of model weights and simulators."""

import base64
import io
import json

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from aisle.nodes.budget_guard import load_limits
from aisle.nodes.pear_backend import (
    ActionDelta,
    ChatClient,
    Frame,
    Observation,
    build_messages,
    parse_action,
)

pytestmark = pytest.mark.unit


def answer(**changes):
    action = {
        "arm": dict.fromkeys(("x", "y", "z", "roll", "pitch", "yaw"), 0),
        "gripper": {"width": 0},
    }
    action["arm"].update(changes)
    return "Next bounded move.\n```json\n" + json.dumps(action) + "\n```"


def observation(stamp=10):
    frames = tuple(
        Frame.from_arrow(
            name,
            pa.array(np.arange(18, dtype=np.uint8)),
            {
                "h": 2,
                "w": 3,
                "enc": "rgb8",
                "sim_time_ns": stamp,
                "env_id": 0,
                "oracle_state": "DO_NOT_FORWARD",
            },
        )
        for name in ("overhead", "wrist")
    )
    return Observation(frames, np.array(load_limits("franka").fallback_qpos), stamp, 0)


def test_rgb_proprioception_only_and_real_image_dimensions():
    """TC-1/2/3/9, CON-4: serialize real RGB plus FK state, never extra metadata."""
    obs = observation()
    messages = build_messages("pick the cube", obs, [])
    serialized = json.dumps(messages)
    assert "DO_NOT_FORWARD" not in serialized
    assert "oracle_state" not in serialized
    assert "depth" not in serialized.lower()
    content = messages[-1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 2
    decoded = Image.open(io.BytesIO(base64.b64decode(images[0]["image_url"]["url"].split(",")[1])))
    assert decoded.size == (3, 2)
    assert np.array_equal(np.asarray(decoded), obs.frames[0].rgb)
    state = json.loads(content[0]["text"].split("\n")[0])
    assert len(state["arm_joint_positions_rad"]) == 7
    assert state["finger_positions_m"] == obs.joints[7:].tolist()
    assert set(state["arm"]) == {"x", "y", "z", "roll", "pitch", "yaw"}


@pytest.mark.parametrize("change", [{"h": 3}, {"enc": "depth"}, {"h": True}, {"sim_time_ns": -1}])
def test_bad_image_contract_refused(change):
    """TC-2/3: refuse malformed sizes, encoding, or timestamps."""
    metadata = {"h": 2, "w": 3, "enc": "rgb8", "sim_time_ns": 1, "env_id": 0, **change}
    with pytest.raises(ValueError):
        Frame.from_arrow("wrist", pa.array(np.arange(18, dtype=np.uint8)), metadata)


def test_action_is_incremental_and_bounded():
    """TC-1, BG-2: PEAR cm/radian deltas stay explicit until the motion adapter."""
    action = parse_action(answer(x=1, z=-2, yaw=0.1))
    assert action == ActionDelta((1.0, 0.0, -2.0), (0.0, 0.0, 0.1), 0.0)


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), 6])
def test_bad_or_oversized_actions_refused(value):
    """BG-2/3: model output cannot inject nonfinite, coerced, or unbounded commands."""
    with pytest.raises(ValueError):
        parse_action(answer(x=value))


@pytest.mark.parametrize(
    "text",
    [
        "print('move robot')",
        "<think>hidden</think>" + answer(),
        answer() + " trailing instructions",
        answer() + answer(),
        answer().replace('"x": 0', '"x": 0, "x": 1'),
        answer().replace('"yaw": 0', '"unknown": 0'),
    ],
)
def test_ambiguous_or_executable_output_is_never_an_action(text):
    """CON-4, BG-3: exactly one final JSON action; no executable-code path."""
    with pytest.raises(ValueError):
        parse_action(text)


def test_http_request_contains_images_seed_and_model_identity():
    """CON-1/5: CPU-only HTTP adapter; deterministic request seed, no ML runtime dependency."""
    calls = []

    def transport(url, body, timeout):
        calls.append((url, body, timeout))
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": answer()}}],
            "usage": {"completion_tokens": 40},
        }

    client = ChatClient("http://127.0.0.1:8000/v1", "pear", transport=transport)
    result = client.infer(build_messages("pick", observation(), []), seed=7)
    assert result.answer == answer()
    assert calls[0][0] == "http://127.0.0.1:8000/v1/chat/completions"
    assert calls[0][1]["seed"] == 7
    assert calls[0][1]["model"] == "pear"
    assert calls[0][1]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.parametrize("finish", ["length", "tool_calls", None])
def test_incomplete_response_never_executes(finish):
    """BG-3: even valid-looking JSON in a truncated answer cannot actuate."""
    client = ChatClient(
        "http://localhost:8000/v1",
        "pear",
        transport=lambda *args: {
            "choices": [{"finish_reason": finish, "message": {"content": answer()}}]
        },
    )
    with pytest.raises(ValueError):
        client.infer([], seed=0)


def test_local_inference_budget_is_explicit():
    """CON-5, BG-2: local token/thinking settings are forwarded, never hidden."""
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": answer()}}]}

    client = ChatClient(
        "http://127.0.0.1:8087/v1",
        "pear",
        max_tokens=512,
        enable_thinking=False,
        transport=transport,
    )
    client.infer([], seed=0)
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("kwargs", [{"max_tokens": True}, {"enable_thinking": "false"}])
def test_local_settings_refuse_coercion(kwargs):
    """CON-5: malformed inference conditions must fail before graph execution."""
    with pytest.raises(ValueError):
        ChatClient("http://127.0.0.1:8087/v1", "pear", **kwargs)


def test_http_client_measures_transport_latency_without_replacing_server_usage():
    """CON-5: an injected monotonic clock measures the remote round trip separately."""
    ticks = iter([100.0, 102.75])
    usage = {"completion_tokens": 7, "wall_time_s": 2.0, "client_wall_time_s": 999}
    client = ChatClient(
        "http://127.0.0.1:8088/v1",
        "pear",
        clock=lambda: next(ticks),
        transport=lambda *args: {
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
            "usage": usage,
        },
    )
    reply = client.infer([], seed=0)
    assert reply.usage == {
        "completion_tokens": 7,
        "wall_time_s": 2.0,
        "client_wall_time_s": 2.75,
    }
    assert usage["client_wall_time_s"] == 999
