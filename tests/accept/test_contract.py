"""SPEC 010 contract acceptance (spec section 4, cases A1..A3) against the live bridge.

Marker `accept` (CON-12): launches dora dataflows with genesis inside the
bridge node. Requires the sim extra and the dora CLI.
"""

import importlib.util
import json
import shutil

import pytest
from contract_capture import assert_rates, write_contract_dataflow

from aisle.embodiment import G1_RIGHT_JOINTS, SO101_JOINTS

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None or shutil.which("dora") is None,
        reason="sim extra or dora CLI not installed",
    ),
]

RATES = {
    "rgb_overhead": 30,
    "rgb_wrist": 30,
    "depth_overhead": 15,
    "joint_state": 100,
    "gripper_state": 100,
    "oracle_state": 30,
    "poses": 15,
}
FRANKA_N_DOF = 9
SO101_N_DOF = len(SO101_JOINTS)
G1_RIGHT_N_DOF = len(G1_RIGHT_JOINTS)  # TC-5 draft: the WIRE is 7+2, not 33


def capture(dataflow, tmp_path, bridge_env, driver_env, duration_s=10.0):
    out = tmp_path / "records.jsonl"
    graph = write_contract_dataflow(
        tmp_path, out, bridge_env=bridge_env, driver_env=driver_env, duration_s=duration_s
    )
    # Await complete protocol evidence, with a separate bounded startup watchdog.
    dataflow.run_until_settled(graph, out, deadline_s=600)
    records = [r for r in dataflow.read(out) if r["id"] != "tick"]
    assert records, "recorder captured nothing"
    previous = {}
    for row in records:
        if row["id"] == "bridge_info":  # the sole pre-loop announcement
            continue
        meta = row["metadata"]
        required = {"turn_epoch", "turn_id", "sim_time_ns", "seq", "env_id"}
        assert required <= meta.keys(), row
        assert all(type(meta[key]) is int for key in required), row
        assert meta["turn_epoch"] == 19 and meta["env_id"] == 0, row
        prior = previous.get(row["id"])
        if prior is not None:
            assert meta["seq"] > prior["seq"] and meta["turn_id"] >= prior["turn_id"], row
        previous[row["id"]] = meta
    reset = next(r for r in records if r["id"] == "reset")
    assert reset["metadata"]["turn_id"] == 0 and reset["metadata"]["sim_time_ns"] == 0
    return records


def test_schema_conformance(tmp_path, dataflow):
    """Acceptance A1 (TC-1, TC-2, TC-3, TC-4, TC-5): 10 s headless run — every
    observed message carries sim_time_ns/env_id/seq metadata, image topics
    carry h/w/enc with h*w*3 payloads, producer rates stay within the ±20%
    contract band, and joint_state serves the franka profile's n_dof with
    names metadata."""
    records = capture(
        dataflow,
        tmp_path,
        bridge_env={"AISLE_SEED": 7},
        driver_env={"DRIVER_MODE": "conformance", "DRIVER_N_DOF": FRANKA_N_DOF},
    )
    by_topic: dict[str, list[dict]] = {}
    for r in records:
        by_topic.setdefault(r["id"], []).append(r)

    for topic, rate in RATES.items():
        msgs = by_topic.get(topic, [])
        assert msgs, f"no {topic} messages observed"
        for m in msgs:
            meta = m["metadata"]
            assert {"sim_time_ns", "env_id", "seq"} <= set(meta), (topic, meta)  # TC-2
        seqs = [int(m["metadata"]["seq"]) for m in msgs]
        assert seqs == sorted(seqs), f"{topic} seq not monotonic"  # TC-2
        measured = assert_rates(topic, msgs, rate, 10.0)
        print(f"{topic}: {json.dumps(measured, sort_keys=True)}")

    for topic in ("rgb_overhead", "rgb_wrist"):
        m = by_topic[topic][0]
        meta = m["metadata"]
        assert meta["enc"] == "rgb8"  # TC-3
        assert m["len"] == int(meta["h"]) * int(meta["w"]) * 3
    depth = by_topic["depth_overhead"][0]
    assert depth["len"] == int(depth["metadata"]["h"]) * int(depth["metadata"]["w"])

    joint = by_topic["joint_state"][0]
    assert joint["len"] == FRANKA_N_DOF  # TC-5 franka n_dof = 7+2
    assert joint["dtype"] == "float"  # TC-1: float32 ("float" is arrow f32)
    assert joint["metadata"]["names"], "joint_state missing names metadata"
    # PD control provably changes state: late joint values differ from first
    first_vals = by_topic["joint_state"][0]["values"]
    late_vals = by_topic["joint_state"][-1]["values"]
    assert first_vals != late_vals, "joint_state never moved under PD commands"

    # TC-1/TC-3 payload typing across the contract table
    assert by_topic["gripper_state"][0]["dtype"] == "float"
    assert by_topic["gripper_state"][0]["len"] == 1
    assert 0.0 <= by_topic["gripper_state"][0]["values"][0] <= 1.0
    oracle = by_topic["oracle_state"][0]
    assert oracle["dtype"] == "float" and oracle["len"] == 35
    poses = by_topic["poses"][0]  # SPEC 010: non-privileged twin of oracle_state
    assert poses["dtype"] == "float" and poses["len"] == 35
    for i in range(5):  # TC-1: quaternions are (x,y,z,w) — w LAST
        block = oracle["values"][i * 7 + 3 : i * 7 + 7]
        assert abs(block[3]) > 0.9, (i, block)
    assert by_topic["rgb_overhead"][0]["dtype"] == "uint8"
    assert by_topic["rgb_wrist"][0]["dtype"] == "uint8"
    assert by_topic["depth_overhead"][0]["dtype"] == "float"

    info = by_topic["bridge_info"][0]
    assert int(info["metadata"]["seq"]) == 0  # BRG-6: the one pre-loop announcement
    assert info["len"] == 1
    assert len(by_topic["bridge_info"]) == 1  # exactly once
    # BRG-3: joint_state documents coalescing in metadata dropped:int
    assert all("dropped" in m["metadata"] for m in by_topic["joint_state"])
    assert any(int(m["metadata"]["dropped"]) > 0 for m in by_topic["joint_state"]), (
        "driver double-sends per turn; some coalescing must be observed"
    )


def test_so101_schema_conformance(tmp_path, dataflow):
    """Acceptance A1 (TC-1, TC-2, TC-5, SCN-4): the official SO-101
    profile publishes and accepts the six-joint wire contract in motor order,
    announces the realized DOF count, and normalizes its revolute gripper."""
    records = capture(
        dataflow,
        tmp_path,
        bridge_env={"AISLE_SEED": 7, "AISLE_EMBODIMENT": "so101"},
        driver_env={"DRIVER_MODE": "conformance", "DRIVER_N_DOF": SO101_N_DOF},
        duration_s=4.0,
    )
    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(record["id"], []).append(record)
    info = json.loads(by_topic["bridge_info"][0]["text"])
    assert info["embodiment"] == "so101" and info["n_dof"] == SO101_N_DOF
    joint = by_topic["joint_state"][0]
    assert joint["len"] == SO101_N_DOF
    assert tuple(joint["metadata"]["names"]) == SO101_JOINTS
    gripper = by_topic["gripper_state"][0]
    assert gripper["len"] == 1
    assert 0.0 <= gripper["values"][0] <= 1.0


def test_g1_right_schema_conformance(tmp_path, dataflow):
    """Acceptance A1 (TC-5 draft, SCN-4, ADR-g1-embodiment): the g1_right
    profile serves a WIRE-sized contract — bridge_info n_dof is 9 (the
    7+2 wire, never the robot's 33 native DOFs), joint_state is 9 long
    with the official G1 right-arm joint names, and gripper_state stays
    the normalized [0, 1] width (0=open, 1=closed)."""
    records = capture(
        dataflow,
        tmp_path,
        bridge_env={"AISLE_SEED": 7, "AISLE_EMBODIMENT": "g1_right"},
        driver_env={"DRIVER_MODE": "conformance", "DRIVER_N_DOF": G1_RIGHT_N_DOF},
        duration_s=4.0,
    )
    by_topic: dict[str, list[dict]] = {}
    for record in records:
        by_topic.setdefault(record["id"], []).append(record)
    info = json.loads(by_topic["bridge_info"][0]["text"])
    assert info["embodiment"] == "g1_right" and info["n_dof"] == G1_RIGHT_N_DOF
    joint = by_topic["joint_state"][0]
    assert joint["len"] == G1_RIGHT_N_DOF
    assert tuple(joint["metadata"]["names"]) == G1_RIGHT_JOINTS
    gripper = by_topic["gripper_state"][0]
    assert gripper["len"] == 1
    assert 0.0 <= gripper["values"][0] <= 1.0


def test_reset_service(tmp_path, dataflow):
    """Acceptance A2 (TC-6, CON-5, RST-1): seeded resets routed THROUGH the
    reset dispatcher node — every request gets a forwarded reset_done
    echoing its request_id within the RST-1 budget, no observation
    interleaves a reset and its reply, and identical seeds reproduce an
    identical first oracle_state after reset."""
    seeds = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1]
    records = capture(
        dataflow,
        tmp_path,
        bridge_env={"AISLE_SEED": 7},
        driver_env={
            "DRIVER_MODE": "reset",
            "DRIVER_RESET_SEEDS": ",".join(str(s) for s in seeds),
            "DRIVER_RESET_SPACING": 10,
        },
    )
    # the DISPATCHER's forwarded replies: proves src/aisle/reset/service.py
    # ran live and preserved TC-6 metadata across both hops
    dones = [r for r in records if r["id"] == "reset_done"]
    assert len(dones) == len(seeds), f"{len(dones)} reset_done for {len(seeds)} requests"
    # RST-1 end-to-end: request arrival and reply arrival are stamped by the
    # recorder's OWN clock, so their delta spans driver -> dispatcher ->
    # bridge teleport -> dispatcher -> reply (t_reset_ms alone would only
    # measure the bridge's internal handler)
    request_wall_t = {
        r["metadata"]["request_id"]: r["wall_t"] for r in records if r["id"] == "reset"
    }
    assert [r["metadata"]["request_id"] for r in dones] == [
        f"req-{i}-{seed}" for i, seed in enumerate(seeds, 1)
    ]
    for done in dones:
        meta = done["metadata"]
        assert meta["request_id"].startswith("req-")  # TC-6 request/reply correlation
        assert done["wall_t"] - request_wall_t[meta["request_id"]] < 2.0  # RST-1
        # bridge-internal handler time is a consistent sub-measurement
        assert 0 <= int(meta["t_reset_ms"]) < 2000

    # CON-5: first oracle_state after a reset is a pure function of the seed
    first_oracle_after: dict[int, str] = {}
    pending_seed = None
    for r in records:
        if r["id"] == "bridge_reset_done":
            pending_seed = int(r["metadata"]["seed"])
        elif r["id"] == "oracle_state" and pending_seed is not None:
            first_oracle_after.setdefault(pending_seed, r["sha256"])
            if r["sha256"] != first_oracle_after[pending_seed]:
                pytest.fail(f"seed {pending_seed} reproduced different oracle_state")
            pending_seed = None
    assert pending_seed is None
    assert len([r for r in records if r["id"] == "bridge_reset_done"]) == len(seeds)
    assert set(first_oracle_after) == set(seeds)

    # TC-6 send-side ordering: dora preserves per-producer order, so the
    # bridge message FOLLOWING each reset_done must be the post-reset
    # oracle snapshot — nothing interleaves the service completion
    # (checked on the bridge's OWN stream: per-producer order holds there,
    # while the dispatcher-forwarded copy crosses producers)
    for i, r in enumerate(records):
        if r["id"] != "bridge_reset_done":
            continue
        following = next(
            (
                x
                for x in records[i + 1 :]
                if x["id"] not in {"reset_done", "reset", "reset_refused"}
            ),
            None,
        )
        assert following is not None and following["id"] == "oracle_state", following


def test_episode_action_lifecycle(tmp_path, dataflow):
    """Acceptance A3 (TC-7, TC-8): a scripted trivial episode CLOSED THROUGH
    THE BRIDGE — the client sends only the goal; the verifier stub emits
    feedback and the schema-valid episode_result only after receiving live
    oracle_state from the bridge, so a dead bridge fails this test."""
    records = capture(
        dataflow,
        tmp_path,
        bridge_env={"AISLE_SEED": 7},
        driver_env={"DRIVER_MODE": "episode"},
    )
    goals = [r for r in records if r["id"] == "episode_goal"]
    feedback = [r for r in records if r["id"] == "episode_feedback"]
    results = [r for r in records if r["id"] == "episode_result"]
    assert goals and feedback and results
    goal_id = goals[0]["metadata"]["goal_id"]
    assert all(f["metadata"]["goal_id"] == goal_id for f in feedback)
    assert results[0]["metadata"]["goal_id"] == goal_id
    result = json.loads(results[0]["text"])
    assert result["status"] in ("success", "fail")
    assert result["failure"] in (
        None,
        "wrong_object",
        "dropped",
        "timeout",
        "never_grasped",
        "collision",
    )
    assert result["goal_id"] == goal_id
    assert result["verifier"] == "oracle"  # TC-8: oracle verdicts are ground truth
    assert isinstance(result["t_end"], float) and isinstance(result["seed"], int)
