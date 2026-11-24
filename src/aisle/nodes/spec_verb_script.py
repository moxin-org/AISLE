"""Scripted SPEC verb sequence for smoke-testing a vendor node (ADR-30 participant).

After the first ``reset_done`` it sends one ``cmd_request`` at a time and waits for
its ``cmd_response`` (retrying ``NOT_READY``), then logs each result. This checks the
SPEC <-> lockstep <-> physics plumbing only; it does not try to solve the task.

``AISLE_VERB_SCRIPT`` may hold a JSON list of ``[verb, params, expected_code]``;
``AISLE_VERB_SCRIPT_OUT`` names a JSON file for the results.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa

NS = "vendor.lerobot.so101"
# The SO-101 starts at home with its open jaw right at the target on the pharmacy shelf
# (TCP x = 0.39 m, shelf front x = 0.40 m). The script lifts the gripper up and back
# first (TCP z rises 0.23 -> 0.40 m along the whole path, x 0.39 -> 0.29 m), works the
# gripper away from the shelf, reopens it, and only then returns to rest. It must not
# touch the scene: the oracle ends the episode on collisions or a dropped target.
LIFT_BACK = {"shoulder_lift": -25.0, "elbow_flex": -10.0}
DEFAULT_SCRIPT = [
    [f"{NS}.enable_torque", {}, "0"],
    [f"{NS}.move_to_joint_state", {"joints": LIFT_BACK}, "0"],
    [f"{NS}.move_to_joint_state", {"joints": {"shoulder_pan": 30.0, "wrist_flex": -20.0}}, "0"],
    [f"{NS}.gripper.set", {"percent": 40}, "0"],
    [f"{NS}.gripper.close", {}, "0"],
    [
        f"{NS}.move_to_joint_state",
        {"joints": {"shoulder_lift": 60.0, "elbow_flex": 30.0}},
        "LIMIT_VIOLATION",
    ],
    [f"{NS}.gripper.open", {}, "0"],
    [f"{NS}.move_to_joint_state", {"joints": {"shoulder_pan": 0.0, "wrist_flex": 0.0}}, "0"],
    [f"{NS}.move_to_named", {"name": "rest"}, "0"],
    [f"{NS}.get_joint_state", {}, "0"],
]
MAX_NOT_READY = 200


class VerbScript:
    def __init__(self, script: list, send) -> None:
        self.script = [tuple(step) for step in script]
        self.send = send
        self.started = False
        self.index = 0
        self.pending: str | None = None
        self.not_ready = 0
        self.results: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.index >= len(self.script)

    def on_reset_done(self) -> None:
        self.started = True

    def on_turn(self, sim_s: float) -> None:
        if not self.started or self.pending is not None or self.finished:
            return
        verb, params, _ = self.script[self.index]
        self.pending = f"script-{self.index}-{self.not_ready}"
        envelope = {
            "request_id": self.pending,
            "verb": verb,
            "params": params,
            "spec_version": "1.0.0",
        }
        self.send("cmd_request", pa.array([json.dumps(envelope)]))
        if self.not_ready == 0:  # retries of a NOT_READY verb share its result entry
            self.results.append({"verb": verb, "sent_sim_s": round(sim_s, 3)})

    def on_response(self, response: dict, sim_s: float) -> None:
        if response.get("request_id") != self.pending:
            return
        self.pending = None
        if response.get("code") == "NOT_READY" and self.not_ready < MAX_NOT_READY:
            self.not_ready += 1
            return
        verb, _, expected = self.script[self.index]
        entry = self.results[-1]
        entry.update(
            code=response.get("code"),
            expected=expected,
            ok=response.get("code") == expected,
            done_sim_s=round(sim_s, 3),
            msg=(response.get("msg") or "")[:160],
        )
        joints = (response.get("data") or {}).get("joints")
        if joints:
            entry["joints"] = {k: round(v, 1) for k, v in joints.items()}
        outcome = f"{entry['code']} (expected {expected})"
        print(f"verb-script: {verb} -> {outcome} at {sim_s:.2f}s {entry['msg']}", flush=True)
        self.index += 1
        self.not_ready = 0


def main() -> None:
    from aisle.topics import make_sender
    from aisle.turn_node import Node

    node = Node()
    sender = make_sender(node)  # TC-2: per-topic seq; the turn wrapper adds the turn stamp
    script = VerbScript(
        json.loads(os.environ.get("AISLE_VERB_SCRIPT", "null")) or DEFAULT_SCRIPT,
        lambda topic, value: sender(topic, value, {}),
    )
    out_path = os.environ.get("AISLE_VERB_SCRIPT_OUT")
    sim_s = 0.0
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        ns = (event.get("metadata") or {}).get("sim_time_ns")
        if ns is not None:
            sim_s = int(ns) / 1e9
        if event["id"] == "reset_done":
            script.on_reset_done()
        elif event["id"] == "cmd_response":
            for raw in event["value"].to_pylist():
                script.on_response(json.loads(raw), sim_s)
            if script.finished and out_path:
                passed = all(r.get("ok") for r in script.results)
                with open(out_path, "w") as f:
                    json.dump({"passed": passed, "results": script.results}, f, indent=2)
                print(f"verb-script: finished, passed={passed}", flush=True)
                out_path = None
        elif event["id"] == "turn":
            script.on_turn(sim_s)


if __name__ == "__main__":
    main()
