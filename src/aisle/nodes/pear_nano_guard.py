"""Independent guard node for the experimental 17-channel Nano wire interface."""

import json
import os
import time
from pathlib import Path

import pyarrow as pa

from aisle.nodes.pear_nano import NanoAdapter, NanoGuard
from aisle.topics import make_sender
from aisle.turn_node import Node


def model_from_environment():
    robot = NanoAdapter(Path(os.environ["AISLE_NANO_URDF"]))
    if robot.sha256 != os.environ["AISLE_NANO_MODEL_SHA256"]:
        raise ValueError("Nano URDF changed after graph preparation")
    return robot


def main():
    robot = model_from_environment()
    guard = NanoGuard(robot)
    node = Node()
    send = make_sender(node)
    began = time.monotonic()
    pending = []
    for event in node:
        if event.get("type") != "INPUT":
            continue
        name, metadata = event["id"], event.get("metadata") or {}
        if name == "reset_done":
            guard.reset()
            began = time.monotonic()
            pending = []
        elif name == "joint_state":
            robot.validate_metadata(name, metadata)
            guard.observe(event["value"].to_numpy(zero_copy_only=False), metadata["sim_time_ns"])
        elif name == "joint_cmd":
            pending.append(event["value"].to_numpy(zero_copy_only=False))
        elif name == "turn":
            for requested in pending:
                if time.monotonic() - began > 1800:
                    command, violations = guard.last.copy(), ["wall_timeout"]
                else:
                    command, violations = guard.apply(requested, metadata["sim_time_ns"])
                send("joint_cmd_safe", pa.array(command, type=pa.float32()), metadata)
                if violations:
                    send(
                        "violation",
                        pa.array(
                            [
                                json.dumps(
                                    {
                                        "reasons": violations,
                                        "corrected_joints": getattr(guard, "corrected_joints", []),
                                    }
                                )
                            ]
                        ),
                        metadata,
                    )
            pending = []


if __name__ == "__main__":
    main()
