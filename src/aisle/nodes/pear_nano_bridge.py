"""Fixed-base Nano/Nexus sensor bridge for non-attesting PEAR arm bring-up."""

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa

from aisle.kinematics import _origin_transform
from aisle.nodes.pear_nano_guard import model_from_environment
from aisle.sim.nexus_backend import _ensure_nexus, _new_scene, pos_lookat_up_to_transform
from aisle.topics import make_sender
from aisle.turns import BridgeTurn, TurnStamp, watermark_metadata


def wheel_floor(model):
    """Place the floor at the actual wheel collider minimum, not a guessed root offset."""
    root = ET.parse(model.path).getroot()
    minimum = np.inf
    for number in (1, 2, 3):
        joint = root.find(f"joint[@name='wheel_{number}_joint']")
        transform = _origin_transform(joint.find("origin"))
        link = root.find(f"link[@name='wheel_{number}']")
        for collider in link.findall("collision"):
            mesh = collider.find("geometry/mesh")
            raw = (model.path.parent / mesh.attrib["filename"]).read_bytes()
            count = int.from_bytes(raw[80:84], "little")
            if len(raw) != 84 + count * 50:
                raise ValueError("Nano wheel collider must be a binary STL")
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            vertices = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].reshape(-1, 3)
            vertices = vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            pose = transform @ _origin_transform(collider.find("origin"))
            minimum = min(minimum, float((vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()))
    if not np.isfinite(minimum):
        raise ValueError("missing Nano wheel support geometry")
    return minimum


class NanoWorld:
    def __init__(self, model, run_dir):
        self.model = model
        # A stationary arm fixture constrains the three wheels at zero. Keeping
        # continuous wheel/floor contacts under a fixed root is unstable on the
        # current Nexus solver. Arm/head joints and all geometry are unchanged.
        root = ET.parse(model.path).getroot()
        self.locked = [f"wheel_{i}_joint" for i in (1, 2, 3)]
        for name in self.locked:
            root.find(f"joint[@name='{name}']").set("type", "fixed")
        for mesh in root.findall(".//mesh"):
            mesh.set("filename", str((model.path.parent / mesh.attrib["filename"]).resolve()))
        fixture = Path(run_dir) / "nano-arm-fixture.urdf"
        ET.ElementTree(root).write(fixture, encoding="utf-8", xml_declaration=True)
        self.fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        engine = _ensure_nexus()
        self.scene = _new_scene(
            engine, {"sim": {"dt": 0.01, "gravity": [0, 0, -9.81]}}, 1, [0.5] * 3
        )
        self.robot = self.scene.add_urdf_robot(fixture, "adora_nano", True)
        floor = wheel_floor(model)
        self.scene.add_box(
            "floor",
            [4, 4, 0.1],
            [0, 0, floor - 0.05],
            fixed=True,
            color=[0.6, 0.6, 0.65, 1],
            friction=0.7,
        )
        self.scene.add_box(
            "table", [0.7, 1, 0.05], [0.6, 0, 0.2], fixed=True, color=[0.4, 0.25, 0.15, 1]
        )
        for i, color in enumerate(([0.9, 0.1, 0.1, 1], [0.1, 0.7, 0.2, 1], [0.1, 0.3, 0.9, 1])):
            self.scene.add_box(
                f"cube_{i}",
                [0.03] * 3,
                [0.55, (i - 1) * 0.12, 0.25],
                fixed=False,
                color=color,
                density=300,
                friction=0.7,
            )
        self.cameras = {}
        # Optical poses are explicit simulation assumptions; URDF camera housings
        # supply the parent links, not measured hardware intrinsics.
        specs = [
            ("head", "head_tilt", [0.03, 0.018, 0], [0, 1, 0], [1, 0, 0]),
            ("left_wrist", "left_wrist_cam", [0.073, 0, 0], [1, 0, 1], [0, 0, -1]),
            ("right_wrist", "right_wrist_cam", [0.073, 0, 0], [1, 0, 1], [0, 0, -1]),
        ]
        for name, link, origin, forward, up in specs:
            camera = self.scene.add_camera((320, 240), 70)
            camera.attach(
                self.robot.get_link(link),
                pos_lookat_up_to_transform(origin, np.asarray(origin) + forward, up),
            )
            self.cameras[name] = camera
        self.scene.build()
        self.wire_indices = [i for i, name in enumerate(model.names) if name not in self.locked]
        self.indices = [
            self.robot.get_joint(model.names[i]).dofs_idx_local[0] for i in self.wire_indices
        ]
        # The generic wrapper has unlimited motors. This Nano simulation uses
        # the actual URDF effort ceilings and modest wheel/gripper gains.
        self.robot.robots[0].set_pd_gains(
            self.indices, max_force=model.effort[self.wire_indices].tolist()
        )
        grips = [
            self.robot.get_joint(model.names[i]).dofs_idx_local[0] for i in model.grippers.values()
        ]
        self.robot.set_dofs_kp([5.0] * 2, grips)
        self.robot.set_dofs_kv([0.75] * 2, grips)
        self.robot.set_qpos(np.zeros(self.robot.n_dofs))
        self.command(np.zeros(17))
        self.floor_z = floor

    def command(self, q):
        q = self.model.joint_vector(q)
        if any(q[self.model.names.index(name)] != 0 for name in self.locked):
            raise ValueError("Nano arm fixture cannot actuate its locked wheels")
        self.robot.control_dofs_position(q[self.wire_indices], self.indices)

    def joints(self):
        q = np.zeros(17)
        q[self.wire_indices] = np.asarray(self.robot.get_qpos())[self.indices]
        return self.model.joint_vector(q)

    def images(self):
        return {
            name: np.asarray(
                camera.render(rgb=True, depth=False, segmentation=False)[0], dtype=np.uint8
            )
            for name, camera in self.cameras.items()
        }


def main():
    from dora import Node

    model = model_from_environment()
    out = Path(os.environ["AISLE_NANO_RUN_DIR"])
    world = NanoWorld(model, out)
    node = Node()
    send = make_sender(node)
    maximum = int(os.environ["AISLE_NANO_DURATION_S"]) * 100
    tick, active, shutdown = 0, None, False
    outputs = [
        "joint_state",
        *["rgb_" + c for c in model.cameras],
        "reset_done",
        "target_request",
        "episode_result",
        "sim_turn",
    ]
    (out / "nano-scene.json").write_text(
        json.dumps(
            {
                **model.description,
                "fixed_base": True,
                "locked_wheel_joints": world.locked,
                "fixture_urdf_sha256": world.fixture_sha256,
                "floor_z": world.floor_z,
                "camera_optics": "nominal, not hardware calibrated",
                "task_success_verifier": False,
            }
        )
        + "\n"
    )
    record = (out / "nano-runtime.jsonl").open("w", buffering=1)

    def publish():
        nonlocal active
        stamp = TurnStamp(1, tick, tick * 10_000_000)
        active = BridgeTurn(stamp)
        metadata = {
            **stamp.metadata(),
            "env_id": 0,
            "goal_id": "nano-0",
            "robot_model_sha256": model.sha256,
        }
        counts = dict.fromkeys(outputs, 0)

        def emit(name, values, extra=None):
            send(name, values, {**metadata, **(extra or {})})
            counts[name] += 1

        if tick == 0:
            emit("reset_done", pa.array([1], type=pa.uint32()))
            emit(
                "target_request",
                pa.array([json.dumps({"instruction": os.environ["AISLE_NANO_INSTRUCTION"]})]),
            )
        q = world.joints()
        emit("joint_state", pa.array(q, type=pa.float32()), {"joint_names": list(model.names)})
        record.write(
            json.dumps({"event": "joint_state", "sim_time_ns": stamp.sim_time_ns, "q": q.tolist()})
            + "\n"
        )
        if tick % 10 == 0:
            for name, rgb in world.images().items():
                emit("rgb_" + name, pa.array(rgb.reshape(-1)), {"h": 240, "w": 320, "enc": "rgb8"})
        if tick >= maximum:
            emit(
                "episode_result",
                pa.array(
                    [
                        json.dumps(
                            {
                                "goal_id": "nano-0",
                                "ok": False,
                                "status": "bringup_time_limit",
                                "task_success_evaluated": False,
                            }
                        )
                    ]
                ),
            )
        counts["sim_turn"] = 1
        send("sim_turn", pa.array([tick], type=pa.uint64()), watermark_metadata(stamp, counts))

    try:
        for event in node:
            if event.get("type") != "INPUT":
                continue
            name, metadata = event["id"], event.get("metadata") or {}
            commands = None
            if name == "tick" and active is None:
                publish()
            elif name == "joint_cmd":
                active.accept(name, event, metadata)
                if active.ready_to_commit:
                    commands = active.finish()
            elif name == "turn_commit":
                shutdown = metadata.get("shutdown") is True
                commands = active.commit(metadata)
            elif name == "policy_result":
                record.write(event["value"][0].as_py() + "\n")
            elif name == "guard_result":
                record.write(
                    json.dumps(
                        {
                            "event": "guard_violation",
                            "details": json.loads(event["value"][0].as_py()),
                        }
                    )
                    + "\n"
                )
            if commands is not None:
                for _, command in commands:
                    q = command["value"].to_numpy(zero_copy_only=False)
                    world.command(q)
                    record.write(
                        json.dumps(
                            {
                                "event": "guarded_command",
                                "q": q.tolist(),
                                "sim_time_ns": metadata["sim_time_ns"],
                            }
                        )
                        + "\n"
                    )
                if shutdown:
                    return
                world.scene.step()
                tick += 1
                publish()
    finally:
        record.close()


if __name__ == "__main__":
    main()
