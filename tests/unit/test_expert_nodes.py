"""Unit tests for the T0 expert pipeline's pure cores (CAP-5 manifests:
oracle-pose, grasp-planner-topdown, task-state-machine) — no dora, no sim
(CON-12)."""

import numpy as np
import pytest

from aisle.nodes.grasp_topdown import (
    HAND_MOUNT_YAW,
    PLACE_DROP_GAP,
    finger_yaw_of,
    neighbour_constraints,
    plan_grasp,
    topdown_quat,
    yaw_of,
)
from aisle.nodes.ik_trajectory import park_read_reply, quat_to_rotation
from aisle.nodes.oracle_pose import select_pose
from aisle.nodes.task_state_machine import TaskStateMachine
from aisle.scenes.pharmacy import MED_NAMES

pytestmark = pytest.mark.unit


def make_poses(n=5):
    blocks = []
    for i in range(n):
        blocks.extend([0.5, -0.1 + 0.06 * i, 0.10, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(blocks, dtype=np.float32)


def assert_straddle(quat_xyzw, expected_yaw):
    """Grip-axis equality mod pi, wrap-tolerant (a -1e-9 finger yaw must
    equal 0, not pi)."""
    assert abs(np.sin(finger_yaw_of(quat_xyzw) - expected_yaw)) < 1e-5


class TestOraclePose:
    def test_selects_target_block_by_med_name(self):
        """CAP-5 oracle-pose: target_pose is the 7-float block of the
        requested med, in scene-manifest order (TC-1)."""
        poses = make_poses()
        for i, name in enumerate(MED_NAMES):
            block = select_pose(poses, name)
            assert block.shape == (7,)
            assert block[1] == pytest.approx(-0.1 + 0.06 * i)

    def test_unknown_med_is_refused(self):
        with pytest.raises(ValueError, match="unknown"):
            select_pose(make_poses(), "aspirin")


class TestGraspTopdown:
    def test_topdown_quat_points_flange_down_with_hand_mount_offset(self):
        """The grasp orientation points the flange z-axis DOWN, and the
        FLANGE yaw carries the hand-mount compensation (issue #92): the
        Franka hand is mounted -45 degrees from the flange about z, so
        the flange target shifts by HAND_MOUNT_YAW to realize the
        planner's grip axis. The uncompensated quat made every
        'axis-aligned' plan a DIAGONAL pinch in the sim — the T10
        'diagonal detents' and the m0 seed-3 hand-corner topple both
        trace to it."""
        quat = topdown_quat(0.0)
        rot = quat_to_rotation(quat)
        assert rot[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)  # flange z down
        assert yaw_of(quat) == pytest.approx(HAND_MOUNT_YAW, abs=1e-6)
        assert finger_yaw_of(quat) == pytest.approx(0.0, abs=1e-6)
        # the two orientation builders must agree for every yaw — grasp
        # (quat) and place/transfer (matrix) paths share the convention.
        # The PHYSICAL offset itself is gated by the Genesis measurement
        # in tests/sim/test_hand_mount.py (PR #93 review: these algebraic
        # relations alone hold for any HAND_MOUNT_YAW value)
        from aisle.nodes.ik_trajectory import topdown_rotation

        for yaw in (0.0, 0.5, np.pi / 2, -1.2):
            assert quat_to_rotation(topdown_quat(yaw)) == pytest.approx(
                topdown_rotation(yaw), abs=1e-9
            )

    def test_grasp_at_top_section_with_yaw(self):
        """CAP-5 grasp-planner-topdown: TCP at the box's TOP section
        (center + half height - grip engagement); with tilt=0 the FINGER
        yaw follows the box yaw so the fingers straddle the narrow
        axis."""
        yaw = 0.5
        quat = (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
        target = np.array([0.5, -0.1, 0.10, *quat], dtype=np.float32)
        size = (0.055, 0.035, 0.090)  # y narrower: yaw unchanged
        grasp, approach, place_z = plan_grasp(target, size, grip=0.025, tray_top_z=0.04)
        assert approach == pytest.approx(0.15)
        # release TCP: tray top + hanging box length + drop gap
        assert place_z == pytest.approx(0.04 + (0.090 - 0.025) + PLACE_DROP_GAP, abs=1e-6)
        assert grasp[:3] == pytest.approx([0.5, -0.1, 0.10 + 0.045 - 0.025], abs=1e-6)
        assert_straddle(grasp[3:], yaw)

    def test_narrow_x_axis_rotates_grip(self):
        """Fingers travel the gripper y-axis: when the box's x side is the
        narrower one, the grasp FINGER yaw turns 90 degrees to straddle
        it."""
        target = np.array([0.5, -0.1, 0.10, 0, 0, 0, 1], dtype=np.float32)
        grasp, _, _ = plan_grasp(target, (0.030, 0.065, 0.110), tray_top_z=0.04)
        assert_straddle(grasp[3:], np.pi / 2)

    def test_grip_axis_avoids_a_close_neighbour(self):
        """A same-level neighbour within the default finger sweep flips the
        grip 90 degrees onto the clearer axis (t10-m0-full seed 8): a box
        offset mostly in y is grasped across x so the open fingers stay
        clear, and with no neighbour the legacy narrow-axis grip stands."""
        # near-square target; neighbour ~0.085 m away in y, ~0 in x
        target = np.array([0.379, 0.20, 0.10, 0, 0, 0, 1], dtype=np.float32)
        size = (0.050, 0.045, 0.085)  # y is the narrower default straddle
        legacy, _, _ = plan_grasp(target, size, tray_top_z=0.04)
        assert_straddle(legacy[3:], 0.0)  # straddles y
        neighbours = [[0.394, 0.115, 0.0325, 0.015]]  # cetirizine, y-offset
        aware, _, _ = plan_grasp(
            target,
            size,
            tray_top_z=0.04,
            neighbours=neighbours,
            finger_open=0.04,
            finger_clear=0.008,
        )
        assert_straddle(aware[3:], np.pi / 2)  # flips to x

    def test_elongated_box_keeps_narrow_grip_despite_neighbour(self):
        """An elongated med can only be gripped across its narrow face; a
        neighbour on that axis must NOT force an infeasible wide-axis grip."""
        target = np.array([0.5, 0.0, 0.10, 0, 0, 0, 1], dtype=np.float32)
        size = (0.070, 0.035, 0.095)  # x half 0.035 > finger_open - finger_clear
        neighbours = [[0.5, 0.09, 0.0175, 0.0175]]  # tempts a flip to x
        grasp, _, _ = plan_grasp(
            target,
            size,
            tray_top_z=0.04,
            neighbours=neighbours,
            finger_open=0.04,
            finger_clear=0.008,
        )
        assert_straddle(grasp[3:], 0.0)  # stays narrow (y)

    def test_front_quat_fingers_straddle_horizontally(self):
        """Issue #92 follow-up (front-mode mount compensation): the HAND
        realized by FRONT_QUAT — flange rotation composed with the local
        Rz(HAND_MOUNT_YAW) mount — must keep the approach axis +x AND the
        finger-travel axis (hand y) HORIZONTAL. The uncompensated quat
        executed the fingers 45 degrees diagonal in the y-z plane
        (measured in Genesis), the front-mode twin of the m0 seed-3
        top-down bug; the physical gate is tests/sim/test_hand_mount.py."""
        import math

        a = HAND_MOUNT_YAW
        rz = np.array(
            [
                [math.cos(a), -math.sin(a), 0.0],
                [math.sin(a), math.cos(a), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        from aisle.nodes.grasp_topdown import FRONT_QUAT

        hand = quat_to_rotation(FRONT_QUAT) @ rz
        assert hand[:, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)  # into the shelf
        assert abs(hand[2, 1]) < 1e-6  # finger travel horizontal (no z tilt)

    def test_front_mode_approaches_horizontally(self):
        """ADR-10: a box under a board is grasped from the shelf FRONT —
        wrist horizontal (approach axis +x), TCP at the box center, and
        the approach distance spans from the front clearance point."""
        target = np.array([0.55, -0.11, 0.11, 0, 0, 0, 1], dtype=np.float32)
        grasp, approach, _ = plan_grasp(
            target, (0.055, 0.035, 0.090), front=True, shelf_front_x=0.40, tray_top_z=0.04
        )
        # z rides up to box_bottom + wrist clearance (0.065 + 0.065 = 0.13),
        # capped at box_top - finger engagement
        assert grasp[:3] == pytest.approx([0.55, -0.11, 0.13], abs=1e-6)
        axis = quat_to_rotation(grasp[3:])[:, 2]
        assert axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)  # into the shelf
        assert approach == pytest.approx(0.55 - (0.40 - 0.06), abs=1e-6)


class TestReadParkReply:
    def test_usable_stamp_arms_strictly_newer_frame_barrier(self):
        """CON-5/TC-2: a completed park with a usable contract stamp
        carries that stamp into the reader request."""
        payload = park_read_reply({"ok": True, "range_m": 0.13}, {"sim_time_ns": 42})
        assert payload == {"ok": True, "range_m": 0.13, "frame_after_sim_time_ns": 42}

    @pytest.mark.parametrize(
        "metadata", [{}, {"sim_time_ns": 0}, {"sim_time_ns": None}, {"sim_time_ns": "bad"}]
    )
    def test_unusable_stamp_refuses_instead_of_falling_back_unbarriered(self, metadata):
        """CON-5/TC-2 (PR #176 review): without a usable park stamp the
        executor must reply fail-closed. Omitting the barrier from an
        otherwise-successful reply reopens the stale-frame wall race."""
        assert park_read_reply({"ok": True, "range_m": 0.13}, metadata) == {
            "ok": False,
            "reason": "missing_park_stamp",
        }


class TestTaskStateMachine:
    def test_goal_emits_target_request_and_feedback_until_result(self):
        """CAP-5 task-state-machine + TC-7: a goal emits a target_request
        naming the med; ticks emit >=1 Hz feedback while active; the result
        ends the episode (feedback stops)."""
        machine = TaskStateMachine()
        out = machine.on_goal({"target_med": "ibuprofen", "timeout_s": 30}, "ep-1")
        assert out == [("target_request", {"target_med": "ibuprofen"}, {"goal_id": "ep-1"})]
        out = machine.on_tick()
        assert out == [
            (
                "episode_feedback",
                {"t": 1, "phase": "executing", "retries": 0},
                {"goal_id": "ep-1"},
            )
        ]
        assert machine.on_result() == []
        assert machine.on_tick() == []  # idle: no feedback

    def test_goal_while_active_is_refused(self):
        """TC-7: actions do not overlap — a second goal while one is active
        is refused (empty emission), the active episode continues."""
        machine = TaskStateMachine()
        machine.on_goal({"target_med": "ibuprofen"}, "ep-1")
        assert machine.on_goal({"target_med": "cetirizine"}, "ep-2") == []
        out = machine.on_tick()
        assert out[0][2] == {"goal_id": "ep-1"}

    def test_violations_are_counted_into_feedback(self):
        machine = TaskStateMachine()
        machine.on_goal({"target_med": "ibuprofen"}, "ep-1")
        machine.on_violation({"reason": "velocity"})
        machine.on_violation({"reason": "velocity"})
        out = machine.on_tick()
        assert out[0][1]["violations"] == {"velocity": 2}

    def test_feedback_t_is_deterministic_episode_ticks(self):
        """CON-5 (T08 review): feedback t counts 1 Hz ticks since the GOAL
        — no wall clock; a second episode restarts at 1."""
        machine = TaskStateMachine()
        machine.on_goal({"target_med": "ibuprofen"}, "ep-1")
        assert [machine.on_tick()[0][1]["t"] for _ in range(3)] == [1, 2, 3]
        machine.on_result()
        machine.on_goal({"target_med": "metformin"}, "ep-2")
        assert machine.on_tick()[0][1]["t"] == 1


class TestT2ScanTour:
    """T2 scan tour (design doc §3, idea I13): candidates from
    target_pose + neighbour rows; read_move -> move_done -> read_request
    -> read_result per candidate; a matching label promotes the
    candidate to grasp_target; refusals and mismatches advance; an
    exhausted tour idles (the episode times out honestly)."""

    META = {"target_med": "metformin", "neighbours": "[]", "sim_time_ns": 7}

    def _touring_machine(self):
        machine = TaskStateMachine(tier="T2")
        machine.on_goal({"target_med": "metformin", "target_sx": 0.10}, "ep-1")
        out = machine.on_target_pose([0.5, 0.1, 0.2], dict(self.META), [[0.5, -0.2], None])
        return machine, out

    def test_tour_starts_at_the_claimed_target_with_the_face_point(self):
        machine, out = self._touring_machine()
        assert out == [("read_move", {"face": [0.45, 0.1, 0.2]}, {"request_id": "ep-1/read0.0"})]

    def test_none_neighbour_rows_are_not_candidates(self):
        machine, _ = self._touring_machine()
        assert len(machine.candidates) == 2  # target + one real row

    def test_move_done_asks_the_reader_relaying_the_camera_pose(self):
        """CON-5/TC-2: the read request carries the completed park's
        sim-time barrier so downstream frame selection cannot depend on
        which queued wrist frame wins a wall-clock delivery race."""
        machine, _ = self._touring_machine()
        pose = {
            "face": [1, 2, 3],
            "cam_pos": [4, 5, 6],
            "cam_rot_cv": list(range(9)),
            "frame_after_sim_time_ns": 42,
        }
        out = machine.on_move_done(
            {"ok": True, "range_m": 0.16, "attempt_used": 0, **pose}, "ep-1/read0.0"
        )
        assert out == [("read_request", {"range_m": 0.16, **pose}, {"request_id": "ep-1/read0.0"})]

    def test_matching_read_promotes_candidate_to_grasp_target(self):
        machine, _ = self._touring_machine()
        machine.on_move_done({"ok": True, "range_m": 0.13}, "ep-1/read0.0")
        read_stamp = {"turn_epoch": 2, "turn_id": 9, "sim_time_ns": 90}
        out = machine.on_read_result(
            {"label": "metformin", "margin": 0.2}, "ep-1/read0.0", read_stamp
        )
        assert out[0][0] == "grasp_target"
        assert out[0][1] == {"pos": [0.5, 0.1, 0.2]}
        # the perception metadata is relayed so the grasp planner sees
        # the same keys target_pose carries, goal_id restamped (TC-7)
        assert out[0][2]["target_med"] == "metformin"
        assert out[0][2]["goal_id"] == "ep-1"
        # Transport provenance comes from the read that promoted the
        # candidate, never the target pose retained several turns earlier.
        assert {key: out[0][2][key] for key in read_stamp} == read_stamp

    def test_default_refusal_advances_one_park_per_candidate(self):
        """MAX_READS_PER_CANDIDATE = 1 by measurement: every correct
        offline read landed on the first tracked entry, and a retry's
        extra home->shelf transit knocked a box 3 cm (live `collision`,
        run 6). A refusal must advance, not re-park."""
        machine, _ = self._touring_machine()
        machine.on_move_done({"ok": True, "range_m": 0.16, "attempt_used": 3}, "ep-1/read0.0")
        out = machine.on_read_result({"label": None, "margin": 0.02}, "ep-1/read0.0")
        assert out == [("read_move", {"face": [0.45, -0.2, 0.2]}, {"request_id": "ep-1/read1.0"})]

    def test_refused_read_retries_from_the_next_ladder_entry_when_enabled(self, monkeypatch):
        """The retry MECHANISM (kept for loop agents to tune): with the
        budget raised, a refusal re-parks the same candidate starting
        past the attempt that refused, then advances."""
        from aisle.nodes import task_state_machine as module

        monkeypatch.setattr(module, "MAX_READS_PER_CANDIDATE", 2)
        machine, _ = self._touring_machine()
        machine.on_move_done({"ok": True, "range_m": 0.13, "attempt_used": 2}, "ep-1/read0.0")
        out = machine.on_read_result({"label": None, "margin": 0.01}, "ep-1/read0.0")
        assert out == [
            (
                "read_move",
                {"face": [0.45, 0.1, 0.2], "attempt_offset": 3},
                {"request_id": "ep-1/read0.1"},
            )
        ]
        machine.on_move_done({"ok": True, "range_m": 0.16, "attempt_used": 4}, "ep-1/read0.1")
        out = machine.on_read_result({"label": None, "margin": 0.02}, "ep-1/read0.1")
        # retries exhausted: next candidate, offset reset
        assert out == [("read_move", {"face": [0.45, -0.2, 0.2]}, {"request_id": "ep-1/read1.0"})]

    def test_foreign_read_advances_without_retry(self):
        """A DIFFERENT med read confidently is a truly different box —
        retrying it would waste tour budget."""
        machine, _ = self._touring_machine()
        machine.on_move_done({"ok": True, "range_m": 0.13, "attempt_used": 0}, "ep-1/read0.0")
        out = machine.on_read_result({"label": "ibuprofen", "margin": 0.3}, "ep-1/read0.0")
        assert out == [("read_move", {"face": [0.45, -0.2, 0.2]}, {"request_id": "ep-1/read1.0"})]
        machine.on_move_done({"ok": True, "range_m": 0.13, "attempt_used": 0}, "ep-1/read1.0")
        out = machine.on_read_result({"label": "ibuprofen", "margin": 0.3}, "ep-1/read1.0")
        assert out == []  # exhausted: idle, the episode times out honestly
        assert machine.candidates is None

    def test_unreachable_face_skips_the_candidate(self):
        machine, _ = self._touring_machine()
        out = machine.on_move_done({"ok": False}, "ep-1/read0.0")
        assert out == [("read_move", {"face": [0.45, -0.2, 0.2]}, {"request_id": "ep-1/read1.0"})]

    def test_stale_request_ids_are_ignored(self):
        machine, _ = self._touring_machine()
        assert machine.on_move_done({"ok": True, "range_m": 0.13}, "ep-1/read9.0") == []
        assert machine.on_read_result({"label": "metformin"}, "ep-1/read9.0") == []

    def test_t1_machine_never_tours(self):
        machine = TaskStateMachine()
        machine.on_goal({"target_med": "metformin"}, "ep-1")
        assert machine.on_target_pose([0.5, 0.1, 0.2], dict(self.META), []) == []

    def test_republished_target_pose_after_promotion_starts_no_second_tour(self):
        """Perception republishes target_pose every frame pair; a tour
        restart after promotion raced the grasp plan out of the executor
        ('one plan at a time') and the first live T2 episode closed
        never_grasped. One tour per goal."""
        machine, _ = self._touring_machine()
        machine.on_move_done({"ok": True, "range_m": 0.13}, "ep-1/read0.0")
        machine.on_read_result({"label": "metformin", "margin": 0.2}, "ep-1/read0.0")
        assert machine.on_target_pose([0.5, 0.1, 0.2], dict(self.META), []) == []
        # a NEW goal tours again
        machine.on_result()
        machine.on_goal({"target_med": "metformin", "target_sx": 0.10}, "ep-2")
        assert machine.on_target_pose([0.5, 0.1, 0.2], dict(self.META), []) != []

    def test_second_target_pose_does_not_restart_a_running_tour(self):
        machine, _ = self._touring_machine()
        assert machine.on_target_pose([0.9, 0.9, 0.9], dict(self.META), []) == []
        assert machine.candidates[0][0] == [0.5, 0.1, 0.2]

    def test_result_and_new_goal_clear_the_tour(self):
        machine, _ = self._touring_machine()
        machine.on_result()
        assert machine.candidates is None
        machine.on_goal({"target_med": "ibuprofen", "target_sx": 0.06}, "ep-2")
        out = machine.on_target_pose([0.4, 0.0, 0.2], dict(self.META), [])
        assert out == [("read_move", {"face": [0.37, 0.0, 0.2]}, {"request_id": "ep-2/read0.0"})]

    def test_scanning_phase_is_reported_in_feedback(self):
        machine, _ = self._touring_machine()
        assert machine.on_tick()[0][1]["phase"] == "scanning"

    def test_out_of_bounds_candidates_are_dropped_not_toured(self):
        """A candidate outside the shelf's occupiable volume is a garbage
        estimate, not a box: the first acceptance probe toured a phantom
        at z=0.38 / x=0.59 and the transit knocked a real box
        (`collision`, run 20260811-161222-dda648). With bounds set, the
        phantom is dropped and the tour starts at the surviving row."""
        bounds = {"x": (0.3, 0.5), "y": (-0.3, 0.3), "z": (0.05, 0.35)}
        machine = TaskStateMachine(tier="T2", candidate_bounds=bounds)
        machine.on_goal({"target_med": "metformin", "target_sx": 0.10}, "ep-1")
        out = machine.on_target_pose([0.59, -0.04, 0.38], dict(self.META), [[0.4, 0.1], None])
        assert len(machine.candidates) == 1  # phantom target dropped on x
        # the neighbour survives with the inherited garbage z CLAMPED
        # into the shelf band (the reader re-snaps z per hypothesis)
        assert out[0][1]["face"] == pytest.approx([0.35, 0.1, 0.35])

    def test_all_candidates_garbage_leaves_the_tour_unlatched(self):
        """Every row out of bounds: no tour, and the latch stays OPEN so
        a later, sane estimate can still start one."""
        bounds = {"x": (0.3, 0.5), "y": (-0.3, 0.3), "z": (0.05, 0.35)}
        machine = TaskStateMachine(tier="T2", candidate_bounds=bounds)
        machine.on_goal({"target_med": "metformin", "target_sx": 0.10}, "ep-1")
        assert machine.on_target_pose([0.59, -0.04, 0.38], dict(self.META), []) == []
        assert machine.candidates is None
        out = machine.on_target_pose([0.4, 0.1, 0.2], dict(self.META), [])
        assert out[0][0] == "read_move"  # sane estimate tours


class TestInContextRetries:
    """HAR-3: plan_done with no verdict arms a retry after a grace
    window; the retry count rides in feedback; max_retries bounds it;
    pass@8 is in-context, never best-of-8 independent episodes."""

    def _machine(self, max_retries=8):
        from aisle.nodes.task_state_machine import TaskStateMachine

        machine = TaskStateMachine(max_retries=max_retries)
        machine.on_goal({"target_med": "ibuprofen"}, "ep-1")
        return machine

    def _tick_until_retry(self, machine, limit=10):
        from aisle.nodes.task_state_machine import RETRY_GRACE_TICKS

        for _ in range(RETRY_GRACE_TICKS + 1):
            out = machine.on_tick()
            if out and out[0][0] == "target_request":
                return out
        return []

    def test_plan_done_without_verdict_retries_after_grace(self):
        machine = self._machine()
        machine.on_plan_done()
        out = self._tick_until_retry(machine)
        assert out[0] == (
            "target_request",
            {"target_med": "ibuprofen"},
            {"goal_id": "ep-1"},
        )
        # the same tick's feedback carries the incremented count
        assert out[1][0] == "episode_feedback"
        assert out[1][1]["retries"] == 1

    def test_no_retry_before_the_grace_window_elapses(self):
        """The grace pin: a retry inside the window could yank the
        DELIVERED box out of the tray while the verdict is in flight —
        the ticks BEFORE retry_due must emit feedback only."""
        from aisle.nodes.task_state_machine import RETRY_GRACE_TICKS

        assert RETRY_GRACE_TICKS >= 2  # oracle verdict latency headroom
        machine = self._machine()
        machine.on_plan_done()
        for _ in range(RETRY_GRACE_TICKS - 1):
            out = machine.on_tick()
            assert [topic for topic, _, _ in out] == ["episode_feedback"]
        assert machine.on_tick()[0][0] == "target_request"

    def test_verdict_within_grace_cancels_the_retry(self):
        """A success verdict lands ~1 tick after the plan finishes;
        retrying anyway could yank the DELIVERED box from the tray."""
        machine = self._machine()
        machine.on_plan_done()
        machine.on_result()
        machine.on_goal({"target_med": "metformin"}, "ep-2")
        for _ in range(8):
            out = machine.on_tick()
            assert all(topic != "target_request" for topic, _, _ in out)
        assert machine.retries == 0

    def test_max_retries_bounds_the_loop(self):
        machine = self._machine(max_retries=2)
        for expected in (1, 2):
            machine.on_plan_done()
            out = self._tick_until_retry(machine)
            assert out and out[1][1]["retries"] == expected
        machine.on_plan_done()  # budget spent: no third retry
        assert self._tick_until_retry(machine) == []
        assert machine.retries == 2

    def test_default_zero_is_one_attempt(self):
        """ADR-10 compatibility: max_retries=0 never retries — the graph
        must opt in (AISLE_MAX_RETRIES, attested)."""
        machine = self._machine(max_retries=0)
        machine.on_plan_done()
        assert self._tick_until_retry(machine) == []

    def test_t2_retry_unlatches_the_tour(self):
        from aisle.nodes.task_state_machine import TaskStateMachine

        machine = TaskStateMachine(tier="T2", max_retries=8)
        machine.on_goal({"target_med": "metformin", "target_sx": 0.10}, "ep-1")
        meta = {"target_med": "metformin", "neighbours": "[]"}
        machine.on_target_pose([0.5, 0.1, 0.2], dict(meta), [])
        machine.on_move_done({"ok": True, "range_m": 0.13, "attempt_used": 0}, "ep-1/read0.0")
        machine.on_read_result({"label": "metformin", "margin": 0.2}, "ep-1/read0.0")
        assert machine.toured is True
        machine.on_plan_done()
        self._tick_until_retry(machine)
        assert machine.toured is False  # a fresh estimate starts a fresh tour
        assert machine.on_target_pose([0.5, 0.1, 0.2], dict(meta), []) != []

    def test_retry_count_resets_per_goal(self):
        machine = self._machine()
        machine.on_plan_done()
        self._tick_until_retry(machine)
        machine.on_result()
        machine.on_goal({"target_med": "metformin"}, "ep-2")
        assert machine.retries == 0
        assert machine.on_tick()[0][1]["retries"] == 0


class TestNeighbourConstraints:
    """The `neighbours` payload contract between both pose sources and the
    grasp planner (TC-9): positional against MED_NAMES, with None slots for
    neighbours the L1 estimator refused. Review finding: the planner's old
    inline strict zip crashed on the shorter list L1's refusal path used to
    publish — these tests drive the CONSUMER with the producer's payloads."""

    # DISTINCT per-med sizes (round-2 review): a uniform fixture let an
    # implementation that reads the TARGET's size for every neighbour pass —
    # per-slot half-extents below pin the per-NAME lookup the code performs
    MEDS = {
        name: {"size": [0.04 + 0.01 * i, 0.03 + 0.01 * i, 0.08]} for i, name in enumerate(MED_NAMES)
    }

    def full_rows(self):
        return [[0.5, -0.1 + 0.06 * i] for i in range(len(MED_NAMES))]

    def test_l0_full_payload_yields_every_non_target_neighbour(self):
        rows = self.full_rows()
        out = neighbour_constraints(rows, MED_NAMES[1], MED_NAMES, self.MEDS)
        assert len(out) == len(MED_NAMES) - 1
        assert [c[:2] for c in out] == [r for i, r in enumerate(rows) if i != 1]
        assert [c[2:4] for c in out] == [
            [pytest.approx(0.02 + 0.005 * i), pytest.approx(0.015 + 0.005 * i)]
            for i in range(len(MED_NAMES))
            if i != 1
        ]
        # the fifth column is the neighbour's own height (palm clearance)
        assert [c[4] for c in out] == [0.08] * (len(MED_NAMES) - 1)

    def test_l1_refused_slot_is_omitted_from_constraints_not_unpacked(self):
        """A None row is a REFUSED neighbour (mask under the occlusion
        floor): it must drop out of the constraints — a permissive plan —
        rather than crash the planner's parse."""
        rows = self.full_rows()
        rows[2] = None
        out = neighbour_constraints(rows, MED_NAMES[1], MED_NAMES, self.MEDS)
        assert len(out) == len(MED_NAMES) - 2

    def test_a_short_payload_is_a_producer_bug_and_fails_loudly(self):
        """strict=True is the drift alarm: a producer that silently DROPS a
        slot (the pre-review L1 behaviour) must fail the parse loudly, not
        misattribute the remaining centres to the wrong meds."""
        with pytest.raises(ValueError):
            neighbour_constraints(self.full_rows()[:-1], MED_NAMES[0], MED_NAMES, self.MEDS)


class TestT4Dialogue:
    """ADR-32 increment one: in T4 the task arrives as dialogue — the
    confirm exchange must complete before any target_request, a
    pre-delivery correction switches the active target and re-confirms
    once, and corrections are counted in dialogue_corrections, never as
    HAR-3 retries."""

    def _machine(self, max_retries=8):
        return TaskStateMachine(tier="T4", max_retries=max_retries)

    def _open(self, machine, med="ibuprofen", goal_id="ep-1"):
        return machine.on_human_msg({"kind": "request", "med": med}, goal_id)

    def test_request_opens_episode_and_confirms_without_target_request(self):
        """ADR-32 §2: on `request` the machine emits robot_msg confirm
        naming the requested med and WAITS — no target_request yet."""
        machine = self._machine()
        out = self._open(machine)
        assert [(topic, payload["kind"], payload["med"]) for topic, payload, _ in out] == [
            ("robot_msg", "confirm", "ibuprofen")
        ]
        assert out[0][2] == {"goal_id": "ep-1"}
        assert machine.goal == {"target_med": "ibuprofen"}

    def test_confirm_reply_releases_target_request(self):
        machine = self._machine()
        self._open(machine)
        out = machine.on_human_msg({"kind": "confirm_reply", "med": "ibuprofen"}, "ep-1")
        assert out == [("target_request", {"target_med": "ibuprofen"}, {"goal_id": "ep-1"})]

    def test_correction_switches_target_and_reconfirms_once(self):
        """ADR-32 §2: a pre-delivery correction switches the active
        target to B and re-confirms; the release then requests B."""
        machine = self._machine()
        self._open(machine)
        out = machine.on_human_msg({"kind": "correction", "med": "cetirizine"}, "ep-1")
        assert [(topic, payload["med"]) for topic, payload, _ in out] == [
            ("robot_msg", "cetirizine")
        ]
        assert machine.goal["target_med"] == "cetirizine"
        out = machine.on_human_msg({"kind": "confirm_reply", "med": "cetirizine"}, "ep-1")
        assert out == [("target_request", {"target_med": "cetirizine"}, {"goal_id": "ep-1"})]

    def test_second_correction_is_ignored(self):
        """ADR-32 §2: the machine re-confirms ONCE — a second correction
        gets no reply and does not move the target."""
        machine = self._machine()
        self._open(machine)
        machine.on_human_msg({"kind": "correction", "med": "cetirizine"}, "ep-1")
        out = machine.on_human_msg({"kind": "correction", "med": "omeprazole"}, "ep-1")
        assert out == []
        assert machine.goal["target_med"] == "cetirizine"

    def test_corrections_count_in_feedback_not_retries(self):
        """ADR-32 §2: dialogue_corrections is its own counter in
        episode_feedback; retries keeps its HAR-3 meaning (pass@1/pass@8
        stay comparable across tiers)."""
        machine = self._machine()
        self._open(machine)
        machine.on_human_msg({"kind": "correction", "med": "cetirizine"}, "ep-1")
        feedback = machine.on_tick()[-1]
        assert feedback[0] == "episode_feedback"
        assert feedback[1]["dialogue_corrections"] == 1
        assert feedback[1]["retries"] == 0
        assert feedback[1]["phase"] == "confirming"

    def test_wrong_goal_and_overlapping_request_are_refused(self):
        """TC-7: episodes do not overlap; messages for another goal_id
        are not this episode's dialogue."""
        machine = self._machine()
        self._open(machine)
        assert machine.on_human_msg({"kind": "request", "med": "metformin"}, "ep-2") == []
        assert machine.on_human_msg({"kind": "confirm_reply", "med": "ibuprofen"}, "ep-2") == []
        assert machine.goal == {"target_med": "ibuprofen"}

    def test_plan_done_before_confirm_counts_protocol_violation(self):
        """ADR-32 §2: a delivery (finished grasp plan) without a completed
        confirm exchange is a dialogue protocol violation, counted in
        feedback — and arms no HAR-3 retry (no legitimate target_request
        existed to re-issue)."""
        machine = self._machine()
        self._open(machine)
        machine.on_plan_done()
        assert machine.retry_due_tick is None
        feedback = machine.on_tick()[-1]
        assert feedback[1]["violations"] == {"dialogue_protocol": 1}

    def test_retries_after_confirm_use_the_corrected_target(self):
        """HAR-3 in T4: an in-flight failure after the exchange re-issues
        target_request for the ACTIVE (corrected) target."""
        from aisle.nodes.task_state_machine import RETRY_GRACE_TICKS

        machine = self._machine()
        self._open(machine)
        machine.on_human_msg({"kind": "correction", "med": "cetirizine"}, "ep-1")
        machine.on_human_msg({"kind": "confirm_reply", "med": "cetirizine"}, "ep-1")
        machine.on_plan_done()
        out = []
        for _ in range(RETRY_GRACE_TICKS + 1):
            out = machine.on_tick()
            if out and out[0][0] == "target_request":
                break
        assert out[0] == ("target_request", {"target_med": "cetirizine"}, {"goal_id": "ep-1"})

    def test_result_resets_dialogue_state(self):
        """The next episode's request starts a clean exchange."""
        machine = self._machine()
        self._open(machine)
        machine.on_human_msg({"kind": "confirm_reply", "med": "ibuprofen"}, "ep-1")
        machine.on_result()
        out = self._open(machine, med="metformin", goal_id="ep-2")
        assert out[0][1] == {
            "kind": "confirm",
            "med": "metformin",
            "text": "you want the metformin, correct?",
        }
        assert machine.confirmed is False
        assert machine.dialogue_corrections == 0

    def test_non_t4_tiers_ignore_human_msg(self):
        """The dialogue path is tier-gated exactly like the T2 tour."""
        machine = TaskStateMachine(tier="T1")
        assert machine.on_human_msg({"kind": "request", "med": "ibuprofen"}, "ep-1") == []


class TestHybridFallback:
    """ENPIRE follow-up 2: the state machine owns the classical/VLA
    arbitration — first retry stays classical, later retries route to
    vla_request, and the two topics are mutually exclusive per retry."""

    def _machine(self, fallback=True):
        from aisle.nodes.task_state_machine import TaskStateMachine

        m = TaskStateMachine(max_retries=8, vla_fallback=fallback)
        m.on_goal({"target_med": "ibuprofen"}, "ep-1")
        return m

    def _retry_topic(self, machine):
        from aisle.nodes.task_state_machine import RETRY_GRACE_TICKS

        machine.on_plan_done()
        for _ in range(RETRY_GRACE_TICKS + 1):
            out = machine.on_tick()
            if out and out[0][0] in ("target_request", "vla_request"):
                return out[0]
        return None

    def test_first_retry_stays_classical_then_routes_to_vla(self):
        machine = self._machine()
        first = self._retry_topic(machine)
        assert first[0] == "target_request"
        second = self._retry_topic(machine)
        assert second == ("vla_request", {"target_med": "ibuprofen"}, {"goal_id": "ep-1"})

    def test_without_the_flag_behavior_is_byte_identical(self):
        machine = self._machine(fallback=False)
        assert self._retry_topic(machine)[0] == "target_request"
        assert self._retry_topic(machine)[0] == "target_request"


class TestReturnItemFlow:
    """T4 inc-2: a request carrying return_med makes the confirmed
    machine emit return_request BEFORE target_request — the executor's
    one-plan-at-a-time rule serializes the return ahead of delivery."""

    def test_confirm_reply_emits_return_then_delivery(self):
        machine = TaskStateMachine(tier="T4")
        machine.on_human_msg(
            {"kind": "request", "med": "amoxicillin", "return_med": "metformin"}, "g"
        )
        out = machine.on_human_msg({"kind": "confirm_reply", "med": "amoxicillin"}, "g")
        assert [(e[0], e[1]) for e in out] == [
            ("return_request", {"return_med": "metformin"}),
            ("target_request", {"target_med": "amoxicillin"}),
        ]

    def test_plain_requests_emit_no_return(self):
        machine = TaskStateMachine(tier="T4")
        machine.on_human_msg({"kind": "request", "med": "amoxicillin"}, "g")
        out = machine.on_human_msg({"kind": "confirm_reply", "med": "amoxicillin"}, "g")
        assert [e[0] for e in out] == ["target_request"]
