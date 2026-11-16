"""task-state-machine node (CAP-5): episode sequencing per TC-7.

A goal opens an episode: emit one target_request naming the med, then
>=1 Hz episode_feedback until the verifier's episode_result closes it.
Violations from the guard are counted into the feedback.

In-context retries (HAR-3): when the executor reports a finished grasp
plan (`plan_done`) and no verdict has closed the episode within a grace
window, the attempt FAILED in-flight (box missed, slipped, was refused)
— the machine re-issues target_request, up to `max_retries` times, and
the retry count rides in every episode_feedback so the rollout client
can record honest pass@8 (never best-of-8 independent episodes). The
grace window is tick-based (CON-5): a success verdict lands within a
tick or two of plan completion, and retrying before it could re-grasp
the DELIVERED box out of the tray. max_retries=0 (the default) is
byte-identical to the old one-attempt behavior (ADR-10).

T2 tier (design doc §3, idea I13): the medicine is identified by its
printed label, so the machine runs a SCAN TOUR before the grasp.
Candidates are the detected target pose plus its neighbour rows (all
box positions — at T2 the detector's identity claims are color-derived
noise; only the positions are trusted). Per candidate: `read_move`
parks the wrist camera at the face (ik-trajectory answers `move_done`
with the ladder range actually used), `read_request` asks the label
reader, and a matching `read_result` promotes the candidate to
`grasp_target` — the same schema the grasp planner consumes as
target_pose, so the grasp pipeline is untouched. A refused read or a
non-matching label advances the tour; an exhausted tour idles and the
episode closes honestly on the verifier's timeout.

T4 tier (design doc §3, ADR-32): the task arrives as DIALOGUE, not as
the TC-7 goal — in T4 graphs `episode_goal` is not wired here at all
(validator rule DIALOGUE_GOAL_LEAK), so the policy never sees the
final corrected answer. On the human's `request`, the machine opens
the episode and emits `robot_msg` confirm naming the requested med,
then WAITS: `confirm_reply` releases the target_request; a pre-delivery
`correction` switches the active target and re-confirms once. Scripted
corrections count in `dialogue_corrections` — they are NOT HAR-3
retries (a correction is not a failure), so pass@1/pass@8 stay
comparable across tiers.
"""

from __future__ import annotations

READ_TIER = "T2"
DIALOGUE_TIER = "T4"
TURN_STAMP_KEYS = ("turn_epoch", "turn_id", "sim_time_ns")
# per candidate: the first read plus this-many-minus-one refusal retries
# from later ladder entries. ONE by measurement: every correct read in
# the offline tour landed on the candidate's FIRST tracked entry, while
# each extra park only added a home->shelf transit — and one such retry
# transit knocked a box 3 cm and closed the first clean live T2 episode
# `collision`. The retry mechanism stays for loop agents to tune.
MAX_READS_PER_CANDIDATE = 1
# ticks (1 Hz, sim-deterministic) between a finished grasp plan and the
# retry it triggers: the oracle judges at 30 Hz, so a SUCCESS verdict
# arrives within ~1 tick of the box settling — retrying sooner could
# yank the delivered box back out of the tray
RETRY_GRACE_TICKS = 3


class TaskStateMachine:
    """Pure core: each handler returns [(topic, payload, metadata), ...].
    Metadata carries goal_id (TC-7) and, for tour topics, request_id
    (TC-6 service pattern)."""

    def __init__(
        self,
        tier: str = "T1",
        candidate_bounds: dict | None = None,
        max_retries: int = 0,
        vla_fallback: bool = False,
    ) -> None:
        self.tier = tier
        self.max_retries = max_retries
        self.vla_fallback = vla_fallback
        self.retries = 0
        self.retry_due_tick: int | None = None  # armed by plan_done
        # T2: {x: (lo, hi), y: (lo, hi), z: (lo, hi)} — a candidate row
        # outside the shelf's occupiable volume is a garbage estimate,
        # not a box (the first acceptance probe toured a phantom at
        # z=0.38/x=0.59 and the transit knocked a real box: `collision`)
        self.candidate_bounds = candidate_bounds
        self.goal: dict | None = None
        self.goal_id: str | None = None
        self.violations: dict[str, int] = {}
        self.ticks = 0
        self.candidates: list | None = None  # [(pos, face)] tour order
        self.tour_idx = 0
        self.tour_meta: dict | None = None  # target_pose metadata to relay
        self.awaiting: str | None = None  # request_id in flight
        # perception republishes target_pose every frame pair; once a
        # tour has run for this goal it must NOT restart — a fresh
        # read_move racing the promoted grasp plan knocked the plan out
        # of the executor ("one plan at a time") and the first live T2
        # episode closed never_grasped
        self.toured = False
        # T4 dialogue (ADR-32): no target_request until the confirm
        # exchange completes; corrections are counted, never retries
        self.confirmed = False
        self.dialogue_corrections = 0
        self.return_med = None  # T4 inc-2: med to take back first
        self.motion_request: tuple[str, str] | None = None
        self.motion_serial = 0
        self.recovery_phase: str | None = None

    def _reset_tour(self) -> None:
        self.candidates = None
        self.tour_idx = 0
        self.tour_meta = None
        self.awaiting = None
        self.read_attempt = 0  # refusal retries spent on this candidate
        self.attempt_offset = 0  # ladder entry the next read_move starts at
        self.last_attempt_used = 0

    def on_goal(self, goal: dict, goal_id: str) -> list:
        if self.goal is not None:  # TC-7: actions do not overlap
            return []
        self.goal, self.goal_id, self.violations = goal, goal_id, {}
        self.ticks = 0
        self.retries = 0
        self.retry_due_tick = None
        self._reset_tour()
        self.toured = False
        return [("target_request", {"target_med": goal["target_med"]}, {"goal_id": goal_id})]

    def on_tick(self) -> list:
        """Feedback t = 1 Hz ticks since the goal (CON-5: deterministic —
        a wall-clock read would make same-seed runs emit different
        payloads, and would span episodes rather than the current one).
        A due retry fires here: the grace window passed with no verdict."""
        if self.goal is None:
            return []
        self.ticks += 1
        emissions: list = []
        if self.retry_due_tick is not None and self.ticks >= self.retry_due_tick:
            self.retry_due_tick = None
            self.retries += 1
            self._reset_tour()
            self.toured = False  # T2: a fresh estimate starts a fresh tour
            # Hybrid fallback (ENPIRE follow-up 2, graph-attested via
            # AISLE_VLA_FALLBACK): the FIRST retry stays classical; from
            # the second failure on, the machine routes the task to the
            # VLA branch instead — the arbitration lives HERE, in the
            # decision node, so the two policies are mutually exclusive
            # by construction (never two writers to the arm).
            topic = "vla_request" if (self.vla_fallback and self.retries >= 2) else "target_request"
            if self.tier == DIALOGUE_TIER:
                emissions.extend(
                    self._request_motion(topic, {"target_med": self.goal["target_med"]})
                )
            else:
                emissions.append(
                    (topic, {"target_med": self.goal["target_med"]}, {"goal_id": self.goal_id})
                )
        if self.tier == DIALOGUE_TIER and not self.confirmed:
            phase = "confirming"
        elif self.candidates is not None:
            phase = "scanning"
        else:
            phase = "executing"
        feedback: dict = {"t": self.ticks, "phase": phase, "retries": self.retries}
        if self.tier == DIALOGUE_TIER:
            feedback["dialogue_corrections"] = self.dialogue_corrections
        if self.violations:
            feedback["violations"] = dict(self.violations)
        emissions.append(("episode_feedback", feedback, {"goal_id": self.goal_id}))
        return emissions

    def on_plan_done(self, goal_id=None, request_id=None) -> list:
        """The executor finished a grasp plan. If the verdict does not
        arrive within the grace window, the attempt failed in-flight —
        arm a retry (HAR-3), bounded by max_retries. In T4 a plan that
        finishes BEFORE the confirm exchange completed means motion ran
        without a confirmed task — a dialogue protocol violation (ADR-32
        §2), counted in feedback, and no retry (there was no legitimate
        target_request to re-issue)."""
        if self.tier == DIALOGUE_TIER:
            if self.motion_request is None or (goal_id, request_id) != self.motion_request:
                if (
                    self.goal is not None
                    and not self.confirmed
                    and goal_id is None
                    and request_id is None
                ):
                    self.violations["dialogue_protocol"] = (
                        self.violations.get("dialogue_protocol", 0) + 1
                    )
                return []
            # The old goal can score before the executor finishes its home
            # stage. Retain that motion boundary across the recovery dialogue.
            self.motion_request = None
            if self.goal is None or not self.confirmed:
                return []
            if self.recovery_phase == "waiting_previous":
                return self._begin_return()
            if self.recovery_phase == "returning":
                self.recovery_phase = None
                return self._request_motion(
                    "target_request", {"target_med": self.goal["target_med"]}
                )
        if self.goal is None:
            return []
        if self.retries >= self.max_retries:
            return []
        self.retry_due_tick = self.ticks + RETRY_GRACE_TICKS
        return []

    def on_result(self, goal_id=None) -> list:
        if goal_id is not None and goal_id != self.goal_id:
            return []
        self.goal = None
        self.retry_due_tick = None
        self._reset_tour()
        self.toured = False
        self.confirmed = False
        self.dialogue_corrections = 0
        self.return_med = None
        self.recovery_phase = None
        return []

    def _request_motion(self, topic: str, payload: dict) -> list:
        self.motion_serial += 1
        request_id = f"{self.goal_id}/{topic}/{self.motion_serial}"
        self.motion_request = (self.goal_id, request_id)
        return [(topic, payload, {"goal_id": self.goal_id, "request_id": request_id})]

    def _begin_return(self) -> list:
        self.recovery_phase = "returning"
        return self._request_motion("return_request", {"return_med": self.return_med})

    # -- T4 dialogue (ADR-32) -------------------------------------------

    def on_human_msg(self, payload: dict, goal_id: str) -> list:
        """The human spoke. `request` opens the episode (the task arrives
        as dialogue — episode_goal is not wired here in T4 graphs);
        `confirm_reply` releases the grasp pipeline; a pre-delivery
        `correction` switches the active target and re-confirms once."""
        if self.tier != DIALOGUE_TIER:
            return []
        kind = payload.get("kind")
        if kind == "request":
            if self.goal is not None:  # TC-7: episodes do not overlap
                return []
            med = payload["med"]
            self.goal, self.goal_id, self.violations = {"target_med": med}, goal_id, {}
            self.return_med = payload.get("return_med")
            self.motion_serial = 0
            self.recovery_phase = None
            if not self.return_med:
                # A fresh initial request starts the next physical episode.
                # A recovery request instead preserves the preceding motion.
                self.motion_request = None
            self.ticks = 0
            self.retries = 0
            self.retry_due_tick = None
            self._reset_tour()
            self.toured = False
            self.confirmed = False
            self.dialogue_corrections = 0
            return [
                (
                    "robot_msg",
                    {"kind": "confirm", "med": med, "text": f"you want the {med}, correct?"},
                    {"goal_id": goal_id},
                )
            ]
        if self.goal is None or goal_id != self.goal_id:
            return []
        if kind == "correction" and not self.confirmed and self.dialogue_corrections == 0:
            med = payload["med"]
            self.goal["target_med"] = med
            self.dialogue_corrections += 1
            return [
                (
                    "robot_msg",
                    {"kind": "confirm", "med": med, "text": f"you want the {med}, correct?"},
                    {"goal_id": goal_id},
                )
            ]
        if kind == "confirm_reply" and not self.confirmed:
            self.confirmed = True
            if self.return_med:
                if self.motion_request is not None:
                    self.recovery_phase = "waiting_previous"
                    return []
                return self._begin_return()
            return self._request_motion("target_request", {"target_med": self.goal["target_med"]})
        return []

    def on_violation(self, violation: dict) -> None:
        reason = violation.get("reason", "unknown")
        self.violations[reason] = self.violations.get(reason, 0) + 1

    # -- T2 scan tour ---------------------------------------------------

    def on_target_pose(self, pos: list, metadata: dict, neighbours: list) -> list:
        """Perception answered. T2: start the tour over every box
        position; other tiers never see this input (graph wiring)."""
        if self.tier != READ_TIER or self.goal is None or self.candidates is not None:
            return []
        if self.toured:
            return []  # one tour per goal (see __init__ note)
        self.toured = True
        target_sx = float(self.goal["target_sx"]) if "target_sx" in self.goal else None
        # neighbour rows are same-level (x, y) centres (TC-9 payload);
        # they inherit the target's z for the face point
        z = float(pos[2])
        rows = [list(pos)[:3]] + [
            [float(r[0]), float(r[1]), z] for r in neighbours if r is not None
        ]
        if self.candidate_bounds is not None:
            # x/y outside the shelf volume = garbage estimate, dropped;
            # z is CLAMPED instead — a garbage claimed z is inherited by
            # every neighbour row, and the reader re-snaps hypothesis z
            # to board geometry anyway
            bounds = self.candidate_bounds
            rows = [
                [row[0], row[1], min(max(row[2], bounds["z"][0]), bounds["z"][1])]
                for row in rows
                if bounds["x"][0] <= row[0] <= bounds["x"][1]
                and bounds["y"][0] <= row[1] <= bounds["y"][1]
            ]
        if not rows:
            self.toured = False  # nothing plausible yet: allow a fresh estimate
            return []
        # face point: the requested med's own x-depth is the best
        # available guess for EVERY candidate — identity is unknown
        # until read, and the read ladder tolerates the ~2 cm spread
        self.candidates = [
            (row, [row[0] - (target_sx or 0.0) / 2.0, row[1], row[2]]) for row in rows
        ]
        self.tour_idx = 0
        # Retain the perception's semantic annotations across the multi-turn
        # scan tour, but not its transport stamp.  A later matching read
        # promotes this candidate in the read-result turn; replaying the
        # original target-pose turn as the grasp output's stamp violates the
        # lockstep causal boundary (and gives non-lockstep consumers a stale
        # observation time).
        self.tour_meta = {
            key: value for key, value in metadata.items() if key not in TURN_STAMP_KEYS
        }
        return self._emit_read_move()

    def _emit_read_move(self) -> list:
        assert self.candidates is not None
        if self.tour_idx >= len(self.candidates):
            self._reset_tour()  # exhausted: idle; the episode times out honestly
            return []
        request_id = f"{self.goal_id}/read{self.tour_idx}.{self.read_attempt}"
        self.awaiting = request_id
        _, face = self.candidates[self.tour_idx]
        request = {"face": face}
        if self.attempt_offset:
            request["attempt_offset"] = self.attempt_offset
        return [("read_move", request, {"request_id": request_id})]

    def _advance_candidate(self) -> list:
        self.tour_idx += 1
        self.read_attempt = 0
        self.attempt_offset = 0
        return self._emit_read_move()

    def on_move_done(self, payload: dict, request_id: str) -> list:
        if self.candidates is None or request_id != self.awaiting:
            return []
        if not payload.get("ok", False):  # ladder exhausted: skip candidate
            return self._advance_candidate()
        self.last_attempt_used = int(payload.get("attempt_used", 0))
        request: dict = {"range_m": float(payload["range_m"])}
        if payload.get("pitched"):
            request["pitched"] = True  # the reader raises its margin floor
        for key in ("face", "cam_pos", "cam_rot_cv", "frame_after_sim_time_ns"):
            # the executor's ACHIEVED camera pose: the reader rectifies
            # the face quad through it (first live tour read a neighbour
            # box confidently wrong without it). Its sim-time barrier
            # selects a strictly later wrist frame independent of wall
            # delivery order.
            if payload.get(key) is not None:
                request[key] = payload[key]
        return [("read_request", request, {"request_id": request_id})]

    def on_read_result(
        self, payload: dict, request_id: str, event_metadata: dict | None = None
    ) -> list:
        if self.candidates is None or request_id != self.awaiting:
            return []
        if payload.get("label") == self.goal["target_med"]:
            pos, _ = self.candidates[self.tour_idx]
            metadata = dict(self.tour_meta or {})
            if event_metadata is not None:
                metadata.update(
                    {key: event_metadata[key] for key in TURN_STAMP_KEYS if key in event_metadata}
                )
            metadata["goal_id"] = self.goal_id
            self._reset_tour()
            return [("grasp_target", {"pos": pos}, metadata)]
        if payload.get("label") is None and self.read_attempt + 1 < MAX_READS_PER_CANDIDATE:
            # REFUSED (not misidentified): the view was bad, not the box —
            # retry this candidate from the next ladder entry (a far-side
            # box reads only from the pitched rungs)
            self.read_attempt += 1
            self.attempt_offset = self.last_attempt_used + 1
            return self._emit_read_move()
        # a DIFFERENT med (or refusals exhausted): keep touring
        return self._advance_candidate()


def main() -> None:
    import json
    import os
    import sys

    import numpy as np
    import pyarrow as pa

    from aisle.topics import env_accepts, env_pin_from_env, make_sender
    from aisle.turn_node import Node

    tier = os.environ.get("AISLE_TASK_TIER", "T1")
    if tier not in ("T1", READ_TIER, DIALOGUE_TIER):
        raise ValueError(
            f"AISLE_TASK_TIER must be T1, {READ_TIER} or {DIALOGUE_TIER}, got {tier!r}"
        )
    target_sx = None
    candidate_bounds = None
    if tier == READ_TIER:
        from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout

        meds = load_meds()
        target_sx = {name: float(spec["size"][0]) for name, spec in meds.items()}
        # the shelf's occupiable volume from public geometry (SCN-2),
        # padded by the largest med half-extent: candidate rows outside
        # it are garbage estimates, not boxes
        layout = resolve_layout(load_physics(), os.environ.get("AISLE_EMBODIMENT", "franka"))
        shelf = layout["shelf"]
        pad = max(max(float(v) for v in spec["size"]) for spec in meds.values())
        half_x, half_y = shelf["level_size"][0] / 2.0, shelf["level_size"][1] / 2.0
        z_lo = shelf["pos"][2] + shelf["level_heights"][0]
        z_hi = shelf["pos"][2] + shelf["level_heights"][-1] + shelf["board_thickness"] + pad
        candidate_bounds = {
            "x": (shelf["pos"][0] - half_x - pad, shelf["pos"][0] + half_x + pad),
            "y": (shelf["pos"][1] - half_y - pad, shelf["pos"][1] + half_y + pad),
            "z": (z_lo, z_hi),
        }

    env_pin = env_pin_from_env(os.environ)
    node = Node()
    send = make_sender(node, env_pin)
    # HAR-3: graph-declared (attested) retry budget; 0 = one attempt
    max_retries_raw = os.environ.get("AISLE_MAX_RETRIES", "0").strip()
    if not max_retries_raw.isdigit() or not 0 <= int(max_retries_raw) <= 8:
        raise ValueError(f"AISLE_MAX_RETRIES must be an int in [0, 8], got {max_retries_raw!r}")
    vla_fallback = os.environ.get("AISLE_VLA_FALLBACK", "").strip() in ("1", "true")
    machine = TaskStateMachine(
        tier=tier,
        candidate_bounds=candidate_bounds,
        max_retries=int(max_retries_raw),
        vla_fallback=vla_fallback,
    )
    lockstep = os.environ.get("AISLE_LOCKSTEP", "0").strip().lower() in ("1", "true", "yes")
    last_tick_sim_ns = -1

    def emit(emissions) -> None:
        for topic, payload, metadata in emissions:
            if topic == "grasp_target":
                send(
                    topic,
                    pa.array(np.asarray(payload["pos"] + [0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
                    metadata,
                )
            else:
                send(topic, pa.array([json.dumps(payload)]), metadata)

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        if not env_accepts(metadata, env_pin):
            continue  # fleet mode (BRG-5): another env's stream
        if event["id"] == "episode_goal":
            goal = json.loads(event["value"][0].as_py())
            if target_sx is not None and goal.get("target_med") in target_sx:
                goal["target_sx"] = target_sx[goal["target_med"]]
            emissions = machine.on_goal(goal, metadata.get("goal_id", ""))
            if not emissions:
                print(f"goal {metadata.get('goal_id')} refused: episode active", file=sys.stderr)
            emit(emissions)
            if lockstep:
                last_tick_sim_ns = int(metadata.get("sim_time_ns", 0))
        elif event["id"] == "episode_result":
            emit(machine.on_result(goal_id=metadata.get("goal_id", "")))
        elif event["id"] == "violation":
            machine.on_violation(json.loads(event["value"][0].as_py()))
        elif event["id"] == "tick" and not lockstep:
            emit(machine.on_tick())
        elif event["id"] == "turn" and lockstep:
            sim_time_ns = int(metadata.get("sim_time_ns", 0))
            if last_tick_sim_ns < 0:
                last_tick_sim_ns = sim_time_ns
            while sim_time_ns - last_tick_sim_ns >= 1_000_000_000:
                last_tick_sim_ns += 1_000_000_000
                emit(machine.on_tick())
        elif event["id"] == "plan_done":
            emit(machine.on_plan_done(metadata.get("goal_id"), metadata.get("request_id")))
        elif event["id"] == "target_pose":
            pos = (
                np.asarray(event["value"].to_numpy(zero_copy_only=False), dtype=np.float64)
                .reshape(-1)[:3]
                .tolist()
            )
            neighbours = json.loads(metadata.get("neighbours", "[]"))
            emit(machine.on_target_pose(pos, metadata, neighbours))
        elif event["id"] == "move_done":
            emit(
                machine.on_move_done(
                    json.loads(event["value"][0].as_py()), metadata.get("request_id", "")
                )
            )
        elif event["id"] == "read_result":
            emit(
                machine.on_read_result(
                    json.loads(event["value"][0].as_py()),
                    metadata.get("request_id", ""),
                    metadata,
                )
            )
        elif event["id"] == "human_msg":
            emit(
                machine.on_human_msg(
                    json.loads(event["value"][0].as_py()), metadata.get("goal_id", "")
                )
            )


if __name__ == "__main__":
    main()
