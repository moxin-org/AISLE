"""Safety exposure ledger and analyzer over synthetic trace rows (SFE-1,
SFE-2, SFE-3, SFE-4, SFE-5, SFE-6, SFE-7, SFE-8, SFE-13, SFE-15; #351).

Rows are hand-built so every exposure rule is exercised on values fixed
independently of the implementation: the 0/8 one-sided exact upper bound is
1 - 0.05 ** (1 / 8) ~ 0.3123.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from aisle.harness.exposure import ExposureError, derive_ledger
from aisle.harness.exposure_report import analyze_ledgers
from aisle.nodes.budget_guard import fk_ee_pos, load_limits

pytestmark = pytest.mark.unit

LIMITS = load_limits("franka")
HOME = np.asarray(LIMITS.fallback_qpos, dtype=np.float32)
MEDS = ["target_med", "other_med"]
SEC = 1_000_000_000


class _Verifier:
    """Frozen judge geometry stand-in: tray is x in [0.5, 0.6]."""

    @staticmethod
    def initial_capture_barrier(latest_oracle_ns: int, reset_sim_ns: int) -> int:
        return reset_sim_ns

    @staticmethod
    def _box_pose(state, idx):
        block = np.asarray(state, dtype=np.float32).reshape(-1)[idx * 7 : idx * 7 + 7]
        return block[:3], block[3:7]

    @staticmethod
    def _center_inside_tray(pos, cfg) -> bool:
        return 0.5 <= float(pos[0]) <= 0.6


def _cfg_factory(timeout_s, initial):
    return SimpleNamespace(knock_epsilon_m=0.02, initial_positions=initial)


def _state(positions: list[list[float]]) -> list[float]:
    out = []
    for p in positions:
        out.extend([*p, 0.0, 0.0, 0.0, 1.0])
    return out


def _row(stamp_ns: int, seq: int, data=None, text=None) -> dict:
    return {"sim_time_ns": stamp_ns, "env_id": 0, "seq": seq, "data": data, "text": text}


def _rows() -> dict[str, list[dict]]:
    """Two episodes: ep-0000 grasps, delivers the target, and knocks the
    other box; ep-0001 has no result yet. One multi-joint clamp, one hold."""
    goal0 = {"target_med": "target_med", "timeout_s": 60.0, "seed": 3, "reset_sim_ns": 0}
    goal1 = {"target_med": "target_med", "timeout_s": 60.0, "seed": 4, "reset_sim_ns": 10 * SEC}
    result0 = {"status": "success", "failure": None, "t_end": 5.0, "seed": 3, "goal_id": "ep-0000"}
    legal = HOME.copy()
    illegal = HOME.copy()
    illegal[0] += 0.5  # velocity clamp on joint 0
    illegal[1] += 0.5  # and joint 1: one proposal, two violation rows
    clamped = HOME.copy()
    clamped[0] += LIMITS.max_step_arr[0]
    clamped[1] += LIMITS.max_step_arr[1]
    joint_cmd = [
        _row(1 * SEC, 1, legal.tolist()),
        _row(2 * SEC, 2, illegal.tolist()),
        _row(3 * SEC, 3, legal.tolist()),
        _row(11 * SEC, 4, legal.tolist()),
    ]
    joint_safe = [
        _row(1 * SEC, 1, legal.tolist()),
        _row(2 * SEC, 2, clamped.tolist()),
        _row(3 * SEC, 3, legal.tolist()),  # wall_timeout hold: output == last safe
        _row(11 * SEC, 4, legal.tolist()),
    ]
    violation = [
        _row(2 * SEC, 1, text=json.dumps({"reason": "velocity", "joint": 0, "seq": 1})),
        _row(2 * SEC, 2, text=json.dumps({"reason": "velocity", "joint": 1, "seq": 2})),
        _row(3 * SEC, 3, text=json.dumps({"reason": "wall_timeout", "seq": 3})),
    ]
    gripper = [
        _row(int(1.5 * SEC), 1, [0.0]),
        _row(int(2.5 * SEC), 2, [1.0]),
        _row(4 * SEC, 3, [0.0]),
    ]
    gripper += [_row(int(11.5 * SEC), 4, [1.0])]  # ep-0001: closes and never reopens
    oracle = [
        _row(int(0.5 * SEC), 1, _state([[0.3, 0.0, 0.1], [0.4, 0.0, 0.1]])),
        _row(int(3.5 * SEC), 2, _state([[0.55, 0.0, 0.1], [0.4, 0.0, 0.1]])),  # target enters tray
        _row(int(4.5 * SEC), 3, _state([[0.55, 0.0, 0.1], [0.45, 0.0, 0.1]])),  # other knocked 5 cm
        _row(int(10.5 * SEC), 4, _state([[0.3, 0.0, 0.1], [0.4, 0.0, 0.1]])),
        _row(int(12 * SEC), 5, _state([[0.3, 0.0, 0.1], [0.4, 0.0, 0.1]])),
    ]
    return {
        "episode_goal": [
            _row(int(0.1 * SEC), 1, text=json.dumps(goal0)),
            _row(int(10.1 * SEC), 2, text=json.dumps(goal1)),
        ],
        "episode_result": [_row(5 * SEC, 1, text=json.dumps(result0))],
        "joint_cmd": joint_cmd,
        "joint_cmd_safe": joint_safe,
        "gripper_cmd": gripper,
        "gripper_cmd_safe": gripper,
        "violation": violation,
        "oracle_state": oracle,
        "joint_state": [_row(1 * SEC, 1, HOME.tolist())],
    }


def _ledger(rows=None, session="run-a", evidence_kind="simulation") -> dict:
    return derive_ledger(
        session,
        rows or _rows(),
        limits=LIMITS,
        fk=fk_ee_pos,
        med_names=MEDS,
        judge_cfg_factory=_cfg_factory,
        verifier=_Verifier,
        source_map={"sha256:planner": {"class": "classical", "role": "planner"}},
        producers={"joint_cmd": "sha256:planner", "gripper_cmd": "sha256:planner"},
        instrument_hash="sha256:instrument",
        trace_hashes={"joint_cmd": "sha256:trace"},
        campaign_id="unit",
        evidence_kind=evidence_kind,
    )


def test_episode_begins_at_goal_even_without_a_result():
    """SFE-3 / SFE-2: both retained goal assignments are episodes; the one
    with no result is started and included but not completed."""
    episodes = _ledger()["episodes"]
    assert [e["goal_id"] for e in episodes] == ["ep-0000", "ep-0001"]
    assert episodes[0]["completed"] and episodes[0]["result"]["status"] == "success"
    assert episodes[1]["started"] and episodes[1]["included"] and not episodes[1]["completed"]


def test_multi_joint_clamp_is_one_proposal_with_one_decision_and_receipt():
    """SFE-4: two violation rows on one request stay one proposal; the
    wall-timeout row is a hold; every proposal has exactly one receipt."""
    joint = [p for p in _ledger()["proposals"] if p["channel"] == "joint_cmd"]
    assert [p["decision"] for p in joint] == ["pass", "clamp", "hold", "pass"]
    assert joint[1]["reasons"] == ["velocity"] and joint[1]["max_correction"] > 0
    assert all(p["receipt_id"].startswith("joint_cmd-safe-") for p in joint)
    assert all(p["controller_class"] == "classical" for p in joint)


def test_reconciliation_refuses_mismatched_proposals_and_outputs():
    """SFE-2 / SFE-4: a decision stream that does not match the proposal
    stream one-to-one is a refusal, never a partial ledger."""
    rows = _rows()
    rows["joint_cmd_safe"] = rows["joint_cmd_safe"][:-1]
    with pytest.raises(ExposureError, match="reconciliation failed"):
        _ledger(rows)


def test_manipulation_attempts_open_close_and_flag_incomplete():
    """SFE-3: an attempt opens on the first close after an arm-motion
    proposal and closes on the next open; the never-reopened attempt in
    ep-0001 is counted and flagged incomplete."""
    episodes = _ledger()["episodes"]
    assert len(episodes[0]["attempts"]) == 1 and episodes[0]["attempts"][0]["complete"]
    assert episodes[0]["attempts"][0]["close_reason"] == "open"
    assert len(episodes[1]["attempts"]) == 1 and not episodes[1]["attempts"][0]["complete"]


def test_deliveries_are_verifier_observed_entries_and_collisions_are_a_proxy():
    """SFE-3 / SFE-5: the target's tray entry is one delivery (not a wrong
    object); the knocked box is a pose-displacement proxy row that names
    its threshold and magnitude and marks contact instrumentation unmeasured."""
    episode = _ledger()["episodes"][0]
    assert len(episode["deliveries"]) == 1 and episode["wrong_object_events"] == []
    assert episode["deliveries"][0]["identity"] == "target_med"
    (collision,) = episode["collisions"]
    assert collision["instrument"] == "oracle_pose_displacement_proxy"
    assert collision["threshold_m"] == 0.02 and collision["magnitude_m"] == pytest.approx(0.05)
    assert collision["contact_instrumentation"] == "unmeasured"


def test_wrong_object_is_a_detected_outcome_row():
    """SFE-1 / SFE-3: a non-target entry is a wrong-object event under the
    verifier_semantic_detection layer, not a gateway decision."""
    rows = _rows()
    rows["oracle_state"][1] = _row(int(3.5 * SEC), 2, _state([[0.3, 0.0, 0.1], [0.55, 0.0, 0.1]]))
    ledger = _ledger(rows)
    assert len(ledger["episodes"][0]["wrong_object_events"]) == 1
    assert "verifier_semantic_detection" in ledger["layers"]
    assert ledger["evidence_kind"] == "simulation"


def test_analyzer_zero_event_bound_uses_the_session_unit():
    """SFE-8 / SFE-6 (issue #508): four exposed sessions with two included
    episodes each bound the wrong-object rate on 4 sessions, not 8 episodes;
    episode, delivery and attempt counts are reported descriptively."""
    report = analyze_ledgers([_ledger(session=f"run-{index}") for index in range(4)])
    assert report["ok"] is True
    primary = report["zero_event"]["wrong_object_primary"]
    assert primary["unit"] == "exposed_session_with_any_event"
    assert primary["denominator"] == 4 and primary["events"] == 0
    assert primary["upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 4), abs=1e-9)
    by_episode = report["zero_event"]["wrong_object_by_episode"]
    assert by_episode == {
        "events": 0,
        "denominator": 8,
        "unit": "included_episode_with_any_event",
        "descriptive": True,
    }
    assert report["zero_event"]["wrong_object_by_delivery"]["descriptive"] is True
    assert report["zero_event"]["wrong_object_by_delivery"]["denominator"] == 4
    session = report["sessions"][0]
    assert session["proposals"]["by_decision"] == {"pass": 6, "clamp": 1, "refuse": 0, "hold": 1}
    assert session["proposals"]["distinct_with_intervention"] == 2
    assert report["by_controller_class"]["classical"]["unit"] == "session"
    assert "prevention" in report["wording"]


def _with_extra_episodes(ledger: dict, copies: int) -> dict:
    """The same session with its episodes repeated: more nested opportunities,
    no additional independent exposure."""
    grown = copy.deepcopy(ledger)
    template = grown["episodes"][0]
    for index in range(copies):
        extra = copy.deepcopy(template)
        extra["goal_id"] = f"ep-extra-{index:04d}"
        extra["proposal_ids"] = []
        extra["attempts"] = []
        extra["deliveries"] = []
        extra["collisions"] = []
        extra["wrong_object_events"] = []
        grown["episodes"].append(extra)
    return grown


def test_more_episodes_in_a_session_do_not_tighten_the_primary_bound():
    """SFE-8: within-session replication is nested, so adding episodes to the
    same sessions leaves the primary denominator and bound unchanged."""
    base = [_ledger(session="a"), _ledger(session="b")]
    grown = [_with_extra_episodes(ledger, 10) for ledger in base]
    before = analyze_ledgers(base)["zero_event"]
    after = analyze_ledgers(grown)["zero_event"]
    assert after["wrong_object_primary"]["denominator"] == 2
    assert after["wrong_object_primary"]["upper_bound"] == pytest.approx(
        before["wrong_object_primary"]["upper_bound"], abs=1e-12
    )
    assert after["wrong_object_by_episode"]["denominator"] == 24
    assert before["wrong_object_by_episode"]["denominator"] == 4


def test_several_events_in_one_session_count_as_one_session_event():
    """SFE-8: the binary outcome is 'session with at least one event'; two
    wrong-object episodes in one session are one session event and two
    descriptive episode events."""
    hit = _ledger(session="hit")
    for episode in hit["episodes"]:
        episode["wrong_object_events"] = [{"delivery_id": f"{episode['goal_id']}-d0"}]
    report = analyze_ledgers([hit, _ledger(session="clean-1"), _ledger(session="clean-2")])
    primary = report["zero_event"]["wrong_object_primary"]
    assert (primary["events"], primary["denominator"]) == (1, 3)
    assert primary["status"] == "events_observed"
    assert report["zero_event"]["wrong_object_by_episode"]["events"] == 2


def test_sessions_without_an_included_episode_are_not_exposure():
    """SFE-8: a session whose episodes are all excluded contributes no
    exposure; with none exposed the analyzer reports no bound, never 0%."""
    excluded = _ledger(session="excluded")
    for episode in excluded["episodes"]:
        episode["included"] = False
    mixed = analyze_ledgers([excluded, _ledger(session="kept")])["zero_event"]
    assert mixed["wrong_object_primary"]["denominator"] == 1
    assert mixed["wrong_object_by_episode"]["denominator"] == 2
    none = analyze_ledgers([excluded])["zero_event"]["wrong_object_primary"]
    assert none["denominator"] == 0 and none["bound"] is None
    assert none["status"] == "no exposure" and "upper_bound" not in none


def test_analyzer_refuses_mixed_evidence_kinds_and_duplicate_sessions():
    """SFE-6 / SFE-15: simulation and hardware rows cannot be pooled;
    duplicate session ids fail reconciliation with ok:false."""
    mixed = analyze_ledgers([_ledger(session="a"), _ledger(session="b", evidence_kind="hardware")])
    assert mixed["ok"] is False and any("mixed evidence kinds" in e for e in mixed["errors"])
    duplicate = analyze_ledgers([_ledger(session="a"), _ledger(session="a")])
    assert duplicate["ok"] is False and any("duplicate session" in e for e in duplicate["errors"])


def test_report_hash_changes_when_a_raw_input_changes():
    """SFE-13: a changed raw ledger changes the report hash."""
    base = _ledger()
    changed = copy.deepcopy(base)
    changed["episodes"][0]["deliveries"][0]["wrong_object"] = True
    changed["episodes"][0]["wrong_object_events"] = [changed["episodes"][0]["deliveries"][0]]
    assert analyze_ledgers([base])["report_sha256"] != analyze_ledgers([changed])["report_sha256"]


def test_events_after_the_verdict_are_flagged_and_kept_out_of_the_primary_bound():
    """SFE-3 / SFE-8: displacement first observed after the verifier's
    verdict is retained as a descriptive row, not a scored-window collision."""
    rows = _rows()
    # ep-0000 verdict lands at t_end 5.0 s after its first post-barrier sample (0.5 s);
    # a knock at 6.0 s is after the verdict.
    rows["oracle_state"][2] = _row(int(4.5 * SEC), 3, _state([[0.55, 0.0, 0.1], [0.4, 0.0, 0.1]]))
    rows["oracle_state"].insert(
        3, _row(int(6.0 * SEC), 6, _state([[0.55, 0.0, 0.1], [0.45, 0.0, 0.1]]))
    )
    ledger = _ledger(rows)
    (collision,) = ledger["episodes"][0]["collisions"]
    assert collision["after_terminal_result"] is True
    report = analyze_ledgers([ledger])
    assert report["zero_event"]["collision_proxy_primary"]["events"] == 0
    assert report["zero_event"]["collision_proxy_after_terminal_result"]["events"] == 1


def test_cli_reads_gzip_ledgers_and_reports_con8(tmp_path):
    """CON-8 / SFE-6: the analyzer CLI accepts gzip ledgers, prints JSON,
    exits 0 iff the report reconciles."""
    import gzip

    from cli_helpers import run_module

    good = tmp_path / "a.json.gz"
    good.write_bytes(gzip.compress(json.dumps(_ledger(session="a")).encode()))
    proc = run_module("aisle.harness.cli", "exposure", "analyze", "--ledger", str(good))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ok"] is True
    proc = run_module(
        "aisle.harness.cli", "exposure", "analyze", "--ledger", str(good), "--ledger", str(good)
    )
    assert proc.returncode == 1 and json.loads(proc.stdout)["ok"] is False


def test_controller_class_comes_from_content_hash_and_unknown_stays_visible():
    """SFE-7: the producer is classified by the content hash in the frozen
    source map, never by filename; an unlisted hash is `unknown`, reported
    separately rather than pooled."""
    classical = _ledger(session="a")
    assert {p["controller_class"] for p in classical["proposals"]} == {"classical"}
    unknown = derive_ledger(
        "b",
        _rows(),
        limits=LIMITS,
        fk=fk_ee_pos,
        med_names=MEDS,
        judge_cfg_factory=_cfg_factory,
        verifier=_Verifier,
        source_map={"sha256:planner": {"class": "classical", "role": "planner"}},
        producers={"joint_cmd": "sha256:other-revision", "gripper_cmd": "sha256:planner"},
        instrument_hash="sha256:instrument",
        trace_hashes={},
        campaign_id="unit",
    )
    joint = [p for p in unknown["proposals"] if p["channel"] == "joint_cmd"]
    assert {p["controller_class"] for p in joint} == {"unknown"}
    report = analyze_ledgers([unknown])
    assert report["by_controller_class"]["unknown"]["valid"] == 4
    assert report["by_controller_class"]["classical"]["valid"] == 4
