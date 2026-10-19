"""Hardware-phase artifacts that can be built before equipment exists
(HWP-1, HWP-2, HWP-3, HWP-5, HWP-6, HWP-7, HWP-9, HWP-11, HWP-13, HWP-14,
HWP-18, HWP-20; SPEC 520, issue #356).

Schemas and fail-closed validators for the station manifest, motor and
workspace calibration artifacts, driver telemetry rows, the interface map,
the safety case and preflight checklist, reset and intervention records,
the physical scorer contract, and the physical protocol; the fourteen-step
execution checklist with every step `pending`; and the analyzer that
regenerates the hardware report with every realized-hardware field
`hardware_pending`. A placeholder or simulated identity blocks physical
execution (HWP-1). Pure over dicts (CON-12).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

HARDWARE_PENDING = "hardware_pending"
PLACEHOLDER_MARKERS = ("placeholder", "simulated", "loopback", "todo", "tbd", "example")
CHECKLIST_STEPS = (
    "acquire and inventory station",
    "inspect and anchor workspace; verify physical estop",
    "pin firmware, software, power, and USB topology",
    "measure motor, workspace, camera, and time calibration; verify independently",
    "sign the safety case and preflight checklist",
    "pass no-load and representative-load limit checks",
    "pass watchdog, lease, held-command, estop, power-loss, and evidence-sink drills",
    "freeze the physical scorer and measure its fidelity against an independent audit",
    "freeze reset and human-intervention protocols",
    "freeze the physical protocol with power or precision target and randomization",
    "run the paired physical task instances in randomized blocks",
    "run the pre-registered blinded live-fault cell if the claim is kept",
    "regenerate the analyzer report and simulation-to-hardware deltas",
    "update the claim matrix and the public status",
)
STEP_STATUSES = ("pending", "passed", "failed")


class HardwarePhaseError(Exception):
    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []


def content_hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _missing(record: dict, required: tuple[str, ...]) -> list[str]:
    return [k for k in required if k not in record]


def _has_placeholder(value: Any) -> str | None:
    if isinstance(value, str):
        lowered = value.lower()
        return next((m for m in PLACEHOLDER_MARKERS if m in lowered), None)
    if isinstance(value, dict):
        for v in value.values():
            hit = _has_placeholder(v)
            if hit:
                return hit
    if isinstance(value, list):
        for v in value:
            hit = _has_placeholder(v)
            if hit:
                return hit
    return None


# ------------------------------------------------------------- HWP-1


STATION_FIELDS = (
    "station_id",
    "arm",
    "gripper",
    "actuators",
    "cameras",
    "firmware_hashes",
    "bus_usb_topology",
    "power_supply",
    "estop_hardware",
    "host",
    "runtime_hashes",
    "workspace_fixture_ids",
    "task_object_ids",
    "responsible_operators",
)


def validate_station_manifest(manifest: dict) -> dict:
    """HWP-1: every device pinned or explicitly hardware_pending; a
    placeholder or simulated identity blocks physical execution."""
    missing = _missing(manifest, STATION_FIELDS)
    if missing:
        raise HardwarePhaseError("station manifest incomplete", missing)
    pending = [k for k in STATION_FIELDS if manifest[k] == HARDWARE_PENDING]
    placeholder = _has_placeholder({k: v for k, v in manifest.items() if v != HARDWARE_PENDING})
    if placeholder:
        raise HardwarePhaseError("placeholder identity blocks physical execution", [placeholder])
    for device in ("arm", "gripper"):
        d = manifest[device]
        if d != HARDWARE_PENDING and not {"manufacturer", "model", "revision", "serial"} <= set(d):
            raise HardwarePhaseError(f"{device} identity incomplete")
    return {
        "physical_execution_allowed": not pending,
        "hardware_pending_fields": pending,
        "manifest_hash": content_hash(manifest),
    }


# ------------------------------------------------------------- HWP-2 / HWP-3


MOTOR_JOINT_FIELDS = (
    "zero",
    "direction",
    "range",
    "encoder_conversion",
    "backlash_repeatability",
    "safe_speed",
    "safe_current",
    "safe_temperature",
)
MOTOR_FIELDS = (
    "station_id",
    "version",
    "date",
    "method",
    "instrument_identity",
    "operator",
    "home_pose",
    "gripper_mapping",
    "max_payload",
    "tool_geometry",
    "joints",
    "expires",
    "signature",
    "artifact_hash",
)


def validate_motor_calibration(artifact: dict, *, station_id: str, today: str) -> dict:
    """HWP-2: the driver refuses a missing, wrong-station, expired,
    out-of-range, incomplete, unsigned, or hash-mismatched artifact before
    torque enable."""
    missing = _missing(artifact, MOTOR_FIELDS)
    if missing:
        raise HardwarePhaseError("motor calibration incomplete", missing)
    if artifact["station_id"] != station_id:
        raise HardwarePhaseError("motor calibration is for another station")
    if artifact["expires"] < today:
        raise HardwarePhaseError("motor calibration expired")
    if not artifact["signature"]:
        raise HardwarePhaseError("motor calibration unsigned")
    body = {k: v for k, v in artifact.items() if k not in ("artifact_hash", "signature")}
    if content_hash(body) != artifact["artifact_hash"]:
        raise HardwarePhaseError("motor calibration hash mismatch")
    for joint, fields in artifact["joints"].items():
        lacking = _missing(fields, MOTOR_JOINT_FIELDS)
        if lacking:
            raise HardwarePhaseError(f"joint {joint} calibration incomplete", lacking)
        lo, hi = fields["range"]
        if not lo < fields["zero"] < hi:
            raise HardwarePhaseError(f"joint {joint} zero outside its measured range")
    return {"torque_enable_allowed": True, "artifact_hash": artifact["artifact_hash"]}


WORKSPACE_FIELDS = (
    "station_id",
    "version",
    "camera_intrinsics",
    "overhead_extrinsics",
    "wrist_extrinsics",
    "depth_scale",
    "frames",
    "target_regions",
    "timestamp_latency_mapping",
    "measurement_uncertainty",
    "repeatability",
    "split",
    "instrument_identities",
    "operator",
)


def validate_workspace_calibration(artifact: dict) -> dict:
    """HWP-3: measured frames, uncertainty, repeatability, and a
    calibration/evaluation split; never simulator scene truth."""
    missing = _missing(artifact, WORKSPACE_FIELDS)
    if missing:
        raise HardwarePhaseError("workspace calibration incomplete", missing)
    if set(artifact["split"]) != {"calibration", "evaluation"} or set(
        artifact["split"]["calibration"]
    ) & set(artifact["split"]["evaluation"]):
        raise HardwarePhaseError("calibration and evaluation samples must be disjoint")
    if _has_placeholder(artifact["frames"]):
        raise HardwarePhaseError("workspace frames must be measured, not placeholders")
    return {"artifact_hash": content_hash(artifact)}


# ------------------------------------------------------------- HWP-5 / HWP-6 / HWP-7


INTERFACE_DIMENSIONS = ("observations", "actions", "cadence", "receipts", "authority")


def validate_interface_map(interface_map: dict) -> dict:
    """HWP-5: one content-addressed driver, adapters, primitives, gateway,
    limits, watchdog, and containment shared by both treatments; semantic
    parity across every dimension."""
    required = (
        "driver_hash",
        "camera_adapter_hashes",
        "primitive_hashes",
        "gateway_hash",
        "limits_hash",
        "watchdog_lease_hash",
        "containment_hash",
        "treatments",
    )
    missing = _missing(interface_map, required)
    if missing:
        raise HardwarePhaseError("interface map incomplete", missing)
    treatments = interface_map["treatments"]
    if set(treatments) != {"typed", "monolithic"}:
        raise HardwarePhaseError("interface map must cover typed and monolithic")
    for name, t in treatments.items():
        if t == HARDWARE_PENDING or t.get("status") == "dependency_pending":
            continue
        for dim in INTERFACE_DIMENSIONS:
            if t.get(dim) != treatments["typed"].get(dim):
                raise HardwarePhaseError(f"interface parity failed on {dim} for {name}")
    return {"map_hash": content_hash(interface_map)}


TELEMETRY_FIELDS = (
    "proposal_id",
    "gateway_decision",
    "receipt_id",
    "requested_positions",
    "transmitted_positions",
    "gripper_command",
    "joint_state",
    "encoder_state",
    "connection_state",
    "torque_state",
    "lease_state",
    "device_error",
    "source_sequence",
    "device_timestamp",
    "host_monotonic_ns",
    "clock_domain",
    "alignment_uncertainty_ns",
)


def validate_telemetry_row(row: dict) -> None:
    """HWP-6 / HWP-7: one proposal binds to zero or one receipt and its
    measured state; every row carries source sequence, device timestamp
    when available, host receipt, clock domain, and alignment uncertainty."""
    missing = _missing(row, TELEMETRY_FIELDS)
    if missing:
        raise HardwarePhaseError("telemetry row incomplete", missing)
    if row["gateway_decision"] not in ("pass", "clamp", "refuse", "hold"):
        raise HardwarePhaseError("telemetry gateway decision unknown")
    if row["gateway_decision"] in ("refuse", "hold") and row["receipt_id"] is not None:
        raise HardwarePhaseError("a refused or held proposal cannot carry a driver receipt")
    if not isinstance(row["host_monotonic_ns"], int) or row["host_monotonic_ns"] < 0:
        raise HardwarePhaseError("host monotonic receipt timestamp required")


def validate_telemetry_stream(rows: list[dict]) -> dict:
    receipts, last_ns = {}, -1
    for row in rows:
        validate_telemetry_row(row)
        if row["receipt_id"] is not None:
            if row["receipt_id"] in receipts:
                raise HardwarePhaseError("receipt bound to two proposals", [row["receipt_id"]])
            receipts[row["receipt_id"]] = row["proposal_id"]
        if row["host_monotonic_ns"] < last_ns:
            raise HardwarePhaseError("host receipt timestamps regress")
        last_ns = row["host_monotonic_ns"]
    return {
        "rows": len(rows),
        "receipts": len(receipts),
        "evidence_kind": rows[0].get("evidence_kind") if rows else None,
    }


# ------------------------------------------------------------- HWP-9 / HWP-11 / HWP-13 / HWP-14


SAFETY_CASE_FIELDS = (
    "safety_operator",
    "trial_conductor",
    "software_controller",
    "torque_enable_authority",
    "start_stop_authority",
    "estop_operate_authority",
    "estop_reset_authority",
    "power_restore_authority",
    "keep_out_zones",
    "anchoring",
    "payload_limits",
    "hazards",
    "speed_current_temperature_limits",
    "preflight_checklist",
    "signatures",
)


def validate_safety_case(case: dict) -> dict:
    """HWP-9: named roles, mapped authorities, declared hazards, a signed
    preflight checklist; an unsigned case blocks."""
    missing = _missing(case, SAFETY_CASE_FIELDS)
    if missing:
        raise HardwarePhaseError("safety case incomplete", missing)
    if not case["signatures"] or any(not s.get("signed_at") for s in case["signatures"]):
        raise HardwarePhaseError("safety case unsigned")
    if any(
        item.get("status") not in ("pending", "passed", "failed")
        for item in case["preflight_checklist"]
    ):
        raise HardwarePhaseError("preflight checklist items need pending/passed/failed")
    return {
        "case_hash": content_hash(case),
        "preflight_all_passed": all(i["status"] == "passed" for i in case["preflight_checklist"]),
    }


INTERVENTION_FIELDS = ("kind", "start_ns", "end_ns", "actor", "reason", "scoring_consequence")
INTERVENTION_KINDS = (
    "workspace_entry",
    "touch",
    "object_move",
    "cable_or_power",
    "estop",
    "rescue",
    "reset",
    "annotation",
)


def validate_intervention_records(records: list[dict], protocol: dict) -> dict:
    """HWP-11: every human action stamped, attributed, reasoned, and mapped
    to a frozen scoring or exclusion consequence."""
    for key in ("allowed_reset_steps", "max_attempts", "timeout_s", "roles", "permitted_tools"):
        if key not in protocol:
            raise HardwarePhaseError("reset protocol incomplete", [key])
    for r in records:
        missing = _missing(r, INTERVENTION_FIELDS)
        if missing:
            raise HardwarePhaseError("intervention record incomplete", missing)
        if r["kind"] not in INTERVENTION_KINDS:
            raise HardwarePhaseError("unknown intervention kind", [r["kind"]])
        if r["end_ns"] < r["start_ns"]:
            raise HardwarePhaseError("intervention ends before it starts")
    return {
        "interventions": len(records),
        "by_kind": {k: sum(1 for r in records if r["kind"] == k) for k in INTERVENTION_KINDS},
    }


SCORER_FIELDS = (
    "success_rule",
    "failure_classes",
    "wrong_object_rule",
    "observation_window",
    "thresholds",
    "missing_data_behavior",
    "output_schema",
    "sensor_inputs",
    "fidelity_audit_source",
)


def validate_scorer_contract(contract: dict) -> dict:
    """HWP-13: a pre-registered physical scorer over recorded sensors with an
    independent fidelity audit source; simulator state is refused."""
    missing = _missing(contract, SCORER_FIELDS)
    if missing:
        raise HardwarePhaseError("scorer contract incomplete", missing)
    if any(
        "oracle_state" in str(s) or "simulator" in str(s).lower() for s in contract["sensor_inputs"]
    ):
        raise HardwarePhaseError("physical scorer must not consume simulator state")
    if "wrong_object" not in contract["failure_classes"]:
        raise HardwarePhaseError("scorer must define the wrong-object class")
    return {"contract_hash": content_hash(contract), "fidelity": HARDWARE_PENDING}


PROTOCOL_FIELDS = (
    "typed_artifact_hash",
    "monolithic_artifact_hash",
    "tasks",
    "instance_bank",
    "perception",
    "station_hash",
    "calibration_hashes",
    "scorer_hash",
    "safety_case_hash",
    "block_order",
    "trial_unit",
    "primary_endpoints",
    "secondary_endpoints",
    "precision_target",
    "sample_size",
    "stopping_rule",
    "exclusion_rules",
    "rerun_rule",
    "reset_protocol_hash",
    "analysis_seed",
)


def validate_physical_protocol(protocol: dict) -> dict:
    """HWP-14: everything frozen before the first trial; oracle perception
    refused; the trial is the unit."""
    missing = _missing(protocol, PROTOCOL_FIELDS)
    if missing:
        raise HardwarePhaseError("physical protocol incomplete", missing)
    if re.search(r"(?<!non-)oracle", str(protocol["perception"]).lower()):
        raise HardwarePhaseError("physical protocol must use non-oracle perception")
    if protocol["trial_unit"] not in ("trial", "physical_robot_session"):
        raise HardwarePhaseError("trial unit must be trial or physical_robot_session")
    return {
        "protocol_hash": content_hash(protocol),
        "frozen": False,
        "reason": "no station exists; freeze after HWP-1..HWP-13 pass",
    }


# ------------------------------------------------------------- HWP-20 / HWP-18


def execution_checklist(statuses: dict[int, str] | None = None, embodiment: str = "so101") -> dict:
    """HWP-20: the fourteen steps, each pending until measured on the pinned
    station; a later step cannot pass while an earlier one has not. G1-G:
    the blocker names the embodiment's station; the so101 wording is pinned
    verbatim by test_hardware_phase.py."""
    statuses = statuses or {}
    steps = []
    for index, text in enumerate(CHECKLIST_STEPS, start=1):
        status = statuses.get(index, "pending")
        if status not in STEP_STATUSES:
            raise HardwarePhaseError("unknown checklist status", [str(index), status])
        steps.append({"step": index, "text": text, "status": status})
    for i, step in enumerate(steps):
        if step["status"] == "passed" and any(s["status"] != "passed" for s in steps[:i]):
            raise HardwarePhaseError(
                "a later step cannot pass before every prior step", [str(step["step"])]
            )
    # G1-G: the blocker names the embodiment's station; the so101 wording
    # is pinned VERBATIM by tests/unit/test_hardware_phase.py:280 — a dict,
    # not .upper(), because "so101".upper() is "SO101" without the hyphen.
    blockers = {
        "so101": "no SO-101 station is available",
        "g1_right": "no G1 station is available",
    }
    blocker = blockers.get(embodiment) or (
        f"no station is available for unknown embodiment {embodiment!r}"
    )
    return {
        "blocker": blocker,
        "steps": steps,
        "all_passed": all(s["status"] == "passed" for s in steps),
    }


def hardware_report(
    *,
    station: dict | None,
    motor_calibration: dict | None,
    workspace_calibration: dict | None,
    telemetry: list[dict] | None,
    interventions: list[dict] | None,
    checklist_statuses: dict[int, str] | None = None,
    embodiment: str = "so101",
) -> dict:
    """HWP-18: regenerate every table; realized-hardware fields stay
    hardware_pending until a calibrated physical instrument produced them;
    simulated and physical rows never merge."""
    checklist = execution_checklist(checklist_statuses, embodiment)
    physical_rows = [r for r in (telemetry or []) if r.get("evidence_kind") == "hardware"]
    simulated_rows = [r for r in (telemetry or []) if r.get("evidence_kind") != "hardware"]
    report = {
        "schema_version": "aisle.hardware-phase.report.v1",
        "station": HARDWARE_PENDING if station is None else validate_station_manifest(station),
        "calibration_validity": {
            "motor": HARDWARE_PENDING if motor_calibration is None else "present",
            "workspace": HARDWARE_PENDING if workspace_calibration is None else "present",
        },
        "scorer_fidelity": HARDWARE_PENDING,
        "trial_flow": HARDWARE_PENDING
        if not physical_rows
        else {"physical_rows": len(physical_rows)},
        "outcomes": HARDWARE_PENDING,
        "interventions": HARDWARE_PENDING
        if interventions is None
        else validate_intervention_records(
            interventions,
            {
                "allowed_reset_steps": [],
                "max_attempts": 0,
                "timeout_s": 0,
                "roles": [],
                "permitted_tools": [],
            },
        ),
        "safety_exposure": HARDWARE_PENDING,
        "telemetry_reconciliation": {
            "physical": HARDWARE_PENDING
            if not physical_rows
            else validate_telemetry_stream(physical_rows),
            "simulated_or_loopback": validate_telemetry_stream(simulated_rows)
            if simulated_rows
            else None,
            "merged": False,
        },
        "sim_to_hardware_deltas": HARDWARE_PENDING,
        "execution_checklist": checklist,
    }
    report["report_hash"] = content_hash({k: v for k, v in report.items() if k != "report_hash"})
    return report
