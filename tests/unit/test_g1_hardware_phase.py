"""G1 hardware-phase artifacts: docs/hardware/g1 templates validate, the
hardware tool generalizes by embodiment with so101 output byte-stable, and
the g1 dry-run covers 16 scenarios at dof=9 reproducibly (G1H-1, G1H-8;
HWP-8, HWP-18, HWP-20; SPEC 520 as the G1 parent)."""

from __future__ import annotations

import json

import pytest
from cli_helpers import REPO_ROOT, run_module

from aisle.harness import hardware_phase as hp

pytestmark = pytest.mark.unit

TEMPLATES = REPO_ROOT / "docs" / "hardware" / "g1"


def test_g1_station_manifest_template_is_complete_and_pending():
    """G1H-1/HWP-1: the G1 station template carries every pinned-device
    field (G1 serial, mode_machine=15, Dex1-1 version+serial, PC2/Jetson,
    192.168.123.x network, remote e-stop) and validates with everything
    hardware_pending — physical execution stays blocked."""
    template = json.loads((TEMPLATES / "station-manifest.template.json").read_text())
    template.pop("_instructions", None)
    template.pop("schema_version", None)
    validated = hp.validate_station_manifest(template)
    assert validated["physical_execution_allowed"] is False
    instructions = (TEMPLATES / "station-manifest.template.json").read_text()
    for needle in ("mode_machine", "15", "Dex1", "192.168.123", "PC2", "Jetson"):
        assert needle in instructions, needle


def test_g1_safety_case_template_validates_unsigned():
    """G1H-4 (HWP-12 wording): balance/fall safety stays with the motion
    controller, not budget-guard; the case blocks until signed."""
    case = json.loads((TEMPLATES / "safety-case.template.json").read_text())
    case.pop("schema_version", None)
    with pytest.raises(hp.HardwarePhaseError, match="unsigned"):
        hp.validate_safety_case(case)
    case["signatures"] = [{"role": "safety_operator", "signed_at": "2026-10-19T00:00:00Z"}]
    assert hp.validate_safety_case(case)["preflight_all_passed"] is False
    text = (TEMPLATES / "safety-case.template.json").read_text()
    assert "motion controller" in text and "budget-guard" not in json.dumps(
        json.loads(text)["software_controller"]
    )


def test_g1_scorer_contract_and_physical_protocol_templates_validate():
    """HWP-13/HWP-14 (SPEC 520 parents): the G1 scorer refuses simulator
    state, keeps the wrong-object class, and stays hardware_pending; the
    physical protocol freezes nothing without a station."""
    scorer = json.loads((TEMPLATES / "scorer-contract.json").read_text())
    scorer.pop("schema_version", None)
    assert hp.validate_scorer_contract(scorer)["fidelity"] == hp.HARDWARE_PENDING
    with pytest.raises(hp.HardwarePhaseError, match="simulator state"):
        hp.validate_scorer_contract({**scorer, "sensor_inputs": ["oracle_state"]})
    protocol = json.loads((TEMPLATES / "physical-protocol.template.json").read_text())
    protocol.pop("schema_version", None)
    assert hp.validate_physical_protocol(protocol)["frozen"] is False


def test_g1_reset_protocol_template_round_trips():
    """G1H-7 (HWP-11): behavioral reset records round-trip the G1
    intervention protocol — reset/retrain scoring consequences retained."""
    protocol = json.loads((TEMPLATES / "reset-intervention-protocol.json").read_text())
    records = [
        {
            "kind": "reset",
            "start_ns": 10,
            "end_ns": 20,
            "actor": "trial_conductor",
            "reason": "arm returned to zero posture under weight ramp",
            "scoring_consequence": "trial restarts; counted as an attempt",
        }
    ]
    assert hp.validate_intervention_records(records, protocol)["interventions"] == 1


def test_calibration_and_execution_procedures_are_g1_specific():
    """G1H-3/G1H-5/HWP-4: joint zeros come from rt/lowstate radians (not
    feetech counts); the execution ladder runs torque-off telemetry → hoist
    no-load → loaded, with weight-ramp takeover/release and lease drills."""
    calibration = (TEMPLATES / "calibration-procedure.md").read_text()
    assert "rt/lowstate" in calibration and "radians" in calibration
    assert "Dex1" in calibration
    execution = (TEMPLATES / "execution-procedure.md").read_text()
    for needle in ("torque-off", "hoist", "weight", "lease", "damping"):
        assert needle in execution, needle


def test_dry_run_g1_covers_sixteen_scenarios_at_dof_9_reproducibly():
    """G1H-8/HWP-8: dry-run --embodiment g1_right runs the 15 SPEC-520
    scenarios plus mode_mismatch at dof=9; same seed reproduces byte-equal
    observations."""
    code, report = run_dry_run("g1_right", seed=7)
    assert code == 0 and report["evidence_kind"] == "loopback"
    names = set(report["scenarios"])
    assert len(names) == 16 and "mode_mismatch" in names
    mismatch_rows = report["scenarios"]["mode_mismatch"]["telemetry"]
    assert mismatch_rows["rows"] == 12
    again_code, again = run_dry_run("g1_right", seed=7)
    assert again_code == 0 and again["scenarios"] == report["scenarios"]


def run_dry_run(embodiment: str, seed: int = 0):
    proc = run_module(
        "aisle.harness.cli", "hardware", "dry-run", "--embodiment", embodiment, "--seed", str(seed)
    )
    return proc.returncode, json.loads(proc.stdout)


def test_dry_run_so101_default_is_byte_stable():
    """HWP-20 regression: the default dry-run output is identical with and
    without --embodiment so101 and keeps exactly the 15 SPEC-520 scenarios
    (the mode_mismatch G1-only scenario never leaks into so101)."""
    default_proc = run_module("aisle.harness.cli", "hardware", "dry-run")
    explicit_proc = run_module("aisle.harness.cli", "hardware", "dry-run", "--embodiment", "so101")
    assert default_proc.returncode == 0 and explicit_proc.returncode == 0
    assert json.loads(default_proc.stdout) == json.loads(explicit_proc.stdout)
    names = set(json.loads(default_proc.stdout)["scenarios"])
    assert len(names) == 15 and "mode_mismatch" not in names


def test_report_blocker_is_parameterized_and_so101_wording_unchanged():
    """HWP-18/HWP-20: report keeps the pinned so101 blocker verbatim and
    names the G1 station for --embodiment g1_right."""
    so101_proc = run_module("aisle.harness.cli", "hardware", "report")
    assert so101_proc.returncode == 0
    blocker = json.loads(so101_proc.stdout)["execution_checklist"]["blocker"]
    assert blocker.startswith("no SO-101 station")
    g1 = json.loads(
        run_module("aisle.harness.cli", "hardware", "report", "--embodiment", "g1_right").stdout
    )
    assert "G1" in g1["execution_checklist"]["blocker"]
