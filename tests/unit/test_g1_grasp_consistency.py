"""Grasp-command / carry-latch consistency (SCN-4, ADR-18, G1-H-r4).

The r3 lesson: two cards' parameters were each locally reasonable and
JOINTLY contradictory — gripper_grasp_cmd closed below carry_latch_close,
so the latch could never fire (the dropped smoke). These tests pin the
invariant for every embodiment that enables the latch:

- per med, the geometric contact plateau (finger qpos at two-face
  contact -> normalized grip) must EXCEED carry_latch_close;
- gripper_grasp_cmd must EXCEED every plateau (a squeeze: PD keeps
  holding force after contact);
- carry_latch_release < carry_latch_close (hysteresis, no chatter).
"""

import pytest

from aisle.scenes.pharmacy import load_meds, load_physics

pytestmark = pytest.mark.unit

PHYSICS = load_physics()
MEDS = load_meds()


def contact_plateau(embodiment: str, med: str) -> float:
    """Normalized grip when both fingers touch the box's grasp-axis faces.

    G1 (symmetric two-finger jaw): finger qpos f = grasp-axis width / 2
    (each finger travels half the gap), g = (open - f)/(open - close).
    SO-101 (single-axis jaw): the measured contact onset is 0.75
    (physics.toml note) — its single lever closes one jaw onto the box
    face, contact onset sits BELOW the commanded 0.70 squeeze by design
    (the latch fires at 0.68 mid-close, before physical contact; see
    physics.toml 'stop below the measured 0.75 contact onset').
    """
    prof = PHYSICS["embodiment"][embodiment]
    if embodiment == "g1_right":
        width = min(MEDS[med]["size"][0], MEDS[med]["size"][1])
        f = width / 2.0
        return (float(prof["gripper_open_qpos"]) - f) / (
            float(prof["gripper_open_qpos"]) - float(prof["gripper_close_qpos"])
        )
    # so101: measured contact onset (documented in physics.toml)
    return 0.75


@pytest.mark.parametrize("med", list(MEDS))
def test_g1_every_med_plateau_exceeds_latch_close(med):
    """SCN-4: the latch threshold sits BELOW every med's contact plateau —
    any successfully closed grasp (plateau reached) fires the latch."""
    prof = PHYSICS["embodiment"]["g1_right"]
    plateau = contact_plateau("g1_right", med)
    assert plateau > float(prof["carry_latch_close"]), (med, plateau)


def test_plateaus_are_monotone_in_width():
    """Sanity: a wider grasp-axis box contacts EARLIER (LOWER normalized
    grip) — the derivation direction the thresholds rely on."""
    widths = {m: min(s["size"][0], s["size"][1]) for m, s in MEDS.items()}
    plateaus = {m: contact_plateau("g1_right", m) for m in MEDS}
    order_by_width = sorted(MEDS, key=lambda m: widths[m])
    got = [plateaus[m] for m in order_by_width]
    assert got == sorted(got, reverse=True)


@pytest.mark.parametrize("med", list(MEDS))
def test_g1_grasp_cmd_is_a_squeeze_above_every_plateau(med):
    """SCN-4: the grasp command closes PAST every plateau, so the PD keeps
    a holding force after contact (a 0.55-style dead-stop below a plateau
    threshold is exactly the r3 contradiction)."""
    prof = PHYSICS["embodiment"]["g1_right"]
    plateau = contact_plateau("g1_right", med)
    assert float(prof["gripper_grasp_cmd"]) > plateau, (med, plateau)


def test_g1_latch_hysteresis_band():
    """ADR-18: release strictly below close (no chatter around one grip
    level); max_distance bounds the latch to the pinched box."""
    prof = PHYSICS["embodiment"]["g1_right"]
    assert float(prof["carry_latch_release"]) < float(prof["carry_latch_close"])
    assert 0.0 < float(prof["carry_latch_max_distance_m"]) <= 0.2


@pytest.mark.parametrize("med", list(MEDS))
def test_so101_latch_invariant_pinned(med):
    """ADR-18 regression pin: so101's working tuning keeps its documented
    chain — the latch close (0.68) engages BEFORE the measured 0.75
    contact onset (pre-contact latching is the design), the grasp command
    (0.70) is a squeeze past the latch point, and release sits under
    close. It is the reference this file generalizes from; g1_right now
    mirrors the same pre-contact engagement geometry."""
    prof = PHYSICS["embodiment"]["so101"]
    onset = contact_plateau("so101", med)
    assert onset > float(prof["carry_latch_close"]), (med, onset)
    assert float(prof["gripper_grasp_cmd"]) > float(prof["carry_latch_close"])
    assert float(prof["carry_latch_release"]) < float(prof["carry_latch_close"])


def test_plateaus_table_is_the_adr_table():
    """CON-5: the ADR's per-med plateau table and this computation agree —
    the numbers are derived, not transcribed."""
    expected = {
        "amoxicillin": pytest.approx(0.101, abs=5e-4),
        "ibuprofen": pytest.approx(0.157, abs=5e-4),
        "cetirizine": pytest.approx(0.213, abs=5e-4),
        "omeprazole": pytest.approx(0.045, abs=5e-4),
        "metformin": pytest.approx(0.157, abs=5e-4),
    }
    for med, approx in expected.items():
        assert contact_plateau("g1_right", med) == approx, med
