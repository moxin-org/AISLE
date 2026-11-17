//! Independent judge thresholds and oracle decoding, one boundary per test.
use aisle_nano::{
    judge::{Judge, Sample, Verdict},
    oracle::{self, Feed, Input, Step},
};

const TICK: u64 = 200_000_000;
const DROP: [f64; 3] = [0.3, 0., 0.2];

fn judge() -> Judge {
    Judge::new([0., 0.], 0.2, DROP)
}
/// Robot at the goal, cube resting untouched away from the placement zone.
fn rest(i: u64) -> Sample {
    Sample {
        stamp: i * TICK,
        base: [0., 0.],
        yaw: 0.,
        cube: [0.6, 0., 0.2],
        finger_contact: false,
        robot_contact: false,
        furniture_contact: false,
    }
}
fn held(i: u64, cube: [f64; 3]) -> Sample {
    Sample {
        cube,
        finger_contact: true,
        robot_contact: true,
        ..rest(i)
    }
}
fn released(i: u64, cube: [f64; 3]) -> Sample {
    Sample { cube, ..rest(i) }
}
fn feed(j: &mut Judge, samples: impl IntoIterator<Item = Sample>) -> Verdict {
    for s in samples {
        j.observe(s).unwrap();
    }
    j.result()
}
/// Samples 0..=3: a supported lift long enough to count as pickup.
fn lifted() -> Judge {
    let mut j = judge();
    let v = feed(&mut j, (0..4).map(|i| held(i, [0.6, 0., 0.32])));
    assert!(v.pickup);
    j
}

/// TC-8: a lift counts only above 7 cm, with finger contact, held for 0.4 s.
#[test]
fn pickup_needs_height_contact_and_duration() {
    let high = [0.6, 0., 0.32];
    let mut j = judge();
    assert!(!feed(&mut j, (0..2).map(|i| held(i, high))).pickup);
    assert!(feed(&mut j, [held(2, high)]).pickup);
    let mut j = judge();
    assert!(!feed(&mut j, (0..8).map(|i| held(i, [0.6, 0., 0.25]))).pickup);
    let mut j = judge();
    assert!(!feed(&mut j, (0..8).map(|i| released(i, high))).pickup);
    assert!((j.result().max_lift_m - 0.12).abs() < 1e-12);
}

/// TC-8: losing finger contact restarts the lift window.
#[test]
fn interrupted_contact_restarts_the_lift_window() {
    let high = [0.6, 0., 0.32];
    let mut j = judge();
    let v = feed(&mut j, [held(0, high), released(1, high), held(2, high)]);
    assert!(!v.pickup);
    assert!(!feed(&mut j, [held(3, high)]).pickup);
    assert!(feed(&mut j, [held(4, high)]).pickup);
}

/// TC-8: placement needs a prior pickup, then 1 s released and motionless.
#[test]
fn placement_needs_pickup_then_a_stable_release() {
    let mut j = lifted();
    feed(&mut j, [held(4, DROP)]);
    assert!(!feed(&mut j, (5..10).map(|i| released(i, DROP))).placed);
    assert!(feed(&mut j, [released(10, DROP)]).placed);

    let mut never_lifted = judge();
    feed(&mut never_lifted, [rest(0), held(1, DROP)]);
    assert!(!feed(&mut never_lifted, (2..12).map(|i| released(i, DROP))).placed);

    let mut rocking = lifted();
    feed(&mut rocking, [held(4, DROP)]);
    let wobble = |i: u64| {
        [
            DROP[0] + if i.is_multiple_of(2) { 0.01 } else { -0.01 },
            0.,
            DROP[2],
        ]
    };
    assert!(!feed(&mut rocking, (5..20).map(|i| released(i, wobble(i)))).placed);
}

/// TC-8: the placement zone is 5 cm in the plane and 2.5 cm in height.
#[test]
fn placement_zone_is_tight() {
    for miss in [
        [DROP[0] + 0.1, 0., DROP[2]],
        [DROP[0], 0.1, DROP[2]],
        [DROP[0], 0., DROP[2] + 0.05],
    ] {
        let mut j = lifted();
        feed(&mut j, [held(4, miss)]);
        assert!(
            !feed(&mut j, (5..20).map(|i| released(i, miss))).placed,
            "{miss:?}"
        );
    }
}

/// TC-8: a cube that leaves the zone and returns unsupported was not placed.
#[test]
fn support_does_not_survive_leaving_the_zone() {
    let mut j = lifted();
    feed(&mut j, [held(4, DROP), released(5, [0.6, 0., 0.2])]);
    assert!(!feed(&mut j, (6..20).map(|i| released(i, DROP))).placed);
}

/// TC-8: every furniture contact sample is counted.
#[test]
fn furniture_contacts_are_counted() {
    let mut j = judge();
    let bump = |i| Sample {
        furniture_contact: true,
        ..rest(i)
    };
    assert_eq!(
        feed(&mut j, [rest(0), bump(1), bump(2), rest(3)]).furniture_contacts,
        2
    );
    assert_eq!(j.result().samples, 4);
}

/// TC-8, MOB-2: arrival is within 12 cm of the goal and stationary for 0.4 s.
#[test]
fn arrival_needs_radius_stop_and_duration() {
    let mut j = judge();
    assert!(!feed(&mut j, (0..3).map(rest)).navigation);
    assert!(feed(&mut j, [rest(3)]).navigation);

    let mut far = judge();
    let away = |i| Sample {
        base: [0.5, 0.],
        ..rest(i)
    };
    assert!(!feed(&mut far, (0..10).map(away)).navigation);

    let mut creeping = judge();
    let creep = |i: u64| Sample {
        base: [-0.05 + 0.01 * i as f64, 0.],
        ..rest(i)
    };
    assert!(!feed(&mut creeping, (0..10).map(creep)).navigation);
}

/// TC-8, MOB-2: a robot turning on the spot at the goal has not stopped.
#[test]
fn spinning_at_the_goal_is_not_arrival() {
    let mut j = judge();
    let spin = |i: u64| Sample {
        yaw: (0.3 * i as f64 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI,
        ..rest(i)
    };
    assert!(!feed(&mut j, (0..40).map(spin)).navigation);
    let still = |i| Sample {
        yaw: spin(39).yaw,
        ..rest(i)
    };
    assert!(feed(&mut j, (40..43).map(still)).navigation);

    // Crossing the +/- pi seam is a small turn, not a full revolution.
    let mut seam = judge();
    let wrap = |i: u64| Sample {
        yaw: (std::f64::consts::PI - 0.002) * if i.is_multiple_of(2) { 1. } else { -1. },
        ..rest(i)
    };
    assert!(feed(&mut seam, (0..4).map(wrap)).navigation);
}

/// TC-8, CON-4: non-finite samples and repeated clocks are refused.
#[test]
fn invalid_samples_are_refused() {
    for bad in [
        Sample {
            base: [f64::NAN, 0.],
            ..rest(0)
        },
        Sample {
            cube: [0.6, f64::INFINITY, 0.2],
            ..rest(0)
        },
        Sample {
            yaw: f64::NAN,
            ..rest(0)
        },
    ] {
        assert!(judge().observe(bad).is_err());
    }
    let mut j = judge();
    j.observe(rest(1)).unwrap();
    assert!(j.observe(rest(1)).is_err());
    assert!(j.observe(rest(0)).is_err());
}

/// TC-8: an older trace without a heading still parses, as a non-turning robot.
#[test]
fn trace_without_heading_still_parses() {
    let old = r#"{"stamp":0,"base":[0.0,0.0],"cube":[0.6,0.0,0.2],"finger_contact":false,"robot_contact":false,"furniture_contact":false}"#;
    assert_eq!(serde_json::from_str::<Sample>(old).unwrap(), rest(0));
}

/// TC-8: oracle columns map to the right contact flags; malformed rows are refused.
#[test]
fn oracle_row_decoding() {
    let s = oracle::sample(7, &[1., 2., 3., 4., 5., 1., 0., 0.], 0.5).unwrap();
    assert_eq!(
        (s.stamp, s.base, s.yaw, s.cube),
        (7, [1., 2.], 0.5, [3., 4., 5.])
    );
    assert!(s.finger_contact && !s.furniture_contact && !s.robot_contact);
    let s = oracle::sample(7, &[1., 2., 3., 4., 5., 0., 1., 0.], 0.).unwrap();
    assert!(!s.finger_contact && s.furniture_contact && !s.robot_contact);
    let s = oracle::sample(7, &[1., 2., 3., 4., 5., 0., 0., 1.], 0.).unwrap();
    assert!(!s.finger_contact && !s.furniture_contact && s.robot_contact);
    assert!(oracle::sample(7, &[1., 2., 3., 4., 5., 0., 0.], 0.).is_err());
    assert!(oracle::sample(7, &[1., 2., 3., 4., 5., 0.5, 0., 0.], 0.).is_err());
    assert!(oracle::sample(7, &[1., f64::NAN, 3., 4., 5., 0., 0., 0.], 0.).is_err());
    assert!(oracle::sample(7, &[1., 2., 3., 4., 5., 0., 0., 0.], f64::NAN).is_err());
}

/// TC-8: heading comes from the state's XYZW base quaternion.
#[test]
fn base_heading_from_state() {
    let mut state = vec![0.; 24];
    let half = 0.4_f64;
    state[5] = half.sin();
    state[6] = half.cos();
    assert!((oracle::base_yaw(&state).unwrap() - 0.8).abs() < 1e-9);
    state[6] = 2.;
    assert!(oracle::base_yaw(&state).is_err());
    assert!(oracle::base_yaw(&state[..23]).is_err());
}

fn verdict(navigation: bool, placed: bool, furniture_contacts: u64) -> Verdict {
    Verdict {
        navigation,
        pickup: placed,
        placed,
        furniture_contacts,
        max_lift_m: 0.,
        samples: 1,
    }
}

/// TC-8: success is per task, and never without a complete, contact-free run.
#[test]
fn task_success_rules() {
    assert!(oracle::success("pickup", true, &verdict(true, true, 0)));
    assert!(!oracle::success("pickup", true, &verdict(true, false, 0)));
    assert!(!oracle::success("pickup", true, &verdict(false, true, 0)));
    assert!(!oracle::success("pickup", true, &verdict(true, true, 1)));
    assert!(!oracle::success("pickup", false, &verdict(true, true, 0)));
    assert!(oracle::success("navigate", true, &verdict(true, false, 0)));
    assert!(!oracle::success(
        "navigate",
        true,
        &verdict(false, false, 0)
    ));
    assert!(!oracle::success("navigate", true, &verdict(true, false, 2)));
    // Inspection never drives to the goal, so it cannot require arrival.
    assert!(oracle::success("inspect", true, &verdict(false, false, 0)));
    assert!(!oracle::success(
        "inspect",
        false,
        &verdict(false, false, 0)
    ));
    assert!(!oracle::success("inspect", true, &verdict(false, false, 1)));
}

fn state(yaw: f64) -> Vec<f64> {
    let mut state = vec![0.; 24];
    state[5] = (yaw / 2.).sin();
    state[6] = (yaw / 2.).cos();
    state
}
const ROW: [f64; 8] = [0., 0., 0.6, 0., 0.2, 0., 0., 0.];

/// TC-8, CON-4: each observation is judged once, from both halves, in order.
#[test]
fn feed_pairs_heading_with_truth_once_per_observation() {
    let mut f = Feed::new([0., 0.], DROP);
    assert_eq!(f.offer(0, 0, Input::Truth(&ROW)).unwrap(), Step::Pending);
    assert_eq!(f.last(), None);
    assert!(f.verdict().is_none());
    assert!(matches!(
        f.offer(0, 0, Input::State(&state(0.))).unwrap(),
        Step::Observed(s) if s.stamp == 0 && s.cube == [0.6, 0., 0.2]
    ));
    // Either arrival order completes the pair.
    assert_eq!(
        f.offer(1, TICK, Input::State(&state(0.))).unwrap(),
        Step::Pending
    );
    assert!(matches!(
        f.offer(1, TICK, Input::Truth(&ROW)).unwrap(),
        Step::Observed(_)
    ));
    // An identical retransmission is acknowledged but not judged twice.
    f.offer(1, TICK, Input::Truth(&ROW)).unwrap();
    assert_eq!(
        f.offer(1, TICK, Input::State(&state(0.))).unwrap(),
        Step::Repeated
    );
    assert_eq!(f.last(), Some(1));
    assert_eq!(f.verdict().unwrap().samples, 2);
    // A heading from an older observation never pairs with newer truth.
    f.offer(1, TICK, Input::State(&state(0.))).unwrap();
    assert_eq!(
        f.offer(2, 2 * TICK, Input::Truth(&ROW)).unwrap(),
        Step::Pending
    );
}

/// TC-8, CON-4: changed retries, gaps and mismatched clocks fail closed.
#[test]
fn feed_rejects_changed_retries_gaps_and_clock_mismatch() {
    let observed = || {
        let mut f = Feed::new([0., 0.], DROP);
        f.offer(0, 0, Input::Truth(&ROW)).unwrap();
        f.offer(0, 0, Input::State(&state(0.))).unwrap();
        f
    };
    let mut f = observed();
    let mut changed = ROW;
    changed[2] = 0.7;
    f.offer(0, 0, Input::Truth(&changed)).unwrap();
    assert!(f.offer(0, 0, Input::State(&state(0.))).is_err());

    let mut f = observed();
    f.offer(0, 0, Input::Truth(&ROW)).unwrap();
    assert!(f.offer(0, 0, Input::State(&state(0.5))).is_err());

    // A late duplicate of an older observation is ignored, not judged or fatal.
    let mut f = observed();
    f.offer(1, TICK, Input::Truth(&ROW)).unwrap();
    f.offer(1, TICK, Input::State(&state(0.))).unwrap();
    f.offer(0, 0, Input::Truth(&ROW)).unwrap();
    assert_eq!(
        f.offer(0, 0, Input::State(&state(0.))).unwrap(),
        Step::Pending
    );
    assert_eq!((f.last(), f.verdict().unwrap().samples), (Some(1), 2));

    let mut f = observed();
    f.offer(2, 2 * TICK, Input::Truth(&ROW)).unwrap();
    assert!(f.offer(2, 2 * TICK, Input::State(&state(0.))).is_err());

    let mut f = observed();
    f.offer(1, TICK, Input::Truth(&ROW)).unwrap();
    assert!(f.offer(1, 2 * TICK, Input::State(&state(0.))).is_err());
}
