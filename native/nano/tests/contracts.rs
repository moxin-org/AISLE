//! Experimental Nano acceptance: requirements are cited per test.
use aisle_nano::{
    judge::{Judge, Sample},
    kinematics::Robot,
    navigation::{Box2, Map},
    perception::{Color, detect},
    safety::Guard,
};
use nalgebra::{Isometry3, Vector3};

fn robot() -> Robot {
    Robot::read(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/robots/adora_nano/model.urdf"
    )))
    .unwrap()
}

/// TC-1, TC-5, MOB-4: both physical arm chains, all 17 joints, identical mirrored reach.
#[test]
fn nano_fk_and_ik_respect_both_chains() {
    let r = robot();
    assert_eq!(r.names.len(), 17);
    let mut q = vec![0.; 17];
    for side in ["left", "right"] {
        for (j, v) in [(1, 0.2), (2, 0.4), (3, -0.6), (4, 0.3), (5, 0.1)] {
            q[r.arm(side).unwrap()[j - 1]] = v;
        }
    }
    let l = r.fk("left_tool_mount", &q).unwrap().translation.vector;
    let p = r.fk("right_tool_mount", &q).unwrap().translation.vector;
    assert!((l - Vector3::new(p.x, -p.y, p.z)).norm() < 1e-6);
    for side in ["left", "right"] {
        let link = format!("{side}_tool_mount");
        let goal = r.fk(&link, &q).unwrap().translation.vector;
        let solved = r
            .ik(side, &link, Vector3::zeros(), goal, None, &[0.; 17])
            .unwrap();
        assert!((r.fk(&link, &solved).unwrap().translation.vector - goal).norm() < 0.003);
        assert!(
            solved
                .iter()
                .zip(&r.limits)
                .all(|(v, l)| *v >= l[0] && *v <= l[1])
        );
    }
    assert!(
        r.ik(
            "left",
            "left_tool_mount",
            Vector3::zeros(),
            Vector3::new(9., 9., 9.),
            None,
            &q
        )
        .is_err()
    );
}

/// TC-1, TC-5: jaw-axis IK for a cube-aligned grasp with the five-DoF Nano arm.
#[test]
fn cube_aligned_jaw_ik() {
    let r = robot();
    for side in ["left", "right"] {
        let sign = if side == "left" { 1. } else { -1. };
        let mut q = vec![0.; 17];
        for goal in [
            [0.404, sign * 0.23, 0.30],
            [0.404, sign * 0.23, 0.20],
            [0.404, sign * 0.23, 0.30],
            [0.34, sign * 0.23, 0.30],
            [0.34, sign * 0.23, 0.215],
        ] {
            let offset = Vector3::new(0.012, 0., -0.12);
            q = r
                .ik_axis(
                    side,
                    &format!("{side}_wrist_housing"),
                    offset,
                    goal.into(),
                    Vector3::x(),
                    Some(Vector3::y() * sign),
                    &q,
                )
                .unwrap();
            let pose = r.fk(&format!("{side}_wrist_housing"), &q).unwrap();
            assert!(
                (pose.transform_point(&offset.into()).coords - Vector3::from(goal)).norm() < 0.003
            );
            assert!((pose.rotation * Vector3::x() - Vector3::y() * sign).norm() < 0.10);
            println!(
                "{side} goal {goal:?} approach {:?}",
                pose.rotation * (-Vector3::z())
            );
        }
    }
}

/// TC-1, TC-5, MOB-4: solve the actual finger-center approach/lift for both arms.
#[test]
fn both_grippers_can_reach_pickup_and_lift() {
    let r = robot();
    for side in ["left", "right"] {
        let sign = if side == "left" { 1. } else { -1. };
        let mut q = vec![0.; 17];
        for goal in [
            [0.41, sign * 0.23, 0.30],
            [0.41, sign * 0.23, 0.20],
            [0.41, sign * 0.23, 0.32],
            [0.37, sign * 0.10, 0.32],
            [0.37, sign * 0.10, 0.215],
        ] {
            let offset = Vector3::new(0.008, 0., -0.11);
            q = r
                .ik(
                    side,
                    &format!("{side}_wrist_housing"),
                    offset,
                    goal.into(),
                    Some(-Vector3::z()),
                    &q,
                )
                .unwrap();
            let pose = r.fk(&format!("{side}_wrist_housing"), &q).unwrap();
            assert!(
                (pose.transform_point(&offset.into()).coords - Vector3::from(goal)).norm() < 0.003
            );
        }
    }
}

/// TC-9: identity from RGB; same-frame depth supplies metric geometry only.
#[test]
fn rgb_depth_perception_rejects_mismatched_or_invalid_frames() {
    let (w, h) = (20, 20);
    let mut rgb = vec![0u8; w * h * 3];
    let depth = vec![1f32; w * h];
    for y in 5..15 {
        for x in 6..14 {
            rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[220, 20, 15]);
        }
    }
    let hits = detect(
        &rgb,
        &depth,
        w,
        h,
        20.,
        Color::Red,
        1,
        1,
        &Isometry3::identity(),
    )
    .unwrap();
    assert_eq!(hits.len(), 1);
    assert!((hits[0].position.z + 1.).abs() < 1e-6);
    assert!(
        detect(
            &rgb,
            &depth,
            w,
            h,
            20.,
            Color::Red,
            1,
            2,
            &Isometry3::identity()
        )
        .is_err()
    );
    assert!(
        detect(
            &rgb,
            &vec![f32::NAN; w * h],
            w,
            h,
            20.,
            Color::Red,
            1,
            1,
            &Isometry3::identity()
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        detect(
            &rgb[..10],
            &depth,
            w,
            h,
            20.,
            Color::Red,
            1,
            1,
            &Isometry3::identity()
        )
        .is_err()
    );
}

/// MOB-2: route around furniture; refuse blocked goals and corner cutting.
#[test]
fn navigation_avoids_inflated_obstacles() {
    let m = Map {
        bounds: [-3., 3., -3., 3.],
        clearance: 0.4,
        obstacles: vec![Box2 {
            center: [0., 0.],
            half: [0.5, 1.],
            clearance: None,
        }],
    };
    let route = m.plan([-2., 0.], [2., 0.]).unwrap();
    assert!(route.len() > 1);
    assert_eq!(route.last(), Some(&[2., 0.]));
    let mut p = [-2., 0.];
    for next in route {
        // Independent geometry: every sampled point stays outside the box
        // grown by the 0.4 m clearance, not merely "clear" by the map's own test.
        for i in 0..=200 {
            let t = i as f64 / 200.;
            let (x, y) = (p[0] + t * (next[0] - p[0]), p[1] + t * (next[1] - p[1]));
            assert!(
                x.abs() > 0.9 || y.abs() > 1.4,
                "({x}, {y}) cuts the obstacle"
            );
        }
        p = next;
    }
    assert!(m.plan([-2., 0.], [0., 0.]).is_err());
}

/// MOB-2: solid-wheel Nano sim turns before driving and overcomes stiction.
#[test]
fn navigation_turns_before_forward_motion() {
    let (v, w) = aisle_nano::navigation::base_command([0., 1.], [0., 0., 0.]).unwrap();
    assert_eq!(v, 0.);
    assert!(w > 0.);
    let (v, w) = aisle_nano::navigation::base_command([0.04, 0.], [0., 0., 0.]).unwrap();
    assert!(v >= 0.07);
    assert_eq!(w, 0.);
    assert!(aisle_nano::navigation::base_command([f64::NAN, 0.], [0., 0., 0.]).is_err());
}

/// BG-1, BG-2, MOB-3: limits, NaN, replayed clocks and concurrent motion fail closed.
#[test]
fn guard_limits_and_interlocks() {
    let r = robot();
    let mut g = Guard::new(r.clone());
    let q = vec![0.; 17];
    let mut cmd = vec![0.; 17];
    cmd[0] = 99.;
    cmd[5] = 0.3;
    let safe = g.apply(0, &q, &cmd).unwrap();
    assert_eq!(safe[0], 0.);
    assert_eq!(safe[5], 0.3);
    assert!(g.apply(0, &q, &cmd).is_err());
    cmd[1] = f64::NAN;
    assert!(g.apply(200_000_000, &q, &cmd).is_err());
    cmd.fill(0.);
    cmd[5] = 0.5;
    let mut at_limit = q.clone();
    at_limit[5] = r.limits[5][1];
    assert_eq!(g.apply(400_000_000, &at_limit, &cmd).unwrap()[5], 0.);
}

/// BG-1, MOB-4: a restarted IK solve cannot command a distant arm branch.
#[test]
fn rejects_discontinuous_ik_branch() {
    let q = [0.; 17];
    let mut next = q;
    next[5] = 2.0;
    assert!(aisle_nano::safety::validate_arm_transition(&q, &next, &[5, 6, 7, 8, 9]).is_err());
    next[5] = 0.5;
    assert!(aisle_nano::safety::validate_arm_transition(&q, &next, &[5, 6, 7, 8, 9]).is_ok());
}

/// TC-8: command completion, teleporting, or a bump alone cannot count as pickup.
#[test]
fn judge_requires_contact_lift_release_and_settling() {
    let mut j = Judge::new([1., 1.], 0.02, [1.2, 1., 0.02]);
    let base = Sample {
        stamp: 0,
        base: [1., 1.],
        yaw: 0.,
        cube: [1., 1., 0.02],
        finger_contact: false,
        robot_contact: false,
        furniture_contact: false,
    };
    j.observe(base).unwrap();
    assert!(!j.result().pickup);
    j.observe(Sample {
        stamp: 200_000_000,
        cube: [1.2, 1., 0.15],
        ..base
    })
    .unwrap();
    assert!(!j.result().pickup);
    let mut j = Judge::new([1., 1.], 0.02, [1.2, 1., 0.02]);
    j.observe(base).unwrap();
    for i in 1..10 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            cube: [1., 1., 0.14],
            finger_contact: true,
            ..base
        })
        .unwrap();
    }
    assert!(j.result().pickup);
    assert!(!j.result().placed);
    for i in 10..21 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            cube: [1.2, 1., 0.02],
            finger_contact: i == 10,
            robot_contact: i == 10,
            ..base
        })
        .unwrap();
    }
    assert!(j.result().placed);
    assert!(j.result().navigation);
    j.observe(Sample {
        stamp: 4_200_000_000,
        robot_contact: true,
        cube: [1.2, 1., 0.02],
        ..base
    })
    .unwrap();
    assert!(
        !j.result().placed,
        "One finger still touching is not released"
    );
    for i in 22..29 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            cube: [1.2, 1., 0.02],
            ..base
        })
        .unwrap();
    }
    assert!(j.result().placed);
    j.observe(Sample {
        stamp: 5_800_000_000,
        base: [2., 2.],
        yaw: 0.,
        cube: [2., 2., 0.02],
        ..base
    })
    .unwrap();
    assert!(
        !j.result().placed,
        "Placement must remain valid at termination"
    );
    assert!(
        !j.result().navigation,
        "Passing through a goal is not final arrival"
    );
    assert!(j.observe(Sample { stamp: 0, ..base }).is_err());
}

/// TC-8: dropping from height into the placement zone is not controlled placement.
#[test]
fn judge_rejects_a_dropped_cube_even_if_it_settles_in_zone() {
    let mut j = Judge::new([0., 0.], 0.2, [0.34, -0.23, 0.2]);
    for i in 0..4 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            base: [0., 0.],
            yaw: 0.,
            cube: [0.404, -0.23, 0.30],
            finger_contact: true,
            robot_contact: true,
            furniture_contact: false,
        })
        .unwrap();
    }
    assert!(j.result().pickup);
    for i in 4..15 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            base: [0., 0.],
            yaw: 0.,
            cube: [0.34, -0.23, 0.2],
            finger_contact: false,
            robot_contact: false,
            furniture_contact: false,
        })
        .unwrap();
    }
    assert!(!j.result().placed);
    // A later two-finger touch cannot retroactively turn an unsupported drop
    // into controlled placement. Support must exist on entry to the zone.
    j.observe(Sample {
        stamp: 3_000_000_000,
        base: [0., 0.],
        yaw: 0.,
        cube: [0.34, -0.23, 0.2],
        finger_contact: true,
        robot_contact: true,
        furniture_contact: false,
    })
    .unwrap();
    for i in 16..25 {
        j.observe(Sample {
            stamp: i * 200_000_000,
            base: [0., 0.],
            yaw: 0.,
            cube: [0.34, -0.23, 0.2],
            finger_contact: false,
            robot_contact: false,
            furniture_contact: false,
        })
        .unwrap();
    }
    assert!(
        !j.result().placed,
        "A post-drop touch is not supported entry"
    );
}

/// TC-8, CON-4: missing oracle intervals cannot establish a sustained grasp.
#[test]
fn judge_rejects_missing_physics_observations() {
    let mut judge = Judge::new([0., 0.], 0.2, [0.3, 0., 0.2]);
    let sample = Sample {
        stamp: 0,
        base: [0., 0.],
        yaw: 0.,
        cube: [0.4, 0., 0.3],
        finger_contact: true,
        robot_contact: true,
        furniture_contact: false,
    };
    judge.observe(sample).unwrap();
    assert!(
        judge
            .observe(Sample {
                stamp: 600_000_000,
                ..sample
            })
            .is_err()
    );
    assert!(!judge.result().pickup);
}

/// MOB-2, CON-8: planning never panics; any returned route is collision-free.
#[test]
fn planner_returns_a_clear_route_or_an_error_near_corners() {
    let corner = Map {
        bounds: [-3., 3., -3., 3.],
        clearance: 0.,
        obstacles: vec![Box2 {
            center: [0.03, -0.03],
            half: [0.1, 0.1],
            clearance: None,
        }],
    };
    let obstacle = |center, half, clearance| Box2 {
        center,
        half,
        clearance,
    };
    let room = Map {
        bounds: [-5.9, 5.9, -4.9, 4.9],
        clearance: 0.55,
        obstacles: vec![
            obstacle([0., 2.6], [1.3, 0.45], None),
            obstacle([0., 1.1], [0.7, 0.4], None),
            obstacle([-3.4, 0.3], [0.6, 1.0], None),
            obstacle([0.85, -2.5], [0.09, 0.6], None),
            obstacle([2.08, -1.43], [0.18, 0.35], Some(0.27)),
        ],
    };
    // Each of these panicked in route smoothing before the start cell was retained.
    let mut cases = vec![(corner, [-0.11, -0.07], [-0.04, -0.17])];
    for (start, goal) in [
        (
            [-1.2664256118774841, 0.2452451701782845],
            [2.177856212725694, 0.14945378842969692],
        ),
        (
            [2.461420635473579, -2.054402715722067],
            [2.1545616975855157, 3.5137309849469283],
        ),
        (
            [-1.277059805746875, 0.23348701851403408],
            [2.4476125689246997, 0.4357812629343689],
        ),
        (
            [1.2539622466345506, 0.167902621306669],
            [-3.458235970170195, 3.41423343687417],
        ),
        (
            [-4.572540214275682, 1.8037297981719131],
            [4.762161451944525, 0.4559079204392935],
        ),
    ] {
        cases.push((room.clone(), start, goal));
    }
    for (map, start, goal) in cases {
        let planned = std::panic::catch_unwind(|| map.plan(start, goal));
        let planned = planned.unwrap_or_else(|_| panic!("planner panicked: {start:?} -> {goal:?}"));
        if let Ok(route) = planned {
            assert_eq!(route.last(), Some(&goal));
            let mut p = start;
            for next in route {
                assert!(map.clear_segment(p, next), "{start:?} -> {goal:?}");
                p = next;
            }
        }
    }
}

/// MOB-2, CON-8: the supervisor pre-flight plans the route the policy will drive.
#[test]
fn preflight_rejects_a_blocked_pickup_docking_point() {
    let goal = [1.9, -2.5];
    let map = aisle_nano::runtime::map(&aisle_nano::runtime::room(0, goal, "right"));
    assert!(map.plan([0., 0.], goal).is_ok());
    assert!(aisle_nano::runtime::route(&map, "pickup", [0., 0.], goal).is_err());
    assert!(aisle_nano::runtime::route(&map, "navigate", [0., 0.], goal).is_ok());
    let open = [1.6, -1.2];
    let map = aisle_nano::runtime::map(&aisle_nano::runtime::room(0, open, "right"));
    let route = aisle_nano::runtime::route(&map, "pickup", [0., 0.], open).unwrap();
    assert_eq!(route.last(), Some(&open));
    assert!(route.contains(&[open[0] - 0.7, open[1]]));
}

/// CON-8: a policy fault is reported as a fault, never as a silent physical failure.
#[test]
fn policy_fault_is_distinguished_from_normal_completion() {
    use aisle_nano::runtime::policy_fault;
    let fault = serde_json::json!({"done":true,"fault":true,"reason":"IK unreachable"});
    assert_eq!(policy_fault(&fault).as_deref(), Some("IK unreachable"));
    let normal = serde_json::json!({"done":true,"fault":false,"reason":"trajectory finished"});
    assert_eq!(policy_fault(&normal), None);
    assert_eq!(policy_fault(&serde_json::json!({})), None);
}

/// MOB-2/3: traversing the room reserves the full base and arm envelope beside other workstations.
#[test]
fn household_docking_clearance_is_local_to_the_final_lane() {
    use aisle_nano::navigation::{Box2, Map};
    let dock = Map {
        bounds: [-3., 3., -3., 3.],
        clearance: 0.55,
        obstacles: vec![Box2 {
            center: [0.5, 0.],
            half: [0.10, 0.08],
            clearance: Some(0.225),
        }],
    };
    let beside = [0.5, -0.36];
    assert!(dock.clear(beside));
    let transit = dock.transit();
    assert!(!transit.clear(beside));
    let start = [-0.9, 0.];
    let dest = [1.6, -0.3];
    let mut from = start;
    for point in transit.plan(start, dest).unwrap() {
        assert!(transit.clear_segment(from, point));
        from = point;
    }
}

/// MOB-2. Regression from the bedroom turnaround. The disc clears the sofa corner by 2 mm.
#[test]
fn leave_a_square_grid_corner_only_with_full_disc_clearance() {
    use aisle_nano::navigation::{Box2, Map};
    let map = Map {
        bounds: [-8., 6., -9., 3.],
        clearance: 0.55,
        obstacles: vec![Box2 {
            center: [-2.13, -0.80],
            half: [1.30, 0.11],
            clearance: None,
        }],
    };
    let start = [-0.639, -0.172];
    assert!(!map.clear(start));
    let route = map.transit_route(start, [1., 1.]).unwrap();
    assert!(map.clear(route[0]));
    assert!(map.transit_route([-0.66, -0.20], [1., 1.]).is_err());
}
