//! Boundary behaviour of the guard, planner, perception, kinematics and base helpers.
use aisle_nano::{
    kinematics::Robot,
    navigation::{Box2, Map},
    perception::{Color, detect},
    runtime,
    safety::{Guard, validate_arm_transition},
};
use nalgebra::{Isometry3, Point3, Vector3};

fn robot() -> Robot {
    Robot::read(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/robots/adora_nano/model.urdf"
    )))
    .unwrap()
}
fn command(values: &[(usize, f64)]) -> Vec<f64> {
    let mut v = vec![0.; 17];
    for (i, x) in values {
        v[*i] = *x;
    }
    v
}

/// BG-1, BG-2: velocity commands are capped at 0.5 rad/s per arm joint and 5 rad/s per wheel.
#[test]
fn guard_caps_joint_and_wheel_velocity() {
    let q = vec![0.; 17];
    let mut g = Guard::new(robot());
    let safe = g.apply(1, &q, &command(&[(5, 99.), (10, -99.)])).unwrap();
    assert_eq!((safe[5], safe[10]), (0.5, -0.5));
    let safe = g
        .apply(2, &q, &command(&[(0, 99.), (1, -99.), (2, 3.)]))
        .unwrap();
    assert_eq!(&safe[..3], &[5., -5., 3.]);
}

/// BG-1: motion stops at both position limits and slows when one is 0.2 s away.
#[test]
fn guard_stops_at_both_position_limits() {
    let r = robot();
    let [lo, hi] = r.limits[15];
    let mut g = Guard::new(r);
    let mut q = vec![0.; 17];
    q[15] = lo;
    assert_eq!(g.apply(1, &q, &command(&[(15, -0.4)])).unwrap()[15], 0.);
    assert_eq!(g.apply(2, &q, &command(&[(15, 0.4)])).unwrap()[15], 0.4);
    q[15] = hi;
    assert_eq!(g.apply(3, &q, &command(&[(15, 0.4)])).unwrap()[15], 0.);
    q[15] = hi - 0.02;
    let slowed = g.apply(4, &q, &command(&[(15, 0.5)])).unwrap()[15];
    assert!((slowed - 0.1).abs() < 1e-9, "{slowed}");
}

/// BG-1, BG-2: wrong-sized state or command vectors are refused, not truncated.
#[test]
fn guard_refuses_wrong_sized_vectors() {
    let mut g = Guard::new(robot());
    assert!(g.apply(1, &[0.; 18], &[0.; 17]).is_err());
    assert!(g.apply(2, &[0.; 17], &[0.; 16]).is_err());
    assert!(g.apply(3, &[0.; 17], &[0.; 17]).is_ok());
    assert!(validate_arm_transition(&[0.; 18], &[0.; 18], &[5, 6, 7, 8, 9]).is_err());
    assert!(validate_arm_transition(&[0.; 17], &[0.; 17], &[17]).is_err());
}

fn walled() -> Map {
    Map {
        bounds: [-3., 3., -3., 3.],
        clearance: 0.4,
        obstacles: vec![Box2 {
            center: [0., 0.],
            half: [0.5, 1.],
            clearance: None,
        }],
    }
}

/// MOB-2: obstacles and room bounds are both grown by the base clearance.
#[test]
fn map_inflates_obstacles_and_bounds() {
    let m = walled();
    assert!(!m.clear([0.7, 0.]));
    assert!(!m.clear([0., 1.2]));
    assert!(m.clear([0.95, 0.]));
    assert!(m.clear([0., 1.45]));
    assert!(!m.clear([2.8, 2.]));
    assert!(!m.clear([2., -2.8]));
    assert!(m.clear([2.5, -2.5]));
    assert!(!m.clear([f64::NAN, 0.]));
    assert!(m.plan([-2., 0.], [2.9, 0.]).is_err());
    assert!(m.plan([-2., 0.], [0.7, 0.]).is_err());
    assert!(!m.clear_segment([-2., 0.], [2., 0.]));
}

/// MOB-2: the room map keeps every fixed piece of furniture and drops the floor.
#[test]
fn room_map_contains_the_furniture() {
    let objects = runtime::room(0, [1.6, -1.2], "right");
    let map = runtime::map(&objects);
    let fixed = objects.iter().filter(|o| o["fixed"] == true).count();
    assert_eq!(map.obstacles.len(), fixed - 1);
    for inside in [[0., 2.6], [0., 1.1], [-3.4, 0.3], [0.85, -2.5], [4.8, 2.]] {
        assert!(!map.clear(inside), "{inside:?}");
    }
    assert!(map.clear([0., 0.]));
    // The low pickup table uses its own, tighter docking clearance.
    let table = map
        .obstacles
        .iter()
        .find(|b| b.clearance.is_some())
        .unwrap();
    assert_eq!(table.clearance, Some(0.27));
    assert!(map.clear([1.6, -1.2]));
}

/// MOB-2: wheel rates follow the three-wheel layout and stay inside 5 rad/s.
#[test]
fn wheel_rates_match_the_base_layout() {
    let turn = runtime::wheel_rates(0., 0., 0.3);
    assert!(turn.iter().all(|w| (w + 1.02).abs() < 1e-9), "{turn:?}");
    let forward = runtime::wheel_rates(0.1, 0., 0.);
    assert_eq!(forward[0], 0.);
    assert!((forward[1] - 0.1 * 3f64.sqrt() / 2. / 0.034).abs() < 1e-9);
    assert_eq!(forward[2], -forward[1]);
    let fast = runtime::wheel_rates(1., 0., 0.);
    assert!((fast[1] - 5.).abs() < 1e-9 && (fast[2] + 5.).abs() < 1e-9);
}

/// TC-9: a camera pose must be a finite, rigid, row-major 4x4 transform.
#[test]
fn camera_pose_must_be_rigid() {
    #[rustfmt::skip]
    let quarter_turn = [
        0., -1., 0., 1.,
        1., 0., 0., 2.,
        0., 0., 1., 3.,
        0., 0., 0., 1.,
    ];
    let pose = runtime::transform(&quarter_turn).unwrap();
    let p = pose.transform_point(&Point3::new(1., 0., 0.));
    assert!((p.coords - Vector3::new(1., 3., 3.)).norm() < 1e-9);
    let mut scaled = quarter_turn;
    scaled[1] = -2.;
    assert!(runtime::transform(&scaled).is_err());
    let mut mirrored = quarter_turn;
    mirrored[10] = -1.;
    assert!(runtime::transform(&mirrored).is_err());
    let mut nan = quarter_turn;
    nan[3] = f64::NAN;
    assert!(runtime::transform(&nan).is_err());
    assert!(runtime::transform(&quarter_turn[..15]).is_err());
}

const W: usize = 40;
const H: usize = 40;
fn image(blobs: &[(usize, usize, usize, [u8; 3])]) -> Vec<u8> {
    let mut rgb = vec![0u8; W * H * 3];
    for (x0, y0, size, color) in blobs {
        for y in *y0..y0 + size {
            for x in *x0..x0 + size {
                rgb[(y * W + x) * 3..(y * W + x) * 3 + 3].copy_from_slice(color);
            }
        }
    }
    rgb
}
fn red(
    rgb: &[u8],
    depth: f32,
    focal: f64,
) -> anyhow::Result<Vec<aisle_nano::perception::Detection>> {
    detect(
        rgb,
        &vec![depth; W * H],
        W,
        H,
        focal,
        Color::Red,
        1,
        1,
        &Isometry3::identity(),
    )
}
const RED: [u8; 3] = [220, 20, 15];

/// TC-9: an off-centre blob back-projects to the right place: +x right, +y up, -z forward.
#[test]
fn perception_back_projects_off_centre_pixels() {
    // Columns 4..12 and rows 2..10 of a 40 px image, focal 40 px, 2 m away.
    let hits = red(&image(&[(4, 2, 8, RED)]), 2., 40.).unwrap();
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!((hit.pixels, hit.bounds), (64, [4, 2, 11, 9]));
    assert!(
        (hit.position - Vector3::new(-0.6, 0.7, -2.)).norm() < 1e-9,
        "{:?}",
        hit.position
    );
    assert!(
        (hit.extent - Vector3::new(0.35, 0.35, 0.)).norm() < 1e-9,
        "{:?}",
        hit.extent
    );
}

/// TC-9: specks, non-red surfaces and out-of-range depth are not detections.
#[test]
fn perception_filters_noise_colour_and_depth() {
    assert!(red(&image(&[(4, 2, 4, RED)]), 2., 40.).unwrap().is_empty());
    assert_eq!(red(&image(&[(4, 2, 5, RED)]), 2., 40.).unwrap().len(), 1);
    for other in [[220, 200, 190], [200, 120, 20], [200, 30, 150]] {
        assert!(
            red(&image(&[(4, 2, 8, other)]), 2., 40.)
                .unwrap()
                .is_empty(),
            "{other:?}"
        );
    }
    for depth in [0.01, 20., f32::INFINITY] {
        assert!(
            red(&image(&[(4, 2, 8, RED)]), depth, 40.)
                .unwrap()
                .is_empty()
        );
    }
    for focal in [0., -40., f64::NAN] {
        assert!(red(&image(&[(4, 2, 8, RED)]), 2., focal).is_err());
    }
}

/// TC-9: detections are ordered largest first, whatever the scan order.
#[test]
fn perception_orders_detections_by_size() {
    let hits = red(&image(&[(2, 2, 5, RED), (20, 20, 10, RED)]), 2., 40.).unwrap();
    assert_eq!(hits.iter().map(|h| h.pixels).collect::<Vec<_>>(), [100, 25]);
}

/// TC-1, TC-5, BG-1: IK never returns a pose outside the joint limits.
#[test]
fn ik_never_leaves_joint_limits() {
    let r = robot();
    for side in ["left", "right"] {
        let arm = r.arm(side).unwrap();
        let link = format!("{side}_tool_mount");
        for joint in 0..4 {
            for sign in [-1., 1.] {
                // A point that this joint could only reach 0.4 rad past its limit.
                let mut at_limit = vec![0.; 17];
                at_limit[arm[1]] = 0.4;
                at_limit[arm[2]] = -0.6;
                at_limit[arm[joint]] = sign * r.limits[arm[joint]][1];
                let mut beyond = at_limit.clone();
                beyond[arm[joint]] += sign * 0.4;
                let goal = r.fk(&link, &beyond).unwrap().translation.vector;
                if let Ok(q) = r.ik(side, &link, Vector3::zeros(), goal, None, &at_limit) {
                    assert!(
                        q.iter()
                            .zip(&r.limits)
                            .all(|(v, l)| *v >= l[0] && *v <= l[1]),
                        "{side} joint {joint}: {q:?}"
                    );
                    assert!((r.fk(&link, &q).unwrap().translation.vector - goal).norm() < 0.003);
                }
            }
        }
    }
}

/// TC-1, TC-5: reaching the position is not enough when a jaw direction was requested.
#[test]
fn ik_does_not_trade_away_the_requested_direction() {
    let r = robot();
    for side in ["left", "right"] {
        let sign = if side == "left" { 1. } else { -1. };
        let link = format!("{side}_wrist_housing");
        let offset = Vector3::new(0.012, 0., -0.12);
        let goal = Vector3::new(0.404, sign * 0.23, 0.30);
        let aligned = r
            .ik_axis(
                side,
                &link,
                offset,
                goal,
                Vector3::x(),
                Some(Vector3::y() * sign),
                &[0.; 17],
            )
            .unwrap();
        // Same point, already reached, but with the jaw asked to point elsewhere.
        for wanted in [
            Vector3::new(0.5, sign * 0.87, 0.),
            Vector3::new(0., sign * 0.8, 0.6),
        ] {
            let wanted = wanted.normalize();
            if let Ok(q) = r.ik_axis(
                side,
                &link,
                offset,
                goal,
                Vector3::x(),
                Some(wanted),
                &aligned,
            ) {
                let pose = r.fk(&link, &q).unwrap();
                assert!(
                    (pose.rotation * Vector3::x() - wanted).norm() < 0.11,
                    "{side} {wanted:?}"
                );
                assert!((pose.transform_point(&offset.into()).coords - goal).norm() < 0.003);
            }
        }
    }
}

/// CON-4, CON-5: a model snapshot holds exactly the URDF and its meshes; unsafe mesh paths are refused.
#[test]
fn model_snapshot_is_complete_and_contained() {
    use aisle_nano::assets;
    let dir = std::env::temp_dir().join(format!("aisle-nano-assets-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src/meshes")).unwrap();
    let urdf = |mesh: &str| {
        format!(
            r#"<robot name="r"><link name="a"><visual><geometry><mesh filename="{mesh}"/></geometry></visual>
            <collision><geometry><mesh filename="./meshes/b.stl"/></geometry></collision></link></robot>"#
        )
    };
    let model = dir.join("src/robot.urdf");
    std::fs::write(&model, urdf("package://adora_nano/meshes/a.stl")).unwrap();
    std::fs::write(dir.join("src/meshes/a.stl"), b"a").unwrap();
    assert!(
        assets::meshes(&model)
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    std::fs::write(dir.join("src/meshes/b.stl"), b"b").unwrap();
    let frozen = assets::snapshot(&model, &dir.join("copy")).unwrap();
    assert_eq!(frozen, dir.join("copy/model.urdf"));
    assert_eq!(std::fs::read(dir.join("copy/meshes/a.stl")).unwrap(), b"a");
    assert_eq!(std::fs::read(dir.join("copy/meshes/b.stl")).unwrap(), b"b");
    let manifest = assets::manifest(&frozen).unwrap();
    assert_eq!(manifest.len(), 3);
    assert_eq!(manifest["model.urdf"], assets::sha256(&model).unwrap());
    // An existing snapshot directory is never merged into.
    assert!(assets::snapshot(&model, &dir.join("copy")).is_err());
    std::fs::write(dir.join("outside.stl"), b"x").unwrap();
    std::fs::write(dir.join("src/model.urdf"), b"x").unwrap();
    for unsafe_path in ["../outside.stl", "/etc/hosts", "model.urdf"] {
        std::fs::write(&model, urdf(unsafe_path)).unwrap();
        assert!(assets::meshes(&model).is_err(), "{unsafe_path}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
