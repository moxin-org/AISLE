use aisle_nano::{
    kinematics::Robot,
    navigation::Map,
    runtime::Config,
    skills::{Controller, Skill},
};
use std::path::PathBuf;

fn controller() -> Controller {
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/robots/adora_nano/model.urdf");
    let robot = Robot::read(&model).unwrap();
    Controller::new(
        Config {
            model,
            output: std::env::temp_dir(),
            map: Map {
                bounds: [-6., 6., -5., 5.],
                clearance: 0.55,
                obstacles: vec![],
            },
            goal: [1.6, -1.2],
            drop: [1.94, -1.43, 0.2],
            task: "pickup".into(),
            arm: "right".into(),
            seed: 0,
            max_frames: 1000,
        },
        robot,
    )
    .unwrap()
}
fn state() -> Vec<f64> {
    let mut s = vec![0.; 24];
    s[6] = 1.;
    s
}

/// TC-7/8: a semantic label cannot change the frozen task or skip pickup.
#[test]
fn skill_catalog_enforces_targets_arm_and_order() {
    let mut c = controller();
    let s = state();
    assert!(
        c.start(
            1,
            Skill::Navigate {
                target: "placement_zone".into()
            },
            0,
            &s
        )
        .is_err()
    );
    assert!(
        c.start(
            2,
            Skill::Place {
                target: "placement_zone".into(),
                arm: "right".into()
            },
            0,
            &s
        )
        .is_err()
    );
    assert!(
        c.start(
            3,
            Skill::Pick {
                object: "red cube".into(),
                arm: "left".into()
            },
            0,
            &s
        )
        .is_err()
    );
    assert!(
        c.start(
            4,
            Skill::Navigate {
                target: "station".into()
            },
            0,
            &s
        )
        .is_ok()
    );
    assert!(
        c.start(
            5,
            Skill::Navigate {
                target: "start".into()
            },
            1,
            &s
        )
        .is_err()
    );
}

/// TC-7, BG-1/2: cancellation terminates the active skill and holds measured joints.
#[test]
fn cancel_stops_navigation_and_allows_new_route() {
    let mut c = controller();
    let s = state();
    c.start(
        1,
        Skill::Navigate {
            target: "station".into(),
        },
        0,
        &s,
    )
    .unwrap();
    let outcome = c.cancel(&s, "operator cancelled").unwrap().unwrap();
    assert_eq!(outcome.status, "cancelled");
    assert_eq!(outcome.skill_id, 1);
    assert!(!c.active());
    assert_eq!(&c.hold(&s).unwrap()[..3], &[0., 0., 0.]);
    c.start(
        2,
        Skill::Navigate {
            target: "start".into(),
        },
        2,
        &s,
    )
    .unwrap();
    assert!(c.active());
}

/// TC-7: cancel is idempotent and malformed state never becomes a command.
#[test]
fn cancel_is_idempotent_and_state_is_validated() {
    let mut c = controller();
    assert!(c.cancel(&state(), "cancel").unwrap().is_none());
    assert!(c.hold(&[f64::NAN; 24]).is_err());
    assert!(
        c.start(
            1,
            Skill::Navigate {
                target: "station".into()
            },
            0,
            &[0.; 3]
        )
        .is_err()
    );
}

/// TC-7: duplicate admission is idempotent, changed payload and retired IDs fail.
#[test]
fn skill_identity_is_exactly_once() {
    let mut c = controller();
    let s = state();
    let skill = Skill::Navigate {
        target: "station".into(),
    };
    c.start(1, skill.clone(), 0, &s).unwrap();
    c.start(1, skill.clone(), 0, &s).unwrap();
    assert!(
        c.start(
            1,
            Skill::Navigate {
                target: "start".into()
            },
            0,
            &s
        )
        .is_err()
    );
    c.cancel(&s, "stop").unwrap();
    assert!(c.start(1, skill, 0, &s).is_err());
}
/// TC-7: model latency may span observations, but retirement and age are strict.
#[test]
fn planner_generation_rejects_late_results() {
    use aisle_nano::admission::Admission;
    let mut a = Admission::default();
    let old = a.request("mission", 100, 1000);
    assert!(a.accepts(&old, 100, 2000));
    assert!(!a.accepts(&old, 101, 2000));
    assert!(!a.accepts(&old, 100, 999));
    assert!(!a.accepts(&old, 100, 31001));
    assert_eq!(a.cancel(), Some(old.clone()));
    let new = a.request("mission", 200, 2000);
    assert_ne!(old, new);
    assert!(!a.accepts(&old, 100, 2001));
    assert!(a.accepts(&new, 200, 2001));
}

/// TC-7/8 and MOB-2. A household command must address the actual dock and a matching item.
#[test]
fn household_skill_labels_are_bound_to_the_current_station() {
    let catalog = aisle_nano::household::stations().unwrap();
    let here = &catalog[0];
    let model =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/robots/adora_nano/model.urdf");
    let config = Config {
        model: model.clone(),
        output: std::env::temp_dir().join(format!("nano-household-skill-{}", std::process::id())),
        map: Map {
            bounds: [-6., 6., -5., 5.],
            clearance: 0.55,
            obstacles: vec![],
        },
        goal: here.goal,
        drop: here.drop,
        task: "household".into(),
        arm: "right".into(),
        seed: 0,
        max_frames: 200,
    };
    std::fs::create_dir_all(&config.output).unwrap();
    aisle_nano::runtime::write(&config.output.join("stations.json"), &catalog).unwrap();
    let output = config.output.clone();
    let mut controller = Controller::new(config, Robot::read(&model).unwrap()).unwrap();
    let mut q = state();
    q[..2].copy_from_slice(&here.goal);
    let pick = |object: &str| Skill::Pick {
        object: object.to_owned(),
        arm: "right".into(),
    };
    assert_eq!(
        controller.capabilities()["available_subgoals"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert!(controller.start(1, pick(&here.object), 0, &q).is_err());
    controller
        .start(
            2,
            Skill::Navigate {
                target: here.station.clone(),
            },
            1,
            &q,
        )
        .unwrap();
    for seq in 2..=12 {
        controller
            .tick(
                seq,
                seq * 200_000_000,
                &q,
                &[],
                &[],
                &nalgebra::Isometry3::identity(),
                1,
                1,
            )
            .unwrap();
    }
    assert_eq!(controller.capabilities()["docked_at"], here.station);
    assert!(
        controller
            .start(3, pick(&catalog[1].object), 13, &q)
            .is_err()
    );
    assert!(
        controller
            .start(
                4,
                Skill::Navigate {
                    target: catalog[1].station.clone()
                },
                14,
                &q
            )
            .is_err()
    );
    controller.start(5, pick(&here.object), 15, &q).unwrap();
    assert!(!controller.placed());
    std::fs::remove_dir_all(output).unwrap();
}
