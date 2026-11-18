//! TC-5/8. Run a captured head RGB-D observation through the production pick planner.
use aisle_nano::{
    kinematics::Robot,
    runtime::Config,
    skills::{Controller, Skill},
};
use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Frame {
    state: Vec<f64>,
    camera: [[f64; 4]; 4],
    width: usize,
    height: usize,
    station: String,
    object: String,
    run: PathBuf,
}
#[test]
#[ignore = "Set AISLE_HOUSEHOLD_RGBD to the exported RGB-D observation directory"]
fn captured_household_grasp_plans_from_rgbd() {
    let root = PathBuf::from(
        std::env::var_os("AISLE_HOUSEHOLD_RGBD").expect("exported RGB-D observation directory"),
    );
    let frame: Frame = serde_json::from_slice(&fs::read(root.join("frame.json")).unwrap()).unwrap();
    let mut config: Config =
        serde_json::from_slice(&fs::read(frame.run.join("policy.json")).unwrap()).unwrap();
    config.output = std::env::temp_dir().join(format!("household-rgbd-{}", std::process::id()));
    fs::create_dir(&config.output).unwrap();
    fs::copy(
        frame.run.join("stations.json"),
        config.output.join("stations.json"),
    )
    .unwrap();
    let plan = config.output.join("ik-plan.json");
    let mut control = Controller::new(config.clone(), Robot::read(&config.model).unwrap()).unwrap();
    let rgb = fs::read(root.join("rgb")).unwrap();
    let bytes = fs::read(root.join("depth")).unwrap();
    let depth = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let m = frame.camera;
    let rotation = Matrix3::from_fn(|i, j| m[i][j]);
    let camera = Isometry3::from_parts(
        Translation3::new(m[0][3], m[1][3], m[2][3]),
        UnitQuaternion::from_matrix(&rotation),
    );
    control
        .start(
            1,
            Skill::Navigate {
                target: frame.station,
            },
            0,
            &frame.state,
        )
        .unwrap();
    for n in 0..12 {
        control
            .tick(
                n,
                n * 200_000_000,
                &frame.state,
                &rgb,
                &depth,
                &camera,
                frame.width,
                frame.height,
            )
            .unwrap();
    }
    control
        .start(
            2,
            Skill::Pick {
                object: frame.object,
                arm: "right".into(),
            },
            12,
            &frame.state,
        )
        .unwrap();
    let (_, failed) = control
        .tick(
            12,
            2_400_000_000,
            &frame.state,
            &rgb,
            &depth,
            &camera,
            frame.width,
            frame.height,
        )
        .unwrap();
    assert!(failed.is_none(), "{failed:?}");
    assert_eq!(control.phase(), "manipulate");
    let result: serde_json::Value = serde_json::from_slice(&fs::read(plan).unwrap()).unwrap();
    assert_eq!(result["poses"].as_array().unwrap().len(), 4);
    std::fs::write(
        root.join("verified-pick-plan.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    fs::remove_dir_all(config.output).unwrap();
}
