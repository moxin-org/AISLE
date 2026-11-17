use anyhow::{Context, Result, bail, ensure};
use nalgebra::{Isometry3, SMatrix, SVector, Translation3, Unit, UnitQuaternion, Vector3};
use std::{collections::HashMap, path::Path};

#[derive(Clone)]
pub struct Robot {
    pub names: Vec<String>,
    pub limits: Vec<[f64; 2]>,
    joints: Vec<urdf_rs::Joint>,
}
impl Robot {
    pub fn read(path: &Path) -> Result<Self> {
        let robot = urdf_rs::read_file(path)?;
        let mut names = Vec::new();
        let mut limits = Vec::new();
        for j in &robot.joints {
            if matches!(
                j.joint_type,
                urdf_rs::JointType::Revolute | urdf_rs::JointType::Continuous
            ) {
                names.push(j.name.clone());
                limits.push(if j.joint_type == urdf_rs::JointType::Continuous {
                    [-f64::INFINITY, f64::INFINITY]
                } else {
                    [j.limit.lower, j.limit.upper]
                });
            } else {
                ensure!(
                    j.joint_type == urdf_rs::JointType::Fixed,
                    "Unsupported joint {}",
                    j.name
                );
            }
        }
        ensure!(names.len() == 17, "Expected Nano's 17 joints");
        ensure!(
            names[..5]
                == [
                    "wheel_1_joint",
                    "wheel_2_joint",
                    "wheel_3_joint",
                    "head_pan_joint",
                    "head_tilt_joint"
                ],
            "Unsupported Nano joint order"
        );
        ensure!(
            names.iter().collect::<std::collections::HashSet<_>>().len() == 17,
            "Duplicate joint name"
        );
        let out = Self {
            names,
            limits,
            joints: robot.joints,
        };
        for s in ["left", "right"] {
            out.arm(s)?;
        }
        Ok(out)
    }
    pub fn arm(&self, side: &str) -> Result<Vec<usize>> {
        ensure!(matches!(side, "left" | "right"), "Unknown arm");
        [
            "j1_yaw",
            "j2_pitch",
            "j3_pitch",
            "j4_pitch",
            "j5_roll",
            "j6_gripper",
        ]
        .iter()
        .map(|j| {
            self.names
                .iter()
                .position(|n| n == &format!("{side}_{j}"))
                .context("Missing Nano joint")
        })
        .collect()
    }
    pub fn fk(&self, link: &str, q: &[f64]) -> Result<Isometry3<f64>> {
        ensure!(
            q.len() == 17 && q.iter().all(|v| v.is_finite()),
            "Invalid joints"
        );
        let mut poses = HashMap::from([("base_link".to_owned(), Isometry3::identity())]);
        for _ in 0..self.joints.len() {
            for j in &self.joints {
                if poses.contains_key(&j.child.link) {
                    continue;
                }
                if let Some(parent) = poses.get(&j.parent.link) {
                    let p = &j.origin;
                    let origin = Isometry3::from_parts(
                        Translation3::new(p.xyz[0], p.xyz[1], p.xyz[2]),
                        UnitQuaternion::from_euler_angles(p.rpy[0], p.rpy[1], p.rpy[2]),
                    );
                    let motion = if let Some(i) = self.names.iter().position(|n| n == &j.name) {
                        let axis = Unit::try_new(Vector3::from(j.axis.xyz.0), 1e-9)
                            .context("Zero joint axis")?;
                        Isometry3::from_parts(
                            Translation3::identity(),
                            UnitQuaternion::from_axis_angle(&axis, q[i]),
                        )
                    } else {
                        Isometry3::identity()
                    };
                    poses.insert(j.child.link.clone(), parent * origin * motion);
                }
            }
            if let Some(p) = poses.get(link) {
                return Ok(*p);
            }
        }
        bail!("Disconnected/unknown link {link}")
    }
    /// Damped least squares, optional local -Z approach direction; no collision claim.
    pub fn ik(
        &self,
        side: &str,
        link: &str,
        offset: Vector3<f64>,
        goal: Vector3<f64>,
        direction: Option<Vector3<f64>>,
        initial: &[f64],
    ) -> Result<Vec<f64>> {
        self.ik_axis(side, link, offset, goal, -Vector3::z(), direction, initial)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ik_axis(
        &self,
        side: &str,
        link: &str,
        offset: Vector3<f64>,
        goal: Vector3<f64>,
        local_axis: Vector3<f64>,
        direction: Option<Vector3<f64>>,
        initial: &[f64],
    ) -> Result<Vec<f64>> {
        ensure!(
            initial.len() == 17
                && goal
                    .iter()
                    .chain(offset.iter())
                    .chain(initial.iter())
                    .all(|x| x.is_finite()),
            "Non-finite IK request"
        );
        let indices = self.arm(side)?;
        let direction = direction
            .map(|d| {
                Unit::try_new(d, 1e-9)
                    .map(|x| *x)
                    .context("Invalid approach direction")
            })
            .transpose()?;
        let value = |q: &[f64]| -> Result<SVector<f64, 6>> {
            let p = self.fk(link, q)?;
            let xyz = p.transform_point(&offset.into()).coords;
            let dir = if direction.is_some() {
                p.rotation * local_axis * 0.12
            } else {
                Vector3::zeros()
            };
            Ok(SVector::from_row_slice(&[
                xyz.x, xyz.y, xyz.z, dir.x, dir.y, dir.z,
            ]))
        };
        let d = direction.unwrap_or(Vector3::zeros()) * 0.12;
        let target = SVector::<f64, 6>::from_row_slice(&[goal.x, goal.y, goal.z, d.x, d.y, d.z]);
        let mut best = f64::INFINITY;
        for restart in 0..8 {
            let mut q = initial.to_vec();
            if restart > 0 {
                for (j, i) in indices[..5].iter().enumerate() {
                    q[*i] = ((restart as f64 * (j + 1) as f64 * 1.731).sin() * 1.3)
                        .clamp(self.limits[*i][0], self.limits[*i][1]);
                }
            }
            for _ in 0..180 {
                let current = value(&q)?;
                let error = target - current;
                best = best.min(error.fixed_rows::<3>(0).norm());
                if error.fixed_rows::<3>(0).norm() < 0.002
                    && error.fixed_rows::<3>(3).norm() < 0.012
                {
                    return Ok(q);
                }
                let mut jac = SMatrix::<f64, 6, 5>::zeros();
                for (col, i) in indices[..5].iter().enumerate() {
                    let mut perturb = q.clone();
                    perturb[*i] += 1e-4;
                    jac.set_column(col, &((value(&perturb)? - current) / 1e-4));
                }
                let lhs = jac.transpose() * jac + SMatrix::<f64, 5, 5>::identity() * 0.0001;
                let Some(delta) = lhs.lu().solve(&(jac.transpose() * error)) else {
                    break;
                };
                for (j, i) in indices[..5].iter().enumerate() {
                    q[*i] = (q[*i] + delta[j].clamp(-0.15, 0.15))
                        .clamp(self.limits[*i][0], self.limits[*i][1]);
                }
            }
        }
        bail!("IK unreachable: best position error {best:.4} m")
    }
}
