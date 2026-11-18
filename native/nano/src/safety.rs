use crate::kinematics::Robot;
use anyhow::{Result, ensure};
pub fn validate_arm_transition(from: &[f64], to: &[f64], joints: &[usize]) -> Result<()> {
    ensure!(
        from.len() == 17 && to.len() == 17,
        "Invalid trajectory pose"
    );
    ensure!(
        joints
            .iter()
            .all(|&i| i < 17 && (to[i] - from[i]).abs() <= 1.5),
        "IK branch discontinuity"
    );
    Ok(())
}
pub struct Guard {
    robot: Robot,
    last: Option<u64>,
}
impl Guard {
    pub fn new(robot: Robot) -> Self {
        Self { robot, last: None }
    }
    pub fn apply(&mut self, stamp: u64, q: &[f64], command: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            self.last.is_none_or(|s| stamp > s),
            "Repeated/regressing command clock"
        );
        ensure!(
            q.len() == 17 && command.len() == 17 && q.iter().chain(command).all(|x| x.is_finite()),
            "Invalid motion data"
        );
        self.last = Some(stamp);
        let mut safe = command.to_vec();
        for (i, v) in safe.iter_mut().enumerate() {
            *v = v.clamp(if i < 3 { -5. } else { -0.5 }, if i < 3 { 5. } else { 0.5 });
            let [lo, hi] = self.robot.limits[i];
            *v = (*v).clamp(((lo - q[i]) / 0.2).min(0.), ((hi - q[i]) / 0.2).max(0.));
        }
        if safe[5..].iter().any(|x| x.abs() > 1e-5) {
            safe[..3].fill(0.);
        }
        // Coarse reach/floor sanity bounds; furniture requires a collision planner.
        let projected: Vec<_> = q.iter().zip(&safe).map(|(q, v)| q + 0.2 * v).collect();
        for side in ["left", "right"] {
            let p = self
                .robot
                .fk(&format!("{side}_tool_mount"), &projected)?
                .translation
                .vector;
            if p.x.abs() > 0.8 || p.y.abs() > 0.8 || !(-0.05..1.).contains(&p.z) {
                for i in self.robot.arm(side)? {
                    safe[i] = 0.;
                }
            }
        }
        Ok(safe)
    }
}
