use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub stamp: u64,
    pub base: [f64; 2],
    pub cube: [f64; 3],
    pub finger_contact: bool,
    pub robot_contact: bool,
    pub furniture_contact: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verdict {
    pub navigation: bool,
    pub pickup: bool,
    pub placed: bool,
    pub furniture_contacts: u64,
    pub max_lift_m: f64,
    pub samples: u64,
}
pub struct Judge {
    goal: [f64; 2],
    initial_z: f64,
    drop: [f64; 3],
    last: Option<Sample>,
    lift_since: Option<u64>,
    release_since: Option<u64>,
    arrival_since: Option<u64>,
    supported_placement: bool,
    v: Verdict,
}
impl Judge {
    pub fn new(goal: [f64; 2], initial_z: f64, drop: [f64; 3]) -> Self {
        Self {
            goal,
            initial_z,
            drop,
            last: None,
            lift_since: None,
            release_since: None,
            arrival_since: None,
            supported_placement: false,
            v: Verdict {
                navigation: false,
                pickup: false,
                placed: false,
                furniture_contacts: 0,
                max_lift_m: 0.,
                samples: 0,
            },
        }
    }
    pub fn observe(&mut self, s: Sample) -> Result<()> {
        ensure!(
            self.last.is_none_or(|p| s.stamp > p.stamp),
            "Regressing judge clock"
        );
        ensure!(
            self.last.is_none_or(|p| s.stamp - p.stamp <= 200_000_000),
            "Missing physics observation"
        );
        ensure!(
            s.base.iter().chain(&s.cube).all(|x| x.is_finite()),
            "Invalid physics sample"
        );
        self.v.samples += 1;
        let stopped = self
            .last
            .is_some_and(|p| crate::navigation::distance(s.base, p.base) < 0.005);
        if crate::navigation::distance(s.base, self.goal) < 0.12 && stopped {
            let since = *self.arrival_since.get_or_insert(s.stamp);
            self.v.navigation = s.stamp - since >= 400_000_000;
        } else {
            self.arrival_since = None;
            self.v.navigation = false;
        }
        if s.furniture_contact {
            self.v.furniture_contacts += 1;
        }
        let lift = s.cube[2] - self.initial_z;
        self.v.max_lift_m = self.v.max_lift_m.max(lift);
        if lift > 0.07 && s.finger_contact {
            let since = *self.lift_since.get_or_insert(s.stamp);
            if s.stamp - since >= 400_000_000 {
                self.v.pickup = true;
            }
        } else {
            self.lift_since = None;
        }
        let in_placement_zone = |cube: &[f64; 3]| {
            (0..3).all(|i| (cube[i] - self.drop[i]).abs() < if i == 2 { 0.025 } else { 0.05 })
        };
        let near = in_placement_zone(&s.cube);
        let entering = near && self.last.is_some_and(|p| !in_placement_zone(&p.cube));
        let stable = self
            .last
            .is_some_and(|p| (0..3).all(|i| (s.cube[i] - p.cube[i]).abs() < 0.003));
        if !near {
            self.supported_placement = false;
        }
        if self.v.pickup && entering && s.finger_contact {
            self.supported_placement = true;
        }
        if self.supported_placement && near && stable && !s.robot_contact {
            let since = *self.release_since.get_or_insert(s.stamp);
            if s.stamp - since >= 1_000_000_000 {
                self.v.placed = true;
            }
        } else {
            self.release_since = None;
            self.v.placed = false;
        }
        self.last = Some(s);
        Ok(())
    }
    pub fn result(&self) -> Verdict {
        self.v.clone()
    }
}
