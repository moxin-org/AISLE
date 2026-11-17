//! Pure pieces of the independent judge node: oracle decoding, observation
//! pairing and task success. None of this reads controller output.
use crate::judge::{Judge, Sample, Verdict};
use anyhow::{Result, ensure};
use nalgebra::{Quaternion, UnitQuaternion};

/// Base heading from the simulator state: position, XYZW quaternion, 17 joints.
pub fn base_yaw(state: &[f64]) -> Result<f64> {
    ensure!(
        state.len() == 24 && state.iter().all(|v| v.is_finite()),
        "Invalid state"
    );
    let norm = state[3..7].iter().map(|v| v * v).sum::<f64>();
    ensure!((norm - 1.).abs() < 0.01, "Invalid base quaternion");
    let rotation =
        UnitQuaternion::new_normalize(Quaternion::new(state[6], state[3], state[4], state[5]));
    Ok(rotation.euler_angles().2)
}
/// One oracle row: base XY, cube XYZ, then finger, furniture and robot-cube contact flags.
pub fn sample(stamp: u64, truth: &[f64], yaw: f64) -> Result<Sample> {
    ensure!(
        truth.len() == 8
            && truth.iter().all(|x| x.is_finite())
            && truth[5..].iter().all(|x| *x == 0. || *x == 1.)
            && yaw.is_finite(),
        "Bad oracle sample"
    );
    Ok(Sample {
        stamp,
        base: [truth[0], truth[1]],
        yaw,
        cube: [truth[2], truth[3], truth[4]],
        finger_contact: truth[5] == 1.,
        furniture_contact: truth[6] == 1.,
        robot_contact: truth[7] == 1.,
    })
}
/// Task success from a complete evaluation. Inspection only captures the scene,
/// so it has no arrival or placement requirement.
pub fn success(task: &str, complete: bool, verdict: &Verdict) -> bool {
    complete
        && verdict.furniture_contacts == 0
        && match task {
            "inspect" => true,
            "navigate" => verdict.navigation,
            _ => verdict.navigation && verdict.placed,
        }
}
pub enum Input<'a> {
    State(&'a [f64]),
    Truth(&'a [f64]),
}
#[derive(Debug, PartialEq)]
pub enum Step {
    /// The other half of this observation has not arrived.
    Pending,
    /// A new observation was judged.
    Observed(Sample),
    /// An identical retransmission of the last judged observation.
    Repeated,
}
/// Pairs each oracle row with the base heading of the same observation and
/// judges every observation exactly once, in order.
pub struct Feed {
    goal: [f64; 2],
    drop: [f64; 3],
    judge: Option<Judge>,
    last: Option<(u64, Sample)>,
    yaw: Option<(u64, u64, f64)>,
    truth: Option<(u64, u64, Vec<f64>)>,
}
impl Feed {
    pub fn new(goal: [f64; 2], drop: [f64; 3]) -> Self {
        Self {
            goal,
            drop,
            judge: None,
            last: None,
            yaw: None,
            truth: None,
        }
    }
    pub fn offer(&mut self, seq: u64, stamp: u64, input: Input) -> Result<Step> {
        match input {
            Input::State(state) => self.yaw = Some((seq, stamp, base_yaw(state)?)),
            Input::Truth(truth) => self.truth = Some((seq, stamp, truth.to_vec())),
        }
        let (Some((a, heading_stamp, yaw)), Some((b, truth_stamp, row))) = (&self.yaw, &self.truth)
        else {
            return Ok(Step::Pending);
        };
        if a != b {
            return Ok(Step::Pending);
        }
        ensure!(heading_stamp == truth_stamp, "Mismatched oracle clocks");
        let sample = sample(*truth_stamp, row, *yaw)?;
        self.yaw = None;
        self.truth = None;
        if let Some((last, previous)) = self.last
            && last == seq
        {
            ensure!(
                previous == sample,
                "Oracle changed an observation during retry"
            );
            return Ok(Step::Repeated);
        }
        // A late duplicate of an observation already left behind changes nothing.
        if self.last.is_some_and(|(last, _)| seq < last) {
            return Ok(Step::Pending);
        }
        ensure!(
            self.last.is_none_or(|(last, _)| seq == last + 1),
            "Missing oracle observation"
        );
        let (goal, drop) = (self.goal, self.drop);
        self.judge
            .get_or_insert_with(|| Judge::new(goal, sample.cube[2], drop))
            .observe(sample)?;
        self.last = Some((seq, sample));
        Ok(Step::Observed(sample))
    }
    /// The last judged observation.
    pub fn last(&self) -> Option<u64> {
        self.last.map(|(seq, _)| seq)
    }
    pub fn verdict(&self) -> Option<Verdict> {
        self.judge.as_ref().map(Judge::result)
    }
}
