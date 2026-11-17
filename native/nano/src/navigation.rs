use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Box2 {
    pub center: [f64; 2],
    pub half: [f64; 2],
    #[serde(default)]
    pub clearance: Option<f64>,
}
pub fn base_command(goal: [f64; 2], pose: [f64; 3]) -> Result<(f64, f64)> {
    ensure!(
        goal.iter().chain(&pose).all(|v| v.is_finite()),
        "Invalid navigation pose"
    );
    let d = distance([pose[0], pose[1]], goal);
    let heading = if d < 0.035 {
        0.
    } else {
        (goal[1] - pose[1]).atan2(goal[0] - pose[0])
    };
    let error = (heading - pose[2] + std::f64::consts::PI).rem_euclid(2. * std::f64::consts::PI)
        - std::f64::consts::PI;
    let turn = if error.abs() < 0.025 {
        0.
    } else {
        error.signum() * (error.abs() * 1.5).clamp(0.12, 0.45)
    };
    let speed = if d < 0.035 || error.abs() > 0.2 {
        0.
    } else {
        (d * 0.8).clamp(0.07, 0.12) * error.cos().powi(3)
    };
    Ok((speed, turn))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Map {
    pub bounds: [f64; 4],
    pub clearance: f64,
    pub obstacles: Vec<Box2>,
}
pub fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
impl Map {
    pub fn clear(&self, p: [f64; 2]) -> bool {
        p.iter().all(|v| v.is_finite())
            && p[0] >= self.bounds[0] + self.clearance
            && p[0] <= self.bounds[1] - self.clearance
            && p[1] >= self.bounds[2] + self.clearance
            && p[1] <= self.bounds[3] - self.clearance
            && self.obstacles.iter().all(|b| {
                (p[0] - b.center[0]).abs() > b.half[0] + b.clearance.unwrap_or(self.clearance)
                    || (p[1] - b.center[1]).abs()
                        > b.half[1] + b.clearance.unwrap_or(self.clearance)
            })
    }
    pub fn clear_segment(&self, a: [f64; 2], b: [f64; 2]) -> bool {
        if !self.clear(a) || !self.clear(b) {
            return false;
        }
        let n = (distance(a, b) / 0.025).ceil() as usize;
        (0..=n.max(1)).all(|i| {
            let t = i as f64 / n.max(1) as f64;
            self.clear([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        })
    }
    pub fn plan(&self, start: [f64; 2], goal: [f64; 2]) -> Result<Vec<[f64; 2]>> {
        ensure!(
            self.clear(start) && self.clear(goal),
            "Start or goal blocked"
        );
        if self.clear_segment(start, goal) {
            return Ok(vec![goal]);
        }
        let step = 0.1;
        let index = |p: [f64; 2]| {
            (
                ((p[0] - self.bounds[0]) / step).round() as i32,
                ((p[1] - self.bounds[2]) / step).round() as i32,
            )
        };
        let point = |i: (i32, i32)| {
            [
                self.bounds[0] + i.0 as f64 * step,
                self.bounds[2] + i.1 as f64 * step,
            ]
        };
        let begin = index(start);
        let end = index(goal);
        ensure!(
            self.clear_segment(start, point(begin)) && self.clear_segment(point(end), goal),
            "Blocked grid endpoint"
        );
        let mut todo = BinaryHeap::from([(0, begin)]);
        let mut cost = HashMap::from([(begin, 0.)]);
        let mut prev = HashMap::new();
        while let Some((_, p)) = todo.pop() {
            if p == end {
                break;
            }
            for d in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let next = (p.0 + d.0, p.1 + d.1);
                if !self.clear_segment(point(p), point(next)) {
                    continue;
                }
                let c = cost[&p] + distance(point(p), point(next));
                if c >= *cost.get(&next).unwrap_or(&f64::INFINITY) {
                    continue;
                }
                cost.insert(next, c);
                prev.insert(next, p);
                todo.push((-((c + distance(point(next), goal)) * 1e6) as i64, next));
            }
        }
        if !cost.contains_key(&end) {
            bail!("No collision-free route");
        }
        let mut rev = vec![goal];
        let mut at = end;
        while at != begin {
            rev.push(point(at));
            at = prev[&at];
        }
        rev.push(start);
        rev.reverse();
        let mut route = Vec::new();
        let mut i = 0;
        while i < rev.len() - 1 {
            let j = (i + 1..rev.len())
                .rev()
                .find(|j| self.clear_segment(rev[i], rev[*j]))
                .unwrap();
            route.push(rev[j]);
            i = j;
        }
        Ok(route)
    }
}
