use anyhow::{Result, ensure};
use nalgebra::{Isometry3, Point3, Vector3};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Color {
    Red,
    Blue,
    Green,
    Yellow,
}
impl Color {
    fn matches(self, p: &[u8]) -> bool {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        match self {
            Self::Red => r > 65. && r > g * 2.5 && r > b * 2.,
            Self::Blue => b > 70. && b > r * 1.8 && b > g * 1.15,
            Self::Green => g > 65. && g > r * 1.8 && g > b * 1.3,
            Self::Yellow => r > 90. && g > 65. && b < g * 0.65 && g > r * 0.65,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Detection {
    pub position: Vector3<f64>,
    pub pixels: usize,
    pub bounds: [usize; 4],
    pub extent: Vector3<f64>,
}
/// Connected color components; no segmentation or simulator object state input.
#[allow(clippy::too_many_arguments)]
pub fn detect(
    rgb: &[u8],
    depth: &[f32],
    w: usize,
    h: usize,
    focal: f64,
    color: Color,
    rgb_stamp: u64,
    depth_stamp: u64,
    camera: &Isometry3<f64>,
) -> Result<Vec<Detection>> {
    ensure!(rgb_stamp == depth_stamp, "Unsynchronized RGB/depth");
    ensure!(
        w > 0
            && h > 0
            && w.checked_mul(h)
                .is_some_and(|n| n.checked_mul(3) == Some(rgb.len()) && n == depth.len()),
        "Invalid image dimensions"
    );
    ensure!(focal.is_finite() && focal > 0., "Invalid intrinsics");
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for root in 0..w * h {
        if seen[root] || !color.matches(&rgb[root * 3..root * 3 + 3]) {
            continue;
        }
        seen[root] = true;
        let mut queue = vec![root];
        let mut i = 0;
        let mut points = Vec::new();
        let mut bounds = [w, h, 0, 0];
        while i < queue.len() {
            let p = queue[i];
            i += 1;
            let x = p % w;
            let y = p / w;
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
            let z = depth[p] as f64;
            if z.is_finite() && z > 0.02 && z < 8. {
                points.push(
                    camera
                        .transform_point(&Point3::new(
                            (x as f64 + 0.5 - w as f64 / 2.) * z / focal,
                            -(y as f64 + 0.5 - h as f64 / 2.) * z / focal,
                            -z,
                        ))
                        .coords,
                );
            }
            for (nx, ny) in [
                (x as isize - 1, y as isize),
                (x as isize + 1, y as isize),
                (x as isize, y as isize - 1),
                (x as isize, y as isize + 1),
            ] {
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let n = ny as usize * w + nx as usize;
                if !seen[n] && color.matches(&rgb[n * 3..n * 3 + 3]) {
                    seen[n] = true;
                    queue.push(n);
                }
            }
        }
        if points.len() < 20 {
            continue;
        }
        let lo = points
            .iter()
            .fold(Vector3::repeat(f64::INFINITY), |a, p| a.inf(p));
        let hi = points
            .iter()
            .fold(Vector3::repeat(-f64::INFINITY), |a, p| a.sup(p));
        out.push(Detection {
            position: (lo + hi) / 2.,
            pixels: points.len(),
            bounds,
            extent: hi - lo,
        });
    }
    out.sort_by_key(|d| std::cmp::Reverse(d.pixels));
    Ok(out)
}
