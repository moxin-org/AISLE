//! Audit the recorded body transforms and per-tick Nexus contact pairs after the run.
use aisle_nano::household::Station;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::Command,
};

#[derive(Deserialize)]
struct Object {
    name: String,
}
#[derive(Deserialize)]
struct Base {
    position: [f64; 3],
}
#[derive(Deserialize)]
struct Frame {
    transforms: Vec<[f64; 16]>,
    base_pose: Base,
}
#[derive(Deserialize)]
struct Scene {
    objects: Vec<Object>,
    frames: Vec<Frame>,
}

fn rows(file: &Path) -> Result<Vec<[i64; 3]>> {
    let out = Command::new("unzip")
        .args(["-p"])
        .arg(file)
        .arg("physics_contacts.npy")
        .output()?;
    ensure!(
        out.status.success(),
        "missing contacts in {}",
        file.display()
    );
    let b = out.stdout;
    ensure!(
        b.starts_with(b"\x93NUMPY\x01\x00") && b.len() >= 10,
        "unsupported contact format"
    );
    let header_end = 10 + u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = std::str::from_utf8(b.get(10..header_end).context("truncated npy")?)?;
    ensure!(
        header.contains("'<i8'")
            && header.contains("'fortran_order': False")
            && header.contains(", 3)"),
        "invalid contact array"
    );
    let (rows, remainder) = b[header_end..].as_chunks::<24>();
    ensure!(remainder.is_empty(), "truncated contact rows");
    Ok(rows
        .iter()
        .map(|r| {
            std::array::from_fn(|i| i64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap()))
        })
        .collect())
}

fn contacts(dataset: &Path, frame: usize, target: usize) -> Result<BTreeMap<i64, BTreeSet<usize>>> {
    let mut ticks: BTreeMap<i64, BTreeSet<usize>> = BTreeMap::new();
    for [t, a, b] in rows(&dataset.join(format!("raw/frame_{frame:06}.npz")))? {
        let other = if a == target as i64 {
            b
        } else if b == target as i64 {
            a
        } else {
            continue;
        };
        ticks.entry(t).or_default().insert(usize::try_from(other)?);
    }
    Ok(ticks)
}

fn in_region(m: &[f64; 16], s: &Station) -> bool {
    // Account for the full rotated rigid box when testing the destination footprint.
    let half = |axis: usize| {
        (0..3)
            .map(|i| m[i * 4 + axis].abs() * s.size[i] / 2.)
            .sum::<f64>()
    };
    let [x, y, rx, ry] = s.region;
    (m[12] - x).abs() + half(0) <= rx
        && (m[13] - y).abs() + half(1) <= ry
        && (m[14] - half(2) - s.surface_z).abs() < 0.008
}

fn check_roster(entries: &[Station]) -> Result<()> {
    let reference = aisle_nano::household::stations()?;
    let pairs = |stations: &[Station]| -> BTreeSet<(String, String, String)> {
        stations
            .iter()
            .map(|s| (s.station.clone(), s.object.clone(), s.destination.clone()))
            .collect()
    };
    ensure!(
        entries.len() == reference.len() && pairs(entries) == pairs(&reference),
        "recording must contain each of the four household deliveries"
    );
    Ok(())
}

pub fn audit(root: &Path) -> Result<Value> {
    let read = |f: &str| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(root.join(f))?)?) };
    let capture = read("capture-complete.json")?;
    let data = root.join("dataset");
    let scene: Scene =
        serde_json::from_slice(&fs::read(data.join("meta/scene/episode_000000.json"))?)?;
    let entries: Vec<Station> = serde_json::from_value(read("stations.json")?)?;
    check_roster(&entries)?;
    let n = scene.frames.len();
    ensure!(
        n >= 12 && capture["frames"].as_u64() == Some(n as u64),
        "incomplete recording"
    );
    let ids: BTreeMap<_, _> = scene
        .objects
        .iter()
        .enumerate()
        .map(|(i, o)| (o.name.as_str(), i))
        .collect();
    let bodies = scene.objects.len();
    ensure!(
        scene.frames.iter().all(|f| f.transforms.len() == bodies),
        "body count changed"
    );
    let mut items = Vec::new();
    for s in &entries {
        let id = *ids.get(s.object.as_str()).context("missing body")?;
        let destination = *ids
            .get(s.destination.as_str())
            .context("missing destination")?;
        let shelf = format!("{}_source_shelf", s.station.replace(' ', "_"));
        let source = *ids.get(shelf.as_str()).context("missing source shelf")?;
        let z0 = scene.frames[0].transforms[id][14];
        let peak = (0..n)
            .max_by(|&a, &b| {
                scene.frames[a].transforms[id][14].total_cmp(&scene.frames[b].transforms[id][14])
            })
            .unwrap();
        let lifted = scene.frames[peak].transforms[id][14] - z0;
        let mut pinched = false;
        let first = peak.saturating_sub(9);
        for (offset, frame) in scene.frames[first..=(peak + 9).min(n - 1)]
            .iter()
            .enumerate()
        {
            if frame.transforms[id][14] - z0 < 0.035 {
                continue;
            }
            if contacts(&data, first + offset, id)?
                .values()
                .any(|peers| peers.range(bodies..).count() >= 2)
            {
                pinched = true;
                break;
            }
        }
        let support = |i, body| -> Result<bool> {
            Ok(contacts(&data, i, id)?
                .values()
                .any(|set| set.contains(&body)))
        };
        let end = &scene.frames[n - 1].transforms[id];
        let stable = scene.frames[n - 10..].iter().all(|f| {
            let m = &f.transforms[id];
            in_region(m, s) && (12..15).all(|i| (m[i] - end[i]).abs() < 0.002)
        });
        let released = contacts(&data, n - 1, id)?
            .values()
            .all(|p| p.range(bodies..).next().is_none());
        let touched_source = support(1, source)?;
        let touched_dest = support(n - 1, destination)?;
        let pass =
            lifted >= 0.04 && pinched && touched_source && stable && released && touched_dest;
        items.push(json!({"object":s.object,"destination":s.destination,"passed":pass,"lift_m":lifted,"lift_frame":peak,"two_robot_links_in_contact_while_lifted":pinched,"initial_source_contact":touched_source,"final_support_contact":touched_dest,"released":released,"settled_inside_for_last_10_frames":stable,"final_center_m":&end[12..15]}));
    }
    let travel: f64 = scene
        .frames
        .windows(2)
        .map(|w| {
            let a = w[0].base_pose.position;
            let b = w[1].base_pose.position;
            (a[0] - b[0]).hypot(a[1] - b[1])
        })
        .sum();
    let diag = read("single-body-diagnostics/judge-progress.json")?;
    let pass_count = items.iter().filter(|v| v["passed"] == true).count();
    let passed = pass_count == entries.len()
        && diag["furniture_contacts"] == 0
        && capture["terminal_acknowledged"] == true;
    let result = json!({"success":passed,"deliveries":pass_count,"required":entries.len(),"base_travel_m":travel,"frames":n,"cameras":capture["cameras"],"furniture_contact_samples":diag["furniture_contacts"],"terminal_acknowledged":capture["terminal_acknowledged"],"items":items});
    aisle_nano::runtime::write(&root.join("household-result.json"), &result)?;
    Ok(result)
}

/// Label the new replay with this household mission after the lossless recorder closes.
pub fn label_replay(root: &Path, result: &Value) -> Result<()> {
    let meta = root.join("dataset/meta");
    let title =
        "Nano household sorting · two garments to laundry baskets and two toys to coffee tables";
    for (name, key, value) in [
        ("tasks.jsonl", "task", json!(title)),
        ("episodes.jsonl", "tasks", json!([title])),
    ] {
        let path = meta.join(name);
        let lines = fs::read_to_string(&path)?;
        let mut output = String::new();
        for line in lines.lines() {
            let mut record: Value = serde_json::from_str(line)?;
            ensure!(record.get(key).is_some(), "missing {key} in {name}");
            record[key] = value.clone();
            output.push_str(&serde_json::to_string(&record)?);
            output.push('\n');
        }
        ensure!(!output.is_empty(), "empty episode metadata");
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, output)?;
        fs::rename(tmp, path)?;
    }
    let description =
        "Octos + DeepSeek Flash household sorting; verified outcomes in household-result.json";
    for name in ["capture.json", "capture-result.json"] {
        let path = meta.join(name);
        let mut v: Value = serde_json::from_slice(&fs::read(&path)?)?;
        v["recording_kind"] = json!(description);
        if name == "capture-result.json" {
            v["task_success"] = result["success"].clone();
        } else {
            v["session"] = json!("Nano household sorting");
        }
        aisle_nano::runtime::write(&path, &v)?;
    }
    aisle_nano::runtime::write(&meta.join("household-result.json"), result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HAR-1: a partial or duplicated roster must never yield a four-delivery verdict.
    #[test]
    fn audit_requires_all_four_distinct_deliveries() {
        let mut entries = aisle_nano::household::stations().unwrap();
        assert!(check_roster(&entries).is_ok());
        assert!(check_roster(&entries[1..]).is_err());
        entries[0] = entries[1].clone();
        assert!(check_roster(&entries).is_err());
    }

    /// SCN-1, HAR-1. A tipped bundle overhanging the edge cannot count as a basket delivery.
    #[test]
    fn destination_uses_full_oriented_footprint_and_support_height() {
        let mut station = aisle_nano::household::stations().unwrap().remove(0);
        station.region = [0., 0., 0.05, 0.05];
        let mut m = [0.; 16];
        m[0] = 1.;
        m[5] = 1.;
        m[10] = 1.;
        m[15] = 1.;
        m[14] = station.surface_z + station.size[2] / 2.;
        assert!(in_region(&m, &station));
        m[12] = 0.03;
        assert!(!in_region(&m, &station));
        m[12] = 0.;
        m[14] += 0.02;
        assert!(!in_region(&m, &station));
        m[0] = 0.;
        m[2] = 1.;
        m[8] = -1.;
        m[10] = 0.;
        m[14] = station.surface_z + station.size[0] / 2.;
        assert!(in_region(&m, &station));
    }
}
