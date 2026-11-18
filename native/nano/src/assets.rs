//! The model a rollout executes: the URDF plus every mesh it references.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

pub fn sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("Reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
/// Referenced mesh paths relative to the model directory. Every one must be an
/// existing file inside that directory.
pub fn meshes(model: &Path) -> Result<BTreeSet<String>> {
    let robot = urdf_rs::read_file(model)?;
    let root = model.parent().context("Model directory")?;
    let mut files = BTreeSet::new();
    for link in &robot.links {
        for geometry in link
            .visual
            .iter()
            .map(|v| &v.geometry)
            .chain(link.collision.iter().map(|c| &c.geometry))
        {
            if let urdf_rs::Geometry::Mesh { filename, .. } = geometry {
                let name = filename
                    .strip_prefix("package://adora_nano/")
                    .unwrap_or(filename);
                ensure!(
                    Path::new(name)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
                    "Model mesh outside the model directory: {name}"
                );
                ensure!(name != "model.urdf", "Model mesh named like the URDF copy");
                ensure!(root.join(name).is_file(), "Model mesh missing: {name}");
                files.insert(name.to_owned());
            }
        }
    }
    Ok(files)
}
/// Copies the URDF and its meshes into a new directory and returns the copied
/// URDF, so later edits to the source cannot change a running experiment.
pub fn snapshot(model: &Path, into: &Path) -> Result<PathBuf> {
    let root = model.parent().context("Model directory")?;
    fs::create_dir(into)?;
    for name in meshes(model)? {
        let to = into.join(&name);
        fs::create_dir_all(to.parent().context("Mesh directory")?)?;
        fs::copy(root.join(&name), to)?;
    }
    let urdf = into.join("model.urdf");
    fs::copy(model, &urdf)?;
    Ok(urdf)
}
/// SHA-256 of the URDF (as `model.urdf`) and each mesh, keyed by relative path.
pub fn manifest(model: &Path) -> Result<BTreeMap<String, String>> {
    let root = model.parent().context("Model directory")?;
    let mut files = BTreeMap::from([("model.urdf".to_owned(), sha256(model)?)]);
    for name in meshes(model)? {
        let hash = sha256(&root.join(&name))?;
        files.insert(name, hash);
    }
    Ok(files)
}
