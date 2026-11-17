use std::process::Command;
fn main() {
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=AISLE_SOURCE_REVISION={revision}");
    // Executable hashes identify the exact working implementation, including
    // uncommitted source. A Git revision alone is not an attestation.
}
