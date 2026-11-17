use anyhow::{Result, ensure};
pub struct CachedCommand {
    pub id: u64,
    stamp: u64,
    request: Vec<f64>,
    response: Vec<f64>,
}
impl CachedCommand {
    pub fn new(id: u64, stamp: u64, request: Vec<f64>, response: Vec<f64>) -> Self {
        Self {
            id,
            stamp,
            request,
            response,
        }
    }
    pub fn replay(&self, id: u64, stamp: u64, request: &[f64]) -> Result<Option<Vec<f64>>> {
        ensure!(id >= self.id, "Old observation request");
        if id > self.id {
            return Ok(None);
        }
        ensure!(
            stamp == self.stamp && request == self.request,
            "Changed command under same observation ID"
        );
        Ok(Some(self.response.clone()))
    }
}
