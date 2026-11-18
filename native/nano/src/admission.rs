//! TC-7: inference admission shared by native Dora and JIT development.
#[derive(Clone)]
pub(crate) struct Pending {
    pub(crate) goal: String,
    pub(crate) stamp: u64,
    pub(crate) captured: u64,
}
/// TC-7: a retired inference can never revive a cancelled skill.
#[derive(Default)]
pub struct Admission {
    pub generation: u64,
    pub(crate) pending: Option<Pending>,
}
impl Admission {
    pub fn request(&mut self, mission: &str, stamp: u64, captured: u64) -> String {
        self.generation += 1;
        let goal = format!("{mission}-g{}", self.generation);
        self.pending = Some(Pending {
            goal: goal.clone(),
            stamp,
            captured,
        });
        goal
    }
    pub fn accepts(&self, id: &str, stamp: u64, at: u64) -> bool {
        self.pending.as_ref().is_some_and(|p| {
            p.goal == id && p.stamp == stamp && at >= p.captured && at - p.captured <= 30000
        })
    }
    pub fn cancel(&mut self) -> Option<String> {
        self.pending.take().map(|p| p.goal)
    }
}
