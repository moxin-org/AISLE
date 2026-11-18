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

/// TC-7: a planner reply is untrusted text. Anything other than bounded JSON is a
/// refused reply for the supervisor to count, never an error that stops control.
pub fn reply(text: &str) -> Result<serde_json::Value, String> {
    if text.len() > 65536 {
        return Err("planner response too large".into());
    }
    serde_json::from_str(text).map_err(|e| format!("planner response is not JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// TC-7: malformed or oversized planner text is refused, not fatal.
    #[test]
    fn planner_reply_is_bounded_json_or_refused() {
        assert_eq!(reply(r#"{"a":1}"#).unwrap()["a"], 1);
        assert!(reply("not json").unwrap_err().contains("not JSON"));
        assert!(reply("").is_err());
        let large = format!("\"{}\"", "x".repeat(65536));
        assert!(reply(&large).unwrap_err().contains("too large"));
    }
    /// TC-7: only the pending request, at its own stamp and within its deadline, is admitted.
    #[test]
    fn retired_or_late_replies_are_not_admitted() {
        let mut a = Admission::default();
        let first = a.request("m", 10, 1000);
        assert!(a.accepts(&first, 10, 2000));
        assert!(!a.accepts(&first, 11, 2000));
        assert!(!a.accepts(&first, 10, 31001));
        assert!(!a.accepts(&first, 10, 999));
        assert_eq!(a.cancel(), Some(first.clone()));
        assert!(!a.accepts(&first, 10, 2000));
        let second = a.request("m", 20, 3000);
        assert_ne!(first, second);
        assert!(!a.accepts(&first, 20, 3000));
        assert!(a.accepts(&second, 20, 3000));
    }
}
