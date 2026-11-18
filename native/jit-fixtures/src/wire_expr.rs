//! CON-3/5: JIT topic-name concatenation preserves scope, mapping and Cow state.
use std::borrow::Cow;
use zenoh_protocol::{core::WireExpr, network::Mapping};

#[test]
fn suffix_matches_upstream_string_and_ownership_semantics() {
    for (prefix, suffix, expected, owned) in [
        (Cow::Borrowed(""), "next", "next", false),
        (Cow::Borrowed(""), "", "", false),
        (Cow::Borrowed("topic"), "", "topic", false),
        (Cow::Borrowed("topic"), "/相机", "topic/相机", true),
        (Cow::Owned("topic".into()), "/cancel", "topic/cancel", true),
        (Cow::Owned("topic".into()), "", "topic", true),
        (Cow::Owned(String::new()), "next", "next", false),
    ] {
        let expression = WireExpr {
            scope: 42,
            suffix: prefix,
            mapping: Mapping::Receiver,
        }
        .with_suffix(suffix);
        assert_eq!(expression.suffix, expected);
        assert_eq!(matches!(expression.suffix, Cow::Owned(_)), owned);
        assert_eq!(expression.scope, 42);
        assert_eq!(expression.mapping, Mapping::Receiver);
    }
}
