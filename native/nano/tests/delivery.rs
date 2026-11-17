use aisle_nano::delivery::CachedCommand;
/// CON-4, CON-5, BG-1: retry delivery never applies motion twice or admits changed commands.
#[test]
fn duplicate_delivery_is_idempotent_but_not_mutable() {
    let c = CachedCommand::new(4, 800_000_000, vec![0.1; 17], vec![0.1; 17]);
    assert_eq!(
        c.replay(4, 800_000_000, &[0.1; 17]).unwrap(),
        Some(vec![0.1; 17])
    );
    assert!(c.replay(4, 800_000_000, &[0.2; 17]).is_err());
    assert!(c.replay(4, 0, &[0.1; 17]).is_err());
    assert!(c.replay(3, 600_000_000, &[0.1; 17]).is_err());
    assert_eq!(c.replay(5, 1_000_000_000, &[0.2; 17]).unwrap(), None);
}
