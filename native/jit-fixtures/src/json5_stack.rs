//! CON-3/5: the real JSON5 derive and stack guard used by Zenoh configuration.
fn main() {
    let value: serde_json::Value =
        json5::from_str("{ // config\n mode: 'peer', values: [1, 2,], }").unwrap();
    assert_eq!(value["mode"], "peer");
    assert_eq!(value["values"][1], 2);
    assert!(json5::from_str::<serde_json::Value>("{broken: [").is_err());
    let remaining = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            // Relax may reserve a larger thread stack for lazy compilation.
            // Compare the probe with real OS bounds, not the requested minimum.
            unsafe extern "C" {
                fn pthread_self() -> *mut std::ffi::c_void;
                fn pthread_get_stackaddr_np(thread: *mut std::ffi::c_void)
                -> *mut std::ffi::c_void;
                fn pthread_get_stacksize_np(thread: *mut std::ffi::c_void) -> usize;
            }
            let (top, size) = unsafe {
                (
                    pthread_get_stackaddr_np(pthread_self()) as usize,
                    pthread_get_stacksize_np(pthread_self()),
                )
            };
            let remaining = stacker::remaining_stack().expect("stack probing must remain active");
            let marker = std::process::id() as u8;
            let pointer = std::hint::black_box(&marker) as *const u8 as usize;
            assert!((top - size..top).contains(&pointer));
            assert!(
                (64 * 1024..size).contains(&remaining),
                "remaining={remaining}, size={size}"
            );
            assert!(remaining.abs_diff(pointer - (top - size)) < 64 * 1024);
            remaining
        })
        .unwrap()
        .join()
        .unwrap();
    println!(
        "{}",
        serde_json::json!({"ok":true,"remaining_stack_bytes":remaining,"malformed_rejected":true})
    );
}
