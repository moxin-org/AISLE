//! CON-3/5: prove actual Zenoh shared-memory unlinking happens at process exit.
use static_init::dynamic;
use std::{num::NonZeroUsize, path::PathBuf};
use zenoh_shm::posix_shm::segment::Segment;

// Ownership moves into the exit registry; the mapping pointer is never exposed
// after that move. Its fd and mapping may be dropped on the exiting thread.
struct Retained {
    _segment: Segment<u64>,
}
unsafe impl Send for Retained {}

#[dynamic(lazy, drop)]
static mut RETAINED: Vec<Retained> = Vec::new();

fn main() {
    let segment = Segment::<u64>::create(NonZeroUsize::new(4096).unwrap()).unwrap();
    let id = segment.id();
    let other = Segment::<u64>::open(id).unwrap();
    let mut checksum = 0_u64;
    for index in 0..4096 {
        let byte = (index % 251) as u8;
        unsafe {
            segment.as_ptr().add(index).write_volatile(byte);
        }
        let observed = unsafe { other.as_ptr().add(index).read_volatile() };
        assert_eq!(observed, byte);
        checksum += observed as u64;
    }
    let output = PathBuf::from(std::env::var_os("AISLE_JIT_SHM_REPORT").unwrap());
    let report = serde_json::json!({"ok":true,"segment_id":id,"bytes":4096,"checksum":checksum});
    std::fs::write(output, report.to_string()).unwrap();
    // Keep real mappings in a lazy static until exit, as Zenoh's global
    // metadata/provider owners do. No stack-local handle is dropped by main.
    // Deliberately forgetting handles would instead keep advisory locks alive,
    // which upstream correctly refuses to unlink even during cleanup.
    RETAINED
        .write()
        .extend([Retained { _segment: segment }, Retained { _segment: other }]);
    match std::env::args().nth(1).as_deref() {
        Some("exit") => std::process::exit(23),
        Some("bypass") => {
            // Negative control: OS-level _exit skips atexit. The harness owns
            // the reported segment and explicitly unlinks it after checking.
            unsafe extern "C" {
                fn _exit(code: i32) -> !;
            }
            unsafe {
                _exit(0);
            }
        }
        _ => {}
    }
}
