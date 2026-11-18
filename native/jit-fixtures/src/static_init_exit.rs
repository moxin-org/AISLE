//! CON-3/5: exercise the lazy-drop mechanism Zenoh uses, including process exit.
use static_init::dynamic;
use std::{fs::OpenOptions, io::Write};

fn record(message: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::var_os("AISLE_JIT_EXIT_LOG").expect("isolated exit log"))
        .unwrap();
    writeln!(file, "{message}").unwrap();
}

struct Resource(&'static str);
impl Drop for Resource {
    fn drop(&mut self) {
        record(self.0);
        if self.0 == "second dropped" && std::env::args().nth(1).as_deref() == Some("nested") {
            assert_eq!(LATE.read().0, "nested dropped");
        }
    }
}

#[dynamic(lazy, drop)]
static mut FIRST: Resource = Resource("first dropped");
#[dynamic(lazy, drop)]
static mut SECOND: Resource = Resource("second dropped");
#[dynamic(lazy, drop)]
static mut UNUSED: Resource = Resource("unused dropped");
#[dynamic(lazy, drop)]
static mut LATE: Resource = Resource("nested dropped");

fn touch() {
    assert_eq!(FIRST.read().0, "first dropped");
    assert_eq!(SECOND.read().0, "second dropped");
}

fn reload_marker() -> &'static str {
    "before reload"
}

fn main() {
    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }
    extern "C" fn early() {
        record("early callback");
    }
    extern "C" fn late() {
        record("late callback");
    }
    assert_eq!(unsafe { atexit(early) }, 0);
    if std::env::args().nth(1).as_deref() == Some("concurrent") {
        let workers: Vec<_> = (0..16).map(|_| std::thread::spawn(touch)).collect();
        for worker in workers {
            worker.join().unwrap();
        }
    } else {
        touch();
    }
    assert_eq!(unsafe { atexit(late) }, 0);
    // A second access must not register a second destructor.
    assert_eq!(FIRST.read().0, "first dropped");
    if std::env::args().nth(1).as_deref() == Some("wait") {
        let release = std::env::var_os("AISLE_JIT_RELEASE_FILE").expect("isolated release file");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let mut previous = String::new();
        while !std::path::Path::new(&release).exists() {
            assert!(std::time::Instant::now() < deadline, "release deadline");
            let current = reload_marker();
            if previous != current {
                record(current);
                previous = current.to_owned();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    record("main finished");
    match std::env::args().nth(1).as_deref() {
        Some("exit") => std::process::exit(23),
        Some("panic") => panic!("intentional exit fixture panic"),
        _ => {}
    }
}
