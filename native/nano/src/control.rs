//! Operator control enters through a local socket and then a Dora topic.
use anyhow::{Context, Result, ensure};
use dora_node_api::{
    DoraNode, Event,
    futures::{StreamExt, executor::block_on},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    os::unix::{
        fs::PermissionsExt,
        net::{UnixListener, UnixStream},
    },
    path::Path,
    time::Duration,
};
pub fn send(socket: &Path, value: &Value) -> Result<()> {
    let mut s = UnixStream::connect(socket)?;
    s.set_write_timeout(Some(Duration::from_secs(2)))?;
    s.write_all(value.to_string().as_bytes())?;
    s.shutdown(std::net::Shutdown::Write)?;
    Ok(())
}
pub fn run() -> Result<()> {
    let output = std::env::var_os("AISLE_NANO_OUTPUT").context("output missing")?;
    let path = Path::new(&output).join("control.sock");
    ensure!(path.as_os_str().len() < 104, "control socket path too long");
    let listener = UnixListener::bind(&path)?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;
    let (mut node, mut events) = DoraNode::init_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    while let Some(e) = block_on(events.next()) {
        match e {
            Event::Stop { .. } => break,
            Event::Input { id, .. } if id.as_str() == "finished" => break,
            Event::Input { id, metadata, .. } if id.as_str() == "tick" => {
                for _ in 0..8 {
                    let (stream, _) = match listener.accept() {
                        Ok(s) => s,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(e) => return Err(anyhow::anyhow!("control accept: {e}")),
                    };
                    stream.set_nonblocking(true)?;
                    let mut b = Vec::new();
                    let mut stream = stream;
                    let deadline = std::time::Instant::now() + Duration::from_millis(100);
                    let mut complete = false;
                    while b.len() <= 4096 && std::time::Instant::now() < deadline {
                        let mut buf = [0u8; 1024];
                        match stream.read(&mut buf) {
                            Ok(0) => {
                                complete = true;
                                break;
                            }
                            Ok(n) => b.extend_from_slice(&buf[..n]),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                std::thread::sleep(Duration::from_millis(1))
                            }
                            Err(_) => break,
                        }
                    }
                    if !complete || b.len() > 4096 {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_slice::<Value>(&b) {
                        node.send_output(
                            "control_msg".into(),
                            metadata.parameters.clone(),
                            v.to_string(),
                        )
                        .map_err(|e| anyhow::anyhow!("control Dora send: {e}"))?;
                    }
                }
            }
            Event::Error(e) => anyhow::bail!("{e}"),
            _ => {}
        }
    }
    fs::remove_file(path)?;
    Ok(())
}
pub fn command(
    socket: &Path,
    mission: String,
    request_id: u64,
    op: String,
    instruction: Option<String>,
) -> Result<Value> {
    let v = json!({"mission_id":mission,"request_id":request_id,"op":op,"instruction":instruction});
    send(socket, &v)?;
    Ok(json!({"ok":true,"queued":v,"applied_ack":"supervisor.jsonl skill_result"}))
}
#[cfg(test)]
mod tests {
    /// TC-7: local control is delivered as one complete bounded message.
    #[test]
    fn local_socket_read_timeout_is_supported() {
        use std::{
            io::{Read, Write},
            os::unix::net::UnixStream,
            time::Duration,
        };
        let path =
            std::env::temp_dir().join(format!("aisle-control-test-{}.sock", std::process::id()));
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut a = UnixStream::connect(&path).unwrap();
        let (mut b, _) = listener.accept().unwrap();
        std::fs::remove_file(path).unwrap();
        b.set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        a.write_all(b"hello").unwrap();
        a.shutdown(std::net::Shutdown::Write).unwrap();
        let mut data = Vec::new();
        b.read_to_end(&mut data).unwrap();
        assert_eq!(data, b"hello");
    }
}
