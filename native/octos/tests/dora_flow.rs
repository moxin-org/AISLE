#![cfg(feature = "test-fixtures")]
use axum::{Json, Router, routing::post};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{process::Stdio, time::Duration};

/// CON-3/4, TC-2/7: real Dora carries stamped subgoals while other timers run;
/// cancellation emits status only. The model here is an explicit HTTP stand-in.
#[tokio::test]
#[ignore = "requires AISLE_TEST_DORA pointing to the existing Dora 1.0.1 CLI"]
async fn actual_dora_roundtrip_and_cancel() {
    let dora = std::env::var_os("AISLE_TEST_DORA").expect("AISLE_TEST_DORA required");
    // Only the outer harness includes cold JIT startup. The node's 20-second
    // operational deadline and all inference/cancellation assertions stay fixed.
    let graph_seconds = std::env::var("AISLE_TEST_JIT_BOOT_SECONDS")
        .map(|v| v.parse::<u64>().expect("integer JIT startup budget"))
        .unwrap_or(30);
    assert!((30..=600).contains(&graph_seconds));
    for (cancel, replace) in [(false, false), (true, false), (false, true)] {
        let temp = tempfile::tempdir().unwrap();
        let started = temp.path().join("inference-started");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = requests.clone();
        let inference_started = started.clone();
        // Cancellation must act on an in-flight request. A wall-clock sleep
        // races cold JIT compilation; hold only the first response instead.
        let app=Router::new().route("/v1/chat/completions",post(move || {
                let counter=counter.clone();let started=inference_started.clone();
                async move {
                    let index=counter.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
                    std::fs::write(started,b"started").unwrap();
                    if index==0 && (cancel||replace) {std::future::pending::<()>().await;}
                    tokio::time::sleep(Duration::from_millis(700)).await;
                    Json(json!({"model":"deepseek-flash","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"kind\":\"pick\",\"object\":\"red cube\",\"arm\":\"right\"}"}}],"usage":{"prompt_tokens":10,"completion_tokens":10}}))
                }
            }));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut png = vec![];
        {
            let mut e = png::Encoder::new(&mut png, 1, 1);
            e.set_color(png::ColorType::Rgb);
            e.set_depth(png::BitDepth::Eight);
            e.write_header()
                .unwrap()
                .write_image_data(&[255, 0, 0])
                .unwrap();
        }
        let url = format!("data:image/png;base64,{}", STANDARD.encode(png));
        let mut content = vec![];
        for name in ["head", "left_wrist", "right_wrist"] {
            content.push(json!({"type":"text","text":format!("Camera: {name}")}));
            content.push(json!({"type":"image_url","image_url":{"url":url}}));
        }
        let request = json!({"model":"deepseek-flash","messages":[{"role":"system","content":"Inspect RGB and choose next subgoal"},{"role":"user","content":content}]});
        let input = temp.path().join("request.json");
        std::fs::write(&input, request.to_string()).unwrap();
        let output = temp.path().join("result.json");
        let graph = json!({"nodes":[
            {"id":"fixture","path":env!("CARGO_BIN_EXE_aisle-octos-fixture"),
             "env":{"FIXTURE_INPUT":input,"FIXTURE_OUTPUT":output,"FIXTURE_INFERENCE_STARTED":started,"FIXTURE_CANCEL":if cancel{"1"}else{"0"},"FIXTURE_REPLACE":if replace{"1"}else{"0"}},
             "inputs":{"tick":"dora/timer/millis/20","proposal":"octos/subgoal_msg","status":"octos/status_result"},
             "outputs":["observation_msg","cancel_msg"]},
            {"id":"octos","path":env!("CARGO_BIN_EXE_aisle-octos"),"args":format!("node --mode subgoal --upstream {upstream}"),
             "env":{"DEEPSEEK_API_KEY":"fixture-key"},
             "inputs":{"observation_msg":{"source":"fixture/observation_msg","queue_size":1},"cancel_msg":"fixture/cancel_msg"},
             "outputs":["subgoal_msg","status_result"]}
        ]});
        let graph_path = temp.path().join("dataflow.yaml");
        std::fs::write(&graph_path, graph.to_string()).unwrap();
        let log_path = temp.path().join("dora.log");
        let log = std::fs::File::create(&log_path).unwrap();
        let mut process = tokio::process::Command::new(&dora)
            .arg("run")
            .arg(&graph_path)
            .current_dir(temp.path())
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let status = tokio::time::timeout(Duration::from_secs(graph_seconds), process.wait()).await;
        if status.is_err() {
            let _ = process.kill().await;
        }
        server.abort();
        if let Some(dir) = std::env::var_os("AISLE_TEST_DORA_ARTIFACTS") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::copy(
                &log_path,
                dir.join(format!("dora-cancel-{cancel}-replace-{replace}.log")),
            )
            .unwrap();
            if output.exists() {
                std::fs::copy(
                    &output,
                    dir.join(format!("result-cancel-{cancel}-replace-{replace}.json")),
                )
                .unwrap();
            }
        }
        assert!(
            status.is_ok_and(|s| s.is_ok_and(|s| s.success())),
            "{}",
            std::fs::read_to_string(&log_path).unwrap()
        );
        let result: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(result["ok"], true);
        assert_eq!(result["cancelled"], cancel);
        assert_eq!(result["replaced"], replace);
        if replace {
            assert_eq!(result["superseded_results"], 1);
        }
        assert!(result["timer_ticks_during_inference"].as_u64().unwrap() > 3);
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            if replace { 2 } else { 1 }
        );
    }
}
