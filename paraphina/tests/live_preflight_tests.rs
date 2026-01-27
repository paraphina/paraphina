#![cfg(feature = "live")]

use std::process::Command;

use tempfile::TempDir;

#[test]
fn live_mode_refuses_to_start_without_guardrails() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    let output = cmd
        .arg("--trade-mode")
        .arg("live")
        .arg("--connector")
        .arg("mock")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .output()
        .expect("run paraphina_live");
    assert!(
        !output.status.success(),
        "expected non-zero exit when live guardrails are missing"
    );
}

#[test]
fn shadow_mode_writes_out_dir_artifacts_without_keys() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("shadow_run");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    let status = cmd
        .arg("--trade-mode")
        .arg("shadow")
        .arg("--connector")
        .arg("mock")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_LIVE_MAX_TICKS", "1")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .status()
        .expect("run paraphina_live");
    assert!(status.success(), "shadow mode should exit cleanly");

    assert!(out_dir.join("config_resolved.json").exists());
    assert!(out_dir.join("build_info.json").exists());
    assert!(out_dir.join("telemetry.jsonl").exists());
    assert!(out_dir.join("summary.json").exists());
}
