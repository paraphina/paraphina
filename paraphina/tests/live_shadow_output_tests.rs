#![cfg(feature = "live_hyperliquid")]

use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

#[test]
fn shadow_output_dir_contains_required_artifacts() {
    let bin = env!("CARGO_BIN_EXE_paraphina_live");
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("shadow_run");

    let status = Command::new(bin)
        .env("PARAPHINA_TRADE_MODE", "shadow")
        .env("PARAPHINA_LIVE_CONNECTOR", "hyperliquid_fixture")
        .env("HL_FIXTURE_DIR", "../tests/fixtures/hyperliquid")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_MAX_TICKS", "5")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:9997")
        .status()
        .expect("run paraphina_live");
    assert!(status.success());

    let telemetry = out_dir.join("telemetry.jsonl");
    let config = out_dir.join("config_resolved.json");
    let build = out_dir.join("build_info.json");
    let specs = out_dir.join("instrument_specs.json");
    let summary = out_dir.join("summary.json");

    assert!(telemetry.exists(), "missing telemetry.jsonl");
    assert!(config.exists(), "missing config_resolved.json");
    assert!(build.exists(), "missing build_info.json");
    assert!(specs.exists(), "missing instrument_specs.json");
    assert!(summary.exists(), "missing summary.json");

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let check = Command::new("python3")
        .current_dir(&repo_root)
        .arg("tools/check_telemetry_contract.py")
        .arg(&telemetry)
        .status()
        .expect("telemetry contract check");
    assert!(check.success());
}
