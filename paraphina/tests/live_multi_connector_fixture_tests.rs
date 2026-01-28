#![cfg(all(feature = "live_hyperliquid", feature = "live_lighter"))]

use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

#[test]
fn multi_connector_fixture_run_is_deterministic_and_healthy() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("multi_run");
    let telemetry_path = out_dir.join("telemetry.jsonl");

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let hl_fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("hyperliquid");
    let lighter_fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("lighter");

    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .env_remove("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS")
        .env_remove("PARAPHINA_LIVE_RECONCILE_MS")
        .env_remove("PARAPHINA_LIVE_KILL_FLATTEN")
        .env_remove("PARAPHINA_LIVE_KILL_SWITCH")
        .env_remove("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
        .env("PARAPHINA_TRADE_MODE", "paper")
        .env("PARAPHINA_LIVE_CONNECTORS", "hyperliquid_fixture,lighter")
        .env("HL_FIXTURE_DIR", &hl_fixture_dir)
        .env("LIGHTER_FIXTURE_DIR", &lighter_fixture_dir)
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_LIVE_MAX_TICKS", "20")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_PAPER_FILL_MODE", "mid")
        .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
        .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "2")
        .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1")
        .output()
        .expect("run paraphina_live");

    assert!(
        output.status.success(),
        "multi-connector run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let data = std::fs::read_to_string(&telemetry_path).expect("read telemetry");
    let mut saw_ready = false;
    let mut saw_fv_true = false;
    let mut expected_tick: u64 = 0;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).expect("parse telemetry JSON");
        let tick = value.get("t").and_then(|v| v.as_u64()).expect("tick");
        assert_eq!(tick, expected_tick, "expected deterministic tick order");
        expected_tick += 1;

        let venue_mid_len = value
            .get("venue_mid_usd")
            .and_then(|v| v.as_array())
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(venue_mid_len, 2, "expected two venues in telemetry");

        let venue_status = value
            .get("venue_status")
            .and_then(|v| v.as_array())
            .expect("venue_status");
        for status in venue_status {
            assert_ne!(status.as_str(), Some("Stale"), "unexpected stale venue");
        }

        let healthy_count = value
            .get("healthy_venues_used_count")
            .and_then(|v| v.as_u64())
            .expect("healthy_venues_used_count");
        if healthy_count == 2 {
            saw_ready = true;
        } else {
            assert!(!saw_ready, "healthy venue count dropped after readiness");
        }

        let fv_available = value
            .get("fv_available")
            .and_then(|v| v.as_bool())
            .expect("fv_available");
        if fv_available {
            saw_fv_true = true;
            assert_eq!(healthy_count, 2, "fv_available before readiness");
        }
    }

    assert!(saw_ready, "expected readiness once both venues are healthy");
    assert!(saw_fv_true, "expected fv_available to become true");
}
