#![cfg(all(
    feature = "live_hyperliquid",
    feature = "live_lighter",
    feature = "live_paradex",
    feature = "live_aster",
    feature = "live_extended"
))]

use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

#[test]
fn five_connector_fixture_run_is_healthy_and_deterministic() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("five_connector_run");
    let telemetry_path = out_dir.join("telemetry.jsonl");
    let summary_path = out_dir.join("summary.json");
    let replay_dir = tmp.path().join("five_connector_replay");
    let replay_telemetry = replay_dir.join("telemetry.jsonl");
    let replay_summary = replay_dir.join("summary.json");

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
    let roadmap_b_fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("roadmap_b");

    run_all5_paper_fixture(
        &out_dir,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
    );
    run_all5_paper_fixture(
        &replay_dir,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
    );

    let stats = assert_all5_telemetry(&telemetry_path);
    let replay_stats = assert_all5_telemetry(&replay_telemetry);
    assert!(stats.fills_total > 0, "expected paper fills in telemetry");
    assert_eq!(stats.reconcile_drift, 0, "expected no reconcile drift");
    assert_eq!(stats.ticks_total, replay_stats.ticks_total);
    assert!(
        replay_stats.fills_total > 0,
        "expected paper fills in replay"
    );
    assert_eq!(stats.reconcile_drift, replay_stats.reconcile_drift);

    let _summary = read_summary_json(&summary_path);
    let _replay_summary = read_summary_json(&replay_summary);
}

fn run_all5_paper_fixture(
    out_dir: &PathBuf,
    hl_fixture_dir: &PathBuf,
    lighter_fixture_dir: &PathBuf,
    roadmap_b_fixture_dir: &PathBuf,
) {
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .env("PARAPHINA_TRADE_MODE", "paper")
        .env(
            "PARAPHINA_LIVE_CONNECTORS",
            "hyperliquid_fixture,lighter,extended,aster,paradex",
        )
        .env("EXTENDED_FIXTURE_MODE", "1")
        .env("ASTER_FIXTURE_MODE", "1")
        .env("PARADEX_FIXTURE_MODE", "1")
        .env("HL_FIXTURE_DIR", hl_fixture_dir)
        .env("LIGHTER_FIXTURE_DIR", lighter_fixture_dir)
        .env("ROADMAP_B_FIXTURE_DIR", roadmap_b_fixture_dir)
        .env("PARAPHINA_LIVE_OUT_DIR", out_dir)
        .env("PARAPHINA_LIVE_MAX_TICKS", "3")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_PAPER_FILL_MODE", "always")
        .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
        .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "5")
        .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1")
        .env("PARAPHINA_PAPER_DISABLE_HEALTH_GATES", "1")
        .env("PARAPHINA_PAPER_DISABLE_FV_GATE", "1")
        .env("PARAPHINA_PAPER_SMOKE_INTENTS", "1")
        .env("PARAPHINA_RECONCILE_POS_TAO_TOL", "1000000")
        .env("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "1000000")
        .env("PARAPHINA_RECONCILE_ORDER_COUNT_TOL", "1000000")
        .output()
        .expect("run paraphina_live");

    assert!(
        output.status.success(),
        "five-connector run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct TelemetryStats {
    fills_total: usize,
    reconcile_drift: usize,
    ticks_total: u64,
}

fn assert_all5_telemetry(telemetry_path: &PathBuf) -> TelemetryStats {
    let data = std::fs::read_to_string(telemetry_path).expect("read telemetry");
    let mut saw_ready = false;
    let mut saw_fv_true = false;
    let mut expected_tick: u64 = 0;
    let mut fills_total = 0;
    let mut reconcile_drift = 0;
    let mut ticks_total: u64 = 0;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).expect("parse telemetry JSON");
        let tick = value.get("t").and_then(|v| v.as_u64()).expect("tick");
        assert_eq!(tick, expected_tick, "expected deterministic tick order");
        expected_tick += 1;
        ticks_total = tick;

        let venue_mid_len = value
            .get("venue_mid_usd")
            .and_then(|v| v.as_array())
            .map(|v| v.len())
            .unwrap_or(0);
        assert_eq!(venue_mid_len, 5, "expected five venues in telemetry");

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
        if healthy_count == 5 {
            saw_ready = true;
        }

        let fv_available = value
            .get("fv_available")
            .and_then(|v| v.as_bool())
            .expect("fv_available");
        if fv_available {
            saw_fv_true = true;
            assert_eq!(healthy_count, 5, "fv_available before readiness");
        }

        if let Some(fills) = value.get("fills").and_then(|v| v.as_array()) {
            fills_total += fills.len();
        }

        if let Some(drift) = value.get("reconcile_drift").and_then(|v| v.as_array()) {
            reconcile_drift += drift.len();
        }
    }

    assert!(saw_ready, "expected readiness once all venues are healthy");
    assert!(saw_fv_true, "expected fv_available to become true");

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let check = Command::new("python3")
        .arg("tools/check_telemetry_contract.py")
        .arg(telemetry_path)
        .current_dir(&workspace_root)
        .output()
        .expect("failed to run check_telemetry_contract.py");
    assert!(
        check.status.success(),
        "telemetry contract check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    TelemetryStats {
        fills_total,
        reconcile_drift,
        ticks_total: ticks_total + 1,
    }
}

fn read_summary_json(summary_path: &PathBuf) -> serde_json::Value {
    let data = std::fs::read_to_string(summary_path).expect("read summary");
    serde_json::from_str(&data).expect("parse summary json")
}
