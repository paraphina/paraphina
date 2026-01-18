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

fn run_all5_fixture(
    trade_mode: &str,
    out_dir: &PathBuf,
    hl_fixture_dir: &PathBuf,
    lighter_fixture_dir: &PathBuf,
    roadmap_b_fixture_dir: &PathBuf,
    paper_mode: bool,
) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    cmd.env("PARAPHINA_TRADE_MODE", trade_mode)
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
        .env("PARAPHINA_LIVE_MAX_TICKS", "20")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl");

    if paper_mode {
        cmd.env("PARAPHINA_PAPER_FILL_MODE", "mid")
            .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
            .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "5")
            .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1");
    }

    let output = cmd.output().expect("run paraphina_live");
    assert!(
        output.status.success(),
        "all-5 run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_all5_telemetry(telemetry_path: &PathBuf, require_readiness: bool) {
    let data = std::fs::read_to_string(telemetry_path).expect("read telemetry");
    let mut saw_ready = false;
    let mut saw_fv_true = false;
    let mut expected_tick: u64 = 0;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).expect("parse telemetry JSON");
        let tick = value
            .get("t")
            .and_then(|v| v.as_u64())
            .expect("tick");
        assert_eq!(tick, expected_tick, "expected deterministic tick order");
        expected_tick += 1;

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
        } else {
            assert!(
                !saw_ready,
                "healthy venue count dropped after readiness"
            );
        }

        let fv_available = value
            .get("fv_available")
            .and_then(|v| v.as_bool())
            .expect("fv_available");
        if fv_available {
            saw_fv_true = true;
            assert_eq!(healthy_count, 5, "fv_available before readiness");
        }
    }

    if require_readiness {
        assert!(saw_ready, "expected readiness once all venues are healthy");
        assert!(saw_fv_true, "expected fv_available to become true");
    }
}

#[test]
fn all5_shadow_and_paper_fixture_runs_are_healthy() {
    let tmp = TempDir::new().expect("temp dir");
    let shadow_out = tmp.path().join("all5_shadow");
    let paper_out = tmp.path().join("all5_paper");
    let shadow_telemetry = shadow_out.join("telemetry.jsonl");
    let paper_telemetry = paper_out.join("telemetry.jsonl");

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let hl_fixture_dir = workspace_root.join("tests").join("fixtures").join("hyperliquid");
    let lighter_fixture_dir = workspace_root.join("tests").join("fixtures").join("lighter");
    let roadmap_b_fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("roadmap_b");

    run_all5_fixture(
        "shadow",
        &shadow_out,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
        false,
    );
    run_all5_fixture(
        "paper",
        &paper_out,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
        true,
    );

    assert_all5_telemetry(&shadow_telemetry, false);
    assert_all5_telemetry(&paper_telemetry, false);
}
