#![cfg(all(
    feature = "live_hyperliquid",
    feature = "live_lighter",
    feature = "live_paradex",
    feature = "live_aster",
    feature = "live_extended"
))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use serde_json::Value;
use tempfile::TempDir;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

struct TelemetryAudit {
    ticks_total: u64,
    fills_total: usize,
    mm_fills_total: usize,
    hedge_records_total: usize,
    hedge_records_with_source: usize,
    hedge_fills_total: usize,
    hedge_fills_with_source: usize,
    reconcile_drift_total: usize,
    max_dispersion_usd: f64,
    final_pnl_total: f64,
    final_pnl_realised: f64,
    final_pnl_unrealised: f64,
    kill_switch_seen: bool,
    kill_reason: Option<String>,
    mm_keep_total: u64,
    mm_replace_total: u64,
    mm_place_total: u64,
}

fn deterministic_probe_out_dir(name: &str) -> PathBuf {
    let out_dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("create deterministic probe out dir");
    out_dir
}

fn pnl_valid_fixture_paths(workspace_root: &PathBuf) -> (PathBuf, PathBuf, PathBuf) {
    let root = workspace_root
        .join("tests")
        .join("fixtures")
        .join("pnl_valid_all5");
    (
        root.join("hyperliquid"),
        root.join("lighter"),
        root.join("roadmap_b"),
    )
}

fn run_all5_paper_fixture(
    out_dir: &PathBuf,
    hl_fixture_dir: &PathBuf,
    lighter_fixture_dir: &PathBuf,
    roadmap_b_fixture_dir: &PathBuf,
    fill_mode: &str,
    max_ticks: u64,
    extra_env: &[(&str, &str)],
) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let mut command = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    command
        .env_remove("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS")
        .env_remove("PARAPHINA_LIVE_RECONCILE_MS")
        .env_remove("PARAPHINA_LIVE_KILL_FLATTEN")
        .env_remove("PARAPHINA_LIVE_KILL_SWITCH")
        .env_remove("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
        .env_remove("ROADMAP_B_FIXTURE_DIR")
        .env_remove("HL_FIXTURE_DIR")
        .env_remove("LIGHTER_FIXTURE_DIR")
        .env_remove("EXTENDED_FIXTURE_DIR")
        .env_remove("ASTER_FIXTURE_DIR")
        .env_remove("PARADEX_FIXTURE_DIR")
        .env_remove("EXTENDED_FIXTURE_MODE")
        .env_remove("ASTER_FIXTURE_MODE")
        .env_remove("PARADEX_FIXTURE_MODE")
        .env_remove("EXTENDED_RECORD_FIXTURES")
        .env_remove("ASTER_RECORD_FIXTURES")
        .env_remove("PARADEX_RECORD_FIXTURES")
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
        .env("PARAPHINA_LIVE_MAX_TICKS", format!("{max_ticks}"))
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_PAPER_FILL_MODE", fill_mode)
        .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
        .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "5")
        // The coherent all-5 fixture uses wall-clock timestamps. Keep stale
        // thresholds above CI/test-host scheduling jitter so fixture health is
        // driven by data coherence, not local load.
        .env("PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE", "60000")
        .env("PARAPHINA_HL_STATE_STALE_MS_OVERRIDE", "60000")
        .env("PARAPHINA_ASTER_STATE_STALE_MS_OVERRIDE", "60000")
        .env("PARAPHINA_LIGHTER_STATE_STALE_MS_OVERRIDE", "60000")
        .env("PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE", "60000")
        .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("run paraphina_live");

    let run_log = out_dir.join("run.log");
    let mut combined = output.stdout.clone();
    combined.extend_from_slice(&output.stderr);
    std::fs::write(&run_log, combined).expect("write run.log");
    assert!(
        output.status.success(),
        "all-5 paper run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn audit_paper_telemetry(workspace_root: &PathBuf, telemetry_path: &PathBuf) -> TelemetryAudit {
    let check = Command::new("python3")
        .arg("tools/check_telemetry_contract.py")
        .arg(telemetry_path)
        .current_dir(workspace_root)
        .output()
        .expect("run telemetry contract check");
    assert!(
        check.status.success(),
        "telemetry contract check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let data = std::fs::read_to_string(telemetry_path).expect("read telemetry");
    let mut expected_tick = 0_u64;
    let mut fills_total = 0_usize;
    let mut mm_fills_total = 0_usize;
    let mut hedge_records_total = 0_usize;
    let mut hedge_records_with_source = 0_usize;
    let mut hedge_fills_total = 0_usize;
    let mut hedge_fills_with_source = 0_usize;
    let mut reconcile_drift_total = 0_usize;
    let mut max_dispersion_usd = 0.0_f64;
    let mut final_pnl_total = 0.0_f64;
    let mut final_pnl_realised = 0.0_f64;
    let mut final_pnl_unrealised = 0.0_f64;
    let mut kill_switch_seen = false;
    let mut kill_reason = None;
    let mut mm_keep_total = 0_u64;
    let mut mm_replace_total = 0_u64;
    let mut mm_place_total = 0_u64;
    let mut saw_ready = false;
    let mut saw_fv_true = false;

    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).expect("parse telemetry JSON");
        assert_eq!(
            value.get("execution_mode").and_then(|v| v.as_str()),
            Some("paper"),
            "expected execution_mode=paper"
        );

        let tick = value.get("t").and_then(|v| v.as_u64()).expect("tick");
        assert_eq!(tick, expected_tick, "expected deterministic tick order");
        expected_tick += 1;

        let venue_mids = value
            .get("venue_mid_usd")
            .and_then(|v| v.as_array())
            .expect("venue_mid_usd");
        assert_eq!(venue_mids.len(), 5, "expected five venues in telemetry");
        let mut mids = Vec::new();
        for mid in venue_mids {
            if let Some(mid) = mid.as_f64() {
                mids.push(mid);
            }
        }
        if mids.len() >= 2 {
            let dispersion = mids.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - mids.iter().copied().fold(f64::INFINITY, f64::min);
            max_dispersion_usd = max_dispersion_usd.max(dispersion);
        }

        let venue_status = value
            .get("venue_status")
            .and_then(|v| v.as_array())
            .expect("venue_status");
        for status in venue_status {
            assert_ne!(status.as_str(), Some("Stale"), "unexpected stale venue");
            assert_ne!(
                status.as_str(),
                Some("Disabled"),
                "unexpected disabled venue"
            );
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
        }

        fills_total += value
            .get("fills")
            .and_then(|v| v.as_array())
            .map(|fills| fills.len())
            .unwrap_or(0);
        if let Some(hedges) = value.get("hedges").and_then(|v| v.as_array()) {
            for hedge in hedges {
                hedge_records_total += 1;
                if hedge
                    .get("source_decision_id")
                    .and_then(|v| v.as_str())
                    .is_some_and(|decision_id| !decision_id.is_empty())
                {
                    hedge_records_with_source += 1;
                }
            }
        }
        if let Some(fills) = value.get("fills").and_then(|v| v.as_array()) {
            for fill in fills {
                match fill.get("purpose").and_then(|v| v.as_str()) {
                    Some("Mm") => mm_fills_total += 1,
                    Some("Hedge") => {
                        hedge_fills_total += 1;
                        if fill
                            .get("source_decision_id")
                            .and_then(|v| v.as_str())
                            .is_some_and(|decision_id| !decision_id.is_empty())
                        {
                            hedge_fills_with_source += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        reconcile_drift_total += value
            .get("reconcile_drift")
            .and_then(|v| v.as_array())
            .map(|drift| drift.len())
            .unwrap_or(0);

        final_pnl_total = value
            .get("pnl_total")
            .and_then(|v| v.as_f64())
            .expect("pnl_total");
        final_pnl_realised = value
            .get("pnl_realised")
            .and_then(|v| v.as_f64())
            .expect("pnl_realised");
        final_pnl_unrealised = value
            .get("pnl_unrealised")
            .and_then(|v| v.as_f64())
            .expect("pnl_unrealised");

        if value
            .get("kill_switch")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            kill_switch_seen = true;
            kill_reason = value
                .get("kill_reason")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned);
        }

        if let Some(summary) = value.get("mm_order_management").and_then(|v| v.as_object()) {
            mm_keep_total += summary
                .get("keep_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            mm_replace_total += summary
                .get("replace_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            mm_place_total += summary
                .get("place_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        } else {
            panic!("expected mm_order_management summary in paper telemetry");
        }
    }

    assert!(expected_tick > 0, "expected at least one telemetry row");
    assert!(saw_ready, "expected readiness once all venues are healthy");
    assert!(
        saw_fv_true,
        "expected fv_available once all venues are healthy"
    );

    TelemetryAudit {
        ticks_total: expected_tick,
        fills_total,
        mm_fills_total,
        hedge_records_total,
        hedge_records_with_source,
        hedge_fills_total,
        hedge_fills_with_source,
        reconcile_drift_total,
        max_dispersion_usd,
        final_pnl_total,
        final_pnl_realised,
        final_pnl_unrealised,
        kill_switch_seen,
        kill_reason,
        mm_keep_total,
        mm_replace_total,
        mm_place_total,
    }
}

fn run_analyzer(workspace_root: &PathBuf, telemetry_path: &PathBuf) -> String {
    let output = Command::new("python3")
        .arg("tools/telemetry_analyzer.py")
        .arg("--telemetry")
        .arg(telemetry_path)
        .current_dir(workspace_root)
        .output()
        .expect("run analyzer");
    assert!(
        output.status.success(),
        "telemetry analyzer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn all5_paper_coherent_fixture_baseline_is_pnl_valid() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("all5_paper_pnl_valid_baseline");
    let telemetry_path = out_dir.join("telemetry.jsonl");
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let (hl_fixture_dir, lighter_fixture_dir, roadmap_b_fixture_dir) =
        pnl_valid_fixture_paths(&workspace_root);

    run_all5_paper_fixture(
        &out_dir,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
        "none",
        8,
        &[],
    );
    let audit = audit_paper_telemetry(&workspace_root, &telemetry_path);

    assert_eq!(audit.fills_total, 0, "baseline should not trade");
    assert_eq!(
        audit.reconcile_drift_total, 0,
        "baseline should not create reconcile drift"
    );
    assert!(
        !audit.kill_switch_seen,
        "baseline should not trip kill switch: {:?}",
        audit.kill_reason
    );
    assert!(
        audit.max_dispersion_usd <= 1.0,
        "coherent fixture dispersion too wide: {}",
        audit.max_dispersion_usd
    );
    assert!(
        audit.final_pnl_total.abs() <= 1e-6,
        "expected flat pnl_total, got {}",
        audit.final_pnl_total
    );
    assert!(
        audit.final_pnl_realised.abs() <= 1e-6,
        "expected flat pnl_realised, got {}",
        audit.final_pnl_realised
    );
    assert!(
        audit.final_pnl_unrealised.abs() <= 1e-6,
        "expected flat pnl_unrealised, got {}",
        audit.final_pnl_unrealised
    );
    assert!(
        (audit.final_pnl_total - (audit.final_pnl_realised + audit.final_pnl_unrealised)).abs()
            <= 1e-6,
        "expected pnl identity to hold"
    );
}

#[test]
fn all5_paper_coherent_fixture_exercises_mm_order_management() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("all5_paper_churn_probe");
    let telemetry_path = out_dir.join("telemetry.jsonl");
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let (hl_fixture_dir, lighter_fixture_dir, roadmap_b_fixture_dir) =
        pnl_valid_fixture_paths(&workspace_root);

    run_all5_paper_fixture(
        &out_dir,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
        "none",
        20,
        &[],
    );
    let audit = audit_paper_telemetry(&workspace_root, &telemetry_path);

    assert!(
        audit.ticks_total >= 20,
        "expected full churn probe tick count, got {}",
        audit.ticks_total
    );
    assert_eq!(
        audit.fills_total, 0,
        "churn probe should leave orders resting"
    );
    assert_eq!(
        audit.reconcile_drift_total, 0,
        "churn probe should not create reconcile drift"
    );
    assert!(
        !audit.kill_switch_seen,
        "churn probe should not trip kill switch: {:?}",
        audit.kill_reason
    );
    assert!(
        audit.max_dispersion_usd <= 1.0,
        "coherent fixture dispersion too wide: {}",
        audit.max_dispersion_usd
    );
    assert!(
        audit.final_pnl_total.abs() <= 1e-6,
        "expected flat pnl_total, got {}",
        audit.final_pnl_total
    );
    assert!(
        audit.mm_place_total > 0,
        "expected initial MM place activity in churn probe"
    );
    assert!(
        audit.mm_keep_total + audit.mm_replace_total > 0,
        "expected keep/replace MM decisions once resting orders exist"
    );
}

#[test]
fn all5_paper_coherent_fixture_produces_hedges_with_source_lineage() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let out_dir = deterministic_probe_out_dir("paraphina_all5_hedge_attr_probe");
    let telemetry_path = out_dir.join("telemetry.jsonl");
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let (hl_fixture_dir, lighter_fixture_dir, roadmap_b_fixture_dir) =
        pnl_valid_fixture_paths(&workspace_root);

    run_all5_paper_fixture(
        &out_dir,
        &hl_fixture_dir,
        &lighter_fixture_dir,
        &roadmap_b_fixture_dir,
        "mid",
        20,
        &[
            ("PARAPHINA_HEDGE_BAND_BASE", "0.01"),
            ("PARAPHINA_HEDGE_MAX_STEP", "0.25"),
            ("PARAPHINA_HEDGE_MIN_DEPTH_USD", "100"),
            ("PARAPHINA_HEDGE_K_HEDGE", "1.0"),
            ("PARAPHINA_HEDGE_LOOP_INTERVAL_MS", "250"),
            ("PARAPHINA_RECONCILE_POS_TAO_TOL", "1000000"),
            ("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "1000000"),
            ("PARAPHINA_RECONCILE_ORDER_COUNT_TOL", "1000000"),
        ],
    );
    let audit = audit_paper_telemetry(&workspace_root, &telemetry_path);
    let analyzer_report = run_analyzer(&workspace_root, &telemetry_path);

    assert!(
        audit.ticks_total >= 20,
        "expected full all-5 hedge probe tick count"
    );
    assert!(
        audit.mm_fills_total > 0,
        "expected MM fills in all-5 hedge probe"
    );
    assert!(
        audit.hedge_records_total > 0,
        "expected hedge records in all-5 hedge probe artifact at {}",
        telemetry_path.display()
    );
    assert!(
        audit.hedge_fills_total > 0,
        "expected hedge fills in all-5 hedge probe artifact at {}",
        telemetry_path.display()
    );
    assert_eq!(
        audit.reconcile_drift_total, 0,
        "all-5 hedge probe should not create reconcile drift"
    );
    assert!(
        !audit.kill_switch_seen,
        "all-5 hedge probe should not trip kill switch: {:?}",
        audit.kill_reason
    );
    assert!(
        audit.max_dispersion_usd <= 1.0,
        "coherent fixture dispersion too wide: {}",
        audit.max_dispersion_usd
    );
    assert!(
        (audit.final_pnl_total - (audit.final_pnl_realised + audit.final_pnl_unrealised)).abs()
            <= 1e-6,
        "expected pnl identity to hold"
    );

    let hedge_record_coverage =
        audit.hedge_records_with_source as f64 / audit.hedge_records_total as f64;
    let hedge_fill_coverage = audit.hedge_fills_with_source as f64 / audit.hedge_fills_total as f64;
    assert!(
        hedge_record_coverage >= 0.90,
        "expected hedge record source coverage >= 90%, got {:.2}%",
        hedge_record_coverage * 100.0
    );
    assert!(
        hedge_fill_coverage >= 0.90,
        "expected hedge fill source coverage >= 90%, got {:.2}%",
        hedge_fill_coverage * 100.0
    );
    assert!(
        analyzer_report.contains("Decision-Level Contribution"),
        "expected decision-level contribution section in analyzer report"
    );
    assert!(
        analyzer_report.contains("Venue Contribution After Hedge"),
        "expected venue contribution after hedge section in analyzer report"
    );
    assert!(
        analyzer_report.contains("Unattributed residual"),
        "expected unattributed residual section in analyzer report"
    );
}
