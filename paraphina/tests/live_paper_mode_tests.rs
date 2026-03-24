#[cfg(feature = "live_hyperliquid")]
use std::path::PathBuf;
#[cfg(feature = "live_hyperliquid")]
use std::process::Command;

#[cfg(feature = "live_hyperliquid")]
use serde_json::Value;
#[cfg(feature = "live_hyperliquid")]
use tempfile::TempDir;

#[cfg(feature = "live_hyperliquid")]
struct PaperRunPaths {
    workspace_root: PathBuf,
    telemetry_path: PathBuf,
    fixture_dir: PathBuf,
}

#[cfg(feature = "live_hyperliquid")]
struct PaperTelemetryAudit {
    ticks_total: u64,
    mm_fills_total: usize,
    hedge_records_total: usize,
    hedge_records_with_source: usize,
    hedge_fills_total: usize,
    hedge_fills_with_source: usize,
    reconcile_drift_total: usize,
    final_pnl_total: f64,
    final_pnl_realised: f64,
    final_pnl_unrealised: f64,
    kill_switch_seen: bool,
    kill_reason: Option<String>,
}

#[cfg(feature = "live_hyperliquid")]
fn hyperliquid_paper_fixture_paths(artifact_dir_name: &str) -> PaperRunPaths {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("hyperliquid");
    let out_dir = std::env::temp_dir().join(artifact_dir_name);
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    PaperRunPaths {
        workspace_root,
        telemetry_path: out_dir.join("telemetry.jsonl"),
        fixture_dir,
    }
}

#[cfg(feature = "live_hyperliquid")]
fn run_hyperliquid_paper_fixture(
    artifact_dir_name: &str,
    extra_env: &[(&str, &str)],
) -> PaperRunPaths {
    let paths = hyperliquid_paper_fixture_paths(artifact_dir_name);
    let out_dir = paths
        .telemetry_path
        .parent()
        .expect("telemetry parent")
        .to_path_buf();

    let mut command = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    command
        .arg("--trade-mode")
        .arg("paper")
        .arg("--connector")
        .arg("hyperliquid-fixture")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_LIVE_MAX_TICKS", "20")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_PAPER_FILL_MODE", "mid")
        .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
        .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "1")
        .env("PARAPHINA_PAPER_DISABLE_FV_GATE", "1")
        .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1")
        .env("PARAPHINA_PAPER_DISABLE_HEALTH_GATES", "1")
        .env("PARAPHINA_PAPER_SMOKE_INTENTS", "1")
        .env("PARAPHINA_RECONCILE_POS_TAO_TOL", "1000000")
        .env("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "1000000")
        .env("PARAPHINA_RECONCILE_ORDER_COUNT_TOL", "1000000")
        .env("HL_FIXTURE_DIR", &paths.fixture_dir);

    for (key, value) in extra_env {
        command.env(key, value);
    }

    let output = command.output().expect("run paraphina_live");

    assert!(
        output.status.success(),
        "paper mode run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("paper_execution=internal"),
        "expected paper execution marker in stderr"
    );
    std::fs::write(out_dir.join("run.log"), &output.stderr).expect("write run log");

    paths
}

#[cfg(feature = "live_hyperliquid")]
fn check_telemetry_contract(workspace_root: &PathBuf, telemetry_path: &PathBuf) {
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
}

#[cfg(feature = "live_hyperliquid")]
fn audit_paper_telemetry(telemetry_path: &PathBuf) -> PaperTelemetryAudit {
    let data = std::fs::read_to_string(telemetry_path).expect("read telemetry");
    let mut ticks_total = 0_u64;
    let mut mm_fills_total = 0_usize;
    let mut hedge_records_total = 0_usize;
    let mut hedge_records_with_source = 0_usize;
    let mut hedge_fills_total = 0_usize;
    let mut hedge_fills_with_source = 0_usize;
    let mut reconcile_drift_total = 0_usize;
    let mut final_pnl_total = 0.0_f64;
    let mut final_pnl_realised = 0.0_f64;
    let mut final_pnl_unrealised = 0.0_f64;
    let mut kill_switch_seen = false;
    let mut kill_reason = None;

    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).expect("parse telemetry JSON");
        assert_eq!(
            value.get("execution_mode").and_then(|v| v.as_str()),
            Some("paper"),
            "expected execution_mode=paper in telemetry"
        );
        ticks_total += 1;
        reconcile_drift_total += value
            .get("reconcile_drift")
            .and_then(|v| v.as_array())
            .map(|items| items.len())
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
    }

    PaperTelemetryAudit {
        ticks_total,
        mm_fills_total,
        hedge_records_total,
        hedge_records_with_source,
        hedge_fills_total,
        hedge_fills_with_source,
        reconcile_drift_total,
        final_pnl_total,
        final_pnl_realised,
        final_pnl_unrealised,
        kill_switch_seen,
        kill_reason,
    }
}

#[cfg(feature = "live_hyperliquid")]
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

#[cfg(feature = "live_hyperliquid")]
#[test]
fn paper_mode_fixture_run_produces_fills_and_valid_telemetry() {
    let _tmp = TempDir::new().expect("temp dir");
    let paths = run_hyperliquid_paper_fixture("paraphina_paper_run_basic", &[]);
    check_telemetry_contract(&paths.workspace_root, &paths.telemetry_path);

    let data = std::fs::read_to_string(&paths.telemetry_path).expect("read telemetry");
    let mut has_paper_mode = false;
    let mut has_fill = false;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).expect("parse telemetry JSON");
        if value.get("execution_mode").and_then(|v| v.as_str()) == Some("paper") {
            has_paper_mode = true;
        }
        if value
            .get("fills")
            .and_then(|v| v.as_array())
            .map(|fills| !fills.is_empty())
            .unwrap_or(false)
        {
            has_fill = true;
        }
    }

    assert!(has_paper_mode, "expected execution_mode=paper in telemetry");
    assert!(has_fill, "expected at least one fill in telemetry");
}

#[cfg(feature = "live_hyperliquid")]
#[test]
fn paper_mode_fixture_run_produces_hedges_with_source_lineage() {
    let paths = run_hyperliquid_paper_fixture(
        "paraphina_mm_hedge_attr_probe",
        &[
            ("PARAPHINA_HEDGE_BAND_BASE", "0.01"),
            ("PARAPHINA_HEDGE_MAX_STEP", "0.25"),
            ("PARAPHINA_HEDGE_MIN_DEPTH_USD", "100"),
            ("PARAPHINA_HEDGE_K_HEDGE", "1.0"),
            ("PARAPHINA_HEDGE_LOOP_INTERVAL_MS", "250"),
        ],
    );
    check_telemetry_contract(&paths.workspace_root, &paths.telemetry_path);
    let audit = audit_paper_telemetry(&paths.telemetry_path);
    let analyzer_report = run_analyzer(&paths.workspace_root, &paths.telemetry_path);

    assert!(audit.ticks_total > 0, "expected telemetry rows");
    assert!(audit.mm_fills_total > 0, "expected MM fills in hedge probe");
    assert!(
        audit.hedge_records_total > 0,
        "expected hedge records in hedge probe artifact at {}",
        paths.telemetry_path.display()
    );
    assert!(
        audit.hedge_fills_total > 0,
        "expected hedge fills in hedge probe artifact at {}",
        paths.telemetry_path.display()
    );
    assert_eq!(
        audit.reconcile_drift_total, 0,
        "expected no reconcile drift in hedge probe"
    );
    assert!(
        !audit.kill_switch_seen,
        "hedge probe should not trip kill switch: {:?}",
        audit.kill_reason
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
        analyzer_report.contains("Hedge source attribution"),
        "expected hedge source attribution section in analyzer report"
    );
    assert!(
        analyzer_report.contains("Hedge fill source attribution"),
        "expected hedge fill source attribution section in analyzer report"
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
