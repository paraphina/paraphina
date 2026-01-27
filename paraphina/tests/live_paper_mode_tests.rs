use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

#[cfg(feature = "live_hyperliquid")]
#[test]
fn paper_mode_fixture_run_produces_fills_and_valid_telemetry() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("paper_run");
    let telemetry_path = out_dir.join("telemetry.jsonl");

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let fixture_dir = workspace_root
        .join("tests")
        .join("fixtures")
        .join("hyperliquid");

    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
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
        .env("HL_FIXTURE_DIR", &fixture_dir)
        .output()
        .expect("run paraphina_live");

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

    let check = Command::new("python3")
        .arg("tools/check_telemetry_contract.py")
        .arg(&telemetry_path)
        .current_dir(&workspace_root)
        .output()
        .expect("run telemetry contract check");
    assert!(
        check.status.success(),
        "telemetry contract check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    let data = std::fs::read_to_string(&telemetry_path).expect("read telemetry");
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
