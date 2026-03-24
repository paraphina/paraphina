#[cfg(all(feature = "live", feature = "live_aster"))]
mod tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::Mutex;

    use serde_json::Value;
    use tempfile::tempdir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());
    const ASTER_INDEX: usize = 2;

    fn deterministic_out_dir(name: &str) -> PathBuf {
        let out_dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&out_dir);
        std::fs::create_dir_all(&out_dir).expect("create deterministic out dir");
        out_dir
    }

    #[test]
    fn aster_fixture_paper_run_is_healthy() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let temp_dir = tempdir().expect("tempdir");
        let out_dir = temp_dir.path().join("aster_fixture_paper");
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let telemetry_path = out_dir.join("telemetry.jsonl");

        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let fixture_dir = workspace_root
            .join("tests")
            .join("fixtures")
            .join("roadmap_b")
            .join("aster");

        let bin_path = env!("CARGO_BIN_EXE_paraphina_live");
        let output = Command::new(bin_path)
            .arg("--trade-mode")
            .arg("paper")
            .arg("--connector")
            .arg("aster")
            .env("ASTER_FIXTURE_MODE", "1")
            .env(
                "ASTER_FIXTURE_DIR",
                fixture_dir.to_string_lossy().to_string(),
            )
            .env(
                "PARAPHINA_LIVE_OUT_DIR",
                out_dir.to_string_lossy().to_string(),
            )
            .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
            .env(
                "PARAPHINA_TELEMETRY_PATH",
                telemetry_path.to_string_lossy().to_string(),
            )
            .env("PARAPHINA_LIVE_MAX_TICKS", "10")
            .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
            .env("PARAPHINA_PAPER_FILL_MODE", "mid")
            .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
            .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "5")
            .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1")
            .output()
            .expect("failed to run paraphina_live in aster fixture mode");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "paraphina_live failed: {}", stderr);
        assert!(telemetry_path.exists(), "telemetry file not found");

        let check = Command::new("python3")
            .arg("tools/check_telemetry_contract.py")
            .arg(&telemetry_path)
            .current_dir(&workspace_root)
            .output()
            .expect("failed to run check_telemetry_contract.py");
        assert!(
            check.status.success(),
            "telemetry contract check failed: {}",
            String::from_utf8_lossy(&check.stderr)
        );
    }

    #[test]
    fn aster_passive_qualification_fixture_produces_mm_fills_inside_mm_program_band() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let fixture_dir = workspace_root
            .join("tests")
            .join("fixtures")
            .join("aster_passive_qualification");
        let out_dir = deterministic_out_dir("paraphina_aster_passive_qual_probe");
        let telemetry_path = out_dir.join("telemetry.jsonl");

        let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
            .arg("--trade-mode")
            .arg("paper")
            .arg("--connector")
            .arg("aster")
            .env("ASTER_FIXTURE_MODE", "1")
            .env(
                "ASTER_FIXTURE_DIR",
                fixture_dir.to_string_lossy().to_string(),
            )
            .env(
                "PARAPHINA_LIVE_OUT_DIR",
                out_dir.to_string_lossy().to_string(),
            )
            .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
            .env(
                "PARAPHINA_TELEMETRY_PATH",
                telemetry_path.to_string_lossy().to_string(),
            )
            .env("PARAPHINA_LIVE_MAX_TICKS", "20")
            .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
            .env("PARAPHINA_PAPER_FILL_MODE", "mid")
            .env("PARAPHINA_PAPER_SLIPPAGE_BPS", "5")
            .env("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF", "1")
            .env("PARAPHINA_PAPER_DISABLE_FV_GATE", "1")
            .env("PARAPHINA_PAPER_DISABLE_HEALTH_GATES", "1")
            .env("PARAPHINA_PAPER_SMOKE_INTENTS", "1")
            .env("PARAPHINA_PAPER_USE_WALLCLOCK_TS", "1")
            .env("PARAPHINA_MM_VENUE_ROLE_ASTER", "fill")
            .env("PARAPHINA_MM_MAX_QUOTE_SPREAD_BPS_ASTER", "10")
            .env("PARAPHINA_MM_MIN_QUOTE_LIFETIME_MS_ASTER", "500")
            .env("PARAPHINA_RECONCILE_POS_TAO_TOL", "1000000")
            .env("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "1000000")
            .env("PARAPHINA_RECONCILE_ORDER_COUNT_TOL", "1000000")
            .output()
            .expect("failed to run aster passive qualification fixture");

        let stderr = String::from_utf8_lossy(&output.stderr);
        std::fs::write(out_dir.join("run.log"), &output.stderr).expect("write run log");
        assert!(
            output.status.success(),
            "aster passive qualification run failed: {}",
            stderr
        );
        assert!(telemetry_path.exists(), "telemetry file not found");

        let check = Command::new("python3")
            .arg("tools/check_telemetry_contract.py")
            .arg(&telemetry_path)
            .current_dir(&workspace_root)
            .output()
            .expect("failed to run check_telemetry_contract.py");
        assert!(
            check.status.success(),
            "telemetry contract check failed: {}",
            String::from_utf8_lossy(&check.stderr)
        );

        let data = std::fs::read_to_string(&telemetry_path).expect("read telemetry");
        let mut ticks_total = 0usize;
        let mut mm_fills_total = 0usize;
        let mut reconcile_drift_total = 0usize;
        let mut kill_switch_seen = false;
        let mut spread_band_ticks = 0usize;
        let mut mm_places_total = 0u64;
        let mut final_pnl_total = 0.0f64;
        let mut final_pnl_realised = 0.0f64;
        let mut final_pnl_unrealised = 0.0f64;

        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).expect("parse telemetry JSON");
            ticks_total += 1;
            assert_eq!(
                value.get("execution_mode").and_then(|v| v.as_str()),
                Some("paper"),
                "expected execution_mode=paper"
            );

            let mid = value
                .get("venue_mid_usd")
                .and_then(|v| v.as_array())
                .and_then(|items| items.get(ASTER_INDEX))
                .and_then(|v| v.as_f64())
                .expect("aster mid");
            let spread = value
                .get("venue_spread_usd")
                .and_then(|v| v.as_array())
                .and_then(|items| items.get(ASTER_INDEX))
                .and_then(|v| v.as_f64())
                .expect("aster spread");
            let spread_bps = spread / mid * 10_000.0;
            if spread_bps <= 10.0 {
                spread_band_ticks += 1;
            }

            let aster_status = value
                .get("venue_status")
                .and_then(|v| v.as_array())
                .and_then(|items| items.get(ASTER_INDEX))
                .and_then(|v| v.as_str())
                .expect("aster status");
            assert_eq!(
                aster_status, "Healthy",
                "expected Aster fixture venue healthy"
            );

            if let Some(fills) = value.get("fills").and_then(|v| v.as_array()) {
                for fill in fills {
                    if fill.get("purpose").and_then(|v| v.as_str()) == Some("Mm") {
                        mm_fills_total += 1;
                    }
                }
            }

            reconcile_drift_total += value
                .get("reconcile_drift")
                .and_then(|v| v.as_array())
                .map(|items| items.len())
                .unwrap_or(0);

            if value
                .get("kill_switch")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                kill_switch_seen = true;
            }

            if let Some(summary) = value.get("mm_order_management").and_then(|v| v.as_object()) {
                mm_places_total += summary
                    .get("place_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }

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
        }

        assert!(ticks_total > 0, "expected telemetry rows");
        assert_eq!(
            spread_band_ticks, ticks_total,
            "all ticks should stay within Aster's 10bps MM spread band"
        );
        assert!(mm_places_total > 0, "expected Aster MM place decisions");
        assert!(mm_fills_total > 0, "expected attributable Aster MM fills");
        assert_eq!(
            reconcile_drift_total, 0,
            "qualification fixture should not produce reconcile drift"
        );
        assert!(
            !kill_switch_seen,
            "qualification fixture should not trip kill switch"
        );
        assert!(
            (final_pnl_total - (final_pnl_realised + final_pnl_unrealised)).abs() <= 1e-6,
            "expected pnl identity to hold"
        );
    }
}
