#[cfg(all(feature = "live", feature = "live_paradex"))]
mod tests {
    use std::process::Command;
    use std::sync::Mutex;

    use tempfile::tempdir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn paradex_fixture_paper_run_is_healthy() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let temp_dir = tempdir().expect("tempdir");
        let out_dir = temp_dir.path().join("paradex_fixture_paper");
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
            .join("paradex");

        let bin_path = env!("CARGO_BIN_EXE_paraphina_live");
        let output = Command::new(bin_path)
            .arg("--trade-mode")
            .arg("paper")
            .arg("--connector")
            .arg("paradex")
            .env("PARADEX_FIXTURE_MODE", "1")
            .env("PARADEX_FIXTURE_DIR", fixture_dir.to_string_lossy().to_string())
            .env("PARAPHINA_LIVE_OUT_DIR", out_dir.to_string_lossy().to_string())
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
            .expect("failed to run paraphina_live in paradex fixture mode");

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
}
