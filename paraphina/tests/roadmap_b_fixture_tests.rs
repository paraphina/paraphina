#[cfg(all(feature = "live", feature = "live_hyperliquid"))]
mod tests {
    use std::process::Command;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn roadmap_b_fixture_run_produces_telemetry() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let temp_dir = tempdir().expect("tempdir");
        let out_dir = temp_dir.path().join("roadmap_b_fixture");
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let telemetry_path = out_dir.join("telemetry.jsonl");

        let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let fixture_dir = workspace_root
            .join("tests")
            .join("fixtures")
            .join("hyperliquid");

        let bin_path = env!("CARGO_BIN_EXE_paraphina_live");
        let output = Command::new(bin_path)
            .arg("--trade-mode")
            .arg("shadow")
            .arg("--connector")
            .arg("hyperliquid-fixture")
            .env("PARAPHINA_LIVE_OUT_DIR", out_dir.to_string_lossy().to_string())
            .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
            .env(
                "PARAPHINA_TELEMETRY_PATH",
                telemetry_path.to_string_lossy().to_string(),
            )
            .env("PARAPHINA_LIVE_CONNECTOR", "hyperliquid_fixture")
            .env("HL_FIXTURE_DIR", fixture_dir.to_string_lossy().to_string())
            .env("PARAPHINA_LIVE_MAX_TICKS", "5")
            .output()
            .expect("failed to run paraphina_live in shadow fixture mode");

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
