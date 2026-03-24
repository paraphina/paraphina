#![cfg(all(feature = "live", feature = "live_aster"))]

use std::process::Command;

use tempfile::TempDir;

#[test]
fn preflight_live_accepts_aster_passive_micro_canary_profile() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("aster_live_preflight_passive_canary");
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .current_dir(&workspace_root)
        .arg("--trade-mode")
        .arg("live")
        .arg("--connector")
        .arg("aster")
        .arg("--preflight")
        .arg("--enable-live-execution")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_LIVE_EXEC_ENABLE", "1")
        .env("PARAPHINA_LIVE_EXECUTION_CONFIRM", "YES")
        .env("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS", "5000")
        .env(
            "PARAPHINA_LIVE_CANARY_PROFILE",
            "configs/prod_canary_aster_passive_min.toml",
        )
        .env("ASTER_WS_URL", "ws://127.0.0.1:1")
        .env("ASTER_REST_URL", "http://127.0.0.1:1")
        .env("ASTER_MARKET", "ETHUSDT")
        .env("ASTER_API_KEY", "test-api-key")
        .env("ASTER_API_SECRET", "test-api-secret")
        .output()
        .expect("run paraphina_live preflight");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected aster passive canary preflight pass; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("PASS canary_profile"),
        "expected canary profile gate pass, stdout: {stdout}"
    );
    assert!(
        stdout.contains("PASS credentials"),
        "expected credentials pass, stdout: {stdout}"
    );
}
