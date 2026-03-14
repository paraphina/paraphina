#![cfg(feature = "live")]

use std::process::Command;

use tempfile::TempDir;

#[test]
fn live_mode_refuses_to_start_without_guardrails() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    let output = cmd
        .arg("--trade-mode")
        .arg("live")
        .arg("--connector")
        .arg("mock")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .output()
        .expect("run paraphina_live");
    assert!(
        !output.status.success(),
        "expected non-zero exit when live guardrails are missing"
    );
}

#[test]
fn shadow_mode_writes_out_dir_artifacts_without_keys() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("shadow_run");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_paraphina_live"));
    let status = cmd
        .arg("--trade-mode")
        .arg("shadow")
        .arg("--connector")
        .arg("mock")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_LIVE_MAX_TICKS", "1")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .status()
        .expect("run paraphina_live");
    assert!(status.success(), "shadow mode should exit cleanly");

    assert!(out_dir.join("config_resolved.json").exists());
    assert!(out_dir.join("build_info.json").exists());
    assert!(out_dir.join("telemetry.jsonl").exists());
    assert!(out_dir.join("summary.json").exists());
}

#[cfg(feature = "live_hyperliquid")]
#[test]
fn preflight_testnet_rejects_hyperliquid_paper_mode() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("hl_testnet_preflight_reject");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .arg("--trade-mode")
        .arg("testnet")
        .arg("--connector")
        .arg("hyperliquid")
        .arg("--preflight")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("HL_WS_URL", "ws://127.0.0.1:1")
        .env("HL_REST_URL", "http://127.0.0.1:1")
        .env("HL_INFO_URL", "http://127.0.0.1:1")
        .env("HL_PRIVATE_KEY", "0x01")
        .env("HL_VAULT_ADDRESS", "0xabc")
        .env("HL_PAPER_MODE", "true")
        .output()
        .expect("run paraphina_live preflight");
    assert!(
        !output.status.success(),
        "expected preflight fail when HL_PAPER_MODE=true in testnet mode"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("FAIL execution_modes"),
        "expected execution_modes gate to fail, stdout: {stdout}"
    );
}

#[cfg(feature = "live_hyperliquid")]
#[test]
fn preflight_testnet_accepts_hyperliquid_live_mode() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("hl_testnet_preflight_accept");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .arg("--trade-mode")
        .arg("testnet")
        .arg("--connector")
        .arg("hyperliquid")
        .arg("--preflight")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("HL_WS_URL", "ws://127.0.0.1:1")
        .env("HL_REST_URL", "http://127.0.0.1:1")
        .env("HL_INFO_URL", "http://127.0.0.1:1")
        .env("HL_PRIVATE_KEY", "0x01")
        .env("HL_VAULT_ADDRESS", "0xabc")
        .env("HL_PAPER_MODE", "false")
        .output()
        .expect("run paraphina_live preflight");
    assert!(
        output.status.success(),
        "expected preflight pass when HL_PAPER_MODE=false in testnet mode"
    );
}

#[cfg(feature = "live_lighter")]
#[test]
fn preflight_testnet_rejects_lighter_paper_mode() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("lighter_testnet_preflight_reject");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .arg("--trade-mode")
        .arg("testnet")
        .arg("--connector")
        .arg("lighter")
        .arg("--preflight")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("LIGHTER_WS_URL", "ws://127.0.0.1:1")
        .env("LIGHTER_HTTP_BASE_URL", "http://127.0.0.1:1")
        .env("LIGHTER_API_KEY_INDEX", "1")
        .env("LIGHTER_ACCOUNT_INDEX", "1")
        .env("LIGHTER_API_PRIVATE_KEY_HEX", "abcd")
        .env("LIGHTER_SIGNER_URL", "http://127.0.0.1:9001")
        .env("LIGHTER_PAPER_MODE", "true")
        .output()
        .expect("run paraphina_live preflight");
    assert!(
        !output.status.success(),
        "expected preflight fail when LIGHTER_PAPER_MODE=true in testnet mode"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("FAIL execution_modes"),
        "expected execution_modes gate to fail, stdout: {stdout}"
    );
}

#[cfg(feature = "live_lighter")]
#[test]
fn preflight_testnet_accepts_lighter_live_mode() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("lighter_testnet_preflight_accept");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .arg("--trade-mode")
        .arg("testnet")
        .arg("--connector")
        .arg("lighter")
        .arg("--preflight")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("LIGHTER_WS_URL", "ws://127.0.0.1:1")
        .env("LIGHTER_HTTP_BASE_URL", "http://127.0.0.1:1")
        .env("LIGHTER_API_KEY_INDEX", "1")
        .env("LIGHTER_ACCOUNT_INDEX", "1")
        .env("LIGHTER_API_PRIVATE_KEY_HEX", "abcd")
        .env("LIGHTER_SIGNER_URL", "http://127.0.0.1:9001")
        .env("LIGHTER_PAPER_MODE", "false")
        .output()
        .expect("run paraphina_live preflight");
    assert!(
        output.status.success(),
        "expected preflight pass when LIGHTER_PAPER_MODE=false in testnet mode"
    );
}

#[cfg(feature = "live_extended")]
#[test]
fn preflight_live_accepts_extended_bridge_auth() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("extended_live_preflight_accept");
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .current_dir(&workspace_root)
        .arg("--trade-mode")
        .arg("live")
        .arg("--connector")
        .arg("extended")
        .arg("--preflight")
        .arg("--enable-live-execution")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_LIVE_EXEC_ENABLE", "1")
        .env("PARAPHINA_LIVE_EXECUTION_CONFIRM", "YES")
        .env("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS", "5000")
        .env("PARAPHINA_LIVE_CANARY_PROFILE", "configs/prod_canary_eth_min.toml")
        .env("EXTENDED_WS_URL", "ws://127.0.0.1:1")
        .env("EXTENDED_REST_URL", "http://127.0.0.1:1")
        .env("EXTENDED_MARKET", "ETH-USD")
        .env("EXTENDED_API_KEY", "test-api-key")
        .env("EXTENDED_TRADER_CMD", "/bin/true")
        .env("EXTENDED_STARK_PRIVATE_KEY", "0x01")
        .env("EXTENDED_STARK_PUBLIC_KEY", "0x02")
        .env("EXTENDED_L2_VAULT", "1")
        .output()
        .expect("run paraphina_live preflight");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected extended bridge auth preflight pass; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("PASS credentials"),
        "expected credentials pass, stdout: {stdout}"
    );
}

#[cfg(feature = "live_hyperliquid")]
#[test]
fn preflight_live_accepts_prod_canary_alias() {
    let tmp = TempDir::new().expect("temp dir");
    let out_dir = tmp.path().join("hl_live_preflight_prod_canary");
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let output = Command::new(env!("CARGO_BIN_EXE_paraphina_live"))
        .current_dir(&workspace_root)
        .arg("--trade-mode")
        .arg("live")
        .arg("--connector")
        .arg("hyperliquid")
        .arg("--preflight")
        .arg("--enable-live-execution")
        .env("PARAPHINA_LIVE_OUT_DIR", &out_dir)
        .env("PARAPHINA_TELEMETRY_MODE", "jsonl")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:0")
        .env("PARAPHINA_LIVE_EXEC_ENABLE", "1")
        .env("PARAPHINA_LIVE_EXECUTION_CONFIRM", "YES")
        .env("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS", "5000")
        .env("PARAPHINA_LIVE_CANARY_PROFILE", "prod_canary")
        .env("HL_WS_URL", "ws://127.0.0.1:1")
        .env("HL_REST_URL", "http://127.0.0.1:1")
        .env("HL_INFO_URL", "http://127.0.0.1:1")
        .env("HL_PRIVATE_KEY", "0x01")
        .env("HL_VAULT_ADDRESS", "0xabc")
        .env("HL_PAPER_MODE", "false")
        .output()
        .expect("run paraphina_live preflight");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected live preflight pass with prod_canary alias; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("PASS canary_profile"),
        "expected canary_profile pass, stdout: {stdout}"
    );
}
