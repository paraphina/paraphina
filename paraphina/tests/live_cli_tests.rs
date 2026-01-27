// CLI integration tests for paraphina_live startup log.

use std::process::Command;

#[test]
#[cfg(feature = "live")]
fn live_startup_log_includes_required_fields() {
    let bin = env!("CARGO_BIN_EXE_paraphina_live");
    let output = Command::new(bin)
        .env("PARAPHINA_TRADE_MODE", "shadow")
        .env("PARAPHINA_LIVE_CONNECTOR", "hyperliquid_fixture")
        .env("HL_FIXTURE_DIR", "../tests/fixtures/hyperliquid")
        .env("PARAPHINA_LIVE_MAX_TICKS", "1")
        .env("PARAPHINA_LIVE_METRICS_ADDR", "127.0.0.1:9999")
        .output()
        .expect("failed to run paraphina_live binary");

    assert!(
        output.status.success(),
        "paraphina_live failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("build_id="), "missing build_id: {stdout}");
    assert!(stdout.contains("cfg_hash=0x"), "missing cfg_hash: {stdout}");
    assert!(
        stdout.contains("trade_mode="),
        "missing trade_mode: {stdout}"
    );
    assert!(
        stdout.contains("connectors="),
        "missing connectors: {stdout}"
    );
    assert!(stdout.contains("venues="), "missing venues: {stdout}");
    assert!(
        stdout.contains("intervals_ms=main:"),
        "missing intervals_ms: {stdout}"
    );
    assert!(
        stdout.contains("metrics_addr=127.0.0.1:9999"),
        "missing metrics addr: {stdout}"
    );
}
