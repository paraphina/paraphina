// tests/config_profile_tests.rs
//
// Note: These tests manipulate environment variables and must run serially.
// Use `cargo test --test config_profile_tests -- --test-threads=1` if flaky.

use paraphina::config::Config;
use std::sync::Mutex;

// Global mutex to serialize tests that touch environment variables.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn env_risk_profile_is_honored_by_from_env_or_default() {
    let _guard = ENV_MUTEX.lock().unwrap();

    // Clean up any stale env var first
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // Set env var for this test process.
    std::env::set_var("PARAPHINA_RISK_PROFILE", "conservative");

    let cfg = Config::from_env_or_default();

    // Conservative in src/config.rs sets daily_loss_limit to 2000.0 (current code truth).
    // If you later change the conservative preset (e.g., to 750.0), update this expected value
    // AND keep the test to prevent silent drift.
    assert_eq!(cfg.risk.daily_loss_limit, 2000.0);

    // Cleanup to avoid polluting other tests.
    std::env::remove_var("PARAPHINA_RISK_PROFILE");
}

#[test]
fn unknown_env_risk_profile_falls_back_to_balanced() {
    let _guard = ENV_MUTEX.lock().unwrap();

    // Clean up any stale env var first
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    std::env::set_var("PARAPHINA_RISK_PROFILE", "definitely_not_a_real_profile");

    let cfg = Config::from_env_or_default();

    // Balanced in src/config.rs uses 5000.0 as the centre loss limit.
    assert_eq!(cfg.risk.daily_loss_limit, 5000.0);

    std::env::remove_var("PARAPHINA_RISK_PROFILE");
}
