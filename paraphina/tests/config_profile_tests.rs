// tests/config_profile_tests.rs
//
// Note: These tests manipulate environment variables and must run serially.
// Use `cargo test --test config_profile_tests -- --test-threads=1` if flaky.

use paraphina::config::{resolve_effective_profile, Config, ProfileSource, RiskProfile};
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

// =============================================================================
// resolve_effective_profile precedence tests
// =============================================================================

#[test]
fn resolve_profile_env_var_overrides_scenario_and_default() {
    let _guard = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // Set env var to aggressive
    std::env::set_var("PARAPHINA_RISK_PROFILE", "aggressive");

    // resolve_effective_profile with no CLI arg and a scenario profile of "balanced"
    let effective = resolve_effective_profile(None, Some("balanced"));

    // Env var should win over scenario
    assert_eq!(effective.profile, RiskProfile::Aggressive);
    assert_eq!(effective.source, ProfileSource::Env);

    std::env::remove_var("PARAPHINA_RISK_PROFILE");
}

#[test]
fn resolve_profile_cli_arg_overrides_env_var() {
    let _guard = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // Set env var to conservative
    std::env::set_var("PARAPHINA_RISK_PROFILE", "conservative");

    // CLI arg is aggressive - should win over env
    let effective = resolve_effective_profile(Some(RiskProfile::Aggressive), Some("balanced"));

    // CLI should win over env
    assert_eq!(effective.profile, RiskProfile::Aggressive);
    assert_eq!(effective.source, ProfileSource::Cli);

    std::env::remove_var("PARAPHINA_RISK_PROFILE");
}

#[test]
fn resolve_profile_scenario_used_when_no_cli_or_env() {
    let _guard = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // No CLI, no env, but scenario specifies conservative
    let effective = resolve_effective_profile(None, Some("conservative"));

    assert_eq!(effective.profile, RiskProfile::Conservative);
    assert_eq!(effective.source, ProfileSource::Scenario);
}

#[test]
fn resolve_profile_default_when_nothing_specified() {
    let _guard = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // No CLI, no env, no scenario
    let effective = resolve_effective_profile(None, None);

    assert_eq!(effective.profile, RiskProfile::Balanced);
    assert_eq!(effective.source, ProfileSource::Default);
}

#[test]
fn resolve_profile_env_var_empty_falls_through_to_scenario() {
    let _guard = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("PARAPHINA_RISK_PROFILE");

    // Empty env var should be treated as unset
    std::env::set_var("PARAPHINA_RISK_PROFILE", "");

    let effective = resolve_effective_profile(None, Some("aggressive"));

    // Empty env should fall through to scenario
    assert_eq!(effective.profile, RiskProfile::Aggressive);
    assert_eq!(effective.source, ProfileSource::Scenario);

    std::env::remove_var("PARAPHINA_RISK_PROFILE");
}

#[test]
fn risk_profile_parse_accepts_various_formats() {
    // Test all accepted formats
    assert_eq!(RiskProfile::parse("balanced"), Some(RiskProfile::Balanced));
    assert_eq!(RiskProfile::parse("BALANCED"), Some(RiskProfile::Balanced));
    assert_eq!(RiskProfile::parse("bal"), Some(RiskProfile::Balanced));
    assert_eq!(RiskProfile::parse("b"), Some(RiskProfile::Balanced));

    assert_eq!(
        RiskProfile::parse("conservative"),
        Some(RiskProfile::Conservative)
    );
    assert_eq!(
        RiskProfile::parse("CONSERVATIVE"),
        Some(RiskProfile::Conservative)
    );
    assert_eq!(RiskProfile::parse("cons"), Some(RiskProfile::Conservative));
    assert_eq!(RiskProfile::parse("c"), Some(RiskProfile::Conservative));

    assert_eq!(
        RiskProfile::parse("aggressive"),
        Some(RiskProfile::Aggressive)
    );
    assert_eq!(
        RiskProfile::parse("AGGRESSIVE"),
        Some(RiskProfile::Aggressive)
    );
    assert_eq!(RiskProfile::parse("agg"), Some(RiskProfile::Aggressive));
    assert_eq!(RiskProfile::parse("a"), Some(RiskProfile::Aggressive));
    assert_eq!(RiskProfile::parse("loose"), Some(RiskProfile::Aggressive));
    assert_eq!(RiskProfile::parse("l"), Some(RiskProfile::Aggressive));

    // Invalid values
    assert_eq!(RiskProfile::parse("invalid"), None);
    assert_eq!(RiskProfile::parse(""), None);
}
