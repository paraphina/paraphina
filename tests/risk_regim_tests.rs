// tests/risk_regim_tests.rs

use paraphina::config::{Config, RiskProfile};
use paraphina::engine::Engine;
use paraphina::state::{GlobalState, RiskRegime};

/// Small harness that gives us a Config, Engine and GlobalState
/// for a given risk profile.
///
/// We Box + leak the config so the Engine can hold a &'static Config
/// without lifetime headaches in tests.
struct Harness {
    cfg: &'static Config,
    engine: Engine<'static>,
    state: GlobalState,
}

fn make_harness(profile: RiskProfile) -> Harness {
    let cfg_box = Box::new(Config::for_profile(profile));
    let cfg: &'static Config = Box::leak(cfg_box);

    let engine = Engine::new(cfg);
    let state = GlobalState::new(cfg);

    Harness { cfg, engine, state }
}

#[test]
fn regime_stays_normal_with_small_exposures() {
    // Use conservative profile – tightest limits.
    let mut h = make_harness(RiskProfile::Conservative);

    // Run one tick to initialise vol + risk limits.
    h.engine.main_tick(&mut h.state, 0);

    // With zero delta / basis / PnL, we should be firmly in Normal regime
    // and the kill switch should be off.
    assert_eq!(h.state.risk_regime, RiskRegime::Normal);
    assert!(!h.state.kill_switch);
}

#[test]
fn hardlimit_and_kill_switch_when_loss_limit_breached() {
    let mut h = make_harness(RiskProfile::Conservative);

    // First tick to make sure limits are initialised.
    h.engine.main_tick(&mut h.state, 0);

    // Config stores a *positive* daily_loss_limit in USD.
    // Risk engine interprets this as a negative PnL threshold.
    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();

    // Drive PnL slightly past the hard loss limit with clean delta/basis.
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.dollar_delta_usd = 0.0;
    h.state.basis_usd = 0.0;

    // Second tick so the risk-regime logic runs on this PnL snapshot.
    h.engine.main_tick(&mut h.state, 1);

    // Once the hard loss limit is breached, we expect:
    // - regime = HardLimit
    // - kill_switch = true
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
}

#[test]
fn hardlimit_disables_mm_quotes_even_if_kill_switch_is_false() {
    // Conservative profile – tightest limits.
    let mut h = make_harness(RiskProfile::Conservative);

    // First tick to initialise fair value + delta limits.
    h.engine.main_tick(&mut h.state, 0);

    // Force a delta hard-limit breach WITHOUT breaching the loss limit.
    // We do this by setting dollar_delta_usd above the computed delta_limit_usd.
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.basis_usd = 0.0;

    // Trigger hardlimit by delta.
    h.state.dollar_delta_usd = h.state.delta_limit_usd + 1.0;

    // Re-run the risk logic.
    h.engine.update_risk_limits_and_regime(&mut h.state);

    // We should be in HardLimit, but kill switch should still be false (no loss breach).
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(!h.state.kill_switch);

    // MM should produce no quotes in HardLimit.
    let quotes = paraphina::mm::compute_mm_quotes(h.cfg, &h.state);
    assert_eq!(quotes.len(), h.cfg.venues.len());
    for q in quotes {
        assert!(q.bid.is_none(), "bid should be None in HardLimit");
        assert!(q.ask.is_none(), "ask should be None in HardLimit");
    }
}
