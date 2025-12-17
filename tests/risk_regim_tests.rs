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

/// Helper: seed deterministic books then run a tick so FV/vol/limits are initialised.
fn init_tick(h: &mut Harness, now_ms: i64) {
    h.engine.seed_dummy_mids(&mut h.state, now_ms);
    h.engine.main_tick(&mut h.state, now_ms);
}

#[test]
fn regime_stays_normal_with_small_exposures() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // With ~0 delta / basis / PnL, we should be Normal and not killed.
    assert_eq!(h.state.risk_regime, RiskRegime::Normal);
    assert!(!h.state.kill_switch);
}

#[test]
fn warning_when_delta_above_warn_but_below_hard() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Place delta between warn and hard thresholds.
    let delta_warn = h.cfg.risk.delta_warn_frac * h.state.delta_limit_usd;
    let delta_hard = h.state.delta_limit_usd;
    let delta_mid = 0.5 * (delta_warn + delta_hard);

    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.basis_usd = 0.0;
    h.state.dollar_delta_usd = delta_mid;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert_eq!(h.state.risk_regime, RiskRegime::Warning);
    assert!(!h.state.kill_switch);
}

#[test]
fn hardlimit_and_kill_switch_when_loss_limit_breached() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();

    // Breach daily loss hard limit, keep delta/basis clean.
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.dollar_delta_usd = 0.0;
    h.state.basis_usd = 0.0;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
}

#[test]
fn kill_switch_latches_once_true() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();

    // Trip kill.
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert!(h.state.kill_switch);
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);

    // Now "recover" PnL; killzn = safe again.
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Kill stays latched until manual reset.
    assert!(h.state.kill_switch);
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
}

#[test]
fn hardlimit_disables_mm_quotes_even_if_kill_switch_is_false() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Force a delta hard-limit breach WITHOUT breaching the loss limit.
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.basis_usd = 0.0;

    // Trigger hardlimit by delta.
    h.state.dollar_delta_usd = h.state.delta_limit_usd + 1.0;

    // Re-run the risk logic.
    h.engine.update_risk_limits_and_regime(&mut h.state);

    // HardLimit, but kill switch should still be false (no loss breach).
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(!h.state.kill_switch);

    // MM should produce no quotes in HardLimit (even if kill is false).
    let quotes = paraphina::mm::compute_mm_quotes(h.cfg, &h.state);
    assert_eq!(quotes.len(), h.cfg.venues.len());
    for q in quotes {
        assert!(q.bid.is_none(), "bid should be None in HardLimit");
        assert!(q.ask.is_none(), "ask should be None in HardLimit");
    }
}
