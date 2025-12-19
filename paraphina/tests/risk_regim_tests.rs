// tests/risk_regim_tests.rs
//
// Milestone C regression tests for risk regime correctness and kill switch semantics.
//
// Tests cover:
// - PnL hard breach triggers Critical (HardLimit) AND kill_switch=true
// - Delta hard breach triggers Critical (HardLimit) AND kill_switch=true
// - Basis hard breach triggers Critical (HardLimit) AND kill_switch=true
// - Liquidation distance hard breach triggers Critical (HardLimit) AND kill_switch=true
// - Kill_switch latches once true (cannot revert to false on subsequent ticks)
// - HardLimit regime disables MM quotes even when triggered by delta/basis (not PnL)

use paraphina::config::{Config, RiskProfile};
use paraphina::engine::Engine;
use paraphina::state::{GlobalState, KillReason, RiskRegime};

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
    assert_eq!(h.state.kill_reason, KillReason::None);
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

    // Milestone C: PnL hard breach MUST trigger kill_switch=true
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::PnlHardBreach);
}

#[test]
fn hardlimit_and_kill_switch_when_delta_limit_breached() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Keep PnL and basis clean, breach delta hard limit
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.basis_usd = 0.0;
    h.state.dollar_delta_usd = h.state.delta_limit_usd + 100.0;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Milestone C: Delta hard breach MUST trigger kill_switch=true
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::DeltaHardBreach);
}

#[test]
fn hardlimit_and_kill_switch_when_basis_limit_breached() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Keep PnL and delta clean, breach basis hard limit
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.dollar_delta_usd = 0.0;
    h.state.basis_usd = h.state.basis_limit_hard_usd + 100.0;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Milestone C: Basis hard breach MUST trigger kill_switch=true
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::BasisHardBreach);
}

#[test]
fn hardlimit_and_kill_switch_when_liquidation_distance_breached() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Keep PnL, delta, basis clean
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.dollar_delta_usd = 0.0;
    h.state.basis_usd = 0.0;

    // Set liquidation distance below critical threshold on at least one venue
    // liq_crit_sigma defaults to 2.0 in config
    let crit_sigma = h.cfg.risk.liq_crit_sigma;
    h.state.venues[0].dist_liq_sigma = crit_sigma - 0.5; // below critical

    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Milestone C: Liquidation distance hard breach MUST trigger kill_switch=true
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::LiquidationDistanceBreach);
}

#[test]
fn kill_switch_latches_once_true() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();

    // Trip kill via PnL breach.
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert!(h.state.kill_switch);
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert_eq!(h.state.kill_reason, KillReason::PnlHardBreach);

    // Now "recover" PnL to safe levels.
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Milestone C: Kill switch MUST remain latched (stays true) until manual reset.
    // Risk regime MUST also stay HardLimit when kill is latched.
    assert!(h.state.kill_switch);
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    // Kill reason is preserved from first trigger
    assert_eq!(h.state.kill_reason, KillReason::PnlHardBreach);
}

#[test]
fn kill_switch_latches_preserves_first_reason() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // First: trigger kill via delta breach
    h.state.daily_realised_pnl = 0.0;
    h.state.daily_unrealised_pnl = 0.0;
    h.state.basis_usd = 0.0;
    h.state.dollar_delta_usd = h.state.delta_limit_usd + 100.0;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::DeltaHardBreach);

    // Now also breach PnL - but kill reason should stay as DeltaHardBreach
    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();
    h.state.daily_realised_pnl = loss_limit - 1.0;

    h.engine.update_risk_limits_and_regime(&mut h.state);

    // Kill reason should be preserved from first trigger
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::DeltaHardBreach);
}

#[test]
fn hardlimit_from_delta_breach_triggers_kill_and_disables_mm() {
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

    // Milestone C: HardLimit MUST trigger kill_switch (unlike before)
    assert_eq!(h.state.risk_regime, RiskRegime::HardLimit);
    assert!(h.state.kill_switch);
    assert_eq!(h.state.kill_reason, KillReason::DeltaHardBreach);

    // MM should produce no quotes when kill switch is active.
    let quotes = paraphina::mm::compute_mm_quotes(h.cfg, &h.state);
    assert_eq!(quotes.len(), h.cfg.venues.len());
    for q in quotes {
        assert!(
            q.bid.is_none(),
            "bid should be None when kill_switch is active"
        );
        assert!(
            q.ask.is_none(),
            "ask should be None when kill_switch is active"
        );
    }
}

#[test]
fn hedge_disabled_when_kill_switch_active() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Create some inventory that would normally trigger hedging
    h.state.q_global_tao = 100.0;
    h.state.dollar_delta_usd = 100.0 * h.state.fair_value.unwrap_or(250.0);

    // Verify hedge would normally produce intents
    let _hedge_before = paraphina::hedge::compute_hedge_orders(h.cfg, &h.state, 0);
    // (We don't assert non-empty here since hedge logic has other conditions)

    // Now trigger kill switch via PnL breach
    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert!(h.state.kill_switch);

    // Hedge should produce no orders when kill switch is active
    let hedge_after = paraphina::hedge::compute_hedge_orders(h.cfg, &h.state, 0);
    assert!(
        hedge_after.is_empty(),
        "hedge should produce no orders when kill_switch is active"
    );
}

#[test]
fn exit_disabled_when_kill_switch_active() {
    let mut h = make_harness(RiskProfile::Conservative);

    init_tick(&mut h, 0);

    // Create some position that might trigger exits
    h.state.q_global_tao = 10.0;
    h.state.venues[0].position_tao = 10.0;
    h.state.venues[0].avg_entry_price = 200.0; // Below fair value for profit

    // Trigger kill switch
    let loss_limit = -h.cfg.risk.daily_loss_limit.abs();
    h.state.daily_realised_pnl = loss_limit - 1.0;
    h.engine.update_risk_limits_and_regime(&mut h.state);

    assert!(h.state.kill_switch);

    // Exit should produce no orders when kill switch is active
    let exit_intents = paraphina::exit::compute_exit_intents(h.cfg, &h.state, 1000);
    assert!(
        exit_intents.is_empty(),
        "exit should produce no orders when kill_switch is active"
    );
}
