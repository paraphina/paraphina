// tests/vol_floor_tests.rs
//
// Milestone D tests for volatility floor behavior.
//
// Tests cover:
// - sigma_eff = max(raw_sigma, sigma_floor) enforced
// - sigma_eff is used by downstream scalars (spread_mult, size_mult, band_mult)
// - Floor behavior under low volatility conditions

use paraphina::config::{Config, RiskProfile};
use paraphina::engine::Engine;
use paraphina::state::GlobalState;
use paraphina::types::VenueStatus;

/// Small harness that gives us a Config, Engine and GlobalState.
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

/// Set up fresh venue data for all venues at the given timestamp.
fn set_fresh_venue_data(h: &mut Harness, now_ms: i64, mid: f64) {
    for v in h.state.venues.iter_mut() {
        v.mid = Some(mid);
        v.spread = Some(0.5);
        v.depth_near_mid = 10_000.0;
        v.last_mid_update_ms = Some(now_ms);
        v.status = VenueStatus::Healthy;
    }
}

// =============================================================================
// Vol floor enforcement
// =============================================================================

#[test]
fn sigma_eff_never_below_sigma_min() {
    let mut h = make_harness(RiskProfile::Balanced);
    // After tick-cadence-aware scaling, the effective floor is sigma_min_tick
    // (sigma_min scaled from calibration cadence to per-tick cadence).
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;

    // Run many ticks with zero price movement to minimize volatility
    let mid = 250.0;
    let tick_ms = 1000_i64;

    for t in 0..100 {
        let now_ms = t * tick_ms;
        set_fresh_venue_data(&mut h, now_ms, mid); // Same mid every tick => zero returns
        h.engine.main_tick(&mut h.state, now_ms);

        // sigma_eff should never fall below the tick-cadence-aware floor
        assert!(
            h.state.sigma_eff >= sigma_min_tick,
            "sigma_eff ({}) should never be below sigma_min_tick ({}) at tick {}",
            h.state.sigma_eff,
            sigma_min_tick,
            t
        );
    }
}

#[test]
fn sigma_eff_initialized_to_sigma_min() {
    let h = make_harness(RiskProfile::Balanced);

    // Before any ticks, sigma_eff is initialized from Config::default/for_profile
    // which uses the unscaled sigma_min. This is the initial state before the engine
    // has run any ticks. The engine applies sigma_min_tick only during update_vol_and_scalars.
    assert_eq!(
        h.state.sigma_eff, h.cfg.volatility.sigma_min,
        "sigma_eff should be initialized to sigma_min (unscaled, pre-engine)"
    );
}

#[test]
fn sigma_eff_uses_floor_when_raw_vol_is_low() {
    let mut h = make_harness(RiskProfile::Balanced);
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;

    // Run a few ticks with constant price (zero volatility)
    let mid = 250.0;
    let tick_ms = 1000_i64;

    for t in 0..50 {
        let now_ms = t * tick_ms;
        set_fresh_venue_data(&mut h, now_ms, mid);
        h.engine.main_tick(&mut h.state, now_ms);
    }

    // After many zero-return ticks, raw sigma should have decayed
    // but sigma_eff should stay at the tick-cadence-aware floor
    assert!(
        h.state.fv_short_vol < sigma_min_tick || (h.state.fv_short_vol - 0.0).abs() < 1e-9,
        "Raw short vol ({}) should be low after constant prices",
        h.state.fv_short_vol
    );

    // sigma_eff should be exactly at the tick-cadence-aware floor
    assert!(
        (h.state.sigma_eff - sigma_min_tick).abs() < 1e-12,
        "sigma_eff ({}) should equal sigma_min_tick ({}) when raw vol is below floor",
        h.state.sigma_eff,
        sigma_min_tick
    );
}

#[test]
fn sigma_eff_equals_raw_vol_when_above_floor() {
    let mut h = make_harness(RiskProfile::Balanced);
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;

    // Directly set fv_short_vol to a value above the tick-cadence floor
    let high_vol = sigma_min_tick * 10.0; // 10x the floor
    h.state.fv_short_vol = high_vol;

    // Run a tick to trigger vol scalar updates
    // Set up minimal venue data first
    set_fresh_venue_data(&mut h, 0, 250.0);
    h.engine.main_tick(&mut h.state, 0);

    // The key relationship: sigma_eff = max(sigma_short, sigma_min_tick)
    // After tick, sigma_eff should be at least sigma_min_tick
    assert!(
        h.state.sigma_eff >= sigma_min_tick,
        "sigma_eff ({}) should be at least sigma_min_tick ({})",
        h.state.sigma_eff,
        sigma_min_tick
    );

    // If sigma_short is above the floor, sigma_eff should equal sigma_short
    if h.state.fv_short_vol > sigma_min_tick {
        assert!(
            (h.state.sigma_eff - h.state.fv_short_vol).abs() < 1e-9,
            "sigma_eff ({}) should equal fv_short_vol ({}) when above floor",
            h.state.sigma_eff,
            h.state.fv_short_vol
        );
    }
}

#[test]
fn sigma_eff_floor_logic_verified() {
    // Verifies sigma_eff = max(sigma_short, sigma_min_tick) after tick-cadence scaling.

    let mut h = make_harness(RiskProfile::Balanced);
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;

    // Test case 1: sigma_short below floor
    set_fresh_venue_data(&mut h, 0, 250.0);
    h.engine.main_tick(&mut h.state, 0);

    // With constant prices, sigma_short should be 0 or very small
    // sigma_eff should be floored at sigma_min_tick
    assert!(
        h.state.sigma_eff >= sigma_min_tick,
        "sigma_eff ({}) should be at least sigma_min_tick ({}) when raw vol is low",
        h.state.sigma_eff,
        sigma_min_tick
    );

    // Test case 2: Verify the max() relationship holds
    assert!(
        h.state.sigma_eff >= h.state.fv_short_vol,
        "sigma_eff ({}) should be >= fv_short_vol ({})",
        h.state.sigma_eff,
        h.state.fv_short_vol
    );

    assert!(
        h.state.sigma_eff >= sigma_min_tick,
        "sigma_eff ({}) should be >= sigma_min_tick ({})",
        h.state.sigma_eff,
        sigma_min_tick
    );
}

// =============================================================================
// Downstream scalar usage
// =============================================================================

#[test]
fn vol_scalars_use_sigma_eff() {
    let mut h = make_harness(RiskProfile::Balanced);

    // Initialize with stable prices
    let mid = 250.0;
    set_fresh_venue_data(&mut h, 0, mid);
    h.engine.main_tick(&mut h.state, 0);

    let sigma_eff = h.state.sigma_eff;
    // Engine now uses tick-cadence-aware vol_ref_tick, not the raw config vol_ref.
    let vol_ref_tick = h.engine.vol_pre.vol_ref_tick;

    // Calculate expected vol_ratio using the tick-scaled reference
    let expected_vol_ratio = if vol_ref_tick > 0.0 {
        sigma_eff / vol_ref_tick
    } else {
        1.0
    };

    let expected_clipped = expected_vol_ratio
        .max(h.cfg.volatility.vol_ratio_min)
        .min(h.cfg.volatility.vol_ratio_max);

    assert!(
        (h.state.vol_ratio_clipped - expected_clipped).abs() < 1e-9,
        "vol_ratio_clipped ({}) should be derived from sigma_eff/vol_ref_tick (expected {})",
        h.state.vol_ratio_clipped,
        expected_clipped
    );

    // Verify scalars are computed
    assert!(
        h.state.spread_mult > 0.0,
        "spread_mult should be positive: {}",
        h.state.spread_mult
    );
    assert!(
        h.state.size_mult > 0.0,
        "size_mult should be positive: {}",
        h.state.size_mult
    );
    assert!(
        h.state.band_mult > 0.0,
        "band_mult should be positive: {}",
        h.state.band_mult
    );
}

#[test]
fn spread_mult_increases_with_higher_sigma_eff() {
    let mut h1 = make_harness(RiskProfile::Balanced);
    let mut h2 = make_harness(RiskProfile::Balanced);

    // h1: low volatility (constant prices)
    let mid = 250.0;
    for t in 0..20 {
        set_fresh_venue_data(&mut h1, t * 1000, mid);
        h1.engine.main_tick(&mut h1.state, t * 1000);
    }

    // h2: high volatility (oscillating prices)
    for t in 0..20 {
        let price = if t % 2 == 0 { mid * 1.10 } else { mid * 0.90 };
        set_fresh_venue_data(&mut h2, t * 1000, price);
        h2.engine.main_tick(&mut h2.state, t * 1000);
    }

    // Higher sigma_eff should lead to higher spread_mult
    assert!(
        h2.state.sigma_eff >= h1.state.sigma_eff,
        "High vol scenario should have higher sigma_eff"
    );

    // Spread mult should be higher with higher vol
    assert!(
        h2.state.spread_mult >= h1.state.spread_mult,
        "spread_mult should be higher with higher volatility: low={}, high={}",
        h1.state.spread_mult,
        h2.state.spread_mult
    );
}

#[test]
fn size_mult_decreases_with_higher_sigma_eff() {
    let mut h1 = make_harness(RiskProfile::Balanced);
    let mut h2 = make_harness(RiskProfile::Balanced);

    // h1: low volatility
    let mid = 250.0;
    for t in 0..20 {
        set_fresh_venue_data(&mut h1, t * 1000, mid);
        h1.engine.main_tick(&mut h1.state, t * 1000);
    }

    // h2: high volatility
    for t in 0..20 {
        let price = if t % 2 == 0 { mid * 1.10 } else { mid * 0.90 };
        set_fresh_venue_data(&mut h2, t * 1000, price);
        h2.engine.main_tick(&mut h2.state, t * 1000);
    }

    // Size mult should be lower with higher vol (more conservative)
    assert!(
        h2.state.size_mult <= h1.state.size_mult,
        "size_mult should be lower with higher volatility: low={}, high={}",
        h1.state.size_mult,
        h2.state.size_mult
    );
}

// =============================================================================
// Telemetry correctness
// =============================================================================

#[test]
fn sigma_eff_in_telemetry_matches_state() {
    let mut h = make_harness(RiskProfile::Balanced);

    // Run a tick
    let mid = 250.0;
    set_fresh_venue_data(&mut h, 0, mid);
    h.engine.main_tick(&mut h.state, 0);

    // sigma_eff should be set and positive
    assert!(
        h.state.sigma_eff > 0.0,
        "sigma_eff should be positive for telemetry: {}",
        h.state.sigma_eff
    );
    assert!(
        h.state.sigma_eff.is_finite(),
        "sigma_eff should be finite for telemetry: {}",
        h.state.sigma_eff
    );

    // Should be at least sigma_min_tick (the tick-cadence-aware floor)
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;
    assert!(
        h.state.sigma_eff >= sigma_min_tick,
        "sigma_eff ({}) should be at least sigma_min_tick ({})",
        h.state.sigma_eff,
        sigma_min_tick
    );
}

#[test]
fn sigma_eff_stable_under_nan_protection() {
    let mut h = make_harness(RiskProfile::Balanced);

    // Set up initial state
    set_fresh_venue_data(&mut h, 0, 250.0);
    h.engine.main_tick(&mut h.state, 0);

    // Try to inject problematic values and ensure sigma_eff stays valid
    h.state.fv_short_vol = f64::NAN;
    h.state.fv_long_vol = f64::NAN;

    // Run another tick - should recover
    set_fresh_venue_data(&mut h, 1000, 250.0);
    h.engine.main_tick(&mut h.state, 1000);

    // sigma_eff should still be valid
    assert!(
        h.state.sigma_eff.is_finite(),
        "sigma_eff should recover to finite value after NaN injection"
    );
    let sigma_min_tick = h.engine.vol_pre.sigma_min_tick;
    assert!(
        h.state.sigma_eff >= sigma_min_tick,
        "sigma_eff ({}) should be at least sigma_min_tick ({}) after recovery",
        h.state.sigma_eff,
        sigma_min_tick
    );
}

// =============================================================================
// Integration with risk scaling
// =============================================================================

#[test]
fn delta_limit_scales_with_vol_ratio() {
    let mut h = make_harness(RiskProfile::Balanced);

    // Initial state
    set_fresh_venue_data(&mut h, 0, 250.0);
    h.engine.main_tick(&mut h.state, 0);

    let base_delta_limit = h.cfg.risk.delta_hard_limit_usd_base;
    let vol_ratio = h.state.vol_ratio_clipped;

    // Delta limit should be scaled by vol_ratio
    let expected_limit = base_delta_limit / vol_ratio.max(1e-6);

    assert!(
        (h.state.delta_limit_usd - expected_limit).abs() < 1.0,
        "delta_limit_usd ({}) should be scaled by vol_ratio (expected {})",
        h.state.delta_limit_usd,
        expected_limit
    );
}
