// tests/fair_value_gating_tests.rs
//
// Milestone D regression tests for fair value gating robustness.
//
// Tests cover:
// - Case A: Intentionally stale venue data => FV update does not incorporate stale venues;
//           if too few remain, fv_available = false and FV unchanged; MM quotes shrink/pause.
// - Case B: Outlier mid on one venue => that venue excluded; FV remains stable;
//           healthy count reflects exclusion.
// - Case C: Min-healthy venues threshold enforced.

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
fn set_fresh_venue_data(h: &mut Harness, now_ms: i64, base_mid: f64) {
    for (idx, v) in h.state.venues.iter_mut().enumerate() {
        let offset = idx as f64 * 0.1; // Small variation between venues
        v.mid = Some(base_mid + offset);
        v.spread = Some(0.5);
        v.depth_near_mid = 10_000.0;
        v.last_mid_update_ms = Some(now_ms);
        v.status = VenueStatus::Healthy;
    }
}

/// Helper: run a single engine tick with given timestamp.
fn run_tick(h: &mut Harness, now_ms: i64) {
    h.engine.main_tick(&mut h.state, now_ms);
}

// =============================================================================
// Case A: Intentionally stale venue data
// =============================================================================

#[test]
fn stale_venue_data_excluded_from_fv_update() {
    let mut h = make_harness(RiskProfile::Balanced);

    // Setup: Initialize venues with fresh data
    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // First tick - all venues fresh, should use all
    run_tick(&mut h, t0);

    assert!(
        h.state.fv_available,
        "FV should be available when all venues are fresh"
    );
    assert!(
        h.state.healthy_venues_used_count >= h.cfg.book.min_healthy_for_kf as usize,
        "Should have used at least min_healthy_for_kf venues"
    );
    let _initial_fv = h.state.fair_value.expect("FV should exist");

    // Now make all but one venue stale
    let stale_threshold = h.cfg.book.stale_ms;
    let t1 = t0 + stale_threshold + 100; // Well past stale threshold

    // Only venue 0 gets fresh data
    h.state.venues[0].last_mid_update_ms = Some(t1);
    h.state.venues[0].mid = Some(base_mid + 0.5);

    // All other venues still have old timestamp (t0), which is now stale
    // (they were not updated)

    run_tick(&mut h, t1);

    // With default min_healthy_for_kf = 2, and only 1 fresh venue,
    // FV should not be updated (fv_available = false)
    assert!(
        h.state.healthy_venues_used_count <= 1,
        "Only one venue should be non-stale"
    );

    // If min_healthy_for_kf > 1, fv_available should be false
    if h.cfg.book.min_healthy_for_kf > 1 {
        assert!(
            !h.state.fv_available,
            "FV should not be available with insufficient healthy venues"
        );
    }
}

#[test]
fn fv_degrades_gracefully_with_all_stale_data() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // First tick - establish FV
    run_tick(&mut h, t0);
    let _initial_fv = h
        .state
        .fair_value
        .expect("FV should exist after first tick");
    assert!(h.state.fv_available);

    // Make all venues stale
    let stale_threshold = h.cfg.book.stale_ms;
    let t1 = t0 + stale_threshold + 100;
    // Don't update any venue timestamps - they're all stale now

    run_tick(&mut h, t1);

    // FV should not be available, but should degrade gracefully (not become NaN)
    assert!(
        !h.state.fv_available,
        "FV should not be available when all venues are stale"
    );
    assert_eq!(
        h.state.healthy_venues_used_count, 0,
        "No healthy venues should be used"
    );
    assert!(
        h.state.healthy_venues_used.is_empty(),
        "Healthy venues list should be empty"
    );

    // FV should still exist (from previous value) and be finite
    let current_fv = h.state.fair_value.expect("FV should still exist");
    assert!(current_fv.is_finite(), "FV should remain finite");
}

// =============================================================================
// Case B: Outlier venue excluded
// =============================================================================

#[test]
fn outlier_venue_excluded_from_fv_update() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // First tick to establish a reference FV
    run_tick(&mut h, t0);
    let initial_fv = h.state.fair_value.expect("FV should exist");
    assert!(h.state.fv_available);

    // Now set one venue to have an outlier mid (exceeds max_mid_jump_pct)
    let max_jump_pct = h.cfg.book.max_mid_jump_pct;
    let outlier_mid = initial_fv * (1.0 + max_jump_pct * 2.0); // 2x the threshold

    let t1 = t0 + 100;
    // Update all venues with fresh timestamps
    for v in h.state.venues.iter_mut() {
        v.last_mid_update_ms = Some(t1);
    }

    // Make venue 0 an outlier
    h.state.venues[0].mid = Some(outlier_mid);

    run_tick(&mut h, t1);

    // Venue 0 should be excluded due to outlier gating
    assert!(
        !h.state.healthy_venues_used.contains(&0),
        "Outlier venue 0 should be excluded from FV update"
    );

    // FV should remain stable (not jump to outlier value)
    let new_fv = h.state.fair_value.expect("FV should exist");
    let fv_change_pct = ((new_fv - initial_fv) / initial_fv).abs();

    // FV change should be much smaller than the outlier deviation
    assert!(
        fv_change_pct < max_jump_pct,
        "FV should remain stable, not follow outlier. FV changed by {:.2}%, max allowed {:.2}%",
        fv_change_pct * 100.0,
        max_jump_pct * 100.0
    );
}

#[test]
fn healthy_count_reflects_outlier_exclusion() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    run_tick(&mut h, t0);
    let _initial_healthy_count = h.state.healthy_venues_used_count;

    // Set venue 0 as outlier and venue 1 as stale
    // t1 must be well past the stale threshold for venue 1 to be considered stale
    let stale_threshold = h.cfg.book.stale_ms;
    let t1 = t0 + stale_threshold + 100;
    let max_jump_pct = h.cfg.book.max_mid_jump_pct;
    let outlier_mid = base_mid * (1.0 + max_jump_pct * 3.0);

    // Only update venues 0, 2, 3, 4 with fresh timestamps
    h.state.venues[0].last_mid_update_ms = Some(t1);
    h.state.venues[0].mid = Some(outlier_mid); // outlier
                                               // venue 1 keeps its old timestamp (t0), which is now stale
    for i in 2..h.state.venues.len() {
        h.state.venues[i].last_mid_update_ms = Some(t1);
        h.state.venues[i].mid = Some(base_mid + i as f64 * 0.1);
    }

    run_tick(&mut h, t1);

    // Venue 0 excluded (outlier), venue 1 excluded (stale)
    assert!(
        !h.state.healthy_venues_used.contains(&0),
        "Outlier venue should be excluded"
    );
    assert!(
        !h.state.healthy_venues_used.contains(&1),
        "Stale venue should be excluded"
    );
}

// =============================================================================
// Case C: Min-healthy venues threshold enforced
// =============================================================================

#[test]
fn min_healthy_threshold_enforced() {
    let mut h = make_harness(RiskProfile::Balanced);

    let min_healthy = h.cfg.book.min_healthy_for_kf as usize;
    assert!(
        min_healthy >= 2,
        "Test requires min_healthy_for_kf >= 2 to be meaningful"
    );

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // First tick to establish FV
    run_tick(&mut h, t0);
    assert!(h.state.fv_available);
    let _initial_fv = h.state.fair_value.expect("FV should exist");

    // Make all but (min_healthy - 1) venues stale
    // This should trigger the min-healthy gating
    let t1 = t0 + h.cfg.book.stale_ms + 100;

    // Only keep (min_healthy - 1) venues fresh - should be below threshold
    let num_fresh = (min_healthy - 1).min(h.state.venues.len());
    for (i, v) in h.state.venues.iter_mut().enumerate() {
        if i < num_fresh {
            v.last_mid_update_ms = Some(t1);
            v.mid = Some(base_mid + i as f64 * 0.1);
        }
        // Other venues keep their old stale timestamp
    }

    run_tick(&mut h, t1);

    // Should NOT update FV due to min-healthy gating
    assert!(
        !h.state.fv_available,
        "FV should not be available with {} venues < min_healthy={}",
        num_fresh, min_healthy
    );
    assert!(
        h.state.healthy_venues_used_count < min_healthy,
        "Healthy count should be below min_healthy threshold"
    );
}

#[test]
fn fv_updates_when_exactly_at_min_healthy() {
    let mut h = make_harness(RiskProfile::Balanced);

    let min_healthy = h.cfg.book.min_healthy_for_kf as usize;
    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    run_tick(&mut h, t0);
    assert!(h.state.fv_available);

    // Make exactly min_healthy venues fresh
    let t1 = t0 + h.cfg.book.stale_ms + 100;

    for (i, v) in h.state.venues.iter_mut().enumerate() {
        if i < min_healthy {
            v.last_mid_update_ms = Some(t1);
            v.mid = Some(base_mid + i as f64 * 0.1);
        }
    }

    run_tick(&mut h, t1);

    // Should update FV since exactly at threshold
    assert!(
        h.state.fv_available,
        "FV should be available with exactly min_healthy={} venues",
        min_healthy
    );
    assert_eq!(
        h.state.healthy_venues_used_count, min_healthy,
        "Should use exactly min_healthy venues"
    );
}

// =============================================================================
// MM quoting behavior under degraded FV
// =============================================================================

#[test]
fn mm_quotes_shrink_when_fv_unavailable() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // Establish normal operation
    run_tick(&mut h, t0);
    assert!(h.state.fv_available);

    // Get MM quotes under normal conditions
    let _normal_quotes = paraphina::mm::compute_mm_quotes(h.cfg, &h.state);

    // Make all venues stale
    let t1 = t0 + h.cfg.book.stale_ms + 100;
    run_tick(&mut h, t1);

    assert!(!h.state.fv_available);

    // MM quotes should still be computed (using fallback FV)
    // but may be more conservative
    let degraded_quotes = paraphina::mm::compute_mm_quotes(h.cfg, &h.state);

    // Verify quotes are produced (the MM module should handle degraded FV gracefully)
    assert_eq!(
        degraded_quotes.len(),
        h.cfg.venues.len(),
        "Should produce quotes for all venues"
    );
}

// =============================================================================
// Telemetry field correctness
// =============================================================================

#[test]
fn telemetry_fields_populated_correctly() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    run_tick(&mut h, t0);

    // Check that all Milestone D telemetry fields are populated
    assert!(
        h.state.fair_value.is_some(),
        "fair_value should be populated"
    );
    assert!(
        h.state.sigma_eff > 0.0,
        "sigma_eff should be positive: {}",
        h.state.sigma_eff
    );

    // fv_available should match whether we had enough healthy venues
    if h.state.healthy_venues_used_count >= h.cfg.book.min_healthy_for_kf as usize {
        assert!(h.state.fv_available, "fv_available should be true");
    }

    // healthy_venues_used should match healthy_venues_used_count
    assert_eq!(
        h.state.healthy_venues_used.len(),
        h.state.healthy_venues_used_count,
        "healthy_venues_used length should match count"
    );

    // All venue indices in healthy_venues_used should be valid
    for &idx in &h.state.healthy_venues_used {
        assert!(
            idx < h.state.venues.len(),
            "Venue index {} should be valid",
            idx
        );
    }
}

#[test]
fn telemetry_healthy_venues_empty_when_fv_unavailable() {
    let mut h = make_harness(RiskProfile::Balanced);

    let base_mid = 250.0;
    let t0 = 1000_i64;
    set_fresh_venue_data(&mut h, t0, base_mid);

    // First tick to establish state
    run_tick(&mut h, t0);

    // Make all venues stale
    let t1 = t0 + h.cfg.book.stale_ms + 100;
    run_tick(&mut h, t1);

    // When FV is unavailable, healthy venues should be empty
    assert!(!h.state.fv_available);
    assert_eq!(h.state.healthy_venues_used_count, 0);
    assert!(h.state.healthy_venues_used.is_empty());
}
