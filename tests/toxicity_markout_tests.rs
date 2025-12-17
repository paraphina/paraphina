// tests/toxicity_markout_tests.rs
//
// Comprehensive tests for the markout-based toxicity system (v2).
//
// These tests verify:
// 1. Adverse markout increases toxicity and can disable a venue.
// 2. Favorable markout does not increase toxicity.
// 3. Pending markouts only apply after horizon time.
// 4. Bounded deque behavior works (max_pending cap).
// 5. Health status transitions based on toxicity thresholds.
// 6. Sell-side markout calculations.
// 7. Multiple markouts accumulate correctly via EWMA.

use paraphina::config::Config;
use paraphina::state::{GlobalState, PendingMarkout};
use paraphina::toxicity::update_toxicity_and_health;
use paraphina::types::{Side, VenueStatus};

/// Helper to create a test config with predictable toxicity settings.
fn make_test_config() -> Config {
    let mut cfg = Config::default();
    cfg.toxicity.tox_med_threshold = 0.5;
    cfg.toxicity.tox_high_threshold = 0.8;
    cfg.toxicity.markout_alpha = 0.5; // 50% blend for easier testing
    cfg.toxicity.markout_scale_usd_per_tao = 1.0; // $1 adverse = tox=1
    cfg.toxicity.markout_horizon_ms = 1000; // 1 second
    cfg.toxicity.max_pending_per_venue = 5;
    cfg
}

/// Helper to setup a venue with valid book data.
fn setup_venue_with_book(state: &mut GlobalState, venue_index: usize, mid: f64) {
    let venue = &mut state.venues[venue_index];
    venue.mid = Some(mid);
    venue.spread = Some(0.1);
    venue.depth_near_mid = 10000.0;
    venue.last_mid_update_ms = Some(0);
}

// =============================================================================
// Test 1: Adverse markout increases toxicity and can disable a venue
// =============================================================================

#[test]
fn test_adverse_buy_markout_increases_toxicity() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 98.0); // Price dropped after buy
    state.venues[0].toxicity = 0.0;

    // Add a pending markout for a BUY at 100, now mid is 98 (adverse)
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Adverse markout = mid_now - price = 98 - 100 = -2
    // tox_instant = clamp(2 / 1, 0, 1) = 1.0
    // tox = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    assert!(
        state.venues[0].toxicity >= 0.4,
        "Adverse markout should increase toxicity significantly, got {}",
        state.venues[0].toxicity
    );
}

#[test]
fn test_severe_adverse_markout_disables_venue() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 90.0); // Price crashed
    state.venues[0].toxicity = 0.7; // Already elevated

    // Add very adverse markout
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Adverse markout = 90 - 100 = -10, tox_instant = 1.0
    // tox = 0.5 * 0.7 + 0.5 * 1.0 = 0.85
    assert!(
        state.venues[0].toxicity >= cfg.toxicity.tox_high_threshold,
        "Severe adverse markout should push toxicity above high threshold"
    );
    assert_eq!(
        state.venues[0].status,
        VenueStatus::Disabled,
        "Venue should be Disabled when toxicity >= tox_high_threshold"
    );
}

// =============================================================================
// Test 2: Favorable markout does not increase toxicity
// =============================================================================

#[test]
fn test_favorable_buy_markout_decreases_toxicity() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 102.0); // Price went UP after buy
    state.venues[0].toxicity = 0.4; // Start with some toxicity

    // Add a pending markout for a BUY at 100, now mid is 102 (favorable!)
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Favorable markout = mid_now - price = 102 - 100 = +2
    // tox_instant = 0 (favorable)
    // tox = 0.5 * 0.4 + 0.5 * 0 = 0.2
    assert!(
        state.venues[0].toxicity < 0.4,
        "Favorable markout should decrease toxicity, got {}",
        state.venues[0].toxicity
    );
    assert!(
        (state.venues[0].toxicity - 0.2).abs() < 0.05,
        "Expected toxicity ~0.2, got {}",
        state.venues[0].toxicity
    );
}

#[test]
fn test_favorable_sell_markout_keeps_toxicity_low() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 98.0); // Price went DOWN after sell (good!)
    state.venues[0].toxicity = 0.0;

    // Add a pending markout for a SELL at 100, now mid is 98 (favorable!)
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Sell,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Favorable sell markout = price - mid_now = 100 - 98 = +2
    // tox_instant = 0 (favorable)
    assert!(
        state.venues[0].toxicity < 0.1,
        "Favorable sell markout should keep toxicity low, got {}",
        state.venues[0].toxicity
    );
}

// =============================================================================
// Test 3: Pending markouts only apply after horizon time
// =============================================================================

#[test]
fn test_markout_not_applied_before_horizon() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 90.0); // Very adverse
    state.venues[0].toxicity = 0.0;

    // Add pending markout that evaluates at t=1000
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    // Run at t=500 (before horizon)
    update_toxicity_and_health(&mut state, &cfg, 500);

    // Toxicity should remain very low (no markout processed)
    assert!(
        state.venues[0].toxicity < 0.1,
        "Markout should not apply before horizon, got {}",
        state.venues[0].toxicity
    );

    // Markout should still be pending
    assert_eq!(
        state.venues[0].pending_markouts.len(),
        1,
        "Markout should remain in queue before horizon"
    );
}

#[test]
fn test_markout_applied_at_exact_horizon() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 90.0);
    state.venues[0].toxicity = 0.0;

    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    // Run at exactly t=1000
    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Markout should have been processed
    assert_eq!(
        state.venues[0].pending_markouts.len(),
        0,
        "Markout should be consumed at horizon"
    );
    assert!(
        state.venues[0].toxicity > 0.3,
        "Toxicity should have increased after markout evaluation"
    );
}

#[test]
fn test_markout_applied_after_horizon() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 90.0);
    state.venues[0].toxicity = 0.0;

    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    // Run at t=2000 (well after horizon)
    update_toxicity_and_health(&mut state, &cfg, 2000);

    // Markout should have been processed
    assert_eq!(state.venues[0].pending_markouts.len(), 0);
    assert!(state.venues[0].toxicity > 0.3);
}

// =============================================================================
// Test 4: Bounded deque behavior (max_pending cap)
// =============================================================================

#[test]
fn test_max_pending_cap_enforced() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 100.0);

    // Add more markouts than max_pending (5)
    for i in 0..10 {
        state.record_pending_markout(
            0,
            Side::Buy,
            1.0,
            100.0,
            i * 100,
            100.0,
            100.0,
            cfg.toxicity.markout_horizon_ms,
            cfg.toxicity.max_pending_per_venue,
        );
    }

    // Should be capped at max_pending
    assert_eq!(
        state.venues[0].pending_markouts.len(),
        cfg.toxicity.max_pending_per_venue,
        "Queue should be capped at max_pending_per_venue"
    );

    // Oldest entries should have been dropped (FIFO)
    // The first entry should have t_fill_ms = 500 (entries 0-4 dropped, 5-9 remain)
    let first = state.venues[0].pending_markouts.front().unwrap();
    assert_eq!(
        first.t_fill_ms, 500,
        "Oldest entries should be dropped when cap exceeded"
    );
}

#[test]
fn test_bounded_deque_fifo_order() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 100.0);

    // Add entries with different timestamps
    for i in 0..3 {
        state.record_pending_markout(
            0,
            Side::Buy,
            1.0,
            100.0 + (i as f64),
            i * 1000,
            100.0,
            100.0,
            cfg.toxicity.markout_horizon_ms,
            cfg.toxicity.max_pending_per_venue,
        );
    }

    // Verify FIFO order
    let prices: Vec<f64> = state.venues[0]
        .pending_markouts
        .iter()
        .map(|pm| pm.price)
        .collect();
    assert_eq!(prices, vec![100.0, 101.0, 102.0]);
}

// =============================================================================
// Test 5: Health status transitions
// =============================================================================

#[test]
fn test_health_status_healthy() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 100.0);
    state.venues[0].toxicity = 0.3; // Below tox_med_threshold (0.5)

    update_toxicity_and_health(&mut state, &cfg, 0);

    assert_eq!(state.venues[0].status, VenueStatus::Healthy);
}

#[test]
fn test_health_status_warning() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 95.0);
    state.venues[0].toxicity = 0.3;

    // Add markout that will push toxicity into warning zone
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 0,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 0);

    // tox = 0.5 * 0.3 + 0.5 * 1.0 = 0.65 (> 0.5, < 0.8)
    assert_eq!(
        state.venues[0].status,
        VenueStatus::Warning,
        "Venue should be Warning when tox_med <= toxicity < tox_high"
    );
}

#[test]
fn test_health_status_disabled() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 90.0);
    state.venues[0].toxicity = 0.7;

    // Add markout that will push toxicity above high threshold
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 0,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 0);

    // tox = 0.5 * 0.7 + 0.5 * 1.0 = 0.85 (>= 0.8)
    assert_eq!(
        state.venues[0].status,
        VenueStatus::Disabled,
        "Venue should be Disabled when toxicity >= tox_high"
    );
}

// =============================================================================
// Test 6: Sell-side markout calculations
// =============================================================================

#[test]
fn test_adverse_sell_markout() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 105.0); // Price went UP after sell (bad!)
    state.venues[0].toxicity = 0.0;

    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 1000,
        side: Side::Sell,
        size_tao: 1.0,
        price: 100.0, // Sold at 100
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 1000);

    // Adverse sell markout = price - mid_now = 100 - 105 = -5
    // tox_instant = clamp(5 / 1, 0, 1) = 1.0
    // tox = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    assert!(
        state.venues[0].toxicity >= 0.4,
        "Adverse sell markout should increase toxicity, got {}",
        state.venues[0].toxicity
    );
}

// =============================================================================
// Test 7: Multiple markouts accumulate correctly via EWMA
// =============================================================================

#[test]
fn test_multiple_markouts_ewma_accumulation() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 98.0);
    state.venues[0].toxicity = 0.0;

    // Add multiple adverse markouts (all ready to evaluate)
    for _ in 0..3 {
        state.venues[0].pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 0,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0,
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
        });
    }

    update_toxicity_and_health(&mut state, &cfg, 0);

    // Each markout has tox_instant = 1.0 (due to -2 markout)
    // EWMA: tox = 0.5 * tox + 0.5 * 1.0 applied 3 times
    // tox_1 = 0.5 * 0 + 0.5 * 1 = 0.5
    // tox_2 = 0.5 * 0.5 + 0.5 * 1 = 0.75
    // tox_3 = 0.5 * 0.75 + 0.5 * 1 = 0.875
    assert!(
        state.venues[0].toxicity > 0.8,
        "Multiple adverse markouts should accumulate via EWMA, got {}",
        state.venues[0].toxicity
    );
}

#[test]
fn test_mixed_markouts_balance_out() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 100.0);
    state.venues[0].toxicity = 0.5;

    // Add one adverse and one favorable markout
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 0,
        side: Side::Buy,
        size_tao: 1.0,
        price: 102.0, // Adverse: bought at 102, now at 100
        fair_at_fill: 102.0,
        mid_at_fill: 102.0,
    });

    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 0,
        side: Side::Buy,
        size_tao: 1.0,
        price: 98.0, // Favorable: bought at 98, now at 100
        fair_at_fill: 98.0,
        mid_at_fill: 98.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 0);

    // First markout: adverse -2, tox_instant = 1.0
    // tox_1 = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    // Second markout: favorable +2, tox_instant = 0.0
    // tox_2 = 0.5 * 0.75 + 0.5 * 0.0 = 0.375

    // The favorable markout should bring toxicity back down
    assert!(
        state.venues[0].toxicity < 0.5,
        "Mixed markouts should balance, got {}",
        state.venues[0].toxicity
    );
}

// =============================================================================
// Test 8: Markout EWMA telemetry field
// =============================================================================

#[test]
fn test_markout_ewma_telemetry_updated() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    setup_venue_with_book(&mut state, 0, 98.0);
    state.venues[0].markout_ewma_usd_per_tao = 0.0;

    // Add adverse markout
    state.venues[0].pending_markouts.push_back(PendingMarkout {
        t_fill_ms: 0,
        t_eval_ms: 0,
        side: Side::Buy,
        size_tao: 1.0,
        price: 100.0,
        fair_at_fill: 100.0,
        mid_at_fill: 100.0,
    });

    update_toxicity_and_health(&mut state, &cfg, 0);

    // markout = 98 - 100 = -2
    // ewma = 0.5 * 0.0 + 0.5 * (-2.0) = -1.0
    assert!(
        (state.venues[0].markout_ewma_usd_per_tao - (-1.0)).abs() < 0.01,
        "Markout EWMA should track actual markout values, got {}",
        state.venues[0].markout_ewma_usd_per_tao
    );
}

// =============================================================================
// Test 9: Edge cases
// =============================================================================

#[test]
fn test_no_mid_sets_high_toxicity() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    // Venue with no mid price
    state.venues[0].mid = None;
    state.venues[0].toxicity = 0.0;

    update_toxicity_and_health(&mut state, &cfg, 0);

    assert_eq!(
        state.venues[0].toxicity, 1.0,
        "Missing mid should result in max toxicity"
    );
}

#[test]
fn test_zero_depth_sets_high_toxicity() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    state.venues[0].mid = Some(100.0);
    state.venues[0].depth_near_mid = 0.0;
    state.venues[0].toxicity = 0.0;

    update_toxicity_and_health(&mut state, &cfg, 0);

    assert_eq!(
        state.venues[0].toxicity, 1.0,
        "Zero depth should result in max toxicity"
    );
}

#[test]
fn test_invalid_fill_not_recorded() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    // Try to record invalid fills
    state.record_pending_markout(
        0,
        Side::Buy,
        0.0, // Zero size
        100.0,
        0,
        100.0,
        100.0,
        cfg.toxicity.markout_horizon_ms,
        cfg.toxicity.max_pending_per_venue,
    );

    state.record_pending_markout(
        0,
        Side::Buy,
        1.0,
        -100.0, // Negative price
        0,
        100.0,
        100.0,
        cfg.toxicity.markout_horizon_ms,
        cfg.toxicity.max_pending_per_venue,
    );

    state.record_pending_markout(
        999, // Invalid venue index
        Side::Buy,
        1.0,
        100.0,
        0,
        100.0,
        100.0,
        cfg.toxicity.markout_horizon_ms,
        cfg.toxicity.max_pending_per_venue,
    );

    // No markouts should have been recorded
    assert_eq!(
        state.venues[0].pending_markouts.len(),
        0,
        "Invalid fills should not create pending markouts"
    );
}

// =============================================================================
// Test 10: Integration with record_pending_markout
// =============================================================================

#[test]
fn test_record_pending_markout_basic() {
    let cfg = make_test_config();
    let mut state = GlobalState::new(&cfg);

    state.record_pending_markout(
        0,
        Side::Buy,
        5.0,
        100.0,
        1000,  // now_ms
        99.0,  // fair
        100.0, // mid
        cfg.toxicity.markout_horizon_ms,
        cfg.toxicity.max_pending_per_venue,
    );

    assert_eq!(state.venues[0].pending_markouts.len(), 1);

    let pm = &state.venues[0].pending_markouts[0];
    assert_eq!(pm.t_fill_ms, 1000);
    assert_eq!(pm.t_eval_ms, 1000 + cfg.toxicity.markout_horizon_ms);
    assert_eq!(pm.side, Side::Buy);
    assert_eq!(pm.size_tao, 5.0);
    assert_eq!(pm.price, 100.0);
    assert_eq!(pm.fair_at_fill, 99.0);
    assert_eq!(pm.mid_at_fill, 100.0);
}
