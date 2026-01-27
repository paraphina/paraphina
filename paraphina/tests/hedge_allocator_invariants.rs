// tests/hedge_allocator_invariants.rs
//
// Deterministic invariant tests for the hedge allocator.
// These tests explore a wide range of parameter combinations and assert
// hard safety invariants that MUST always hold for hedge allocation.
//
// Invariants enforced:
// 1) No NaNs/inf in any computed metrics or outputs.
// 2) Total requested hedge size is bounded by max_step_tao.
// 3) Per-venue allocation respects per-venue cap (min of max_venue_tao_per_tick,
//    venue.max_order_size, depth_fraction * depth / fair).
// 4) When margin_available_usd is the binding constraint, allocated additional
//    abs exposure must not exceed the margin-derived cap.
// 5) Hedging direction must not increase absolute inventory exposure (if global
//    position is long, hedge must net sell; if short, net buy).
// 6) Output ordering is stable and deterministic (same inputs => same ordered intents).

#[path = "hedge_testkit.rs"]
mod hedge_testkit;

use hedge_testkit::{approx_eq, is_finite, HedgeTestCase, Xorshift64};
use paraphina::hedge::{compute_abs_limit_after_trade, compute_hedge_orders, compute_hedge_plan};
use paraphina::types::OrderIntent;
use paraphina::types::Side;

/// Extractors for OrderIntent enum used in hedge allocator invariants.
/// Hedge allocator should emit Place/Replace intents only in these invariants.
fn intent_place_like_fields(intent: &OrderIntent) -> (usize, Side, f64, f64) {
    match intent {
        OrderIntent::Place(pi) => (pi.venue_index, pi.side, pi.size, pi.price),
        OrderIntent::Replace(ri) => (ri.venue_index, ri.side, ri.size, ri.price),
        other => panic!("Expected Place/Replace OrderIntent, got: {:?}", other),
    }
}
fn intent_venue_index(intent: &OrderIntent) -> usize {
    intent_place_like_fields(intent).0
}
fn intent_side(intent: &OrderIntent) -> Side {
    intent_place_like_fields(intent).1
}
fn intent_size(intent: &OrderIntent) -> f64 {
    intent_place_like_fields(intent).2
}
fn intent_price(intent: &OrderIntent) -> f64 {
    intent_place_like_fields(intent).3
}

// ============================================================================
// CONSTANTS
// ============================================================================

/// Fixed seed for deterministic testing.
const SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Number of random test cases to generate.
/// Keep this small for fast CI (<1s target).
const NUM_RANDOM_CASES: u64 = 200;

// ============================================================================
// INVARIANT 1: No NaNs/Infs
// ============================================================================

/// Assert that all outputs are finite (no NaN or infinity).
#[test]
fn invariant_no_nans_or_infs_in_outputs() {
    let mut rng = Xorshift64::new(SEED);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        for intent in &intents {
            assert!(
                is_finite(intent_size(intent)),
                "Case {case_id}: Intent size is NaN/inf: {:?}\nTest case: {test_case:?}",
                intent_size(intent)
            );
            assert!(
                is_finite(intent_price(intent)),
                "Case {case_id}: Intent price is NaN/inf: {:?}\nTest case: {test_case:?}",
                intent_price(intent)
            );
            assert!(
                intent_size(intent) >= 0.0,
                "Case {case_id}: Intent size is negative: {}\nTest case: {test_case:?}",
                intent_size(intent)
            );
            assert!(
                intent_price(intent) > 0.0,
                "Case {case_id}: Intent price is non-positive: {}\nTest case: {test_case:?}",
                intent_price(intent)
            );
        }

        // Also check the plan's desired_delta if present
        if let Some(plan) = compute_hedge_plan(&cfg, &state, 0) {
            assert!(
                is_finite(plan.desired_delta),
                "Case {case_id}: Plan desired_delta is NaN/inf: {:?}\nTest case: {test_case:?}",
                plan.desired_delta
            );
        }
    }
}

// ============================================================================
// INVARIANT 2: Total Hedge Size <= max_step_tao
// ============================================================================

/// Assert that total hedge size never exceeds max_step_tao.
#[test]
fn invariant_total_hedge_bounded_by_max_step() {
    let mut rng = Xorshift64::new(SEED + 1);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);
        let total_size: f64 = intents.iter().map(intent_size).sum();

        // Allow small floating-point tolerance
        let tolerance = 1e-6;
        assert!(
            total_size <= cfg.hedge.max_step_tao + tolerance,
            "Case {case_id}: Total hedge size ({total_size}) exceeds max_step_tao ({})\n\
             Intents: {intents:?}\nTest case: {test_case:?}",
            cfg.hedge.max_step_tao
        );
    }
}

// ============================================================================
// INVARIANT 3: Per-Venue Cap Respected
// ============================================================================

/// Assert that each venue's allocation respects its per-venue cap.
#[test]
fn invariant_per_venue_cap_respected() {
    let mut rng = Xorshift64::new(SEED + 2);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        for intent in &intents {
            let vcfg = &cfg.venues[intent_venue_index(intent)];
            let v = &state.venues[intent_venue_index(intent)];

            // Compute the per-venue cap
            let fair = state.fair_value.unwrap_or(100.0);
            let depth_cap = cfg.hedge.depth_fraction * v.depth_near_mid / fair;
            let max_venue = cfg.hedge.max_venue_tao_per_tick;
            let max_order = vcfg.max_order_size;
            let expected_cap = max_venue.min(max_order).min(depth_cap);

            // Allow tolerance for floating-point and lot-size rounding
            let tolerance = 1e-6;
            assert!(
                intent_size(intent) <= expected_cap + tolerance,
                "Case {case_id}: Venue {} allocation ({}) exceeds per-venue cap ({})\n\
                 max_venue={max_venue}, max_order={max_order}, depth_cap={depth_cap}\n\
                 Test case: {test_case:?}",
                intent_venue_index(intent),
                intent_size(intent),
                expected_cap
            );
        }
    }
}

// ============================================================================
// INVARIANT 4: Margin Constraint Respected
// ============================================================================

/// Assert that margin constraints are respected when increasing exposure.
#[test]
fn invariant_margin_constraint_respected() {
    let mut rng = Xorshift64::new(SEED + 3);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        for intent in &intents {
            let v = &state.venues[intent_venue_index(intent)];
            let fair = state.fair_value.unwrap_or(100.0);

            // Compute the margin-based abs limit
            let abs_limit = compute_abs_limit_after_trade(
                v.position_tao,
                v.margin_available_usd,
                cfg.hedge.max_leverage,
                cfg.hedge.margin_safety_buffer,
                fair,
            );

            // Compute the new position after this intent
            let dq = match intent_side(intent) {
                Side::Buy => intent_size(intent),
                Side::Sell => -intent_size(intent),
            };
            let new_position = v.position_tao + dq;

            // If this trade increases absolute exposure, check the limit
            let old_abs = v.position_tao.abs();
            let new_abs = new_position.abs();

            if new_abs > old_abs {
                // Margin constraint should be respected
                let tolerance = 1e-6;
                assert!(
                    new_abs <= abs_limit + tolerance,
                    "Case {case_id}: Venue {} new abs position ({new_abs}) exceeds margin limit ({abs_limit})\n\
                     old_position={}, dq={dq}, margin_available={}, leverage={}, safety={}\n\
                     Test case: {test_case:?}",
                    intent_venue_index(intent),
                    v.position_tao,
                    v.margin_available_usd,
                    cfg.hedge.max_leverage,
                    cfg.hedge.margin_safety_buffer
                );
            }
        }
    }
}

// ============================================================================
// INVARIANT 5: Hedge Direction Correct
// ============================================================================

/// Assert that hedge direction reduces global inventory exposure.
/// If global position is long, hedge must net sell; if short, net buy.
#[test]
fn invariant_hedge_direction_reduces_exposure() {
    let mut rng = Xorshift64::new(SEED + 4);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        if intents.is_empty() {
            // No hedge = deadband or other gating, which is fine
            continue;
        }

        // All intents should be the same side
        let expected_side = if state.q_global_tao > 0.0 {
            Side::Sell // Long position -> sell to reduce
        } else {
            Side::Buy // Short position -> buy to cover
        };

        for intent in &intents {
            assert_eq!(
                intent_side(intent),
                expected_side,
                "Case {case_id}: Hedge direction wrong. Global q={}, expected {:?}, got {:?}\n\
                 Test case: {test_case:?}",
                state.q_global_tao,
                expected_side,
                intent_side(intent)
            );
        }
    }
}

// ============================================================================
// INVARIANT 6: Output Determinism
// ============================================================================

/// Assert that same inputs produce identical outputs across multiple runs.
#[test]
fn invariant_output_determinism() {
    let mut rng = Xorshift64::new(SEED + 5);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        // Run allocator multiple times with identical inputs
        let intents1 = compute_hedge_orders(&cfg, &state, 0);
        let intents2 = compute_hedge_orders(&cfg, &state, 0);
        let intents3 = compute_hedge_orders(&cfg, &state, 0);

        // Verify identical length
        assert_eq!(
            intents1.len(),
            intents2.len(),
            "Case {case_id}: Non-deterministic number of intents (run1={}, run2={})\n\
             Test case: {test_case:?}",
            intents1.len(),
            intents2.len()
        );
        assert_eq!(
            intents1.len(),
            intents3.len(),
            "Case {case_id}: Non-deterministic number of intents (run1={}, run3={})\n\
             Test case: {test_case:?}",
            intents1.len(),
            intents3.len()
        );

        // Verify identical contents
        for i in 0..intents1.len() {
            assert_eq!(
                intent_venue_index(&intents1[i]),
                intent_venue_index(&intents2[i]),
                "Case {case_id}: Non-deterministic venue order at position {i}\n\
                 Test case: {test_case:?}"
            );
            assert_eq!(
                intent_side(&intents1[i]),
                intent_side(&intents2[i]),
                "Case {case_id}: Non-deterministic side at position {i}\n\
                 Test case: {test_case:?}"
            );
            assert!(
                approx_eq(intent_size(&intents1[i]), intent_size(&intents2[i]), 1e-12),
                "Case {case_id}: Non-deterministic size at position {i}: {} vs {}\n\
                 Test case: {test_case:?}",
                intent_size(&intents1[i]),
                intent_size(&intents2[i])
            );
            assert!(
                approx_eq(
                    intent_price(&intents1[i]),
                    intent_price(&intents2[i]),
                    1e-12
                ),
                "Case {case_id}: Non-deterministic price at position {i}: {} vs {}\n\
                 Test case: {test_case:?}",
                intent_price(&intents1[i]),
                intent_price(&intents2[i])
            );
        }
    }
}

// ============================================================================
// SYSTEMATIC ENUMERATION TESTS
// ============================================================================

/// Systematically test critical parameter combinations.
#[test]
fn systematic_parameter_combinations() {
    // Test grid for key parameters
    let q_values = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let fair_values = [50.0, 100.0, 200.0];
    let margin_values = [0.0, 100.0, 10000.0];
    let leverage_values = [1.0, 10.0];

    let mut case_count = 0;

    for &q in &q_values {
        for &fair in &fair_values {
            for &margin in &margin_values {
                for &leverage in &leverage_values {
                    case_count += 1;
                    run_systematic_case(case_count, q, fair, margin, leverage);
                }
            }
        }
    }
}

fn run_systematic_case(case_id: u64, q: f64, fair: f64, margin: f64, leverage: f64) {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 0.5; // Small deadband
    cfg.hedge.band_vol_mult = 0.0;
    cfg.hedge.k_hedge = 0.5;
    cfg.hedge.max_step_tao = 20.0;
    cfg.hedge.max_venue_tao_per_tick = 10.0;
    cfg.hedge.min_depth_usd = 100.0;
    cfg.hedge.depth_fraction = 0.5;
    cfg.hedge.margin_safety_buffer = 0.95;
    cfg.hedge.max_leverage = leverage;
    cfg.hedge.chunk_size_tao = 2.0;

    // Zero out variable costs for predictability
    cfg.hedge.funding_weight = 0.0;
    cfg.hedge.basis_weight = 0.0;
    cfg.hedge.frag_penalty = 0.0;
    cfg.hedge.liq_penalty_scale = 0.0;
    cfg.hedge.slippage_buffer = 0.0;

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(fair);
    state.fair_value_prev = fair;
    state.vol_ratio_clipped = 1.0;
    state.q_global_tao = q;

    // Set up venues
    for i in 0..3 {
        state.venues[i].mid = Some(fair);
        state.venues[i].spread = Some(fair * 0.005);
        state.venues[i].depth_near_mid = 100_000.0;
        state.venues[i].margin_available_usd = margin;
        state.venues[i].dist_liq_sigma = 10.0;
        state.venues[i].status = paraphina::types::VenueStatus::Healthy;
        state.venues[i].last_mid_update_ms = Some(0);
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);

    // Check all invariants
    let total_size: f64 = intents.iter().map(intent_size).sum();
    assert!(
        total_size <= cfg.hedge.max_step_tao + 1e-6,
        "Systematic case {case_id} (q={q}, fair={fair}, margin={margin}, leverage={leverage}): \
         Total size ({total_size}) exceeds max_step_tao ({})",
        cfg.hedge.max_step_tao
    );

    for intent in &intents {
        assert!(
            is_finite(intent_size(intent)) && is_finite(intent_price(intent)),
            "Systematic case {case_id}: NaN/inf in output"
        );
        assert!(
            intent_size(intent) >= 0.0 && intent_price(intent) > 0.0,
            "Systematic case {case_id}: Invalid size/price"
        );

        let expected_cap = cfg.hedge.max_venue_tao_per_tick;
        assert!(
            intent_size(intent) <= expected_cap + 1e-6,
            "Systematic case {case_id}: Per-venue cap violated"
        );
    }

    // Check direction if hedging
    if !intents.is_empty() && q.abs() > cfg.hedge.band_base_tao {
        let expected_side = if q > 0.0 { Side::Sell } else { Side::Buy };
        for intent in &intents {
            assert_eq!(
                intent_side(intent),
                expected_side,
                "Systematic case {case_id}: Wrong hedge direction for q={q}"
            );
        }
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

/// Test behavior with zero margin (should still work, just limited).
#[test]
fn edge_case_zero_margin() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 1.0;
    cfg.hedge.band_vol_mult = 0.0;
    cfg.hedge.k_hedge = 0.5;
    cfg.hedge.max_step_tao = 10.0;
    cfg.hedge.max_venue_tao_per_tick = 10.0;
    cfg.hedge.margin_safety_buffer = 0.95;
    cfg.hedge.max_leverage = 10.0;
    cfg.hedge.min_depth_usd = 100.0;
    cfg.hedge.depth_fraction = 0.5;

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;
    state.vol_ratio_clipped = 1.0;
    state.q_global_tao = 10.0;

    // Set up venues with zero margin
    for i in 0..3 {
        state.venues[i].mid = Some(100.0);
        state.venues[i].spread = Some(0.5);
        state.venues[i].depth_near_mid = 100_000.0;
        state.venues[i].margin_available_usd = 0.0; // Zero margin!
        state.venues[i].position_tao = 0.0; // Starting flat
        state.venues[i].dist_liq_sigma = 10.0;
        state.venues[i].status = paraphina::types::VenueStatus::Healthy;
        state.venues[i].last_mid_update_ms = Some(0);
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);

    // With zero margin and flat position, any trade would increase exposure
    // and be blocked by margin constraint.
    for intent in &intents {
        assert!(
            is_finite(intent_size(intent)),
            "Output should be finite even with zero margin"
        );
        // With zero margin, the margin cap should limit allocation significantly
        // But we still shouldn't crash or produce NaNs
    }
}

/// Test behavior with zero global inventory (should be in deadband).
#[test]
fn edge_case_zero_inventory() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 1.0;
    cfg.hedge.band_vol_mult = 0.0;

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.q_global_tao = 0.0;

    for i in 0..3 {
        state.venues[i].mid = Some(100.0);
        state.venues[i].spread = Some(0.5);
        state.venues[i].depth_near_mid = 100_000.0;
        state.venues[i].status = paraphina::types::VenueStatus::Healthy;
        state.venues[i].last_mid_update_ms = Some(0);
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);
    assert!(intents.is_empty(), "Zero inventory should be in deadband");
}

/// Test behavior with exactly at deadband boundary.
#[test]
fn edge_case_at_deadband_boundary() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 5.0;
    cfg.hedge.band_vol_mult = 0.0; // band_vol = 5.0 exactly

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.vol_ratio_clipped = 1.0;
    state.q_global_tao = 5.0; // Exactly at boundary

    for i in 0..3 {
        state.venues[i].mid = Some(100.0);
        state.venues[i].spread = Some(0.5);
        state.venues[i].depth_near_mid = 100_000.0;
        state.venues[i].status = paraphina::types::VenueStatus::Healthy;
        state.venues[i].last_mid_update_ms = Some(0);
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);
    // At boundary (<=), should still be in deadband
    assert!(
        intents.is_empty(),
        "At deadband boundary should produce no hedge"
    );
}

/// Test behavior with very large inventory (tests clamping).
#[test]
fn edge_case_very_large_inventory() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 1.0;
    cfg.hedge.band_vol_mult = 0.0;
    cfg.hedge.k_hedge = 1.0; // Full hedge
    cfg.hedge.max_step_tao = 10.0; // But capped at 10

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.vol_ratio_clipped = 1.0;
    state.q_global_tao = 1000.0; // Huge inventory

    for i in 0..3 {
        state.venues[i].mid = Some(100.0);
        state.venues[i].spread = Some(0.5);
        state.venues[i].depth_near_mid = 100_000.0;
        state.venues[i].margin_available_usd = 1_000_000.0;
        state.venues[i].dist_liq_sigma = 10.0;
        state.venues[i].status = paraphina::types::VenueStatus::Healthy;
        state.venues[i].last_mid_update_ms = Some(0);
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);
    let total: f64 = intents.iter().map(intent_size).sum();

    assert!(
        total <= cfg.hedge.max_step_tao + 1e-6,
        "Large inventory should still respect max_step_tao cap"
    );
}

/// Test all venues disabled (should return empty).
#[test]
fn edge_case_all_venues_disabled() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 1.0;

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.q_global_tao = 10.0;

    for i in 0..state.venues.len() {
        state.venues[i].status = paraphina::types::VenueStatus::Disabled;
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);
    assert!(
        intents.is_empty(),
        "All venues disabled should return empty"
    );
}

/// Test single venue available.
#[test]
fn edge_case_single_venue() {
    let mut cfg = paraphina::config::Config::default();
    cfg.hedge.band_base_tao = 1.0;
    cfg.hedge.band_vol_mult = 0.0;
    cfg.hedge.k_hedge = 0.5;
    cfg.hedge.max_step_tao = 10.0;
    cfg.hedge.min_depth_usd = 100.0;
    cfg.hedge.depth_fraction = 0.5;

    let mut state = paraphina::state::GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;
    state.vol_ratio_clipped = 1.0;

    // Only venue 0 enabled
    state.venues[0].mid = Some(100.0);
    state.venues[0].spread = Some(0.5);
    state.venues[0].depth_near_mid = 100_000.0;
    state.venues[0].margin_available_usd = 100_000.0;
    state.venues[0].dist_liq_sigma = 10.0;
    state.venues[0].status = paraphina::types::VenueStatus::Healthy;
    state.venues[0].last_mid_update_ms = Some(0);
    // Set position on venue 0 to make q_global_tao = 10.0 after recompute
    state.venues[0].position_tao = 10.0;

    // Disable all others
    for i in 1..state.venues.len() {
        state.venues[i].status = paraphina::types::VenueStatus::Disabled;
    }

    state.recompute_after_fills(&cfg);

    let intents = compute_hedge_orders(&cfg, &state, 0);

    assert!(
        !intents.is_empty(),
        "Single enabled venue should produce intents"
    );
    assert!(
        intents.iter().all(|i| intent_venue_index(i) == 0),
        "All intents should be for venue 0"
    );
}

// ============================================================================
// CHUNKING INVARIANTS
// ============================================================================

/// Test that output contains at most one order per venue (aggregation).
#[test]
fn invariant_single_order_per_venue() {
    let mut rng = Xorshift64::new(SEED + 6);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        // Check for duplicate venue indices
        let mut seen = std::collections::HashSet::new();
        for intent in &intents {
            assert!(
                seen.insert(intent_venue_index(intent)),
                "Case {case_id}: Venue {} appears multiple times - not properly aggregated\n\
                 Test case: {test_case:?}",
                intent_venue_index(intent)
            );
        }
    }
}

/// Test that venue ordering is consistent (sorted by venue_index).
#[test]
fn invariant_venue_ordering_consistent() {
    let mut rng = Xorshift64::new(SEED + 7);

    for case_id in 0..NUM_RANDOM_CASES {
        let test_case = HedgeTestCase::random(&mut rng, case_id);
        let cfg = test_case.build_config();
        let state = test_case.build_state(&cfg);

        let intents = compute_hedge_orders(&cfg, &state, 0);

        // Check that intents are sorted by venue_index
        for i in 1..intents.len() {
            assert!(
                intent_venue_index(&intents[i - 1]) <= intent_venue_index(&intents[i]),
                "Case {case_id}: Intents not sorted by venue_index\n\
                 Test case: {test_case:?}"
            );
        }
    }
}
