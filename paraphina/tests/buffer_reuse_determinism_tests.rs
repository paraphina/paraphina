// tests/buffer_reuse_determinism_tests.rs
//
// Tests to verify that buffer-reusing (*_into) methods produce identical results
// to their allocating counterparts, ensuring the optimization doesn't break determinism.

use paraphina::config::Config;
use paraphina::engine::Engine;
use paraphina::exit::compute_exit_intents_into;
use paraphina::hedge::compute_hedge_orders_into;
use paraphina::mm::{
    compute_mm_quotes, compute_mm_quotes_into, compute_mm_quotes_into_with_scratch,
    mm_quotes_to_order_intents, mm_quotes_to_order_intents_into, MmQuote, MmScratch,
};
use paraphina::state::GlobalState;
use paraphina::types::{OrderIntent, TimestampMs, VenueStatus};
use paraphina::{exit, hedge};

/// Create a test configuration.
fn create_test_config() -> Config {
    Config::default()
}

/// Create a realistic test state with market data.
fn create_test_state(cfg: &Config) -> GlobalState {
    let mut state = GlobalState::new(cfg);

    // Set fair value and market conditions
    state.fair_value = Some(300.0);
    state.fair_value_prev = 300.0;
    state.sigma_eff = 0.02;
    state.spread_mult = 1.0;
    state.size_mult = 1.0;
    state.vol_ratio_clipped = 1.0;
    state.delta_limit_usd = 100_000.0;

    // Set up venue states with valid data
    for (i, v) in state.venues.iter_mut().enumerate() {
        v.mid = Some(300.0 + i as f64 * 0.1);
        v.spread = Some(1.00);
        v.depth_near_mid = 10_000.0;
        v.margin_available_usd = 10_000.0;
        v.dist_liq_sigma = 10.0;
        v.status = VenueStatus::Healthy;
        v.toxicity = 0.0;
        v.last_mid_update_ms = Some(1000);
    }

    state
}

/// Helper to compare two slices of OrderIntent for equality.
fn intents_equal(a: &[OrderIntent], b: &[OrderIntent]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (ia, ib) in a.iter().zip(b.iter()) {
        if ia.venue_index != ib.venue_index
            || ia.venue_id != ib.venue_id
            || ia.side != ib.side
            || (ia.price - ib.price).abs() > 1e-12
            || (ia.size - ib.size).abs() > 1e-12
            || ia.purpose != ib.purpose
        {
            return false;
        }
    }
    true
}

/// Helper to compare two slices of MmQuote for equality.
fn mm_quotes_equal(a: &[MmQuote], b: &[MmQuote]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (qa, qb) in a.iter().zip(b.iter()) {
        if qa.venue_index != qb.venue_index || qa.venue_id != qb.venue_id {
            return false;
        }

        match (&qa.bid, &qb.bid) {
            (Some(ba), Some(bb)) => {
                if (ba.price - bb.price).abs() > 1e-12 || (ba.size - bb.size).abs() > 1e-12 {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        match (&qa.ask, &qb.ask) {
            (Some(aa), Some(ab)) => {
                if (aa.price - ab.price).abs() > 1e-12 || (aa.size - ab.size).abs() > 1e-12 {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
    }
    true
}

// =============================================================================
// MM Module Tests
// =============================================================================

#[test]
fn test_compute_mm_quotes_into_matches_allocating() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    // Allocating version
    let quotes_alloc = compute_mm_quotes(&cfg, &state);

    // Buffer-reusing version
    let mut quotes_buf: Vec<MmQuote> = Vec::new();
    compute_mm_quotes_into(&cfg, &state, &mut quotes_buf);

    assert!(
        mm_quotes_equal(&quotes_alloc, &quotes_buf),
        "compute_mm_quotes_into produces different results than compute_mm_quotes"
    );
}

#[test]
fn test_mm_quotes_to_order_intents_into_matches_allocating() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let quotes = compute_mm_quotes(&cfg, &state);

    // Allocating version
    let intents_alloc = mm_quotes_to_order_intents(&quotes);

    // Buffer-reusing version
    let mut intents_buf: Vec<OrderIntent> = Vec::new();
    mm_quotes_to_order_intents_into(&quotes, &mut intents_buf);

    assert!(
        intents_equal(&intents_alloc, &intents_buf),
        "mm_quotes_to_order_intents_into produces different results"
    );
}

#[test]
fn test_mm_into_buffer_reuse_preserves_capacity() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    // Pre-allocate with large capacity
    let mut quotes_buf: Vec<MmQuote> = Vec::with_capacity(100);
    let mut intents_buf: Vec<OrderIntent> = Vec::with_capacity(200);

    // Run multiple times
    for _ in 0..10 {
        compute_mm_quotes_into(&cfg, &state, &mut quotes_buf);
        mm_quotes_to_order_intents_into(&quotes_buf, &mut intents_buf);
    }

    // Capacity should be preserved or increased, never reallocated smaller
    assert!(
        quotes_buf.capacity() >= 100,
        "Buffer capacity should be preserved"
    );
    assert!(
        intents_buf.capacity() >= 200,
        "Buffer capacity should be preserved"
    );
}

#[test]
fn test_compute_mm_quotes_into_with_scratch_matches_allocating() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    // Allocating version
    let quotes_alloc = compute_mm_quotes(&cfg, &state);

    // Buffer-reusing version with scratch
    let mut quotes_buf: Vec<MmQuote> = Vec::new();
    let mut scratch = MmScratch::new();
    compute_mm_quotes_into_with_scratch(&cfg, &state, &mut quotes_buf, &mut scratch);

    assert!(
        mm_quotes_equal(&quotes_alloc, &quotes_buf),
        "compute_mm_quotes_into_with_scratch produces different results than compute_mm_quotes"
    );
}

#[test]
fn test_mm_scratch_buffer_reuse_preserves_capacity() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    // Pre-allocate with large capacity
    let mut quotes_buf: Vec<MmQuote> = Vec::with_capacity(100);
    let mut scratch = MmScratch::with_capacity(50);

    let initial_scratch_capacity = scratch.venue_targets_capacity();
    assert!(
        initial_scratch_capacity >= 50,
        "Initial scratch capacity should be >= 50"
    );

    // Run multiple times
    for _ in 0..10 {
        compute_mm_quotes_into_with_scratch(&cfg, &state, &mut quotes_buf, &mut scratch);
    }

    // Capacity should be preserved or increased, never reallocated smaller
    assert!(
        quotes_buf.capacity() >= 100,
        "Quotes buffer capacity should be preserved"
    );
    assert!(
        scratch.venue_targets_capacity() >= initial_scratch_capacity,
        "Scratch buffer capacity should be preserved"
    );
}

#[test]
fn test_mm_scratch_determinism_across_ticks() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);

    let mut quotes_buf_scratch: Vec<MmQuote> = Vec::new();
    let mut scratch = MmScratch::new();

    // Simulate multiple ticks and verify consistency with allocating version
    for tick in 0..5 {
        // Modify state slightly each tick
        state.q_global_tao = tick as f64 * 0.5;
        state.recompute_after_fills(&cfg);

        let quotes_alloc = compute_mm_quotes(&cfg, &state);
        compute_mm_quotes_into_with_scratch(&cfg, &state, &mut quotes_buf_scratch, &mut scratch);

        assert!(
            mm_quotes_equal(&quotes_alloc, &quotes_buf_scratch),
            "Tick {}: scratch version diverges from allocating version",
            tick
        );
    }
}

// =============================================================================
// Exit Module Tests
// =============================================================================

#[test]
fn test_compute_exit_intents_into_matches_allocating() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);

    // Add some inventory to trigger exit logic
    state.q_global_tao = 5.0;
    state.venues[0].position_tao = 3.0;
    state.venues[1].position_tao = 2.0;
    state.recompute_after_fills(&cfg);

    let now_ms: TimestampMs = 2000;

    // Allocating version
    let intents_alloc = exit::compute_exit_intents(&cfg, &state, now_ms);

    // Buffer-reusing version
    let mut intents_buf: Vec<OrderIntent> = Vec::new();
    compute_exit_intents_into(&cfg, &state, now_ms, &mut intents_buf);

    assert!(
        intents_equal(&intents_alloc, &intents_buf),
        "compute_exit_intents_into produces different results"
    );
}

#[test]
fn test_exit_into_buffer_reuse() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);
    state.q_global_tao = 5.0;
    state.venues[0].position_tao = 3.0;
    state.venues[1].position_tao = 2.0;
    state.recompute_after_fills(&cfg);

    let mut intents_buf: Vec<OrderIntent> = Vec::with_capacity(50);

    for t in 0..10 {
        let now_ms: TimestampMs = 1000 + t * 1000;
        compute_exit_intents_into(&cfg, &state, now_ms, &mut intents_buf);
    }

    assert!(
        intents_buf.capacity() >= 50,
        "Buffer capacity should be preserved"
    );
}

// =============================================================================
// Hedge Module Tests
// =============================================================================

#[test]
fn test_compute_hedge_orders_into_matches_allocating() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);

    // Add some delta to trigger hedge logic
    state.dollar_delta_usd = 5000.0;
    state.q_global_tao = 20.0;
    state.venues[0].position_tao = 10.0;
    state.venues[1].position_tao = 10.0;
    state.recompute_after_fills(&cfg);

    let now_ms: TimestampMs = 2000;

    // Allocating version
    let intents_alloc = hedge::compute_hedge_orders(&cfg, &state, now_ms);

    // Buffer-reusing version
    let mut intents_buf: Vec<OrderIntent> = Vec::new();
    compute_hedge_orders_into(&cfg, &state, now_ms, &mut intents_buf);

    assert!(
        intents_equal(&intents_alloc, &intents_buf),
        "compute_hedge_orders_into produces different results"
    );
}

#[test]
fn test_hedge_into_buffer_reuse() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);
    state.dollar_delta_usd = 5000.0;
    state.q_global_tao = 20.0;
    state.venues[0].position_tao = 10.0;
    state.recompute_after_fills(&cfg);

    let mut intents_buf: Vec<OrderIntent> = Vec::with_capacity(50);

    for t in 0..10 {
        let now_ms: TimestampMs = 1000 + t * 1000;
        compute_hedge_orders_into(&cfg, &state, now_ms, &mut intents_buf);
    }

    assert!(
        intents_buf.capacity() >= 50,
        "Buffer capacity should be preserved"
    );
}

// =============================================================================
// Full Tick Loop Simulation Tests
// =============================================================================

#[test]
fn test_full_tick_loop_determinism() {
    // Simulate multiple ticks and verify the buffer-reusing path produces
    // the same results as the allocating path.

    let cfg = create_test_config();
    let engine = Engine::new(&cfg);

    let num_ticks = 50;
    let num_venues = cfg.venues.len();

    // --- Run with allocating versions ---
    let mut state_alloc = GlobalState::new(&cfg);
    let mut results_alloc: Vec<(Vec<OrderIntent>, Vec<OrderIntent>, Vec<OrderIntent>)> = Vec::new();

    for t in 0..num_ticks {
        let now_ms: TimestampMs = 1000 + t * 1000;

        engine.seed_dummy_mids(&mut state_alloc, now_ms);
        engine.main_tick(&mut state_alloc, now_ms);

        if state_alloc.kill_switch {
            break;
        }

        let mm_quotes = compute_mm_quotes(&cfg, &state_alloc);
        let mm_intents = mm_quotes_to_order_intents(&mm_quotes);
        let exit_intents = exit::compute_exit_intents(&cfg, &state_alloc, now_ms);
        let hedge_intents = hedge::compute_hedge_orders(&cfg, &state_alloc, now_ms);

        results_alloc.push((
            mm_intents.clone(),
            exit_intents.clone(),
            hedge_intents.clone(),
        ));

        // Apply fills (simple simulation)
        for intent in mm_intents
            .iter()
            .chain(exit_intents.iter())
            .chain(hedge_intents.iter())
        {
            state_alloc.apply_perp_fill(
                intent.venue_index,
                intent.side,
                intent.size,
                intent.price,
                5.0, // 5 bps fee
            );
        }
        state_alloc.recompute_after_fills(&cfg);
    }

    // --- Run with buffer-reusing versions ---
    let mut state_buf = GlobalState::new(&cfg);
    let mut mm_quotes_buf: Vec<MmQuote> = Vec::with_capacity(num_venues);
    let mut mm_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues * 2);
    let mut exit_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues);
    let mut hedge_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues);

    let mut results_buf: Vec<(Vec<OrderIntent>, Vec<OrderIntent>, Vec<OrderIntent>)> = Vec::new();

    for t in 0..num_ticks {
        let now_ms: TimestampMs = 1000 + t * 1000;

        engine.seed_dummy_mids(&mut state_buf, now_ms);
        engine.main_tick(&mut state_buf, now_ms);

        if state_buf.kill_switch {
            break;
        }

        compute_mm_quotes_into(&cfg, &state_buf, &mut mm_quotes_buf);
        mm_quotes_to_order_intents_into(&mm_quotes_buf, &mut mm_intents_buf);
        compute_exit_intents_into(&cfg, &state_buf, now_ms, &mut exit_intents_buf);
        compute_hedge_orders_into(&cfg, &state_buf, now_ms, &mut hedge_intents_buf);

        results_buf.push((
            mm_intents_buf.clone(),
            exit_intents_buf.clone(),
            hedge_intents_buf.clone(),
        ));

        // Apply fills
        for intent in mm_intents_buf
            .iter()
            .chain(exit_intents_buf.iter())
            .chain(hedge_intents_buf.iter())
        {
            state_buf.apply_perp_fill(
                intent.venue_index,
                intent.side,
                intent.size,
                intent.price,
                5.0,
            );
        }
        state_buf.recompute_after_fills(&cfg);
    }

    // --- Compare results ---
    assert_eq!(
        results_alloc.len(),
        results_buf.len(),
        "Different number of ticks executed"
    );

    for (tick, ((mm_a, exit_a, hedge_a), (mm_b, exit_b, hedge_b))) in
        results_alloc.iter().zip(results_buf.iter()).enumerate()
    {
        assert!(
            intents_equal(mm_a, mm_b),
            "MM intents differ at tick {}",
            tick
        );
        assert!(
            intents_equal(exit_a, exit_b),
            "Exit intents differ at tick {}",
            tick
        );
        assert!(
            intents_equal(hedge_a, hedge_b),
            "Hedge intents differ at tick {}",
            tick
        );
    }

    // Final state should be identical
    assert!(
        (state_alloc.q_global_tao - state_buf.q_global_tao).abs() < 1e-9,
        "Final q_global differs"
    );
    assert!(
        (state_alloc.daily_pnl_total - state_buf.daily_pnl_total).abs() < 1e-9,
        "Final PnL differs"
    );
}

#[test]
fn test_repeated_buffer_clears() {
    // Verify that buffers are properly cleared between calls.

    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let mut mm_quotes_buf: Vec<MmQuote> = Vec::with_capacity(10);
    let mut intents_buf: Vec<OrderIntent> = Vec::with_capacity(20);

    // Run first call
    compute_mm_quotes_into(&cfg, &state, &mut mm_quotes_buf);
    mm_quotes_to_order_intents_into(&mm_quotes_buf, &mut intents_buf);

    let len1 = intents_buf.len();

    // Run second call - buffers should be cleared and repopulated
    compute_mm_quotes_into(&cfg, &state, &mut mm_quotes_buf);
    mm_quotes_to_order_intents_into(&mm_quotes_buf, &mut intents_buf);

    let len2 = intents_buf.len();

    // Lengths should be the same since state is identical
    assert_eq!(len1, len2, "Buffer not properly cleared between calls");
}

// =============================================================================
// Engine seed_dummy_mids Optimization Tests
// =============================================================================

/// Compute a simple checksum of engine state for determinism verification.
fn compute_state_checksum(state: &GlobalState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash fair value and volatility state
    if let Some(fv) = state.fair_value {
        fv.to_bits().hash(&mut hasher);
    }
    state.fair_value_prev.to_bits().hash(&mut hasher);
    state.kf_x_hat.to_bits().hash(&mut hasher);
    state.kf_p.to_bits().hash(&mut hasher);
    state.fv_short_vol.to_bits().hash(&mut hasher);
    state.fv_long_vol.to_bits().hash(&mut hasher);
    state.sigma_eff.to_bits().hash(&mut hasher);

    // Hash per-venue state
    for v in &state.venues {
        if let Some(mid) = v.mid {
            mid.to_bits().hash(&mut hasher);
        }
        if let Some(spread) = v.spread {
            spread.to_bits().hash(&mut hasher);
        }
        v.depth_near_mid.to_bits().hash(&mut hasher);
        v.local_vol_short.to_bits().hash(&mut hasher);
        v.local_vol_long.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

#[test]
fn test_seed_dummy_mids_determinism_regression() {
    // Run the same tick loop twice with identical initial conditions
    // and verify that all outputs are byte-identical.

    let cfg = create_test_config();
    let engine = Engine::new(&cfg);

    let num_ticks = 100;

    // --- First run ---
    let mut state1 = GlobalState::new(&cfg);
    let mut checksums1: Vec<u64> = Vec::with_capacity(num_ticks);

    for t in 0..num_ticks {
        let now_ms: TimestampMs = 1000 + t as i64 * 100;

        engine.seed_dummy_mids(&mut state1, now_ms);
        engine.main_tick(&mut state1, now_ms);

        checksums1.push(compute_state_checksum(&state1));
    }

    // --- Second run (identical inputs) ---
    let mut state2 = GlobalState::new(&cfg);
    let mut checksums2: Vec<u64> = Vec::with_capacity(num_ticks);

    for t in 0..num_ticks {
        let now_ms: TimestampMs = 1000 + t as i64 * 100;

        engine.seed_dummy_mids(&mut state2, now_ms);
        engine.main_tick(&mut state2, now_ms);

        checksums2.push(compute_state_checksum(&state2));
    }

    // --- Verify checksums match tick-by-tick ---
    assert_eq!(
        checksums1.len(),
        checksums2.len(),
        "Different number of ticks executed"
    );

    for (tick, (c1, c2)) in checksums1.iter().zip(checksums2.iter()).enumerate() {
        assert_eq!(
            c1, c2,
            "State checksum mismatch at tick {}: {} != {}",
            tick, c1, c2
        );
    }

    // --- Verify final state is identical ---
    assert!(
        (state1.fair_value.unwrap_or(0.0) - state2.fair_value.unwrap_or(0.0)).abs() < 1e-15,
        "Final fair value differs"
    );
    assert!(
        (state1.kf_x_hat - state2.kf_x_hat).abs() < 1e-15,
        "Final kf_x_hat differs"
    );
    assert!(
        (state1.sigma_eff - state2.sigma_eff).abs() < 1e-15,
        "Final sigma_eff differs"
    );

    // Verify per-venue state is identical
    for (i, (v1, v2)) in state1.venues.iter().zip(state2.venues.iter()).enumerate() {
        assert!(
            (v1.mid.unwrap_or(0.0) - v2.mid.unwrap_or(0.0)).abs() < 1e-15,
            "Venue {} mid differs",
            i
        );
        assert!(
            (v1.local_vol_short - v2.local_vol_short).abs() < 1e-15,
            "Venue {} local_vol_short differs",
            i
        );
        assert!(
            (v1.local_vol_long - v2.local_vol_long).abs() < 1e-15,
            "Venue {} local_vol_long differs",
            i
        );
    }
}

/// Test seed_dummy_mids in isolation (Opt7 verification).
///
/// This test verifies that seed_dummy_mids:
/// 1. Produces bit-exact identical results across two independent runs with same inputs
/// 2. Does not cause any heap allocations (scratch buffers are untouched)
/// 3. Maintains deterministic ordering across venues
///
/// Note: The function is deterministic for a given (state, now_ms) pair, NOT idempotent.
/// Calling with the same now_ms twice will update local_vol based on the mid change
/// (which is 0 when mid doesn't change), so the EWMA will decay. This is correct behavior.
#[test]
fn test_seed_dummy_mids_isolated_determinism_and_no_alloc() {
    let cfg = create_test_config();
    let engine = Engine::new(&cfg);
    let num_venues = cfg.venues.len();

    // --- Run 1: First state ---
    let mut state1 = GlobalState::new(&cfg);

    // Record initial scratch buffer capacities (should not change since
    // seed_dummy_mids doesn't use scratch buffers - this verifies that claim)
    let initial_mids_cap = state1.scratch_mids_capacity();
    let initial_kf_obs_cap = state1.scratch_kf_obs_capacity();

    // Run seed_dummy_mids multiple times
    for t in 0..100 {
        let now_ms: TimestampMs = 1000 + t * 50;
        engine.seed_dummy_mids(&mut state1, now_ms);
    }

    // Verify scratch buffers were NOT touched (seed_dummy_mids doesn't use them)
    assert_eq!(
        state1.scratch_mids_capacity(),
        initial_mids_cap,
        "scratch_mids capacity changed unexpectedly"
    );
    assert_eq!(
        state1.scratch_kf_obs_capacity(),
        initial_kf_obs_cap,
        "scratch_kf_obs capacity changed unexpectedly"
    );

    // --- Run 2: Second state with identical inputs (fresh state, same sequence) ---
    let mut state2 = GlobalState::new(&cfg);
    for t in 0..100 {
        let now_ms: TimestampMs = 1000 + t * 50;
        engine.seed_dummy_mids(&mut state2, now_ms);
    }

    // --- Verify bit-exact identical venue state ---
    for i in 0..num_venues {
        let v1 = &state1.venues[i];
        let v2 = &state2.venues[i];

        // Mid values must be bit-exact
        assert_eq!(
            v1.mid.map(|m| m.to_bits()),
            v2.mid.map(|m| m.to_bits()),
            "Venue {} mid not bit-exact",
            i
        );

        // Spread values must be bit-exact
        assert_eq!(
            v1.spread.map(|s| s.to_bits()),
            v2.spread.map(|s| s.to_bits()),
            "Venue {} spread not bit-exact",
            i
        );

        // Depth must be bit-exact
        assert_eq!(
            v1.depth_near_mid.to_bits(),
            v2.depth_near_mid.to_bits(),
            "Venue {} depth not bit-exact",
            i
        );

        // Last update timestamp must match
        assert_eq!(
            v1.last_mid_update_ms, v2.last_mid_update_ms,
            "Venue {} last_mid_update_ms differs",
            i
        );

        // Local volatilities must be bit-exact
        assert_eq!(
            v1.local_vol_short.to_bits(),
            v2.local_vol_short.to_bits(),
            "Venue {} local_vol_short not bit-exact",
            i
        );
        assert_eq!(
            v1.local_vol_long.to_bits(),
            v2.local_vol_long.to_bits(),
            "Venue {} local_vol_long not bit-exact",
            i
        );
    }

    // --- Run 3: Verify a third independent run also matches ---
    // This confirms reproducibility across any number of runs.
    let mut state3 = GlobalState::new(&cfg);
    for t in 0..100 {
        let now_ms: TimestampMs = 1000 + t * 50;
        engine.seed_dummy_mids(&mut state3, now_ms);
    }

    // Verify state3 matches state1 bit-exactly
    for i in 0..num_venues {
        let v1 = &state1.venues[i];
        let v3 = &state3.venues[i];

        assert_eq!(
            v1.mid.map(|m| m.to_bits()),
            v3.mid.map(|m| m.to_bits()),
            "Venue {} mid not bit-exact on 3rd run",
            i
        );
        assert_eq!(
            v1.local_vol_short.to_bits(),
            v3.local_vol_short.to_bits(),
            "Venue {} local_vol_short not bit-exact on 3rd run",
            i
        );
        assert_eq!(
            v1.local_vol_long.to_bits(),
            v3.local_vol_long.to_bits(),
            "Venue {} local_vol_long not bit-exact on 3rd run",
            i
        );
    }

    // Verify no scratch buffer growth after 3 runs
    assert_eq!(
        state3.scratch_mids_capacity(),
        initial_mids_cap,
        "scratch_mids capacity grew unexpectedly"
    );
    assert_eq!(
        state3.scratch_kf_obs_capacity(),
        initial_kf_obs_cap,
        "scratch_kf_obs capacity grew unexpectedly"
    );
}

#[test]
fn test_engine_scratch_buffer_capacity_preserved() {
    // Verify that scratch buffers in GlobalState preserve their capacity
    // across multiple tick cycles (no shrinking/reallocating).

    let cfg = create_test_config();
    let engine = Engine::new(&cfg);
    let mut state = GlobalState::new(&cfg);

    let num_venues = cfg.venues.len();

    // Record initial capacity after construction using accessor methods
    let initial_mids_cap = state.scratch_mids_capacity();
    let initial_kf_obs_cap = state.scratch_kf_obs_capacity();

    // Verify initial capacity is at least venue count (pre-allocated)
    assert!(
        initial_mids_cap >= num_venues,
        "scratch_mids should be pre-allocated: {} < {}",
        initial_mids_cap,
        num_venues
    );
    assert!(
        initial_kf_obs_cap >= num_venues,
        "scratch_kf_obs should be pre-allocated: {} < {}",
        initial_kf_obs_cap,
        num_venues
    );

    // Run many ticks
    for t in 0..200 {
        let now_ms: TimestampMs = 1000 + t * 100;
        engine.seed_dummy_mids(&mut state, now_ms);
        engine.main_tick(&mut state, now_ms);
    }

    // Verify capacity was preserved (not shrunk)
    assert!(
        state.scratch_mids_capacity() >= initial_mids_cap,
        "scratch_mids capacity shrunk: {} < {}",
        state.scratch_mids_capacity(),
        initial_mids_cap
    );
    assert!(
        state.scratch_kf_obs_capacity() >= initial_kf_obs_cap,
        "scratch_kf_obs capacity shrunk: {} < {}",
        state.scratch_kf_obs_capacity(),
        initial_kf_obs_cap
    );

    // Verify buffers are cleared between ticks (not accumulating)
    // After the last main_tick, buffers should have been used and cleared
    // for the next potential tick. Let's run one more seed + tick and check lengths.
    let now_ms: TimestampMs = 1000 + 200 * 100;
    engine.seed_dummy_mids(&mut state, now_ms);
    engine.main_tick(&mut state, now_ms);

    // After main_tick, scratch buffers should be populated with current tick's data
    // (not accumulated from all previous ticks)
    assert!(
        state.scratch_mids_len() <= num_venues,
        "scratch_mids accumulated instead of cleared: {} > {}",
        state.scratch_mids_len(),
        num_venues
    );
    assert!(
        state.scratch_kf_obs_len() <= num_venues,
        "scratch_kf_obs accumulated instead of cleared: {} > {}",
        state.scratch_kf_obs_len(),
        num_venues
    );
}

// =============================================================================
// Toxicity Determinism Tests
// =============================================================================

use paraphina::state::PendingMarkout;
use paraphina::toxicity::update_toxicity_and_health;
use paraphina::types::Side;

/// Test that update_toxicity_and_health produces identical results across repeated calls.
///
/// This verifies determinism is preserved for toxicity calculations.
#[test]
fn test_toxicity_update_determinism() {
    let cfg = create_test_config();
    let num_venues = cfg.venues.len();

    // --- Run 1: First state ---
    let mut state1 = GlobalState::new(&cfg);

    // Setup venues with valid market data
    for (i, v) in state1.venues.iter_mut().enumerate() {
        v.mid = Some(100.0 + i as f64 * 0.5);
        v.spread = Some(0.1);
        v.depth_near_mid = 10_000.0;
        v.last_mid_update_ms = Some(0);
        v.toxicity = 0.1;
    }

    // Add some pending markouts to process
    for v in state1.venues.iter_mut() {
        v.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 99.0,
            fair_at_fill: 99.0,
            mid_at_fill: 99.0,
        });
        v.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 100,
            t_eval_ms: 1100,
            side: Side::Sell,
            size_tao: 0.5,
            price: 101.0,
            fair_at_fill: 101.0,
            mid_at_fill: 101.0,
        });
    }

    // Run toxicity update
    state1.sigma_eff = 0.02;
    update_toxicity_and_health(&mut state1, &cfg, 1500);

    // Capture results
    let tox1: Vec<f64> = state1.venues.iter().map(|v| v.toxicity).collect();
    let status1: Vec<_> = state1.venues.iter().map(|v| v.status).collect();
    let markout_ewma1: Vec<f64> = state1
        .venues
        .iter()
        .map(|v| v.markout_ewma_usd_per_tao)
        .collect();

    // --- Run 2: Second state with identical inputs ---
    let mut state2 = GlobalState::new(&cfg);

    for (i, v) in state2.venues.iter_mut().enumerate() {
        v.mid = Some(100.0 + i as f64 * 0.5);
        v.spread = Some(0.1);
        v.depth_near_mid = 10_000.0;
        v.last_mid_update_ms = Some(0);
        v.toxicity = 0.1;
    }

    for v in state2.venues.iter_mut() {
        v.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 99.0,
            fair_at_fill: 99.0,
            mid_at_fill: 99.0,
        });
        v.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 100,
            t_eval_ms: 1100,
            side: Side::Sell,
            size_tao: 0.5,
            price: 101.0,
            fair_at_fill: 101.0,
            mid_at_fill: 101.0,
        });
    }

    state2.sigma_eff = 0.02;
    update_toxicity_and_health(&mut state2, &cfg, 1500);

    // --- Verify bit-exact identical results ---
    for i in 0..num_venues {
        assert_eq!(
            tox1[i].to_bits(),
            state2.venues[i].toxicity.to_bits(),
            "Venue {} toxicity not bit-exact: {} vs {}",
            i,
            tox1[i],
            state2.venues[i].toxicity
        );
        assert_eq!(
            status1[i], state2.venues[i].status,
            "Venue {} status differs",
            i
        );
        assert_eq!(
            markout_ewma1[i].to_bits(),
            state2.venues[i].markout_ewma_usd_per_tao.to_bits(),
            "Venue {} markout_ewma not bit-exact",
            i
        );
    }
}
