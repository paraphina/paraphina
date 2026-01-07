// tests/buffer_reuse_determinism_tests.rs
//
// Tests to verify that buffer-reusing (*_into) methods produce identical results
// to their allocating counterparts, ensuring the optimization doesn't break determinism.

use paraphina::config::Config;
use paraphina::engine::Engine;
use paraphina::exit::compute_exit_intents_into;
use paraphina::hedge::compute_hedge_orders_into;
use paraphina::mm::{
    compute_mm_quotes, compute_mm_quotes_into, mm_quotes_to_order_intents,
    mm_quotes_to_order_intents_into, MmQuote,
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
