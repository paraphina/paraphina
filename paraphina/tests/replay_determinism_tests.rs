// tests/replay_determinism_tests.rs
//
// Deterministic replay tests for Paraphina (Milestone H).
//
// These tests verify that the strategy core produces identical action streams
// when given the same inputs. This is critical for:
// - Debugging and reproducing issues
// - Backtesting and research
// - Regulatory compliance (auditable execution)
//
// Test approach:
// 1. Create a deterministic feed of events/ticks
// 2. Run pass A: compute actions and record them
// 3. Reset state, run pass B with same feed
// 4. Assert action streams are identical

use paraphina::config::Config;
use paraphina::engine::Engine;
use paraphina::state::GlobalState;
use paraphina::strategy_core::{compute_actions, StrategyInput};
use paraphina::types::{TimestampMs, VenueStatus};
use paraphina::StrategyOutput;

/// Create a deterministic test configuration.
fn create_test_config() -> Config {
    Config::default()
}

/// Create initial state with deterministic values.
fn create_test_state(cfg: &Config) -> GlobalState {
    let mut state = GlobalState::new(cfg);

    // Set deterministic fair value and market conditions
    state.fair_value = Some(300.0);
    state.fair_value_prev = 300.0;
    state.sigma_eff = 0.02;
    state.spread_mult = 1.0;
    state.size_mult = 1.0;
    state.vol_ratio_clipped = 1.0;
    state.delta_limit_usd = 100_000.0;

    // Set up venue states with deterministic data
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

/// Simulate a deterministic tick with synthetic market data.
fn simulate_tick(engine: &Engine, state: &mut GlobalState, now_ms: TimestampMs) {
    // Seed synthetic books
    engine.seed_dummy_mids(state, now_ms);
    // Run main tick (FV, vol, toxicity, risk)
    engine.main_tick(state, now_ms);
}

/// Run strategy for multiple ticks and collect action batches.
fn run_strategy_pass(
    cfg: &Config,
    initial_state: &GlobalState,
    num_ticks: u64,
    seed: Option<u64>,
) -> Vec<StrategyOutput> {
    let engine = Engine::new(cfg);
    let mut state = initial_state.clone();
    let mut outputs = Vec::new();

    let base_ms: TimestampMs = seed.map(|s| (s % 10_000) as i64).unwrap_or(0);
    let dt_ms: TimestampMs = 1_000;

    for tick in 0..num_ticks {
        let now_ms = base_ms + (tick as i64) * dt_ms;

        // Simulate tick (updates fair value, vol, etc.)
        simulate_tick(&engine, &mut state, now_ms);

        // Compute actions (pure strategy)
        let input = StrategyInput {
            cfg,
            state: &state,
            now_ms,
            tick_index: tick,
            run_seed: seed,
        };

        let output = compute_actions(input);
        outputs.push(output);

        // Note: In a full integration test, we would also execute the actions
        // and apply fills to state. For this test, we just verify the action
        // computation is deterministic.
    }

    outputs
}

#[test]
fn test_replay_determinism_single_tick() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    // Pass A
    let input_a = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(42),
    };
    let output_a = compute_actions(input_a);

    // Pass B (identical inputs)
    let input_b = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(42),
    };
    let output_b = compute_actions(input_b);

    // Compare outputs (intents + logs)
    assert_eq!(output_a.mm_intents, output_b.mm_intents, "MM intents should match");
    assert_eq!(output_a.exit_intents, output_b.exit_intents, "Exit intents should match");
    assert_eq!(output_a.hedge_intents, output_b.hedge_intents, "Hedge intents should match");
    assert_eq!(output_a.logs, output_b.logs, "Logs should match");


}

#[test]
fn test_replay_determinism_multi_tick() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);
    let num_ticks = 10;
    let seed = Some(12345u64);

    // Pass A
    let outputs_a = run_strategy_pass(&cfg, &state, num_ticks, seed);

    // Pass B (identical)
    let outputs_b = run_strategy_pass(&cfg, &state, num_ticks, seed);

    assert_eq!(
        outputs_a.len(),
        outputs_b.len(),
        "Batch counts should match"
    );

    for (tick, (a, b)) in outputs_a.iter().zip(outputs_b.iter()).enumerate() {
        assert_eq!(a.mm_intents, b.mm_intents, "Tick {} MM intents differ", tick);
        assert_eq!(a.exit_intents, b.exit_intents, "Tick {} Exit intents differ", tick);
        assert_eq!(a.hedge_intents, b.hedge_intents, "Tick {} Hedge intents differ", tick);
        assert_eq!(a.logs, b.logs, "Tick {} Logs differ", tick);
    }
}

#[test]
fn test_replay_determinism_with_inventory() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);

    // Add some initial inventory
    state.q_global_tao = 5.0;
    state.venues[0].position_tao = 3.0;
    state.venues[1].position_tao = 2.0;
    state.recompute_after_fills(&cfg);

    // Same inputs => identical outputs
    let input_a = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 2000,
        tick_index: 1,
        run_seed: Some(99),
    };
    let input_b = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 2000,
        tick_index: 1,
        run_seed: Some(99),
    };

    let output_a = compute_actions(input_a);
    let output_b = compute_actions(input_b);

    assert_eq!(output_a.mm_intents, output_b.mm_intents);
    assert_eq!(output_a.exit_intents, output_b.exit_intents);
    assert_eq!(output_a.hedge_intents, output_b.hedge_intents);
    assert_eq!(output_a.logs, output_b.logs);
}



#[test]
fn test_strategy_output_intents_are_well_formed() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let input = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: None,
    };

    let output = compute_actions(input);

    // All produced intents should be Place/Replace with finite positive price/size.
    for it in output
        .mm_intents
        .iter()
        .chain(output.exit_intents.iter())
        .chain(output.hedge_intents.iter())
    {
        match it {
            paraphina::types::OrderIntent::Place(pi) => {
                assert!(pi.price.is_finite() && pi.price > 0.0);
                assert!(pi.size.is_finite() && pi.size > 0.0);
            }
            paraphina::types::OrderIntent::Replace(ri) => {
                assert!(ri.price.is_finite() && ri.price > 0.0);
                assert!(ri.size.is_finite() && ri.size > 0.0);
            }
            other => panic!("Unexpected intent variant from strategy core: {:?}", other),
        }
    }
}

/// Fast performance test - ensure strategy computation is quick.
#[test]
fn test_replay_performance() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);
    let num_ticks = 100;

    let start = std::time::Instant::now();
    let _ = run_strategy_pass(&cfg, &state, num_ticks, Some(42));
    let elapsed = start.elapsed();

    // Should complete in well under 1 second
    assert!(
        elapsed.as_millis() < 1000,
        "100 ticks should complete in <1s, took {:?}",
        elapsed
    );
}
