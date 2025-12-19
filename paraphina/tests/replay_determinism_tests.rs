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

use paraphina::actions::{Action, ActionBatch};
use paraphina::config::Config;
use paraphina::engine::Engine;
use paraphina::io::noop::compare_batches;
use paraphina::state::GlobalState;
use paraphina::strategy_core::{compute_actions, StrategyInput};
use paraphina::types::{TimestampMs, VenueStatus};

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
) -> Vec<ActionBatch> {
    let engine = Engine::new(cfg);
    let mut state = initial_state.clone();
    let mut batches = Vec::new();

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
        batches.push(output.batch);

        // Note: In a full integration test, we would also execute the actions
        // and apply fills to state. For this test, we just verify the action
        // computation is deterministic.
    }

    batches
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

    // Compare action batches
    assert_eq!(
        output_a.batch.actions.len(),
        output_b.batch.actions.len(),
        "Action counts should match"
    );

    for (i, (a, b)) in output_a
        .batch
        .actions
        .iter()
        .zip(output_b.batch.actions.iter())
        .enumerate()
    {
        assert_eq!(a, b, "Action {} should be identical", i);
    }
}

#[test]
fn test_replay_determinism_multi_tick() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);
    let num_ticks = 10;
    let seed = Some(12345u64);

    // Pass A
    let batches_a = run_strategy_pass(&cfg, &state, num_ticks, seed);

    // Pass B (identical)
    let batches_b = run_strategy_pass(&cfg, &state, num_ticks, seed);

    assert_eq!(
        batches_a.len(),
        batches_b.len(),
        "Batch counts should match"
    );

    for (tick, (batch_a, batch_b)) in batches_a.iter().zip(batches_b.iter()).enumerate() {
        let result = compare_batches(batch_a, batch_b);
        assert!(
            result.is_ok(),
            "Tick {} batches differ: {}",
            tick,
            result.unwrap_err()
        );
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

    // Pass A
    let input_a = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 2000,
        tick_index: 1,
        run_seed: Some(99),
    };
    let output_a = compute_actions(input_a);

    // Pass B
    let input_b = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 2000,
        tick_index: 1,
        run_seed: Some(99),
    };
    let output_b = compute_actions(input_b);

    let result = compare_batches(&output_a.batch, &output_b.batch);
    assert!(
        result.is_ok(),
        "Batches with inventory differ: {}",
        result.unwrap_err()
    );
}

#[test]
fn test_action_ids_deterministic() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let input = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 42,
        run_seed: Some(1),
    };

    let output1 = compute_actions(input.clone());
    let output2 = compute_actions(input);

    // Collect action IDs
    let ids1: Vec<String> = output1
        .batch
        .actions
        .iter()
        .map(|a| match a {
            Action::PlaceOrder(po) => po.action_id.clone(),
            Action::CancelOrder(co) => co.action_id.clone(),
            Action::CancelAll(ca) => ca.action_id.clone(),
            Action::SetKillSwitch(ks) => ks.action_id.clone(),
            Action::Log(l) => l.action_id.clone(),
        })
        .collect();

    let ids2: Vec<String> = output2
        .batch
        .actions
        .iter()
        .map(|a| match a {
            Action::PlaceOrder(po) => po.action_id.clone(),
            Action::CancelOrder(co) => co.action_id.clone(),
            Action::CancelAll(ca) => ca.action_id.clone(),
            Action::SetKillSwitch(ks) => ks.action_id.clone(),
            Action::Log(l) => l.action_id.clone(),
        })
        .collect();

    assert_eq!(ids1, ids2, "Action IDs should be deterministic");

    // All IDs should contain tick index
    for id in &ids1 {
        assert!(
            id.contains("t42_"),
            "Action ID '{}' should contain tick index",
            id
        );
    }
}

#[test]
fn test_different_seeds_produce_different_batches() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let input1 = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(1),
    };

    let input2 = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(2), // Different seed
    };

    let output1 = compute_actions(input1);
    let output2 = compute_actions(input2);

    // Batches should have same actions (seed only affects timebase offset in sim)
    // but metadata should differ
    assert_eq!(output1.batch.run_seed, Some(1));
    assert_eq!(output2.batch.run_seed, Some(2));
}

#[test]
fn test_batch_metadata_correct() {
    let cfg = create_test_config();
    let state = create_test_state(&cfg);

    let input = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 5000,
        tick_index: 10,
        run_seed: Some(42),
    };

    let output = compute_actions(input);

    assert_eq!(output.batch.now_ms, 5000);
    assert_eq!(output.batch.tick_index, 10);
    assert_eq!(output.batch.run_seed, Some(42));
    assert_eq!(output.batch.config_version, cfg.version);
}

#[test]
fn test_kill_switch_determinism() {
    let cfg = create_test_config();
    let mut state = create_test_state(&cfg);

    // Activate kill switch
    state.kill_switch = true;
    state.kill_reason = paraphina::state::KillReason::PnlHardBreach;

    let input1 = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(1),
    };

    let input2 = StrategyInput {
        cfg: &cfg,
        state: &state,
        now_ms: 1000,
        tick_index: 0,
        run_seed: Some(1),
    };

    let output1 = compute_actions(input1);
    let output2 = compute_actions(input2);

    let result = compare_batches(&output1.batch, &output2.batch);
    assert!(
        result.is_ok(),
        "Kill switch batches differ: {}",
        result.unwrap_err()
    );

    // Should only have a log action (no trading)
    assert_eq!(output1.batch.actions.len(), 1);
    match &output1.batch.actions[0] {
        Action::Log(log) => {
            assert!(log.message.contains("Kill switch"));
        }
        _ => panic!("Expected Log action when kill switch active"),
    }
}

#[test]
fn test_strategy_output_intents_match_actions() {
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

    // Count PlaceOrder actions
    let place_order_count = output
        .batch
        .actions
        .iter()
        .filter(|a| matches!(a, Action::PlaceOrder(_)))
        .count();

    // Total intents should equal PlaceOrder actions
    let total_intents =
        output.mm_intents.len() + output.exit_intents.len() + output.hedge_intents.len();

    assert_eq!(
        place_order_count, total_intents,
        "PlaceOrder actions should match intent count"
    );
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
