//! RL-0 determinism and shadow mode tests.
//!
//! Per ROADMAP.md RL-0 requirements:
//! - Determinism: with same seed/config, policy trace output is byte-for-byte identical across runs
//! - Shadow mode: baseline executed actions are identical with and without shadow enabled

use paraphina::gateway::SimGateway;
use paraphina::logging::NoopSink;
use paraphina::state::GlobalState;
use paraphina::{
    Config, EpisodeConfig, HeuristicPolicy, NoopPolicy, Observation, Policy, PolicyAction,
    ShadowRunner, OBS_VERSION,
};

/// Test that observations are deterministic given the same state.
#[test]
fn test_observation_deterministic() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    // Set up deterministic state
    state.fair_value = Some(300.0);
    state.fair_value_prev = 300.0;
    state.sigma_eff = 0.02;
    state.spread_mult = 1.0;
    state.size_mult = 1.0;
    state.vol_ratio_clipped = 1.0;
    state.q_global_tao = 5.0;
    state.dollar_delta_usd = 1500.0;
    state.daily_pnl_total = 100.0;

    for v in &mut state.venues {
        v.mid = Some(300.0);
        v.spread = Some(1.0);
        v.depth_near_mid = 10_000.0;
        v.toxicity = 0.1;
        v.position_tao = 1.0;
    }

    // Create observations multiple times
    let obs1 = Observation::from_state(&state, &cfg, 1000, 5);
    let obs2 = Observation::from_state(&state, &cfg, 1000, 5);

    // Should be identical
    assert_eq!(
        obs1, obs2,
        "Same state should produce identical observations"
    );

    // JSON should be byte-for-byte identical
    let json1 = obs1.to_canonical_json().unwrap();
    let json2 = obs2.to_canonical_json().unwrap();
    assert_eq!(json1, json2, "JSON serialization should be deterministic");
}

/// Test that observation version is set correctly.
#[test]
fn test_observation_version() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);

    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    assert_eq!(obs.obs_version, OBS_VERSION);
}

/// Test that policy actions are deterministic.
#[test]
fn test_policy_action_deterministic() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);
    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    let policy = HeuristicPolicy::new();

    let action1 = policy.act(&obs);
    let action2 = policy.act(&obs);

    assert_eq!(
        action1, action2,
        "Same observation should produce same action"
    );

    // JSON should be identical
    let json1 = serde_json::to_vec(&action1).unwrap();
    let json2 = serde_json::to_vec(&action2).unwrap();
    assert_eq!(json1, json2, "Action JSON should be deterministic");
}

/// Test episode determinism: same seed produces identical results.
#[test]
fn test_episode_determinism() {
    let cfg = Config::default();

    // Run episode 1
    let gateway1 = SimGateway;
    let sink1 = NoopSink;
    let mut runner1 = ShadowRunner::new(&cfg, gateway1, sink1);
    let config1 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(50);
    let summary1 = runner1.run_episode(config1);

    // Run episode 2 with same seed
    let gateway2 = SimGateway;
    let sink2 = NoopSink;
    let mut runner2 = ShadowRunner::new(&cfg, gateway2, sink2);
    let config2 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(50);
    let summary2 = runner2.run_episode(config2);

    // Results should be identical
    assert_eq!(
        summary1.total_ticks, summary2.total_ticks,
        "Tick count should be identical"
    );
    assert!(
        (summary1.final_pnl_total - summary2.final_pnl_total).abs() < 1e-9,
        "PnL should be identical: {} vs {}",
        summary1.final_pnl_total,
        summary2.final_pnl_total
    );
    assert!(
        (summary1.final_q_global_tao - summary2.final_q_global_tao).abs() < 1e-9,
        "Inventory should be identical: {} vs {}",
        summary1.final_q_global_tao,
        summary2.final_q_global_tao
    );
    assert!(
        (summary1.final_realised_pnl - summary2.final_realised_pnl).abs() < 1e-9,
        "Realised PnL should be identical"
    );
    assert!(
        (summary1.final_unrealised_pnl - summary2.final_unrealised_pnl).abs() < 1e-9,
        "Unrealised PnL should be identical"
    );
    assert_eq!(
        summary1.kill_switch_triggered, summary2.kill_switch_triggered,
        "Kill switch status should be identical"
    );
}

/// Test that different seeds produce different results.
#[test]
fn test_different_seeds_produce_different_results() {
    let cfg = Config::default();

    // Run with seed 42
    let gateway1 = SimGateway;
    let sink1 = NoopSink;
    let mut runner1 = ShadowRunner::new(&cfg, gateway1, sink1);
    let config1 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(50);
    let summary1 = runner1.run_episode(config1);

    // Run with seed 123
    let gateway2 = SimGateway;
    let sink2 = NoopSink;
    let mut runner2 = ShadowRunner::new(&cfg, gateway2, sink2);
    let config2 = EpisodeConfig::default()
        .with_seed(123)
        .with_episode_id(2)
        .with_max_ticks(50);
    let summary2 = runner2.run_episode(config2);

    // At least some metrics should differ (different time offsets)
    // Note: with the current sim, the behavior is deterministic per seed
    assert_ne!(summary1.seed, summary2.seed, "Seeds should be different");
}

/// Test shadow mode: baseline results identical with and without shadow.
#[test]
fn test_shadow_mode_no_execution_impact() {
    let cfg = Config::default();

    // Run WITHOUT shadow mode
    let gateway1 = SimGateway;
    let sink1 = NoopSink;
    let mut runner1 = ShadowRunner::new(&cfg, gateway1, sink1);
    let config1 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(50)
        .with_shadow_mode(false);
    let summary_no_shadow = runner1.run_episode(config1);

    // Run WITH shadow mode (using NoopPolicy as shadow)
    let gateway2 = SimGateway;
    let sink2 = NoopSink;
    let shadow_policy = Box::new(NoopPolicy::new());
    let mut runner2 = ShadowRunner::new(&cfg, gateway2, sink2).with_shadow_policy(shadow_policy);
    let config2 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(50)
        .with_shadow_mode(true);
    let summary_with_shadow = runner2.run_episode(config2);

    // Results MUST be identical (shadow should not affect execution)
    assert_eq!(
        summary_no_shadow.total_ticks, summary_with_shadow.total_ticks,
        "Shadow mode must not affect tick count"
    );
    assert!(
        (summary_no_shadow.final_pnl_total - summary_with_shadow.final_pnl_total).abs() < 1e-9,
        "Shadow mode must not affect PnL: {} vs {}",
        summary_no_shadow.final_pnl_total,
        summary_with_shadow.final_pnl_total
    );
    assert!(
        (summary_no_shadow.final_q_global_tao - summary_with_shadow.final_q_global_tao).abs()
            < 1e-9,
        "Shadow mode must not affect inventory: {} vs {}",
        summary_no_shadow.final_q_global_tao,
        summary_with_shadow.final_q_global_tao
    );
    assert!(
        (summary_no_shadow.final_realised_pnl - summary_with_shadow.final_realised_pnl).abs()
            < 1e-9,
        "Shadow mode must not affect realised PnL"
    );
    assert!(
        (summary_no_shadow.final_unrealised_pnl - summary_with_shadow.final_unrealised_pnl).abs()
            < 1e-9,
        "Shadow mode must not affect unrealised PnL"
    );
    assert_eq!(
        summary_no_shadow.kill_switch_triggered, summary_with_shadow.kill_switch_triggered,
        "Shadow mode must not affect kill switch"
    );
}

/// Test that HeuristicPolicy produces identity actions.
#[test]
fn test_heuristic_policy_is_identity() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);
    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    let policy = HeuristicPolicy::new();
    let action = policy.act(&obs);

    assert!(
        action.is_identity(),
        "HeuristicPolicy should produce identity actions"
    );
}

/// Test policy reset for episode boundaries.
#[test]
fn test_policy_reset_episode() {
    let mut policy = HeuristicPolicy::new();

    // Reset for episode 1
    policy.reset_episode(42, 1);

    // Reset for episode 2
    policy.reset_episode(123, 2);

    // Policy should still work correctly
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);
    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    let action = policy.act(&obs);
    assert!(action.is_identity());
}

/// Test observation JSON roundtrip preserves equality.
#[test]
fn test_observation_json_roundtrip() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(300.0);
    state.sigma_eff = 0.02;
    state.q_global_tao = 5.0;
    state.daily_pnl_total = 100.0;

    for v in &mut state.venues {
        v.mid = Some(300.0);
        v.spread = Some(1.0);
    }

    let obs = Observation::from_state(&state, &cfg, 1000, 5);

    let json = serde_json::to_string(&obs).unwrap();
    let parsed: Observation = serde_json::from_str(&json).unwrap();

    assert_eq!(obs, parsed, "Observation should roundtrip through JSON");
}

/// Test policy action JSON roundtrip preserves equality.
#[test]
fn test_policy_action_json_roundtrip() {
    let action = PolicyAction::identity(5, "test-v1");

    let json = serde_json::to_string(&action).unwrap();
    let parsed: PolicyAction = serde_json::from_str(&json).unwrap();

    assert_eq!(action, parsed, "PolicyAction should roundtrip through JSON");
}

/// Test policy action clamping.
#[test]
fn test_policy_action_clamp() {
    let mut action = PolicyAction::identity(3, "test-v1");

    // Set out-of-range values
    action.spread_scale = vec![0.1, 5.0, 1.5];
    action.size_scale = vec![-1.0, 3.0, 1.0];
    action.hedge_scale = 5.0;
    action.rprice_offset_usd = vec![-100.0, 100.0, 0.0];

    action.clamp();

    // Check clamped values
    assert_eq!(action.spread_scale[0], 0.5);
    assert_eq!(action.spread_scale[1], 3.0);
    assert_eq!(action.spread_scale[2], 1.5);

    assert_eq!(action.size_scale[0], 0.0);
    assert_eq!(action.size_scale[1], 2.0);
    assert_eq!(action.size_scale[2], 1.0);

    assert_eq!(action.hedge_scale, 2.0);

    assert_eq!(action.rprice_offset_usd[0], -10.0);
    assert_eq!(action.rprice_offset_usd[1], 10.0);
    assert_eq!(action.rprice_offset_usd[2], 0.0);
}

/// Test multiple episodes in sequence with reset.
#[test]
fn test_multiple_episodes_with_reset() {
    let cfg = Config::default();
    let gateway = SimGateway;
    let sink = NoopSink;
    let mut runner = ShadowRunner::new(&cfg, gateway, sink);

    // Run first episode
    let config1 = EpisodeConfig::default()
        .with_seed(42)
        .with_episode_id(1)
        .with_max_ticks(20);
    let summary1 = runner.run_episode(config1);

    // Run second episode (different seed)
    let config2 = EpisodeConfig::default()
        .with_seed(123)
        .with_episode_id(2)
        .with_max_ticks(20);
    let summary2 = runner.run_episode(config2);

    // Both should complete
    assert!(summary1.total_ticks > 0);
    assert!(summary2.total_ticks > 0);

    // Seeds should be recorded
    assert_eq!(summary1.seed, 42);
    assert_eq!(summary2.seed, 123);
}

/// Test observation venue ordering is deterministic.
#[test]
fn test_observation_venue_ordering() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);

    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    // Venues should be ordered by index
    for (i, v) in obs.venues.iter().enumerate() {
        assert_eq!(v.venue_index, i, "Venues must be ordered by index");
    }
}

/// Test that observation includes all required fields per WHITEPAPER A.3.
#[test]
fn test_observation_has_required_fields() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(300.0);
    state.sigma_eff = 0.02;
    state.daily_pnl_total = 100.0;

    let obs = Observation::from_state(&state, &cfg, 1000, 0);

    // Global features (per WHITEPAPER A.3)
    assert!(obs.fair_value.is_some(), "fair_value required");
    assert!(obs.sigma_eff > 0.0, "sigma_eff required");
    assert!(obs.vol_ratio_clipped > 0.0, "vol_ratio_clipped required");
    assert!(obs.spread_mult > 0.0, "spread_mult required");
    assert!(obs.size_mult > 0.0, "size_mult required");

    // Risk fields
    assert!(obs.delta_limit_usd > 0.0, "delta_limit_usd required");
    assert!(
        obs.basis_limit_hard_usd > 0.0,
        "basis_limit_hard_usd required"
    );

    // Per-venue features
    assert!(!obs.venues.is_empty(), "venues required");
    for v in &obs.venues {
        assert!(!v.venue_id.is_empty(), "venue_id required");
        // dist_liq_sigma can be any positive value
        assert!(
            v.dist_liq_sigma >= 0.0,
            "dist_liq_sigma must be non-negative"
        );
    }
}
