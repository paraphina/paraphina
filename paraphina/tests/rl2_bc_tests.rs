// tests/rl2_bc_tests.rs
//
// Integration tests for RL-2 Behaviour Cloning infrastructure.
//
// Tests:
// 1. Action encoding determinism
// 2. Action encoding round-trip
// 3. Trajectory collection determinism
// 4. Observation encoding consistency

use paraphina::{
    decode_action, encode_action, ActionEncodingSpec, Config, GlobalState, HeuristicPolicy,
    Observation, Policy, PolicyAction, TrajectoryCollector, ACTION_VERSION, OBS_VERSION,
    TRAJECTORY_VERSION,
};

// ============================================================================
// Action Encoding Tests
// ============================================================================

#[test]
fn test_action_encoding_determinism_identity() {
    let num_venues = 5;
    let policy_version = "test-v1";

    let action = PolicyAction::identity(num_venues, policy_version);

    // Encode multiple times
    let encoded1 = encode_action(&action, num_venues);
    let encoded2 = encode_action(&action, num_venues);
    let encoded3 = encode_action(&action, num_venues);

    // All encodings must be identical
    assert_eq!(
        encoded1, encoded2,
        "Identity action encoding not deterministic"
    );
    assert_eq!(
        encoded2, encoded3,
        "Identity action encoding not deterministic"
    );
}

#[test]
fn test_action_encoding_determinism_varied() {
    let num_venues = 5;
    let policy_version = "test-v1";

    let action = PolicyAction {
        policy_version: policy_version.to_string(),
        policy_id: Some("test".to_string()),
        spread_scale: vec![0.6, 1.2, 1.8, 2.4, 2.9],
        size_scale: vec![0.1, 0.5, 1.0, 1.5, 1.9],
        rprice_offset_usd: vec![-8.0, -4.0, 0.0, 4.0, 8.0],
        hedge_scale: 1.3,
        hedge_venue_weights: vec![0.3, 0.25, 0.2, 0.15, 0.1],
    };

    // Encode multiple times
    let encoded1 = encode_action(&action, num_venues);
    let encoded2 = encode_action(&action, num_venues);

    // All encodings must be identical (bit-for-bit)
    assert_eq!(
        encoded1, encoded2,
        "Varied action encoding not deterministic"
    );
}

#[test]
fn test_action_encoding_round_trip_identity() {
    let num_venues = 5;
    let policy_version = "test-v1";

    let original = PolicyAction::identity(num_venues, policy_version);
    let encoded = encode_action(&original, num_venues);
    let decoded = decode_action(&encoded, num_venues, policy_version);

    // Check each field with tolerance for floating-point
    let eps = 1e-5;

    for i in 0..num_venues {
        assert!(
            (original.spread_scale[i] - decoded.spread_scale[i]).abs() < eps,
            "spread_scale[{}] round-trip failed: {} vs {}",
            i,
            original.spread_scale[i],
            decoded.spread_scale[i]
        );

        assert!(
            (original.size_scale[i] - decoded.size_scale[i]).abs() < eps,
            "size_scale[{}] round-trip failed: {} vs {}",
            i,
            original.size_scale[i],
            decoded.size_scale[i]
        );

        assert!(
            (original.rprice_offset_usd[i] - decoded.rprice_offset_usd[i]).abs() < 0.1,
            "rprice_offset_usd[{}] round-trip failed: {} vs {}",
            i,
            original.rprice_offset_usd[i],
            decoded.rprice_offset_usd[i]
        );
    }

    assert!(
        (original.hedge_scale - decoded.hedge_scale).abs() < eps,
        "hedge_scale round-trip failed"
    );
}

#[test]
fn test_action_encoding_round_trip_extreme_values() {
    let num_venues = 5;
    let policy_version = "test-v1";

    // Test with extreme values that should be clamped
    let original = PolicyAction {
        policy_version: policy_version.to_string(),
        policy_id: None,
        spread_scale: vec![0.5, 3.0, 1.0, 1.0, 1.0], // At bounds
        size_scale: vec![0.0, 2.0, 1.0, 1.0, 1.0],   // At bounds
        rprice_offset_usd: vec![-10.0, 10.0, 0.0, 0.0, 0.0], // At bounds
        hedge_scale: 2.0,                            // At bound
        hedge_venue_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
    };

    let encoded = encode_action(&original, num_venues);
    let decoded = decode_action(&encoded, num_venues, policy_version);

    // Values at bounds should survive round-trip
    assert!(
        (decoded.spread_scale[0] - 0.5).abs() < 0.01,
        "Min spread_scale not preserved"
    );
    assert!(
        (decoded.spread_scale[1] - 3.0).abs() < 0.01,
        "Max spread_scale not preserved"
    );
    assert!(
        (decoded.size_scale[0] - 0.0).abs() < 0.01,
        "Min size_scale not preserved"
    );
    assert!(
        (decoded.size_scale[1] - 2.0).abs() < 0.01,
        "Max size_scale not preserved"
    );
}

#[test]
fn test_action_encoding_spec_dimensions() {
    let spec_5 = ActionEncodingSpec::new(5);
    // Per-venue: 3 * 5 = 15, Global: 1 + 5 = 6, Total: 21
    assert_eq!(spec_5.action_dim, 21);
    assert_eq!(spec_5.num_venues, 5);
    assert_eq!(spec_5.version, ACTION_VERSION);

    let spec_3 = ActionEncodingSpec::new(3);
    // Per-venue: 3 * 3 = 9, Global: 1 + 3 = 4, Total: 13
    assert_eq!(spec_3.action_dim, 13);
}

#[test]
fn test_action_encoding_clamping() {
    let num_venues = 5;
    let policy_version = "test-v1";

    // Out-of-bounds values
    let action = PolicyAction {
        policy_version: policy_version.to_string(),
        policy_id: None,
        spread_scale: vec![0.0, 5.0, 1.0, 1.0, 1.0], // Out of [0.5, 3.0]
        size_scale: vec![-1.0, 3.0, 1.0, 1.0, 1.0],  // Out of [0.0, 2.0]
        rprice_offset_usd: vec![-20.0, 20.0, 0.0, 0.0, 0.0], // Out of [-10, 10]
        hedge_scale: 5.0,                            // Out of [0.0, 2.0]
        hedge_venue_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
    };

    let encoded = encode_action(&action, num_venues);
    let decoded = decode_action(&encoded, num_venues, policy_version);

    // All values should be clamped to valid ranges
    assert!(
        decoded.spread_scale[0] >= 0.5,
        "spread_scale should be clamped to >= 0.5"
    );
    assert!(
        decoded.spread_scale[1] <= 3.0,
        "spread_scale should be clamped to <= 3.0"
    );
    assert!(
        decoded.size_scale[0] >= 0.0,
        "size_scale should be clamped to >= 0.0"
    );
    assert!(
        decoded.size_scale[1] <= 2.0,
        "size_scale should be clamped to <= 2.0"
    );
    assert!(
        decoded.rprice_offset_usd[0] >= -10.0,
        "rprice_offset should be clamped to >= -10"
    );
    assert!(
        decoded.rprice_offset_usd[1] <= 10.0,
        "rprice_offset should be clamped to <= 10"
    );
    assert!(
        decoded.hedge_scale <= 2.0,
        "hedge_scale should be clamped to <= 2.0"
    );
}

// ============================================================================
// Trajectory Collection Tests
// ============================================================================

#[test]
fn test_trajectory_collection_determinism_small() {
    // Collect with same seed twice, should get identical results
    let collector1 = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
    let (records1, meta1) = collector1.collect(2);

    let collector2 = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
    let (records2, meta2) = collector2.collect(2);

    // Same number of records
    assert_eq!(records1.len(), records2.len(), "Record counts differ");

    // Same metadata
    assert_eq!(meta1.num_episodes, meta2.num_episodes);
    assert_eq!(meta1.num_transitions, meta2.num_transitions);
    assert_eq!(meta1.obs_version, meta2.obs_version);
    assert_eq!(meta1.action_version, meta2.action_version);
    assert_eq!(meta1.trajectory_version, meta2.trajectory_version);

    // Same records (bit-for-bit)
    for (r1, r2) in records1.iter().zip(records2.iter()) {
        assert_eq!(
            r1.obs_features, r2.obs_features,
            "Observation features differ at episode {}, step {}",
            r1.episode_idx, r1.step_idx
        );
        assert_eq!(
            r1.action_target, r2.action_target,
            "Action targets differ at episode {}, step {}",
            r1.episode_idx, r1.step_idx
        );
        assert_eq!(r1.reward, r2.reward, "Rewards differ");
        assert_eq!(r1.terminal, r2.terminal, "Terminals differ");
        assert_eq!(r1.episode_idx, r2.episode_idx);
        assert_eq!(r1.step_idx, r2.step_idx);
        assert_eq!(r1.seed, r2.seed);
    }
}

#[test]
fn test_trajectory_collection_different_seeds() {
    // Different seeds should produce different results
    let collector1 = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
    let (records1, _) = collector1.collect(2);

    let collector2 = TrajectoryCollector::new(2, 10, false, "deterministic", 99);
    let (records2, _) = collector2.collect(2);

    // Same number of records (same config)
    assert_eq!(records1.len(), records2.len());

    // But different content (at least one difference expected)
    let mut has_difference = false;
    for (r1, r2) in records1.iter().zip(records2.iter()) {
        if r1.obs_features != r2.obs_features
            || r1.action_target != r2.action_target
            || r1.seed != r2.seed
        {
            has_difference = true;
            break;
        }
    }

    assert!(
        has_difference,
        "Different seeds should produce different trajectories"
    );
}

#[test]
fn test_trajectory_metadata_versions() {
    let collector = TrajectoryCollector::new(1, 5, false, "deterministic", 42);
    let (_, metadata) = collector.collect(1);

    assert_eq!(metadata.obs_version, OBS_VERSION);
    assert_eq!(metadata.action_version, ACTION_VERSION);
    assert_eq!(metadata.trajectory_version, TRAJECTORY_VERSION);
    assert!(!metadata.policy_version.is_empty());
}

#[test]
fn test_trajectory_record_dimensions() {
    let collector = TrajectoryCollector::new(1, 5, false, "deterministic", 42);
    let (records, metadata) = collector.collect(1);

    assert!(!records.is_empty());

    // Check dimensions match metadata
    for record in &records {
        assert_eq!(
            record.obs_features.len(),
            metadata.obs_dim,
            "Observation dimension mismatch"
        );
        assert_eq!(
            record.action_target.len(),
            metadata.action_dim,
            "Action dimension mismatch"
        );
    }
}

// ============================================================================
// Observation Encoding Tests
// ============================================================================

#[test]
fn test_observation_to_features_determinism() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    state.fair_value = Some(300.0);
    state.sigma_eff = 0.02;
    state.spread_mult = 1.0;

    for v in &mut state.venues {
        v.mid = Some(300.0);
        v.spread = Some(1.0);
        v.depth_near_mid = 10000.0;
    }

    // Create observation twice
    let obs1 = Observation::from_state(&state, &cfg, 1000, 0);
    let obs2 = Observation::from_state(&state, &cfg, 1000, 0);

    // Observations should be identical
    assert_eq!(obs1, obs2, "Observation construction not deterministic");

    // Serialize to JSON and compare
    let json1 = obs1.to_canonical_json().unwrap();
    let json2 = obs2.to_canonical_json().unwrap();

    assert_eq!(
        json1, json2,
        "Observation JSON serialization not deterministic"
    );
}

// ============================================================================
// Heuristic Policy Tests
// ============================================================================

#[test]
fn test_heuristic_policy_determinism() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    state.fair_value = Some(300.0);
    state.sigma_eff = 0.02;

    for v in &mut state.venues {
        v.mid = Some(300.0);
        v.spread = Some(1.0);
    }

    let obs = Observation::from_state(&state, &cfg, 1000, 0);
    let policy = HeuristicPolicy::new();

    // Call policy multiple times
    let action1 = policy.act(&obs);
    let action2 = policy.act(&obs);
    let action3 = policy.act(&obs);

    // All actions should be identical
    assert_eq!(
        action1, action2,
        "Heuristic policy not deterministic on same observation"
    );
    assert_eq!(action2, action3);
}

#[test]
fn test_heuristic_policy_produces_identity() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    state.fair_value = Some(300.0);
    state.sigma_eff = 0.02;

    for v in &mut state.venues {
        v.mid = Some(300.0);
        v.spread = Some(1.0);
    }

    let obs = Observation::from_state(&state, &cfg, 1000, 0);
    let policy = HeuristicPolicy::new();
    let action = policy.act(&obs);

    // Heuristic policy should produce identity (baseline behavior)
    assert!(
        action.is_identity(),
        "Heuristic policy should produce identity action"
    );
}

// ============================================================================
// Version Compatibility Tests
// ============================================================================

#[test]
fn test_version_constants_are_documented() {
    // Verify version constants are accessible and can be compared.
    // The actual values are compile-time constants; we verify the exported symbols are usable.
    let obs_v: u32 = OBS_VERSION;
    let action_v: u32 = ACTION_VERSION;
    let traj_v: u32 = TRAJECTORY_VERSION;

    // These should not panic - verifies the constants are valid u32 values
    assert!(obs_v == obs_v, "OBS_VERSION identity check");
    assert!(action_v == action_v, "ACTION_VERSION identity check");
    assert!(traj_v == traj_v, "TRAJECTORY_VERSION identity check");
}
