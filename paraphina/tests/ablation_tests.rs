// tests/ablation_tests.rs
//
// Tests for the ablation harness feature.
//
// These tests verify:
// - CLI parses multiple --ablation flags and sorts/dedups them
// - Unknown ablation returns an error
// - config_resolved.json and run_summary.json include ablations
// - checksum differs when ablations differ

use paraphina::sim_eval::{AblationError, AblationSet, VALID_ABLATION_IDS};
use paraphina::{AblationSet as LibAblationSet, BuildInfo, RunSummary, ScenarioSpec};

// =============================================================================
// AblationSet tests
// =============================================================================

#[test]
fn test_parse_multiple_ablations_sorts_and_dedups() {
    // Test that multiple ablations are sorted and deduplicated
    let ids = vec![
        "disable_risk_regime".to_string(),
        "disable_vol_floor".to_string(),
        "disable_vol_floor".to_string(), // duplicate
        "disable_toxicity_gate".to_string(),
    ];

    let set = AblationSet::from_ids(&ids).expect("Should parse valid ablations");

    // Should be sorted and deduped
    assert_eq!(set.ablations.len(), 3);
    assert_eq!(set.ablations[0], "disable_risk_regime");
    assert_eq!(set.ablations[1], "disable_toxicity_gate");
    assert_eq!(set.ablations[2], "disable_vol_floor");
}

#[test]
fn test_unknown_ablation_returns_error() {
    // Test that unknown ablation IDs return an error with clear message
    let ids = vec![
        "disable_vol_floor".to_string(),
        "invalid_ablation".to_string(),
    ];

    let result = AblationSet::from_ids(&ids);
    assert!(result.is_err());

    let err = result.unwrap_err();
    // Error message should be helpful
    let msg = format!("{}", err);
    assert!(msg.contains("invalid_ablation"));
    assert!(msg.contains("Valid ablation IDs"));

    match err {
        AblationError::UnknownAblation { id, valid } => {
            assert_eq!(id, "invalid_ablation");
            assert!(valid.contains(&"disable_vol_floor".to_string()));
            assert!(valid.contains(&"disable_toxicity_gate".to_string()));
            assert!(valid.contains(&"disable_fair_value_gating".to_string()));
            assert!(valid.contains(&"disable_risk_regime".to_string()));
        }
    }
}

#[test]
fn test_valid_ablation_ids_are_complete() {
    // Verify all expected ablation IDs are present
    assert!(VALID_ABLATION_IDS.contains(&"disable_vol_floor"));
    assert!(VALID_ABLATION_IDS.contains(&"disable_fair_value_gating"));
    assert!(VALID_ABLATION_IDS.contains(&"disable_toxicity_gate"));
    assert!(VALID_ABLATION_IDS.contains(&"disable_risk_regime"));
    assert_eq!(VALID_ABLATION_IDS.len(), 4);
}

// =============================================================================
// RunSummary and ConfigResolved ablation tests
// =============================================================================

fn make_test_scenario() -> ScenarioSpec {
    use paraphina::sim_eval::{
        Engine, Horizon, InitialState, Invariants, MarketModel, MicrostructureModel, Rng,
    };

    ScenarioSpec {
        scenario_id: "test_ablation".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: Horizon {
            steps: 100,
            dt_seconds: 0.5,
        },
        rng: Rng {
            base_seed: 42,
            num_seeds: 1,
        },
        initial_state: InitialState {
            risk_profile: "balanced".to_string(),
            init_q_tao: 0.0,
        },
        market_model: MarketModel::default(),
        microstructure_model: MicrostructureModel::default(),
        latency_spike: None,
        partial_fill: paraphina::sim_eval::PartialFillModel::default(),
        cancel_storm: None,
        invariants: Invariants::default(),
    }
}

#[test]
fn test_run_summary_includes_ablations() {
    let scenario = make_test_scenario();
    let build_info = BuildInfo::for_test();
    let kill_switch = paraphina::KillSwitchInfo::default();

    let ablations = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();

    let summary = RunSummary::with_ablations(
        &scenario,
        42,
        build_info,
        100.0,
        50.0,
        kill_switch,
        &ablations,
    );

    // Ablations should be in the summary
    assert_eq!(summary.ablations.len(), 2);
    assert!(summary
        .ablations
        .contains(&"disable_toxicity_gate".to_string()));
    assert!(summary.ablations.contains(&"disable_vol_floor".to_string()));
}

#[test]
fn test_config_resolved_includes_ablations() {
    use paraphina::ConfigResolved;

    let scenario = make_test_scenario();
    let ablations = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_risk_regime".to_string(),
    ])
    .unwrap();

    let config = ConfigResolved::from_scenario_with_ablations(&scenario, &ablations);

    // Ablations should be in the config
    assert_eq!(config.ablations.len(), 2);
    assert!(config
        .ablations
        .contains(&"disable_risk_regime".to_string()));
    assert!(config.ablations.contains(&"disable_vol_floor".to_string()));
}

// =============================================================================
// Checksum tests
// =============================================================================

#[test]
fn test_checksum_differs_with_ablations() {
    let scenario = make_test_scenario();
    let build_info = BuildInfo::for_test();
    let kill_switch = paraphina::KillSwitchInfo::default();

    // Baseline (no ablations)
    let summary_baseline = RunSummary::new(
        &scenario,
        42,
        build_info.clone(),
        100.0,
        50.0,
        kill_switch.clone(),
    );

    // With one ablation
    let ablations1 = LibAblationSet::from_ids(&["disable_vol_floor".to_string()]).unwrap();
    let summary_ablated1 = RunSummary::with_ablations(
        &scenario,
        42,
        build_info.clone(),
        100.0,
        50.0,
        kill_switch.clone(),
        &ablations1,
    );

    // With a different ablation
    let ablations2 = LibAblationSet::from_ids(&["disable_toxicity_gate".to_string()]).unwrap();
    let summary_ablated2 = RunSummary::with_ablations(
        &scenario,
        42,
        build_info.clone(),
        100.0,
        50.0,
        kill_switch.clone(),
        &ablations2,
    );

    // With multiple ablations
    let ablations3 = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();
    let summary_ablated3 = RunSummary::with_ablations(
        &scenario,
        42,
        build_info,
        100.0,
        50.0,
        kill_switch,
        &ablations3,
    );

    // All checksums should be different
    assert_ne!(
        summary_baseline.determinism.checksum, summary_ablated1.determinism.checksum,
        "Baseline and ablated checksums should differ"
    );
    assert_ne!(
        summary_ablated1.determinism.checksum, summary_ablated2.determinism.checksum,
        "Different ablations should produce different checksums"
    );
    assert_ne!(
        summary_ablated2.determinism.checksum, summary_ablated3.determinism.checksum,
        "Different ablation sets should produce different checksums"
    );
}

#[test]
fn test_checksum_same_with_same_ablations() {
    let scenario = make_test_scenario();
    let build_info = BuildInfo::for_test();
    let kill_switch = paraphina::KillSwitchInfo::default();

    // Same ablations in different order should produce same checksum
    let ablations1 = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();

    let ablations2 = LibAblationSet::from_ids(&[
        "disable_toxicity_gate".to_string(),
        "disable_vol_floor".to_string(),
    ])
    .unwrap();

    let summary1 = RunSummary::with_ablations(
        &scenario,
        42,
        build_info.clone(),
        100.0,
        50.0,
        kill_switch.clone(),
        &ablations1,
    );

    let summary2 = RunSummary::with_ablations(
        &scenario,
        42,
        build_info,
        100.0,
        50.0,
        kill_switch,
        &ablations2,
    );

    assert_eq!(
        summary1.determinism.checksum, summary2.determinism.checksum,
        "Same ablations in different order should produce same checksum"
    );
}

// =============================================================================
// Output directory naming tests
// =============================================================================

#[test]
fn test_ablation_short_hash_baseline() {
    let empty = LibAblationSet::new();
    assert_eq!(empty.short_hash(), "baseline");
}

#[test]
fn test_ablation_short_hash_deterministic() {
    let ablations1 = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();

    let ablations2 = LibAblationSet::from_ids(&[
        "disable_toxicity_gate".to_string(),
        "disable_vol_floor".to_string(),
    ])
    .unwrap();

    // Same ablations in different order should produce same hash
    assert_eq!(ablations1.short_hash(), ablations2.short_hash());
    assert_ne!(ablations1.short_hash(), "baseline");
}

#[test]
fn test_ablation_short_hash_unique() {
    let ablations1 = LibAblationSet::from_ids(&["disable_vol_floor".to_string()]).unwrap();
    let ablations2 = LibAblationSet::from_ids(&["disable_toxicity_gate".to_string()]).unwrap();
    let ablations3 = LibAblationSet::from_ids(&["disable_risk_regime".to_string()]).unwrap();

    // Different ablations should produce different hashes
    assert_ne!(ablations1.short_hash(), ablations2.short_hash());
    assert_ne!(ablations2.short_hash(), ablations3.short_hash());
    assert_ne!(ablations1.short_hash(), ablations3.short_hash());
}

// =============================================================================
// Ablation flag helper tests
// =============================================================================

#[test]
fn test_ablation_helper_methods() {
    let ablations = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();

    assert!(ablations.disable_vol_floor());
    assert!(!ablations.disable_fair_value_gating());
    assert!(ablations.disable_toxicity_gate());
    assert!(!ablations.disable_risk_regime());
}

#[test]
fn test_empty_ablation_set_all_false() {
    let empty = LibAblationSet::new();

    assert!(!empty.disable_vol_floor());
    assert!(!empty.disable_fair_value_gating());
    assert!(!empty.disable_toxicity_gate());
    assert!(!empty.disable_risk_regime());
}

// =============================================================================
// Serialization tests
// =============================================================================

#[test]
fn test_ablation_set_json_serialization() {
    let ablations = LibAblationSet::from_ids(&[
        "disable_vol_floor".to_string(),
        "disable_toxicity_gate".to_string(),
    ])
    .unwrap();

    let json = serde_json::to_string(&ablations).expect("Should serialize");
    let parsed: LibAblationSet = serde_json::from_str(&json).expect("Should deserialize");

    assert_eq!(ablations, parsed);
}

#[test]
fn test_run_summary_json_includes_ablations() {
    let scenario = make_test_scenario();
    let build_info = BuildInfo::for_test();
    let kill_switch = paraphina::KillSwitchInfo::default();

    let ablations = LibAblationSet::from_ids(&["disable_vol_floor".to_string()]).unwrap();

    let summary = RunSummary::with_ablations(
        &scenario,
        42,
        build_info,
        100.0,
        50.0,
        kill_switch,
        &ablations,
    );

    let json = serde_json::to_string_pretty(&summary).expect("Should serialize");

    // JSON should contain ablations field
    assert!(json.contains("\"ablations\""));
    assert!(json.contains("disable_vol_floor"));
}
