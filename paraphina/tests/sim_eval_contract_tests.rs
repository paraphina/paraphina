// tests/sim_eval_contract_tests.rs
//
// Contract tests for the sim_eval module per Option B requirements.
//
// Tests:
// 1. Parsing scenarios/v1/synth_baseline.yaml succeeds
// 2. Required output fields are present
// 3. Determinism: two runs with same scenario+seed produce same checksum

use paraphina::config::{Config, RiskProfile};
use paraphina::metrics::DrawdownTracker;
use paraphina::rl::sim_env::SimEnvConfig;
use paraphina::rl::{PolicyAction, SimEnv};
use paraphina::sim_eval::{
    BuildInfo, Engine, ExpectKillSwitch, KillSwitchInfo, MarketModelType, PnlLinearityCheck,
    RunSummary, ScenarioSpec, SuiteSpec, SyntheticProcess,
};

// --------------------------------------------------------------------------
// Test 1: Parsing scenarios/v1/synth_baseline.yaml succeeds
// --------------------------------------------------------------------------

#[test]
fn test_parse_synth_baseline_yaml() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scenarios/v1/synth_baseline.yaml");

    let spec = ScenarioSpec::from_yaml_file(&path).expect("Should parse synth_baseline.yaml");

    // Verify required fields
    assert_eq!(spec.scenario_id, "synth_baseline");
    assert_eq!(spec.scenario_version, 1);
    assert_eq!(spec.engine, Engine::RlSimEnv);
    assert_eq!(spec.horizon.steps, 2000);
    assert!((spec.horizon.dt_seconds - 0.25).abs() < 1e-9);
    assert_eq!(spec.rng.base_seed, 42);
    assert_eq!(spec.rng.num_seeds, 5);
    assert_eq!(spec.initial_state.risk_profile, "balanced");
    assert!((spec.initial_state.init_q_tao - 0.0).abs() < 1e-9);
    assert_eq!(spec.market_model.model_type, MarketModelType::Synthetic);

    // Verify synthetic config
    let synth = spec
        .market_model
        .synthetic
        .as_ref()
        .expect("Should have synthetic config");
    assert_eq!(synth.process, SyntheticProcess::Gbm);
    assert!((synth.params.vol - 0.015).abs() < 1e-9);

    // Verify invariants
    assert_eq!(
        spec.invariants.expect_kill_switch,
        ExpectKillSwitch::Allowed
    );
    assert_eq!(
        spec.invariants.pnl_linearity_check,
        PnlLinearityCheck::Disabled
    );
}

#[test]
fn test_parse_synth_jump_yaml() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scenarios/v1/synth_jump.yaml");

    let spec = ScenarioSpec::from_yaml_file(&path).expect("Should parse synth_jump.yaml");

    assert_eq!(spec.scenario_id, "synth_jump");
    assert_eq!(spec.scenario_version, 1);
    assert_eq!(spec.initial_state.risk_profile, "conservative");
    assert!((spec.initial_state.init_q_tao - 5.0).abs() < 1e-9);

    // Verify jump diffusion params
    let synth = spec
        .market_model
        .synthetic
        .as_ref()
        .expect("Should have synthetic config");
    assert_eq!(synth.process, SyntheticProcess::JumpDiffusionStub);
    assert!((synth.params.vol - 0.02).abs() < 1e-9);
    assert!((synth.params.jump_intensity - 0.0005).abs() < 1e-9);

    // Verify microstructure
    assert!((spec.microstructure_model.fees_bps_taker - 1.0).abs() < 1e-9);
    assert!((spec.microstructure_model.latency_ms - 5.0).abs() < 1e-9);
}

#[test]
fn test_parse_historical_stub_yaml() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scenarios/v1/historical_replay_stub.yaml");

    let spec =
        ScenarioSpec::from_yaml_file(&path).expect("Should parse historical_replay_stub.yaml");

    assert_eq!(spec.scenario_id, "historical_stub");
    assert_eq!(spec.engine, Engine::ReplayStub);
    assert_eq!(
        spec.market_model.model_type,
        MarketModelType::HistoricalStub
    );

    let hist = spec
        .market_model
        .historical_stub
        .as_ref()
        .expect("Should have historical_stub config");
    assert!(!hist.dataset_id.is_empty());
}

// --------------------------------------------------------------------------
// Test 2: Required output fields are present
// --------------------------------------------------------------------------

#[test]
fn test_run_summary_required_fields() {
    // Create a minimal scenario spec
    let spec = ScenarioSpec {
        scenario_id: "test_scenario".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: paraphina::sim_eval::Horizon {
            steps: 100,
            dt_seconds: 0.5,
        },
        rng: paraphina::sim_eval::Rng {
            base_seed: 42,
            num_seeds: 1,
        },
        initial_state: paraphina::sim_eval::InitialState {
            risk_profile: "balanced".to_string(),
            init_q_tao: 0.0,
        },
        market_model: paraphina::sim_eval::MarketModel::default(),
        microstructure_model: paraphina::sim_eval::MicrostructureModel::default(),
        invariants: paraphina::sim_eval::Invariants::default(),
    };

    let build_info = BuildInfo::for_test();
    let kill_switch = KillSwitchInfo::default();

    let summary = RunSummary::new(&spec, 42, build_info, 123.45, 67.89, kill_switch);

    // Verify all required fields per docs/SIM_OUTPUT_SCHEMA.md
    // Top-level
    assert_eq!(summary.scenario_id, "test_scenario");
    assert_eq!(summary.scenario_version, 1);
    assert_eq!(summary.seed, 42);

    // build_info
    assert!(!summary.build_info.git_sha.is_empty());
    // dirty can be true or false

    // config
    assert_eq!(summary.config.risk_profile, "balanced");
    assert!((summary.config.init_q_tao - 0.0).abs() < 1e-9);
    assert!((summary.config.dt_seconds - 0.5).abs() < 1e-9);
    assert_eq!(summary.config.steps, 100);

    // results
    assert!((summary.results.final_pnl_usd - 123.45).abs() < 1e-9);
    assert!((summary.results.max_drawdown_usd - 67.89).abs() < 1e-9);

    // kill_switch
    assert!(!summary.results.kill_switch.triggered);
    assert!(summary.results.kill_switch.step.is_none());
    assert!(summary.results.kill_switch.reason.is_none());

    // determinism
    assert!(!summary.determinism.checksum.is_empty());
    // SHA256 hex is 64 chars
    assert_eq!(summary.determinism.checksum.len(), 64);
}

#[test]
fn test_run_summary_with_kill_switch() {
    let spec = ScenarioSpec {
        scenario_id: "test_kill".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: paraphina::sim_eval::Horizon {
            steps: 100,
            dt_seconds: 1.0,
        },
        rng: paraphina::sim_eval::Rng::default(),
        initial_state: paraphina::sim_eval::InitialState::default(),
        market_model: paraphina::sim_eval::MarketModel::default(),
        microstructure_model: paraphina::sim_eval::MicrostructureModel::default(),
        invariants: paraphina::sim_eval::Invariants::default(),
    };

    let build_info = BuildInfo::for_test();
    let kill_switch = KillSwitchInfo {
        triggered: true,
        step: Some(42),
        reason: Some("PnlHardBreach".to_string()),
    };

    let summary = RunSummary::new(&spec, 1, build_info, -5000.0, 5000.0, kill_switch);

    assert!(summary.results.kill_switch.triggered);
    assert_eq!(summary.results.kill_switch.step, Some(42));
    assert_eq!(
        summary.results.kill_switch.reason,
        Some("PnlHardBreach".to_string())
    );
}

// --------------------------------------------------------------------------
// Test 3: Determinism - same scenario+seed => same checksum
// --------------------------------------------------------------------------

/// Helper function to run a simulation and return the checksum.
fn run_simulation_for_checksum(seed: u64, steps: u64) -> String {
    let config = Config::for_profile(RiskProfile::Balanced);
    let mut env_config = SimEnvConfig::deterministic();
    env_config.max_ticks = steps;
    env_config.dt_ms = 250; // 0.25 seconds

    let mut env = SimEnv::new(config, env_config);
    env.reset(Some(seed));

    let action = PolicyAction::identity(env.num_venues(), "test");
    let mut dd_tracker = DrawdownTracker::new();
    let mut kill_switch = KillSwitchInfo::default();

    for step in 0..steps {
        let result = env.step(&action);
        dd_tracker.update(result.info.pnl_total);

        if result.info.kill_switch && !kill_switch.triggered {
            kill_switch.triggered = true;
            kill_switch.step = Some(step);
            kill_switch.reason = result.info.kill_reason.clone();
        }

        if result.done {
            break;
        }
    }

    let state = env.state();
    let final_pnl = state.daily_pnl_total;
    let max_drawdown = dd_tracker.max_drawdown();

    // Create summary to get checksum
    let spec = ScenarioSpec {
        scenario_id: "determinism_test".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: paraphina::sim_eval::Horizon {
            steps,
            dt_seconds: 0.25,
        },
        rng: paraphina::sim_eval::Rng {
            base_seed: seed,
            num_seeds: 1,
        },
        initial_state: paraphina::sim_eval::InitialState::default(),
        market_model: paraphina::sim_eval::MarketModel::default(),
        microstructure_model: paraphina::sim_eval::MicrostructureModel::default(),
        invariants: paraphina::sim_eval::Invariants::default(),
    };

    let build_info = BuildInfo::for_test();
    let summary = RunSummary::new(
        &spec,
        seed,
        build_info,
        final_pnl,
        max_drawdown,
        kill_switch,
    );

    summary.determinism.checksum
}

#[test]
fn test_determinism_same_seed_same_checksum() {
    let seed = 42u64;
    let steps = 100u64;

    // Run twice with identical parameters
    let checksum1 = run_simulation_for_checksum(seed, steps);
    let checksum2 = run_simulation_for_checksum(seed, steps);

    assert_eq!(
        checksum1, checksum2,
        "Same scenario+seed should produce identical checksums"
    );
}

#[test]
fn test_determinism_different_seeds_different_checksums() {
    let steps = 100u64;

    let checksum1 = run_simulation_for_checksum(42, steps);
    let checksum2 = run_simulation_for_checksum(43, steps);

    assert_ne!(
        checksum1, checksum2,
        "Different seeds should produce different checksums"
    );
}

#[test]
fn test_determinism_consistency_across_runs() {
    // Run multiple times to ensure consistency
    let seed = 12345u64;
    let steps = 50u64;

    let checksums: Vec<String> = (0..5)
        .map(|_| run_simulation_for_checksum(seed, steps))
        .collect();

    // All checksums should be identical
    let first = &checksums[0];
    for (i, checksum) in checksums.iter().enumerate() {
        assert_eq!(
            checksum, first,
            "Run {} produced different checksum: {} vs {}",
            i, checksum, first
        );
    }
}

// --------------------------------------------------------------------------
// Test: Seed expansion
// --------------------------------------------------------------------------

#[test]
fn test_seed_expansion() {
    let spec = ScenarioSpec {
        scenario_id: "test".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: paraphina::sim_eval::Horizon::default(),
        rng: paraphina::sim_eval::Rng {
            base_seed: 100,
            num_seeds: 5,
        },
        initial_state: paraphina::sim_eval::InitialState::default(),
        market_model: paraphina::sim_eval::MarketModel::default(),
        microstructure_model: paraphina::sim_eval::MicrostructureModel::default(),
        invariants: paraphina::sim_eval::Invariants::default(),
    };

    let seeds = spec.expand_seeds();

    assert_eq!(seeds.len(), 5);
    assert_eq!(seeds[0], (0, 100));
    assert_eq!(seeds[1], (1, 101));
    assert_eq!(seeds[2], (2, 102));
    assert_eq!(seeds[3], (3, 103));
    assert_eq!(seeds[4], (4, 104));
}

// --------------------------------------------------------------------------
// Test: Validation
// --------------------------------------------------------------------------

#[test]
fn test_validation_empty_scenario_id() {
    let yaml = r#"
scenario_id: ""
scenario_version: 1
"#;
    let result = ScenarioSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("scenario_id"));
}

#[test]
fn test_validation_zero_steps() {
    let yaml = r#"
scenario_id: test
scenario_version: 1
horizon:
  steps: 0
  dt_seconds: 1.0
"#;
    let result = ScenarioSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("steps"));
}

#[test]
fn test_validation_missing_synthetic_config() {
    let yaml = r#"
scenario_id: test
scenario_version: 1
market_model:
  type: synthetic
"#;
    let result = ScenarioSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("synthetic"));
}

// --------------------------------------------------------------------------
// Test: Suite parsing
// --------------------------------------------------------------------------

#[test]
fn test_parse_ci_smoke_suite_yaml() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scenarios/suites/ci_smoke_v1.yaml");

    let suite = SuiteSpec::from_yaml_file(&path).expect("Should parse ci_smoke_v1.yaml");

    assert_eq!(suite.suite_id, "ci_smoke_v1");
    assert_eq!(suite.suite_version, 1);
    assert_eq!(suite.repeat_runs, 2);
    assert_eq!(suite.out_dir, "runs/ci");
    assert_eq!(suite.scenarios.len(), 3);
    assert_eq!(suite.scenarios[0].path, "scenarios/v1/synth_baseline.yaml");
    assert_eq!(suite.scenarios[1].path, "scenarios/v1/synth_jump.yaml");
    assert_eq!(
        suite.scenarios[2].path,
        "scenarios/v1/historical_replay_stub.yaml"
    );
}

#[test]
fn test_suite_parsing_inline() {
    let yaml = r#"
suite_id: test_suite
suite_version: 1
repeat_runs: 3
out_dir: test_out
scenarios:
  - path: scenario1.yaml
  - path: scenario2.yaml
"#;

    let suite = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
    assert_eq!(suite.suite_id, "test_suite");
    assert_eq!(suite.suite_version, 1);
    assert_eq!(suite.repeat_runs, 3);
    assert_eq!(suite.out_dir, "test_out");
    assert_eq!(suite.scenarios.len(), 2);
}

#[test]
fn test_suite_default_values() {
    let yaml = r#"
suite_id: minimal
suite_version: 1
scenarios:
  - path: test.yaml
"#;

    let suite = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
    assert_eq!(suite.repeat_runs, 1); // Default
    assert_eq!(suite.out_dir, "runs/ci"); // Default
}

#[test]
fn test_suite_validation_empty_id() {
    let yaml = r#"
suite_id: ""
suite_version: 1
scenarios:
  - path: test.yaml
"#;

    let result = SuiteSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("suite_id"));
}

#[test]
fn test_suite_validation_empty_scenarios() {
    let yaml = r#"
suite_id: test
suite_version: 1
scenarios: []
"#;

    let result = SuiteSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("scenarios"));
}

#[test]
fn test_suite_validation_zero_repeat_runs() {
    let yaml = r#"
suite_id: test
suite_version: 1
repeat_runs: 0
scenarios:
  - path: test.yaml
"#;

    let result = SuiteSpec::from_yaml_str(yaml);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("repeat_runs"));
}

// --------------------------------------------------------------------------
// Test: Determinism gate (simulates what the suite runner does)
// --------------------------------------------------------------------------

#[test]
fn test_determinism_gate_passes_for_identical_runs() {
    // Simulate the determinism gate: run N times and compare checksums
    let seed = 999u64;
    let steps = 50u64;
    let repeat_runs = 3;

    let checksums: Vec<String> = (0..repeat_runs)
        .map(|_| run_simulation_for_checksum(seed, steps))
        .collect();

    // Verify all checksums are identical (determinism gate passes)
    let first = &checksums[0];
    let all_same = checksums.iter().all(|c| c == first);
    assert!(
        all_same,
        "Determinism gate should pass: all checksums should be identical"
    );
}

#[test]
fn test_determinism_gate_would_fail_for_different_seeds() {
    // Demonstrate that different seeds produce different checksums
    // (this is how a non-deterministic implementation would fail the gate)
    let steps = 50u64;

    let checksum1 = run_simulation_for_checksum(100, steps);
    let checksum2 = run_simulation_for_checksum(101, steps);

    // Different seeds should give different checksums
    assert_ne!(
        checksum1, checksum2,
        "Different seeds should produce different checksums (gate would catch non-determinism)"
    );
}

// --------------------------------------------------------------------------
// Test: Schema completeness (verify RunSummary has all required fields)
// --------------------------------------------------------------------------

#[test]
fn test_schema_completeness_all_fields_present() {
    let spec = ScenarioSpec {
        scenario_id: "schema_test".to_string(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: paraphina::sim_eval::Horizon {
            steps: 100,
            dt_seconds: 0.5,
        },
        rng: paraphina::sim_eval::Rng::default(),
        initial_state: paraphina::sim_eval::InitialState {
            risk_profile: "balanced".to_string(),
            init_q_tao: 0.0,
        },
        market_model: paraphina::sim_eval::MarketModel::default(),
        microstructure_model: paraphina::sim_eval::MicrostructureModel::default(),
        invariants: paraphina::sim_eval::Invariants::default(),
    };

    let summary = RunSummary::new(
        &spec,
        42,
        BuildInfo::for_test(),
        100.0,
        50.0,
        KillSwitchInfo::default(),
    );

    // Verify all required fields per docs/SIM_OUTPUT_SCHEMA.md are present and valid
    // Top-level
    assert!(!summary.scenario_id.is_empty(), "scenario_id required");
    assert!(summary.scenario_version >= 1, "scenario_version >= 1");
    // seed is always present (u64)

    // build_info
    assert!(!summary.build_info.git_sha.is_empty(), "git_sha required");
    // dirty is always present (bool)

    // config
    assert!(
        !summary.config.risk_profile.is_empty(),
        "risk_profile required"
    );
    assert!(summary.config.steps > 0, "steps > 0");
    assert!(summary.config.dt_seconds > 0.0, "dt_seconds > 0");
    // init_q_tao can be any value

    // results
    // final_pnl_usd and max_drawdown_usd can be any values
    // kill_switch fields have defaults

    // determinism
    assert!(
        !summary.determinism.checksum.is_empty(),
        "checksum required"
    );
    assert_eq!(
        summary.determinism.checksum.len(),
        64,
        "checksum must be 64 hex chars (SHA256)"
    );
}
