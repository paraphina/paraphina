// src/sim_eval/mod.rs
//
// Simulation & Evaluation runner module (Option B per ROADMAP.md).
//
// This module provides:
// - ScenarioSpec: Versioned scenario definition (YAML-parsed)
// - SuiteSpec: Suite manifest for grouping scenarios with CI gates
// - RunSummary: Output schema with determinism checksum
// - BuildInfo: Git SHA and dirty flag for reproducibility
// - AblationSet: Ablation harness for research experiments
//
// Design principle: scenarios fully define runs, outputs are CI-comparable.

pub mod ablation;
pub mod output;
pub mod scenario;
pub mod suite;
pub mod summarize;

pub use ablation::{
    print_ablations, AblationError, AblationSet, ABLATION_DESCRIPTIONS, VALID_ABLATION_IDS,
};
pub use output::{
    create_output_dir, create_output_dir_with_ablations, write_build_info, write_config_resolved,
    write_config_resolved_with_ablations, BuildInfo, ConfigResolved, DeterminismInfo,
    KillSwitchInfo, MetricRecord, MetricsWriter, ResultsInfo, RunSummary,
};
pub use scenario::{
    Engine, ExpectKillSwitch, HistoricalStubConfig, Horizon, InitialState, Invariants, MarketModel,
    MarketModelType, MicrostructureModel, PnlLinearityCheck, Rng, ScenarioError, ScenarioSpec,
    SyntheticConfig, SyntheticParams, SyntheticProcess, SCENARIO_SCHEMA_VERSION,
};
pub use suite::{ScenarioRef, SuiteError, SuiteSpec, SUITE_SCHEMA_VERSION};
pub use summarize::{summarize, SummarizeResult, SummaryRow};
