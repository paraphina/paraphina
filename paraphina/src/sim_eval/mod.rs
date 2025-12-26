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
// - EvidencePack: Audit-grade output bundle with provenance and integrity
//
// Design principle: scenarios fully define runs, outputs are CI-comparable.

pub mod ablation;
pub mod env_override;
pub mod evidence_pack;
pub mod evidence_pack_verify;
pub mod output;
pub mod report;
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
pub use report::{
    check_gates, compute_stats, generate_report, load_run_summaries, match_runs,
    print_console_summary, run_report, write_json_report, write_markdown_report, MatchedRun,
    ReportArgs, ReportResult, ResearchReport, ResultsMinimal, RunSummaryMinimal, VariantReport,
    VariantStats,
};
pub use scenario::{
    Engine, ExpectKillSwitch, HistoricalStubConfig, Horizon, InitialState, Invariants, MarketModel,
    MarketModelType, MicrostructureModel, PnlLinearityCheck, Rng, ScenarioError, ScenarioSpec,
    SyntheticConfig, SyntheticParams, SyntheticProcess, SCENARIO_SCHEMA_VERSION,
};
pub use suite::{
    InlineScenario, ScenarioRef, SuiteError, SuiteInvariants, SuiteSpec, SUITE_SCHEMA_VERSION,
};
pub use summarize::{
    get_summary_rows, summarize, summarize_with_format, OutputFormat, SummarizeResult, SummaryRow,
};

// Evidence Pack v1 (per docs/EVIDENCE_PACK.md)
pub use evidence_pack::{write_evidence_pack, write_root_evidence_pack};

// Evidence Pack v1 verification
pub use evidence_pack_verify::{
    verify_evidence_pack_dir, verify_evidence_pack_tree, EvidencePackVerificationReport,
};

// Scoped environment variable overrides for inline scenarios
pub use env_override::{parse_env_overrides, with_env_overrides};
