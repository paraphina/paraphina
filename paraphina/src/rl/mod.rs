// src/rl/mod.rs
//
// RL-0, RL-1, and RL-2 Foundations for Paraphina (per ROADMAP.md and WHITEPAPER.md Appendix A).
//
// This module provides the interfaces and instrumentation needed to evolve
// Paraphina from a deterministic heuristic strategy into an RL-ready system,
// while preserving safety invariants.
//
// Key components:
// - Observation: Versioned, serializable state snapshot for policy input
// - Policy: Trait for strategy decision-making (heuristic or learned)
// - PolicyAction: Bounded control surface for policy outputs
// - RLTelemetry: Logging of policy inputs/outputs and reward components
// - ShadowRunner: Parallel execution of baseline + shadow policies
// - DomainRandConfig: Domain randomisation for training environments (RL-1)
// - SimEnv: Gym-style simulation environment (RL-1)
// - ActionEncoding: Versioned action vector encoding for BC (RL-2)
// - TrajectoryCollector: Dataset generation from heuristic policy (RL-2)
//
// Design principle: "Policy learns decisions, engine enforces safety"

pub mod action_encoding;
pub mod domain_rand;
pub mod observation;
pub mod policy;
pub mod research_budgets;
pub mod runner;
pub mod safe_pipeline;
pub mod safety;
pub mod sim_env;
pub mod telemetry;
pub mod trajectory;

// Re-exports for convenience
pub use action_encoding::{decode_action, encode_action, ActionEncodingSpec, ACTION_VERSION};
pub use domain_rand::{DomainRandConfig, DomainRandSample, DomainRandSampler};
pub use observation::{Observation, VenueObservation, OBS_VERSION};
pub use policy::{HeuristicPolicy, NoopPolicy, Policy, PolicyAction, HEURISTIC_POLICY_VERSION};
pub use research_budgets::{alignment_budget_for_profile, ResearchAlignmentBudget};
pub use runner::{EpisodeConfig, EpisodeSummary, ShadowRunner, TerminationReason};
pub use safe_pipeline::{run_safe_pipeline, SafePipelineSummary};
pub use safety::{SafetyLayer, SafetyResult};
pub use sim_env::{SimEnv, StepResult, VecEnv};
pub use telemetry::{RLTelemetry, RewardComponents, RewardWeights, TickRecord};
pub use trajectory::{
    TrajectoryCollector, TrajectoryMetadata, TrajectoryRecord, TrajectoryWriter, TRAJECTORY_VERSION,
};
