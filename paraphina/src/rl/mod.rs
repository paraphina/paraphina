// src/rl/mod.rs
//
// RL-0 and RL-1 Foundations for Paraphina (per ROADMAP.md and WHITEPAPER.md Appendix A).
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
//
// Design principle: "Policy learns decisions, engine enforces safety"

pub mod domain_rand;
pub mod observation;
pub mod policy;
pub mod runner;
pub mod sim_env;
pub mod telemetry;

// Re-exports for convenience
pub use domain_rand::{DomainRandConfig, DomainRandSample, DomainRandSampler};
pub use observation::{Observation, VenueObservation, OBS_VERSION};
pub use policy::{HeuristicPolicy, NoopPolicy, Policy, PolicyAction, HEURISTIC_POLICY_VERSION};
pub use runner::{EpisodeConfig, EpisodeSummary, ShadowRunner, TerminationReason};
pub use sim_env::{SimEnv, StepResult, VecEnv};
pub use telemetry::{RLTelemetry, RewardComponents, RewardWeights, TickRecord};
