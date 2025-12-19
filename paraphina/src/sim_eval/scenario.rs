// src/sim_eval/scenario.rs
//
// Scenario specification parsing and validation per docs/SIM_EVAL_SPEC.md.
//
// A scenario fully defines a reproducible simulation run:
// - scenario_id + scenario_version for tracking
// - engine selection (rl_sim_env, core_sim, replay_stub)
// - horizon (steps or duration)
// - RNG seeds
// - initial state (risk profile, inventory)
// - market model (synthetic or historical stub)
// - microstructure model (fees, latency)
// - invariants (kill switch expectations, etc.)

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Current scenario schema version.
pub const SCENARIO_SCHEMA_VERSION: u32 = 1;

/// Engine type for simulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    /// Use the Gym-style SimEnv from rl module.
    #[default]
    RlSimEnv,
    /// Use the core simulation engine directly.
    CoreSim,
    /// Replay stub for historical data (not implemented in v1).
    ReplayStub,
}

/// Horizon configuration for a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Horizon {
    /// Number of simulation steps.
    pub steps: u64,
    /// Time delta per step in seconds.
    pub dt_seconds: f64,
}

impl Default for Horizon {
    fn default() -> Self {
        Self {
            steps: 1000,
            dt_seconds: 1.0,
        }
    }
}

/// RNG configuration for seed expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rng {
    /// Base seed for reproducibility.
    pub base_seed: u64,
    /// Number of seeds to run (runner expands as base_seed + k).
    pub num_seeds: u32,
}

impl Default for Rng {
    fn default() -> Self {
        Self {
            base_seed: 42,
            num_seeds: 1,
        }
    }
}

/// Initial state configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialState {
    /// Risk profile name (maps to PARAPHINA_RISK_PROFILE values).
    pub risk_profile: String,
    /// Initial inventory in TAO.
    pub init_q_tao: f64,
}

impl Default for InitialState {
    fn default() -> Self {
        Self {
            risk_profile: "balanced".to_string(),
            init_q_tao: 0.0,
        }
    }
}

/// Synthetic market process type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticProcess {
    /// Geometric Brownian Motion.
    #[default]
    Gbm,
    /// Jump diffusion stub (simplified).
    JumpDiffusionStub,
}

/// Synthetic market model parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyntheticParams {
    /// Volatility (annualized or per-step depending on model).
    #[serde(default)]
    pub vol: f64,
    /// Drift rate.
    #[serde(default)]
    pub drift: f64,
    /// Jump intensity (for jump diffusion).
    #[serde(default)]
    pub jump_intensity: f64,
    /// Jump size sigma (for jump diffusion).
    #[serde(default)]
    pub jump_sigma: f64,
}

/// Synthetic market model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticConfig {
    /// Process type.
    pub process: SyntheticProcess,
    /// Process parameters.
    #[serde(default)]
    pub params: SyntheticParams,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            process: SyntheticProcess::Gbm,
            params: SyntheticParams {
                vol: 0.015,
                drift: 0.0,
                jump_intensity: 0.0,
                jump_sigma: 0.0,
            },
        }
    }
}

/// Historical stub configuration (placeholder).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoricalStubConfig {
    /// Dataset identifier (no data in repo; pointer only).
    pub dataset_id: String,
}

/// Market model type selector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarketModelType {
    #[default]
    Synthetic,
    HistoricalStub,
}

/// Market model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketModel {
    /// Market model type.
    #[serde(rename = "type")]
    pub model_type: MarketModelType,
    /// Synthetic configuration (if type == synthetic).
    #[serde(default)]
    pub synthetic: Option<SyntheticConfig>,
    /// Historical stub configuration (if type == historical_stub).
    #[serde(default)]
    pub historical_stub: Option<HistoricalStubConfig>,
}

impl Default for MarketModel {
    fn default() -> Self {
        Self {
            model_type: MarketModelType::Synthetic,
            synthetic: Some(SyntheticConfig::default()),
            historical_stub: None,
        }
    }
}

/// Microstructure model (fees and latency).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureModel {
    /// Maker fee in basis points.
    pub fees_bps_maker: f64,
    /// Taker fee in basis points.
    pub fees_bps_taker: f64,
    /// Latency in milliseconds (constant in v1).
    pub latency_ms: f64,
}

impl Default for MicrostructureModel {
    fn default() -> Self {
        Self {
            fees_bps_maker: 0.0,
            fees_bps_taker: 0.0,
            latency_ms: 0.0,
        }
    }
}

/// Kill switch expectation for invariant checking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectKillSwitch {
    /// Kill switch must always trigger.
    Always,
    /// Kill switch must never trigger.
    Never,
    /// Kill switch may or may not trigger.
    #[default]
    Allowed,
}

/// PnL linearity check setting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PnlLinearityCheck {
    Enabled,
    #[default]
    Disabled,
}

/// Invariants (assertions) for the scenario.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Invariants {
    /// Kill switch expectation.
    #[serde(default)]
    pub expect_kill_switch: ExpectKillSwitch,
    /// PnL linearity check (for q0 sweeps).
    #[serde(default)]
    pub pnl_linearity_check: PnlLinearityCheck,
}

/// Complete scenario specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSpec {
    /// Unique scenario identifier.
    pub scenario_id: String,
    /// Schema version (starts at 1).
    pub scenario_version: u32,
    /// Engine to use.
    #[serde(default)]
    pub engine: Engine,
    /// Horizon configuration.
    #[serde(default)]
    pub horizon: Horizon,
    /// RNG configuration.
    #[serde(default)]
    pub rng: Rng,
    /// Initial state.
    #[serde(default)]
    pub initial_state: InitialState,
    /// Market model.
    #[serde(default)]
    pub market_model: MarketModel,
    /// Microstructure model.
    #[serde(default)]
    pub microstructure_model: MicrostructureModel,
    /// Invariants.
    #[serde(default)]
    pub invariants: Invariants,
}

impl ScenarioSpec {
    /// Load a scenario from a YAML file.
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, ScenarioError> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|e| ScenarioError::IoError {
            path: path.as_ref().display().to_string(),
            source: e.to_string(),
        })?;
        Self::from_yaml_str(&contents)
    }

    /// Parse a scenario from a YAML string.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, ScenarioError> {
        let spec: ScenarioSpec =
            serde_yaml::from_str(yaml).map_err(|e| ScenarioError::ParseError {
                source: e.to_string(),
            })?;
        spec.validate()?;
        Ok(spec)
    }

    /// Validate the scenario specification.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        // Validate scenario_id
        if self.scenario_id.is_empty() {
            return Err(ScenarioError::ValidationError {
                field: "scenario_id".to_string(),
                message: "scenario_id cannot be empty".to_string(),
            });
        }

        // Validate scenario_version
        if self.scenario_version == 0 {
            return Err(ScenarioError::ValidationError {
                field: "scenario_version".to_string(),
                message: "scenario_version must be >= 1".to_string(),
            });
        }

        // Validate horizon
        if self.horizon.steps == 0 {
            return Err(ScenarioError::ValidationError {
                field: "horizon.steps".to_string(),
                message: "steps must be > 0".to_string(),
            });
        }
        if self.horizon.dt_seconds <= 0.0 {
            return Err(ScenarioError::ValidationError {
                field: "horizon.dt_seconds".to_string(),
                message: "dt_seconds must be > 0".to_string(),
            });
        }

        // Validate rng
        if self.rng.num_seeds == 0 {
            return Err(ScenarioError::ValidationError {
                field: "rng.num_seeds".to_string(),
                message: "num_seeds must be >= 1".to_string(),
            });
        }

        // Validate market model consistency
        match self.market_model.model_type {
            MarketModelType::Synthetic => {
                if self.market_model.synthetic.is_none() {
                    return Err(ScenarioError::ValidationError {
                        field: "market_model.synthetic".to_string(),
                        message: "synthetic config required when type is synthetic".to_string(),
                    });
                }
            }
            MarketModelType::HistoricalStub => {
                if self.market_model.historical_stub.is_none() {
                    return Err(ScenarioError::ValidationError {
                        field: "market_model.historical_stub".to_string(),
                        message: "historical_stub config required when type is historical_stub"
                            .to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Expand seeds into a list of (seed_index, actual_seed) pairs.
    pub fn expand_seeds(&self) -> Vec<(u32, u64)> {
        (0..self.rng.num_seeds)
            .map(|k| (k, self.rng.base_seed.wrapping_add(k as u64)))
            .collect()
    }
}

/// Errors that can occur when working with scenarios.
#[derive(Debug, Clone)]
pub enum ScenarioError {
    IoError { path: String, source: String },
    ParseError { source: String },
    ValidationError { field: String, message: String },
}

impl std::fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScenarioError::IoError { path, source } => {
                write!(f, "Failed to read scenario file '{}': {}", path, source)
            }
            ScenarioError::ParseError { source } => {
                write!(f, "Failed to parse scenario YAML: {}", source)
            }
            ScenarioError::ValidationError { field, message } => {
                write!(f, "Scenario validation error in '{}': {}", field, message)
            }
        }
    }
}

impl std::error::Error for ScenarioError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_synth_baseline() {
        let yaml = r#"
scenario_id: synth_baseline
scenario_version: 1

engine: rl_sim_env

horizon:
  steps: 2000
  dt_seconds: 0.25

rng:
  base_seed: 42
  num_seeds: 5

initial_state:
  risk_profile: balanced
  init_q_tao: 0.0

market_model:
  type: synthetic
  synthetic:
    process: gbm
    params:
      vol: 0.015
      drift: 0.0

microstructure_model:
  fees_bps_maker: 0.0
  fees_bps_taker: 0.0
  latency_ms: 0.0

invariants:
  expect_kill_switch: allowed
  pnl_linearity_check: disabled
"#;

        let spec = ScenarioSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.scenario_id, "synth_baseline");
        assert_eq!(spec.scenario_version, 1);
        assert_eq!(spec.engine, Engine::RlSimEnv);
        assert_eq!(spec.horizon.steps, 2000);
        assert_eq!(spec.horizon.dt_seconds, 0.25);
        assert_eq!(spec.rng.base_seed, 42);
        assert_eq!(spec.rng.num_seeds, 5);
        assert_eq!(spec.initial_state.risk_profile, "balanced");
        assert_eq!(spec.initial_state.init_q_tao, 0.0);
        assert_eq!(spec.market_model.model_type, MarketModelType::Synthetic);
        assert!(spec.market_model.synthetic.is_some());
        let synth = spec.market_model.synthetic.as_ref().unwrap();
        assert_eq!(synth.process, SyntheticProcess::Gbm);
        assert!((synth.params.vol - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_expand_seeds() {
        let yaml = r#"
scenario_id: test
scenario_version: 1
rng:
  base_seed: 100
  num_seeds: 3
"#;
        let spec = ScenarioSpec::from_yaml_str(yaml).expect("Should parse");
        let seeds = spec.expand_seeds();
        assert_eq!(seeds, vec![(0, 100), (1, 101), (2, 102)]);
    }

    #[test]
    fn test_validation_empty_id() {
        let yaml = r#"
scenario_id: ""
scenario_version: 1
"#;
        let result = ScenarioSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_zero_version() {
        let yaml = r#"
scenario_id: test
scenario_version: 0
"#;
        let result = ScenarioSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }
}
