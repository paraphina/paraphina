// src/sim_eval/suite.rs
//
// Suite manifest specification for CI smoke testing.
//
// A suite groups multiple scenarios together with:
// - Determinism checking via repeated runs
// - Common output directory
// - CI gate enforcement
// - Inline env_overrides for adversarial/stress testing

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Current suite schema version.
pub const SUITE_SCHEMA_VERSION: u32 = 2;

/// Reference to a scenario within a suite.
///
/// Supports two modes:
/// 1. File-based: `path: scenarios/v1/foo.yaml`
/// 2. Inline: `id: smoke_test`, `env_overrides: { KEY: VAL }`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioRef {
    /// Path-based scenario reference.
    Path {
        /// Path to the scenario YAML file (relative to repo root).
        path: String,
    },
    /// Inline scenario with env_overrides (runs as subprocess).
    Inline(InlineScenario),
}

/// Inline scenario definition for suites with env_overrides.
///
/// These scenarios run as subprocesses with the merged environment:
/// - Base: parent process environment
/// - Layer: suite-level env_overrides
/// - Top: scenario-level env_overrides (wins on conflict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineScenario {
    /// Unique scenario identifier (used for output directory naming).
    pub id: String,
    /// Random seed for determinism.
    #[serde(default)]
    pub seed: u64,
    /// Risk profile name.
    #[serde(default = "default_profile")]
    pub profile: String,
    /// Environment variable overrides (scenario-level).
    /// Uses BTreeMap for stable ordering.
    #[serde(default)]
    pub env_overrides: BTreeMap<String, String>,
    /// Optional adversarial parameters (not used by sim_eval, but preserved).
    #[serde(default)]
    pub adversarial_params: BTreeMap<String, serde_yaml::Value>,
}

fn default_profile() -> String {
    "balanced".to_string()
}

/// Invariants block for inline scenarios (optional).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuiteInvariants {
    /// Whether evidence pack must verify.
    #[serde(default = "default_true")]
    pub evidence_pack_valid: bool,
    /// Expected kill switch behavior: "allowed", "never", "always".
    #[serde(default = "default_expect_kill_switch")]
    pub expect_kill_switch: String,
}

fn default_true() -> bool {
    true
}

fn default_expect_kill_switch() -> String {
    "allowed".to_string()
}

/// Suite manifest specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSpec {
    /// Unique suite identifier.
    pub suite_id: String,
    /// Schema version.
    pub suite_version: u32,
    /// Number of times to repeat each run for determinism checking.
    /// If > 1, all runs must produce identical checksums.
    #[serde(default = "default_repeat_runs")]
    pub repeat_runs: u32,
    /// Output directory for run artifacts.
    #[serde(default = "default_out_dir")]
    pub out_dir: String,
    /// Suite-level environment variable overrides.
    /// Applied to all scenarios before scenario-level overrides.
    /// Uses BTreeMap for stable ordering.
    #[serde(default)]
    pub env_overrides: BTreeMap<String, String>,
    /// List of scenario references (path-based or inline).
    pub scenarios: Vec<ScenarioRef>,
    /// Optional invariants block.
    #[serde(default)]
    pub invariants: Option<SuiteInvariants>,
}

fn default_repeat_runs() -> u32 {
    1
}

fn default_out_dir() -> String {
    "runs/ci".to_string()
}

impl ScenarioRef {
    /// Get the scenario path (for path-based scenarios).
    pub fn path(&self) -> Option<&str> {
        match self {
            ScenarioRef::Path { path } => Some(path.as_str()),
            ScenarioRef::Inline(_) => None,
        }
    }

    /// Get the inline scenario (for inline scenarios).
    pub fn inline(&self) -> Option<&InlineScenario> {
        match self {
            ScenarioRef::Path { .. } => None,
            ScenarioRef::Inline(inline) => Some(inline),
        }
    }

    /// Check if this is an inline scenario.
    pub fn is_inline(&self) -> bool {
        matches!(self, ScenarioRef::Inline(_))
    }

    /// Get a display identifier for this scenario.
    pub fn display_id(&self) -> String {
        match self {
            ScenarioRef::Path { path } => path.clone(),
            ScenarioRef::Inline(inline) => inline.id.clone(),
        }
    }
}

impl InlineScenario {
    /// Merge environment overrides: suite-level + scenario-level.
    /// Scenario-level overrides win on conflict.
    pub fn merge_env(&self, suite_env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        let mut merged = suite_env.clone();
        for (k, v) in &self.env_overrides {
            merged.insert(k.clone(), v.clone());
        }
        merged
    }
}

impl SuiteSpec {
    /// Load a suite from a YAML file.
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, SuiteError> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|e| SuiteError::IoError {
            path: path.as_ref().display().to_string(),
            source: e.to_string(),
        })?;
        Self::from_yaml_str(&contents)
    }

    /// Parse a suite from a YAML string.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, SuiteError> {
        let spec: SuiteSpec = serde_yaml::from_str(yaml).map_err(|e| SuiteError::ParseError {
            source: e.to_string(),
        })?;
        spec.validate()?;
        Ok(spec)
    }

    /// Validate the suite specification.
    pub fn validate(&self) -> Result<(), SuiteError> {
        if self.suite_id.is_empty() {
            return Err(SuiteError::ValidationError {
                field: "suite_id".to_string(),
                message: "suite_id cannot be empty".to_string(),
            });
        }

        if self.suite_version == 0 {
            return Err(SuiteError::ValidationError {
                field: "suite_version".to_string(),
                message: "suite_version must be >= 1".to_string(),
            });
        }

        if self.repeat_runs == 0 {
            return Err(SuiteError::ValidationError {
                field: "repeat_runs".to_string(),
                message: "repeat_runs must be >= 1".to_string(),
            });
        }

        if self.scenarios.is_empty() {
            return Err(SuiteError::ValidationError {
                field: "scenarios".to_string(),
                message: "scenarios list cannot be empty".to_string(),
            });
        }

        for (i, scenario) in self.scenarios.iter().enumerate() {
            match scenario {
                ScenarioRef::Path { path } => {
                    if path.is_empty() {
                        return Err(SuiteError::ValidationError {
                            field: format!("scenarios[{}].path", i),
                            message: "scenario path cannot be empty".to_string(),
                        });
                    }
                }
                ScenarioRef::Inline(inline) => {
                    if inline.id.is_empty() {
                        return Err(SuiteError::ValidationError {
                            field: format!("scenarios[{}].id", i),
                            message: "inline scenario id cannot be empty".to_string(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if any scenario is inline (has env_overrides).
    pub fn has_inline_scenarios(&self) -> bool {
        self.scenarios.iter().any(|s| s.is_inline())
    }
}

/// Errors that can occur when working with suites.
#[derive(Debug, Clone)]
pub enum SuiteError {
    IoError { path: String, source: String },
    ParseError { source: String },
    ValidationError { field: String, message: String },
}

impl std::fmt::Display for SuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuiteError::IoError { path, source } => {
                write!(f, "Failed to read suite file '{}': {}", path, source)
            }
            SuiteError::ParseError { source } => {
                write!(f, "Failed to parse suite YAML: {}", source)
            }
            SuiteError::ValidationError { field, message } => {
                write!(f, "Suite validation error in '{}': {}", field, message)
            }
        }
    }
}

impl std::error::Error for SuiteError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ci_smoke_suite() {
        let yaml = r#"
suite_id: ci_smoke_v1
suite_version: 1

repeat_runs: 2

out_dir: runs/ci

scenarios:
  - path: scenarios/v1/synth_baseline.yaml
  - path: scenarios/v1/synth_jump.yaml
"#;

        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.suite_id, "ci_smoke_v1");
        assert_eq!(spec.suite_version, 1);
        assert_eq!(spec.repeat_runs, 2);
        assert_eq!(spec.out_dir, "runs/ci");
        assert_eq!(spec.scenarios.len(), 2);
        assert_eq!(
            spec.scenarios[0].path(),
            Some("scenarios/v1/synth_baseline.yaml")
        );
        assert_eq!(
            spec.scenarios[1].path(),
            Some("scenarios/v1/synth_jump.yaml")
        );
        assert!(!spec.has_inline_scenarios());
    }

    #[test]
    fn test_default_values() {
        let yaml = r#"
suite_id: test
suite_version: 1
scenarios:
  - path: test.yaml
"#;
        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.repeat_runs, 1);
        assert_eq!(spec.out_dir, "runs/ci");
    }

    #[test]
    fn test_validation_empty_suite_id() {
        let yaml = r#"
suite_id: ""
suite_version: 1
scenarios:
  - path: test.yaml
"#;
        let result = SuiteSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_empty_scenarios() {
        let yaml = r#"
suite_id: test
suite_version: 1
scenarios: []
"#;
        let result = SuiteSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_zero_repeat_runs() {
        let yaml = r#"
suite_id: test
suite_version: 1
repeat_runs: 0
scenarios:
  - path: test.yaml
"#;
        let result = SuiteSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_inline_scenario_with_env_overrides() {
        let yaml = r#"
suite_id: adversarial_test
suite_version: 1
repeat_runs: 2
out_dir: runs/adv

scenarios:
  - id: smoke_balanced_0
    seed: 42
    profile: balanced
    env_overrides:
      PARAPHINA_RISK_PROFILE: balanced
      PARAPHINA_VOL_REF: "0.2000"
      PARAPHINA_MM_SIZE_ETA: "0.5000"
"#;

        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.suite_id, "adversarial_test");
        assert_eq!(spec.scenarios.len(), 1);
        assert!(spec.has_inline_scenarios());

        let scenario = &spec.scenarios[0];
        assert!(scenario.is_inline());
        let inline = scenario.inline().expect("Should be inline");
        assert_eq!(inline.id, "smoke_balanced_0");
        assert_eq!(inline.seed, 42);
        assert_eq!(inline.profile, "balanced");
        assert_eq!(inline.env_overrides.len(), 3);
        assert_eq!(
            inline.env_overrides.get("PARAPHINA_RISK_PROFILE"),
            Some(&"balanced".to_string())
        );
        assert_eq!(
            inline.env_overrides.get("PARAPHINA_VOL_REF"),
            Some(&"0.2000".to_string())
        );
    }

    #[test]
    fn test_parse_suite_level_env_overrides() {
        let yaml = r#"
suite_id: env_test
suite_version: 1
out_dir: runs/env

env_overrides:
  PARAPHINA_DAILY_LOSS_LIMIT: "500.0"
  PARAPHINA_RISK_PROFILE: conservative

scenarios:
  - id: scenario_1
    seed: 1
    env_overrides:
      PARAPHINA_RISK_PROFILE: aggressive
      PARAPHINA_VOL_REF: "0.15"
"#;

        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.env_overrides.len(), 2);
        assert_eq!(
            spec.env_overrides.get("PARAPHINA_DAILY_LOSS_LIMIT"),
            Some(&"500.0".to_string())
        );

        let inline = spec.scenarios[0].inline().expect("Should be inline");
        let merged = inline.merge_env(&spec.env_overrides);

        // Scenario-level overrides suite-level
        assert_eq!(
            merged.get("PARAPHINA_RISK_PROFILE"),
            Some(&"aggressive".to_string())
        );
        // Suite-level is preserved if not overridden
        assert_eq!(
            merged.get("PARAPHINA_DAILY_LOSS_LIMIT"),
            Some(&"500.0".to_string())
        );
        // Scenario-level addition
        assert_eq!(merged.get("PARAPHINA_VOL_REF"), Some(&"0.15".to_string()));
    }

    #[test]
    fn test_mixed_path_and_inline_scenarios() {
        let yaml = r#"
suite_id: mixed_test
suite_version: 1
out_dir: runs/mixed

scenarios:
  - path: scenarios/v1/synth_baseline.yaml
  - id: inline_1
    seed: 99
    env_overrides:
      PARAPHINA_RISK_PROFILE: aggressive
"#;

        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        assert_eq!(spec.scenarios.len(), 2);
        assert!(spec.has_inline_scenarios());

        assert!(spec.scenarios[0].path().is_some());
        assert!(!spec.scenarios[0].is_inline());

        assert!(spec.scenarios[1].inline().is_some());
        assert!(spec.scenarios[1].is_inline());
    }

    #[test]
    fn test_validation_empty_inline_id() {
        let yaml = r#"
suite_id: test
suite_version: 1
scenarios:
  - id: ""
    seed: 1
"#;
        let result = SuiteSpec::from_yaml_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_env_overrides_btreemap_ordering() {
        let yaml = r#"
suite_id: order_test
suite_version: 1
scenarios:
  - id: test
    env_overrides:
      ZEBRA: "z"
      APPLE: "a"
      MANGO: "m"
"#;
        let spec = SuiteSpec::from_yaml_str(yaml).expect("Should parse");
        let inline = spec.scenarios[0].inline().expect("Should be inline");

        // BTreeMap keys are sorted
        let keys: Vec<_> = inline.env_overrides.keys().collect();
        assert_eq!(keys, vec!["APPLE", "MANGO", "ZEBRA"]);
    }
}
