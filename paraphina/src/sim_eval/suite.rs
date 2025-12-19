// src/sim_eval/suite.rs
//
// Suite manifest specification for CI smoke testing.
//
// A suite groups multiple scenarios together with:
// - Determinism checking via repeated runs
// - Common output directory
// - CI gate enforcement

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Current suite schema version.
pub const SUITE_SCHEMA_VERSION: u32 = 1;

/// Reference to a scenario within a suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioRef {
    /// Path to the scenario YAML file (relative to repo root).
    pub path: String,
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
    /// List of scenario references.
    pub scenarios: Vec<ScenarioRef>,
}

fn default_repeat_runs() -> u32 {
    1
}

fn default_out_dir() -> String {
    "runs/ci".to_string()
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
            if scenario.path.is_empty() {
                return Err(SuiteError::ValidationError {
                    field: format!("scenarios[{}].path", i),
                    message: "scenario path cannot be empty".to_string(),
                });
            }
        }

        Ok(())
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
        assert_eq!(spec.scenarios[0].path, "scenarios/v1/synth_baseline.yaml");
        assert_eq!(spec.scenarios[1].path, "scenarios/v1/synth_jump.yaml");
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
}
