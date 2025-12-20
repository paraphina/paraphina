// src/sim_eval/output.rs
//
// Output schema for simulation runs per docs/SIM_OUTPUT_SCHEMA.md.
//
// Required outputs:
// - run_summary.json: Small, stable summary for CI comparison
// - config_resolved.json: Fully resolved scenario + defaults
// - build_info.json: Git SHA, dirty flag
// - metrics.jsonl: Optional streaming metrics (tick or periodic)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

use super::ablation::AblationSet;
use super::scenario::ScenarioSpec;

/// Output schema version.
pub const OUTPUT_SCHEMA_VERSION: u32 = 1;

/// Build information for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Git SHA of the build.
    pub git_sha: String,
    /// Whether the working directory has uncommitted changes.
    pub dirty: bool,
}

impl BuildInfo {
    /// Capture build info from git.
    pub fn capture() -> Self {
        let (git_sha, dirty) = get_git_info();
        Self { git_sha, dirty }
    }

    /// Create build info for testing.
    pub fn for_test() -> Self {
        Self {
            git_sha: "test_sha".to_string(),
            dirty: false,
        }
    }
}

/// Configuration snapshot in run summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResolved {
    /// Risk profile used.
    pub risk_profile: String,
    /// Initial inventory in TAO.
    pub init_q_tao: f64,
    /// Time delta per step in seconds.
    pub dt_seconds: f64,
    /// Number of steps.
    pub steps: u64,
    /// Active ablations (sorted list of ablation IDs).
    #[serde(default)]
    pub ablations: Vec<String>,
}

impl ConfigResolved {
    /// Create from scenario spec.
    pub fn from_scenario(spec: &ScenarioSpec) -> Self {
        Self {
            risk_profile: spec.initial_state.risk_profile.clone(),
            init_q_tao: spec.initial_state.init_q_tao,
            dt_seconds: spec.horizon.dt_seconds,
            steps: spec.horizon.steps,
            ablations: Vec::new(),
        }
    }

    /// Create from scenario spec with ablations.
    pub fn from_scenario_with_ablations(spec: &ScenarioSpec, ablations: &AblationSet) -> Self {
        Self {
            risk_profile: spec.initial_state.risk_profile.clone(),
            init_q_tao: spec.initial_state.init_q_tao,
            dt_seconds: spec.horizon.dt_seconds,
            steps: spec.horizon.steps,
            ablations: ablations.ablations.clone(),
        }
    }
}

/// Kill switch information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KillSwitchInfo {
    /// Whether the kill switch was triggered.
    pub triggered: bool,
    /// Step at which it was triggered (if any).
    pub step: Option<u64>,
    /// Reason for triggering (if any).
    pub reason: Option<String>,
}

/// Results section of run summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsInfo {
    /// Final PnL in USD.
    pub final_pnl_usd: f64,
    /// Maximum drawdown in USD.
    pub max_drawdown_usd: f64,
    /// Kill switch information.
    pub kill_switch: KillSwitchInfo,
}

/// Determinism information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismInfo {
    /// Checksum of key time series or summary payload.
    pub checksum: String,
}

/// Complete run summary per docs/SIM_OUTPUT_SCHEMA.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Scenario identifier.
    pub scenario_id: String,
    /// Scenario version.
    pub scenario_version: u32,
    /// Seed used for this run.
    pub seed: u64,
    /// Build information.
    pub build_info: BuildInfo,
    /// Configuration snapshot.
    pub config: ConfigResolved,
    /// Results.
    pub results: ResultsInfo,
    /// Determinism information.
    pub determinism: DeterminismInfo,
    /// Active ablations (sorted list of ablation IDs).
    #[serde(default)]
    pub ablations: Vec<String>,
}

impl RunSummary {
    /// Create a new run summary (baseline, no ablations).
    pub fn new(
        scenario: &ScenarioSpec,
        seed: u64,
        build_info: BuildInfo,
        final_pnl_usd: f64,
        max_drawdown_usd: f64,
        kill_switch: KillSwitchInfo,
    ) -> Self {
        Self::with_ablations(
            scenario,
            seed,
            build_info,
            final_pnl_usd,
            max_drawdown_usd,
            kill_switch,
            &AblationSet::new(),
        )
    }

    /// Create a new run summary with ablations.
    pub fn with_ablations(
        scenario: &ScenarioSpec,
        seed: u64,
        build_info: BuildInfo,
        final_pnl_usd: f64,
        max_drawdown_usd: f64,
        kill_switch: KillSwitchInfo,
        ablations: &AblationSet,
    ) -> Self {
        let config = ConfigResolved::from_scenario_with_ablations(scenario, ablations);
        let results = ResultsInfo {
            final_pnl_usd,
            max_drawdown_usd,
            kill_switch,
        };

        // Compute checksum from deterministic fields including ablations
        let checksum = Self::compute_checksum(&config, &results, seed, ablations);

        Self {
            scenario_id: scenario.scenario_id.clone(),
            scenario_version: scenario.scenario_version,
            seed,
            build_info,
            config,
            results,
            determinism: DeterminismInfo { checksum },
            ablations: ablations.ablations.clone(),
        }
    }

    /// Compute checksum from deterministic run data.
    ///
    /// The checksum covers:
    /// - seed
    /// - config (risk_profile, init_q_tao, dt_seconds, steps)
    /// - results (final_pnl_usd, max_drawdown_usd, kill_switch)
    /// - ablations (sorted list of ablation IDs)
    pub fn compute_checksum(
        config: &ConfigResolved,
        results: &ResultsInfo,
        seed: u64,
        ablations: &AblationSet,
    ) -> String {
        let mut hasher = Sha256::new();

        // Seed
        hasher.update(seed.to_le_bytes());

        // Config
        hasher.update(config.risk_profile.as_bytes());
        hasher.update(config.init_q_tao.to_le_bytes());
        hasher.update(config.dt_seconds.to_le_bytes());
        hasher.update(config.steps.to_le_bytes());

        // Ablations (sorted, joined by "|")
        hasher.update(ablations.checksum_bytes());

        // Results (use fixed-precision to ensure determinism)
        // Round to 6 decimal places for floating point stability
        let pnl_rounded = (results.final_pnl_usd * 1_000_000.0).round() as i64;
        let dd_rounded = (results.max_drawdown_usd * 1_000_000.0).round() as i64;
        hasher.update(pnl_rounded.to_le_bytes());
        hasher.update(dd_rounded.to_le_bytes());

        // Kill switch
        hasher.update([results.kill_switch.triggered as u8]);
        if let Some(step) = results.kill_switch.step {
            hasher.update(step.to_le_bytes());
        }

        // Return hex-encoded hash
        let hash = hasher.finalize();
        hex_encode(&hash)
    }

    /// Write the run summary to a file.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}

/// Single metric record for metrics.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Step index.
    pub step: u64,
    /// Current PnL in USD.
    pub pnl_usd: f64,
    /// Current inventory in TAO.
    pub inventory_q_tao: f64,
    /// Effective volatility (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_eff: Option<f64>,
    /// Quoted spread (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread_quoted: Option<f64>,
}

/// Writer for streaming metrics to JSONL file.
pub struct MetricsWriter {
    writer: BufWriter<File>,
}

impl MetricsWriter {
    /// Create a new metrics writer.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self { writer })
    }

    /// Write a single metric record.
    pub fn write_record(&mut self, record: &MetricRecord) -> std::io::Result<()> {
        let json = serde_json::to_string(record)?;
        writeln!(self.writer, "{}", json)?;
        Ok(())
    }

    /// Flush the writer.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Create the output directory structure.
///
/// Creates: runs/<scenario_id>/<git_sha>/<seed>/
pub fn create_output_dir(
    base_dir: &Path,
    scenario_id: &str,
    git_sha: &str,
    seed: u64,
) -> std::io::Result<std::path::PathBuf> {
    let path = base_dir
        .join(scenario_id)
        .join(git_sha)
        .join(seed.to_string());
    fs::create_dir_all(&path)?;
    Ok(path)
}

/// Create the output directory structure with ablation hash.
///
/// Creates: runs/<scenario_id>/<git_sha>/<ablation_hash>/<seed>/
///
/// The ablation_hash is "baseline" for empty ablation set, otherwise a 6-char hex hash.
/// This ensures different ablation sets write to different directories.
pub fn create_output_dir_with_ablations(
    base_dir: &Path,
    scenario_id: &str,
    git_sha: &str,
    seed: u64,
    ablations: &AblationSet,
) -> std::io::Result<std::path::PathBuf> {
    let ablation_hash = ablations.short_hash();
    let path = base_dir
        .join(scenario_id)
        .join(git_sha)
        .join(ablation_hash)
        .join(seed.to_string());
    fs::create_dir_all(&path)?;
    Ok(path)
}

/// Write config_resolved.json (full scenario + defaults).
pub fn write_config_resolved<P: AsRef<Path>>(
    path: P,
    scenario: &ScenarioSpec,
) -> std::io::Result<()> {
    write_config_resolved_with_ablations(path, scenario, &AblationSet::new())
}

/// Write config_resolved.json with ablations included.
pub fn write_config_resolved_with_ablations<P: AsRef<Path>>(
    path: P,
    scenario: &ScenarioSpec,
    ablations: &AblationSet,
) -> std::io::Result<()> {
    // Create a combined structure that includes both scenario and ablations
    #[derive(Serialize)]
    struct ConfigResolvedFull<'a> {
        #[serde(flatten)]
        scenario: &'a ScenarioSpec,
        ablations: &'a Vec<String>,
    }

    let full = ConfigResolvedFull {
        scenario,
        ablations: &ablations.ablations,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &full)?;
    Ok(())
}

/// Write build_info.json.
pub fn write_build_info<P: AsRef<Path>>(path: P, build_info: &BuildInfo) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, build_info)?;
    Ok(())
}

/// Get git SHA and dirty status.
fn get_git_info() -> (String, bool) {
    // Try to get git SHA
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Check if dirty
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false);

    (sha, dirty)
}

/// Hex-encode bytes.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_scenario() -> ScenarioSpec {
        ScenarioSpec {
            scenario_id: "test_scenario".to_string(),
            scenario_version: 1,
            engine: super::super::scenario::Engine::RlSimEnv,
            horizon: super::super::scenario::Horizon {
                steps: 100,
                dt_seconds: 0.5,
            },
            rng: super::super::scenario::Rng {
                base_seed: 42,
                num_seeds: 1,
            },
            initial_state: super::super::scenario::InitialState {
                risk_profile: "balanced".to_string(),
                init_q_tao: 0.0,
            },
            market_model: super::super::scenario::MarketModel::default(),
            microstructure_model: super::super::scenario::MicrostructureModel::default(),
            invariants: super::super::scenario::Invariants::default(),
        }
    }

    #[test]
    fn test_run_summary_creation() {
        let scenario = make_test_scenario();
        let build_info = BuildInfo::for_test();
        let kill_switch = KillSwitchInfo::default();

        let summary = RunSummary::new(&scenario, 42, build_info, 100.0, 50.0, kill_switch);

        assert_eq!(summary.scenario_id, "test_scenario");
        assert_eq!(summary.scenario_version, 1);
        assert_eq!(summary.seed, 42);
        assert_eq!(summary.config.risk_profile, "balanced");
        assert_eq!(summary.results.final_pnl_usd, 100.0);
        assert_eq!(summary.results.max_drawdown_usd, 50.0);
        assert!(!summary.determinism.checksum.is_empty());
    }

    #[test]
    fn test_checksum_determinism() {
        let scenario = make_test_scenario();
        let build_info = BuildInfo::for_test();
        let kill_switch = KillSwitchInfo::default();

        let summary1 = RunSummary::new(
            &scenario,
            42,
            build_info.clone(),
            100.0,
            50.0,
            kill_switch.clone(),
        );
        let summary2 = RunSummary::new(&scenario, 42, build_info, 100.0, 50.0, kill_switch);

        assert_eq!(summary1.determinism.checksum, summary2.determinism.checksum);
    }

    #[test]
    fn test_checksum_differs_for_different_results() {
        let scenario = make_test_scenario();
        let build_info = BuildInfo::for_test();
        let kill_switch = KillSwitchInfo::default();

        let summary1 = RunSummary::new(
            &scenario,
            42,
            build_info.clone(),
            100.0,
            50.0,
            kill_switch.clone(),
        );
        let summary2 = RunSummary::new(&scenario, 42, build_info, 200.0, 50.0, kill_switch);

        assert_ne!(summary1.determinism.checksum, summary2.determinism.checksum);
    }

    #[test]
    fn test_checksum_differs_with_ablations() {
        let scenario = make_test_scenario();
        let build_info = BuildInfo::for_test();
        let kill_switch = KillSwitchInfo::default();

        // Baseline (no ablations)
        let summary_baseline = RunSummary::new(
            &scenario,
            42,
            build_info.clone(),
            100.0,
            50.0,
            kill_switch.clone(),
        );

        // With ablations
        let ablations = AblationSet::from_ids(&["disable_vol_floor".to_string()]).unwrap();
        let summary_ablated = RunSummary::with_ablations(
            &scenario,
            42,
            build_info,
            100.0,
            50.0,
            kill_switch,
            &ablations,
        );

        // Checksums should differ due to ablations
        assert_ne!(
            summary_baseline.determinism.checksum,
            summary_ablated.determinism.checksum
        );

        // Ablations should be in the summary
        assert!(summary_baseline.ablations.is_empty());
        assert_eq!(summary_ablated.ablations, vec!["disable_vol_floor"]);
    }

    #[test]
    fn test_config_resolved_includes_ablations() {
        let scenario = make_test_scenario();
        let ablations = AblationSet::from_ids(&[
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ])
        .unwrap();

        let config = ConfigResolved::from_scenario_with_ablations(&scenario, &ablations);

        assert_eq!(config.ablations.len(), 2);
        assert!(config
            .ablations
            .contains(&"disable_toxicity_gate".to_string()));
        assert!(config.ablations.contains(&"disable_vol_floor".to_string()));
    }
}
