// src/sim_eval/report.rs
//
// Report command: generates combined baseline-vs-ablations research reports
// in Markdown and JSON format with regression gating support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::summarize::discover_run_summaries;

// =============================================================================
// Data Structures
// =============================================================================

/// A minimal run summary parsed from run_summary.json for report matching.
/// Uses #[serde(default)] to gracefully handle missing fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummaryMinimal {
    /// Scenario identifier.
    pub scenario_id: String,
    /// Seed used for this run.
    pub seed: u64,
    /// Results section.
    #[serde(default)]
    pub results: ResultsMinimal,
    /// Active ablations.
    #[serde(default)]
    pub ablations: Vec<String>,
    /// Source path (not serialized from JSON, set after parsing).
    #[serde(skip)]
    pub source_path: PathBuf,
}

/// Minimal results info for report matching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultsMinimal {
    /// Final PnL in USD.
    #[serde(default)]
    pub final_pnl_usd: f64,
    /// Maximum drawdown in USD.
    #[serde(default)]
    pub max_drawdown_usd: f64,
    /// Kill switch information.
    #[serde(default)]
    pub kill_switch: KillSwitchMinimal,
}

/// Minimal kill switch info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KillSwitchMinimal {
    /// Whether the kill switch was triggered.
    #[serde(default)]
    pub triggered: bool,
    /// Step at which it was triggered.
    #[serde(default)]
    pub step: Option<u64>,
    /// Reason for triggering.
    #[serde(default)]
    pub reason: Option<String>,
}

/// A matched pair of baseline and variant runs.
#[derive(Debug, Clone, Serialize)]
pub struct MatchedRun {
    /// Scenario identifier.
    pub scenario_id: String,
    /// Random seed.
    pub seed: u64,
    /// Baseline final PnL.
    pub baseline_pnl: f64,
    /// Variant final PnL.
    pub variant_pnl: f64,
    /// Delta: variant_pnl - baseline_pnl.
    pub delta_pnl: f64,
    /// Baseline source path.
    pub baseline_path: PathBuf,
    /// Variant source path.
    pub variant_path: PathBuf,
}

/// Aggregated statistics for a variant comparison.
#[derive(Debug, Clone, Serialize)]
pub struct VariantStats {
    /// Variant name.
    pub variant_name: String,
    /// Number of matched runs.
    pub count: usize,
    /// Mean baseline PnL.
    pub mean_baseline_pnl: f64,
    /// Mean variant PnL.
    pub mean_variant_pnl: f64,
    /// Mean delta (variant - baseline).
    pub mean_delta: f64,
    /// Worst (most negative) delta.
    pub worst_delta: f64,
    /// Best (most positive) delta.
    pub best_delta: f64,
    /// Standard deviation of deltas.
    pub std_delta: f64,
}

/// Full report for a single variant comparison.
#[derive(Debug, Clone, Serialize)]
pub struct VariantReport {
    /// Aggregated statistics.
    pub stats: VariantStats,
    /// All matched runs (sorted by scenario_id, then seed).
    pub matched_runs: Vec<MatchedRun>,
}

/// Complete research report containing all variant comparisons.
#[derive(Debug, Clone, Serialize)]
pub struct ResearchReport {
    /// Baseline directory path.
    pub baseline_dir: PathBuf,
    /// Variant reports keyed by variant name.
    pub variants: HashMap<String, VariantReport>,
    /// Report generation timestamp (ISO 8601).
    pub generated_at: String,
    /// Whether regression gates passed.
    pub gates_passed: bool,
    /// Gate failure messages (if any).
    pub gate_failures: Vec<String>,
}

/// Arguments for the report command.
#[derive(Debug, Clone)]
pub struct ReportArgs {
    /// Baseline runs directory.
    pub baseline_dir: PathBuf,
    /// Variant directories: (name, path).
    pub variants: Vec<(String, PathBuf)>,
    /// Output path for Markdown report.
    pub out_md: PathBuf,
    /// Output path for JSON report.
    pub out_json: PathBuf,
    /// Maximum allowed regression in USD (gate).
    pub gate_max_regression_usd: Option<f64>,
    /// Maximum allowed regression as percentage (gate).
    pub gate_max_regression_pct: Option<f64>,
}

/// Result of the report command.
#[derive(Debug)]
pub enum ReportResult {
    /// Success with number of variants analyzed.
    Success { variants: usize, gates_passed: bool },
    /// No baseline runs found.
    NoBaselineRuns,
    /// Error during processing.
    Error(String),
}

// =============================================================================
// Core Logic
// =============================================================================

/// Parse all run_summary.json files under a directory into minimal structs.
pub fn load_run_summaries(dir: &Path) -> std::io::Result<Vec<RunSummaryMinimal>> {
    let paths = discover_run_summaries(dir)?;
    let mut summaries = Vec::with_capacity(paths.len());

    for path in paths {
        if let Some(summary) = parse_run_summary_minimal(&path) {
            summaries.push(summary);
        }
    }

    Ok(summaries)
}

/// Parse a single run_summary.json into a minimal struct.
fn parse_run_summary_minimal(path: &Path) -> Option<RunSummaryMinimal> {
    let contents = fs::read_to_string(path).ok()?;
    let mut summary: RunSummaryMinimal = serde_json::from_str(&contents).ok()?;
    summary.source_path = path.to_path_buf();
    Some(summary)
}

/// Create a lookup key from scenario_id and seed.
fn make_key(scenario_id: &str, seed: u64) -> String {
    format!("{}:{}", scenario_id, seed)
}

/// Match baseline runs to variant runs by (scenario_id, seed).
pub fn match_runs(
    baseline: &[RunSummaryMinimal],
    variant: &[RunSummaryMinimal],
) -> Vec<MatchedRun> {
    // Build a map of baseline runs by (scenario_id, seed)
    let baseline_map: HashMap<String, &RunSummaryMinimal> = baseline
        .iter()
        .map(|s| (make_key(&s.scenario_id, s.seed), s))
        .collect();

    // Match variant runs to baseline
    let mut matched: Vec<MatchedRun> = variant
        .iter()
        .filter_map(|v| {
            let key = make_key(&v.scenario_id, v.seed);
            baseline_map.get(&key).map(|b| MatchedRun {
                scenario_id: v.scenario_id.clone(),
                seed: v.seed,
                baseline_pnl: b.results.final_pnl_usd,
                variant_pnl: v.results.final_pnl_usd,
                delta_pnl: v.results.final_pnl_usd - b.results.final_pnl_usd,
                baseline_path: b.source_path.clone(),
                variant_path: v.source_path.clone(),
            })
        })
        .collect();

    // Sort deterministically by scenario_id, then seed
    matched.sort_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then_with(|| a.seed.cmp(&b.seed))
    });

    matched
}

/// Compute aggregated statistics from matched runs.
pub fn compute_stats(variant_name: &str, matched: &[MatchedRun]) -> VariantStats {
    if matched.is_empty() {
        return VariantStats {
            variant_name: variant_name.to_string(),
            count: 0,
            mean_baseline_pnl: 0.0,
            mean_variant_pnl: 0.0,
            mean_delta: 0.0,
            worst_delta: 0.0,
            best_delta: 0.0,
            std_delta: 0.0,
        };
    }

    let count = matched.len();
    let sum_baseline: f64 = matched.iter().map(|m| m.baseline_pnl).sum();
    let sum_variant: f64 = matched.iter().map(|m| m.variant_pnl).sum();
    let sum_delta: f64 = matched.iter().map(|m| m.delta_pnl).sum();

    let mean_baseline_pnl = sum_baseline / count as f64;
    let mean_variant_pnl = sum_variant / count as f64;
    let mean_delta = sum_delta / count as f64;

    let worst_delta = matched
        .iter()
        .map(|m| m.delta_pnl)
        .fold(f64::INFINITY, f64::min);
    let best_delta = matched
        .iter()
        .map(|m| m.delta_pnl)
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute standard deviation
    let variance: f64 = matched
        .iter()
        .map(|m| {
            let diff = m.delta_pnl - mean_delta;
            diff * diff
        })
        .sum::<f64>()
        / count as f64;
    let std_delta = variance.sqrt();

    VariantStats {
        variant_name: variant_name.to_string(),
        count,
        mean_baseline_pnl,
        mean_variant_pnl,
        mean_delta,
        worst_delta,
        best_delta,
        std_delta,
    }
}

/// Check regression gates and return failure messages.
pub fn check_gates(
    stats: &VariantStats,
    max_regression_usd: Option<f64>,
    max_regression_pct: Option<f64>,
) -> Vec<String> {
    let mut failures = Vec::new();

    // Check USD regression gate
    if let Some(max_usd) = max_regression_usd {
        // worst_delta is the most negative delta, so if it's worse than -max_usd, it fails
        if stats.worst_delta < -max_usd {
            failures.push(format!(
                "Variant '{}': worst delta ${:.2} exceeds max regression ${:.2}",
                stats.variant_name, stats.worst_delta, max_usd
            ));
        }
    }

    // Check percentage regression gate
    if let Some(max_pct) = max_regression_pct {
        // Calculate regression percentage based on mean baseline
        if stats.mean_baseline_pnl.abs() > 1e-9 {
            let regression_pct = (stats.worst_delta / stats.mean_baseline_pnl.abs()) * 100.0;
            if regression_pct < -max_pct {
                failures.push(format!(
                    "Variant '{}': worst regression {:.2}% exceeds max {:.2}%",
                    stats.variant_name,
                    regression_pct.abs(),
                    max_pct
                ));
            }
        }
    }

    failures
}

/// Generate the complete research report.
pub fn generate_report(args: &ReportArgs) -> Result<ResearchReport, String> {
    // Load baseline runs
    let baseline_runs = load_run_summaries(&args.baseline_dir)
        .map_err(|e| format!("Failed to load baseline runs: {}", e))?;

    if baseline_runs.is_empty() {
        return Err("No baseline runs found".to_string());
    }

    let mut variants = HashMap::new();
    let mut all_gate_failures = Vec::new();

    // Process each variant
    for (name, variant_dir) in &args.variants {
        let variant_runs = load_run_summaries(variant_dir)
            .map_err(|e| format!("Failed to load variant '{}' runs: {}", name, e))?;

        let matched = match_runs(&baseline_runs, &variant_runs);
        let stats = compute_stats(name, &matched);

        // Check gates
        let gate_failures = check_gates(
            &stats,
            args.gate_max_regression_usd,
            args.gate_max_regression_pct,
        );
        all_gate_failures.extend(gate_failures);

        variants.insert(
            name.clone(),
            VariantReport {
                stats,
                matched_runs: matched,
            },
        );
    }

    let gates_passed = all_gate_failures.is_empty();

    Ok(ResearchReport {
        baseline_dir: args.baseline_dir.clone(),
        variants,
        generated_at: chrono_stub(),
        gates_passed,
        gate_failures: all_gate_failures,
    })
}

/// Simple timestamp without external dependency.
fn chrono_stub() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s since epoch", duration.as_secs())
}

// =============================================================================
// Output Writers
// =============================================================================

/// Write the JSON report to a file.
pub fn write_json_report(report: &ResearchReport, path: &Path) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, report)?;
    Ok(())
}

/// Write the Markdown report to a file.
pub fn write_markdown_report(report: &ResearchReport, path: &Path) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# Research Report: Baseline vs Ablations")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "**Baseline Directory:** `{}`",
        report.baseline_dir.display()
    )?;
    writeln!(writer, "**Generated:** {}", report.generated_at)?;
    writeln!(writer)?;

    // Gate status
    if report.gates_passed {
        writeln!(writer, "## ✓ All Gates Passed")?;
    } else {
        writeln!(writer, "## ✗ Gate Failures")?;
        writeln!(writer)?;
        for failure in &report.gate_failures {
            writeln!(writer, "- {}", failure)?;
        }
    }
    writeln!(writer)?;

    // Sort variant names for deterministic output
    let mut variant_names: Vec<_> = report.variants.keys().collect();
    variant_names.sort();

    // Summary table
    writeln!(writer, "## Summary")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| Variant | Runs | Mean Baseline PnL | Mean Variant PnL | Mean Δ | Worst Δ | Best Δ |"
    )?;
    writeln!(
        writer,
        "|---------|------|-------------------|------------------|--------|---------|--------|"
    )?;

    for name in &variant_names {
        if let Some(vr) = report.variants.get(*name) {
            writeln!(
                writer,
                "| {} | {} | ${:.2} | ${:.2} | ${:.2} | ${:.2} | ${:.2} |",
                vr.stats.variant_name,
                vr.stats.count,
                vr.stats.mean_baseline_pnl,
                vr.stats.mean_variant_pnl,
                vr.stats.mean_delta,
                vr.stats.worst_delta,
                vr.stats.best_delta,
            )?;
        }
    }
    writeln!(writer)?;

    // Detailed per-variant sections
    for name in &variant_names {
        if let Some(vr) = report.variants.get(*name) {
            writeln!(writer, "## Variant: {}", name)?;
            writeln!(writer)?;
            writeln!(
                writer,
                "| Scenario | Seed | Baseline PnL | Variant PnL | Δ PnL |"
            )?;
            writeln!(
                writer,
                "|----------|------|--------------|-------------|-------|"
            )?;

            for run in &vr.matched_runs {
                writeln!(
                    writer,
                    "| {} | {} | ${:.2} | ${:.2} | ${:.2} |",
                    run.scenario_id, run.seed, run.baseline_pnl, run.variant_pnl, run.delta_pnl,
                )?;
            }
            writeln!(writer)?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Print a concise console summary.
pub fn print_console_summary(report: &ResearchReport) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("RESEARCH REPORT: Baseline vs Ablations");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Baseline: {}", report.baseline_dir.display());
    println!();

    // Sort variant names for deterministic output
    let mut variant_names: Vec<_> = report.variants.keys().collect();
    variant_names.sort();

    for name in &variant_names {
        if let Some(vr) = report.variants.get(*name) {
            println!(
                "  {} | runs={} mean_Δ=${:.2} worst_Δ=${:.2} best_Δ=${:.2}",
                name,
                vr.stats.count,
                vr.stats.mean_delta,
                vr.stats.worst_delta,
                vr.stats.best_delta,
            );
        }
    }

    println!();
    if report.gates_passed {
        println!("✓ ALL GATES PASSED");
    } else {
        println!("✗ {} GATE FAILURES:", report.gate_failures.len());
        for failure in &report.gate_failures {
            println!("  - {}", failure);
        }
    }
}

/// Main entry point for the report command.
pub fn run_report(args: ReportArgs) -> ReportResult {
    match generate_report(&args) {
        Ok(report) => {
            // Write outputs
            if let Err(e) = write_json_report(&report, &args.out_json) {
                return ReportResult::Error(format!("Failed to write JSON report: {}", e));
            }
            if let Err(e) = write_markdown_report(&report, &args.out_md) {
                return ReportResult::Error(format!("Failed to write Markdown report: {}", e));
            }

            // Print console summary
            print_console_summary(&report);

            println!();
            println!("Output written to:");
            println!("  Markdown: {}", args.out_md.display());
            println!("  JSON:     {}", args.out_json.display());

            ReportResult::Success {
                variants: report.variants.len(),
                gates_passed: report.gates_passed,
            }
        }
        Err(e) if e.contains("No baseline runs") => ReportResult::NoBaselineRuns,
        Err(e) => ReportResult::Error(e),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    /// Create a test run_summary.json file.
    fn create_test_run_summary(
        dir: &Path,
        scenario_id: &str,
        seed: u64,
        pnl: f64,
        ablations: &[&str],
    ) -> PathBuf {
        let summary = serde_json::json!({
            "scenario_id": scenario_id,
            "scenario_version": 1,
            "seed": seed,
            "build_info": {
                "git_sha": "abc123def456",
                "dirty": false
            },
            "config": {
                "risk_profile": "balanced",
                "init_q_tao": 0.0,
                "dt_seconds": 0.5,
                "steps": 100
            },
            "results": {
                "final_pnl_usd": pnl,
                "max_drawdown_usd": 10.0,
                "kill_switch": {
                    "triggered": false,
                    "step": null,
                    "reason": null
                }
            },
            "determinism": {
                "checksum": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            },
            "ablations": ablations
        });

        fs::create_dir_all(dir).unwrap();
        let path = dir.join("run_summary.json");
        let mut file = File::create(&path).unwrap();
        write!(file, "{}", serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        path
    }

    #[test]
    fn test_match_runs_basic() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create baseline runs
        let baseline_dir = base.join("baseline");
        create_test_run_summary(
            &baseline_dir.join("scenario_a").join("sha").join("42"),
            "scenario_a",
            42,
            100.0,
            &[],
        );
        create_test_run_summary(
            &baseline_dir.join("scenario_a").join("sha").join("99"),
            "scenario_a",
            99,
            150.0,
            &[],
        );

        // Create variant runs
        let variant_dir = base.join("variant");
        create_test_run_summary(
            &variant_dir.join("scenario_a").join("sha").join("42"),
            "scenario_a",
            42,
            110.0,
            &["disable_vol_floor"],
        );
        create_test_run_summary(
            &variant_dir.join("scenario_a").join("sha").join("99"),
            "scenario_a",
            99,
            140.0,
            &["disable_vol_floor"],
        );

        // Load and match
        let baseline = load_run_summaries(&baseline_dir).unwrap();
        let variant = load_run_summaries(&variant_dir).unwrap();

        assert_eq!(baseline.len(), 2);
        assert_eq!(variant.len(), 2);

        let matched = match_runs(&baseline, &variant);
        assert_eq!(matched.len(), 2);

        // Verify sorting (by scenario_id, then seed)
        assert_eq!(matched[0].seed, 42);
        assert_eq!(matched[1].seed, 99);

        // Verify deltas
        assert!((matched[0].delta_pnl - 10.0).abs() < 1e-9); // 110 - 100
        assert!((matched[1].delta_pnl - (-10.0)).abs() < 1e-9); // 140 - 150
    }

    #[test]
    fn test_compute_stats() {
        let matched = vec![
            MatchedRun {
                scenario_id: "s1".to_string(),
                seed: 1,
                baseline_pnl: 100.0,
                variant_pnl: 110.0,
                delta_pnl: 10.0,
                baseline_path: PathBuf::new(),
                variant_path: PathBuf::new(),
            },
            MatchedRun {
                scenario_id: "s1".to_string(),
                seed: 2,
                baseline_pnl: 200.0,
                variant_pnl: 180.0,
                delta_pnl: -20.0,
                baseline_path: PathBuf::new(),
                variant_path: PathBuf::new(),
            },
        ];

        let stats = compute_stats("test_variant", &matched);

        assert_eq!(stats.count, 2);
        assert!((stats.mean_baseline_pnl - 150.0).abs() < 1e-9);
        assert!((stats.mean_variant_pnl - 145.0).abs() < 1e-9);
        assert!((stats.mean_delta - (-5.0)).abs() < 1e-9);
        assert!((stats.worst_delta - (-20.0)).abs() < 1e-9);
        assert!((stats.best_delta - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_gate_regression_usd() {
        let stats = VariantStats {
            variant_name: "test".to_string(),
            count: 2,
            mean_baseline_pnl: 100.0,
            mean_variant_pnl: 90.0,
            mean_delta: -10.0,
            worst_delta: -25.0,
            best_delta: 5.0,
            std_delta: 15.0,
        };

        // Should pass with high threshold
        let failures = check_gates(&stats, Some(30.0), None);
        assert!(failures.is_empty());

        // Should fail with low threshold
        let failures = check_gates(&stats, Some(20.0), None);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("worst delta"));
    }

    #[test]
    fn test_gate_regression_pct() {
        let stats = VariantStats {
            variant_name: "test".to_string(),
            count: 2,
            mean_baseline_pnl: 100.0,
            mean_variant_pnl: 90.0,
            mean_delta: -10.0,
            worst_delta: -25.0, // 25% of baseline
            best_delta: 5.0,
            std_delta: 15.0,
        };

        // Should pass with high threshold
        let failures = check_gates(&stats, None, Some(30.0));
        assert!(failures.is_empty());

        // Should fail with low threshold
        let failures = check_gates(&stats, None, Some(20.0));
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("regression"));
    }

    #[test]
    fn test_full_report_generation() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create baseline runs
        let baseline_dir = base.join("baseline");
        create_test_run_summary(
            &baseline_dir.join("scenario_a").join("sha").join("42"),
            "scenario_a",
            42,
            100.0,
            &[],
        );
        create_test_run_summary(
            &baseline_dir.join("scenario_b").join("sha").join("42"),
            "scenario_b",
            42,
            200.0,
            &[],
        );

        // Create variant runs
        let variant_dir = base.join("variant_ablation");
        create_test_run_summary(
            &variant_dir.join("scenario_a").join("sha").join("42"),
            "scenario_a",
            42,
            105.0,
            &["disable_vol_floor"],
        );
        create_test_run_summary(
            &variant_dir.join("scenario_b").join("sha").join("42"),
            "scenario_b",
            42,
            195.0,
            &["disable_vol_floor"],
        );

        let args = ReportArgs {
            baseline_dir: baseline_dir.clone(),
            variants: vec![("ablation".to_string(), variant_dir)],
            out_md: base.join("report.md"),
            out_json: base.join("report.json"),
            gate_max_regression_usd: Some(10.0),
            gate_max_regression_pct: None,
        };

        let report = generate_report(&args).unwrap();

        assert!(report.gates_passed);
        assert_eq!(report.variants.len(), 1);

        let ablation_report = report.variants.get("ablation").unwrap();
        assert_eq!(ablation_report.stats.count, 2);
        assert!((ablation_report.stats.mean_delta - 0.0).abs() < 1e-9); // (+5 + -5) / 2 = 0
    }

    #[test]
    fn test_missing_fields_handled_gracefully() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create a minimal JSON with missing optional fields
        let dir = base.join("minimal").join("scenario").join("sha").join("1");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run_summary.json");
        let json = serde_json::json!({
            "scenario_id": "minimal",
            "seed": 1
            // No results, no ablations, no other fields
        });
        let mut file = File::create(&path).unwrap();
        write!(file, "{}", serde_json::to_string_pretty(&json).unwrap()).unwrap();

        // Should parse without crashing
        let summaries = load_run_summaries(base).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].scenario_id, "minimal");
        assert_eq!(summaries[0].seed, 1);
        assert_eq!(summaries[0].results.final_pnl_usd, 0.0); // Default
    }

    #[test]
    fn test_deterministic_sorting() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create runs in non-sorted order
        let baseline_dir = base.join("baseline");
        create_test_run_summary(
            &baseline_dir.join("z_scenario").join("sha").join("99"),
            "z_scenario",
            99,
            100.0,
            &[],
        );
        create_test_run_summary(
            &baseline_dir.join("a_scenario").join("sha").join("1"),
            "a_scenario",
            1,
            100.0,
            &[],
        );
        create_test_run_summary(
            &baseline_dir.join("a_scenario").join("sha").join("50"),
            "a_scenario",
            50,
            100.0,
            &[],
        );

        let variant_dir = base.join("variant");
        create_test_run_summary(
            &variant_dir.join("z_scenario").join("sha").join("99"),
            "z_scenario",
            99,
            110.0,
            &[],
        );
        create_test_run_summary(
            &variant_dir.join("a_scenario").join("sha").join("1"),
            "a_scenario",
            1,
            110.0,
            &[],
        );
        create_test_run_summary(
            &variant_dir.join("a_scenario").join("sha").join("50"),
            "a_scenario",
            50,
            110.0,
            &[],
        );

        let baseline = load_run_summaries(&baseline_dir).unwrap();
        let variant = load_run_summaries(&variant_dir).unwrap();
        let matched = match_runs(&baseline, &variant);

        // Verify deterministic sorting: a_scenario:1, a_scenario:50, z_scenario:99
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0].scenario_id, "a_scenario");
        assert_eq!(matched[0].seed, 1);
        assert_eq!(matched[1].scenario_id, "a_scenario");
        assert_eq!(matched[1].seed, 50);
        assert_eq!(matched[2].scenario_id, "z_scenario");
        assert_eq!(matched[2].seed, 99);
    }
}
