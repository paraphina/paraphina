// src/sim_eval/summarize.rs
//
// Summarize command: discovers and aggregates run_summary.json files
// into a readable table output (fixed-width text or Markdown).

use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::output::RunSummary;

/// Output format for summarize command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Fixed-width text table (default).
    #[default]
    Text,
    /// Markdown table.
    Markdown,
}

/// A parsed run entry for the summary table.
#[derive(Debug, Clone)]
pub struct SummaryRow {
    /// Suite identifier (extracted from path or empty).
    pub suite_id: String,
    /// Scenario identifier.
    pub scenario_id: String,
    /// Random seed used.
    pub seed: u64,
    /// Ablations applied (default empty).
    pub ablations: Vec<String>,
    /// Run status (OK, KILLED).
    pub status: String,
    /// Whether kill switch was triggered (for sorting/filtering).
    pub kill_switch_triggered: bool,
    /// Final PnL value.
    pub pnl: f64,
    /// Maximum drawdown in USD.
    pub max_drawdown_usd: f64,
    /// Determinism checksum.
    pub checksum: String,
    /// Relative path to the run_summary.json file.
    pub path: PathBuf,
}

/// Format the display status for a run.
///
/// Returns:
/// - "KILL@<N>" if the kill switch was triggered at step N
/// - "KILL" if triggered but step is unknown
/// - "OK" if not triggered
fn display_status(triggered: bool, step: Option<u64>) -> String {
    if triggered {
        match step {
            Some(n) => format!("KILL@{}", n),
            None => "KILL".to_string(),
        }
    } else {
        "OK".to_string()
    }
}

impl SummaryRow {
    /// Create a row from a parsed RunSummary and path info.
    pub fn from_run_summary(summary: &RunSummary, path: PathBuf, suite_id: String) -> Self {
        let kill_triggered = summary.results.kill_switch.triggered;
        let kill_step = summary.results.kill_switch.step;
        let status = display_status(kill_triggered, kill_step);

        Self {
            suite_id,
            scenario_id: summary.scenario_id.clone(),
            seed: summary.seed,
            ablations: summary.ablations.clone(),
            status,
            kill_switch_triggered: kill_triggered,
            pnl: summary.results.final_pnl_usd,
            max_drawdown_usd: summary.results.max_drawdown_usd,
            checksum: summary.determinism.checksum.clone(),
            path,
        }
    }

    /// Create a row from a generic JSON Value with fallback parsing.
    pub fn from_json_value(value: &Value, path: PathBuf, suite_id: String) -> Option<Self> {
        let scenario_id = value.get("scenario_id")?.as_str()?.to_string();
        let seed = value.get("seed")?.as_u64()?;

        // Try to get PnL from results.final_pnl_usd or fallback locations
        let pnl = value
            .get("results")
            .and_then(|r| r.get("final_pnl_usd"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Try to get max_drawdown_usd
        let max_drawdown_usd = value
            .get("results")
            .and_then(|r| r.get("max_drawdown_usd"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Try to get checksum
        let checksum = value
            .get("determinism")
            .and_then(|d| d.get("checksum"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Try to get kill switch status and step from various locations
        let kill_switch = value.get("results").and_then(|r| r.get("kill_switch"));

        let kill_triggered = kill_switch
            .and_then(|k| k.get("triggered"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Try to get kill step from kill_switch.step
        let kill_step = kill_switch
            .and_then(|k| k.get("step"))
            .and_then(|v| v.as_u64());

        // Check if status is already formatted as "KILL@<N>" string
        let status = if let Some(existing_status) = value.get("status").and_then(|v| v.as_str()) {
            // If there's a pre-formatted status string, use it
            existing_status.to_string()
        } else {
            display_status(kill_triggered, kill_step)
        };

        // Try to get ablations from top-level or config
        let ablations = value
            .get("ablations")
            .and_then(|a| a.as_array())
            .or_else(|| {
                value
                    .get("config")
                    .and_then(|c| c.get("ablations"))
                    .and_then(|a| a.as_array())
            })
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Some(Self {
            suite_id,
            scenario_id,
            seed,
            ablations,
            status,
            kill_switch_triggered: kill_triggered,
            pnl,
            max_drawdown_usd,
            checksum,
            path,
        })
    }
}

/// Recursively discover all run_summary.json files under a directory.
pub fn discover_run_summaries(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    discover_recursive(dir, &mut results)?;
    Ok(results)
}

fn discover_recursive(dir: &Path, results: &mut Vec<PathBuf>) -> io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            discover_recursive(&path, results)?;
        } else if path
            .file_name()
            .map(|n| n == "run_summary.json")
            .unwrap_or(false)
        {
            results.push(path);
        }
    }

    Ok(())
}

/// Try to extract suite_id from the directory path.
///
/// Expected structure: runs/<suite_id>/<scenario_id>/<git_sha>/<seed>/run_summary.json
/// or: runs/<scenario_id>/<git_sha>/<seed>/run_summary.json (no suite_id)
///
/// We look for common suite directory names like "ci" or patterns.
fn extract_suite_id(summary_path: &Path, runs_dir: &Path) -> String {
    // Get the relative path from runs_dir to the summary
    if let Ok(rel_path) = summary_path.strip_prefix(runs_dir) {
        let components: Vec<_> = rel_path.components().collect();
        // If path is like: ci/scenario_id/sha/seed/run_summary.json (5 components)
        // then first component might be suite_id
        if components.len() >= 5 {
            if let Some(std::path::Component::Normal(name)) = components.first() {
                return name.to_string_lossy().to_string();
            }
        }
    }
    String::new()
}

/// Parse a single run_summary.json file into a SummaryRow.
pub fn parse_run_summary(path: &Path, runs_dir: &Path) -> Option<SummaryRow> {
    let contents = fs::read_to_string(path).ok()?;
    let suite_id = extract_suite_id(path, runs_dir);

    // Make path relative to runs_dir
    let relative_path = path
        .strip_prefix(runs_dir)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|_| path.to_path_buf());

    // First try to parse as RunSummary struct
    if let Ok(summary) = serde_json::from_str::<RunSummary>(&contents) {
        return Some(SummaryRow::from_run_summary(
            &summary,
            relative_path,
            suite_id,
        ));
    }

    // Fallback: parse as generic JSON Value
    if let Ok(value) = serde_json::from_str::<Value>(&contents) {
        return SummaryRow::from_json_value(&value, relative_path, suite_id);
    }

    None
}

/// Parse all discovered run summaries.
pub fn parse_all_summaries(paths: &[PathBuf], runs_dir: &Path) -> Vec<SummaryRow> {
    paths
        .iter()
        .filter_map(|p| parse_run_summary(p, runs_dir))
        .collect()
}

/// Sort rows by scenario_id asc, seed asc (stable sorting).
pub fn sort_rows(rows: &mut [SummaryRow]) {
    rows.sort_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then_with(|| a.seed.cmp(&b.seed))
    });
}

/// Sort rows by suite_id, scenario_id, ablations, seed (legacy sorting).
pub fn sort_rows_legacy(rows: &mut [SummaryRow]) {
    rows.sort_by(|a, b| {
        a.suite_id
            .cmp(&b.suite_id)
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
            .then_with(|| a.ablations.cmp(&b.ablations))
            .then_with(|| a.seed.cmp(&b.seed))
    });
}

/// Format ablations as a display string.
fn format_ablations(ablations: &[String]) -> String {
    if ablations.is_empty() {
        "(none)".to_string()
    } else {
        ablations.join(", ")
    }
}

/// Shorten checksum for display (first 12 chars).
fn shorten_checksum(checksum: &str) -> &str {
    if checksum.len() >= 12 {
        &checksum[..12]
    } else {
        checksum
    }
}

/// Print the summary as a fixed-width text table.
pub fn print_table<W: Write>(rows: &[SummaryRow], mut writer: W) -> io::Result<()> {
    // Header
    writeln!(
        writer,
        "{:<15} {:<25} {:<10} {:<20} {:<8} {:>12} {:>12} {:<14} PATH",
        "SUITE_ID", "SCENARIO_ID", "SEED", "ABLATIONS", "STATUS", "PNL", "MAX_DD", "CHECKSUM",
    )?;
    writeln!(
        writer,
        "{:-<15} {:-<25} {:-<10} {:-<20} {:-<8} {:->12} {:->12} {:-<14} {:-<30}",
        "", "", "", "", "", "", "", "", ""
    )?;

    // Rows
    for row in rows {
        writeln!(
            writer,
            "{:<15} {:<25} {:<10} {:<20} {:<8} {:>12.2} {:>12.2} {:<14} {}",
            row.suite_id,
            row.scenario_id,
            row.seed,
            format_ablations(&row.ablations),
            row.status,
            row.pnl,
            row.max_drawdown_usd,
            shorten_checksum(&row.checksum),
            row.path.display()
        )?;
    }

    Ok(())
}

/// Print the summary as a Markdown table.
///
/// Columns: scenario_id, seed, ablations, final_pnl_usd, max_drawdown_usd,
/// status (OK/KILLED), checksum (shortened), path.
pub fn print_markdown_table<W: Write>(rows: &[SummaryRow], mut writer: W) -> io::Result<()> {
    // Header
    writeln!(
        writer,
        "| scenario_id | seed | ablations | final_pnl_usd | max_drawdown_usd | status | checksum | path |"
    )?;
    writeln!(
        writer,
        "|-------------|------|-----------|---------------|------------------|--------|----------|------|"
    )?;

    // Rows
    for row in rows {
        writeln!(
            writer,
            "| {} | {} | {} | {:.2} | {:.2} | {} | `{}` | `{}` |",
            row.scenario_id,
            row.seed,
            format_ablations(&row.ablations),
            row.pnl,
            row.max_drawdown_usd,
            row.status,
            shorten_checksum(&row.checksum),
            row.path.display()
        )?;
    }

    Ok(())
}

/// Result of the summarize command.
#[derive(Debug)]
pub enum SummarizeResult {
    /// Success with number of rows found.
    Success(usize),
    /// No run_summary.json files found.
    NoFilesFound,
    /// Files found but none could be parsed.
    NoParseable,
}

/// Main entry point: discover, parse, sort, and print summaries (default text format).
pub fn summarize<W: Write>(runs_dir: &Path, writer: W) -> io::Result<SummarizeResult> {
    summarize_with_format(runs_dir, writer, OutputFormat::Text)
}

/// Main entry point with format selection: discover, parse, sort, and print summaries.
pub fn summarize_with_format<W: Write>(
    runs_dir: &Path,
    writer: W,
    format: OutputFormat,
) -> io::Result<SummarizeResult> {
    // Discover all run_summary.json files
    let paths = discover_run_summaries(runs_dir)?;

    if paths.is_empty() {
        return Ok(SummarizeResult::NoFilesFound);
    }

    // Parse all summaries
    let mut rows = parse_all_summaries(&paths, runs_dir);

    if rows.is_empty() {
        return Ok(SummarizeResult::NoParseable);
    }

    // Sort by scenario_id, seed (stable sorting per requirements)
    sort_rows(&mut rows);

    // Print in requested format
    match format {
        OutputFormat::Text => print_table(&rows, writer)?,
        OutputFormat::Markdown => print_markdown_table(&rows, writer)?,
    }

    Ok(SummarizeResult::Success(rows.len()))
}

/// Get parsed rows without printing (useful for testing and other tools).
pub fn get_summary_rows(runs_dir: &Path) -> io::Result<Vec<SummaryRow>> {
    let paths = discover_run_summaries(runs_dir)?;
    let mut rows = parse_all_summaries(&paths, runs_dir);
    sort_rows(&mut rows);
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_summary(dir: &Path, scenario_id: &str, seed: u64, pnl: f64) -> PathBuf {
        create_test_summary_full(dir, scenario_id, seed, pnl, 10.0, &[], false)
    }

    fn create_test_summary_full(
        dir: &Path,
        scenario_id: &str,
        seed: u64,
        pnl: f64,
        max_drawdown: f64,
        ablations: &[&str],
        kill_switch_triggered: bool,
    ) -> PathBuf {
        let summary = serde_json::json!({
            "scenario_id": scenario_id,
            "scenario_version": 1,
            "seed": seed,
            "build_info": {
                "git_sha": "abc123def456789012345678901234567890abcd",
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
                "max_drawdown_usd": max_drawdown,
                "kill_switch": {
                    "triggered": kill_switch_triggered,
                    "step": if kill_switch_triggered { Some(50) } else { None },
                    "reason": if kill_switch_triggered { Some("max_loss") } else { None }
                }
            },
            "determinism": {
                "checksum": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            },
            "ablations": ablations
        });

        let path = dir.join("run_summary.json");
        let mut file = File::create(&path).unwrap();
        write!(file, "{}", serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        path
    }

    #[test]
    fn test_discover_run_summaries() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create nested structure
        let dir1 = base.join("scenario1").join("sha1").join("42");
        fs::create_dir_all(&dir1).unwrap();
        create_test_summary(&dir1, "scenario1", 42, 100.0);

        let dir2 = base.join("scenario2").join("sha2").join("99");
        fs::create_dir_all(&dir2).unwrap();
        create_test_summary(&dir2, "scenario2", 99, 200.0);

        let paths = discover_run_summaries(base).unwrap();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_parse_run_summary() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        let dir = base.join("test_scenario").join("sha").join("42");
        fs::create_dir_all(&dir).unwrap();
        let path = create_test_summary(&dir, "test_scenario", 42, 123.45);

        let row = parse_run_summary(&path, base).unwrap();
        assert_eq!(row.scenario_id, "test_scenario");
        assert_eq!(row.seed, 42);
        assert_eq!(row.pnl, 123.45);
        assert_eq!(row.status, "OK");
    }

    #[test]
    fn test_sort_rows() {
        let mut rows = vec![
            SummaryRow {
                suite_id: "b".to_string(),
                scenario_id: "z_scenario".to_string(),
                seed: 1,
                ablations: vec![],
                status: "OK".to_string(),
                kill_switch_triggered: false,
                pnl: 0.0,
                max_drawdown_usd: 0.0,
                checksum: "abc".to_string(),
                path: PathBuf::from("p1"),
            },
            SummaryRow {
                suite_id: "a".to_string(),
                scenario_id: "a_scenario".to_string(),
                seed: 2,
                ablations: vec![],
                status: "OK".to_string(),
                kill_switch_triggered: false,
                pnl: 0.0,
                max_drawdown_usd: 0.0,
                checksum: "def".to_string(),
                path: PathBuf::from("p2"),
            },
            SummaryRow {
                suite_id: "a".to_string(),
                scenario_id: "a_scenario".to_string(),
                seed: 1,
                ablations: vec![],
                status: "OK".to_string(),
                kill_switch_triggered: false,
                pnl: 0.0,
                max_drawdown_usd: 0.0,
                checksum: "ghi".to_string(),
                path: PathBuf::from("p3"),
            },
        ];

        sort_rows(&mut rows);

        // After sorting by scenario_id asc, seed asc
        assert_eq!(rows[0].scenario_id, "a_scenario");
        assert_eq!(rows[0].seed, 1);
        assert_eq!(rows[1].scenario_id, "a_scenario");
        assert_eq!(rows[1].seed, 2);
        assert_eq!(rows[2].scenario_id, "z_scenario");
        assert_eq!(rows[2].seed, 1);
    }

    #[test]
    fn test_print_table() {
        let rows = vec![SummaryRow {
            suite_id: "ci".to_string(),
            scenario_id: "test".to_string(),
            seed: 42,
            ablations: vec![],
            status: "OK".to_string(),
            kill_switch_triggered: false,
            pnl: 123.45,
            max_drawdown_usd: 25.50,
            checksum: "abcdef1234567890abcdef1234567890".to_string(),
            path: PathBuf::from("test/run_summary.json"),
        }];

        let mut output = Vec::new();
        print_table(&rows, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("ci"));
        assert!(output_str.contains("test"));
        assert!(output_str.contains("42"));
        assert!(output_str.contains("123.45"));
        assert!(output_str.contains("abcdef123456")); // Shortened checksum
    }

    /// Test that ablations, pnl, checksum, and max_drawdown are correctly surfaced.
    #[test]
    fn test_ablations_pnl_checksum_surfaced() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create a run with ablations and specific PnL/checksum values
        let dir = base.join("test_scenario").join("sha").join("42");
        fs::create_dir_all(&dir).unwrap();
        let path = create_test_summary_full(
            &dir,
            "ablation_test",
            42,
            -150.75,                                         // Specific PnL
            35.25,                                           // Specific max drawdown
            &["disable_vol_floor", "disable_toxicity_gate"], // Ablations
            false,
        );

        let row = parse_run_summary(&path, base).unwrap();

        // Verify all fields are correctly parsed
        assert_eq!(row.scenario_id, "ablation_test");
        assert_eq!(row.seed, 42);
        assert_eq!(row.pnl, -150.75);
        assert_eq!(row.max_drawdown_usd, 35.25);
        assert_eq!(row.ablations.len(), 2);
        assert!(row.ablations.contains(&"disable_vol_floor".to_string()));
        assert!(row.ablations.contains(&"disable_toxicity_gate".to_string()));
        assert_eq!(
            row.checksum,
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        );
        assert_eq!(row.status, "OK");
        assert!(!row.kill_switch_triggered);
    }

    /// Test that kill switch triggered status is correctly shown as KILL@<step>.
    #[test]
    fn test_kill_switch_status() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        let dir = base.join("kill_scenario").join("sha").join("99");
        fs::create_dir_all(&dir).unwrap();
        let path = create_test_summary_full(
            &dir,
            "kill_test",
            99,
            -500.0,
            100.0,
            &[],
            true, // Kill switch triggered
        );

        let row = parse_run_summary(&path, base).unwrap();

        assert_eq!(row.status, "KILL@50");
        assert!(row.kill_switch_triggered);
    }

    /// Test Markdown output format includes all required columns.
    #[test]
    fn test_markdown_output_format() {
        let rows = vec![
            SummaryRow {
                suite_id: "research".to_string(),
                scenario_id: "scenario_a".to_string(),
                seed: 1,
                ablations: vec!["disable_vol_floor".to_string()],
                status: "OK".to_string(),
                kill_switch_triggered: false,
                pnl: 100.50,
                max_drawdown_usd: 25.75,
                checksum: "abcdef1234567890abcdef1234567890".to_string(),
                path: PathBuf::from("scenario_a/sha/1/run_summary.json"),
            },
            SummaryRow {
                suite_id: "research".to_string(),
                scenario_id: "scenario_b".to_string(),
                seed: 2,
                ablations: vec![],
                status: "KILLED".to_string(),
                kill_switch_triggered: true,
                pnl: -200.25,
                max_drawdown_usd: 150.00,
                checksum: "fedcba0987654321fedcba0987654321".to_string(),
                path: PathBuf::from("scenario_b/sha/2/run_summary.json"),
            },
        ];

        let mut output = Vec::new();
        print_markdown_table(&rows, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Verify Markdown table structure
        assert!(output_str.contains("| scenario_id |"));
        assert!(output_str.contains("| seed |"));
        assert!(output_str.contains("| ablations |"));
        assert!(output_str.contains("| final_pnl_usd |"));
        assert!(output_str.contains("| max_drawdown_usd |"));
        assert!(output_str.contains("| status |"));
        assert!(output_str.contains("| checksum |"));
        assert!(output_str.contains("| path |"));

        // Verify data rows
        assert!(output_str.contains("| scenario_a |"));
        assert!(output_str.contains("| 1 |"));
        assert!(output_str.contains("disable_vol_floor"));
        assert!(output_str.contains("100.50"));
        assert!(output_str.contains("25.75"));
        assert!(output_str.contains("| OK |"));
        assert!(output_str.contains("`abcdef123456`")); // Shortened checksum

        assert!(output_str.contains("| scenario_b |"));
        assert!(output_str.contains("| KILLED |"));
        assert!(output_str.contains("-200.25"));
    }

    /// Test stable sorting by scenario_id asc, seed asc.
    #[test]
    fn test_stable_sorting() {
        let temp = tempdir().unwrap();
        let base = temp.path();

        // Create runs in non-sorted order
        let scenarios = [
            ("z_scenario", 99),
            ("a_scenario", 50),
            ("a_scenario", 1),
            ("m_scenario", 10),
        ];

        for (scenario_id, seed) in scenarios {
            let dir = base.join(scenario_id).join("sha").join(seed.to_string());
            fs::create_dir_all(&dir).unwrap();
            create_test_summary(&dir, scenario_id, seed, 100.0);
        }

        let rows = get_summary_rows(base).unwrap();

        // Verify sorting order: a_scenario:1, a_scenario:50, m_scenario:10, z_scenario:99
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].scenario_id, "a_scenario");
        assert_eq!(rows[0].seed, 1);
        assert_eq!(rows[1].scenario_id, "a_scenario");
        assert_eq!(rows[1].seed, 50);
        assert_eq!(rows[2].scenario_id, "m_scenario");
        assert_eq!(rows[2].seed, 10);
        assert_eq!(rows[3].scenario_id, "z_scenario");
        assert_eq!(rows[3].seed, 99);
    }
}
