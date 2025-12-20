// src/sim_eval/summarize.rs
//
// Summarize command: discovers and aggregates run_summary.json files
// into a readable table output.

use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::output::RunSummary;

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
    /// Run status (OK, KILL, ERROR).
    pub status: String,
    /// Final PnL value.
    pub pnl: f64,
    /// Determinism checksum.
    pub checksum: String,
    /// Relative path to the run_summary.json file.
    pub path: PathBuf,
}

impl SummaryRow {
    /// Create a row from a parsed RunSummary and path info.
    pub fn from_run_summary(summary: &RunSummary, path: PathBuf, suite_id: String) -> Self {
        let status = if summary.results.kill_switch.triggered {
            format!(
                "KILL@{}",
                summary
                    .results
                    .kill_switch
                    .step
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            )
        } else {
            "OK".to_string()
        };

        Self {
            suite_id,
            scenario_id: summary.scenario_id.clone(),
            seed: summary.seed,
            ablations: Vec::new(), // Default to empty
            status,
            pnl: summary.results.final_pnl_usd,
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

        // Try to get checksum
        let checksum = value
            .get("determinism")
            .and_then(|d| d.get("checksum"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Try to get kill switch status
        let kill_triggered = value
            .get("results")
            .and_then(|r| r.get("kill_switch"))
            .and_then(|k| k.get("triggered"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let kill_step = value
            .get("results")
            .and_then(|r| r.get("kill_switch"))
            .and_then(|k| k.get("step"))
            .and_then(|v| v.as_u64());

        let status = if kill_triggered {
            format!(
                "KILL@{}",
                kill_step.map(|s| s.to_string()).unwrap_or_default()
            )
        } else {
            "OK".to_string()
        };

        // Try to get ablations
        let ablations = value
            .get("ablations")
            .and_then(|a| a.as_array())
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
            pnl,
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

/// Sort rows by suite_id, scenario_id, ablations, seed.
pub fn sort_rows(rows: &mut [SummaryRow]) {
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
        "[]".to_string()
    } else {
        format!("[{}]", ablations.join(","))
    }
}

/// Print the summary table to stdout.
pub fn print_table<W: Write>(rows: &[SummaryRow], mut writer: W) -> io::Result<()> {
    // Header
    writeln!(
        writer,
        "{:<15} {:<25} {:<10} {:<15} {:<12} {:>12} {:<18} PATH",
        "SUITE_ID", "SCENARIO_ID", "SEED", "ABLATIONS", "STATUS", "PNL", "CHECKSUM",
    )?;
    writeln!(
        writer,
        "{:-<15} {:-<25} {:-<10} {:-<15} {:-<12} {:->12} {:-<18} {:-<30}",
        "", "", "", "", "", "", "", ""
    )?;

    // Rows
    for row in rows {
        let checksum_short = if row.checksum.len() >= 16 {
            &row.checksum[..16]
        } else {
            &row.checksum
        };

        writeln!(
            writer,
            "{:<15} {:<25} {:<10} {:<15} {:<12} {:>12.2} {:<18} {}",
            row.suite_id,
            row.scenario_id,
            row.seed,
            format_ablations(&row.ablations),
            row.status,
            row.pnl,
            checksum_short,
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

/// Main entry point: discover, parse, sort, and print summaries.
pub fn summarize<W: Write>(runs_dir: &Path, writer: W) -> io::Result<SummarizeResult> {
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

    // Sort by suite_id, scenario_id, ablations, seed
    sort_rows(&mut rows);

    // Print table
    print_table(&rows, writer)?;

    Ok(SummarizeResult::Success(rows.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn create_test_summary(dir: &Path, scenario_id: &str, seed: u64, pnl: f64) -> PathBuf {
        let summary = serde_json::json!({
            "scenario_id": scenario_id,
            "scenario_version": 1,
            "seed": seed,
            "build_info": {
                "git_sha": "abc123",
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
            }
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
                scenario_id: "s1".to_string(),
                seed: 1,
                ablations: vec![],
                status: "OK".to_string(),
                pnl: 0.0,
                checksum: "abc".to_string(),
                path: PathBuf::from("p1"),
            },
            SummaryRow {
                suite_id: "a".to_string(),
                scenario_id: "s1".to_string(),
                seed: 2,
                ablations: vec![],
                status: "OK".to_string(),
                pnl: 0.0,
                checksum: "def".to_string(),
                path: PathBuf::from("p2"),
            },
            SummaryRow {
                suite_id: "a".to_string(),
                scenario_id: "s1".to_string(),
                seed: 1,
                ablations: vec![],
                status: "OK".to_string(),
                pnl: 0.0,
                checksum: "ghi".to_string(),
                path: PathBuf::from("p3"),
            },
        ];

        sort_rows(&mut rows);

        assert_eq!(rows[0].suite_id, "a");
        assert_eq!(rows[0].seed, 1);
        assert_eq!(rows[1].suite_id, "a");
        assert_eq!(rows[1].seed, 2);
        assert_eq!(rows[2].suite_id, "b");
    }

    #[test]
    fn test_print_table() {
        let rows = vec![SummaryRow {
            suite_id: "ci".to_string(),
            scenario_id: "test".to_string(),
            seed: 42,
            ablations: vec![],
            status: "OK".to_string(),
            pnl: 123.45,
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
        assert!(output_str.contains("abcdef1234567890"));
    }
}
