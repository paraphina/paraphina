// tests/sim_eval_summarize_tests.rs
//
// Integration tests for the sim_eval summarize command.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

use paraphina::sim_eval::{summarize, SummarizeResult};

/// Helper to create a fake run_summary.json file.
fn create_run_summary(
    dir: &Path,
    scenario_id: &str,
    seed: u64,
    pnl: f64,
    kill_triggered: bool,
    kill_step: Option<u64>,
) {
    let summary = serde_json::json!({
        "scenario_id": scenario_id,
        "scenario_version": 1,
        "seed": seed,
        "build_info": {
            "git_sha": "abc123def456abc123def456abc123def456abc1",
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
                "triggered": kill_triggered,
                "step": kill_step,
                "reason": if kill_triggered { Some("max_drawdown") } else { None }
            }
        },
        "determinism": {
            "checksum": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        }
    });

    fs::create_dir_all(dir).expect("Failed to create directory");
    let path = dir.join("run_summary.json");
    let mut file = File::create(&path).expect("Failed to create file");
    write!(file, "{}", serde_json::to_string_pretty(&summary).unwrap())
        .expect("Failed to write file");
}

#[test]
fn test_summarize_finds_multiple_runs() {
    let temp = tempdir().expect("Failed to create temp dir");
    let base = temp.path();

    // Create 3 fake run summaries with different scenarios and seeds
    create_run_summary(
        &base.join("scenario_alpha").join("sha1").join("42"),
        "scenario_alpha",
        42,
        150.75,
        false,
        None,
    );
    create_run_summary(
        &base.join("scenario_beta").join("sha2").join("100"),
        "scenario_beta",
        100,
        -25.50,
        true,
        Some(50),
    );
    create_run_summary(
        &base.join("scenario_alpha").join("sha1").join("99"),
        "scenario_alpha",
        99,
        200.00,
        false,
        None,
    );

    let mut output = Vec::new();
    let result = summarize(base, &mut output).expect("summarize should not fail");

    match result {
        SummarizeResult::Success(count) => {
            assert_eq!(count, 3, "Should find 3 runs");
        }
        other => panic!("Expected Success, got {:?}", other),
    }

    let output_str = String::from_utf8(output).expect("Output should be valid UTF-8");

    // Check that all scenarios are present
    assert!(
        output_str.contains("scenario_alpha"),
        "Output should contain scenario_alpha"
    );
    assert!(
        output_str.contains("scenario_beta"),
        "Output should contain scenario_beta"
    );

    // Check that seeds are present
    assert!(output_str.contains("42"), "Output should contain seed 42");
    assert!(output_str.contains("99"), "Output should contain seed 99");
    assert!(output_str.contains("100"), "Output should contain seed 100");

    // Check that PnL values are present
    assert!(
        output_str.contains("150.75"),
        "Output should contain PnL 150.75"
    );
    assert!(
        output_str.contains("-25.50"),
        "Output should contain PnL -25.50"
    );
    assert!(
        output_str.contains("200.00"),
        "Output should contain PnL 200.00"
    );

    // Check status
    assert!(output_str.contains("OK"), "Output should contain OK status");
    assert!(
        output_str.contains("KILL@50"),
        "Output should contain KILL@50 status"
    );
}

#[test]
fn test_summarize_output_is_sorted() {
    let temp = tempdir().expect("Failed to create temp dir");
    let base = temp.path();

    // Create runs out of order
    create_run_summary(
        &base.join("z_scenario").join("sha").join("1"),
        "z_scenario",
        1,
        100.0,
        false,
        None,
    );
    create_run_summary(
        &base.join("a_scenario").join("sha").join("2"),
        "a_scenario",
        2,
        200.0,
        false,
        None,
    );
    create_run_summary(
        &base.join("a_scenario").join("sha").join("1"),
        "a_scenario",
        1,
        300.0,
        false,
        None,
    );

    let mut output = Vec::new();
    let result = summarize(base, &mut output).expect("summarize should not fail");

    match result {
        SummarizeResult::Success(count) => {
            assert_eq!(count, 3);
        }
        other => panic!("Expected Success, got {:?}", other),
    }

    let output_str = String::from_utf8(output).expect("Output should be valid UTF-8");
    let lines: Vec<&str> = output_str.lines().collect();

    // Skip header lines (first 2), check data rows
    assert!(lines.len() >= 5, "Should have header + 3 data rows");

    // Find data rows (skip header lines)
    let data_lines: Vec<&str> = lines.iter().skip(2).cloned().collect();

    // First data row should be a_scenario with seed 1
    assert!(
        data_lines[0].contains("a_scenario") && data_lines[0].contains(" 1 "),
        "First row should be a_scenario seed 1, got: {}",
        data_lines[0]
    );
    // Second should be a_scenario with seed 2
    assert!(
        data_lines[1].contains("a_scenario") && data_lines[1].contains(" 2 "),
        "Second row should be a_scenario seed 2, got: {}",
        data_lines[1]
    );
    // Third should be z_scenario
    assert!(
        data_lines[2].contains("z_scenario"),
        "Third row should be z_scenario, got: {}",
        data_lines[2]
    );
}

#[test]
fn test_summarize_no_files_found() {
    let temp = tempdir().expect("Failed to create temp dir");
    let base = temp.path();

    // Empty directory
    let mut output = Vec::new();
    let result = summarize(base, &mut output).expect("summarize should not fail");

    match result {
        SummarizeResult::NoFilesFound => {}
        other => panic!("Expected NoFilesFound, got {:?}", other),
    }
}

#[test]
fn test_summarize_with_nested_directories() {
    let temp = tempdir().expect("Failed to create temp dir");
    let base = temp.path();

    // Create deeply nested run summaries
    create_run_summary(
        &base.join("ci").join("scenario1").join("abc123").join("42"),
        "scenario1",
        42,
        100.0,
        false,
        None,
    );
    create_run_summary(
        &base.join("ci").join("scenario2").join("def456").join("99"),
        "scenario2",
        99,
        200.0,
        false,
        None,
    );

    let mut output = Vec::new();
    let result = summarize(base, &mut output).expect("summarize should not fail");

    match result {
        SummarizeResult::Success(count) => {
            assert_eq!(count, 2, "Should find 2 runs");
        }
        other => panic!("Expected Success, got {:?}", other),
    }

    let output_str = String::from_utf8(output).expect("Output should be valid UTF-8");

    // Check that suite_id is extracted from path
    assert!(
        output_str.contains("ci"),
        "Output should contain suite_id 'ci'"
    );
}

#[test]
fn test_summarize_table_has_expected_columns() {
    let temp = tempdir().expect("Failed to create temp dir");
    let base = temp.path();

    create_run_summary(
        &base.join("test").join("sha").join("1"),
        "test_scenario",
        1,
        123.45,
        false,
        None,
    );

    let mut output = Vec::new();
    let result = summarize(base, &mut output).expect("summarize should not fail");

    match result {
        SummarizeResult::Success(_) => {}
        other => panic!("Expected Success, got {:?}", other),
    }

    let output_str = String::from_utf8(output).expect("Output should be valid UTF-8");
    let header = output_str.lines().next().expect("Should have header");

    // Check all required columns are in header
    assert!(
        header.contains("SUITE_ID"),
        "Header should contain SUITE_ID"
    );
    assert!(
        header.contains("SCENARIO_ID"),
        "Header should contain SCENARIO_ID"
    );
    assert!(header.contains("SEED"), "Header should contain SEED");
    assert!(
        header.contains("ABLATIONS"),
        "Header should contain ABLATIONS"
    );
    assert!(header.contains("STATUS"), "Header should contain STATUS");
    assert!(header.contains("PNL"), "Header should contain PNL");
    assert!(
        header.contains("CHECKSUM"),
        "Header should contain CHECKSUM"
    );
    assert!(header.contains("PATH"), "Header should contain PATH");
}
