// tests/monte_carlo_threaded_determinism_tests.rs
//
// Integration tests for deterministic multi-threaded Monte Carlo execution.
//
// These tests verify that:
// 1. Running with --threads 1 and --threads 4 produces identical mc_summary.json
// 2. Running with --threads 1 and --threads 4 produces identical mc_runs.jsonl
// 3. The --quiet flag does not produce interleaved output in threaded mode

use std::fs;
use std::process::Command;
use tempfile::tempdir;

/// Helper to run monte_carlo binary with given arguments.
fn run_monte_carlo(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_monte_carlo"))
        .args(args)
        .output()
        .expect("Failed to execute monte_carlo binary")
}

/// Parse JSONL file and return records sorted by run_index.
fn parse_jsonl_sorted(path: &std::path::Path) -> Vec<serde_json::Value> {
    let content = fs::read_to_string(path).expect("Failed to read JSONL file");
    let mut records: Vec<serde_json::Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("Failed to parse JSONL line"))
        .collect();

    // Sort by run_index for comparison
    records.sort_by_key(|r| r.get("run_index").and_then(|v| v.as_u64()).unwrap_or(0));
    records
}

/// Compare two JSON files for equality (ignoring formatting).
fn json_files_equal(path1: &std::path::Path, path2: &std::path::Path) -> bool {
    let content1 = fs::read_to_string(path1).expect("Failed to read JSON file 1");
    let content2 = fs::read_to_string(path2).expect("Failed to read JSON file 2");

    let json1: serde_json::Value =
        serde_json::from_str(&content1).expect("Failed to parse JSON file 1");
    let json2: serde_json::Value =
        serde_json::from_str(&content2).expect("Failed to parse JSON file 2");

    json1 == json2
}

#[test]
fn test_threaded_determinism_summary_identical() {
    // Run with threads=1 and threads=4, verify mc_summary.json is identical

    let dir1 = tempdir().expect("Failed to create temp dir 1");
    let dir4 = tempdir().expect("Failed to create temp dir 4");

    let runs = "12";
    let ticks = "50";
    let seed = "42";

    // Run with threads=1
    let output1 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "1",
        "--quiet",
        "--output-dir",
        dir1.path().to_str().unwrap(),
    ]);
    assert!(
        output1.status.success(),
        "monte_carlo --threads 1 failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    // Run with threads=4
    let output4 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "4",
        "--quiet",
        "--output-dir",
        dir4.path().to_str().unwrap(),
    ]);
    assert!(
        output4.status.success(),
        "monte_carlo --threads 4 failed: {}",
        String::from_utf8_lossy(&output4.stderr)
    );

    // Compare mc_summary.json
    let summary1_path = dir1.path().join("mc_summary.json");
    let summary4_path = dir4.path().join("mc_summary.json");

    assert!(
        summary1_path.exists(),
        "mc_summary.json not found for threads=1"
    );
    assert!(
        summary4_path.exists(),
        "mc_summary.json not found for threads=4"
    );

    // Load and compare summaries
    let summary1: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&summary1_path).expect("Failed to read summary 1"),
    )
    .expect("Failed to parse summary 1");
    let summary4: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&summary4_path).expect("Failed to read summary 4"),
    )
    .expect("Failed to parse summary 4");

    // Compare aggregate stats (should be identical)
    assert_eq!(
        summary1.get("aggregate"),
        summary4.get("aggregate"),
        "Aggregate stats differ between threads=1 and threads=4"
    );

    // Compare tail_risk (should be identical)
    assert_eq!(
        summary1.get("tail_risk"),
        summary4.get("tail_risk"),
        "Tail risk metrics differ between threads=1 and threads=4"
    );

    // Compare runs array (should be identical)
    assert_eq!(
        summary1.get("runs"),
        summary4.get("runs"),
        "Run records differ between threads=1 and threads=4"
    );
}

#[test]
fn test_threaded_determinism_jsonl_identical() {
    // Run with threads=1 and threads=4, verify mc_runs.jsonl records match

    let dir1 = tempdir().expect("Failed to create temp dir 1");
    let dir4 = tempdir().expect("Failed to create temp dir 4");

    let runs = "12";
    let ticks = "50";
    let seed = "12345";

    // Run with threads=1
    let output1 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "1",
        "--quiet",
        "--output-dir",
        dir1.path().to_str().unwrap(),
    ]);
    assert!(output1.status.success(), "threads=1 run failed");

    // Run with threads=4
    let output4 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "4",
        "--quiet",
        "--output-dir",
        dir4.path().to_str().unwrap(),
    ]);
    assert!(output4.status.success(), "threads=4 run failed");

    // Parse and sort JSONL records
    let jsonl1_path = dir1.path().join("mc_runs.jsonl");
    let jsonl4_path = dir4.path().join("mc_runs.jsonl");

    let records1 = parse_jsonl_sorted(&jsonl1_path);
    let records4 = parse_jsonl_sorted(&jsonl4_path);

    assert_eq!(
        records1.len(),
        records4.len(),
        "JSONL record counts differ"
    );

    // Compare each record
    for (i, (r1, r4)) in records1.iter().zip(records4.iter()).enumerate() {
        assert_eq!(
            r1, r4,
            "JSONL record {} differs:\nthreads=1: {:?}\nthreads=4: {:?}",
            i, r1, r4
        );
    }
}

#[test]
fn test_threaded_determinism_with_jitter() {
    // Test determinism with jitter enabled (more complex scenario)

    let dir1 = tempdir().expect("Failed to create temp dir 1");
    let dir4 = tempdir().expect("Failed to create temp dir 4");

    let runs = "8";
    let ticks = "30";
    let seed = "999";
    let jitter_ms = "100";

    // Run with threads=1
    let output1 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--jitter-ms",
        jitter_ms,
        "--threads",
        "1",
        "--quiet",
        "--output-dir",
        dir1.path().to_str().unwrap(),
    ]);
    assert!(output1.status.success(), "threads=1 run with jitter failed");

    // Run with threads=4
    let output4 = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--jitter-ms",
        jitter_ms,
        "--threads",
        "4",
        "--quiet",
        "--output-dir",
        dir4.path().to_str().unwrap(),
    ]);
    assert!(output4.status.success(), "threads=4 run with jitter failed");

    // Compare summaries
    let summary1_path = dir1.path().join("mc_summary.json");
    let summary4_path = dir4.path().join("mc_summary.json");

    assert!(
        json_files_equal(&summary1_path, &summary4_path),
        "Summaries differ with jitter enabled"
    );

    // Compare JSONL
    let jsonl1_path = dir1.path().join("mc_runs.jsonl");
    let jsonl4_path = dir4.path().join("mc_runs.jsonl");

    let records1 = parse_jsonl_sorted(&jsonl1_path);
    let records4 = parse_jsonl_sorted(&jsonl4_path);

    assert_eq!(records1, records4, "JSONL records differ with jitter enabled");
}

#[test]
fn test_threaded_jsonl_sorted_by_run_index() {
    // Verify that mc_runs.jsonl is written in sorted run_index order

    let dir = tempdir().expect("Failed to create temp dir");

    let runs = "16";
    let ticks = "20";
    let seed = "777";

    // Run with threads=4 (parallel execution)
    let output = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "4",
        "--quiet",
        "--output-dir",
        dir.path().to_str().unwrap(),
    ]);
    assert!(output.status.success(), "monte_carlo failed");

    // Read JSONL and verify order
    let jsonl_path = dir.path().join("mc_runs.jsonl");
    let content = fs::read_to_string(&jsonl_path).expect("Failed to read JSONL");

    let mut prev_run_index: Option<usize> = None;
    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value =
            serde_json::from_str(line).expect("Failed to parse JSONL line");
        let run_index = record
            .get("run_index")
            .and_then(|v| v.as_u64())
            .expect("Missing run_index") as usize;

        if let Some(prev) = prev_run_index {
            assert!(
                run_index > prev,
                "JSONL not sorted: line {} has run_index {} after {}",
                line_num,
                run_index,
                prev
            );
        }
        prev_run_index = Some(run_index);
    }
}

#[test]
fn test_threaded_quiet_no_interleaved_output() {
    // Verify that --quiet with threads > 1 does not produce interleaved per-run output

    let dir = tempdir().expect("Failed to create temp dir");

    let runs = "12";
    let ticks = "30";
    let seed = "123";

    // Run with threads=4 and --quiet
    let output = run_monte_carlo(&[
        "--runs",
        runs,
        "--ticks",
        ticks,
        "--seed",
        seed,
        "--threads",
        "4",
        "--quiet",
        "--output-dir",
        dir.path().to_str().unwrap(),
    ]);
    assert!(output.status.success(), "monte_carlo failed");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // With --quiet, we should NOT see per-run lines like "run 1/12"
    // We should only see the header, summary, and file output messages
    let lines: Vec<&str> = stdout.lines().collect();

    for line in &lines {
        // Should not contain per-run output patterns
        assert!(
            !line.starts_with("run ") || !line.contains("pnl="),
            "Found per-run output line in --quiet mode: {}",
            line
        );
    }

    // Should contain summary
    assert!(
        stdout.contains("SUMMARY"),
        "Missing SUMMARY in quiet mode output"
    );
}

#[test]
fn test_threaded_schema_version_preserved() {
    // Verify that McRunJsonlRecord still has schema_version: 1

    let dir = tempdir().expect("Failed to create temp dir");

    let output = run_monte_carlo(&[
        "--runs",
        "4",
        "--ticks",
        "10",
        "--seed",
        "1",
        "--threads",
        "2",
        "--quiet",
        "--output-dir",
        dir.path().to_str().unwrap(),
    ]);
    assert!(output.status.success(), "monte_carlo failed");

    let jsonl_path = dir.path().join("mc_runs.jsonl");
    let content = fs::read_to_string(&jsonl_path).expect("Failed to read JSONL");

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value =
            serde_json::from_str(line).expect("Failed to parse JSONL line");

        // Verify schema_version is present and equals 1
        let schema_version = record
            .get("schema_version")
            .expect("Missing schema_version field");
        assert_eq!(
            schema_version.as_u64(),
            Some(1),
            "schema_version should be 1, got: {:?}",
            schema_version
        );

        // Verify all required fields are present
        assert!(record.get("run_index").is_some(), "Missing run_index");
        assert!(record.get("seed").is_some(), "Missing seed");
        assert!(record.get("pnl_total").is_some(), "Missing pnl_total");
        assert!(record.get("max_drawdown").is_some(), "Missing max_drawdown");
        assert!(record.get("kill_switch").is_some(), "Missing kill_switch");
        assert!(record.get("kill_tick").is_some(), "Missing kill_tick");
        assert!(record.get("kill_reason").is_some(), "Missing kill_reason");
        assert!(
            record.get("ticks_executed").is_some(),
            "Missing ticks_executed"
        );
        assert!(
            record.get("max_abs_delta_usd").is_some(),
            "Missing max_abs_delta_usd"
        );
        assert!(
            record.get("max_abs_basis_usd").is_some(),
            "Missing max_abs_basis_usd"
        );
        assert!(
            record.get("max_abs_q_tao").is_some(),
            "Missing max_abs_q_tao"
        );
        assert!(
            record.get("max_venue_toxicity").is_some(),
            "Missing max_venue_toxicity"
        );
    }
}

#[test]
fn test_threads_cli_argument_parsing() {
    // Test that --threads argument is parsed correctly

    let dir = tempdir().expect("Failed to create temp dir");

    // Test --threads N style
    let output1 = run_monte_carlo(&[
        "--runs",
        "2",
        "--ticks",
        "5",
        "--seed",
        "1",
        "--threads",
        "2",
        "--quiet",
        "--output-dir",
        dir.path().to_str().unwrap(),
    ]);
    assert!(output1.status.success(), "--threads N style failed");

    // Test --threads=N style
    let dir2 = tempdir().expect("Failed to create temp dir 2");
    let output2 = run_monte_carlo(&[
        "--runs",
        "2",
        "--ticks",
        "5",
        "--seed",
        "1",
        "--threads=2",
        "--quiet",
        "--output-dir",
        dir2.path().to_str().unwrap(),
    ]);
    assert!(output2.status.success(), "--threads=N style failed");

    // Test --threads 0 should fail
    let dir3 = tempdir().expect("Failed to create temp dir 3");
    let output3 = run_monte_carlo(&[
        "--runs",
        "2",
        "--ticks",
        "5",
        "--seed",
        "1",
        "--threads",
        "0",
        "--quiet",
        "--output-dir",
        dir3.path().to_str().unwrap(),
    ]);
    assert!(
        !output3.status.success(),
        "--threads 0 should fail but succeeded"
    );
}

#[test]
fn test_single_thread_default() {
    // Verify default behavior (threads=1) still works

    let dir = tempdir().expect("Failed to create temp dir");

    // Run without --threads (should default to 1)
    let output = run_monte_carlo(&[
        "--runs",
        "4",
        "--ticks",
        "10",
        "--seed",
        "1",
        "--quiet",
        "--output-dir",
        dir.path().to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "monte_carlo without --threads failed"
    );

    // Verify outputs exist
    assert!(
        dir.path().join("mc_summary.json").exists(),
        "mc_summary.json missing"
    );
    assert!(
        dir.path().join("mc_runs.jsonl").exists(),
        "mc_runs.jsonl missing"
    );
}

