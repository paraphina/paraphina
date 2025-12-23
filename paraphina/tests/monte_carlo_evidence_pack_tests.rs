// tests/monte_carlo_evidence_pack_tests.rs
//
// Integration tests for monte_carlo evidence pack generation and verification.
//
// These tests verify that the monte_carlo binary correctly generates
// Evidence Pack v1 artifacts that pass the strict verifier.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use paraphina::sim_eval::{verify_evidence_pack_dir, verify_evidence_pack_tree};

/// Get the path to the monte_carlo binary built by cargo.
fn monte_carlo_bin() -> &'static str {
    env!("CARGO_BIN_EXE_monte_carlo")
}

/// Get the path to the sim_eval binary built by cargo.
fn sim_eval_bin() -> &'static str {
    env!("CARGO_BIN_EXE_sim_eval")
}

/// Create a unique temp directory using stdlib only.
fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    let pid = std::process::id();
    let dir_name = format!("{}_{}_{}", prefix, nanos, pid);
    let path = std::env::temp_dir().join(dir_name);
    fs::create_dir_all(&path).expect("Failed to create temp directory");
    path
}

/// Remove a temp directory, ignoring errors.
fn cleanup_temp_dir(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

// =============================================================================
// Test 1: monte_carlo generates valid evidence pack
// =============================================================================

#[test]
fn test_monte_carlo_generates_evidence_pack() {
    let temp_dir = make_temp_dir("mc_evidence_test");

    // Run monte_carlo with minimal parameters to temp directory
    let output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "10",
            "--output-dir",
            temp_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Monte carlo should succeed
    assert_eq!(
        exit_code, 0,
        "monte_carlo should exit with code 0. stdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Check that evidence_pack directory was created
    let evidence_pack_dir = temp_dir.join("evidence_pack");
    assert!(
        evidence_pack_dir.exists(),
        "evidence_pack/ directory should exist"
    );

    // Check required files exist
    assert!(
        evidence_pack_dir.join("manifest.json").exists(),
        "evidence_pack/manifest.json should exist"
    );
    assert!(
        evidence_pack_dir.join("suite.yaml").exists(),
        "evidence_pack/suite.yaml should exist"
    );
    assert!(
        evidence_pack_dir.join("SHA256SUMS").exists(),
        "evidence_pack/SHA256SUMS should exist"
    );

    // Check that mc_summary.json was written
    assert!(
        temp_dir.join("mc_summary.json").exists(),
        "mc_summary.json should exist"
    );

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 2: monte_carlo evidence pack passes Rust verifier
// =============================================================================

#[test]
fn test_monte_carlo_evidence_pack_verifies() {
    let temp_dir = make_temp_dir("mc_verify_test");

    // Run monte_carlo
    let output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "3",
            "--ticks",
            "5",
            "--output-dir",
            temp_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Verify the evidence pack using the library function
    let report =
        verify_evidence_pack_dir(&temp_dir).expect("Evidence pack verification should succeed");

    assert_eq!(report.packs_verified, 1, "Should verify 1 pack");
    assert!(
        report.files_verified >= 3,
        "Should verify at least 3 files (manifest, suite, mc_summary)"
    );

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 3: monte_carlo evidence pack passes tree verifier
// =============================================================================

#[test]
fn test_monte_carlo_evidence_pack_verifies_tree() {
    let temp_dir = make_temp_dir("mc_tree_test");
    let run_dir = temp_dir.join("runs/test_run");
    fs::create_dir_all(&run_dir).expect("Failed to create run directory");

    // Run monte_carlo
    let output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "5",
            "--output-dir",
            run_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Verify tree from parent runs directory
    let runs_dir = temp_dir.join("runs");
    let report = verify_evidence_pack_tree(&runs_dir)
        .expect("Evidence pack tree verification should succeed");

    assert_eq!(report.packs_verified, 1, "Should verify 1 pack");

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 4: monte_carlo evidence pack passes CLI verifier
// =============================================================================

#[test]
fn test_monte_carlo_cli_verify_evidence_pack() {
    let temp_dir = make_temp_dir("mc_cli_verify_test");

    // Run monte_carlo
    let mc_output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "5",
            "--output-dir",
            temp_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        mc_output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Verify using sim_eval CLI
    let verify_output = Command::new(sim_eval_bin())
        .args(["verify-evidence-pack", temp_dir.to_str().unwrap()])
        .output()
        .expect("Failed to execute sim_eval verify-evidence-pack");

    let exit_code = verify_output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&verify_output.stdout);
    let stderr = String::from_utf8_lossy(&verify_output.stderr);

    assert_eq!(
        exit_code, 0,
        "sim_eval verify-evidence-pack should exit with code 0. stdout: {}\nstderr: {}",
        stdout, stderr
    );

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 5: monte_carlo evidence pack passes CLI tree verifier
// =============================================================================

#[test]
fn test_monte_carlo_cli_verify_evidence_tree() {
    let temp_dir = make_temp_dir("mc_cli_tree_test");
    let run_dir = temp_dir.join("runs/test_mc");
    fs::create_dir_all(&run_dir).expect("Failed to create run directory");

    // Run monte_carlo
    let mc_output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "5",
            "--output-dir",
            run_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        mc_output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Verify tree using sim_eval CLI
    let runs_dir = temp_dir.join("runs");
    let verify_output = Command::new(sim_eval_bin())
        .args(["verify-evidence-tree", runs_dir.to_str().unwrap()])
        .output()
        .expect("Failed to execute sim_eval verify-evidence-tree");

    let exit_code = verify_output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&verify_output.stdout);
    let stderr = String::from_utf8_lossy(&verify_output.stderr);

    assert_eq!(
        exit_code, 0,
        "sim_eval verify-evidence-tree should exit with code 0. stdout: {}\nstderr: {}",
        stdout, stderr
    );

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 6: monte_carlo with CSV option includes CSV in evidence pack
// =============================================================================

#[test]
fn test_monte_carlo_with_csv_in_evidence_pack() {
    let temp_dir = make_temp_dir("mc_csv_test");

    // Run monte_carlo with CSV output
    let output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "5",
            "--output-dir",
            temp_dir.to_str().unwrap(),
            "--csv",
            "mc_runs.csv",
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Check that CSV was written
    let csv_path = temp_dir.join("mc_runs.csv");
    assert!(csv_path.exists(), "mc_runs.csv should exist");

    // Verify the evidence pack (CSV should be included in hash verification)
    let report =
        verify_evidence_pack_dir(&temp_dir).expect("Evidence pack verification should succeed");
    assert!(
        report.files_verified >= 4,
        "Should verify at least 4 files (manifest, suite, mc_summary, csv)"
    );

    // Read SHA256SUMS and check CSV is included
    let sha256sums_path = temp_dir.join("evidence_pack/SHA256SUMS");
    let sha256sums_content =
        fs::read_to_string(&sha256sums_path).expect("Failed to read SHA256SUMS");
    assert!(
        sha256sums_content.contains("mc_runs.csv"),
        "SHA256SUMS should include mc_runs.csv. Content:\n{}",
        sha256sums_content
    );

    cleanup_temp_dir(&temp_dir);
}

// =============================================================================
// Test 7: monte_carlo manifest.json contains required fields
// =============================================================================

#[test]
fn test_monte_carlo_manifest_structure() {
    let temp_dir = make_temp_dir("mc_manifest_test");

    // Run monte_carlo
    let output = Command::new(monte_carlo_bin())
        .args([
            "--runs",
            "2",
            "--ticks",
            "5",
            "--output-dir",
            temp_dir.to_str().unwrap(),
            "--quiet",
        ])
        .output()
        .expect("Failed to execute monte_carlo");

    assert_eq!(
        output.status.code().unwrap_or(-1),
        0,
        "monte_carlo should succeed"
    );

    // Read and parse manifest.json
    let manifest_path = temp_dir.join("evidence_pack/manifest.json");
    let manifest_content =
        fs::read_to_string(&manifest_path).expect("Failed to read manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_content).expect("Failed to parse manifest.json");

    // Check required fields
    assert_eq!(
        manifest.get("evidence_pack_schema_version"),
        Some(&serde_json::json!("v1")),
        "evidence_pack_schema_version should be 'v1'"
    );
    assert!(
        manifest.get("generated_at_unix_ms").is_some(),
        "generated_at_unix_ms should be present"
    );
    assert!(
        manifest.get("paraphina_version").is_some(),
        "paraphina_version should be present"
    );
    assert!(
        manifest.get("repository").is_some(),
        "repository should be present"
    );
    assert!(manifest.get("suite").is_some(), "suite should be present");
    assert!(
        manifest.get("artifacts").is_some(),
        "artifacts should be present"
    );

    // Check artifacts is a non-empty array
    let artifacts = manifest.get("artifacts").unwrap().as_array().unwrap();
    assert!(!artifacts.is_empty(), "artifacts should not be empty");

    cleanup_temp_dir(&temp_dir);
}
