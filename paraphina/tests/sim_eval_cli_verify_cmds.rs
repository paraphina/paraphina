// tests/sim_eval_cli_verify_cmds.rs
//
// Minimal, dependency-free integration tests for the verifier CLI commands.
// These tests use std::process::Command to execute the compiled sim_eval binary,
// ensuring CLI argument parsing works correctly for verify-evidence-pack and
// verify-evidence-tree subcommands.
//
// The goal is to prevent regressions of the "Unexpected argument: runs" class of bug
// by verifying that positional arguments are accepted and routed to verifier code
// (exit 3), not rejected by parsing (exit 2).

use std::fs;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

/// Get the path to the sim_eval binary built by cargo.
fn sim_eval_bin() -> &'static str {
    env!("CARGO_BIN_EXE_sim_eval")
}

/// Create a unique temp directory using stdlib only (SystemTime nanos + pid).
/// Returns the path to the created directory.
fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    let pid = std::process::id();
    let dir_name = format!("{}_{}_{}_{}", prefix, nanos, pid, rand_suffix());
    let path = std::env::temp_dir().join(dir_name);
    fs::create_dir_all(&path).expect("Failed to create temp directory");
    path
}

/// Generate a small random suffix using time-based pseudo-randomness.
fn rand_suffix() -> u32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .subsec_nanos();
    nanos % 100000
}

/// Remove a temp directory, ignoring errors.
fn cleanup_temp_dir(path: &std::path::Path) {
    let _ = fs::remove_dir_all(path);
}

/// Helper to get combined stdout and stderr as a string.
fn combined_output(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{}{}", stdout, stderr)
}

// =============================================================================
// Test 1: help_mentions_verify_commands
// =============================================================================

#[test]
fn help_mentions_verify_commands() {
    let output = Command::new(sim_eval_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute sim_eval --help");

    let combined = combined_output(&output);

    assert!(
        combined.contains("verify-evidence-pack"),
        "Help output should mention verify-evidence-pack. Got:\n{}",
        combined
    );
    assert!(
        combined.contains("verify-evidence-tree"),
        "Help output should mention verify-evidence-tree. Got:\n{}",
        combined
    );
}

// =============================================================================
// Test 2: verify_tree_missing_arg_is_usage_error
// =============================================================================

#[test]
fn verify_tree_missing_arg_is_usage_error() {
    let output = Command::new(sim_eval_bin())
        .arg("verify-evidence-tree")
        .output()
        .expect("Failed to execute sim_eval verify-evidence-tree");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = combined_output(&output);

    assert_eq!(
        exit_code, 2,
        "Missing arg should exit with code 2. Got {} with output:\n{}",
        exit_code, combined
    );

    // Check for usage information (case insensitive)
    let combined_lower = combined.to_lowercase();
    assert!(
        combined_lower.contains("usage:"),
        "Output should contain 'Usage:' or 'USAGE:'. Got:\n{}",
        combined
    );
    assert!(
        combined.contains("verify-evidence-tree"),
        "Output should contain 'verify-evidence-tree'. Got:\n{}",
        combined
    );
}

// =============================================================================
// Test 3: verify_pack_missing_arg_is_usage_error
// =============================================================================

#[test]
fn verify_pack_missing_arg_is_usage_error() {
    let output = Command::new(sim_eval_bin())
        .arg("verify-evidence-pack")
        .output()
        .expect("Failed to execute sim_eval verify-evidence-pack");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = combined_output(&output);

    assert_eq!(
        exit_code, 2,
        "Missing arg should exit with code 2. Got {} with output:\n{}",
        exit_code, combined
    );

    assert!(
        combined.contains("verify-evidence-pack"),
        "Output should contain 'verify-evidence-pack'. Got:\n{}",
        combined
    );
}

// =============================================================================
// Test 4: verify_tree_accepts_positional_root
// =============================================================================

#[test]
fn verify_tree_accepts_positional_root() {
    let temp_dir = make_temp_dir("sim_eval_cli_test_tree");

    let output = Command::new(sim_eval_bin())
        .arg("verify-evidence-tree")
        .arg(&temp_dir)
        .output()
        .expect("Failed to execute sim_eval verify-evidence-tree <tempdir>");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = combined_output(&output);

    // Cleanup before assertions to ensure cleanup happens even on failure
    cleanup_temp_dir(&temp_dir);

    // Exit code 3 means positional arg was accepted and routed to verifier,
    // but verification failed (expected since temp dir is empty).
    // Exit code 2 would mean argument parsing rejected the positional arg.
    assert_eq!(
        exit_code, 3,
        "Empty temp dir should exit with code 3 (verification failure, not usage error). \
         Exit code 2 would indicate positional arg was rejected. Got {} with output:\n{}",
        exit_code, combined
    );

    // Should contain a verifier-origin error about missing evidence packs
    let has_relevant_error = combined.contains("No evidence_pack")
        || combined.contains("SHA256SUMS")
        || combined.contains("evidence_pack");
    assert!(
        has_relevant_error,
        "Output should contain verifier error (No evidence_pack, SHA256SUMS, or evidence_pack). Got:\n{}",
        combined
    );
}

// =============================================================================
// Test 5: verify_pack_accepts_positional_output_root
// =============================================================================

#[test]
fn verify_pack_accepts_positional_output_root() {
    let temp_dir = make_temp_dir("sim_eval_cli_test_pack");

    let output = Command::new(sim_eval_bin())
        .arg("verify-evidence-pack")
        .arg(&temp_dir)
        .output()
        .expect("Failed to execute sim_eval verify-evidence-pack <tempdir>");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = combined_output(&output);

    // Cleanup before assertions
    cleanup_temp_dir(&temp_dir);

    // Exit code 3 means positional arg was accepted and routed to verifier,
    // but verification failed (expected since temp dir is empty).
    assert_eq!(
        exit_code, 3,
        "Empty temp dir should exit with code 3 (verification failure, not usage error). \
         Exit code 2 would indicate positional arg was rejected. Got {} with output:\n{}",
        exit_code, combined
    );

    // Should contain a verifier-origin error
    let has_relevant_error = combined.contains("SHA256SUMS") || combined.contains("evidence_pack");
    assert!(
        has_relevant_error,
        "Output should contain verifier error (SHA256SUMS or evidence_pack). Got:\n{}",
        combined
    );
}
