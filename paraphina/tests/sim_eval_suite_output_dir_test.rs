// tests/sim_eval_suite_output_dir_test.rs
//
// Regression test for sim_eval suite --output-dir feature.
//
// This test verifies that:
// 1. sim_eval suite accepts --output-dir option
// 2. Suite runs write output to the specified directory (exactly as given)
// 3. Evidence packs are created and verifiable under the specified directory
//
// Note: Suite only appends ablation suffix when ablations are actually active.
// With no ablations, output goes exactly to the specified directory.

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

use paraphina::sim_eval::verify_evidence_pack_tree;

/// Get the path to the sim_eval binary built by cargo.
fn sim_eval_bin() -> &'static str {
    env!("CARGO_BIN_EXE_sim_eval")
}

/// Get workspace root (parent of the test crate).
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Recursively check if a directory contains at least one evidence_pack/SHA256SUMS file.
fn has_evidence_pack(dir: &Path) -> bool {
    if !dir.exists() {
        return false;
    }

    // Check for evidence_pack/SHA256SUMS directly under this dir
    if dir.join("evidence_pack/SHA256SUMS").exists() {
        return true;
    }

    // Recursively check subdirectories
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && has_evidence_pack(&path) {
                return true;
            }
        }
    }

    false
}

/// Get the effective output directory.
/// Suite no longer appends __baseline suffix when no ablations are specified.
/// Suffix is only added when ablations are actually active.
fn effective_output_dir(base: &Path) -> std::path::PathBuf {
    base.to_path_buf()
}

// =============================================================================
// Test: suite --output-dir works and creates verifiable evidence packs
// =============================================================================

#[test]
fn suite_output_dir_creates_evidence_pack() {
    // Create a temp directory for output
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_dir_base = temp_dir.path().join("test_output");
    let output_dir = effective_output_dir(&output_dir_base);

    // Get path to ci_smoke_v1.yaml suite (smallest suite)
    let suite_path = workspace_root().join("scenarios/suites/ci_smoke_v1.yaml");
    assert!(
        suite_path.exists(),
        "Suite file should exist: {:?}",
        suite_path
    );

    // Run sim_eval suite with --output-dir (pass the base, not the effective)
    let output = Command::new(sim_eval_bin())
        .arg("suite")
        .arg(&suite_path)
        .arg("--output-dir")
        .arg(&output_dir_base)
        .current_dir(workspace_root())
        .output()
        .expect("Failed to execute sim_eval suite");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    // The suite should complete (exit 0) or have gate failures (exit 1),
    // but NOT exit 2 (usage error from unknown option)
    let exit_code = output.status.code().expect("Process terminated by signal");
    assert_ne!(
        exit_code, 2,
        "--output-dir should be accepted, not rejected as unknown option. Exit code: {}. Output:\n{}",
        exit_code, combined
    );

    // Verify output directory was created (exactly as specified)
    assert!(
        output_dir.exists(),
        "Output directory should be created: {:?}. Output:\n{}",
        output_dir,
        combined
    );

    // Verify evidence pack exists under the output directory
    assert!(
        has_evidence_pack(&output_dir),
        "Evidence pack should exist under {:?}. Output:\n{}",
        output_dir,
        combined
    );

    // Verify evidence pack tree passes verification
    let verify_result = verify_evidence_pack_tree(&output_dir);
    assert!(
        verify_result.is_ok(),
        "Evidence pack verification should succeed. Error: {:?}",
        verify_result.err()
    );

    let report = verify_result.unwrap();
    assert!(
        report.packs_verified >= 1,
        "At least 1 evidence pack should be verified. Got: {}",
        report.packs_verified
    );
    assert!(
        report.files_verified >= 1,
        "At least 1 file should be verified. Got: {}",
        report.files_verified
    );
}

#[test]
fn suite_output_dir_equals_syntax() {
    // Test --output-dir=<path> syntax (equals sign variant)
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_dir_base = temp_dir.path().join("equals_syntax_test");
    let output_dir = effective_output_dir(&output_dir_base);

    let suite_path = workspace_root().join("scenarios/suites/ci_smoke_v1.yaml");

    let output = Command::new(sim_eval_bin())
        .arg("suite")
        .arg(&suite_path)
        .arg(format!("--output-dir={}", output_dir_base.display()))
        .current_dir(workspace_root())
        .output()
        .expect("Failed to execute sim_eval suite");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Should not be rejected as unknown option
    assert_ne!(
        exit_code, 2,
        "--output-dir=<path> syntax should be accepted. Exit code: {}. Output:\n{}",
        exit_code, combined
    );

    // Output directory should be created
    assert!(
        output_dir.exists(),
        "Output directory should be created with = syntax: {:?}. Output:\n{}",
        output_dir,
        combined
    );
}

#[test]
fn suite_output_dir_before_suite_path() {
    // Test that --output-dir works when placed before the suite path
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_dir_base = temp_dir.path().join("order_test");
    let output_dir = effective_output_dir(&output_dir_base);

    let suite_path = workspace_root().join("scenarios/suites/ci_smoke_v1.yaml");

    // Put --output-dir BEFORE the suite path
    let output = Command::new(sim_eval_bin())
        .arg("suite")
        .arg("--output-dir")
        .arg(&output_dir_base)
        .arg(&suite_path)
        .current_dir(workspace_root())
        .output()
        .expect("Failed to execute sim_eval suite");

    let exit_code = output.status.code().expect("Process terminated by signal");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Should not be rejected as unknown option
    assert_ne!(
        exit_code, 2,
        "--output-dir before suite path should work. Exit code: {}. Output:\n{}",
        exit_code, combined
    );

    // Output directory should be created
    assert!(
        output_dir.exists(),
        "Output directory should be created when --output-dir comes before path: {:?}. Output:\n{}",
        output_dir,
        combined
    );
}

#[test]
fn suite_help_mentions_output_dir() {
    // Verify that --help mentions --output-dir for suite
    let output = Command::new(sim_eval_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute sim_eval --help");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that SUITE OPTIONS section mentions --output-dir
    assert!(
        combined.contains("SUITE OPTIONS"),
        "Help should have SUITE OPTIONS section"
    );
    assert!(
        combined.contains("--output-dir"),
        "Help should mention --output-dir for suite. Got:\n{}",
        combined
    );
}

#[test]
fn suite_inline_env_overrides_creates_evidence_packs() {
    // Test that a suite with inline env_overrides creates verifiable evidence packs
    // This uses a minimal inline suite YAML created on the fly
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_dir = temp_dir.path().join("inline_test_output");

    // Create a minimal suite YAML with inline env_overrides
    let suite_content = r#"
suite_id: inline_test_suite
suite_version: 1
repeat_runs: 1

scenarios:
  - id: inline_scenario_1
    seed: 42
    profile: balanced
    env_overrides:
      PARAPHINA_RISK_PROFILE: balanced
      PARAPHINA_VOL_REF: "0.20"
"#;

    let suite_path = temp_dir.path().join("inline_test_suite.yaml");
    fs::write(&suite_path, suite_content).expect("Failed to write suite YAML");

    // Run sim_eval suite
    let output = Command::new(sim_eval_bin())
        .arg("suite")
        .arg(&suite_path)
        .arg("--output-dir")
        .arg(&output_dir)
        .current_dir(workspace_root())
        .output()
        .expect("Failed to execute sim_eval suite");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    // Suite should complete successfully
    let exit_code = output.status.code().expect("Process terminated by signal");
    assert_eq!(
        exit_code, 0,
        "Suite with inline env_overrides should succeed. Exit code: {}. Output:\n{}",
        exit_code, combined
    );

    // Verify output directory was created
    assert!(
        output_dir.exists(),
        "Output directory should be created: {:?}. Output:\n{}",
        output_dir,
        combined
    );

    // Verify evidence pack exists under the output directory
    assert!(
        has_evidence_pack(&output_dir),
        "Evidence pack should exist under {:?}. Output:\n{}",
        output_dir,
        combined
    );

    // Verify evidence pack tree passes verification
    let verify_result = verify_evidence_pack_tree(&output_dir);
    assert!(
        verify_result.is_ok(),
        "Evidence pack verification should succeed for inline scenarios. Error: {:?}",
        verify_result.err()
    );

    let report = verify_result.unwrap();
    // Should have at least 2 packs: one per inline scenario + suite-level
    assert!(
        report.packs_verified >= 1,
        "At least 1 evidence pack should be verified. Got: {}",
        report.packs_verified
    );
}

#[test]
fn suite_help_mentions_env_overrides() {
    // Verify that --help mentions env_overrides support
    let output = Command::new(sim_eval_bin())
        .arg("--help")
        .output()
        .expect("Failed to execute sim_eval --help");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that help mentions env_overrides
    assert!(
        combined.contains("env_overrides"),
        "Help should mention env_overrides support. Got:\n{}",
        combined
    );
}
