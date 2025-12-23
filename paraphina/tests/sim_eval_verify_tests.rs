// tests/sim_eval_verify_tests.rs
//
// Integration tests for the sim_eval verify-evidence-pack and verify-evidence-tree commands.
//
// These tests verify the CLI wiring by calling the library functions directly,
// matching the pattern used by sim_eval_summarize_tests.rs.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

use paraphina::sim_eval::{
    verify_evidence_pack_dir, verify_evidence_pack_tree, EvidencePackVerificationReport,
};

/// Helper to create a file with specific content.
fn create_file(path: &Path, content: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)
}

/// Compute SHA256 of content directly.
fn compute_sha256(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Create a valid evidence pack for testing.
fn create_valid_evidence_pack(dir: &Path) -> std::io::Result<()> {
    let evidence_pack_dir = dir.join("evidence_pack");
    fs::create_dir_all(&evidence_pack_dir)?;

    // Create manifest.json
    let manifest_content = b"{\"evidence_pack_schema_version\":\"v1\"}";
    create_file(&evidence_pack_dir.join("manifest.json"), manifest_content)?;

    // Create suite.yaml
    let suite_content = b"suite_id: test\n";
    create_file(&evidence_pack_dir.join("suite.yaml"), suite_content)?;

    // Create an extra artifact
    let artifact_content = b"{\"result\": \"ok\"}";
    create_file(&dir.join("results/run_001.json"), artifact_content)?;

    // Create SHA256SUMS with correct hashes
    let manifest_hash = compute_sha256(manifest_content);
    let suite_hash = compute_sha256(suite_content);
    let artifact_hash = compute_sha256(artifact_content);

    let sha256sums_content = format!(
        "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n{}  results/run_001.json\n",
        manifest_hash, suite_hash, artifact_hash
    );
    create_file(
        &evidence_pack_dir.join("SHA256SUMS"),
        sha256sums_content.as_bytes(),
    )?;

    Ok(())
}

// =============================================================================
// verify-evidence-pack tests
// =============================================================================

#[test]
fn test_verify_evidence_pack_success() {
    let dir = tempdir().expect("Failed to create temp dir");
    create_valid_evidence_pack(dir.path()).expect("Failed to create evidence pack");

    let report = verify_evidence_pack_dir(dir.path()).expect("Verification should succeed");

    assert_eq!(report.packs_verified, 1, "Should verify 1 pack");
    assert_eq!(report.files_verified, 3, "Should verify 3 files");
}

#[test]
fn test_verify_evidence_pack_missing_sha256sums() {
    let dir = tempdir().expect("Failed to create temp dir");
    fs::create_dir_all(dir.path().join("evidence_pack")).expect("Failed to create dir");

    let result = verify_evidence_pack_dir(dir.path());
    assert!(result.is_err(), "Should fail when SHA256SUMS is missing");

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("SHA256SUMS not found"),
        "Error should mention SHA256SUMS: {}",
        err
    );
}

#[test]
fn test_verify_evidence_pack_hash_mismatch() {
    let dir = tempdir().expect("Failed to create temp dir");
    let evidence_pack_dir = dir.path().join("evidence_pack");
    fs::create_dir_all(&evidence_pack_dir).expect("Failed to create dir");

    let manifest_content = b"{}";
    let suite_content = b"test";
    create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
    create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

    let manifest_hash = compute_sha256(manifest_content);
    let wrong_hash = "b".repeat(64); // Wrong hash

    let sha256sums = format!(
        "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n",
        manifest_hash, wrong_hash
    );
    create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

    let result = verify_evidence_pack_dir(dir.path());
    assert!(result.is_err(), "Should fail on hash mismatch");

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Hash mismatch"),
        "Error should mention hash mismatch: {}",
        err
    );
}

// =============================================================================
// verify-evidence-tree tests
// =============================================================================

#[test]
fn test_verify_evidence_tree_multiple_packs() {
    let dir = tempdir().expect("Failed to create temp dir");

    // Create two evidence packs in different subdirectories
    let pack1 = dir.path().join("runs/scenario1");
    let pack2 = dir.path().join("runs/scenario2");

    create_valid_evidence_pack(&pack1).expect("Failed to create pack1");
    create_valid_evidence_pack(&pack2).expect("Failed to create pack2");

    let report = verify_evidence_pack_tree(dir.path()).expect("Verification should succeed");

    assert_eq!(report.packs_verified, 2, "Should verify 2 packs");
    assert_eq!(
        report.files_verified, 6,
        "Should verify 6 files (3 per pack)"
    );
}

#[test]
fn test_verify_evidence_tree_no_packs() {
    let dir = tempdir().expect("Failed to create temp dir");
    fs::create_dir_all(dir.path().join("some/nested/dir")).expect("Failed to create dir");

    let result = verify_evidence_pack_tree(dir.path());
    assert!(result.is_err(), "Should fail when no packs found");

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No evidence_pack/SHA256SUMS files found"),
        "Error should mention no packs found: {}",
        err
    );
}

#[test]
fn test_verify_evidence_tree_one_bad_pack() {
    let dir = tempdir().expect("Failed to create temp dir");

    // Create one valid pack
    let pack1 = dir.path().join("runs/scenario1");
    create_valid_evidence_pack(&pack1).expect("Failed to create pack1");

    // Create one invalid pack (missing file)
    let pack2 = dir.path().join("runs/scenario2");
    let evidence_pack_dir = pack2.join("evidence_pack");
    fs::create_dir_all(&evidence_pack_dir).expect("Failed to create dir");

    let manifest_content = b"{}";
    create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
    // Don't create suite.yaml

    let manifest_hash = compute_sha256(manifest_content);
    let fake_hash = "a".repeat(64);
    let sha256sums = format!(
        "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n",
        manifest_hash, fake_hash
    );
    create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

    let result = verify_evidence_pack_tree(dir.path());
    assert!(
        result.is_err(),
        "Should fail when any pack fails verification"
    );
}

// =============================================================================
// Report structure tests
// =============================================================================

#[test]
fn test_report_default_is_empty() {
    let report = EvidencePackVerificationReport::default();
    assert_eq!(report.packs_verified, 0);
    assert_eq!(report.files_verified, 0);
}
