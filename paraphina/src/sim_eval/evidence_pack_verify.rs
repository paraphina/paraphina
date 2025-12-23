// src/sim_eval/evidence_pack_verify.rs
//
// Evidence Pack v1 verifier per docs/EVIDENCE_PACK.md.
//
// Provides verification of evidence pack integrity by:
// - Parsing SHA256SUMS file with strict format validation
// - Rejecting unsafe paths (absolute, traversal, symlinks)
// - Computing file hashes in Rust (no shell-out)
// - Requiring manifest.json and suite.yaml entries

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};

// ============================================================================
// Public API
// ============================================================================

/// Result of verifying one or more evidence packs.
#[derive(Debug, Clone, Default)]
pub struct EvidencePackVerificationReport {
    /// Number of evidence packs verified.
    pub packs_verified: usize,
    /// Total number of files verified across all packs.
    pub files_verified: usize,
}

/// Verify a single evidence pack under `output_root/evidence_pack/`.
///
/// Checks:
/// - SHA256SUMS exists and parses correctly
/// - All referenced files exist and are regular files (not symlinks)
/// - All hashes match
/// - Required entries (manifest.json, suite.yaml) are present in SHA256SUMS
///
/// # Arguments
///
/// * `output_root` - Base output directory (evidence_pack/ must exist inside)
///
/// # Errors
///
/// Returns an error if:
/// - evidence_pack/SHA256SUMS is missing or malformed
/// - Any referenced file is missing, is a symlink, or has wrong hash
/// - Path safety checks fail (absolute paths, traversal, etc.)
/// - Required entries manifest.json or suite.yaml are missing from SHA256SUMS
pub fn verify_evidence_pack_dir(output_root: &Path) -> Result<EvidencePackVerificationReport> {
    let sha256sums_path = output_root.join("evidence_pack/SHA256SUMS");

    if !sha256sums_path.exists() {
        bail!("SHA256SUMS not found at {}", sha256sums_path.display());
    }

    // Parse SHA256SUMS
    let entries = parse_sha256sums(&sha256sums_path)?;

    if entries.is_empty() {
        bail!("SHA256SUMS is empty (no entries)");
    }

    // Check for required entries
    check_required_entries(&entries)?;

    // Check for duplicate paths
    check_no_duplicates(&entries)?;

    // Verify each entry
    let mut files_verified = 0;
    for entry in &entries {
        verify_entry(output_root, entry)?;
        files_verified += 1;
    }

    Ok(EvidencePackVerificationReport {
        packs_verified: 1,
        files_verified,
    })
}

/// Find and verify all evidence packs under `root`.
///
/// Recursively finds all `evidence_pack/SHA256SUMS` files under `root`
/// and verifies each pack.
///
/// # Arguments
///
/// * `root` - Root directory to search for evidence packs
///
/// # Errors
///
/// Returns an error if any evidence pack fails verification.
pub fn verify_evidence_pack_tree(root: &Path) -> Result<EvidencePackVerificationReport> {
    let sha256sums_files = find_sha256sums_files(root)?;

    if sha256sums_files.is_empty() {
        bail!(
            "No evidence_pack/SHA256SUMS files found under {}",
            root.display()
        );
    }

    let mut total_report = EvidencePackVerificationReport::default();

    for sha256sums_path in sha256sums_files {
        // The output_root is the parent of evidence_pack/
        let evidence_pack_dir = sha256sums_path
            .parent()
            .context("SHA256SUMS has no parent directory")?;
        let output_root = evidence_pack_dir
            .parent()
            .context("evidence_pack has no parent directory")?;

        let report = verify_evidence_pack_dir(output_root).with_context(|| {
            format!(
                "Failed to verify evidence pack at {}",
                output_root.display()
            )
        })?;

        total_report.packs_verified += report.packs_verified;
        total_report.files_verified += report.files_verified;
    }

    Ok(total_report)
}

// ============================================================================
// SHA256SUMS parsing
// ============================================================================

/// A parsed entry from SHA256SUMS.
#[derive(Debug, Clone)]
struct Sha256Entry {
    /// The expected SHA256 hash (64 hex chars, lowercase).
    hash: String,
    /// The relative path (normalized, no leading "./").
    path: String,
}

/// Parse a SHA256SUMS file with strict format validation.
///
/// Accepts formats:
/// - "<64hex>  <relpath>" (standard sha256sum output, two spaces)
/// - "<64hex> *<relpath>" (binary mode indicator)
///
/// Allows:
/// - Empty lines
/// - Trailing whitespace
/// - CRLF line endings
/// - Leading "./" in path (stripped)
///
/// Rejects:
/// - Invalid hex characters in hash
/// - Hash not exactly 64 characters
/// - Absolute paths
/// - Windows path prefixes
/// - Parent directory components ("..")
/// - Paths containing NUL bytes
fn parse_sha256sums(path: &Path) -> Result<Vec<Sha256Entry>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open SHA256SUMS: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut entries = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .with_context(|| format!("Failed to read line {} of SHA256SUMS", line_num + 1))?;

        // Handle CRLF by trimming end
        let line = line.trim_end();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        let entry = parse_sha256sums_line(line)
            .with_context(|| format!("Invalid SHA256SUMS line {}: {:?}", line_num + 1, line))?;

        entries.push(entry);
    }

    Ok(entries)
}

/// Parse a single line from SHA256SUMS.
fn parse_sha256sums_line(line: &str) -> Result<Sha256Entry> {
    // Format 1: "<64hex>  <relpath>" (two spaces)
    // Format 2: "<64hex> *<relpath>" (space + asterisk for binary mode)

    // Must have at least 64 chars for hash + separator + at least 1 char for path
    if line.len() < 67 {
        bail!("Line too short to be valid");
    }

    let hash_part = &line[..64];
    let separator = &line[64..66];
    let path_part = &line[66..];

    // Validate hash is 64 lowercase hex characters
    if !is_valid_hex(hash_part) {
        bail!("Invalid hash: must be 64 lowercase hex characters");
    }

    // Validate separator: must be "  " or " *"
    if separator != "  " && separator != " *" {
        bail!("Invalid separator: expected two spaces or space+asterisk after hash");
    }

    // Normalize and validate path
    let normalized_path = normalize_and_validate_path(path_part)?;

    Ok(Sha256Entry {
        hash: hash_part.to_lowercase(),
        path: normalized_path,
    })
}

/// Check if a string is valid lowercase hexadecimal.
fn is_valid_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Normalize a path from SHA256SUMS and validate it for safety.
fn normalize_and_validate_path(path: &str) -> Result<String> {
    // Reject NUL bytes
    if path.contains('\0') {
        bail!("Path contains NUL byte");
    }

    // Reject Windows prefixes
    if is_windows_path(path) {
        bail!("Windows path prefix not allowed: {:?}", path);
    }

    // Strip leading "./" repeatedly
    let mut normalized = path.to_string();
    while normalized.starts_with("./") {
        normalized = normalized[2..].to_string();
    }

    // Check if empty after normalization
    if normalized.is_empty() {
        bail!("Path is empty after normalization");
    }

    // Parse as Path and validate components
    let path_obj = Path::new(&normalized);

    // Reject absolute paths
    if path_obj.is_absolute() {
        bail!("Absolute path not allowed: {:?}", normalized);
    }

    // Reject parent directory components
    for component in path_obj.components() {
        match component {
            Component::ParentDir => {
                bail!(
                    "Parent directory component (..) not allowed: {:?}",
                    normalized
                );
            }
            Component::Prefix(_) => {
                bail!("Path prefix not allowed: {:?}", normalized);
            }
            _ => {}
        }
    }

    Ok(normalized)
}

/// Check if a path looks like a Windows path.
fn is_windows_path(path: &str) -> bool {
    // Check for drive letters (e.g., "C:\", "D:/")
    if path.len() >= 2 {
        let bytes = path.as_bytes();
        if bytes[0].is_ascii_alphabetic() && (bytes[1] == b':') {
            return true;
        }
    }

    // Check for UNC paths (e.g., "\\server\share", "//server/share")
    if path.starts_with("\\\\") || path.starts_with("//") {
        return true;
    }

    false
}

// ============================================================================
// Required entries check
// ============================================================================

/// Check that SHA256SUMS includes entries for manifest.json and suite.yaml.
fn check_required_entries(entries: &[Sha256Entry]) -> Result<()> {
    let has_manifest = entries
        .iter()
        .any(|e| e.path == "evidence_pack/manifest.json");
    let has_suite = entries.iter().any(|e| e.path == "evidence_pack/suite.yaml");

    if !has_manifest {
        bail!("Required entry missing from SHA256SUMS: evidence_pack/manifest.json");
    }

    if !has_suite {
        bail!("Required entry missing from SHA256SUMS: evidence_pack/suite.yaml");
    }

    Ok(())
}

/// Check for duplicate paths in entries.
fn check_no_duplicates(entries: &[Sha256Entry]) -> Result<()> {
    let mut seen = HashSet::new();
    for entry in entries {
        if !seen.insert(&entry.path) {
            bail!("Duplicate path in SHA256SUMS: {:?}", entry.path);
        }
    }
    Ok(())
}

// ============================================================================
// Entry verification
// ============================================================================

/// Verify a single SHA256SUMS entry.
fn verify_entry(output_root: &Path, entry: &Sha256Entry) -> Result<()> {
    let file_path = output_root.join(&entry.path);

    // Canonicalize both paths to ensure file is under output_root
    let canonical_root = output_root.canonicalize().with_context(|| {
        format!(
            "Failed to canonicalize output_root: {}",
            output_root.display()
        )
    })?;

    // First check if file exists (before canonicalize, which would fail)
    if !file_path.exists() {
        bail!(
            "Referenced file does not exist: {} (from entry {:?})",
            file_path.display(),
            entry.path
        );
    }

    let canonical_file = file_path
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize file path: {}", file_path.display()))?;

    // Verify the file is under output_root
    if !canonical_file.starts_with(&canonical_root) {
        bail!(
            "Path escapes output_root: {:?} resolves to {} which is outside {}",
            entry.path,
            canonical_file.display(),
            canonical_root.display()
        );
    }

    // Check if it's a symlink (reject symlinks)
    let metadata = fs::symlink_metadata(&file_path)
        .with_context(|| format!("Failed to get metadata for: {}", file_path.display()))?;

    if metadata.file_type().is_symlink() {
        bail!("Symlinks not allowed: {:?} is a symlink", entry.path);
    }

    if !metadata.is_file() {
        bail!("Not a regular file: {:?}", entry.path);
    }

    // Compute hash using streaming reads
    let actual_hash = hash_file_sha256(&file_path)?;

    // Compare hashes (case-insensitive)
    if actual_hash.to_lowercase() != entry.hash.to_lowercase() {
        bail!(
            "Hash mismatch for {:?}: expected {}, got {}",
            entry.path,
            entry.hash,
            actual_hash
        );
    }

    Ok(())
}

/// Compute SHA256 hash of a file using streaming reads.
fn hash_file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(hex_encode(&hash))
}

/// Hex-encode bytes to lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ============================================================================
// Tree search
// ============================================================================

/// Find all `evidence_pack/SHA256SUMS` files under a root directory.
fn find_sha256sums_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    find_sha256sums_recursive(root, &mut results)?;
    Ok(results)
}

fn find_sha256sums_recursive(dir: &Path, results: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let entries = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    for entry in entries {
        let entry = entry
            .with_context(|| format!("Failed to read directory entry in: {}", dir.display()))?;
        let path = entry.path();

        if path.is_dir() {
            // Check if this is an evidence_pack directory with SHA256SUMS
            if path
                .file_name()
                .map(|n| n == "evidence_pack")
                .unwrap_or(false)
            {
                let sha256sums = path.join("SHA256SUMS");
                if sha256sums.exists() && sha256sums.is_file() {
                    results.push(sha256sums);
                }
            } else {
                // Recurse into subdirectory
                find_sha256sums_recursive(&path, results)?;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

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
        hex_encode(&hash)
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

    // ========================================================================
    // Happy path tests
    // ========================================================================

    #[test]
    fn test_verify_valid_evidence_pack() {
        let dir = tempdir().unwrap();
        create_valid_evidence_pack(dir.path()).unwrap();

        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.packs_verified, 1);
        assert_eq!(report.files_verified, 3); // manifest, suite, artifact
    }

    #[test]
    fn test_verify_with_binary_mode_indicator() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        // Use binary mode format: "<hash> *<path>"
        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let sha256sums = format!(
            "{} *evidence_pack/manifest.json\n{} *evidence_pack/suite.yaml\n",
            manifest_hash, suite_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.files_verified, 2);
    }

    #[test]
    fn test_verify_with_leading_dot_slash() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        // Use paths with leading "./"
        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let sha256sums = format!(
            "{}  ./evidence_pack/manifest.json\n{}  ././evidence_pack/suite.yaml\n",
            manifest_hash, suite_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.files_verified, 2);
    }

    #[test]
    fn test_verify_with_empty_lines_and_crlf() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        // Include empty lines and CRLF
        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let sha256sums = format!(
            "\r\n{}  evidence_pack/manifest.json\r\n\n{}  evidence_pack/suite.yaml\r\n\n",
            manifest_hash, suite_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.files_verified, 2);
    }

    #[test]
    fn test_verify_evidence_pack_tree() {
        let dir = tempdir().unwrap();

        // Create two evidence packs in different subdirectories
        let pack1 = dir.path().join("runs/scenario1");
        let pack2 = dir.path().join("runs/scenario2");

        create_valid_evidence_pack(&pack1).unwrap();
        create_valid_evidence_pack(&pack2).unwrap();

        let report = verify_evidence_pack_tree(dir.path()).unwrap();
        assert_eq!(report.packs_verified, 2);
        assert_eq!(report.files_verified, 6); // 3 files per pack
    }

    // ========================================================================
    // Error case tests
    // ========================================================================

    #[test]
    fn test_error_missing_sha256sums() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("evidence_pack")).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("SHA256SUMS not found"));
    }

    #[test]
    fn test_error_missing_referenced_file() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let missing_hash = "a".repeat(64); // Fake hash for missing file

        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n{}  missing_file.json\n",
            manifest_hash, suite_hash, missing_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn test_error_hash_mismatch() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        let manifest_hash = compute_sha256(manifest_content);
        let wrong_hash = "b".repeat(64); // Wrong hash for suite.yaml

        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n",
            manifest_hash, wrong_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Hash mismatch"));
    }

    #[test]
    fn test_error_path_traversal() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let bad_hash = "c".repeat(64);

        // Path with parent directory traversal - this should fail during parsing
        let sha256sums = format!("{}  ../escape.txt\n", bad_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        // Use the full error chain to check for the underlying cause
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("Parent directory component") || err.contains(".."),
            "Got error: {}",
            err
        );
    }

    #[test]
    fn test_error_absolute_path() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let bad_hash = "d".repeat(64);

        // Absolute path - this should fail during parsing
        let sha256sums = format!("{}  /etc/passwd\n", bad_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        // Use the full error chain to check for the underlying cause
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("Absolute path not allowed") || err.contains("/etc/passwd"),
            "Got error: {}",
            err
        );
    }

    #[test]
    fn test_error_missing_required_manifest() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        let suite_hash = compute_sha256(suite_content);

        // Missing manifest.json entry
        let sha256sums = format!("{}  evidence_pack/suite.yaml\n", suite_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Required entry missing"));
        assert!(err.contains("manifest.json"));
    }

    #[test]
    fn test_error_missing_required_suite() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();

        let manifest_hash = compute_sha256(manifest_content);

        // Missing suite.yaml entry
        let sha256sums = format!("{}  evidence_pack/manifest.json\n", manifest_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Required entry missing"));
        assert!(err.contains("suite.yaml"));
    }

    #[test]
    fn test_error_duplicate_paths() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);

        // Duplicate entry
        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n{}  evidence_pack/manifest.json\n",
            manifest_hash, suite_hash, manifest_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Duplicate path"));
    }

    #[test]
    fn test_error_invalid_hash_format() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        // Invalid hash (not 64 hex chars)
        let sha256sums = b"not_a_valid_hash  evidence_pack/manifest.json\n";
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too short") || err.contains("Invalid"));
    }

    #[test]
    fn test_error_windows_path_drive() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let bad_hash = "e".repeat(64);
        // Windows drive path - should fail during parsing
        let sha256sums = format!("{}  C:\\Windows\\System32\\config\\sam\n", bad_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        // Use the full error chain to check for the underlying cause
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("Windows path prefix") || err.contains("C:\\"),
            "Got error: {}",
            err
        );
    }

    #[test]
    fn test_error_windows_unc_path() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let bad_hash = "f".repeat(64);
        // Windows UNC path - should fail during parsing
        let sha256sums = format!("{}  \\\\server\\share\\file.txt\n", bad_hash);
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        // Use the full error chain to check for the underlying cause
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("Windows path prefix") || err.contains("\\\\server"),
            "Got error: {}",
            err
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_error_symlink_rejected() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        // Create a regular file and a symlink to it
        let real_file = dir.path().join("real_file.txt");
        create_file(&real_file, b"real content").unwrap();

        let symlink_path = dir.path().join("link_file.txt");
        symlink(&real_file, &symlink_path).unwrap();

        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let link_hash = compute_sha256(b"real content");

        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n{}  link_file.txt\n",
            manifest_hash, suite_hash, link_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Symlinks not allowed") || err.contains("symlink"));
    }

    #[test]
    fn test_error_empty_sha256sums() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        // Empty SHA256SUMS
        create_file(&evidence_pack_dir.join("SHA256SUMS"), b"").unwrap();

        let result = verify_evidence_pack_dir(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty"));
    }

    #[test]
    fn test_error_no_packs_in_tree() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("some/nested/dir")).unwrap();

        let result = verify_evidence_pack_tree(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No evidence_pack/SHA256SUMS files found"));
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_uppercase_hash_accepted() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();

        // Use uppercase hex
        let manifest_hash = compute_sha256(manifest_content).to_uppercase();
        let suite_hash = compute_sha256(suite_content).to_uppercase();

        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n",
            manifest_hash, suite_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        // Should still pass (case-insensitive comparison)
        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.files_verified, 2);
    }

    #[test]
    fn test_deeply_nested_artifact() {
        let dir = tempdir().unwrap();
        let evidence_pack_dir = dir.path().join("evidence_pack");
        fs::create_dir_all(&evidence_pack_dir).unwrap();

        let manifest_content = b"{}";
        let suite_content = b"test";
        let nested_content = b"deeply nested";

        create_file(&evidence_pack_dir.join("manifest.json"), manifest_content).unwrap();
        create_file(&evidence_pack_dir.join("suite.yaml"), suite_content).unwrap();
        create_file(&dir.path().join("a/b/c/d/e/nested.txt"), nested_content).unwrap();

        let manifest_hash = compute_sha256(manifest_content);
        let suite_hash = compute_sha256(suite_content);
        let nested_hash = compute_sha256(nested_content);

        let sha256sums = format!(
            "{}  evidence_pack/manifest.json\n{}  evidence_pack/suite.yaml\n{}  a/b/c/d/e/nested.txt\n",
            manifest_hash, suite_hash, nested_hash
        );
        create_file(&evidence_pack_dir.join("SHA256SUMS"), sha256sums.as_bytes()).unwrap();

        let report = verify_evidence_pack_dir(dir.path()).unwrap();
        assert_eq!(report.files_verified, 3);
    }

    #[test]
    fn test_hex_validation() {
        assert!(is_valid_hex(&"a".repeat(64)));
        assert!(is_valid_hex(&"0123456789abcdef".repeat(4)));
        assert!(is_valid_hex(&("ABCDEF".repeat(10) + "abcd")));

        // Wrong length
        assert!(!is_valid_hex(&"a".repeat(63)));
        assert!(!is_valid_hex(&"a".repeat(65)));

        // Non-hex characters
        assert!(!is_valid_hex(&("g".to_owned() + &"a".repeat(63))));
    }

    #[test]
    fn test_windows_path_detection() {
        assert!(is_windows_path("C:\\Windows"));
        assert!(is_windows_path("D:/path"));
        assert!(is_windows_path("\\\\server\\share"));
        assert!(is_windows_path("//server/share"));

        assert!(!is_windows_path("normal/path"));
        assert!(!is_windows_path("./relative"));
        assert!(!is_windows_path("evidence_pack/manifest.json"));
    }
}
