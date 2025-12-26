// src/sim_eval/evidence_pack.rs
//
// Evidence Pack v1 writer per docs/EVIDENCE_PACK.md.
//
// Provides an audit-grade output bundle with:
// - Provenance (git commit, Cargo.lock hash, schema hash)
// - Integrity (SHA256 hashes for all artifacts)
// - Determinism (stable field ordering, atomic writes)

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Paraphina version from Cargo.toml (compile-time).
const PARAPHINA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Evidence Pack schema version.
const SCHEMA_VERSION: &str = "v1";

// ============================================================================
// Manifest types (field order guaranteed by struct definition + serde)
// ============================================================================

/// Repository metadata for provenance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryInfo {
    /// Git commit SHA (null if not available).
    pub git_commit: Option<String>,
    /// SHA256 of Cargo.lock (null if not available).
    pub cargo_lock_sha256: Option<String>,
    /// SHA256 of docs/SIM_OUTPUT_SCHEMA.md (null if not available).
    pub sim_output_schema_sha256: Option<String>,
}

/// Suite file information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteInfo {
    /// Original path to suite YAML (relative to repo root).
    pub source_path: String,
    /// Destination path within evidence pack.
    pub copied_to: String,
    /// SHA256 of the suite file.
    pub sha256: String,
}

/// Single artifact entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArtifactEntry {
    /// Path relative to output_root.
    pub path: String,
    /// SHA256 hash of the file.
    pub sha256: String,
}

/// Complete manifest.json structure per docs/EVIDENCE_PACK.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version (always "v1" for this implementation).
    pub evidence_pack_schema_version: String,
    /// Unix timestamp in milliseconds when pack was generated.
    pub generated_at_unix_ms: u64,
    /// Paraphina version from Cargo.toml.
    pub paraphina_version: String,
    /// Repository metadata.
    pub repository: RepositoryInfo,
    /// Suite file information.
    pub suite: SuiteInfo,
    /// Artifact entries (sorted by path, excludes manifest.json itself).
    pub artifacts: Vec<ArtifactEntry>,
}

// ============================================================================
// Hashing utilities (streaming, no full-file memory load)
// ============================================================================

/// Compute SHA256 hash of a file using streaming reads.
///
/// Returns the hash as a lowercase hex string prefixed with "sha256:".
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
    Ok(format!("sha256:{}", hex_encode(&hash)))
}

/// Compute raw SHA256 hash of a file (without prefix), for SHA256SUMS format.
fn hash_file_sha256_raw(path: &Path) -> Result<String> {
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
// Atomic file write utilities
// ============================================================================

/// Write data to a file atomically (write to temp, then rename).
///
/// The temp file is created in the same directory to ensure rename works
/// (same filesystem).
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("Path has no parent: {}", path.display()))?;

    // Create temp file in same directory
    let temp_name = format!(
        ".tmp_{}_{}",
        std::process::id(),
        path.file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default()
    );
    let temp_path = parent.join(&temp_name);

    // Write to temp file
    let mut file = File::create(&temp_path)
        .with_context(|| format!("Failed to create temp file: {}", temp_path.display()))?;
    file.write_all(data)
        .with_context(|| format!("Failed to write temp file: {}", temp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("Failed to sync temp file: {}", temp_path.display()))?;

    // Rename temp to final (atomic on POSIX)
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "Failed to rename {} to {}",
            temp_path.display(),
            path.display()
        )
    })?;

    Ok(())
}

/// Copy a file atomically to destination.
fn atomic_copy(src: &Path, dst: &Path) -> Result<()> {
    let data =
        fs::read(src).with_context(|| format!("Failed to read source file: {}", src.display()))?;
    atomic_write(dst, &data)
}

// ============================================================================
// Path validation utilities
// ============================================================================

/// Normalize a path to use forward slashes (for cross-platform consistency).
fn normalize_path_str(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Check if a path escapes the root (has .. components or is absolute).
fn path_escapes_root(path: &Path) -> bool {
    // Check for absolute paths
    if path.is_absolute() {
        return true;
    }

    // Check for .. components
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return true;
        }
    }

    false
}

// ============================================================================
// Best-effort repository metadata
// ============================================================================

/// Get git commit SHA with environment variable fallback.
///
/// Precedence:
/// 1. PARAPHINA_GIT_SHA env var
/// 2. GITHUB_SHA env var
/// 3. `git rev-parse HEAD` in CARGO_MANIFEST_DIR
fn get_git_commit() -> Option<String> {
    // Check env vars first
    if let Ok(sha) = std::env::var("PARAPHINA_GIT_SHA") {
        if !sha.is_empty() {
            return Some(sha);
        }
    }
    if let Ok(sha) = std::env::var("GITHUB_SHA") {
        if !sha.is_empty() {
            return Some(sha);
        }
    }

    // Try git rev-parse HEAD
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
}

/// Get workspace root (parent of CARGO_MANIFEST_DIR for workspace members).
fn get_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // For workspace members, the workspace root is typically one level up
    manifest_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest_dir)
}

/// Hash Cargo.lock from workspace root if present.
fn get_cargo_lock_sha256() -> Option<String> {
    let cargo_lock_path = get_workspace_root().join("Cargo.lock");
    if cargo_lock_path.exists() {
        hash_file_sha256(&cargo_lock_path).ok()
    } else {
        None
    }
}

/// Hash docs/SIM_OUTPUT_SCHEMA.md from workspace root if present.
fn get_sim_output_schema_sha256() -> Option<String> {
    let schema_path = get_workspace_root().join("docs/SIM_OUTPUT_SCHEMA.md");
    if schema_path.exists() {
        hash_file_sha256(&schema_path).ok()
    } else {
        None
    }
}

/// Collect all repository metadata with best-effort retrieval.
fn collect_repository_info() -> RepositoryInfo {
    RepositoryInfo {
        git_commit: get_git_commit(),
        cargo_lock_sha256: get_cargo_lock_sha256(),
        sim_output_schema_sha256: get_sim_output_schema_sha256(),
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Write an Evidence Pack v1 to the output directory.
///
/// Creates `output_root/evidence_pack/` containing:
/// - `suite.yaml`: Byte-identical copy of the input suite file
/// - `manifest.json`: Structured metadata with artifact hashes
/// - `SHA256SUMS`: Standard checksum file for verification
///
/// # Arguments
///
/// * `output_root` - Base output directory (evidence_pack/ will be created inside)
/// * `suite_source_path` - Path to the original suite YAML file
/// * `artifact_paths_relative_to_output_root` - Paths of artifact files relative to output_root
///
/// # Errors
///
/// Returns an error if:
/// - Any artifact path escapes output_root (.. traversal or absolute paths)
/// - Any artifact file does not exist
/// - File I/O operations fail
///
/// # Example
///
/// ```ignore
/// use std::path::PathBuf;
///
/// write_evidence_pack(
///     Path::new("./runs/scenario_123"),
///     Path::new("./suites/stress_test.yaml"),
///     &[
///         PathBuf::from("results/run_001.json"),
///         PathBuf::from("results/run_002.json"),
///     ],
/// )?;
/// ```
pub fn write_evidence_pack(
    output_root: &Path,
    suite_source_path: &Path,
    artifact_paths_relative_to_output_root: &[PathBuf],
) -> Result<()> {
    // Validate artifact paths don't escape output_root
    for artifact_path in artifact_paths_relative_to_output_root {
        if path_escapes_root(artifact_path) {
            bail!(
                "Artifact path escapes output_root (contains .. or is absolute): {}",
                artifact_path.display()
            );
        }
    }

    // Create evidence_pack directory
    let evidence_pack_dir = output_root.join("evidence_pack");
    fs::create_dir_all(&evidence_pack_dir).with_context(|| {
        format!(
            "Failed to create evidence_pack dir: {}",
            evidence_pack_dir.display()
        )
    })?;

    // Copy suite file to evidence_pack/suite.yaml (byte-identical)
    let suite_dest = evidence_pack_dir.join("suite.yaml");
    atomic_copy(suite_source_path, &suite_dest).with_context(|| {
        format!(
            "Failed to copy suite file from {}",
            suite_source_path.display()
        )
    })?;

    // Compute suite hash
    let suite_sha256 = hash_file_sha256(&suite_dest)?;

    // Verify all artifact files exist and compute their hashes
    let mut artifacts: Vec<ArtifactEntry> = Vec::new();

    // Add suite.yaml as first artifact
    artifacts.push(ArtifactEntry {
        path: "evidence_pack/suite.yaml".to_string(),
        sha256: suite_sha256.clone(),
    });

    // Add user-provided artifacts
    for artifact_rel_path in artifact_paths_relative_to_output_root {
        let artifact_abs_path = output_root.join(artifact_rel_path);
        if !artifact_abs_path.exists() {
            bail!(
                "Artifact file does not exist: {}",
                artifact_abs_path.display()
            );
        }

        let sha256 = hash_file_sha256(&artifact_abs_path)?;
        let path_str = normalize_path_str(artifact_rel_path);

        artifacts.push(ArtifactEntry {
            path: path_str,
            sha256,
        });
    }

    // Sort artifacts by path (lexicographic) - excludes manifest.json by construction
    artifacts.sort_by(|a, b| a.path.cmp(&b.path));

    // Build manifest
    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    // Compute suite source path relative to workspace root for display
    let suite_source_path_display = suite_source_path.to_string_lossy().replace('\\', "/");

    let manifest = Manifest {
        evidence_pack_schema_version: SCHEMA_VERSION.to_string(),
        generated_at_unix_ms,
        paraphina_version: PARAPHINA_VERSION.to_string(),
        repository: collect_repository_info(),
        suite: SuiteInfo {
            source_path: suite_source_path_display,
            copied_to: "evidence_pack/suite.yaml".to_string(),
            sha256: suite_sha256,
        },
        artifacts,
    };

    // Serialize manifest to JSON with stable field order (serde preserves struct field order)
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("Failed to serialize manifest to JSON")?;

    // Write manifest.json atomically
    let manifest_path = evidence_pack_dir.join("manifest.json");
    atomic_write(&manifest_path, manifest_json.as_bytes())?;

    // Compute manifest hash
    let manifest_hash = hash_file_sha256_raw(&manifest_path)?;

    // Build SHA256SUMS content
    // Order: manifest.json, suite.yaml, then artifacts (sorted by path)
    let mut sha256sums_lines: Vec<String> = Vec::new();

    // manifest.json first
    sha256sums_lines.push(format!("{}  evidence_pack/manifest.json", manifest_hash));

    // suite.yaml second
    let suite_hash_raw = hash_file_sha256_raw(&suite_dest)?;
    sha256sums_lines.push(format!("{}  evidence_pack/suite.yaml", suite_hash_raw));

    // All artifacts from manifest (sorted by path, which they already are)
    for artifact in &manifest.artifacts {
        // Skip suite.yaml (already added above)
        if artifact.path == "evidence_pack/suite.yaml" {
            continue;
        }
        let artifact_abs_path = output_root.join(&artifact.path);
        let hash_raw = hash_file_sha256_raw(&artifact_abs_path)?;
        sha256sums_lines.push(format!("{}  {}", hash_raw, artifact.path));
    }

    let sha256sums_content = sha256sums_lines.join("\n") + "\n";

    // Write SHA256SUMS atomically
    let sha256sums_path = evidence_pack_dir.join("SHA256SUMS");
    atomic_write(&sha256sums_path, sha256sums_content.as_bytes())?;

    Ok(())
}

// ============================================================================
// Root Evidence Pack (simplified, for Python batch outputs)
// ============================================================================

/// Directories to exclude from root evidence pack scanning.
const EXCLUDED_DIRS: &[&str] = &[
    "evidence_pack",
    ".git",
    "target",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
];

/// Write a root-level Evidence Pack for a directory.
///
/// This is a simplified evidence pack that:
/// - Scans all files under `root` (excluding evidence_pack/ and common junk dirs)
/// - Generates `<root>/evidence_pack/SHA256SUMS` with deterministic ordering
/// - Generates `<root>/evidence_pack/manifest.json` with minimal metadata
/// - Creates a synthetic `<root>/evidence_pack/suite.yaml` for verifier compatibility
///
/// Unlike `write_evidence_pack`, this does not require a pre-existing suite file
/// or explicit artifact list - it discovers all files automatically.
///
/// # Arguments
///
/// * `root` - Directory to create evidence pack for
///
/// # Returns
///
/// Number of files hashed on success.
///
/// # Errors
///
/// Returns an error if:
/// - File I/O operations fail
/// - Root directory does not exist
pub fn write_root_evidence_pack(root: &Path) -> Result<usize> {
    if !root.exists() {
        bail!("Root directory does not exist: {}", root.display());
    }

    if !root.is_dir() {
        bail!("Root path is not a directory: {}", root.display());
    }

    // Collect all files recursively, excluding evidence_pack/ and junk dirs
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files_recursive(root, &mut files)?;

    // Sort by relative path for determinism
    files.sort();

    // Create evidence_pack directory
    let evidence_pack_dir = root.join("evidence_pack");
    fs::create_dir_all(&evidence_pack_dir).with_context(|| {
        format!(
            "Failed to create evidence_pack dir: {}",
            evidence_pack_dir.display()
        )
    })?;

    // Create minimal suite.yaml (for verifier compatibility)
    let suite_content = format!(
        "# Auto-generated root evidence pack\n# Generated at: {}\ntype: root_evidence_pack\nroot: {}\n",
        generate_timestamp_str(),
        root.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string())
    );
    let suite_path = evidence_pack_dir.join("suite.yaml");
    atomic_write(&suite_path, suite_content.as_bytes())?;
    let suite_hash = hash_file_sha256_raw(&suite_path)?;

    // Build manifest
    let mut artifacts: Vec<ArtifactEntry> = Vec::new();

    // Add suite.yaml
    artifacts.push(ArtifactEntry {
        path: "evidence_pack/suite.yaml".to_string(),
        sha256: format!("sha256:{}", suite_hash),
    });

    // Hash all discovered files
    for file_path in &files {
        let rel_path = file_path
            .strip_prefix(root)
            .with_context(|| format!("Failed to get relative path for: {}", file_path.display()))?;
        let hash = hash_file_sha256(file_path)?;
        let path_str = normalize_path_str(rel_path);

        artifacts.push(ArtifactEntry {
            path: path_str,
            sha256: hash,
        });
    }

    // Sort artifacts by path
    artifacts.sort_by(|a, b| a.path.cmp(&b.path));

    // Build manifest
    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let manifest = Manifest {
        evidence_pack_schema_version: SCHEMA_VERSION.to_string(),
        generated_at_unix_ms,
        paraphina_version: PARAPHINA_VERSION.to_string(),
        repository: collect_repository_info(),
        suite: SuiteInfo {
            source_path: "(generated)".to_string(),
            copied_to: "evidence_pack/suite.yaml".to_string(),
            sha256: format!("sha256:{}", suite_hash),
        },
        artifacts: artifacts.clone(),
    };

    // Write manifest.json
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("Failed to serialize manifest to JSON")?;
    let manifest_path = evidence_pack_dir.join("manifest.json");
    atomic_write(&manifest_path, manifest_json.as_bytes())?;
    let manifest_hash = hash_file_sha256_raw(&manifest_path)?;

    // Build SHA256SUMS
    let mut sha256sums_lines: Vec<String> = Vec::new();

    // manifest.json first
    sha256sums_lines.push(format!("{}  evidence_pack/manifest.json", manifest_hash));

    // suite.yaml second
    sha256sums_lines.push(format!("{}  evidence_pack/suite.yaml", suite_hash));

    // All artifacts from manifest (sorted by path)
    for artifact in &artifacts {
        // Skip suite.yaml (already added)
        if artifact.path == "evidence_pack/suite.yaml" {
            continue;
        }
        let artifact_abs_path = root.join(&artifact.path);
        let hash_raw = hash_file_sha256_raw(&artifact_abs_path)?;
        sha256sums_lines.push(format!("{}  {}", hash_raw, artifact.path));
    }

    let sha256sums_content = sha256sums_lines.join("\n") + "\n";

    // Write SHA256SUMS atomically
    let sha256sums_path = evidence_pack_dir.join("SHA256SUMS");
    atomic_write(&sha256sums_path, sha256sums_content.as_bytes())?;

    Ok(files.len())
}

/// Recursively collect all files under a directory, excluding certain paths.
fn collect_files_recursive(current: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(current)
        .with_context(|| format!("Failed to read directory: {}", current.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("Failed to read entry in: {}", current.display()))?;
        let path = entry.path();
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        if path.is_dir() {
            // Skip excluded directories
            if EXCLUDED_DIRS.contains(&file_name.as_str()) {
                continue;
            }
            collect_files_recursive(&path, files)?;
        } else if path.is_file() {
            // Skip hidden files (starting with .)
            if !file_name.starts_with('.') {
                files.push(path);
            }
        }
        // Skip symlinks and other non-regular files
    }

    Ok(())
}

/// Generate a timestamp string for suite.yaml comment.
fn generate_timestamp_str() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let total_secs = now.as_secs();
    let secs_per_min = 60;
    let secs_per_hour = 3600;
    let secs_per_day = 86400;

    let days_since_epoch = total_secs / secs_per_day;
    let time_of_day = total_secs % secs_per_day;

    let hours = time_of_day / secs_per_hour;
    let minutes = (time_of_day % secs_per_hour) / secs_per_min;
    let seconds = time_of_day % secs_per_min;

    let (year, month, day) = days_to_ymd(days_since_epoch as i64);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let mut remaining = days;
    let mut year = 1970;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days_in_month in &days_in_months {
        if remaining < *days_in_month {
            break;
        }
        remaining -= *days_in_month;
        month += 1;
    }

    let day = remaining + 1;

    (year, month, day as u32)
}

/// Check if a year is a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tempfile::tempdir;

    /// Helper to create a dummy file with specific content.
    fn create_dummy_file(path: &Path, content: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)
    }

    /// Helper to compute SHA256 of content directly.
    fn compute_sha256(content: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        hex_encode(&hash)
    }

    #[test]
    fn test_hex_encode() {
        let bytes = [0xde, 0xad, 0xbe, 0xef];
        assert_eq!(hex_encode(&bytes), "deadbeef");
    }

    #[test]
    fn test_path_escapes_root() {
        assert!(path_escapes_root(Path::new("../foo")));
        assert!(path_escapes_root(Path::new("foo/../bar/../..")));
        assert!(path_escapes_root(Path::new("/absolute/path")));

        assert!(!path_escapes_root(Path::new("foo/bar")));
        assert!(!path_escapes_root(Path::new("foo")));
        assert!(!path_escapes_root(Path::new("nested/deep/path/file.txt")));
    }

    #[test]
    fn test_normalize_path_str() {
        // On Unix, backslashes are valid in filenames but we normalize anyway
        let path = Path::new("foo/bar/baz.txt");
        assert_eq!(normalize_path_str(path), "foo/bar/baz.txt");
    }

    #[test]
    fn test_hash_file_sha256() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let content = "Hello, World!";
        create_dummy_file(&file_path, content).unwrap();

        let hash = hash_file_sha256(&file_path).unwrap();
        let expected = format!("sha256:{}", compute_sha256(content.as_bytes()));
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_atomic_write() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("atomic_test.txt");
        let content = b"atomic content";

        atomic_write(&file_path, content).unwrap();

        assert!(file_path.exists());
        assert_eq!(fs::read(&file_path).unwrap(), content);
    }

    #[test]
    fn test_write_evidence_pack_basic() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        // Create suite file
        let suite_path = dir.path().join("suite.yaml");
        let suite_content = "scenario_id: test\nversion: 1\n";
        create_dummy_file(&suite_path, suite_content).unwrap();

        // Create artifact files
        let artifact1_path = output_root.join("results/run_001.json");
        let artifact1_content = r#"{"result": "ok"}"#;
        create_dummy_file(&artifact1_path, artifact1_content).unwrap();

        let artifact2_path = output_root.join("nested/deep/run_002.json");
        let artifact2_content = r#"{"result": "also_ok"}"#;
        create_dummy_file(&artifact2_path, artifact2_content).unwrap();

        // Write evidence pack
        let artifacts = vec![
            PathBuf::from("results/run_001.json"),
            PathBuf::from("nested/deep/run_002.json"),
        ];
        write_evidence_pack(&output_root, &suite_path, &artifacts).unwrap();

        // Verify evidence_pack directory exists
        let evidence_pack_dir = output_root.join("evidence_pack");
        assert!(evidence_pack_dir.exists());

        // Verify manifest.json exists and parses
        let manifest_path = evidence_pack_dir.join("manifest.json");
        assert!(manifest_path.exists());
        let manifest_content = fs::read_to_string(&manifest_path).unwrap();
        let manifest: Manifest = serde_json::from_str(&manifest_content).unwrap();

        // Verify required fields
        assert_eq!(manifest.evidence_pack_schema_version, "v1");
        assert!(!manifest.paraphina_version.is_empty());
        assert!(manifest.generated_at_unix_ms > 0);

        // Verify suite info
        assert_eq!(manifest.suite.copied_to, "evidence_pack/suite.yaml");
        assert!(manifest.suite.sha256.starts_with("sha256:"));

        // Verify suite.yaml exists and matches
        let suite_dest = evidence_pack_dir.join("suite.yaml");
        assert!(suite_dest.exists());
        assert_eq!(fs::read_to_string(&suite_dest).unwrap(), suite_content);

        // Verify SHA256SUMS exists
        let sha256sums_path = evidence_pack_dir.join("SHA256SUMS");
        assert!(sha256sums_path.exists());
    }

    #[test]
    fn test_manifest_artifacts_sorted() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        // Create suite file
        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite content").unwrap();

        // Create artifacts with unsorted names
        create_dummy_file(&output_root.join("z_last.json"), "{}").unwrap();
        create_dummy_file(&output_root.join("a_first.json"), "{}").unwrap();
        create_dummy_file(&output_root.join("m_middle.json"), "{}").unwrap();

        let artifacts = vec![
            PathBuf::from("z_last.json"),
            PathBuf::from("a_first.json"),
            PathBuf::from("m_middle.json"),
        ];
        write_evidence_pack(&output_root, &suite_path, &artifacts).unwrap();

        // Parse manifest
        let manifest_path = output_root.join("evidence_pack/manifest.json");
        let manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();

        // Verify artifacts are sorted by path
        let paths: Vec<&str> = manifest.artifacts.iter().map(|a| a.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "a_first.json",
                "evidence_pack/suite.yaml",
                "m_middle.json",
                "z_last.json"
            ]
        );
    }

    #[test]
    fn test_manifest_excludes_itself() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        create_dummy_file(&output_root.join("artifact.json"), "{}").unwrap();

        write_evidence_pack(&output_root, &suite_path, &[PathBuf::from("artifact.json")]).unwrap();

        let manifest_path = output_root.join("evidence_pack/manifest.json");
        let manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();

        // Ensure manifest.json is not in artifacts list
        for artifact in &manifest.artifacts {
            assert_ne!(artifact.path, "evidence_pack/manifest.json");
        }
    }

    #[test]
    fn test_sha256sums_covers_all_files() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite content").unwrap();

        create_dummy_file(&output_root.join("artifact1.json"), r#"{"a":1}"#).unwrap();
        create_dummy_file(&output_root.join("sub/artifact2.json"), r#"{"b":2}"#).unwrap();

        write_evidence_pack(
            &output_root,
            &suite_path,
            &[
                PathBuf::from("artifact1.json"),
                PathBuf::from("sub/artifact2.json"),
            ],
        )
        .unwrap();

        // Read SHA256SUMS
        let sha256sums_path = output_root.join("evidence_pack/SHA256SUMS");
        let sha256sums_content = fs::read_to_string(&sha256sums_path).unwrap();

        // Parse lines
        let paths_in_sums: HashSet<&str> = sha256sums_content
            .lines()
            .filter_map(|line| line.split_whitespace().nth(1))
            .collect();

        // Must include: manifest.json, suite.yaml, artifact1.json, sub/artifact2.json
        assert!(paths_in_sums.contains("evidence_pack/manifest.json"));
        assert!(paths_in_sums.contains("evidence_pack/suite.yaml"));
        assert!(paths_in_sums.contains("artifact1.json"));
        assert!(paths_in_sums.contains("sub/artifact2.json"));
    }

    #[test]
    fn test_sha256sums_hashes_match() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        let suite_content = "test suite content";
        create_dummy_file(&suite_path, suite_content).unwrap();

        let artifact_content = r#"{"test": "data"}"#;
        create_dummy_file(&output_root.join("artifact.json"), artifact_content).unwrap();

        write_evidence_pack(&output_root, &suite_path, &[PathBuf::from("artifact.json")]).unwrap();

        // Read SHA256SUMS and verify each hash
        let sha256sums_path = output_root.join("evidence_pack/SHA256SUMS");
        let sha256sums_content = fs::read_to_string(&sha256sums_path).unwrap();

        for line in sha256sums_content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(parts.len(), 2, "Invalid SHA256SUMS line format");

            let expected_hash = parts[0];
            let file_path_str = parts[1];
            let file_path = output_root.join(file_path_str);

            // Recompute hash and compare
            let actual_hash = hash_file_sha256_raw(&file_path).unwrap();
            assert_eq!(
                actual_hash, expected_hash,
                "Hash mismatch for {}",
                file_path_str
            );
        }
    }

    #[test]
    fn test_error_on_path_escape_parent() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        let result = write_evidence_pack(
            &output_root,
            &suite_path,
            &[PathBuf::from("../escaped.json")],
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("escapes output_root"));
    }

    #[test]
    fn test_error_on_absolute_path() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        let result = write_evidence_pack(
            &output_root,
            &suite_path,
            &[PathBuf::from("/absolute/path.json")],
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("escapes output_root"));
    }

    #[test]
    fn test_error_on_missing_artifact() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        let result = write_evidence_pack(
            &output_root,
            &suite_path,
            &[PathBuf::from("nonexistent.json")],
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn test_suite_byte_identical_copy() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        // Create suite with specific binary content (including trailing newlines, etc.)
        let suite_path = dir.path().join("suite.yaml");
        let suite_content = b"scenario_id: test\nsteps: 100\n\n# trailing comment\n";
        fs::write(&suite_path, suite_content).unwrap();

        write_evidence_pack(&output_root, &suite_path, &[]).unwrap();

        // Verify byte-identical copy
        let copied_suite = output_root.join("evidence_pack/suite.yaml");
        let copied_content = fs::read(&copied_suite).unwrap();
        assert_eq!(copied_content, suite_content);
    }

    #[test]
    fn test_repository_info_fields_present() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        write_evidence_pack(&output_root, &suite_path, &[]).unwrap();

        // Parse manifest and verify repository fields are present (may be null)
        let manifest_path = output_root.join("evidence_pack/manifest.json");
        let manifest_content = fs::read_to_string(&manifest_path).unwrap();

        // Parse as serde_json::Value to check field presence
        let manifest_value: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();

        assert!(manifest_value.get("repository").is_some());
        let repo = manifest_value.get("repository").unwrap();
        assert!(repo.get("git_commit").is_some());
        assert!(repo.get("cargo_lock_sha256").is_some());
        assert!(repo.get("sim_output_schema_sha256").is_some());
    }

    #[test]
    fn test_nested_artifact_paths() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().join("output");
        fs::create_dir_all(&output_root).unwrap();

        let suite_path = dir.path().join("suite.yaml");
        create_dummy_file(&suite_path, "suite").unwrap();

        // Create deeply nested artifact
        let nested_path = output_root.join("a/b/c/d/artifact.json");
        create_dummy_file(&nested_path, "{}").unwrap();

        write_evidence_pack(
            &output_root,
            &suite_path,
            &[PathBuf::from("a/b/c/d/artifact.json")],
        )
        .unwrap();

        // Verify it's in the manifest
        let manifest_path = output_root.join("evidence_pack/manifest.json");
        let manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();

        let nested_artifact = manifest
            .artifacts
            .iter()
            .find(|a| a.path == "a/b/c/d/artifact.json");
        assert!(nested_artifact.is_some());
    }

    #[test]
    fn test_deterministic_manifest_json_output() {
        // Run twice and verify the manifest structure is identical (ignoring timestamp)
        let dir1 = tempdir().unwrap();
        let output_root1 = dir1.path().join("output");
        fs::create_dir_all(&output_root1).unwrap();

        let dir2 = tempdir().unwrap();
        let output_root2 = dir2.path().join("output");
        fs::create_dir_all(&output_root2).unwrap();

        let suite_content = "suite: identical";

        let suite_path1 = dir1.path().join("suite.yaml");
        create_dummy_file(&suite_path1, suite_content).unwrap();

        let suite_path2 = dir2.path().join("suite.yaml");
        create_dummy_file(&suite_path2, suite_content).unwrap();

        let artifact_content = r#"{"same": "content"}"#;

        create_dummy_file(&output_root1.join("artifact.json"), artifact_content).unwrap();
        create_dummy_file(&output_root2.join("artifact.json"), artifact_content).unwrap();

        write_evidence_pack(
            &output_root1,
            &suite_path1,
            &[PathBuf::from("artifact.json")],
        )
        .unwrap();
        write_evidence_pack(
            &output_root2,
            &suite_path2,
            &[PathBuf::from("artifact.json")],
        )
        .unwrap();

        // Parse both manifests
        let manifest1: Manifest = serde_json::from_str(
            &fs::read_to_string(output_root1.join("evidence_pack/manifest.json")).unwrap(),
        )
        .unwrap();
        let manifest2: Manifest = serde_json::from_str(
            &fs::read_to_string(output_root2.join("evidence_pack/manifest.json")).unwrap(),
        )
        .unwrap();

        // Compare fields (except timestamp and repository which may vary)
        assert_eq!(
            manifest1.evidence_pack_schema_version,
            manifest2.evidence_pack_schema_version
        );
        assert_eq!(manifest1.paraphina_version, manifest2.paraphina_version);
        assert_eq!(manifest1.suite.sha256, manifest2.suite.sha256);
        assert_eq!(manifest1.suite.copied_to, manifest2.suite.copied_to);
        assert_eq!(manifest1.artifacts, manifest2.artifacts);
    }

    // ========================================================================
    // write_root_evidence_pack tests
    // ========================================================================

    #[test]
    fn test_write_root_evidence_pack_basic() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("output");
        fs::create_dir_all(&root).unwrap();

        // Create some files
        create_dummy_file(&root.join("file1.txt"), "content1").unwrap();
        create_dummy_file(&root.join("file2.json"), r#"{"a":1}"#).unwrap();
        create_dummy_file(&root.join("subdir/nested.txt"), "nested").unwrap();

        // Write root evidence pack
        let file_count = write_root_evidence_pack(&root).unwrap();
        assert_eq!(file_count, 3);

        // Verify evidence_pack directory exists
        let evidence_pack_dir = root.join("evidence_pack");
        assert!(evidence_pack_dir.exists());

        // Verify SHA256SUMS exists
        let sha256sums_path = evidence_pack_dir.join("SHA256SUMS");
        assert!(sha256sums_path.exists());

        // Verify manifest.json exists
        let manifest_path = evidence_pack_dir.join("manifest.json");
        assert!(manifest_path.exists());

        // Verify suite.yaml exists (for verifier compatibility)
        let suite_path = evidence_pack_dir.join("suite.yaml");
        assert!(suite_path.exists());
    }

    #[test]
    fn test_write_root_evidence_pack_excludes_junk_dirs() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("output");
        fs::create_dir_all(&root).unwrap();

        // Create a regular file
        create_dummy_file(&root.join("good_file.txt"), "content").unwrap();

        // Create files in excluded directories
        create_dummy_file(&root.join("__pycache__/cache.pyc"), "cache").unwrap();
        create_dummy_file(&root.join(".git/objects/abc"), "git object").unwrap();
        create_dummy_file(&root.join("target/debug/binary"), "binary").unwrap();

        // Write root evidence pack
        let file_count = write_root_evidence_pack(&root).unwrap();
        assert_eq!(file_count, 1); // Only good_file.txt

        // Verify SHA256SUMS only contains good_file.txt (plus evidence_pack files)
        let sha256sums_path = root.join("evidence_pack/SHA256SUMS");
        let sha256sums_content = fs::read_to_string(&sha256sums_path).unwrap();

        assert!(!sha256sums_content.contains("__pycache__"));
        assert!(!sha256sums_content.contains(".git"));
        assert!(!sha256sums_content.contains("target"));
        assert!(sha256sums_content.contains("good_file.txt"));
    }

    #[test]
    fn test_write_root_evidence_pack_deterministic() {
        let dir1 = tempdir().unwrap();
        let root1 = dir1.path().join("output");
        fs::create_dir_all(&root1).unwrap();

        let dir2 = tempdir().unwrap();
        let root2 = dir2.path().join("output");
        fs::create_dir_all(&root2).unwrap();

        // Create identical files in both directories
        create_dummy_file(&root1.join("z_file.txt"), "z content").unwrap();
        create_dummy_file(&root1.join("a_file.txt"), "a content").unwrap();
        create_dummy_file(&root1.join("m_file.txt"), "m content").unwrap();

        create_dummy_file(&root2.join("z_file.txt"), "z content").unwrap();
        create_dummy_file(&root2.join("a_file.txt"), "a content").unwrap();
        create_dummy_file(&root2.join("m_file.txt"), "m content").unwrap();

        // Write root evidence packs
        write_root_evidence_pack(&root1).unwrap();
        write_root_evidence_pack(&root2).unwrap();

        // Compare SHA256SUMS (excluding evidence_pack lines which have timestamps)
        let sha256sums1 = fs::read_to_string(root1.join("evidence_pack/SHA256SUMS")).unwrap();
        let sha256sums2 = fs::read_to_string(root2.join("evidence_pack/SHA256SUMS")).unwrap();

        // Extract user file lines (not evidence_pack/* lines)
        let user_lines1: Vec<&str> = sha256sums1
            .lines()
            .filter(|l| !l.contains("evidence_pack/"))
            .collect();
        let user_lines2: Vec<&str> = sha256sums2
            .lines()
            .filter(|l| !l.contains("evidence_pack/"))
            .collect();

        assert_eq!(user_lines1, user_lines2);

        // Verify files are sorted alphabetically
        let paths1: Vec<&str> = user_lines1
            .iter()
            .filter_map(|l| l.split_whitespace().nth(1))
            .collect();
        let mut sorted_paths1 = paths1.clone();
        sorted_paths1.sort();
        assert_eq!(paths1, sorted_paths1, "Paths should be sorted");
    }

    #[test]
    fn test_write_root_evidence_pack_error_nonexistent() {
        let result = write_root_evidence_pack(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn test_write_root_evidence_pack_error_not_directory() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("not_a_dir.txt");
        create_dummy_file(&file_path, "content").unwrap();

        let result = write_root_evidence_pack(&file_path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a directory"));
    }

    #[test]
    fn test_write_root_evidence_pack_verifiable() {
        use super::super::evidence_pack_verify::verify_evidence_pack_dir;

        let dir = tempdir().unwrap();
        let root = dir.path().join("output");
        fs::create_dir_all(&root).unwrap();

        // Create some files
        create_dummy_file(&root.join("data.json"), r#"{"key": "value"}"#).unwrap();
        create_dummy_file(&root.join("nested/file.txt"), "nested content").unwrap();

        // Write evidence pack
        write_root_evidence_pack(&root).unwrap();

        // Verify with the verifier
        let report = verify_evidence_pack_dir(&root).unwrap();
        assert!(report.files_verified >= 4); // At least manifest, suite, data.json, nested/file.txt
        assert_eq!(report.packs_verified, 1);
    }
}
