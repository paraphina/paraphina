# Evidence Pack Manifest

**Version:** 1.0  
**Status:** Active

## Overview

The Evidence Pack Manifest provides a deterministic, verifiable record of all files produced during a Phase AB run. It enables:

1. **Integrity verification**: Ensure evidence pack files haven't been tampered with
2. **Reproducibility**: Capture all metadata needed to reproduce a run
3. **CI integration**: Upload artifacts with cryptographic guarantees

## Manifest Format

The manifest is a JSON file (`manifest.json`) stored inside the evidence pack directory:

```json
{
  "schema_version": 1,
  "created_utc": "2025-01-04T12:00:00.000000Z",
  "metadata": {
    "cli_args": ["smoke", "--auto-generate-phasea", "--out-dir", "runs/ci/phase_ab_smoke", "--seed", "12345"],
    "seed": 12345,
    "python_version": "3.11.0",
    "git_commit": "abc123def456..."
  },
  "files": [
    {"path": "confidence_report.json", "bytes": 1234, "sha256": "abc123..."},
    {"path": "confidence_report.md", "bytes": 2345, "sha256": "def456..."},
    {"path": "phase_ab_manifest.json", "bytes": 789, "sha256": "ghi789..."}
  ]
}
```

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `1` for this version |
| `created_utc` | string | ISO-8601 timestamp (UTC) when manifest was created |
| `metadata.cli_args` | array | CLI arguments used to invoke the command |
| `metadata.seed` | int | Random seed (if provided) |
| `metadata.python_version` | string | Python version used |
| `metadata.git_commit` | string | Git commit hash (if in a git repo) |
| `files` | array | List of files in the evidence pack |
| `files[].path` | string | Relative path (POSIX-style, forward slashes) |
| `files[].bytes` | int | File size in bytes |
| `files[].sha256` | string | Lowercase hex SHA-256 hash |

### Key Properties

- **Deterministic ordering**: Files are sorted alphabetically by path
- **POSIX paths**: All paths use forward slashes (`/`) regardless of platform
- **Self-exclusion**: `manifest.json` is never included in the file list
- **Trailing newline**: Manifest file ends with a newline

## Directory Structure

After a successful Phase AB smoke run with `--out-dir runs/ci/phase_ab_smoke`:

```
runs/ci/phase_ab_smoke/
├── phase_ab_manifest.json      # Phase AB run metadata
├── confidence_report.json       # Machine-readable gating report
├── confidence_report.md         # Human-readable gating report
├── phase_ab_summary.json        # CI summary (outcome, paths, seed, etc.)
└── evidence_pack/
    ├── manifest.json            # Evidence pack manifest (this file)
    ├── phase_ab_manifest.json   # Copy for verification
    ├── confidence_report.json   # Copy for verification
    └── confidence_report.md     # Copy for verification
```

## Verification

### Local Verification

To verify an evidence pack after download or retrieval:

```bash
python3 -m batch_runs.phase_ab.cli verify-evidence runs/ci/phase_ab_smoke/evidence_pack
```

**Success output:**
```
OK: Evidence pack verified (3 files)
```

**Failure output (example - tampered file):**
```
FAIL: Evidence pack verification failed (1 error(s)):

Hash mismatch: confidence_report.json
  Expected SHA-256: abc123...
  Actual SHA-256: def456...
```

### Verification Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Verification passed |
| 2 | Verification failed (missing files, hash mismatch, extra files) |

### Programmatic Verification

```python
from batch_runs.evidence_pack import verify_manifest, ManifestError

try:
    verify_manifest("/path/to/evidence_pack")
    print("Verification passed")
except ManifestError as e:
    print(f"Verification failed: {e}")
```

## CI Integration

### GitHub Actions Artifact

The Phase AB smoke workflow uploads the evidence pack as a GitHub Actions artifact named **`phase-ab-evidence-pack`**.

To find it:
1. Go to the GitHub Actions run
2. Click on the "phase-ab-smoke" job
3. Scroll to "Artifacts" section
4. Download "phase-ab-evidence-pack"

### Verifying Downloaded Artifacts

After downloading the artifact:

```bash
# Unzip the artifact
unzip phase-ab-evidence-pack.zip -d artifact

# Verify
python3 -m batch_runs.phase_ab.cli verify-evidence artifact/evidence_pack
```

### Workflow Configuration

The CI workflow:
1. Runs smoke with deterministic seed: `--seed 12345`
2. Outputs to fixed location: `--out-dir runs/ci/phase_ab_smoke`
3. Verifies the evidence pack before uploading
4. Uploads artifact even if smoke step fails (`if: always()`)
5. Writes a job summary with outcome and file counts

## API Reference

### `write_manifest(evidence_dir, metadata, manifest_name="manifest.json")`

Write a deterministic manifest for an evidence pack.

**Parameters:**
- `evidence_dir`: Path to evidence pack directory
- `metadata`: Dict with metadata (cli_args, seed, etc.)
- `manifest_name`: Name of manifest file (default: `manifest.json`)

**Returns:** Path to the written manifest file

### `verify_manifest(evidence_dir, manifest_name="manifest.json", allow_extra=False)`

Verify an evidence pack against its manifest.

**Parameters:**
- `evidence_dir`: Path to evidence pack directory
- `manifest_name`: Name of manifest file
- `allow_extra`: If `True`, allow extra files not in manifest

**Raises:** `ManifestError` if verification fails

### `compute_sha256(path)`

Compute SHA-256 hash of a file.

**Parameters:**
- `path`: Path to file

**Returns:** Lowercase hex-encoded SHA-256 hash

### `ManifestError`

Exception raised for manifest verification failures. Contains detailed, actionable error messages.

## Failure Modes

| Condition | Error Message |
|-----------|---------------|
| Manifest missing | "Manifest not found: /path/to/manifest.json" |
| File missing | "Missing file: <path>\n  Expected SHA-256: ..." |
| Size mismatch | "Size mismatch: <path>\n  Expected: N bytes\n  Actual: M bytes" |
| Hash mismatch | "Hash mismatch: <path>\n  Expected SHA-256: ...\n  Actual SHA-256: ..." |
| Extra files | "Extra files not in manifest (N found):\n  - file1.txt\n  - file2.txt" |
| Invalid JSON | "Invalid JSON in manifest: /path/to/manifest.json" |
| Wrong schema | "Unsupported manifest schema version: N" |

## Design Rationale

### Why SHA-256?

- Cryptographically secure hash function
- Standard in file integrity verification
- Supported by Python's stdlib (`hashlib`)

### Why POSIX paths?

- Cross-platform determinism
- Windows paths would break verification on Linux/macOS
- JSON doesn't escape forward slashes

### Why exclude manifest from file list?

- Chicken-and-egg: can't hash a file that contains its own hash
- Industry standard practice (e.g., MANIFEST.MF in Java JARs)

### Why sorted files?

- Deterministic output regardless of filesystem enumeration order
- Makes diffing manifests meaningful
- Required for reproducible builds

## Testing

Run unit tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Or specifically:

```bash
python3 -m unittest tests.test_evidence_pack_manifest -v
```

## Changelog

### v1.0 (2025-01-04)

- Initial implementation
- Support for write_manifest, verify_manifest, compute_sha256
- Integration with Phase AB CLI (smoke, run, verify-evidence commands)
- GitHub Actions artifact upload

