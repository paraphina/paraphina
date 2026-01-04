"""
manifest.py

Deterministic manifest generation and verification for evidence packs.

All functions use stdlib only. No third-party dependencies.

Manifest format (JSON):
{
    "schema_version": 1,
    "created_utc": "2025-01-04T12:00:00.000000Z",
    "metadata": {
        "cli_args": [...],
        "seed": 12345,
        "python_version": "3.11.0",
        "git_commit": "abc123..."
    },
    "files": [
        {"path": "relative/path.ext", "bytes": 1234, "sha256": "abc..."},
        ...
    ]
}

Files are sorted by path (POSIX-style, forward slashes) for determinism.
The manifest file itself (manifest.json) is excluded from the file list.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Exception
# =============================================================================

class ManifestError(Exception):
    """
    Exception raised for manifest verification failures.
    
    Provides human-readable error messages with actionable information.
    """
    pass


# =============================================================================
# Helpers
# =============================================================================

def compute_sha256(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Lowercase hex-encoded SHA-256 hash
        
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> Optional[str]:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _collect_files(evidence_dir: Path, manifest_name: str) -> List[Dict[str, Any]]:
    """
    Collect all files in evidence_dir recursively.
    
    Args:
        evidence_dir: Root directory to scan
        manifest_name: Name of manifest file to exclude
        
    Returns:
        List of file info dicts, sorted by POSIX path
    """
    files: List[Dict[str, Any]] = []
    
    for root, _dirs, filenames in os.walk(evidence_dir):
        root_path = Path(root)
        for filename in filenames:
            # Skip manifest file
            if filename == manifest_name:
                continue
            
            file_path = root_path / filename
            rel_path = file_path.relative_to(evidence_dir)
            # Use POSIX-style paths for cross-platform determinism
            posix_path = rel_path.as_posix()
            
            file_size = file_path.stat().st_size
            file_hash = compute_sha256(file_path)
            
            files.append({
                "path": posix_path,
                "bytes": file_size,
                "sha256": file_hash,
            })
    
    # Sort by path for deterministic ordering
    files.sort(key=lambda f: f["path"])
    return files


# =============================================================================
# Write Manifest
# =============================================================================

def write_manifest(
    evidence_dir: Path,
    metadata: Dict[str, Any],
    manifest_name: str = "manifest.json",
) -> Path:
    """
    Write a deterministic manifest.json for an evidence pack.
    
    Args:
        evidence_dir: Directory containing evidence pack files
        metadata: Metadata dict to include (cli_args, seed, etc.)
                  Will be augmented with python_version and git_commit
        manifest_name: Name of manifest file (default: manifest.json)
        
    Returns:
        Path to the written manifest file
        
    Raises:
        FileNotFoundError: If evidence_dir does not exist
    """
    evidence_dir = Path(evidence_dir).resolve()
    
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory does not exist: {evidence_dir}")
    
    if not evidence_dir.is_dir():
        raise NotADirectoryError(f"Evidence path is not a directory: {evidence_dir}")
    
    # Build metadata with required fields
    full_metadata = dict(metadata)  # Copy
    full_metadata.setdefault("python_version", sys.version.split()[0])
    if "git_commit" not in full_metadata:
        full_metadata["git_commit"] = _get_git_commit()
    
    # Collect files
    files = _collect_files(evidence_dir, manifest_name)
    
    # Build manifest
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": full_metadata,
        "files": files,
    }
    
    # Write with deterministic formatting
    manifest_path = evidence_dir / manifest_name
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")  # Trailing newline
    
    return manifest_path


# =============================================================================
# Verify Manifest
# =============================================================================

def verify_manifest(
    evidence_dir: Path,
    manifest_name: str = "manifest.json",
    allow_extra: bool = False,
) -> None:
    """
    Verify an evidence pack against its manifest.
    
    Checks:
    1. Manifest file exists
    2. All listed files exist
    3. All SHA-256 hashes match
    4. No extra files in directory (unless allow_extra=True)
    
    Args:
        evidence_dir: Directory containing evidence pack
        manifest_name: Name of manifest file (default: manifest.json)
        allow_extra: If True, allow extra files not in manifest
        
    Raises:
        ManifestError: If verification fails (with detailed message)
    """
    evidence_dir = Path(evidence_dir).resolve()
    manifest_path = evidence_dir / manifest_name
    
    # Check manifest exists
    if not manifest_path.exists():
        raise ManifestError(
            f"Manifest not found: {manifest_path}\n"
            f"Evidence pack verification requires a manifest file."
        )
    
    # Parse manifest
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ManifestError(
            f"Invalid JSON in manifest: {manifest_path}\n"
            f"Parse error: {e}"
        )
    
    # Validate schema version
    schema_version = manifest.get("schema_version")
    if schema_version != 1:
        raise ManifestError(
            f"Unsupported manifest schema version: {schema_version}\n"
            f"Expected: 1"
        )
    
    # Get file list from manifest
    manifest_files = manifest.get("files", [])
    if not isinstance(manifest_files, list):
        raise ManifestError(
            f"Invalid manifest format: 'files' must be a list"
        )
    
    # Build set of expected paths
    expected_paths = set()
    for file_info in manifest_files:
        expected_paths.add(file_info["path"])
    
    # Verify each file in manifest
    errors: List[str] = []
    
    for file_info in manifest_files:
        rel_path = file_info["path"]
        expected_hash = file_info["sha256"]
        expected_bytes = file_info["bytes"]
        
        file_path = evidence_dir / rel_path
        
        # Check file exists
        if not file_path.exists():
            errors.append(
                f"Missing file: {rel_path}\n"
                f"  Expected SHA-256: {expected_hash}"
            )
            continue
        
        # Check file size (fast check before hash)
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"Size mismatch: {rel_path}\n"
                f"  Expected: {expected_bytes} bytes\n"
                f"  Actual: {actual_bytes} bytes"
            )
            continue
        
        # Check hash
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            errors.append(
                f"Hash mismatch: {rel_path}\n"
                f"  Expected SHA-256: {expected_hash}\n"
                f"  Actual SHA-256: {actual_hash}"
            )
    
    # Check for extra files (unless allow_extra)
    if not allow_extra:
        actual_files = set()
        for root, _dirs, filenames in os.walk(evidence_dir):
            root_path = Path(root)
            for filename in filenames:
                if filename == manifest_name:
                    continue
                file_path = root_path / filename
                rel_path = file_path.relative_to(evidence_dir).as_posix()
                actual_files.add(rel_path)
        
        extra_files = actual_files - expected_paths
        if extra_files:
            extra_list = sorted(extra_files)
            errors.append(
                f"Extra files not in manifest ({len(extra_list)} found):\n"
                + "\n".join(f"  - {p}" for p in extra_list[:10])
                + (f"\n  ... and {len(extra_list) - 10} more" if len(extra_list) > 10 else "")
            )
    
    # Raise if any errors
    if errors:
        error_msg = f"Evidence pack verification failed ({len(errors)} error(s)):\n\n"
        error_msg += "\n\n".join(errors)
        raise ManifestError(error_msg)


def count_manifest_files(manifest_path: Path) -> int:
    """
    Count the number of files listed in a manifest.
    
    Args:
        manifest_path: Path to manifest.json
        
    Returns:
        Number of files in the manifest
        
    Raises:
        FileNotFoundError: If manifest does not exist
        json.JSONDecodeError: If manifest is invalid JSON
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return len(manifest.get("files", []))

