"""
evidence_pack - Evidence Pack Manifest utilities.

This module provides stdlib-only helpers for creating and verifying
deterministic evidence pack manifests.

Usage:
    from batch_runs.evidence_pack import (
        compute_sha256,
        write_manifest,
        verify_manifest,
        ManifestError,
    )
"""

from batch_runs.evidence_pack.manifest import (
    ManifestError,
    compute_sha256,
    count_manifest_files,
    verify_manifest,
    write_manifest,
)

__all__ = [
    "ManifestError",
    "compute_sha256",
    "count_manifest_files",
    "verify_manifest",
    "write_manifest",
]

