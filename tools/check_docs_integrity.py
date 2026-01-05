#!/usr/bin/env python3
"""
Docs Integrity Gate: Validates whitepaper references and locks canonical spec.

This tool provides institutional-grade enforcement of documentation integrity:
1. Validates "Implemented:" references in docs/WHITEPAPER.md point to real files
   (and optionally validates referenced symbols exist via substring check).
2. Locks the canonical spec appendix (Part II) behind a committed SHA256 baseline.

Exit codes:
    0 - OK: all checks passed
    1 - Internal error (script bug, I/O error, etc.)
    2 - Integrity violation (missing file, missing symbol, hash mismatch)

Usage:
    python3 tools/check_docs_integrity.py
    python3 tools/check_docs_integrity.py --update-canonical-hash
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import NamedTuple


# Constants
WHITEPAPER_PATH = Path("docs/WHITEPAPER.md")
BASELINE_HASH_PATH = Path("docs/CANONICAL_SPEC_V1_SHA256.txt")
CANONICAL_MARKER = "<!-- CANONICAL_SPEC_V1_BEGIN -->"

# Known file extensions for path detection
KNOWN_EXTENSIONS = {".rs", ".py", ".md", ".yml", ".yaml", ".toml"}

# Regex for a valid symbol identifier (last component after ::)
SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RefError(NamedTuple):
    """A reference validation error."""
    doc_path: str
    line: int
    message: str


def is_path_like(token: str) -> bool:
    """
    Check if a token looks like a file path.
    Must contain '/' AND end with a known extension.
    """
    if "/" not in token:
        return False
    
    # Handle symbol annotations: path/to/file.rs::Symbol::method
    base_path = token.split("::")[0] if "::" in token else token
    
    # Check extension
    for ext in KNOWN_EXTENSIONS:
        if base_path.endswith(ext):
            return True
    
    return False


def parse_implemented_refs(content: str, doc_path: str) -> list[tuple[int, str]]:
    """
    Parse lines containing 'Implemented:' and extract backticked tokens
    that look like file paths.
    
    Returns list of (line_number, token) tuples.
    """
    refs = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, start=1):
        if "Implemented:" not in line:
            continue
        
        # Extract all backticked tokens
        backtick_pattern = re.compile(r"`([^`]+)`")
        for match in backtick_pattern.finditer(line):
            token = match.group(1)
            if is_path_like(token):
                refs.append((line_num, token))
    
    return refs


def validate_ref(token: str, repo_root: Path) -> tuple[bool, str | None]:
    """
    Validate a file reference token.
    
    Token format: path/to/file.ext or path/to/file.ext::Symbol::method
    
    Returns (is_valid, error_message).
    Returns (True, None) to skip validation for example paths.
    """
    # Split path from symbol annotation
    if "::" in token:
        path_part, symbol_part = token.split("::", 1)
    else:
        path_part = token
        symbol_part = None
    
    # Get the first directory component
    parts = path_part.split("/")
    if parts:
        first_dir = parts[0]
        # If the first directory doesn't exist, this is likely an example path
        # (e.g., "path/to/file.rs" in documentation)
        if not (repo_root / first_dir).exists():
            return (True, None)  # Skip validation for example paths
    
    # Check if file exists
    file_path = repo_root / path_part
    if not file_path.exists():
        return (False, f"missing file: {path_part}")
    
    # If symbol annotation present, validate it
    if symbol_part:
        # Get the last identifier (e.g., "method" from "Symbol::method")
        identifiers = symbol_part.split("::")
        last_ident = identifiers[-1]
        
        # Only check if it looks like a valid identifier
        if SYMBOL_PATTERN.match(last_ident):
            try:
                file_content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return (False, f"cannot read file: {path_part}")
            
            # Simple substring check for the symbol
            if last_ident not in file_content:
                return (False, f"missing symbol: {last_ident} in {path_part}")
    
    return (True, "")


def validate_implemented_refs(whitepaper_content: str, repo_root: Path) -> list[RefError]:
    """
    Validate all Implemented: references in the whitepaper.
    Returns list of errors sorted by (doc_path, line, message).
    """
    errors: list[RefError] = []
    doc_path = str(WHITEPAPER_PATH)
    
    refs = parse_implemented_refs(whitepaper_content, doc_path)
    
    for line_num, token in refs:
        is_valid, error_msg = validate_ref(token, repo_root)
        # Skip if validation was skipped (example paths) or if valid
        if not is_valid and error_msg is not None:
            errors.append(RefError(
                doc_path=doc_path,
                line=line_num,
                message=error_msg,
            ))
    
    # Sort errors for deterministic output
    errors.sort(key=lambda e: (e.doc_path, e.line, e.message))
    return errors


def compute_canonical_hash(content: str, marker: str) -> tuple[str | None, str]:
    """
    Compute SHA256 of the canonical spec content after the marker.
    
    Returns (hash_hex, error_message). If marker not found, returns (None, error).
    """
    # Find the marker line
    marker_pos = content.find(marker)
    if marker_pos == -1:
        return (None, f"Canonical spec marker '{marker}' not found in {WHITEPAPER_PATH}")
    
    # Find the end of the marker line
    marker_line_end = content.find('\n', marker_pos)
    if marker_line_end == -1:
        # Marker is at end of file with no newline
        canonical_text = ""
    else:
        # Content after the marker line
        canonical_text = content[marker_line_end + 1:]
    
    # Normalize to LF
    canonical_text = canonical_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Compute SHA256
    hash_hex = hashlib.sha256(canonical_text.encode('utf-8')).hexdigest()
    return (hash_hex, "")


def check_canonical_hash(
    whitepaper_content: str,
    baseline_path: Path,
    update_mode: bool,
) -> tuple[bool, str, int]:
    """
    Check or update the canonical spec hash.
    
    Returns (success, message, ref_count_placeholder).
    The ref_count_placeholder is not used here, kept for interface consistency.
    """
    computed_hash, error = compute_canonical_hash(whitepaper_content, CANONICAL_MARKER)
    if computed_hash is None:
        return (False, error, 0)
    
    if update_mode:
        # Write the computed hash to baseline file
        try:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(computed_hash + "\n", encoding="utf-8")
            return (True, f"Updated {baseline_path} with hash: {computed_hash}", 0)
        except OSError as e:
            return (False, f"Failed to write baseline hash: {e}", 0)
    
    # Check mode: compare against baseline
    if not baseline_path.exists():
        return (
            False,
            f"Baseline hash file missing: {baseline_path}\n"
            f"Run with --update-canonical-hash to create it:\n"
            f"  python3 tools/check_docs_integrity.py --update-canonical-hash",
            0,
        )
    
    try:
        expected_hash = baseline_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        return (False, f"Failed to read baseline hash: {e}", 0)
    
    if computed_hash != expected_hash:
        return (
            False,
            f"Canonical spec hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {computed_hash}\n"
            f"\n"
            f"The canonical spec (Part II) has been modified.\n"
            f"Either revert the spec change, or if intentional, run:\n"
            f"  python3 tools/check_docs_integrity.py --update-canonical-hash",
            0,
        )
    
    return (True, "canonical hash OK", 0)


def run_checks(repo_root: Path, update_mode: bool = False) -> int:
    """
    Run all integrity checks.
    
    Returns exit code: 0=OK, 1=internal error, 2=integrity violation.
    """
    whitepaper_path = repo_root / WHITEPAPER_PATH
    baseline_path = repo_root / BASELINE_HASH_PATH
    
    # Read whitepaper
    try:
        whitepaper_content = whitepaper_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {WHITEPAPER_PATH} not found", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ERROR: Failed to read {WHITEPAPER_PATH}: {e}", file=sys.stderr)
        return 1
    
    errors: list[RefError] = []
    
    # Check 1: Validate Implemented: references
    ref_errors = validate_implemented_refs(whitepaper_content, repo_root)
    errors.extend(ref_errors)
    ref_count = len(parse_implemented_refs(whitepaper_content, str(WHITEPAPER_PATH)))
    
    # If there are reference errors, print them
    if ref_errors:
        for err in ref_errors:
            print(f"{err.doc_path}:{err.line} {err.message}")
    
    # Check 2: Canonical spec hash
    hash_ok, hash_msg, _ = check_canonical_hash(whitepaper_content, baseline_path, update_mode)
    
    if update_mode:
        # In update mode, we report the update result
        if ref_errors:
            print(f"\nFAILED: {len(ref_errors)} reference error(s) found.")
            print("Cannot update canonical hash with failing reference checks.")
            return 2
        if not hash_ok:
            print(f"ERROR: {hash_msg}", file=sys.stderr)
            return 1
        print(hash_msg)
        print(f"OK: docs integrity checks passed ({ref_count} references checked; canonical hash updated).")
        return 0
    
    # Normal check mode
    if not hash_ok:
        print(f"\n{hash_msg}")
        errors.append(RefError(doc_path=str(WHITEPAPER_PATH), line=0, message="canonical hash mismatch"))
    
    if errors:
        return 2
    
    print(f"OK: docs integrity checks passed ({ref_count} references checked; canonical hash OK).")
    return 0


def main() -> int:
    """Main entry point."""
    update_mode = "--update-canonical-hash" in sys.argv
    
    # Determine repo root (script is in tools/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Sanity check: ensure we're in the right place
    if not (repo_root / "docs").exists():
        print("ERROR: Cannot find docs/ directory. Run from repo root.", file=sys.stderr)
        return 1
    
    try:
        return run_checks(repo_root, update_mode)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

