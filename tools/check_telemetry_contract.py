#!/usr/bin/env python3
"""
Telemetry Contract Gate: Validates telemetry JSONL files against schema v1.

This tool enforces the telemetry schema contract defined in:
- docs/TELEMETRY_SCHEMA_V1.md (human-readable)
- schemas/telemetry_schema_v1.json (machine-readable)

Exit codes:
    0 - OK: all records valid (or no telemetry files to validate)
    1 - Contract violation (schema error, missing required fields, type mismatch, etc.)
    2 - File not found or CLI usage error
    3 - Internal error (schema file missing, JSON parse error in schema, etc.)

Usage:
    python3 tools/check_telemetry_contract.py                     # Check current directory
    python3 tools/check_telemetry_contract.py path/to/file.jsonl  # Check specific file
    python3 tools/check_telemetry_contract.py path/to/dir/        # Check directory
    python3 tools/check_telemetry_contract.py --help              # Show help
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple


# Maximum number of errors to display before truncating
MAX_ERRORS_DISPLAY = 10

# Common telemetry filenames to auto-detect in a directory
TELEMETRY_FILENAMES = [
    "telemetry.jsonl",
    "mc_runs.jsonl",
    "metrics.jsonl",
    "trajectories.jsonl",
]


class ValidationError(NamedTuple):
    """A single validation error."""
    line: int
    message: str


def load_schema(schema_path: Path) -> dict[str, Any] | None:
    """
    Load the machine-readable schema from JSON file.
    
    Returns None on error (and prints error message).
    """
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}", file=sys.stderr)
        return None
    
    try:
        content = schema_path.read_text(encoding="utf-8")
        schema = json.loads(content)
        return schema
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in schema file: {e}", file=sys.stderr)
        return None
    except OSError as e:
        print(f"ERROR: Cannot read schema file: {e}", file=sys.stderr)
        return None


def is_finite_number(value: Any) -> bool:
    """Check if value is a finite number (not NaN, not Inf)."""
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, bool):
        return False
    return math.isfinite(value)


def check_type(value: Any, expected_type: str | list[str], field_name: str) -> str | None:
    """
    Check if value matches the expected type.
    
    Returns error message if type mismatch, None if OK.
    """
    # Handle union types (e.g., ["number", "null"])
    if isinstance(expected_type, list):
        for t in expected_type:
            if check_type(value, t, field_name) is None:
                return None
        return f"field '{field_name}' expected one of {expected_type}, got {type(value).__name__}"
    
    if expected_type == "integer":
        # JSON integers are Python int, but float with .0 is also acceptable
        if isinstance(value, bool):
            return f"field '{field_name}' expected integer, got boolean"
        if isinstance(value, int):
            return None
        if isinstance(value, float) and value == int(value) and math.isfinite(value):
            return None
        return f"field '{field_name}' expected integer, got {type(value).__name__}"
    
    elif expected_type == "number":
        if isinstance(value, bool):
            return f"field '{field_name}' expected number, got boolean"
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                return f"field '{field_name}' is not finite (NaN/Inf not allowed)"
            return None
        return f"field '{field_name}' expected number, got {type(value).__name__}"
    
    elif expected_type == "string":
        if isinstance(value, str):
            return None
        return f"field '{field_name}' expected string, got {type(value).__name__}"
    
    elif expected_type == "boolean":
        if isinstance(value, bool):
            return None
        return f"field '{field_name}' expected boolean, got {type(value).__name__}"
    
    elif expected_type == "null":
        if value is None:
            return None
        return f"field '{field_name}' expected null, got {type(value).__name__}"
    
    elif expected_type == "array_of_integer":
        if not isinstance(value, list):
            return f"field '{field_name}' expected array, got {type(value).__name__}"
        for i, elem in enumerate(value):
            if isinstance(elem, bool):
                return f"field '{field_name}[{i}]' expected integer, got boolean"
            if not isinstance(elem, int):
                if isinstance(elem, float) and elem == int(elem) and math.isfinite(elem):
                    continue
                return f"field '{field_name}[{i}]' expected integer, got {type(elem).__name__}"
        return None
    
    else:
        # Unknown type in schema - treat as internal error
        return f"field '{field_name}' has unknown type '{expected_type}' in schema"


def validate_record(
    record: dict[str, Any],
    schema: dict[str, Any],
    line_num: int,
    prev_tick: int | None,
) -> tuple[list[ValidationError], int | None]:
    """
    Validate a single telemetry record against the schema.
    
    Returns (list of errors, current tick value or None).
    """
    errors: list[ValidationError] = []
    
    required_fields = schema.get("required_fields", [])
    optional_fields = schema.get("optional_fields", [])
    field_types = schema.get("field_types", {})
    enums = schema.get("enums", {})
    expected_schema_version = schema.get("invariants", {}).get("schema_version_value", 1)
    
    # Check required fields exist
    for field in required_fields:
        if field not in record:
            errors.append(ValidationError(line_num, f"missing required field: {field}"))
    
    # Check schema_version value
    if "schema_version" in record:
        if record["schema_version"] != expected_schema_version:
            errors.append(ValidationError(
                line_num,
                f"schema_version mismatch: expected {expected_schema_version}, got {record['schema_version']}"
            ))
    
    # Check types for all present fields
    all_known_fields = set(required_fields) | set(optional_fields)
    for field, value in record.items():
        if field in field_types:
            type_error = check_type(value, field_types[field], field)
            if type_error:
                errors.append(ValidationError(line_num, type_error))
        # Note: We don't error on unknown fields (forwards compatibility)
    
    # Check enum constraints
    for field, allowed_values in enums.items():
        if field in record and record[field] not in allowed_values:
            errors.append(ValidationError(
                line_num,
                f"field '{field}' has invalid value '{record[field]}', expected one of {allowed_values}"
            ))
    
    # Check tick monotonicity
    current_tick = None
    if "t" in record:
        t = record["t"]
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            current_tick = int(t)
            if prev_tick is not None and current_tick <= prev_tick:
                errors.append(ValidationError(
                    line_num,
                    f"tick not monotonically increasing: t={current_tick} (prev={prev_tick})"
                ))
    
    return errors, current_tick


def validate_file(file_path: Path, schema: dict[str, Any]) -> list[ValidationError]:
    """
    Validate all records in a JSONL file against the schema.
    
    Returns list of all validation errors.
    """
    errors: list[ValidationError] = []
    prev_tick: int | None = None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                # Parse JSON
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(ValidationError(line_num, f"invalid JSON: {e}"))
                    continue
                
                # Must be an object
                if not isinstance(record, dict):
                    errors.append(ValidationError(line_num, f"expected JSON object, got {type(record).__name__}"))
                    continue
                
                # Validate record
                record_errors, current_tick = validate_record(record, schema, line_num, prev_tick)
                errors.extend(record_errors)
                
                if current_tick is not None:
                    prev_tick = current_tick
    
    except OSError as e:
        errors.append(ValidationError(0, f"file read error: {e}"))
    
    return errors


def find_telemetry_file(path: Path) -> Path | None:
    """
    Find a telemetry file given a path.
    
    If path is a file, return it.
    If path is a directory, look for common telemetry filenames.
    Returns None if not found.
    """
    if path.is_file():
        return path
    
    if path.is_dir():
        for name in TELEMETRY_FILENAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
    
    return None


def validate_telemetry_path(input_path: Path, schema: dict[str, Any]) -> int:
    """
    Validate telemetry at the given path.
    
    Returns exit code: 0=OK, 1=contract violation, 2=not found.
    """
    # Find telemetry file
    telemetry_file = find_telemetry_file(input_path)
    if telemetry_file is None:
        if input_path.is_dir():
            print(f"ERROR: No telemetry file found in directory: {input_path}", file=sys.stderr)
            print(f"  Looked for: {', '.join(TELEMETRY_FILENAMES)}", file=sys.stderr)
        else:
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return 2
    
    # Validate
    errors = validate_file(telemetry_file, schema)
    
    if errors:
        # Count records (for summary)
        try:
            with open(telemetry_file, "r", encoding="utf-8") as f:
                total_lines = sum(1 for line in f if line.strip())
        except OSError:
            total_lines = "?"
        
        print(f"FAILED: {len(errors)} error(s) in {telemetry_file}")
        print(f"  (Total records scanned: {total_lines})")
        print()
        
        # Show first N errors
        for i, err in enumerate(errors[:MAX_ERRORS_DISPLAY]):
            if err.line > 0:
                print(f"  Line {err.line}: {err.message}")
            else:
                print(f"  {err.message}")
        
        if len(errors) > MAX_ERRORS_DISPLAY:
            print(f"  ... and {len(errors) - MAX_ERRORS_DISPLAY} more error(s)")
        
        return 1
    
    # Success
    try:
        with open(telemetry_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
    except OSError:
        total_lines = "?"
    
    print(f"OK: {total_lines} record(s) validated against schema v{schema.get('schema_version', '?')}")
    print(f"  File: {telemetry_file}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="check_telemetry_contract",
        description="""
Telemetry Contract Gate: Validates telemetry JSONL files against schema v1.

Enforces the telemetry schema contract defined in:
  - docs/TELEMETRY_SCHEMA_V1.md (human-readable)
  - schemas/telemetry_schema_v1.json (machine-readable)
""",
        epilog="""
Exit codes:
  0 - OK: all records valid (or no telemetry files to validate)
  1 - Contract violation (schema error, missing required fields, etc.)
  2 - File not found or CLI usage error
  3 - Internal error (schema file missing, etc.)

Examples:
  %(prog)s                          # Check current directory for telemetry files
  %(prog)s path/to/file.jsonl       # Check specific JSONL file
  %(prog)s path/to/run_directory/   # Check directory for telemetry files
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to telemetry JSONL file or directory. "
             "If omitted, checks the current directory for telemetry files.",
    )
    
    return parser


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determine repo root (script is in tools/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Load schema
    schema_path = repo_root / "schemas" / "telemetry_schema_v1.json"
    schema = load_schema(schema_path)
    if schema is None:
        return 3
    
    # Determine input path
    if args.path is not None:
        input_path = Path(args.path)
    else:
        # Default to current directory
        input_path = Path.cwd()
    
    # If path is a directory, look for telemetry files
    if input_path.is_dir():
        telemetry_file = find_telemetry_file(input_path)
        if telemetry_file is None:
            # No telemetry files found - this is OK (nothing to validate)
            print(f"OK: No telemetry files found to validate in {input_path}")
            print(f"  (Looked for: {', '.join(TELEMETRY_FILENAMES)})")
            return 0
    
    # Validate the path
    return validate_telemetry_path(input_path, schema)


if __name__ == "__main__":
    raise SystemExit(main())
