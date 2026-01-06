#!/usr/bin/env python3
"""
Telemetry Contract Gate: Validates telemetry JSONL files against their schemas.

This tool enforces telemetry schema contracts defined in:
- docs/TELEMETRY_SCHEMA_V1.md (human-readable documentation)
- schemas/telemetry_schema_v1.json (per-tick telemetry)
- schemas/mc_runs_schema_v1.json (Monte Carlo per-run records)

File-to-schema mapping:
  telemetry.jsonl    → telemetry_schema_v1.json
  mc_runs.jsonl      → mc_runs_schema_v1.json
  metrics.jsonl      → (unmapped - fails loudly)
  trajectories.jsonl → (unmapped - fails loudly)

Exit codes:
    0 - OK: all records valid
    1 - Contract violation (schema error, missing required fields, type mismatch, etc.)
    2 - File not found, CLI usage error, or unmapped file type

Usage:
    python3 tools/check_telemetry_contract.py [PATH]
    python3 tools/check_telemetry_contract.py --help
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple


# Maximum number of errors to display before truncating
MAX_ERRORS_DISPLAY = 10

# File-to-schema mapping: filename -> schema filename
# Files not in this mapping will fail with a clear error
FILE_SCHEMA_MAP: dict[str, str] = {
    "telemetry.jsonl": "telemetry_schema_v1.json",
    "mc_runs.jsonl": "mc_runs_schema_v1.json",
}

# Files that are recognized but have no schema yet
UNMAPPED_FILES: set[str] = {
    "metrics.jsonl",
    "trajectories.jsonl",
}

# All known telemetry filenames for directory scanning
TELEMETRY_FILENAMES = list(FILE_SCHEMA_MAP.keys()) + list(UNMAPPED_FILES)


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
    prev_index: int | None,
    index_field: str | None,
) -> tuple[list[ValidationError], int | None]:
    """
    Validate a single record against the schema.
    
    Args:
        record: The JSON record to validate
        schema: The schema to validate against
        line_num: Line number for error messages
        prev_index: Previous index value (for monotonicity check)
        index_field: Field name to check for monotonicity (e.g., "t" or "run_index")
    
    Returns (list of errors, current index value or None).
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
    
    # Check index monotonicity if applicable
    current_index = None
    if index_field and index_field in record:
        idx_val = record[index_field]
        if isinstance(idx_val, (int, float)) and not isinstance(idx_val, bool):
            current_index = int(idx_val)
            if prev_index is not None and current_index <= prev_index:
                errors.append(ValidationError(
                    line_num,
                    f"{index_field} not monotonically increasing: {index_field}={current_index} (prev={prev_index})"
                ))
    
    return errors, current_index


def validate_file(
    file_path: Path,
    schema: dict[str, Any],
    index_field: str | None = None,
) -> list[ValidationError]:
    """
    Validate all records in a JSONL file against the schema.
    
    Args:
        file_path: Path to the JSONL file
        schema: Schema to validate against
        index_field: Field to check for monotonicity (e.g., "t" for telemetry, "run_index" for mc_runs)
    
    Returns list of all validation errors.
    """
    errors: list[ValidationError] = []
    prev_index: int | None = None
    
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
                record_errors, current_index = validate_record(
                    record, schema, line_num, prev_index, index_field
                )
                errors.extend(record_errors)
                
                if current_index is not None:
                    prev_index = current_index
    
    except OSError as e:
        errors.append(ValidationError(0, f"file read error: {e}"))
    
    return errors


def get_schema_for_file(filename: str, repo_root: Path) -> tuple[dict[str, Any] | None, str | None, int]:
    """
    Get the appropriate schema for a given filename.
    
    Returns (schema, index_field, exit_code).
    - exit_code 0 means success
    - exit_code 2 means unmapped file (fail loudly)
    """
    basename = Path(filename).name
    
    if basename in FILE_SCHEMA_MAP:
        schema_file = FILE_SCHEMA_MAP[basename]
        schema_path = repo_root / "schemas" / schema_file
        schema = load_schema(schema_path)
        if schema is None:
            return None, None, 2
        
        # Determine index field for monotonicity check
        if basename == "telemetry.jsonl":
            index_field = "t"
        elif basename == "mc_runs.jsonl":
            index_field = "run_index"
        else:
            index_field = None
        
        return schema, index_field, 0
    
    if basename in UNMAPPED_FILES:
        print(f"ERROR: No schema defined for '{basename}'", file=sys.stderr)
        print(f"  To add schema support:", file=sys.stderr)
        print(f"    1. Create schemas/{basename.replace('.jsonl', '_schema_v1.json')}", file=sys.stderr)
        print(f"    2. Add mapping to FILE_SCHEMA_MAP in tools/check_telemetry_contract.py", file=sys.stderr)
        print(f"  See docs/TELEMETRY_SCHEMA_V1.md for schema format.", file=sys.stderr)
        return None, None, 2
    
    # Unknown file - treat as potential telemetry, try with telemetry schema
    # This allows validating arbitrary .jsonl files against the default schema
    schema_path = repo_root / "schemas" / "telemetry_schema_v1.json"
    schema = load_schema(schema_path)
    if schema is None:
        return None, None, 2
    return schema, "t", 0


def validate_single_file(file_path: Path, repo_root: Path) -> int:
    """
    Validate a single file against its appropriate schema.
    
    Returns exit code: 0=OK, 1=contract violation, 2=error.
    """
    schema, index_field, exit_code = get_schema_for_file(file_path.name, repo_root)
    if exit_code != 0:
        return exit_code
    
    if schema is None:
        return 2
    
    errors = validate_file(file_path, schema, index_field)
    
    if errors:
        # Count records (for summary)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for line in f if line.strip())
        except OSError:
            total_lines = "?"
        
        print(f"FAILED: {len(errors)} error(s) in {file_path}")
        print(f"  (Total records scanned: {total_lines})")
        print()
        
        # Show first N errors
        for err in errors[:MAX_ERRORS_DISPLAY]:
            if err.line > 0:
                print(f"  Line {err.line}: {err.message}")
            else:
                print(f"  {err.message}")
        
        if len(errors) > MAX_ERRORS_DISPLAY:
            print(f"  ... and {len(errors) - MAX_ERRORS_DISPLAY} more error(s)")
        
        return 1
    
    # Success
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())
    except OSError:
        total_lines = "?"
    
    print(f"OK: {total_lines} record(s) validated against schema v{schema.get('schema_version', '?')}")
    print(f"  File: {file_path}")
    return 0


def find_and_validate_directory(dir_path: Path, repo_root: Path) -> int:
    """
    Find and validate all telemetry files in a directory.
    
    Returns exit code: 0=all OK (or no files), 1=any contract violation, 2=any error.
    """
    validated_count = 0
    worst_exit_code = 0
    
    for filename in TELEMETRY_FILENAMES:
        file_path = dir_path / filename
        if file_path.exists():
            print(f"--- Validating {filename} ---")
            exit_code = validate_single_file(file_path, repo_root)
            validated_count += 1
            
            # Track worst exit code (1 > 2 > 0 for prioritizing contract violations)
            if exit_code == 1:
                worst_exit_code = 1
            elif exit_code == 2 and worst_exit_code == 0:
                worst_exit_code = 2
            
            print()
    
    if validated_count == 0:
        print(f"OK: No telemetry files found to validate in {dir_path}")
        print(f"  (Looked for: {', '.join(TELEMETRY_FILENAMES)})")
        return 0
    
    if worst_exit_code == 0:
        print(f"=== All {validated_count} file(s) validated successfully ===")
    
    return worst_exit_code


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="check_telemetry_contract",
        description="""
Telemetry Contract Gate: Validates telemetry JSONL files against their schemas.

Supports multiple file types with per-file schema mapping:
  telemetry.jsonl    → schemas/telemetry_schema_v1.json (per-tick data)
  mc_runs.jsonl      → schemas/mc_runs_schema_v1.json (Monte Carlo runs)

Unmapped files (metrics.jsonl, trajectories.jsonl) will fail with instructions
on how to add schema support.
""",
        epilog="""
Exit codes:
  0 - OK: all records valid (or no telemetry files found)
  1 - Contract violation (schema error, missing required fields, etc.)
  2 - File not found, CLI usage error, or unmapped file type

Examples:
  %(prog)s                          # Check current directory
  %(prog)s path/to/file.jsonl       # Check specific JSONL file
  %(prog)s path/to/run_directory/   # Check all telemetry files in directory

Schema documentation: docs/TELEMETRY_SCHEMA_V1.md
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to JSONL file or directory. "
             "If omitted, checks the current directory.",
    )
    
    return parser


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determine repo root (script is in tools/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Determine input path
    if args.path is not None:
        input_path = Path(args.path)
    else:
        input_path = Path.cwd()
    
    # Validate based on path type
    if input_path.is_file():
        return validate_single_file(input_path, repo_root)
    elif input_path.is_dir():
        return find_and_validate_directory(input_path, repo_root)
    else:
        print(f"ERROR: Path not found: {input_path}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
