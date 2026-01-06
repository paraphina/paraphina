"""
Unit tests for the Telemetry Contract Gate.

Tests the telemetry schema validation functionality using temp files
and subprocess execution to ensure the validator tool works correctly
as a standalone script.
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Import from tools package for unit testing internal functions
from tools.check_telemetry_contract import (
    ValidationError,
    check_type,
    is_finite_number,
    load_schema,
    validate_file,
    validate_record,
)


class TestIsFiniteNumber(unittest.TestCase):
    """Test the is_finite_number function."""
    
    def test_finite_integers(self):
        """Finite integers should return True."""
        self.assertTrue(is_finite_number(0))
        self.assertTrue(is_finite_number(1))
        self.assertTrue(is_finite_number(-100))
        self.assertTrue(is_finite_number(999999999))
    
    def test_finite_floats(self):
        """Finite floats should return True."""
        self.assertTrue(is_finite_number(0.0))
        self.assertTrue(is_finite_number(1.5))
        self.assertTrue(is_finite_number(-0.001))
        self.assertTrue(is_finite_number(1e10))
    
    def test_nan_returns_false(self):
        """NaN should return False."""
        self.assertFalse(is_finite_number(float('nan')))
    
    def test_inf_returns_false(self):
        """Infinity should return False."""
        self.assertFalse(is_finite_number(float('inf')))
        self.assertFalse(is_finite_number(float('-inf')))
    
    def test_non_numbers_return_false(self):
        """Non-numeric types should return False."""
        self.assertFalse(is_finite_number("123"))
        self.assertFalse(is_finite_number(None))
        self.assertFalse(is_finite_number([1, 2, 3]))
        self.assertFalse(is_finite_number({"a": 1}))
    
    def test_bool_returns_false(self):
        """Booleans should return False (not treated as numbers)."""
        self.assertFalse(is_finite_number(True))
        self.assertFalse(is_finite_number(False))


class TestCheckType(unittest.TestCase):
    """Test the check_type function."""
    
    def test_integer_type_valid(self):
        """Valid integers should pass."""
        self.assertIsNone(check_type(0, "integer", "test"))
        self.assertIsNone(check_type(42, "integer", "test"))
        self.assertIsNone(check_type(-100, "integer", "test"))
    
    def test_integer_type_float_with_no_fraction(self):
        """Float with no fractional part should pass as integer."""
        self.assertIsNone(check_type(42.0, "integer", "test"))
        self.assertIsNone(check_type(0.0, "integer", "test"))
    
    def test_integer_type_invalid(self):
        """Invalid integer values should fail."""
        self.assertIsNotNone(check_type(1.5, "integer", "test"))
        self.assertIsNotNone(check_type("42", "integer", "test"))
        self.assertIsNotNone(check_type(True, "integer", "test"))
        self.assertIsNotNone(check_type(None, "integer", "test"))
    
    def test_number_type_valid(self):
        """Valid numbers should pass."""
        self.assertIsNone(check_type(0, "number", "test"))
        self.assertIsNone(check_type(1.5, "number", "test"))
        self.assertIsNone(check_type(-100.5, "number", "test"))
    
    def test_number_type_nan_fails(self):
        """NaN should fail number type check."""
        error = check_type(float('nan'), "number", "test")
        self.assertIsNotNone(error)
        self.assertIn("not finite", error)
    
    def test_number_type_inf_fails(self):
        """Infinity should fail number type check."""
        error = check_type(float('inf'), "number", "test")
        self.assertIsNotNone(error)
        self.assertIn("not finite", error)
    
    def test_number_type_bool_fails(self):
        """Boolean should fail number type check."""
        self.assertIsNotNone(check_type(True, "number", "test"))
        self.assertIsNotNone(check_type(False, "number", "test"))
    
    def test_string_type_valid(self):
        """Valid strings should pass."""
        self.assertIsNone(check_type("", "string", "test"))
        self.assertIsNone(check_type("hello", "string", "test"))
        self.assertIsNone(check_type("Normal", "string", "test"))
    
    def test_string_type_invalid(self):
        """Non-strings should fail string type check."""
        self.assertIsNotNone(check_type(123, "string", "test"))
        self.assertIsNotNone(check_type(None, "string", "test"))
    
    def test_boolean_type_valid(self):
        """Valid booleans should pass."""
        self.assertIsNone(check_type(True, "boolean", "test"))
        self.assertIsNone(check_type(False, "boolean", "test"))
    
    def test_boolean_type_invalid(self):
        """Non-booleans should fail boolean type check."""
        self.assertIsNotNone(check_type(0, "boolean", "test"))
        self.assertIsNotNone(check_type(1, "boolean", "test"))
        self.assertIsNotNone(check_type("true", "boolean", "test"))
    
    def test_null_type_valid(self):
        """None should pass null type check."""
        self.assertIsNone(check_type(None, "null", "test"))
    
    def test_null_type_invalid(self):
        """Non-null values should fail null type check."""
        self.assertIsNotNone(check_type(0, "null", "test"))
        self.assertIsNotNone(check_type("", "null", "test"))
    
    def test_union_type_number_or_null(self):
        """Union types should accept any valid type."""
        # Should accept number
        self.assertIsNone(check_type(1.5, ["number", "null"], "test"))
        # Should accept null
        self.assertIsNone(check_type(None, ["number", "null"], "test"))
        # Should reject string
        self.assertIsNotNone(check_type("hello", ["number", "null"], "test"))
    
    def test_array_of_integer_valid(self):
        """Valid integer arrays should pass."""
        self.assertIsNone(check_type([], "array_of_integer", "test"))
        self.assertIsNone(check_type([1, 2, 3], "array_of_integer", "test"))
        self.assertIsNone(check_type([0], "array_of_integer", "test"))
    
    def test_array_of_integer_invalid(self):
        """Invalid integer arrays should fail."""
        # Not an array
        self.assertIsNotNone(check_type("not array", "array_of_integer", "test"))
        # Contains non-integer
        self.assertIsNotNone(check_type([1, "two", 3], "array_of_integer", "test"))
        # Contains boolean
        self.assertIsNotNone(check_type([1, True, 3], "array_of_integer", "test"))


class TestLoadSchema(unittest.TestCase):
    """Test the load_schema function."""
    
    def test_load_valid_schema(self):
        """Should load a valid schema file."""
        # Use the real schema file
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        
        schema = load_schema(schema_path)
        
        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("schema_version", schema["required_fields"])
    
    def test_load_missing_file_returns_none(self):
        """Should return None for missing file."""
        schema_path = Path("/nonexistent/path/schema.json")
        
        # Capture stderr
        stderr_capture = io.StringIO()
        with contextlib.redirect_stderr(stderr_capture):
            schema = load_schema(schema_path)
        
        self.assertIsNone(schema)
    
    def test_load_invalid_json_returns_none(self):
        """Should return None for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_path = Path(f.name)
        
        try:
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                schema = load_schema(temp_path)
            self.assertIsNone(schema)
        finally:
            temp_path.unlink()


class TestValidateRecord(unittest.TestCase):
    """Test the validate_record function."""
    
    def setUp(self):
        """Load the schema for tests."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        self.schema = load_schema(schema_path)
        self.assertIsNotNone(self.schema, "Failed to load schema for tests")
    
    def _make_valid_record(self, **overrides) -> dict:
        """Create a valid record with optional overrides."""
        record = {
            "schema_version": 1,
            "t": 0,
            "pnl_realised": 0.0,
            "pnl_unrealised": 0.0,
            "pnl_total": 0.0,
            "risk_regime": "Normal",
            "kill_switch": False,
            "kill_reason": "None",
            "q_global_tao": 0.0,
            "dollar_delta_usd": 0.0,
            "basis_usd": 0.0,
        }
        record.update(overrides)
        return record
    
    def test_valid_record_passes(self):
        """A valid record should produce no errors."""
        record = self._make_valid_record()
        errors, tick = validate_record(record, self.schema, 1, None)
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(tick, 0)
    
    def test_missing_required_field_fails(self):
        """Missing required field should produce error."""
        record = self._make_valid_record()
        del record["schema_version"]
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("schema_version" in e.message for e in errors))
    
    def test_wrong_schema_version_fails(self):
        """Wrong schema_version should produce error."""
        record = self._make_valid_record(schema_version=999)
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("schema_version mismatch" in e.message for e in errors))
    
    def test_invalid_risk_regime_fails(self):
        """Invalid risk_regime enum value should produce error."""
        record = self._make_valid_record(risk_regime="InvalidRegime")
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("risk_regime" in e.message and "invalid value" in e.message for e in errors))
    
    def test_all_risk_regimes_valid(self):
        """All valid risk_regime values should pass."""
        for regime in ["Normal", "Warning", "HardLimit"]:
            record = self._make_valid_record(risk_regime=regime)
            errors, _ = validate_record(record, self.schema, 1, None)
            self.assertEqual(len(errors), 0, f"regime '{regime}' should be valid")
    
    def test_tick_monotonicity_enforced(self):
        """Non-monotonic tick should produce error."""
        record = self._make_valid_record(t=5)
        
        # With prev_tick=10, t=5 should fail
        errors, tick = validate_record(record, self.schema, 1, prev_tick=10)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("monotonic" in e.message for e in errors))
    
    def test_tick_monotonicity_ok_when_increasing(self):
        """Increasing tick should pass monotonicity check."""
        record = self._make_valid_record(t=11)
        
        errors, tick = validate_record(record, self.schema, 1, prev_tick=10)
        
        # Filter out monotonicity errors specifically
        mono_errors = [e for e in errors if "monotonic" in e.message]
        self.assertEqual(len(mono_errors), 0)
        self.assertEqual(tick, 11)
    
    def test_optional_fields_validated_when_present(self):
        """Optional fields should be type-checked when present."""
        record = self._make_valid_record(
            fv_available=True,
            fair_value=250.5,
            sigma_eff=0.02,
            healthy_venues_used_count=3,
            healthy_venues_used=[0, 1, 2],
        )
        
        errors, _ = validate_record(record, self.schema, 1, None)
        self.assertEqual(len(errors), 0)
    
    def test_optional_field_wrong_type_fails(self):
        """Optional field with wrong type should produce error."""
        record = self._make_valid_record(fv_available="yes")  # Should be boolean
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("fv_available" in e.message for e in errors))
    
    def test_fair_value_null_valid(self):
        """fair_value can be null."""
        record = self._make_valid_record(fair_value=None)
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        # Filter errors for fair_value specifically
        fv_errors = [e for e in errors if "fair_value" in e.message]
        self.assertEqual(len(fv_errors), 0)
    
    def test_nan_in_numeric_field_fails(self):
        """NaN value in numeric field should fail."""
        record = self._make_valid_record(pnl_total=float('nan'))
        
        errors, _ = validate_record(record, self.schema, 1, None)
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("not finite" in e.message or "NaN" in e.message for e in errors))


class TestValidateFile(unittest.TestCase):
    """Test the validate_file function."""
    
    def setUp(self):
        """Load the schema for tests."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        self.schema = load_schema(schema_path)
        self.assertIsNotNone(self.schema, "Failed to load schema for tests")
    
    def _make_valid_record(self, tick: int = 0, **overrides) -> dict:
        """Create a valid record with optional overrides."""
        record = {
            "schema_version": 1,
            "t": tick,
            "pnl_realised": 0.0,
            "pnl_unrealised": 0.0,
            "pnl_total": 0.0,
            "risk_regime": "Normal",
            "kill_switch": False,
            "kill_reason": "None",
            "q_global_tao": 0.0,
            "dollar_delta_usd": 0.0,
            "basis_usd": 0.0,
        }
        record.update(overrides)
        return record
    
    def test_valid_file_passes(self):
        """A valid JSONL file should produce no errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(5):
                record = self._make_valid_record(tick=i)
                f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)
        
        try:
            errors = validate_file(temp_path, self.schema)
            self.assertEqual(len(errors), 0)
        finally:
            temp_path.unlink()
    
    def test_invalid_json_line_fails(self):
        """Invalid JSON line should produce error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            record = self._make_valid_record(tick=0)
            f.write(json.dumps(record) + "\n")
            f.write("not valid json {{\n")
            temp_path = Path(f.name)
        
        try:
            errors = validate_file(temp_path, self.schema)
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("invalid JSON" in e.message for e in errors))
        finally:
            temp_path.unlink()
    
    def test_missing_required_field_in_file_fails(self):
        """File with missing required field should produce error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            record = self._make_valid_record(tick=0)
            del record["kill_switch"]  # Remove required field
            f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)
        
        try:
            errors = validate_file(temp_path, self.schema)
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("kill_switch" in e.message for e in errors))
        finally:
            temp_path.unlink()
    
    def test_empty_lines_skipped(self):
        """Empty lines in JSONL should be skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            record0 = self._make_valid_record(tick=0)
            record1 = self._make_valid_record(tick=1)
            f.write(json.dumps(record0) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace-only line
            f.write(json.dumps(record1) + "\n")
            temp_path = Path(f.name)
        
        try:
            errors = validate_file(temp_path, self.schema)
            self.assertEqual(len(errors), 0)
        finally:
            temp_path.unlink()


class TestValidatorSubprocess(unittest.TestCase):
    """Test the validator as a subprocess (integration tests)."""
    
    def _get_validator_path(self) -> Path:
        """Get path to the validator script."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "check_telemetry_contract.py"
    
    def _make_valid_record(self, tick: int = 0, **overrides) -> dict:
        """Create a valid record with optional overrides."""
        record = {
            "schema_version": 1,
            "t": tick,
            "pnl_realised": 0.0,
            "pnl_unrealised": 0.0,
            "pnl_total": 0.0,
            "risk_regime": "Normal",
            "kill_switch": False,
            "kill_reason": "None",
            "q_global_tao": 0.0,
            "dollar_delta_usd": 0.0,
            "basis_usd": 0.0,
        }
        record.update(overrides)
        return record
    
    def test_valid_file_exit_0(self):
        """Valid file should exit with code 0."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(3):
                record = self._make_valid_record(tick=i)
                f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)
        
        try:
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(temp_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("OK", result.stdout)
        finally:
            temp_path.unlink()
    
    def test_invalid_file_exit_1(self):
        """File with contract violation should exit with code 1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            record = self._make_valid_record(tick=0)
            del record["schema_version"]  # Remove required field
            f.write(json.dumps(record) + "\n")
            temp_path = Path(f.name)
        
        try:
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(temp_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("FAILED", result.stdout)
        finally:
            temp_path.unlink()
    
    def test_missing_file_exit_2(self):
        """Missing file should exit with code 2."""
        result = subprocess.run(
            [sys.executable, str(self._get_validator_path()), "/nonexistent/file.jsonl"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2, f"stdout: {result.stdout}\nstderr: {result.stderr}")
    
    def test_help_flag_exit_0(self):
        """--help should print help and exit with code 0."""
        result = subprocess.run(
            [sys.executable, str(self._get_validator_path()), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("telemetry", result.stdout.lower())
    
    def test_no_args_exit_0(self):
        """No arguments in empty directory should exit with code 0 (no files to validate)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path())],
                capture_output=True,
                text=True,
                cwd=tmpdir,  # Run in empty temp directory
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("OK", result.stdout)
            self.assertIn("No telemetry files found", result.stdout)
    
    def test_no_args_with_valid_file_exit_0(self):
        """No arguments with valid telemetry file in cwd should exit 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid telemetry file in the temp directory
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            record = self._make_valid_record(tick=0)
            with open(telemetry_path, "w") as f:
                f.write(json.dumps(record) + "\n")
            
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path())],
                capture_output=True,
                text=True,
                cwd=tmpdir,  # Run in directory with telemetry file
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("OK", result.stdout)
            self.assertIn("1 record(s) validated", result.stdout)


class TestSchemaGeneratedTelemetry(unittest.TestCase):
    """
    Test that telemetry generated from schema definition validates successfully.
    
    This ensures the schema file itself is self-consistent.
    """
    
    def test_schema_driven_record_validates(self):
        """A record generated from schema required fields should validate."""
        # Load schema
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        schema = load_schema(schema_path)
        self.assertIsNotNone(schema)
        
        # Build a minimal valid record from schema
        required_fields = schema["required_fields"]
        field_types = schema["field_types"]
        
        record = {}
        for field in required_fields:
            ftype = field_types.get(field, "string")
            if ftype == "integer":
                record[field] = 0
            elif ftype == "number":
                record[field] = 0.0
            elif ftype == "string":
                # Use valid enum value if applicable
                enums = schema.get("enums", {})
                if field in enums:
                    record[field] = enums[field][0]
                else:
                    record[field] = ""
            elif ftype == "boolean":
                record[field] = False
            elif isinstance(ftype, list):
                # Union type - use first non-null type
                for t in ftype:
                    if t != "null":
                        if t == "number":
                            record[field] = 0.0
                        elif t == "integer":
                            record[field] = 0
                        break
        
        # Set required values
        record["schema_version"] = 1
        record["risk_regime"] = "Normal"
        record["kill_reason"] = "None"
        
        # Validate
        errors, _ = validate_record(record, schema, 1, None)
        self.assertEqual(len(errors), 0, f"Schema-driven record should validate. Errors: {errors}")
    
    def test_schema_file_has_all_required_keys(self):
        """Schema file should have all expected top-level keys."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        schema = load_schema(schema_path)
        
        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("optional_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("enums", schema)
        self.assertIn("invariants", schema)
        self.assertIn("schema_version", schema)
        self.assertEqual(schema["schema_version"], 1)


if __name__ == '__main__':
    unittest.main()

