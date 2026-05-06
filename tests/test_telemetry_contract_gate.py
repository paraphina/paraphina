"""
Unit tests for the Telemetry Contract Gate.

Tests the telemetry schema validation functionality using temp files
and subprocess execution to ensure the validator tool works correctly
as a standalone script.

Supports multiple schemas:
- telemetry.jsonl → telemetry_schema_v1.json
- mc_runs.jsonl → mc_runs_schema_v1.json
"""

import contextlib
import hashlib
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
from tools.phase51b_lighter_account_limits import _sanitize_response_headers


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

    def test_array_of_string_valid(self):
        """Valid string arrays should pass."""
        self.assertIsNone(check_type([], "array_of_string", "test"))
        self.assertIsNone(check_type(["a", "b"], "array_of_string", "test"))

    def test_array_of_string_invalid(self):
        """Invalid string arrays should fail."""
        self.assertIsNotNone(check_type("not array", "array_of_string", "test"))
        self.assertIsNotNone(check_type(["a", 2], "array_of_string", "test"))
        self.assertIsNotNone(check_type(["a", None], "array_of_string", "test"))

    def test_object_type(self):
        """JSON objects should pass object type checks."""
        self.assertIsNone(check_type({}, "object", "test"))
        self.assertIsNone(check_type({"MAKER": 1}, "object", "test"))
        self.assertIsNotNone(check_type([], "object", "test"))
        self.assertIsNotNone(check_type("not object", "object", "test"))


class TestLoadSchema(unittest.TestCase):
    """Test the load_schema function."""
    
    def test_load_valid_telemetry_schema(self):
        """Should load the telemetry schema file."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v1.json"
        
        schema = load_schema(schema_path)
        
        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("schema_version", schema["required_fields"])
    
    def test_load_valid_mc_runs_schema(self):
        """Should load the mc_runs schema file."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "mc_runs_schema_v1.json"
        
        schema = load_schema(schema_path)
        
        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("schema_version", schema["required_fields"])
        self.assertIn("run_index", schema["required_fields"])

    def test_load_valid_telemetry_v2_schema(self):
        """Should load the telemetry v2 schema file."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "telemetry_schema_v2.json"

        schema = load_schema(schema_path)

        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("schema_version", schema["required_fields"])
        self.assertIn("event_type", schema["required_fields"])
    
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


class TestValidateTelemetryRecord(unittest.TestCase):
    """Test the validate_record function for telemetry.jsonl schema."""
    
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
        errors, tick = validate_record(record, self.schema, 1, None, "t")
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(tick, 0)
    
    def test_missing_required_field_fails(self):
        """Missing required field should produce error."""
        record = self._make_valid_record()
        del record["schema_version"]
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("schema_version" in e.message for e in errors))
    
    def test_wrong_schema_version_fails(self):
        """Wrong schema_version should produce error."""
        record = self._make_valid_record(schema_version=999)
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("schema_version mismatch" in e.message for e in errors))
    
    def test_invalid_risk_regime_fails(self):
        """Invalid risk_regime enum value should produce error."""
        record = self._make_valid_record(risk_regime="InvalidRegime")
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("risk_regime" in e.message and "invalid value" in e.message for e in errors))
    
    def test_all_risk_regimes_valid(self):
        """All valid risk_regime values should pass."""
        for regime in ["Normal", "Warning", "HardLimit"]:
            record = self._make_valid_record(risk_regime=regime)
            errors, _ = validate_record(record, self.schema, 1, None, "t")
            self.assertEqual(len(errors), 0, f"regime '{regime}' should be valid")
    
    def test_tick_monotonicity_enforced(self):
        """Non-monotonic tick should produce error."""
        record = self._make_valid_record(t=5)
        
        # With prev_tick=10, t=5 should fail
        errors, tick = validate_record(record, self.schema, 1, prev_index=10, index_field="t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("monotonic" in e.message for e in errors))
    
    def test_tick_monotonicity_ok_when_increasing(self):
        """Increasing tick should pass monotonicity check."""
        record = self._make_valid_record(t=11)
        
        errors, tick = validate_record(record, self.schema, 1, prev_index=10, index_field="t")
        
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
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        self.assertEqual(len(errors), 0)
    
    def test_optional_field_wrong_type_fails(self):
        """Optional field with wrong type should produce error."""
        record = self._make_valid_record(fv_available="yes")  # Should be boolean
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("fv_available" in e.message for e in errors))
    
    def test_fair_value_null_valid(self):
        """fair_value can be null."""
        record = self._make_valid_record(fair_value=None)
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        # Filter errors for fair_value specifically
        fv_errors = [e for e in errors if "fair_value" in e.message]
        self.assertEqual(len(fv_errors), 0)
    
    def test_nan_in_numeric_field_fails(self):
        """NaN value in numeric field should fail."""
        record = self._make_valid_record(pnl_total=float('nan'))
        
        errors, _ = validate_record(record, self.schema, 1, None, "t")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("not finite" in e.message or "NaN" in e.message for e in errors))


class TestValidateMcRunsRecord(unittest.TestCase):
    """Test the validate_record function for mc_runs.jsonl schema."""
    
    def setUp(self):
        """Load the mc_runs schema for tests."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "mc_runs_schema_v1.json"
        self.schema = load_schema(schema_path)
        self.assertIsNotNone(self.schema, "Failed to load mc_runs schema for tests")
    
    def _make_valid_mc_run_record(self, **overrides) -> dict:
        """Create a valid mc_runs record with optional overrides."""
        record = {
            "schema_version": 1,
            "run_index": 0,
            "seed": 12345,
            "pnl_total": -50.0,
            "max_drawdown": 100.0,
            "kill_switch": False,
            "kill_tick": None,
            "kill_reason": "None",
            "ticks_executed": 100,
            "max_abs_delta_usd": 1000.0,
            "max_abs_basis_usd": 500.0,
            "max_abs_q_tao": 10.0,
            "max_venue_toxicity": 0.5,
        }
        record.update(overrides)
        return record
    
    def test_valid_mc_run_record_passes(self):
        """A valid mc_runs record should produce no errors."""
        record = self._make_valid_mc_run_record()
        errors, idx = validate_record(record, self.schema, 1, None, "run_index")
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(idx, 0)
    
    def test_missing_required_field_fails(self):
        """Missing required field in mc_runs should produce error."""
        record = self._make_valid_mc_run_record()
        del record["schema_version"]
        
        errors, _ = validate_record(record, self.schema, 1, None, "run_index")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("schema_version" in e.message for e in errors))
    
    def test_missing_run_index_fails(self):
        """Missing run_index should produce error."""
        record = self._make_valid_mc_run_record()
        del record["run_index"]
        
        errors, _ = validate_record(record, self.schema, 1, None, "run_index")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("run_index" in e.message for e in errors))
    
    def test_run_index_monotonicity(self):
        """run_index should be monotonically increasing."""
        record = self._make_valid_mc_run_record(run_index=5)
        
        # With prev_index=10, run_index=5 should fail
        errors, idx = validate_record(record, self.schema, 1, prev_index=10, index_field="run_index")
        
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("monotonic" in e.message for e in errors))
    
    def test_kill_tick_can_be_null(self):
        """kill_tick can be null."""
        record = self._make_valid_mc_run_record(kill_tick=None)
        
        errors, _ = validate_record(record, self.schema, 1, None, "run_index")
        
        kill_tick_errors = [e for e in errors if "kill_tick" in e.message]
        self.assertEqual(len(kill_tick_errors), 0)
    
    def test_kill_tick_can_be_integer(self):
        """kill_tick can be an integer."""
        record = self._make_valid_mc_run_record(kill_tick=50)
        
        errors, _ = validate_record(record, self.schema, 1, None, "run_index")
        
        kill_tick_errors = [e for e in errors if "kill_tick" in e.message]
        self.assertEqual(len(kill_tick_errors), 0)
    
    def test_extra_fields_allowed(self):
        """Extra fields should be allowed (forward compatibility)."""
        record = self._make_valid_mc_run_record(
            extra_field="some_value",
            another_extra=123,
        )
        
        errors, _ = validate_record(record, self.schema, 1, None, "run_index")
        
        self.assertEqual(len(errors), 0, "Extra fields should not cause errors")


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
            errors = validate_file(temp_path, self.schema, index_field="t")
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
            errors = validate_file(temp_path, self.schema, index_field="t")
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
            errors = validate_file(temp_path, self.schema, index_field="t")
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
            errors = validate_file(temp_path, self.schema, index_field="t")
            self.assertEqual(len(errors), 0)
        finally:
            temp_path.unlink()


class TestValidatorSubprocess(unittest.TestCase):
    """Test the validator as a subprocess (integration tests)."""

    def test_phase51b_lighter_response_header_sanitizer_keeps_only_limit_headers(self):
        """Lighter read-only probes should persist only non-secret quota headers."""
        sanitized = _sanitize_response_headers({
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "58",
            "Retry-After": "1",
            "Set-Cookie": "must-not-appear",
            "Authorization": "must-not-appear",
            "X-Request-Id": "not-a-limit-field",
        })
        self.assertEqual(sanitized["X-RateLimit-Limit"], "60")
        self.assertEqual(sanitized["X-RateLimit-Remaining"], "58")
        self.assertEqual(sanitized["Retry-After"], "1")
        self.assertNotIn("Set-Cookie", sanitized)
        self.assertNotIn("Authorization", sanitized)
        self.assertNotIn("X-Request-Id", sanitized)
    
    def _get_validator_path(self) -> Path:
        """Get path to the validator script."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "check_telemetry_contract.py"

    def _get_phase51_shadow_path(self) -> Path:
        """Get path to the Phase 5.1 shadow harness."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51_ev_shadow.py"

    def _get_phase51b_lighter_account_limits_path(self) -> Path:
        """Get path to the Phase 5.1b Lighter account/native-limit collector."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51b_lighter_account_limits.py"

    def _get_phase51b_acceptance_path(self) -> Path:
        """Get path to the Phase 5.1b evidence acceptance gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51b_accept_evidence.py"

    def _get_phase51c_label_lake_path(self) -> Path:
        """Get path to the Phase 5.1c label-lake scaffold."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_label_lake.py"

    def _get_phase51c_observed_labels_path(self) -> Path:
        """Get path to the Phase 5.1c observed-label extractor."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_observed_labels.py"

    def _get_phase51c_join_holdout_path(self) -> Path:
        """Get path to the Phase 5.1c deterministic join/holdout gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_join_holdout.py"

    def _get_phase51c_lighter_trade_backfill_path(self) -> Path:
        """Get path to the Phase 5.1c Lighter native trade backfill collector."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_lighter_trade_backfill.py"

    def _get_phase51c_pfill_outcome_path(self) -> Path:
        """Get path to the Phase 5.1c order-level P_fill outcome gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_pfill_outcome_labels.py"

    def _get_phase51c_pfill_calibration_readiness_path(self) -> Path:
        """Get path to the Phase 5.1c P_fill calibration-readiness gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_pfill_calibration_readiness.py"

    def _get_phase51c_pfill_censoring_audit_path(self) -> Path:
        """Get path to the Phase 5.1c P_fill censoring audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_pfill_censoring_audit.py"

    def _get_phase51c_queue_churn_path(self) -> Path:
        """Get path to the Phase 5.1c queue/churn label gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_queue_churn_labels.py"

    def _get_phase51c_markout_calibration_readiness_path(self) -> Path:
        """Get path to the Phase 5.1c markout calibration-readiness gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_markout_calibration_readiness.py"

    def _get_phase51c_lighter_attribution_gap_audit_path(self) -> Path:
        """Get path to the Phase 5.1c Lighter attribution-gap audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51c_lighter_attribution_gap_audit.py"

    def _get_phase51e_lifecycle_truth_audit_path(self) -> Path:
        """Get path to the Phase 5.1e lifecycle/native-truth audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51e_lifecycle_truth_audit.py"

    def _get_phase51f_canonical_pfill_outcome_path(self) -> Path:
        """Get path to the Phase 5.1f canonical P_fill outcome review gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51f_canonical_pfill_outcome_rebuild.py"

    def _get_phase51g_pfill_quarantine_review_path(self) -> Path:
        """Get path to the Phase 5.1g P_fill quarantine review gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51g_pfill_quarantine_review.py"

    def _get_phase51h_observed_pfill_feature_audit_path(self) -> Path:
        """Get path to the Phase 5.1h observed P_fill feature audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51h_observed_pfill_feature_audit.py"

    def _get_phase51i_pfill_feature_matrix_admissibility_path(self) -> Path:
        """Get path to the Phase 5.1i P_fill feature-matrix admissibility gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51i_pfill_feature_matrix_admissibility.py"

    def _get_phase51j_observed_horizon_recovery_path(self) -> Path:
        """Get path to the Phase 5.1j observed-horizon recovery gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51j_observed_horizon_recovery.py"

    def _get_phase51k_filled_horizon_timebase_recovery_path(self) -> Path:
        """Get path to the Phase 5.1k filled-horizon timebase recovery gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51k_filled_horizon_timebase_recovery.py"

    def _get_phase51l_filled_horizon_source_key_recovery_path(self) -> Path:
        """Get path to the Phase 5.1l filled-horizon source-key recovery gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51l_filled_horizon_source_key_recovery.py"

    def _get_phase51n_lighter_native_limit_time_alignment_path(self) -> Path:
        """Get path to the Phase 5.1n Lighter native-limit time-alignment gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51n_lighter_native_limit_time_alignment.py"

    def _get_phase51n_maker_taker_attribution_recovery_path(self) -> Path:
        """Get path to the Phase 5.1n maker/taker attribution recovery gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51n_maker_taker_attribution_recovery.py"

    def _get_phase51o_native_role_source_inventory_path(self) -> Path:
        """Get path to the Phase 5.1o native role source inventory gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51o_native_role_source_inventory.py"

    def _get_phase51p_lighter_native_role_canonical_join_path(self) -> Path:
        """Get path to the Phase 5.1p Lighter native role canonical join gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51p_lighter_native_role_canonical_join.py"

    def _get_phase51q_forward_native_evidence_capture_path(self) -> Path:
        """Get path to the Phase 5.1q forward native evidence gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51q_forward_native_evidence_capture.py"

    def _get_phase51r_forward_native_source_acquisition_path(self) -> Path:
        """Get path to the Phase 5.1r forward native source-acquisition gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51r_forward_native_source_acquisition.py"

    def _get_phase51s_local_native_source_acquisition_path(self) -> Path:
        """Get path to the Phase 5.1s local native source-acquisition gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51s_local_native_source_acquisition.py"

    def _get_phase51t_source_link_sidecar_builder_path(self) -> Path:
        """Get path to the Phase 5.1t source-link sidecar builder."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51t_source_link_sidecar_builder.py"

    def _get_phase51u_forward_capture_target_manifest_path(self) -> Path:
        """Get path to the Phase 5.1u forward capture target-manifest gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51u_forward_capture_target_manifest.py"

    def _get_phase51v_forward_capture_bundle_readiness_path(self) -> Path:
        """Get path to the Phase 5.1v forward capture bundle-readiness gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51v_forward_capture_bundle_readiness.py"

    def _get_phase51w_forward_capture_request_pack_path(self) -> Path:
        """Get path to the Phase 5.1w forward capture request-pack gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51w_forward_capture_request_pack.py"

    def _get_phase51x_hyperliquid_native_role_adapter_path(self) -> Path:
        """Get path to the Phase 5.1x Hyperliquid native-role adapter."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51x_hyperliquid_native_role_adapter.py"

    def _get_phase51y_all5_native_role_adapter_path(self) -> Path:
        """Get path to the Phase 5.1y all-venue native-role adapter."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51y_all5_native_role_adapter.py"

    def _get_phase51z_readonly_native_role_capture_path(self) -> Path:
        """Get path to the Phase 5.1z read-only native-role capture tool."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51z_readonly_native_role_capture.py"

    def _get_phase51z_source_link_request_pack_path(self) -> Path:
        """Get path to the Phase 5.1z source-link request-pack tool."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51z_source_link_request_pack.py"

    def _get_phase51aa_lighter_ws_account_trades_snapshot_path(self) -> Path:
        """Get path to the Phase 5.1aa Lighter WS account-trades snapshot tool."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51aa_lighter_ws_account_trades_snapshot.py"

    def _get_phase51ab_lighter_native_limit_pressure_source_path(self) -> Path:
        """Get path to the Phase 5.1ab Lighter native-limit pressure source gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51ab_lighter_native_limit_pressure_source.py"

    def _get_phase51ac_source_link_reuse_audit_path(self) -> Path:
        """Get path to the Phase 5.1ac source-link reuse audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51ac_source_link_reuse_audit.py"

    def _get_phase51ad_source_link_sidecar_materialize_path(self) -> Path:
        """Get path to the Phase 5.1ad source-link sidecar materializer gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51ad_source_link_sidecar_materialize.py"

    def _get_phase51ae_candidate_manifest_compose_path(self) -> Path:
        """Get path to the Phase 5.1ae candidate-manifest composition gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51ae_candidate_manifest_compose.py"

    def _get_phase51af_local_source_retrieval_audit_path(self) -> Path:
        """Get path to the Phase 5.1af local source retrieval audit gate."""
        script_dir = Path(__file__).parent.parent
        return script_dir / "tools" / "phase51af_local_source_retrieval_audit.py"

    def _make_valid_telemetry_record(self, tick: int = 0, **overrides) -> dict:
        """Create a valid telemetry record."""
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
    
    def _make_valid_mc_run_record(self, run_index: int = 0, **overrides) -> dict:
        """Create a valid mc_runs record."""
        record = {
            "schema_version": 1,
            "run_index": run_index,
            "seed": 12345 + run_index,
            "pnl_total": -50.0,
            "max_drawdown": 100.0,
            "kill_switch": False,
            "kill_tick": None,
            "kill_reason": "None",
            "ticks_executed": 100,
            "max_abs_delta_usd": 1000.0,
            "max_abs_basis_usd": 500.0,
            "max_abs_q_tao": 10.0,
            "max_venue_toxicity": 0.5,
        }
        record.update(overrides)
        return record

    def _make_valid_telemetry_v2_record(self, event_seq: int = 0, **overrides) -> dict:
        """Create a valid telemetry v2 record."""
        record = {
            "schema_version": 2,
            "event_type": "V2_RUN_CONTEXT",
            "event_seq": event_seq,
            "timestamp_local_ns": 1_700_000_000_000_000_000 + event_seq,
            "run_id": "phase51_test",
            "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
            "no_live_flag": True,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_capital_escalation": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        }
        record.update(overrides)
        return record
    
    def test_valid_telemetry_file_exit_0(self):
        """Valid telemetry file should exit with code 0."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(3):
                record = self._make_valid_telemetry_record(tick=i)
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

    def test_valid_telemetry_v2_file_exit_0(self):
        """Valid schema_version 2 telemetry file should exit with code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            with open(telemetry_path, "w") as f:
                f.write(json.dumps(self._make_valid_telemetry_v2_record(0)) + "\n")
                f.write(json.dumps(self._make_valid_telemetry_v2_record(
                    1,
                    event_type="V2_EV_EVALUATED",
                    candidate_id="cand-1",
                    decision="HOLD",
                    EV_hat=-0.1,
                    EV_lcb_alpha=-0.2,
                    alpha=0.05,
                )) + "\n")

            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("schema v2", result.stdout)

    def test_valid_telemetry_v2_lighter_account_limit_events_exit_0(self):
        """Phase 5.1b Lighter account/native-limit events should validate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            with open(telemetry_path, "w") as f:
                f.write(json.dumps(self._make_valid_telemetry_v2_record(0)) + "\n")
                f.write(json.dumps(self._make_valid_telemetry_v2_record(
                    1,
                    event_type="V2_LIGHTER_ACCOUNT_LIMITS",
                    venue_id="lighter",
                    account_index=123,
                    account_limits_status="OBSERVED",
                    account_limits_source_sha256="abc123",
                    account_limits_source_endpoint=None,
                    account_limits_raw_keys=["active_orders_per_account_limit"],
                    sendtx_per_minute_limit=60,
                    sendtx_per_minute_remaining=None,
                    rest_requests_per_minute_limit=None,
                    weighted_requests_per_minute_limit=None,
                    pending_orders_per_account_limit=None,
                    pending_orders_per_market_limit=None,
                    active_orders_per_account_limit=1000,
                    active_orders_per_market_limit=100,
                    volume_quota_remaining=None,
                    rate_limit_headroom_status="OBSERVED",
                    decision="HOLD",
                    admissible_for_financial_claim=False,
                )) + "\n")

            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("schema v2", result.stdout)

    def test_mixed_telemetry_versions_fail(self):
        """Mixed v1/v2 telemetry.jsonl files should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            with open(telemetry_path, "w") as f:
                f.write(json.dumps(self._make_valid_telemetry_record(0)) + "\n")
                f.write(json.dumps(self._make_valid_telemetry_v2_record(1)) + "\n")

            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("schema_version mismatch", result.stdout)

    def test_unknown_telemetry_schema_version_fails(self):
        """Unknown telemetry schema versions should fail without fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            with open(telemetry_path, "w") as f:
                f.write(json.dumps(self._make_valid_telemetry_v2_record(0, schema_version=999)) + "\n")

            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("Unsupported schema_version", result.stderr)

    def test_telemetry_v2_unsafe_boolean_invariants_fail(self):
        """Schema v2 should reject records that grant live/canary/capital/risk authority."""
        cases = [
            ("no_live_flag", False, "must be true"),
            ("approved_for_live", True, "must be false"),
            ("approved_for_canary", True, "must be false"),
            ("approved_for_capital_escalation", True, "must be false"),
            ("live_orders_allowed", True, "must be false"),
            ("capital_change_allowed", True, "must be false"),
            ("risk_limit_relaxation_allowed", True, "must be false"),
            ("capital_escalation_flag", True, "must be false"),
            ("risk_limit_override_flag", True, "must be false"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for field, unsafe_value, expected_text in cases:
                telemetry_path = Path(tmpdir) / "telemetry.jsonl"
                record = self._make_valid_telemetry_v2_record(0, **{field: unsafe_value})
                telemetry_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

                result = subprocess.run(
                    [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    1,
                    f"field={field} stdout={result.stdout} stderr={result.stderr}",
                )
                self.assertIn(field, result.stdout)
                self.assertIn(expected_text, result.stdout)

    def test_phase51_shadow_harness_emits_valid_hold_artifact(self):
        """Phase 5.1 shadow harness should emit source-linked HOLD-only v2 artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "phase5_sample.telemetry.jsonl"
            output_root = tmp_path / "phase51_runs"
            run_id = "phase51_contract_test"
            source_record = self._make_valid_telemetry_record(
                tick=42,
                fair_value=100.25,
                config_version_id="test-config",
                quote_levels=[
                    {
                        "venue_id": "lighter",
                        "side": "Bid",
                        "price": 100.0,
                        "size_final": 0.01,
                        "edge_local": 0.12,
                        "candidate_edge_pre_utility": 0.15,
                        "edge_threshold": 0.1,
                        "book_age_ms": 7,
                        "book_stale_threshold_ms": 12000,
                        "distance_to_touch_bps": 2.5,
                        "size_raw": 1.2,
                        "size_margin_cap": 4.0,
                        "quote_state": "active",
                        "suppression_reason": None,
                        "utility_tier": "full",
                        "utility_role": "fill",
                        "utility_reason": "healthy",
                    },
                    {
                        "venue_id": "paradex",
                        "side": "Ask",
                        "price": 100.5,
                        "size_final": 0.01,
                        "edge_local": 0.1,
                    },
                ],
            )
            suppressed_record = self._make_valid_telemetry_record(
                tick=43,
                fair_value=100.25,
                config_version_id="test-config",
                quote_levels=[],
            )
            input_path.write_text(
                json.dumps(source_record) + json.dumps(suppressed_record) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51_shadow_path()),
                    "--input-telemetry",
                    str(input_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    run_id,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")

            run_dir = output_root / run_id
            telemetry_path = run_dir / "telemetry.jsonl"
            self.assertTrue(telemetry_path.exists())
            validator = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(validator.returncode, 0, f"stdout: {validator.stdout}\nstderr: {validator.stderr}")

            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ev_records = [r for r in records if r["event_type"] == "V2_EV_EVALUATED"]
            replay_labels = [r for r in records if r["event_type"] == "V2_REPLAY_LABEL"]
            forbidden_execution_events = {
                "V2_ORDER_INTENT",
                "V2_ORDER_LIFECYCLE",
                "V2_FILL_OBSERVED",
                "V2_FAST_HEDGE_DECISION",
                "V2_HEDGE_LIFECYCLE",
            }
            self.assertEqual(len(ev_records), 1)
            self.assertEqual(len(replay_labels), 1)
            self.assertTrue(all(r["no_live_flag"] is True for r in records))
            self.assertTrue(all(r["approved_for_live"] is False for r in records))
            self.assertTrue(all(r["approved_for_canary"] is False for r in records))
            self.assertTrue(all(r["approved_for_capital_escalation"] is False for r in records))
            self.assertTrue(all(r["live_orders_allowed"] is False for r in records))
            self.assertTrue(all(r["capital_change_allowed"] is False for r in records))
            self.assertTrue(all(r["risk_limit_relaxation_allowed"] is False for r in records))
            self.assertFalse(forbidden_execution_events.intersection({r["event_type"] for r in records}))
            self.assertTrue(all(r.get("decision") == "HOLD" for r in records if "decision" in r))
            self.assertTrue(all(
                r.get("admissible_for_financial_claim") is False
                for r in records
                if "admissible_for_financial_claim" in r
            ))

            ev = ev_records[0]
            self.assertEqual(ev["source_t"], 42)
            self.assertEqual(ev["source_line"], 1)
            self.assertEqual(ev["instrument_id"], "ETH-PERP")
            self.assertEqual(ev["passive_price"], 100.0)
            self.assertEqual(ev["candidate_size_Q"], 0.01)
            self.assertEqual(ev["quote_state"], "active")
            self.assertEqual(ev["calibration_status"], "SPARSE")
            self.assertEqual(ev["calibration_sample_count"], 0)
            self.assertFalse(ev["pair_conditioned_flag"])
            self.assertFalse(ev["fast_hedge_allowed"])
            self.assertEqual(ev["fast_hedge_serialization_state"], "NOT_APPLICABLE_NONLIVE_SHADOW")
            self.assertFalse(ev["residual_state_required"])
            self.assertEqual(ev["residual_state_status"], "NO_FILL_NO_RESIDUAL")
            self.assertEqual(ev["action_owner"], "NO_ACTION_NONLIVE_SHADOW")
            self.assertEqual(ev["double_action_prevention_state"], "NO_EXECUTION_EVENTS_EMITTED")
            self.assertEqual(ev["min_quote_candidates_required"], 1000)
            self.assertEqual(ev["min_fill_labels_required"], 200)
            self.assertEqual(ev["min_hedge_labels_required"], 100)
            self.assertIn("missing_phase51_calibration", ev["binding_constraints"])
            self.assertEqual(ev["decision_reason_primary"], "phase51_calibration_hold")
            self.assertIn("missing_pfill_calibration", ev["decision_reason_secondary_list"])
            self.assertIn("missing_tail_risk_calibration", ev["decision_reason_secondary_list"])

            label = replay_labels[0]
            self.assertEqual(label["candidate_id"], ev["candidate_id"])
            self.assertEqual(label["label_type"], "COUNTERFACTUAL_DECISION")
            self.assertEqual(label["source_t"], 42)
            self.assertEqual(label["source_line"], 1)
            self.assertEqual(label["source_record_sha256"], ev["source_record_sha256"])
            self.assertEqual(label["run_id"], ev["run_id"])
            self.assertEqual(label["baseline_commit"], ev["baseline_commit"])
            self.assertEqual(label["label_confidence"], 1.0)

            summary = json.loads((run_dir / "ev_shadow_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidates_evaluated"], 1)
            self.assertEqual(summary["replay_labels_emitted"], 1)
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["calibration_status"], "SPARSE")
            self.assertEqual(summary["hold_reason_counts"]["missing_pfill_calibration"], 1)
            gate = json.loads((run_dir / "gate_result.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "HOLD")
            self.assertFalse(gate["approved_for_live"])
            self.assertFalse(gate["approved_for_canary"])
            self.assertFalse(gate["approved_for_capital_escalation"])

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            artifact_index = json.loads(
                (run_dir / "evidence_pack" / "artifact_index.json").read_text(encoding="utf-8")
            )
            for artifact in (manifest, artifact_index):
                metadata = artifact["metadata"]
                self.assertEqual(metadata["run_id"], run_id)
                self.assertEqual(metadata["baseline_commit"], ev["baseline_commit"])
                self.assertTrue(metadata["no_live_flag"])
                self.assertEqual(metadata["input_sha256"], hashlib.sha256(input_path.read_bytes()).hexdigest())
                self.assertEqual(metadata["input_artifact_mode"], "copy")
                self.assertFalse(metadata["approved_for_live"])
                self.assertFalse(metadata["approved_for_canary"])
                self.assertFalse(metadata["approved_for_capital_escalation"])
                self.assertFalse(metadata["live_orders_allowed"])
                self.assertFalse(metadata["capital_change_allowed"])
                self.assertFalse(metadata["risk_limit_relaxation_allowed"])

            ref_run_id = f"{run_id}_reference"
            reference = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51_shadow_path()),
                    "--input-telemetry",
                    str(input_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    ref_run_id,
                    "--input-artifact-mode",
                    "reference",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(reference.returncode, 0, f"stdout: {reference.stdout}\nstderr: {reference.stderr}")
            ref_dir = output_root / ref_run_id
            self.assertFalse((ref_dir / "input_telemetry.source.jsonl").exists())
            ref_manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(ref_manifest["metadata"]["input_artifact_mode"], "reference")
            self.assertEqual(ref_manifest["metadata"]["input_sha256"], hashlib.sha256(input_path.read_bytes()).hexdigest())

            telemetry_hash = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
            manifest_hash = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
            repeat = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51_shadow_path()),
                    "--input-telemetry",
                    str(input_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    run_id,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeat.returncode, 0, f"stdout: {repeat.stdout}\nstderr: {repeat.stderr}")
            self.assertEqual(hashlib.sha256(telemetry_path.read_bytes()).hexdigest(), telemetry_hash)
            self.assertEqual(hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(), manifest_hash)

    def test_phase51_shadow_harness_rejects_live_or_relaxed_specs(self):
        """Phase 5.1 shadow harness should fail closed on live/capital/risk spec drift."""
        script_dir = Path(__file__).parent.parent
        base_spec = json.loads((script_dir / "configs" / "phase51_lighter_only_ev_shadow.json").read_text())
        cases = [
            ("wrong_run_mode", {"run_mode": "LIVE"}, "run_mode must be SHADOW"),
            ("no_live_false", {"no_live_flag": False}, "no_live_flag must be true"),
            ("capital_escalation_true", {"capital_escalation_flag": True}, "capital escalation must be false"),
            ("risk_override_true", {"risk_limit_override_flag": True}, "risk limit override must be false"),
            ("wrong_venue", {"venue_id": "hyperliquid"}, "must be Lighter-only"),
            (
                "live_orders_allowed",
                {"constraints": {**base_spec["constraints"], "live_orders_allowed": True}},
                "live_orders_allowed must be false",
            ),
            (
                "capital_change_allowed",
                {"constraints": {**base_spec["constraints"], "capital_change_allowed": True}},
                "capital_change_allowed must be false",
            ),
            (
                "risk_limit_relaxation_allowed",
                {"constraints": {**base_spec["constraints"], "risk_limit_relaxation_allowed": True}},
                "risk_limit_relaxation_allowed must be false",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for name, overrides, expected_error in cases:
                spec = dict(base_spec)
                spec.update(overrides)
                spec_path = tmp_path / f"{name}.json"
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(self._get_phase51_shadow_path()),
                        "--spec",
                        str(spec_path),
                        "--output-root",
                        str(tmp_path / "runs"),
                        "--run-id",
                        name,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2, f"case={name} stdout={result.stdout} stderr={result.stderr}")
                self.assertIn(expected_error, result.stderr)

    def test_phase51b_lighter_account_limits_collector_emits_valid_hold_artifact(self):
        """Phase 5.1b collector should emit read-only HOLD v2 evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_root = tmp_path / "phase51b_runs"
            run_id = "phase51b_contract_test"
            account_path = tmp_path / "account.json"
            limits_path = tmp_path / "account_limits.json"
            active_orders_path = tmp_path / "active_orders.json"
            inactive_orders_path = tmp_path / "inactive_orders.json"
            order_books_path = tmp_path / "order_books.json"
            trades_path = tmp_path / "trades.json"
            trade_export_path = tmp_path / "trade_export.json"
            env_path = tmp_path / "lighter.env"
            env_path.write_text(
                "LIGHTER_ACCOUNT_INDEX='123'\n"
                "LIGHTER_AUTH_TOKEN='should-not-appear-in-artifacts'\n",
                encoding="utf-8",
            )
            account_path.write_text(json.dumps({
                "accounts": [{
                    "account_index": 123,
                    "l1_address": "0xabc",
                    "account_type": "STANDARD",
                    "auth_token": "should-redact",
                    "authToken": "should-redact",
                    "accessToken": "should-redact",
                    "jwt": "should-redact",
                    "password": "should-redact",
                    "credential": "should-redact",
                }]
            }), encoding="utf-8")
            limits_path.write_text(json.dumps({
                "sendtx_per_minute_limit": 60,
                "sendtx_per_minute_remaining": 59,
                "rest_requests_per_minute_limit": 60,
                "rest_requests_per_minute_remaining": 58,
                "weighted_requests_per_minute_limit": 24000,
                "weighted_requests_per_minute_remaining": 23999,
                "active_orders_per_account_limit": 1500,
                "active_orders_per_market_limit": 1000,
                "volume_quota_remaining": 12.5,
                "user_tier": "premium",
                "user_tier_name": "premium",
                "current_maker_fee_tick": 40,
                "current_taker_fee_tick": 280,
                "effective_lit_stakes": "0.00000000",
            }), encoding="utf-8")
            active_orders_path.write_text(json.dumps({
                "active_orders": [
                    {"order_id": "1", "market_id": 0, "status": "OPEN"},
                    {"order_id": "2", "market_id": 0, "status": "PENDING"},
                    {"order_id": "3", "market_id": 8, "status": "OPEN"},
                ]
            }), encoding="utf-8")
            inactive_orders_path.write_text(json.dumps({
                "orders": [
                    {
                        "order_id": "inactive-1",
                        "client_order_id": "client-inactive-1",
                        "market_index": 0,
                        "status": "FILLED",
                    }
                ],
                "next_cursor": "cursor-should-hash",
            }), encoding="utf-8")
            order_books_path.write_text(json.dumps({
                "order_books": [{
                    "market_id": 0,
                    "symbol": "ETH-USD",
                    "maker_fee": "0.0040",
                    "taker_fee": "0.0280",
                    "supported_price_decimals": 2,
                    "supported_size_decimals": 4,
                }]
            }), encoding="utf-8")
            trades_path.write_text(json.dumps({
                "trades": [
                    {"trade_id": "t1", "ask_account_id": 123, "bid_account_id": 456, "is_maker_ask": True},
                    {"trade_id": "t2", "ask_account_id": 123, "bid_account_id": 456, "is_maker_ask": False},
                    {"trade_id": "t3"},
                ]
            }), encoding="utf-8")
            trade_export_path.write_text(json.dumps({
                "data_url": "https://example.invalid/presigned?signature=should-redact",
                "rows": [
                    {
                        "trade_id": "te1",
                        "ask_id": "ask-1",
                        "bid_id": "bid-1",
                        "ask_account_id": 123,
                        "bid_account_id": 456,
                        "is_maker_ask": True,
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51b_lighter_account_limits_path()),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    run_id,
                    "--account-json",
                    str(account_path),
                    "--account-limits-json",
                    str(limits_path),
                    "--active-orders-json",
                    str(active_orders_path),
                    "--inactive-orders-json",
                    str(inactive_orders_path),
                    "--order-books-json",
                    str(order_books_path),
                    "--trades-json",
                    str(trades_path),
                    "--trade-export-json",
                    str(trade_export_path),
                    "--env-file",
                    str(env_path),
                    "--market-symbol",
                    "ETH-USD",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")

            run_dir = output_root / run_id
            telemetry_path = run_dir / "telemetry.jsonl"
            self.assertTrue(telemetry_path.exists())
            validator = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(telemetry_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(validator.returncode, 0, f"stdout: {validator.stdout}\nstderr: {validator.stderr}")

            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            event_types = {record["event_type"] for record in records}
            self.assertIn("V2_LIGHTER_ACCOUNT_PROFILE", event_types)
            self.assertIn("V2_LIGHTER_ACCOUNT_LIMITS", event_types)
            self.assertIn("V2_LIGHTER_OFFICIAL_LIMITS_DOC", event_types)
            self.assertIn("V2_LIGHTER_ACTIVE_ORDERS", event_types)
            self.assertIn("V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE", event_types)
            self.assertTrue(all(r["no_live_flag"] is True for r in records))
            self.assertTrue(all(r["approved_for_live"] is False for r in records))
            self.assertTrue(all(r["live_orders_allowed"] is False for r in records))
            self.assertTrue(all(r.get("decision") == "HOLD" for r in records if "decision" in r))
            self.assertTrue(all(
                r.get("admissible_for_financial_claim") is False
                for r in records
                if "admissible_for_financial_claim" in r
            ))

            profile = next(r for r in records if r["event_type"] == "V2_LIGHTER_ACCOUNT_PROFILE")
            self.assertEqual(profile["lighter_account_type"], "STANDARD")
            self.assertEqual(profile["market_id"], 0)
            self.assertEqual(profile["maker_fee_raw"], "0.0040")
            self.assertEqual(profile["taker_fee_raw"], "0.0280")
            self.assertAlmostEqual(profile["maker_fee_bps"], 0.4)
            self.assertAlmostEqual(profile["taker_fee_bps"], 2.8)
            self.assertEqual(profile["price_decimals"], 2)
            self.assertEqual(profile["size_decimals"], 4)

            limits = next(r for r in records if r["event_type"] == "V2_LIGHTER_ACCOUNT_LIMITS")
            self.assertEqual(limits["sendtx_per_minute_limit"], 60)
            self.assertEqual(limits["sendtx_per_minute_remaining"], 59)
            self.assertEqual(limits["rest_requests_per_minute_limit"], 60)
            self.assertEqual(limits["rest_requests_per_minute_remaining"], 58)
            self.assertEqual(limits["weighted_requests_per_minute_limit"], 24000)
            self.assertEqual(limits["weighted_requests_per_minute_remaining"], 23999)
            self.assertEqual(limits["active_orders_per_account_limit"], 1500)
            self.assertEqual(limits["active_orders_per_market_limit"], 1000)
            self.assertEqual(limits["lighter_user_tier"], "premium")
            self.assertEqual(limits["lighter_user_tier_name"], "premium")
            self.assertEqual(limits["current_maker_fee_tick"], 40)
            self.assertEqual(limits["current_taker_fee_tick"], 280)
            self.assertEqual(limits["effective_lit_stakes"], "0.00000000")

            official_limits = next(r for r in records if r["event_type"] == "V2_LIGHTER_OFFICIAL_LIMITS_DOC")
            self.assertEqual(official_limits["official_active_orders_per_account_limit"], 1500)
            self.assertEqual(official_limits["official_active_orders_per_market_limit"], 1000)
            self.assertTrue(official_limits["official_doc_cap_not_event_time_usage"])

            active_orders = next(r for r in records if r["event_type"] == "V2_LIGHTER_ACTIVE_ORDERS")
            self.assertEqual(active_orders["active_orders_count_total"], 3)
            self.assertEqual(active_orders["active_orders_count_market"], 2)
            self.assertEqual(active_orders["pending_orders_count_total"], 1)
            self.assertEqual(active_orders["active_order_limit_source"], "API_ACCOUNT_LIMITS")
            self.assertEqual(active_orders["open_order_limit_status"], "OBSERVED_API_ACCOUNT_LIMIT")
            self.assertEqual(active_orders["native_limit_time_alignment_status"], "CURRENT_SNAPSHOT_NOT_LABEL_EVENT_TIME")
            self.assertEqual(active_orders["active_order_headroom_account"], 1497)
            self.assertEqual(active_orders["active_order_headroom_market"], 998)

            trades = next(r for r in records if r["event_type"] == "V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE")
            self.assertEqual(trades["trade_sample_count"], 3)
            self.assertEqual(trades["maker_trade_count"], 1)
            self.assertEqual(trades["taker_trade_count"], 1)
            self.assertEqual(trades["unknown_role_trade_count"], 1)

            sanitized_account = json.loads(
                (run_dir / "source_snapshots" / "account.sanitized.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sanitized_account["accounts"][0]["auth_token"], "<redacted>")
            self.assertEqual(sanitized_account["accounts"][0]["authToken"], "<redacted>")
            self.assertEqual(sanitized_account["accounts"][0]["accessToken"], "<redacted>")
            self.assertEqual(sanitized_account["accounts"][0]["jwt"], "<redacted>")
            self.assertEqual(sanitized_account["accounts"][0]["password"], "<redacted>")
            self.assertEqual(sanitized_account["accounts"][0]["credential"], "<redacted>")
            summary = json.loads((run_dir / "lighter_account_native_limits_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["phase51b_capture_complete"])
            self.assertIn("inactive_orders", summary["source_names"])
            self.assertIn("trade_export", summary["source_names"])
            gate = json.loads((run_dir / "gate_result.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "HOLD")
            self.assertTrue(gate["approved_for_nonlive_evidence_review"])
            self.assertFalse(gate["approved_for_calibration_label_ingestion"])
            self.assertEqual(
                gate["calibration_label_ingestion_hold_reason"],
                "requires_external_schema_validation_and_secret_audit",
            )
            for artifact in run_dir.rglob("*"):
                if artifact.is_file():
                    content = artifact.read_text(encoding="utf-8", errors="ignore")
                    self.assertNotIn("should-not-appear-in-artifacts", content, str(artifact))
                    self.assertNotIn("should-redact", content, str(artifact))
                    self.assertNotIn("inactive-1", content, str(artifact))
                    self.assertNotIn("client-inactive-1", content, str(artifact))
                    self.assertNotIn("cursor-should-hash", content, str(artifact))
                    self.assertNotIn("presigned?signature", content, str(artifact))
            sanitized_inactive = json.loads(
                (run_dir / "source_snapshots" / "inactive_orders.sanitized.json").read_text(encoding="utf-8")
            )
            self.assertIn("order_id_sha256", sanitized_inactive["orders"][0])
            self.assertIn("client_order_id_sha256", sanitized_inactive["orders"][0])
            self.assertIn("next_cursor_sha256", sanitized_inactive)
            sanitized_export = json.loads(
                (run_dir / "source_snapshots" / "trade_export.sanitized.json").read_text(encoding="utf-8")
            )
            self.assertTrue(sanitized_export["data_url_present"])
            self.assertIn("data_url_sha256", sanitized_export)
            self.assertIn("ask_id_sha256", sanitized_export["rows"][0])
            self.assertIn("bid_id_sha256", sanitized_export["rows"][0])
            self.assertFalse(gate["approved_for_live"])

    def test_phase51b_lighter_account_limits_rejects_live_or_sendtx_specs(self):
        """Phase 5.1b collector should fail closed on live or sendTx spec drift."""
        script_dir = Path(__file__).parent.parent
        base_spec = json.loads((script_dir / "configs" / "phase51b_lighter_account_native_limits.json").read_text())
        cases = [
            ("wrong_run_mode", {"run_mode": "LIVE"}, "run_mode must be READ_ONLY"),
            ("no_live_false", {"no_live_flag": False}, "no_live_flag must be true"),
            ("capital_escalation_true", {"capital_escalation_flag": True}, "capital escalation must be false"),
            ("risk_override_true", {"risk_limit_override_flag": True}, "risk limit override must be false"),
            ("wrong_venue", {"venue_id": "hyperliquid"}, "must be Lighter-only"),
            (
                "sendtx_allowed",
                {"constraints": {**base_spec["constraints"], "sendtx_allowed": True}},
                "sendtx_allowed must be false",
            ),
            (
                "live_orders_allowed",
                {"constraints": {**base_spec["constraints"], "live_orders_allowed": True}},
                "live_orders_allowed must be false",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            account_limits_path = tmp_path / "account_limits.json"
            account_limits_path.write_text(json.dumps({"active_orders_per_account_limit": 10}), encoding="utf-8")
            for name, overrides, expected_error in cases:
                spec = dict(base_spec)
                spec.update(overrides)
                spec_path = tmp_path / f"{name}.json"
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(self._get_phase51b_lighter_account_limits_path()),
                        "--spec",
                        str(spec_path),
                        "--output-root",
                        str(tmp_path / "runs"),
                        "--run-id",
                        name,
                        "--account-limits-json",
                        str(account_limits_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2, f"case={name} stdout={result.stdout} stderr={result.stderr}")
                self.assertIn(expected_error, result.stderr)

    def test_phase51b_acceptance_promotes_only_calibration_ingestion(self):
        """Phase 5.1b acceptance should promote only to 5.1c calibration ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_dir = tmp_path / "phase51b_accept"
            run_dir.mkdir()
            env_path = tmp_path / "lighter.env"
            env_path.write_text("LIGHTER_AUTH_TOKEN='secret-token-value'\n", encoding="utf-8")
            records = [
                self._make_valid_telemetry_v2_record(1, run_id="phase51b_accept_test"),
                self._make_valid_telemetry_v2_record(
                    2,
                    run_id="phase51b_accept_test",
                    event_type="V2_LIGHTER_ACCOUNT_PROFILE",
                    venue_id="lighter",
                    account_index=123,
                    lighter_account_type="ACCOUNT_TYPE_0",
                    lighter_account_profile_status="OBSERVED",
                    market_id=0,
                    market_symbol="ETH",
                    market_metadata_status="OBSERVED",
                    decision="HOLD",
                    admissible_for_financial_claim=False,
                ),
                self._make_valid_telemetry_v2_record(
                    3,
                    run_id="phase51b_accept_test",
                    event_type="V2_LIGHTER_ACCOUNT_LIMITS",
                    venue_id="lighter",
                    account_index=123,
                    account_limits_status="OBSERVED",
                    account_limits_source_sha256="abc",
                    account_limits_source_endpoint=None,
                    account_limits_raw_keys=["user_tier_name"],
                    lighter_user_tier_name="premium",
                    rate_limit_headroom_status="OBSERVED",
                    decision="HOLD",
                    admissible_for_financial_claim=False,
                ),
                self._make_valid_telemetry_v2_record(
                    4,
                    run_id="phase51b_accept_test",
                    event_type="V2_LIGHTER_ACTIVE_ORDERS",
                    venue_id="lighter",
                    account_index=123,
                    market_id=0,
                    active_orders_status="OBSERVED",
                    active_orders_source_sha256="def",
                    active_orders_source_endpoint=None,
                    active_orders_count_total=0,
                    active_orders_count_market=0,
                    pending_orders_count_total=0,
                    active_order_status_keys=[],
                    active_order_sample_hash="empty",
                    open_order_limit_status="UNKNOWN",
                    decision="HOLD",
                    admissible_for_financial_claim=False,
                ),
            ]
            with (run_dir / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            (run_dir / "gate_result.json").write_text(json.dumps({
                "status": "HOLD",
                "phase51b_capture_complete": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
            }), encoding="utf-8")
            (run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "metadata": {"run_id": "phase51b_accept_test"},
                "files": [],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51b_acceptance_path()),
                    str(run_dir),
                    "--sensitive-env-file",
                    str(env_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            acceptance = json.loads((run_dir / "phase51b_acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(acceptance["status"], "PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION")
            self.assertTrue(acceptance["approved_for_calibration_label_ingestion"])
            self.assertFalse(acceptance["approved_for_live"])
            self.assertFalse(acceptance["approved_for_canary"])
            self.assertFalse(acceptance["approved_for_capital_escalation"])
            self.assertFalse(acceptance["secret_scan"]["sensitive_value_leak_found"])
            self.assertIn("lighter_rest_or_weighted_limit_not_observed", acceptance["limitations"])
            self.assertIn("lighter_rest_or_weighted_remaining_not_observed", acceptance["limitations"])

    def test_phase51c_label_lake_scaffold_holds_without_fill_markout_balance(self):
        """Phase 5.1c label lake should preserve labels and keep model training blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "phase5_source.jsonl"
            ev_path = tmp_path / "ev_shadow" / "telemetry.jsonl"
            ev_path.parent.mkdir()
            acceptance_path = tmp_path / "phase51b_acceptance.json"
            output_root = tmp_path / "label_lake_runs"
            source_record = {
                "schema_version": 1,
                "t": 42,
                "orders": [{
                    "action": "place",
                    "status": "ack",
                    "venue_id": "lighter",
                    "decision_id": "d42_mm_v2_buy",
                    "order_id": "order-1",
                    "client_order_id": "client-1",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "post_only": True,
                    "reduce_only": False,
                }],
            }
            source_path.write_text(
                json.dumps(source_record) + json.dumps({"schema_version": 1, "t": 43, "orders": []}) + "\n",
                encoding="utf-8",
            )
            ev_records = [
                self._make_valid_telemetry_v2_record(1, run_id="ev_shadow_test"),
                self._make_valid_telemetry_v2_record(
                    2,
                    run_id="ev_shadow_test",
                    event_type="V2_EV_EVALUATED",
                    venue_id="lighter",
                    candidate_id="cand-1",
                    source_line=1,
                    source_t=42,
                    source_record_sha256="abc",
                    side="BID",
                    layer="TOUCH",
                    decision="HOLD",
                    decision_reason_primary="missing_pfill_calibration",
                    decision_reason_secondary_list=["counterfactual_only_nonfinancial"],
                    calibration_bucket_id="lighter:BID:TOUCH",
                    calibration_status="SPARSE",
                    admissible_for_financial_claim=False,
                ),
            ]
            with ev_path.open("w", encoding="utf-8") as f:
                for record in ev_records:
                    f.write(json.dumps(record) + "\n")
            acceptance_path.write_text(json.dumps({
                "status": "PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": ["lighter_open_order_limit_headroom_unknown"],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_label_lake_path()),
                    "--source-telemetry",
                    str(source_path),
                    "--ev-shadow-telemetry",
                    str(ev_path),
                    "--phase51b-acceptance",
                    str(acceptance_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_label_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_label_test"
            summary = json.loads((run_dir / "label_lake_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["quote_decision_labels"], 1)
            self.assertEqual(summary["order_lifecycle_labels"], 1)
            self.assertEqual(summary["fill_label_status"], "MISSING")
            self.assertEqual(summary["markout_label_status"], "MISSING")
            self.assertEqual(summary["balance_reconciliation_status"], "MISSING")
            self.assertEqual(summary["native_limit_pressure_status"], "UNKNOWN")
            self.assertFalse(summary["approved_for_model_training"])
            labels = [
                json.loads(line)
                for line in (run_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([label["label_type"] for label in labels], [
                "QUOTE_DECISION_LABEL",
                "ORDER_LIFECYCLE_LABEL",
                "LABEL_COVERAGE_SUMMARY",
            ])
            self.assertTrue(all(label["approved_for_live"] is False for label in labels))
            self.assertTrue(all(label["admissible_for_model_training"] is False for label in labels))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

    def test_phase51c_observed_labels_extracts_fills_and_balance_without_promotion(self):
        """Observed labels should capture fills/balances while keeping training blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "phase5_source.jsonl"
            balance_pre = tmp_path / "balance_pre_snapshot.json"
            balance_post = tmp_path / "balance_post_snapshot.json"
            balance_comparison = tmp_path / "balance_snapshot_comparison.json"
            lighter_trades = tmp_path / "lighter_trades.json"
            output_root = tmp_path / "observed_label_runs"
            source_record_1 = {
                "schema_version": 1,
                "t": 7,
                "kf_last_update_ms": 1700000000000,
                "fair_value": 100.0,
                "fills": [{
                    "venue_id": "lighter",
                    "venue_index": 3,
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "fee_bps": 0.0,
                    "purpose": "Mm",
                    "decision_id": "d7_mm_v3_buy",
                    "order_id": "123",
                    "client_order_id": "client-1",
                    "fill_time_ms": 1700000000000,
                    "markout_pnl_short": None,
                }],
            }
            source_record_2 = {
                "schema_version": 1,
                "t": 8,
                "kf_last_update_ms": 1700000001000,
                "fair_value": 101.0,
                "fills": [{
                    "venue_id": "lighter",
                    "venue_index": 3,
                    "side": "Sell",
                    "price": 101.0,
                    "size": 0.02,
                    "fee_bps": 0.0,
                    "purpose": "Mm",
                    "decision_id": "d8_mm_v3_sell",
                    "order_id": "order-2",
                    "client_order_id": "client-2",
                    "fill_time_ms": 1700000000250,
                    "markout_pnl_short": 0.001,
                    "maker_or_taker": "maker",
                }],
            }
            source_path.write_text(
                json.dumps(source_record_1) + json.dumps(source_record_2) + "\n",
                encoding="utf-8",
            )
            balance_pre.write_text(json.dumps({
                "schema_version": 1,
                "total_balance_usd": "100.00000000",
                "venues": ["lighter"],
            }), encoding="utf-8")
            balance_post.write_text(json.dumps({
                "schema_version": 1,
                "total_balance_usd": "100.00100000",
                "venues": ["lighter"],
            }), encoding="utf-8")
            balance_comparison.write_text(json.dumps({
                "schema_version": 1,
                "generated_at_utc": "2026-05-02T00:00:00Z",
                "venues": ["lighter"],
                "total": {
                    "pre_usd": "100.00000000",
                    "post_usd": "100.00100000",
                    "delta_usd": "0.00100000",
                },
                "per_venue": {
                    "lighter": {
                        "pre_balance_usd": "100.00000000",
                        "post_balance_usd": "100.00100000",
                        "delta_usd": "0.00100000",
                    },
                },
            }), encoding="utf-8")
            lighter_trades.write_text(json.dumps({
                "trades": [{
                    "bid_id": 123,
                    "bid_id_str": "123",
                    "bid_client_id": 123,
                    "bid_client_id_str": "123",
                    "ask_id": 456,
                    "ask_id_str": "456",
                    "is_maker_ask": False,
                }],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_observed_labels_path()),
                    "--source-telemetry",
                    str(source_path),
                    "--balance-pre",
                    str(balance_pre),
                    "--balance-post",
                    str(balance_post),
                    "--balance-comparison",
                    str(balance_comparison),
                    "--lighter-trades-json",
                    str(lighter_trades),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_observed_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--markout-horizons-ms",
                    "1000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_observed_test"
            summary = json.loads((run_dir / "observed_label_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["fill_labels"], 2)
            self.assertEqual(summary["markout_labels"], 2)
            self.assertEqual(summary["balance_reconciliation_labels"], 1)
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "observed_label_pack_requires_quote_join_holdout_and_board_review")
            self.assertEqual(summary["fill_label_status"], "OBSERVED")
            self.assertEqual(summary["markout_label_status"], "OBSERVED")
            self.assertEqual(summary["balance_reconciliation_status"], "OBSERVED")
            self.assertEqual(summary["maker_taker_role_counts"]["MAKER"], 2)
            self.assertEqual(summary["maker_taker_role_counts_by_venue"]["lighter"]["MAKER"], 2)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            labels = [
                json.loads(line)
                for line in (run_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([label["label_type"] for label in labels], [
                "OBSERVED_FILL_LABEL",
                "OBSERVED_MARKOUT_LABEL",
                "OBSERVED_FILL_LABEL",
                "OBSERVED_MARKOUT_LABEL",
                "BALANCE_RECONCILIATION_LABEL",
            ])
            self.assertEqual(labels[0]["maker_taker_attribution_status"], "OBSERVED")
            self.assertEqual(labels[0]["maker_taker_attribution_source"], "lighter_trades_json")
            self.assertEqual(labels[1]["markout_horizon_ms"], 1000)
            self.assertEqual(labels[2]["maker_taker_role"], "MAKER")
            self.assertTrue(all(label["approved_for_live"] is False for label in labels))
            self.assertTrue(all(label["admissible_for_model_training"] is False for label in labels))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

    def test_phase51c_join_holdout_joins_labels_without_training_promotion(self):
        """Join/holdout pack should link observed fills to quote/order labels and remain HOLD."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            label_lake_run = tmp_path / "label_lake"
            observed_run = tmp_path / "observed"
            output_root = tmp_path / "join_runs"
            label_lake_run.mkdir()
            observed_run.mkdir()
            source_sha = "source-sha-1"
            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "label_lake_test",
                "source_telemetry_sha256": source_sha,
                "ev_shadow_telemetry_sha256": "ev-sha-1",
                "quote_decision_labels": 1,
                "order_lifecycle_labels": 1,
            }), encoding="utf-8")
            label_lake_labels = [
                {
                    "label_type": "QUOTE_DECISION_LABEL",
                    "source_line": 7,
                    "source_t": 42,
                    "venue_id": "lighter",
                    "side": "BID",
                    "candidate_id": "cand-1",
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "source_line": 7,
                    "source_t": 42,
                    "venue_id": "lighter",
                    "side": "Buy",
                    "decision_id": "decision-1",
                    "order_id_hash": "order-hash-1",
                    "client_order_id_hash": "client-hash-1",
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
            ]
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in label_lake_labels:
                    f.write(json.dumps(label) + "\n")
            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "run_id": "observed_test",
                "source_telemetry_sha256": source_sha,
                "fill_labels": 1,
                "markout_labels": 1,
                "balance_reconciliation_labels": 1,
                "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
            }), encoding="utf-8")
            observed_labels = [
                {
                    "label_type": "OBSERVED_FILL_LABEL",
                    "fill_id": "fill-5",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "fill_time_ms": 1700000000000,
                    "decision_id": "decision-1",
                    "order_id_hash": "order-hash-1",
                    "client_order_id_hash": "client-hash-1",
                    "maker_taker_role": "MAKER",
                    "maker_taker_attribution_status": "OBSERVED",
                    "maker_taker_attribution_source": "lighter_trades_json",
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "OBSERVED_MARKOUT_LABEL",
                    "fill_id": "fill-5",
                    "markout_horizon_ms": 1000,
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "BALANCE_RECONCILIATION_LABEL",
                    "balance_reconciliation_status": "OBSERVED",
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
            ]
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in observed_labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_join_holdout_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--observed-run",
                    str(observed_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_join_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_join_test"
            summary = json.loads((run_dir / "join_holdout_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "deterministic_join_requires_board_review")
            self.assertEqual(summary["fill_labels"], 1)
            self.assertEqual(summary["order_join_count"], 1)
            self.assertEqual(summary["candidate_join_count"], 1)
            self.assertEqual(summary["complete_join_count"], 1)
            self.assertEqual(summary["markout_join_count"], 1)
            self.assertEqual(summary["balance_reconciliation_labels"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            labels = [
                json.loads(line)
                for line in (run_dir / "joined_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["candidate_id"], "cand-1")
            self.assertEqual(labels[0]["join_status"], "COMPLETE_FOR_NONLIVE_REVIEW")
            self.assertIn(labels[0]["holdout_split"], {"TRAIN", "HOLDOUT"})
            self.assertFalse(labels[0]["approved_for_model_training"])
            self.assertFalse(labels[0]["approved_for_live"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "run_id": "observed_test",
                "source_telemetry_sha256": "different-source-sha",
                "fill_labels": 1,
                "markout_labels": 1,
                "balance_reconciliation_labels": 1,
            }), encoding="utf-8")
            mismatch_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_join_holdout_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--observed-run",
                    str(observed_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_join_mismatch_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(mismatch_result.returncode, 2)
            self.assertIn("source_telemetry_sha256", mismatch_result.stderr)

    def test_phase51c_lighter_trade_backfill_ingests_offline_pages_without_promotion(self):
        """Trade backfill should paginate/normalize native role evidence without promotion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            page_path = tmp_path / "trades_page.json"
            output_root = tmp_path / "trade_backfill_runs"
            page_path.write_text(json.dumps({
                "next_cursor": "cursor-1",
                "trades": [
                    {
                        "ask_account_id": 123,
                        "ask_id": 11,
                        "ask_id_str": "11",
                        "ask_client_id": 111,
                        "ask_client_id_str": "111",
                        "bid_account_id": 456,
                        "bid_id": 22,
                        "bid_id_str": "22",
                        "bid_client_id": 222,
                        "bid_client_id_str": "222",
                        "is_maker_ask": True,
                        "timestamp": 1777448154939,
                        "trade_id": "raw-trade-id-one",
                        "trade_id_str": "raw-trade-id-one",
                        "tx_hash": "raw-tx-hash-one",
                    },
                    {
                        "ask_account_id": 456,
                        "ask_id": 33,
                        "ask_id_str": "33",
                        "bid_account_id": 123,
                        "bid_id": 44,
                        "bid_id_str": "44",
                        "is_maker_ask": False,
                        "timestamp": 1777448164939,
                        "trade_id": "raw-trade-id-two",
                        "tx_hash": "raw-tx-hash-two",
                    },
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_lighter_trade_backfill_path()),
                    "--page-json",
                    str(page_path),
                    "--account-index",
                    "123",
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_trade_backfill_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--stop-at-or-before-ms",
                    "1777448154939",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_trade_backfill_test"
            summary = json.loads((run_dir / "lighter_trade_backfill_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "native_trade_backfill_readonly_attribution_input_only")
            self.assertEqual(summary["source_mode"], "offline_page_json")
            self.assertEqual(summary["trade_count"], 2)
            self.assertEqual(summary["timestamp_min_ms"], 1777448154939)
            self.assertEqual(summary["timestamp_max_ms"], 1777448164939)
            self.assertTrue(summary["complete_to_requested_stop"])
            self.assertEqual(summary["role_counts_for_account"], {"maker": 2, "taker": 0, "unknown": 0})
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            trades = json.loads((run_dir / "source_snapshots" / "trades_backfill.sanitized.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trades["trades"]), 2)
            pages = json.loads(
                (run_dir / "source_snapshots" / "trades_backfill_pages.sanitized.json").read_text(encoding="utf-8")
            )
            raw_identifier_keys = {
                "ask_id",
                "ask_id_str",
                "ask_client_id",
                "ask_client_id_str",
                "bid_id",
                "bid_id_str",
                "bid_client_id",
                "bid_client_id_str",
                "trade_id",
                "trade_id_str",
                "tx_hash",
            }
            for trade in trades["trades"]:
                self.assertTrue(raw_identifier_keys.isdisjoint(trade.keys()), trade)
            for page in pages["pages"]:
                for trade in page["payload"]["trades"]:
                    self.assertTrue(raw_identifier_keys.isdisjoint(trade.keys()), trade)
            self.assertIn("ask_id_sha256", trades["trades"][0])
            self.assertIn("ask_id_str_sha256", trades["trades"][0])
            self.assertIn("ask_client_id_sha256", trades["trades"][0])
            self.assertIn("bid_id_sha256", trades["trades"][0])
            self.assertIn("trade_id_sha256", trades["trades"][0])
            self.assertIn("tx_hash_sha256", trades["trades"][0])
            sanitized_text = json.dumps({"pages": pages, "trades": trades}, sort_keys=True)
            self.assertNotIn("raw-trade-id-one", sanitized_text)
            self.assertNotIn("raw-tx-hash-one", sanitized_text)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

    def test_phase51c_lighter_trade_backfill_rejects_unknown_raw_identifier_key(self):
        """Trade backfill should fail closed if an unknown raw identifier-like key survives redaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            page_path = tmp_path / "trades_page.json"
            output_root = tmp_path / "trade_backfill_runs"
            page_path.write_text(json.dumps({
                "trades": [
                    {
                        "ask_account_id": 123,
                        "bid_account_id": 456,
                        "is_maker_ask": True,
                        "timestamp": 1777448154939,
                        "unexpected_venue_id": "raw-unexpected-id",
                    },
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_lighter_trade_backfill_path()),
                    "--page-json",
                    str(page_path),
                    "--account-index",
                    "123",
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_trade_backfill_bad_identifier_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("redaction left raw identifier-like keys", result.stderr)

    def test_phase51c_pfill_outcome_labels_filled_cancelled_censored_without_promotion(self):
        """P_fill outcome labels should distinguish filled, terminal unfilled, and censored orders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            label_lake_run = tmp_path / "label_lake"
            join_run = tmp_path / "join_holdout"
            output_root = tmp_path / "pfill_runs"
            label_lake_run.mkdir()
            join_run.mkdir()
            source_sha = "source-sha-pfill"
            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "label_lake_pfill_test",
                "source_telemetry_sha256": source_sha,
                "order_lifecycle_labels": 4,
            }), encoding="utf-8")
            order_labels = [
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 1,
                    "source_line": 10,
                    "source_t": 100,
                    "source_order_index": 0,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "sent",
                    "decision_id": "d-fill",
                    "order_id_hash": "order-fill",
                    "client_order_id_hash": "client-fill",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 2,
                    "source_line": 11,
                    "source_t": 101,
                    "source_order_index": 0,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "sent",
                    "decision_id": "d-cancel",
                    "order_id_hash": "order-cancel",
                    "client_order_id_hash": "client-cancel",
                    "side": "Sell",
                    "price": 101.0,
                    "size": 0.02,
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 3,
                    "source_line": 12,
                    "source_t": 102,
                    "source_order_index": 0,
                    "venue_id": "lighter",
                    "action": "cancel",
                    "status": "cancelled",
                    "decision_id": "d-cancel",
                    "order_id_hash": "order-cancel",
                    "client_order_id_hash": "client-cancel",
                    "side": "Sell",
                    "price": 101.0,
                    "size": 0.02,
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 4,
                    "source_line": 13,
                    "source_t": 103,
                    "source_order_index": 0,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "sent",
                    "decision_id": "d-censor",
                    "order_id_hash": "order-censor",
                    "client_order_id_hash": "client-censor",
                    "side": "Buy",
                    "price": 99.0,
                    "size": 0.01,
                    "approved_for_live": False,
                    "admissible_for_model_training": False,
                },
            ]
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in order_labels:
                    f.write(json.dumps(label) + "\n")
            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "run_id": "join_pfill_test",
                "source_telemetry_sha256": source_sha,
                "fill_labels": 1,
            }), encoding="utf-8")
            with (join_run / "joined_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "fill_id": "fill-1",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.004,
                    "fill_time_ms": 1700000000001,
                    "maker_taker_role": "MAKER",
                    "order_join_status": "JOINED",
                    "order_label_seq": 1,
                    "order_source_line": 10,
                    "order_source_t": 100,
                    "order_source_order_index": 0,
                    "order_id_hash": "order-fill",
                    "client_order_id_hash": "client-fill",
                    "order_decision_id": "d-fill",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_outcome_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_pfill_test"
            summary = json.loads((run_dir / "pfill_outcome_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["order_label_count"], 3)
            self.assertEqual(summary["filled_count"], 1)
            self.assertEqual(summary["not_filled_count"], 1)
            self.assertEqual(summary["censored_count"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            labels = [
                json.loads(line)
                for line in (run_dir / "pfill_order_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_decision = {label["decision_id"]: label for label in labels}
            self.assertEqual(by_decision["d-fill"]["p_fill_outcome"], 1.0)
            self.assertEqual(by_decision["d-fill"]["outcome_status"], "OBSERVED_FILLED")
            self.assertEqual(by_decision["d-fill"]["fill_count"], 1)
            self.assertEqual(by_decision["d-fill"]["filled_size_total"], 0.004)
            self.assertEqual(by_decision["d-cancel"]["p_fill_outcome"], 0.0)
            self.assertEqual(by_decision["d-cancel"]["outcome_status"], "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL")
            self.assertIsNone(by_decision["d-censor"]["p_fill_outcome"])
            self.assertEqual(by_decision["d-censor"]["outcome_status"], "CENSORED_OR_UNOBSERVED")
            self.assertTrue(all(label["approved_for_live"] is False for label in labels))
            self.assertTrue(all(label["approved_for_model_training"] is False for label in labels))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "run_id": "join_pfill_test",
                "source_telemetry_sha256": "different-source-sha",
                "fill_labels": 1,
            }), encoding="utf-8")
            mismatch = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_outcome_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_mismatch_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("source_telemetry_sha256", mismatch.stderr)

    def test_phase51c_pfill_calibration_readiness_preserves_splits_and_censoring(self):
        """P_fill readiness should summarize immutable splits without approving training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pfill_run = tmp_path / "pfill_outcome"
            output_root = tmp_path / "pfill_readiness"
            pfill_run.mkdir()
            labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "order_key": "order-1",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "terminal_action_first": None,
                    "observed_horizon_source_ticks": 5,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "order_key": "order-2",
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "terminal_action_first": "cancel",
                    "observed_horizon_source_ticks": 2,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "order_key": "order-3",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "terminal_action_first": None,
                    "observed_horizon_source_ticks": None,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_source",
                "gate_status": "HOLD",
                "gate_reason": "pfill_outcome_contains_censored_orders",
                "source_telemetry_sha256": "source-sha-readiness",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_readiness_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--min-observed-per-bucket",
                    "2",
                    "--min-holdout-observed-per-bucket",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_pfill_readiness_test"
            summary = json.loads((run_dir / "pfill_calibration_readiness_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "pfill_calibration_contains_censored_orders")
            self.assertEqual(summary["order_label_count"], 3)
            self.assertEqual(summary["observed_count"], 2)
            self.assertEqual(summary["filled_count"], 1)
            self.assertEqual(summary["not_filled_count"], 1)
            self.assertEqual(summary["censored_count"], 1)
            self.assertEqual(summary["missing_observed_horizon_count"], 1)
            self.assertEqual(summary["terminal_action_counts"]["cancel"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            buckets = [
                json.loads(line)
                for line in (run_dir / "pfill_calibration_buckets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            global_bucket = next(bucket for bucket in buckets if bucket["bucket_id"] == "GLOBAL")
            self.assertEqual(global_bucket["observed_fill_rate"], 0.5)
            self.assertIsNotNone(global_bucket["observed_fill_rate_ci_low_95"])
            split_manifest = [
                json.loads(line)
                for line in (run_dir / "pfill_order_split_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(split_manifest), 3)
            self.assertEqual({row["order_key"] for row in split_manifest}, {"order-1", "order-2", "order-3"})
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

            labels.append({
                **labels[0],
                "order_holdout_split": "HOLDOUT",
            })
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            conflict = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_readiness_conflict_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(conflict.returncode, 2)
            self.assertIn("conflicting order_holdout_split", conflict.stderr)

            labels = labels[:3]
            labels[0]["approved_for_live"] = True
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            unsafe_label = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_readiness_unsafe_label_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_label.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe_label.stderr)

            labels[0]["approved_for_live"] = False
            labels[0]["p_fill_outcome"] = 0.0
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            bad_outcome = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_readiness_bad_outcome_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad_outcome.returncode, 2)
            self.assertIn("OBSERVED_FILLED P_fill labels must carry p_fill_outcome=1.0", bad_outcome.stderr)

    def test_phase51c_pfill_censoring_audit_classifies_without_training_promotion(self):
        """P_fill censoring audit should classify censored rows and stay HOLD-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pfill_run = tmp_path / "pfill"
            label_lake_run = tmp_path / "lake"
            output_root = tmp_path / "audit"
            pfill_run.mkdir()
            label_lake_run.mkdir()
            source_sha = "source-sha-censor"
            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "lake_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "record_count": 100,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "filled",
                    "order_holdout_split": "TRAIN",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "order_label_seq": 1,
                    "order_source_line": 10,
                    "order_source_t": 10,
                    "order_id_hash": "order-filled",
                    "client_order_id_hash": "client-filled",
                    "fill_count": 1,
                    "terminal_event_count": 0,
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "cancelled",
                    "order_holdout_split": "HOLDOUT",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "order_label_seq": 2,
                    "order_source_line": 20,
                    "order_source_t": 20,
                    "order_id_hash": "order-cancel",
                    "client_order_id_hash": "client-cancel",
                    "fill_count": 0,
                    "terminal_event_count": 1,
                    "terminal_action_first": "cancel",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "boundary",
                    "order_holdout_split": "TRAIN",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "order_label_seq": 3,
                    "order_source_line": 98,
                    "order_source_t": 98,
                    "order_id_hash": "order-boundary",
                    "client_order_id_hash": "client-boundary",
                    "fill_count": 0,
                    "terminal_event_count": 0,
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "missing-terminal",
                    "order_holdout_split": "TRAIN",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "order_label_seq": 4,
                    "order_source_line": 30,
                    "order_source_t": 30,
                    "order_id_hash": "order-open",
                    "client_order_id_hash": "client-open",
                    "fill_count": 0,
                    "terminal_event_count": 0,
                    "venue_id": "paradex",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "pfill_outcome_contains_censored_orders",
                "source_telemetry_sha256": source_sha,
                "label_lake_run": str(label_lake_run),
                "order_label_count": 4,
                "filled_count": 1,
                "not_filled_count": 1,
                "censored_count": 2,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_censoring_audit_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_censor_audit_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--boundary-source-line-margin",
                    "5",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_pfill_censor_audit_test"
            summary = json.loads((run_dir / "pfill_censoring_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["order_label_count"], 4)
            self.assertEqual(summary["observed_count"], 2)
            self.assertEqual(summary["censored_count"], 2)
            self.assertEqual(summary["reason_counts"]["SOURCE_WINDOW_BOUNDARY"], 1)
            self.assertEqual(summary["reason_counts"]["NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            audit_labels = [
                json.loads(line)
                for line in (run_dir / "pfill_censoring_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(audit_labels), 4)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            labels[0]["approved_for_live"] = True
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_censoring_audit_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_censor_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

            labels[0]["approved_for_live"] = False
            labels.append({
                **labels[0],
                "order_holdout_split": "HOLDOUT",
            })
            with (pfill_run / "pfill_outcome_summary.json").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "gate_status": "HOLD",
                    "gate_reason": "pfill_outcome_contains_censored_orders",
                    "source_telemetry_sha256": source_sha,
                    "label_lake_run": str(label_lake_run),
                    "order_label_count": 5,
                    "filled_count": 2,
                    "not_filled_count": 1,
                    "censored_count": 2,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }))
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            split_conflict = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_censoring_audit_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_pfill_censor_split_conflict_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(split_conflict.returncode, 2)
            self.assertIn("conflicting order_holdout_split", split_conflict.stderr)

    def test_phase51e_lifecycle_truth_audit_canonicalizes_aliases_hold_only(self):
        """Lifecycle truth audit should explain censored aliases without relabeling/training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            label_lake_run = tmp_path / "label_lake"
            join_run = tmp_path / "join"
            pfill_run = tmp_path / "pfill"
            lighter_gap_run = tmp_path / "lighter_gap"
            observed_run = tmp_path / "observed"
            backfill_run = tmp_path / "backfill"
            output_root = tmp_path / "phase51e"
            label_lake_run.mkdir()
            join_run.mkdir()
            pfill_run.mkdir()
            lighter_gap_run.mkdir()
            observed_run.mkdir()
            (backfill_run / "source_snapshots").mkdir(parents=True)
            source_sha = "source-sha-phase51e"

            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "lake_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            lifecycle_labels = [
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 10,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "intent",
                    "decision_id": "d1",
                    "client_order_id_hash": "client-1",
                    "source_line": 10,
                    "source_t": 10,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 11,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "ack",
                    "decision_id": "d1",
                    "order_id_hash": "client-1",
                    "client_order_id_hash": "client-1",
                    "source_line": 11,
                    "source_t": 11,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 12,
                    "venue_id": "lighter",
                    "action": "cancel",
                    "status": "ack",
                    "decision_id": "d1",
                    "order_id_hash": "client-1",
                    "source_line": 12,
                    "source_t": 12,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 20,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "intent",
                    "decision_id": "d2",
                    "client_order_id_hash": "client-2",
                    "source_line": 20,
                    "source_t": 20,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 21,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "ack",
                    "decision_id": "d2",
                    "order_id_hash": "client-2",
                    "client_order_id_hash": "client-2",
                    "source_line": 21,
                    "source_t": 21,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 30,
                    "venue_id": "lighter",
                    "action": "place",
                    "status": "intent",
                    "decision_id": "d3",
                    "client_order_id_hash": "client-3",
                    "source_line": 30,
                    "source_t": 30,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 31,
                    "venue_id": "lighter",
                    "action": "cancel_all",
                    "status": "intent",
                    "source_line": 31,
                    "source_t": 31,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in lifecycle_labels:
                    f.write(json.dumps(label) + "\n")

            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "run_id": "join_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (join_run / "joined_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "venue_id": "lighter",
                    "fill_id": "fill-client-2",
                    "order_id_hash": "client-2",
                    "client_order_id_hash": "client-2",
                    "maker_taker_role": "UNKNOWN",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")

            pfill_labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "order-client-1-intent",
                    "order_label_seq": 10,
                    "order_source_line": 10,
                    "order_source_t": 10,
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "client_order_id_hash": "client-1",
                    "fill_count": 0,
                    "terminal_event_count": 0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "order-client-2-intent",
                    "order_label_seq": 20,
                    "order_source_line": 20,
                    "order_source_t": 20,
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "client_order_id_hash": "client-2",
                    "fill_count": 0,
                    "terminal_event_count": 0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "order-client-3-intent",
                    "order_label_seq": 30,
                    "order_source_line": 30,
                    "order_source_t": 30,
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "client_order_id_hash": "client-3",
                    "fill_count": 0,
                    "terminal_event_count": 0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "pfill_outcome_contains_censored_orders",
                "source_telemetry_sha256": source_sha,
                "label_lake_run": str(label_lake_run),
                "join_holdout_run": str(join_run),
                "order_label_count": 3,
                "filled_count": 0,
                "not_filled_count": 0,
                "censored_count": 3,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")

            raw_telemetry = tmp_path / "source.telemetry.jsonl"
            raw_fill = {
                "venue_id": "lighter",
                "side": "Sell",
                "price": 100.0,
                "size": 0.01,
                "fill_time_ms": 1000,
                "order_id": "raw-order",
                "client_order_id": "raw-client",
                "decision_id": "raw-decision",
            }
            raw_telemetry.write_text(json.dumps({"t": 1, "fills": [raw_fill]}) + "\n", encoding="utf-8")
            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "run_id": "observed_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "source_telemetry": str(raw_telemetry),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "labels.jsonl").write_text("", encoding="utf-8")
            trades_payload = {
                "account_index": 7,
                "trades": [
                    {
                        "trade_id": 99,
                        "trade_id_str": "99",
                        "timestamp": 1000,
                        "transaction_time": 1000,
                        "price": "100.0",
                        "size": "0.01",
                        "ask_account_id": 7,
                        "bid_account_id": 8,
                        "ask_id": "raw-order",
                        "ask_client_id": "raw-client",
                        "bid_id": "other-order",
                        "bid_client_id": "other-client",
                        "is_maker_ask": True,
                    },
                ],
            }
            trades_path = backfill_run / "source_snapshots" / "trades_backfill.sanitized.json"
            trades_path.write_text(json.dumps(trades_payload), encoding="utf-8")
            (backfill_run / "lighter_trade_backfill_summary.json").write_text(json.dumps({
                "run_id": "backfill_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "trades_path": str(trades_path),
                "trades_sha256": hashlib.sha256(trades_path.read_bytes()).hexdigest(),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (lighter_gap_run / "lighter_attribution_gap_summary.json").write_text(json.dumps({
                "run_id": "gap_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "observed_run": str(observed_run),
                "lighter_trade_backfill_run": str(backfill_run),
                "lighter_fill_count": 1,
                "observed_role_counts": {"UNKNOWN": 1},
                "gap_reason_counts": {"NO_NATIVE_TRADE_MATCH": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (lighter_gap_run / "lighter_attribution_gap_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label_type": "LIGHTER_ATTRIBUTION_GAP_AUDIT_LABEL",
                    "fill_id": "unknown-lighter-fill",
                    "fill_time_ms": 1000,
                    "side": "BID",
                    "price": 100.0,
                    "size": 0.01,
                    "observed_maker_taker_role": "UNKNOWN",
                    "native_role_if_determinable": None,
                    "gap_reason": "NO_NATIVE_TRADE_MATCH",
                    "gap_reason_detail": "no native trade matches identity or time/price/size",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51e_lifecycle_truth_audit_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--lighter-attribution-gap-run",
                    str(lighter_gap_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51e_lifecycle_truth_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51e_lifecycle_truth_test"
            summary = json.loads((run_dir / "lifecycle_truth_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["order_label_count"], 3)
            self.assertEqual(summary["current_censored_count"], 3)
            self.assertEqual(summary["canonical_status_counts"]["CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW"], 1)
            self.assertEqual(summary["canonical_status_counts"]["CENSORED_TO_CANONICAL_FILLED_REVIEW"], 1)
            self.assertEqual(summary["canonical_status_counts"]["CANCEL_ALL_SCOPE_REVIEW"], 1)
            self.assertEqual(summary["lighter_native_gap_reason_counts"], {"NO_NATIVE_TRADE_MATCH": 1})
            self.assertEqual(summary["lighter_raw_native_truth_label_count"], 1)
            self.assertEqual(summary["lighter_raw_native_match_status_counts"], {"MATCHED_NATIVE_ID": 1})
            self.assertEqual(summary["lighter_raw_native_role_counts"], {"MAKER": 1})
            self.assertFalse(summary["approved_for_live"])
            self.assertFalse(summary["admissible_for_ev_admission"])
            labels = [
                json.loads(line)
                for line in (run_dir / "order_lifecycle_truth_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({label["canonical_status"] for label in labels}, {
                "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW",
                "CENSORED_TO_CANONICAL_FILLED_REVIEW",
                "CANCEL_ALL_SCOPE_REVIEW",
            })
            native_gap_labels = [
                json.loads(line)
                for line in (run_dir / "lighter_native_identity_gap_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(native_gap_labels), 1)
            self.assertEqual(native_gap_labels[0]["gap_reason"], "NO_NATIVE_TRADE_MATCH")
            raw_native_labels = [
                json.loads(line)
                for line in (run_dir / "lighter_raw_native_truth_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(raw_native_labels), 1)
            self.assertEqual(raw_native_labels[0]["native_trade_match_status"], "MATCHED_NATIVE_ID")
            self.assertEqual(raw_native_labels[0]["native_role"], "MAKER")
            self.assertIn("decision_id_hash", raw_native_labels[0])
            self.assertNotIn("decision_id", raw_native_labels[0])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            pfill_labels[0]["approved_for_live"] = True
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51e_lifecycle_truth_audit_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51e_lifecycle_truth_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51f_canonical_pfill_outcome_rebuild_collapses_groups_hold_only(self):
        """Canonical P_fill rebuild should collapse lifecycle groups and keep review quarantines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            lifecycle_run = tmp_path / "phase51e"
            pfill_run = tmp_path / "pfill"
            output_root = tmp_path / "phase51f"
            readiness_root = tmp_path / "readiness"
            lifecycle_run.mkdir()
            pfill_run.mkdir()
            source_sha = "source-sha-phase51f"

            pfill_labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "raw-fill-observed",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "decision_id": "d-fill",
                    "order_id_hash": "order-fill",
                    "client_order_id_hash": "client-fill",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "fill_count": 1,
                    "filled_size_total": 0.01,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "raw-fill-alias",
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "decision_id": "d-fill",
                    "order_id_hash": "order-fill",
                    "client_order_id_hash": "client-fill",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "fill_count": 0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "raw-direct-terminal",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "raw-duplicate-only",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "raw-cancel-all",
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "pfill_outcome_contains_censored_orders",
                "source_telemetry_sha256": source_sha,
                "order_label_count": 5,
                "filled_count": 1,
                "not_filled_count": 0,
                "censored_count": 4,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")

            truth_rows = [
                {
                    "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "source_pfill_run_id": "pfill_source",
                    "source_pfill_run_path": str(pfill_run),
                    "order_key": "raw-fill-observed",
                    "order_label_seq": 1,
                    "canonical_group_id": "group-fill",
                    "canonical_status": "STAYS_FILLED",
                    "current_outcome_status": "OBSERVED_FILLED",
                    "current_p_fill_outcome": 1.0,
                    "canonical_direct_terminal_count": 0,
                    "venue_id": "lighter",
                    "side": "BID",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "source_pfill_run_id": "pfill_source",
                    "source_pfill_run_path": str(pfill_run),
                    "order_key": "raw-fill-alias",
                    "order_label_seq": 2,
                    "canonical_group_id": "group-fill",
                    "canonical_status": "CENSORED_TO_CANONICAL_FILLED_REVIEW",
                    "current_outcome_status": "CENSORED_OR_UNOBSERVED",
                    "current_p_fill_outcome": None,
                    "canonical_direct_terminal_count": 0,
                    "venue_id": "lighter",
                    "side": "BID",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "source_pfill_run_id": "pfill_source",
                    "source_pfill_run_path": str(pfill_run),
                    "order_key": "raw-direct-terminal",
                    "order_label_seq": 3,
                    "canonical_group_id": "group-terminal",
                    "canonical_status": "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW",
                    "current_outcome_status": "CENSORED_OR_UNOBSERVED",
                    "current_p_fill_outcome": None,
                    "canonical_direct_terminal_count": 1,
                    "venue_id": "lighter",
                    "side": "ASK",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "source_pfill_run_id": "pfill_source",
                    "source_pfill_run_path": str(pfill_run),
                    "order_key": "raw-duplicate-only",
                    "order_label_seq": 4,
                    "canonical_group_id": "group-duplicate",
                    "canonical_status": "DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW",
                    "current_outcome_status": "CENSORED_OR_UNOBSERVED",
                    "current_p_fill_outcome": None,
                    "canonical_direct_terminal_count": 0,
                    "venue_id": "lighter",
                    "side": "BID",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "source_pfill_run_id": "pfill_source",
                    "source_pfill_run_path": str(pfill_run),
                    "order_key": "raw-cancel-all",
                    "order_label_seq": 5,
                    "canonical_group_id": "group-cancel-all",
                    "canonical_status": "CANCEL_ALL_SCOPE_REVIEW",
                    "current_outcome_status": "CENSORED_OR_UNOBSERVED",
                    "current_p_fill_outcome": None,
                    "canonical_direct_terminal_count": 0,
                    "venue_id": "lighter",
                    "side": "ASK",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (lifecycle_run / "lifecycle_truth_audit_summary.json").write_text(json.dumps({
                "run_id": "phase51e_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "phase51e_canonical_lifecycle_reviewable_movements_found",
                "source_telemetry_sha256_list": [source_sha],
                "order_label_count": 5,
                "canonical_group_count": 4,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (lifecycle_run / "order_lifecycle_truth_labels.jsonl").open("w", encoding="utf-8") as f:
                for row in truth_rows:
                    f.write(json.dumps(row) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51f_canonical_pfill_outcome_path()),
                    "--lifecycle-truth-run",
                    str(lifecycle_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51f_canonical_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51f_canonical_test"
            summary = json.loads((run_dir / "pfill_outcome_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["source_label_count"], 5)
            self.assertEqual(summary["canonical_group_count"], 4)
            self.assertEqual(summary["order_label_count"], 4)
            self.assertEqual(summary["filled_count"], 1)
            self.assertEqual(summary["not_filled_count"], 1)
            self.assertEqual(summary["censored_count"], 2)
            self.assertEqual(summary["split_conflict_count"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            labels = [
                json.loads(line)
                for line in (run_dir / "canonical_pfill_order_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(labels), 4)
            by_group = {label["canonical_group_id"]: label for label in labels}
            self.assertEqual(by_group["group-fill"]["outcome_status"], "OBSERVED_FILLED")
            self.assertEqual(by_group["group-fill"]["p_fill_outcome"], 1.0)
            self.assertNotIn("decision_id", by_group["group-fill"])
            self.assertTrue(by_group["group-fill"]["decision_id_present"])
            self.assertIsNotNone(by_group["group-fill"]["decision_id_hash"])
            self.assertNotIn("d-fill", json.dumps(labels))
            self.assertEqual(by_group["group-terminal"]["outcome_status"], "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL")
            self.assertEqual(by_group["group-terminal"]["p_fill_outcome"], 0.0)
            self.assertEqual(by_group["group-duplicate"]["outcome_status"], "CENSORED_OR_UNOBSERVED")
            self.assertIsNone(by_group["group-duplicate"]["p_fill_outcome"])
            self.assertEqual(by_group["group-cancel-all"]["outcome_status"], "CENSORED_OR_UNOBSERVED")
            self.assertTrue(all(label["approved_for_model_training"] is False for label in labels))

            source_manifest = [
                json.loads(line)
                for line in (run_dir / "source_to_canonical_order_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(source_manifest), 5)
            quarantined = [
                json.loads(line)
                for line in (run_dir / "quarantined_review_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(quarantined), 2)
            split_conflicts = [
                json.loads(line)
                for line in (run_dir / "split_conflict_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(split_conflicts), 1)

            readiness = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(run_dir),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51f_readiness_compat_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--min-observed-per-bucket",
                    "1",
                    "--min-holdout-observed-per-bucket",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(readiness.returncode, 0, f"stdout: {readiness.stdout}\nstderr: {readiness.stderr}")
            readiness_summary = json.loads(
                (readiness_root / "phase51f_readiness_compat_test" / "pfill_calibration_readiness_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(readiness_summary["gate_status"], "HOLD")
            self.assertEqual(readiness_summary["order_label_count"], 4)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            truth_rows[0]["approved_for_live"] = True
            with (lifecycle_run / "order_lifecycle_truth_labels.jsonl").open("w", encoding="utf-8") as f:
                for row in truth_rows:
                    f.write(json.dumps(row) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51f_canonical_pfill_outcome_path()),
                    "--lifecycle-truth-run",
                    str(lifecycle_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51f_canonical_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51g_pfill_quarantine_review_emits_observed_only_diagnostic_pack(self):
        """P_fill quarantine review should exclude review groups from numeric outcomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            canonical_run = tmp_path / "phase51f"
            output_root = tmp_path / "phase51g"
            readiness_root = tmp_path / "readiness"
            canonical_run.mkdir()
            source_sha = "source-sha-phase51g"
            labels = [
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 1,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-filled",
                    "canonical_group_id": "group-filled",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "decision_id": "raw-phase51g-decision",
                    "client_order_id": "raw-client-id-must-not-propagate",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "source_label_count": 1,
                    "source_canonical_status_counts": {"STAYS_FILLED": 1},
                    "source_current_status_counts": {"OBSERVED_FILLED": 1},
                    "source_old_split_conflict": False,
                    "source_old_split_values": ["TRAIN"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 2,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-not-filled",
                    "canonical_group_id": "group-not-filled",
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "source_label_count": 1,
                    "source_canonical_status_counts": {"STAYS_NOT_FILLED": 1},
                    "source_current_status_counts": {"OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL": 1},
                    "source_old_split_conflict": False,
                    "source_old_split_values": ["HOLDOUT"],
                    "terminal_action_first": "cancel",
                    "observed_horizon_source_ticks": 2,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 3,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-duplicate",
                    "canonical_group_id": "group-duplicate",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "hyperliquid",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "source_label_count": 2,
                    "source_canonical_status_counts": {"DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW": 2},
                    "source_current_status_counts": {"CENSORED_OR_UNOBSERVED": 2},
                    "source_old_split_conflict": False,
                    "source_old_split_values": ["TRAIN"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 4,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-replace",
                    "canonical_group_id": "group-replace",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "source_label_count": 3,
                    "source_canonical_status_counts": {"CENSORED_TO_REPLACE_CHAIN_REVIEW": 3},
                    "source_current_status_counts": {"CENSORED_OR_UNOBSERVED": 3},
                    "source_old_split_conflict": True,
                    "source_old_split_values": ["HOLDOUT", "TRAIN"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 5,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-cancel-all",
                    "canonical_group_id": "group-cancel-all",
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "aster",
                    "side": "Buy",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "source_label_count": 1,
                    "source_canonical_status_counts": {"CANCEL_ALL_SCOPE_REVIEW": 1},
                    "source_current_status_counts": {"CENSORED_OR_UNOBSERVED": 1},
                    "source_old_split_conflict": False,
                    "source_old_split_values": ["HOLDOUT"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 6,
                    "run_id": "phase51f_source",
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-no-terminal",
                    "canonical_group_id": "group-no-terminal",
                    "order_holdout_split": "TRAIN",
                    "venue_id": "hyperliquid",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "source_label_count": 1,
                    "source_canonical_status_counts": {"REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW": 1},
                    "source_current_status_counts": {"CENSORED_OR_UNOBSERVED": 1},
                    "source_old_split_conflict": False,
                    "source_old_split_values": ["TRAIN"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
            ]
            (canonical_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "phase51f_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "phase51f_canonical_pfill_contains_quarantined_review_groups",
                "source_telemetry_sha256": source_sha,
                "order_label_count": 6,
                "filled_count": 1,
                "not_filled_count": 1,
                "censored_count": 4,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (canonical_run / "canonical_pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51g_pfill_quarantine_review_path()),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51g_quarantine_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51g_quarantine_test"
            summary = json.loads((run_dir / "quarantine_review_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["order_label_count"], 6)
            self.assertEqual(summary["observed_count"], 2)
            self.assertEqual(summary["filled_count"], 1)
            self.assertEqual(summary["not_filled_count"], 1)
            self.assertEqual(summary["censored_count"], 4)
            self.assertEqual(summary["exclusion_reason_counts"]["EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL"], 1)
            self.assertEqual(summary["exclusion_reason_counts"]["EXCLUDED_REPLACE_CHAIN_REVIEW"], 1)
            self.assertEqual(summary["exclusion_reason_counts"]["EXCLUDED_CANCEL_ALL_SCOPE_REVIEW"], 1)
            self.assertEqual(summary["exclusion_reason_counts"]["RIGHT_CENSORED_NO_TERMINAL"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["admissible_for_ev_admission"])

            review_labels = [
                json.loads(line)
                for line in (run_dir / "quarantine_review_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(review_labels), 6)
            excluded = [row for row in review_labels if not row["included_in_observed_only_pack"]]
            self.assertEqual(len(excluded), 4)
            self.assertTrue(all(row["input_p_fill_outcome"] is None for row in excluded))
            self.assertTrue(all(row["approved_for_model_training"] is False for row in review_labels))

            compat_dir = run_dir / "observed_only_pfill_outcome"
            compat_summary = json.loads((compat_dir / "pfill_outcome_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(compat_summary["order_label_count"], 2)
            self.assertEqual(compat_summary["filled_count"], 1)
            self.assertEqual(compat_summary["not_filled_count"], 1)
            self.assertEqual(compat_summary["censored_count"], 0)
            compat_labels = [
                json.loads(line)
                for line in (compat_dir / "pfill_order_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(compat_labels), 2)
            self.assertEqual({row["outcome_status"] for row in compat_labels}, {
                "OBSERVED_FILLED",
                "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
            })
            self.assertTrue(compat_labels[0]["decision_id_present"])
            self.assertIsNotNone(compat_labels[0]["decision_id_hash"])
            self.assertNotIn("decision_id", {key for row in compat_labels for key in row})
            self.assertNotIn("client_order_id", {key for row in compat_labels for key in row})
            self.assertNotIn("raw-phase51g-decision", json.dumps(compat_labels))
            self.assertNotIn("raw-client-id-must-not-propagate", json.dumps(compat_labels))

            readiness = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_pfill_calibration_readiness_path()),
                    "--pfill-outcome-run",
                    str(compat_dir),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51g_observed_readiness_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--min-observed-per-bucket",
                    "1",
                    "--min-holdout-observed-per-bucket",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(readiness.returncode, 0, f"stdout: {readiness.stdout}\nstderr: {readiness.stderr}")
            readiness_summary = json.loads(
                (readiness_root / "phase51g_observed_readiness_test" / "pfill_calibration_readiness_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(readiness_summary["order_label_count"], 2)
            self.assertEqual(readiness_summary["censored_count"], 0)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            labels[0]["approved_for_live"] = True
            with (canonical_run / "canonical_pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51g_pfill_quarantine_review_path()),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51g_quarantine_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51h_observed_pfill_feature_audit_reconciles_features_without_raw_ids(self):
        """Observed P_fill feature audit should join source features and redact raw IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            quarantine_run = tmp_path / "quarantine"
            canonical_run = tmp_path / "canonical"
            queue_run = tmp_path / "queue"
            markout_run = tmp_path / "markout"
            horizon_recovery_run = tmp_path / "horizon_recovery"
            filled_horizon_recovery_run = tmp_path / "filled_horizon_recovery"
            output_root = tmp_path / "phase51h"
            for path in [
                observed_run,
                quarantine_run,
                canonical_run,
                queue_run,
                markout_run,
                horizon_recovery_run,
                filled_horizon_recovery_run,
            ]:
                path.mkdir()
            source_sha = "source-sha-phase51h"
            baseline = "18dd09512288a85e440d3977e32432c3aabc1190"
            pfill_labels = [
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 1,
                    "run_id": "phase51g_observed",
                    "baseline_commit": baseline,
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-filled",
                    "canonical_group_id": "group-filled",
                    "source_order_keys": ["source-filled-a", "source-filled-b"],
                    "source_label_count": 2,
                    "order_holdout_split": "TRAIN",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                    "observed_horizon_source_ticks": None,
                    "terminal_event_count": 1,
                    "terminal_action_first": "canonical_direct_terminal",
                    "decision_id": "raw-decision-id-must-not-emit",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 2,
                    "run_id": "phase51g_observed",
                    "baseline_commit": baseline,
                    "source_telemetry_sha256": source_sha,
                    "order_key": "canonical-not-filled",
                    "canonical_group_id": "group-not-filled",
                    "source_order_keys": ["source-not-filled"],
                    "source_label_count": 1,
                    "order_holdout_split": "HOLDOUT",
                    "venue_id": "aster",
                    "side": "Sell",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "fill_count": 0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
                    "observed_horizon_source_ticks": None,
                    "terminal_event_count": 1,
                    "terminal_action_first": "cancel",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "admissible_for_ev_admission": False,
                },
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51g_observed",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51g_observed_only_binary_diagnostic_requires_board_review",
                "excluded_quarantine_count": 4,
                "excluded_quarantine_reason_counts": {
                    "EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL": 1,
                    "EXCLUDED_REPLACE_CHAIN_REVIEW": 1,
                    "EXCLUDED_CANCEL_ALL_SCOPE_REVIEW": 1,
                    "RIGHT_CENSORED_NO_TERMINAL": 1,
                },
                "observed_only_pack_warning": "diagnostic only; excluded censored groups are not negative outcomes",
                "order_label_count": 2,
                "filled_count": 1,
                "not_filled_count": 1,
                "censored_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")

            (quarantine_run / "quarantine_review_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51g_quarantine",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51g_quarantine_review_observed_only_diagnostic_pack",
                "approved_for_live": False,
                "approved_for_model_training": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (quarantine_run / "source_reconciliation_manifest.jsonl").open("w", encoding="utf-8") as f:
                for group_id, order_key in [("group-filled", "canonical-filled"), ("group-not-filled", "canonical-not-filled")]:
                    f.write(json.dumps({
                        "canonical_group_id": group_id,
                        "canonical_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": "lighter" if group_id == "group-filled" else "aster",
                        "review_status": "BINARY_OBSERVED_FILLED_DIAGNOSTIC",
                        "included_in_observed_only_pack": True,
                    }) + "\n")

            with (canonical_run / "source_to_canonical_order_manifest.jsonl").open("w", encoding="utf-8") as f:
                for source_key in ["source-filled-a", "source-filled-b"]:
                    f.write(json.dumps({
                        "canonical_group_id": "group-filled",
                        "canonical_order_key": "canonical-filled",
                        "source_order_key": source_key,
                        "source_telemetry_sha256": source_sha,
                    }) + "\n")
                f.write(json.dumps({
                    "canonical_group_id": "group-not-filled",
                    "canonical_order_key": "canonical-not-filled",
                    "source_order_key": "source-not-filled",
                    "source_telemetry_sha256": source_sha,
                }) + "\n")

            (queue_run / "queue_churn_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "queue_run",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "queue_churn_native_limit_pressure_unknown",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            queue_labels = [
                ("source-filled-a", "lighter", "PARTIAL_ACTIVE_ORDER_COUNT_OBSERVED_LIMIT_UNKNOWN", 1),
                ("source-filled-b", "lighter", "PARTIAL_ACTIVE_ORDER_COUNT_OBSERVED_LIMIT_UNKNOWN", 0),
                ("source-not-filled", "aster", "UNKNOWN_NON_LIGHTER_NO_NATIVE_LIMIT_INPUT", 0),
            ]
            with (queue_run / "queue_churn_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, (order_key, venue, native_status, replace_count) in enumerate(queue_labels, start=1):
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "QUEUE_CHURN_LABEL",
                        "label_seq": seq,
                        "run_id": "queue_run",
                        "baseline_commit": baseline,
                        "source_telemetry_sha256": source_sha,
                        "order_key": order_key,
                        "venue_id": venue,
                        "side": "BID",
                        "lifecycle_join_status": "JOINED",
                        "queue_reset_proxy_event_count": replace_count,
                        "replace_event_count": replace_count,
                        "cancel_event_count": 1,
                        "cancel_all_event_count": 0,
                        "churn_event_count": replace_count + 1,
                        "native_limit_pressure_status": native_status,
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "admissible_for_ev_admission": False,
                    }) + "\n")

            (markout_run / "markout_calibration_readiness_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "markout_run",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "markout_readiness_sparse_buckets",
                "source_telemetry_sha256_list": [source_sha],
                "approved_for_live": False,
                "approved_for_model_training": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")

            (horizon_recovery_run / "observed_horizon_recovery_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51j_recovery",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51j_observed_horizon_recovery_partial_horizon_missing",
                "approved_for_live": False,
                "approved_for_model_training": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
                "label_count": 2,
                "recovery_status_counts": {
                    "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE": 1,
                    "RECOVERED_TERMINAL_SOURCE_TICKS": 1,
                },
            }), encoding="utf-8")
            with (horizon_recovery_run / "observed_horizon_recovery_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, (group_id, order_key, venue, side, status, effective) in enumerate([
                    ("group-filled", "canonical-filled", "lighter", "BID", "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE", None),
                    ("group-not-filled", "canonical-not-filled", "aster", "ASK", "RECOVERED_TERMINAL_SOURCE_TICKS", 3),
                ], start=1):
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "PHASE51J_OBSERVED_HORIZON_RECOVERY_LABEL",
                        "label_seq": seq,
                        "run_id": "phase51j_recovery",
                        "baseline_commit": baseline,
                        "gate_status": "HOLD",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "approved_for_canary": False,
                        "approved_for_capital_escalation": False,
                        "admissible_for_financial_claim": False,
                        "admissible_for_ev_admission": False,
                        "no_live_flag": True,
                        "live_orders_allowed": False,
                        "capital_change_allowed": False,
                        "risk_limit_relaxation_allowed": False,
                        "raw_identifier_redaction_status": "PASS",
                        "canonical_group_id": group_id,
                        "canonical_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": venue,
                        "side": side,
                        "recovery_status": status,
                        "effective_observed_horizon_source_ticks": effective,
                    }) + "\n")

            (filled_horizon_recovery_run / "filled_horizon_timebase_recovery_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51k_recovery",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51k_filled_horizon_source_tick_complete_nonlive_hold",
                "approved_for_live": False,
                "approved_for_model_training": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
                "label_count": 2,
                "recovery_status_counts": {
                    "RECOVERED_SOURCE_TICKS": 1,
                    "PRESERVED_EXISTING_SOURCE_TICKS": 1,
                },
            }), encoding="utf-8")
            with (filled_horizon_recovery_run / "filled_horizon_timebase_recovery_labels.jsonl").open(
                "w",
                encoding="utf-8",
            ) as f:
                for seq, (group_id, order_key, venue, side, status, effective) in enumerate([
                    ("group-filled", "canonical-filled", "lighter", "BID", "RECOVERED_SOURCE_TICKS", 7),
                    ("group-not-filled", "canonical-not-filled", "aster", "ASK", "PRESERVED_EXISTING_SOURCE_TICKS", 3),
                ], start=1):
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "PHASE51K_FILLED_HORIZON_TIMEBASE_RECOVERY_LABEL",
                        "label_seq": seq,
                        "run_id": "phase51k_recovery",
                        "baseline_commit": baseline,
                        "gate_status": "HOLD",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "approved_for_canary": False,
                        "approved_for_capital_escalation": False,
                        "admissible_for_financial_claim": False,
                        "admissible_for_ev_admission": False,
                        "no_live_flag": True,
                        "live_orders_allowed": False,
                        "capital_change_allowed": False,
                        "risk_limit_relaxation_allowed": False,
                        "raw_identifier_redaction_status": "PASS",
                        "canonical_group_id": group_id,
                        "canonical_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": venue,
                        "side": side,
                        "recovery_status": status,
                        "recovery_timebase": "SOURCE_TICKS",
                        "effective_observed_horizon_source_ticks": effective,
                        "recovered_observed_horizon_exchange_ms": None,
                    }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51h_observed_pfill_feature_audit_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--quarantine-review-run",
                    str(quarantine_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--queue-churn-run",
                    str(queue_run),
                    "--markout-readiness-run",
                    str(markout_run),
                    "--horizon-recovery-run",
                    str(horizon_recovery_run),
                    "--filled-horizon-recovery-run",
                    str(filled_horizon_recovery_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51h_feature_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--min-observed-per-bucket",
                    "1",
                    "--min-holdout-observed-per-bucket",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51h_feature_test"
            summary = json.loads((run_dir / "pfill_feature_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["label_count"], 2)
            self.assertEqual(summary["queue_churn_joined_all_count"], 2)
            self.assertEqual(summary["native_limit_partial_count"], 1)
            self.assertEqual(summary["markout_source_available_count"], 2)
            self.assertEqual(summary["raw_identifier_input_present_count"], 1)
            self.assertEqual(summary["horizon_recovery_applied_count"], 1)
            self.assertEqual(summary["horizon_recovered_terminal_count"], 1)
            self.assertEqual(summary["filled_horizon_recovery_applied_count"], 1)
            self.assertEqual(summary["filled_horizon_recovered_source_tick_count"], 1)
            self.assertEqual(summary["filled_horizon_unrecovered_count"], 0)
            self.assertEqual(summary["excluded_quarantine_count"], 4)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["admissible_for_ev_admission"])

            labels = [
                json.loads(line)
                for line in (run_dir / "pfill_feature_coverage_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(labels), 2)
            self.assertTrue(labels[0]["raw_identifier_input_present"])
            self.assertNotIn("decision_id", labels[0])
            self.assertNotIn("raw-decision-id-must-not-emit", json.dumps(labels))
            self.assertEqual(labels[0]["queue_churn_join_status"], "JOINED_ALL_SOURCE_KEYS")
            self.assertEqual(labels[0]["native_limit_pressure_status"], "PARTIAL")
            self.assertTrue(labels[0]["filled_horizon_recovery_applied"])
            self.assertEqual(labels[0]["observed_horizon_source_ticks"], 7)
            self.assertEqual(labels[1]["maker_taker_feature_status"], "NO_FILL_NOT_APPLICABLE")
            self.assertTrue(labels[1]["horizon_recovery_applied"])
            self.assertEqual(labels[1]["observed_horizon_source_ticks"], 3)

            buckets = [
                json.loads(line)
                for line in (run_dir / "pfill_feature_bucket_readiness.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            global_bucket = next(row for row in buckets if row["bucket_id"] == "GLOBAL")
            self.assertIn("raw_identifier_present_in_input_not_emitted", global_bucket["gate_reasons"])

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            with (queue_run / "queue_churn_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, (order_key, venue, native_status, replace_count) in enumerate(queue_labels[:2], start=1):
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "QUEUE_CHURN_LABEL",
                        "label_seq": seq,
                        "run_id": "queue_run",
                        "baseline_commit": baseline,
                        "source_telemetry_sha256": source_sha,
                        "order_key": order_key,
                        "venue_id": venue,
                        "side": "BID",
                        "lifecycle_join_status": "JOINED",
                        "queue_reset_proxy_event_count": replace_count,
                        "replace_event_count": replace_count,
                        "cancel_event_count": 1,
                        "cancel_all_event_count": 0,
                        "churn_event_count": replace_count + 1,
                        "native_limit_pressure_status": native_status,
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "admissible_for_ev_admission": False,
                    }) + "\n")
            missing_queue = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51h_observed_pfill_feature_audit_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--quarantine-review-run",
                    str(quarantine_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--queue-churn-run",
                    str(queue_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51h_missing_queue_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing_queue.returncode, 2)
            self.assertIn("missing queue/churn rows", missing_queue.stderr)

            pfill_labels[0]["approved_for_live"] = True
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51h_observed_pfill_feature_audit_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--quarantine-review-run",
                    str(quarantine_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--queue-churn-run",
                    str(queue_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51h_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51j_observed_horizon_recovery_recovers_terminal_horizons_hold_only(self):
        """Observed-horizon recovery should recover terminal source ticks without emitting raw IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_run = tmp_path / "phase51h"
            canonical_run = tmp_path / "canonical"
            lifecycle_run = tmp_path / "lifecycle_truth"
            label_lake_run = tmp_path / "label_lake"
            output_root = tmp_path / "phase51j"
            for path in [feature_run, canonical_run, lifecycle_run, label_lake_run]:
                path.mkdir()
            source_sha = "source-sha-phase51j"
            baseline = "18dd09512288a85e440d3977e32432c3aabc1190"
            group_terminal = f"{source_sha}:order_label_seq:101"
            group_existing = f"{source_sha}:order_label_seq:201"
            group_fill = f"{source_sha}:order_label_seq:301"

            feature_summary = {
                "schema_version": 1,
                "run_id": "phase51h_redacted",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51h_missing_observed_horizon_features",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "source_telemetry_sha256_list": [source_sha],
                "raw_identifier_input_present_count": 0,
                "label_count": 3,
                "observed_horizon_available_count": 1,
                "observed_horizon_missing_count": 2,
            }
            (feature_run / "pfill_feature_audit_summary.json").write_text(json.dumps(feature_summary), encoding="utf-8")
            feature_labels = [
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                    "label_seq": 1,
                    "run_id": "phase51h_redacted",
                    "baseline_commit": baseline,
                    "gate_status": "HOLD",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "canonical_group_id": group_terminal,
                    "canonical_order_key": "canonical-terminal",
                    "source_telemetry_sha256": source_sha,
                    "venue_id": "lighter",
                    "side": "BID",
                    "order_holdout_split": "TRAIN",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "observed_horizon_source_ticks": None,
                    "raw_identifier_input_present": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                    "label_seq": 2,
                    "run_id": "phase51h_redacted",
                    "baseline_commit": baseline,
                    "gate_status": "HOLD",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "canonical_group_id": group_existing,
                    "canonical_order_key": "canonical-existing",
                    "source_telemetry_sha256": source_sha,
                    "venue_id": "aster",
                    "side": "ASK",
                    "order_holdout_split": "HOLDOUT",
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "observed_horizon_source_ticks": 2,
                    "raw_identifier_input_present": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                    "label_seq": 3,
                    "run_id": "phase51h_redacted",
                    "baseline_commit": baseline,
                    "gate_status": "HOLD",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "canonical_group_id": group_fill,
                    "canonical_order_key": "canonical-fill",
                    "source_telemetry_sha256": source_sha,
                    "venue_id": "lighter",
                    "side": "BID",
                    "order_holdout_split": "TRAIN",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "observed_horizon_source_ticks": None,
                    "raw_identifier_input_present": False,
                },
            ]
            with (feature_run / "pfill_feature_coverage_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in feature_labels:
                    f.write(json.dumps(label) + "\n")

            (canonical_run / "canonical_pfill_outcome_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "canonical",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (canonical_run / "source_to_canonical_order_manifest.jsonl").open("w", encoding="utf-8") as f:
                for group_id, order_key in [
                    (group_terminal, "source-terminal"),
                    (group_existing, "source-existing"),
                    (group_fill, "source-fill"),
                ]:
                    f.write(json.dumps({
                        "canonical_group_id": group_id,
                        "canonical_order_key": f"canonical-{order_key}",
                        "source_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                    }) + "\n")

            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "label_lake",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            lifecycle_labels = [
                (101, "place", "intent", 10, "raw-decision-terminal"),
                (102, "cancel", "ack", 13, "raw-decision-terminal"),
                (201, "place", "intent", 20, "raw-decision-existing"),
                (202, "cancel", "ack", 22, "raw-decision-existing"),
                (301, "place", "intent", 30, "raw-decision-fill"),
            ]
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, action, status, source_t, decision_id in lifecycle_labels:
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "ORDER_LIFECYCLE_LABEL",
                        "label_seq": seq,
                        "run_id": "label_lake",
                        "baseline_commit": baseline,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": "lighter",
                        "action": action,
                        "status": status,
                        "source_t": source_t,
                        "decision_id": decision_id,
                        "client_order_id_hash": hashlib.sha256(decision_id.encode("utf-8")).hexdigest(),
                        "approved_for_model_training": False,
                        "approved_for_live": False,
                        "approved_for_canary": False,
                        "admissible_for_ev_admission": False,
                    }) + "\n")
            labels_sha = hashlib.sha256((label_lake_run / "labels.jsonl").read_bytes()).hexdigest()

            (lifecycle_run / "lifecycle_truth_audit_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "lifecycle_truth",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51e_test",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
                "source_inputs": [
                    {
                        "source_telemetry_sha256": source_sha,
                        "label_lake_run": str(label_lake_run),
                        "label_lake_labels_sha256": labels_sha,
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51j_observed_horizon_recovery_path()),
                    "--feature-audit-run",
                    str(feature_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--lifecycle-truth-run",
                    str(lifecycle_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51j_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51j_recovery_test"
            summary = json.loads((run_dir / "observed_horizon_recovery_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(summary["label_count"], 3)
            self.assertEqual(summary["input_observed_horizon_missing_count"], 2)
            self.assertEqual(summary["recovered_terminal_horizon_count"], 1)
            self.assertEqual(summary["effective_observed_horizon_missing_count"], 1)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["admissible_for_ev_admission"])

            labels = [
                json.loads(line)
                for line in (run_dir / "observed_horizon_recovery_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_group = {label["canonical_group_id"]: label for label in labels}
            self.assertEqual(by_group[group_terminal]["recovery_status"], "RECOVERED_TERMINAL_SOURCE_TICKS")
            self.assertEqual(by_group[group_terminal]["effective_observed_horizon_source_ticks"], 3)
            self.assertEqual(by_group[group_existing]["recovery_status"], "PRESERVED_EXISTING_OBSERVED_HORIZON")
            self.assertEqual(by_group[group_fill]["recovery_status"], "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE")
            self.assertNotIn("decision_id", json.dumps(labels))
            self.assertNotIn("raw-decision-terminal", json.dumps(labels))

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            feature_labels[0]["decision_id"] = "raw-id-blocks-input"
            with (feature_run / "pfill_feature_coverage_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in feature_labels:
                    f.write(json.dumps(label) + "\n")
            raw_id = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51j_observed_horizon_recovery_path()),
                    "--feature-audit-run",
                    str(feature_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--lifecycle-truth-run",
                    str(lifecycle_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51j_raw_id_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw_id.returncode, 2)
            self.assertIn("raw identifier field", raw_id.stderr)

    def test_phase51k_filled_horizon_timebase_recovery_recovers_source_ticks_hold_only(self):
        """Filled-horizon recovery should use source ticks and avoid emitting raw fill IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_run = tmp_path / "phase51h"
            canonical_run = tmp_path / "canonical"
            lifecycle_run = tmp_path / "lifecycle_truth"
            label_lake_run = tmp_path / "label_lake"
            join_run = tmp_path / "join_holdout"
            observed_run = tmp_path / "observed_labels"
            output_root = tmp_path / "phase51k"
            for path in [feature_run, canonical_run, lifecycle_run, label_lake_run, join_run, observed_run]:
                path.mkdir()
            source_sha = "source-sha-phase51k"
            baseline = "18dd09512288a85e440d3977e32432c3aabc1190"
            group_filled = f"{source_sha}:order_label_seq:101"
            group_not_filled = f"{source_sha}:order_label_seq:201"
            group_existing = f"{source_sha}:order_label_seq:301"

            (feature_run / "pfill_feature_audit_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51h_redacted",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51h_missing_observed_horizon_features",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_input_present_count": 0,
                "label_count": 3,
            }), encoding="utf-8")
            feature_labels = [
                (1, group_filled, "canonical-filled", "OBSERVED_FILLED", 1.0, None),
                (2, group_not_filled, "canonical-not-filled", "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL", 0.0, None),
                (3, group_existing, "canonical-existing", "OBSERVED_FILLED", 1.0, 2),
            ]
            with (feature_run / "pfill_feature_coverage_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, group_id, order_key, outcome, pfill, horizon in feature_labels:
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                        "label_seq": seq,
                        "run_id": "phase51h_redacted",
                        "baseline_commit": baseline,
                        "gate_status": "HOLD",
                        "approved_for_model_training": False,
                        "approved_for_live": False,
                        "approved_for_canary": False,
                        "approved_for_capital_escalation": False,
                        "admissible_for_financial_claim": False,
                        "admissible_for_ev_admission": False,
                        "no_live_flag": True,
                        "live_orders_allowed": False,
                        "capital_change_allowed": False,
                        "risk_limit_relaxation_allowed": False,
                        "canonical_group_id": group_id,
                        "canonical_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": "lighter",
                        "side": "BID",
                        "order_holdout_split": "TRAIN",
                        "outcome_status": outcome,
                        "p_fill_outcome": pfill,
                        "observed_horizon_source_ticks": horizon,
                        "raw_identifier_input_present": False,
                    }) + "\n")

            (canonical_run / "canonical_pfill_outcome_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "canonical",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "order_label_count": 3,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (canonical_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, group_id, order_key, order_source_t, fill_count in [
                    (101, group_filled, "canonical-filled", 10, 1),
                    (201, group_not_filled, "canonical-not-filled", 20, 0),
                    (301, group_existing, "canonical-existing", 30, 1),
                ]:
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "label_seq": seq,
                        "run_id": "canonical",
                        "baseline_commit": baseline,
                        "canonical_group_id": group_id,
                        "order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "order_label_seq": seq,
                        "order_source_t": order_source_t,
                        "fill_count": fill_count,
                        "approved_for_model_training": False,
                        "approved_for_live": False,
                        "approved_for_canary": False,
                        "admissible_for_ev_admission": False,
                    }) + "\n")

            (label_lake_run / "labels.jsonl").write_text("", encoding="utf-8")
            labels_sha = hashlib.sha256((label_lake_run / "labels.jsonl").read_bytes()).hexdigest()
            (lifecycle_run / "lifecycle_truth_audit_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "lifecycle_truth",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51e_test",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
                "source_inputs": [
                    {
                        "source_telemetry_sha256": source_sha,
                        "label_lake_run": str(label_lake_run),
                        "label_lake_labels_sha256": labels_sha,
                        "join_holdout_run": str(join_run),
                    }
                ],
            }), encoding="utf-8")

            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "join_holdout",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51c_test_join",
                "source_telemetry_sha256": source_sha,
                "observed_run": str(observed_run),
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            raw_fill_id = "raw-fill-id-must-not-emit"
            with (join_run / "joined_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "schema_version": 1,
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "label_seq": 1,
                    "run_id": "join_holdout",
                    "baseline_commit": baseline,
                    "order_label_seq": 101,
                    "fill_id": raw_fill_id,
                    "fill_time_ms": 1710000000001,
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "admissible_for_ev_admission": False,
                }) + "\n")
            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "observed_labels",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "observed_label_pack_test",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "schema_version": 1,
                    "label_type": "OBSERVED_FILL_LABEL",
                    "label_seq": 1,
                    "run_id": "observed_labels",
                    "baseline_commit": baseline,
                    "fill_id": raw_fill_id,
                    "source_t": 13,
                    "decision_id": "raw-decision-must-not-emit",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "admissible_for_ev_admission": False,
                }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51k_filled_horizon_timebase_recovery_path()),
                    "--feature-audit-run",
                    str(feature_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--lifecycle-truth-run",
                    str(lifecycle_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51k_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51k_recovery_test"
            summary = json.loads((run_dir / "filled_horizon_timebase_recovery_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(summary["input_filled_missing_horizon_count"], 1)
            self.assertEqual(summary["recovered_source_tick_count"], 1)
            self.assertEqual(summary["still_missing_filled_horizon_count"], 0)

            output_text = (run_dir / "filled_horizon_timebase_recovery_labels.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(raw_fill_id, output_text)
            self.assertNotIn("raw-decision-must-not-emit", output_text)
            labels = [json.loads(line) for line in output_text.splitlines() if line.strip()]
            by_group = {label["canonical_group_id"]: label for label in labels}
            self.assertEqual(by_group[group_filled]["recovery_status"], "RECOVERED_SOURCE_TICKS")
            self.assertEqual(by_group[group_filled]["effective_observed_horizon_source_ticks"], 3)
            self.assertEqual(by_group[group_not_filled]["recovery_status"], "NOT_FILLED_NOT_APPLICABLE")
            self.assertEqual(by_group[group_existing]["recovery_status"], "PRESERVED_EXISTING_SOURCE_TICKS")
            self.assertEqual(by_group[group_existing]["effective_observed_horizon_source_ticks"], 2)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

    def test_phase51l_filled_horizon_source_key_recovery_recovers_source_and_hash_paths_hold_only(self):
        """Source-key recovery should recover 5.1k MISSING_JOIN rows without raw identifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            phase51k_run = tmp_path / "phase51k"
            canonical_run = tmp_path / "canonical"
            source_pfill_run = tmp_path / "source_pfill"
            observed_run = tmp_path / "observed_labels"
            output_root = tmp_path / "phase51l"
            for path in [phase51k_run, canonical_run, source_pfill_run, observed_run]:
                path.mkdir()
            source_sha = "source-sha-phase51l"
            baseline = "18dd09512288a85e440d3977e32432c3aabc1190"
            group_source = f"{source_sha}:order_label_seq:101"
            group_hash = f"{source_sha}:order_label_seq:201"
            group_existing = f"{source_sha}:order_label_seq:301"
            group_not_filled = f"{source_sha}:order_label_seq:401"

            (phase51k_run / "filled_horizon_timebase_recovery_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "phase51k_recovery",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51k_filled_horizon_timebase_partial",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
                "label_count": 4,
            }), encoding="utf-8")
            phase51k_rows = [
                (1, group_source, "canonical-source", "OBSERVED_FILLED", 1.0, "MISSING_JOIN", None, 10, "aster"),
                (2, group_hash, "canonical-hash", "OBSERVED_FILLED", 1.0, "MISSING_JOIN", None, 20, "paradex"),
                (3, group_existing, "canonical-existing", "OBSERVED_FILLED", 1.0, "PRESERVED_EXISTING_SOURCE_TICKS", 2, 30, "extended"),
                (4, group_not_filled, "canonical-not-filled", "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL", 0.0, "PRESERVED_EXISTING_SOURCE_TICKS", 3, 40, "aster"),
            ]
            with (phase51k_run / "filled_horizon_timebase_recovery_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, group_id, order_key, outcome, pfill, status, effective, order_source_t, venue in phase51k_rows:
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "PHASE51K_FILLED_HORIZON_TIMEBASE_RECOVERY_LABEL",
                        "label_seq": seq,
                        "run_id": "phase51k_recovery",
                        "baseline_commit": baseline,
                        "gate_status": "HOLD",
                        "approved_for_model_training": False,
                        "approved_for_live": False,
                        "approved_for_canary": False,
                        "approved_for_capital_escalation": False,
                        "admissible_for_financial_claim": False,
                        "admissible_for_ev_admission": False,
                        "no_live_flag": True,
                        "live_orders_allowed": False,
                        "capital_change_allowed": False,
                        "risk_limit_relaxation_allowed": False,
                        "raw_identifier_redaction_status": "PASS",
                        "canonical_group_id": group_id,
                        "canonical_order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": venue,
                        "side": "BID",
                        "order_holdout_split": "TRAIN",
                        "outcome_status": outcome,
                        "p_fill_outcome": pfill,
                        "fill_count": 1 if pfill == 1.0 else 0,
                        "order_source_t": order_source_t,
                        "recovery_status": status,
                        "recovery_timebase": "SOURCE_TICKS" if effective is not None else "NONE",
                        "effective_observed_horizon_source_ticks": effective,
                    }) + "\n")

            (canonical_run / "canonical_pfill_outcome_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "canonical",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "order_label_count": 4,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            canonical_rows = [
                (101, group_source, "canonical-source", 10, ["source-key-a"], None, None, "aster"),
                (201, group_hash, "canonical-hash", 20, ["source-key-b"], None, None, "paradex"),
                (301, group_existing, "canonical-existing", 30, ["source-key-c"], None, None, "extended"),
                (401, group_not_filled, "canonical-not-filled", 40, ["source-key-d"], None, None, "aster"),
            ]
            with (canonical_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for seq, group_id, order_key, order_source_t, source_keys, order_hash, client_hash, venue in canonical_rows:
                    f.write(json.dumps({
                        "schema_version": 1,
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "label_seq": seq,
                        "run_id": "canonical",
                        "baseline_commit": baseline,
                        "canonical_group_id": group_id,
                        "order_key": order_key,
                        "source_telemetry_sha256": source_sha,
                        "venue_id": venue,
                        "order_source_t": order_source_t,
                        "source_order_keys": source_keys,
                        "order_id_hash": order_hash,
                        "client_order_id_hash": client_hash,
                        "approved_for_model_training": False,
                        "approved_for_live": False,
                        "approved_for_canary": False,
                        "admissible_for_ev_admission": False,
                    }) + "\n")

            (source_pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "source_pfill",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "pfill_outcome_sparse_observed_fills",
                "source_telemetry_sha256": source_sha,
                "order_label_count": 2,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            raw_decision_id = "raw-decision-id-must-not-emit-phase51l"
            source_labels = [
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 1,
                    "run_id": "source_pfill",
                    "baseline_commit": baseline,
                    "source_telemetry_sha256": source_sha,
                    "order_key": "source-key-a",
                    "venue_id": "aster",
                    "side": "Buy",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "fill_count": 1,
                    "order_source_t": 12,
                    "observed_horizon_source_ticks": 3,
                    "decision_id": raw_decision_id,
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "admissible_for_ev_admission": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": 2,
                    "run_id": "source_pfill",
                    "baseline_commit": baseline,
                    "source_telemetry_sha256": source_sha,
                    "order_key": "source-key-b",
                    "venue_id": "paradex",
                    "side": "Buy",
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "fill_count": 1,
                    "order_source_t": 21,
                    "observed_horizon_source_ticks": None,
                    "order_id_hash": "hash-order-b",
                    "client_order_id_hash": "hash-client-b",
                    "decision_id": "raw-decision-id-must-not-emit-hash",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "admissible_for_ev_admission": False,
                },
            ]
            with (source_pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in source_labels:
                    f.write(json.dumps(label) + "\n")

            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "observed_labels",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "observed_label_pack_partial_maker_taker_attribution",
                "source_telemetry_sha256": source_sha,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "admissible_for_ev_admission": False,
            }), encoding="utf-8")
            raw_fill_id = "raw-fill-id-must-not-emit-phase51l"
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "schema_version": 1,
                    "label_type": "OBSERVED_FILL_LABEL",
                    "label_seq": 1,
                    "run_id": "observed_labels",
                    "baseline_commit": baseline,
                    "venue_id": "paradex",
                    "order_id_hash": "hash-order-b",
                    "client_order_id_hash": "hash-client-b",
                    "fill_id": raw_fill_id,
                    "decision_id": "raw-observed-decision-must-not-emit",
                    "source_t": 26,
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "admissible_for_ev_admission": False,
                }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51l_filled_horizon_source_key_recovery_path()),
                    "--phase51k-recovery-run",
                    str(phase51k_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--source-pfill-run",
                    str(source_pfill_run),
                    "--observed-label-run",
                    str(observed_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51l_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51l_recovery_test"
            summary = json.loads((run_dir / "filled_horizon_source_key_recovery_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(summary["target_missing_join_count"], 2)
            self.assertEqual(summary["source_pfill_horizon_recovered_count"], 1)
            self.assertEqual(summary["observed_fill_hash_recovered_count"], 1)
            self.assertEqual(summary["recovered_source_tick_count"], 2)
            self.assertEqual(summary["still_missing_filled_horizon_count"], 0)
            self.assertEqual(summary["gate_reason"], "phase51l_filled_horizon_source_key_complete_nonlive_hold")

            output_text = (run_dir / "filled_horizon_source_key_recovery_labels.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(raw_decision_id, output_text)
            self.assertNotIn(raw_fill_id, output_text)
            self.assertNotIn("decision_id", output_text)
            self.assertNotIn("fill_id", output_text)
            labels = [json.loads(line) for line in output_text.splitlines() if line.strip()]
            by_group = {label["canonical_group_id"]: label for label in labels}
            self.assertEqual(by_group[group_source]["recovery_status"], "RECOVERED_FROM_SOURCE_PFILL_HORIZON")
            self.assertEqual(by_group[group_source]["effective_observed_horizon_source_ticks"], 5)
            self.assertEqual(by_group[group_hash]["recovery_status"], "RECOVERED_FROM_OBSERVED_FILL_HASH")
            self.assertEqual(by_group[group_hash]["effective_observed_horizon_source_ticks"], 6)
            self.assertEqual(by_group[group_existing]["recovery_status"], "PRESERVED_EXISTING_SOURCE_TICKS")
            self.assertEqual(by_group[group_not_filled]["recovery_status"], "PRESERVED_EXISTING_SOURCE_TICKS")

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

    def test_phase51i_pfill_feature_matrix_admissibility_holds_redacted_matrix(self):
        """Feature-matrix admissibility should require redacted input and remain HOLD-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feature_run = tmp_path / "phase51h"
            output_root = tmp_path / "phase51i"
            feature_run.mkdir()
            baseline = "18dd09512288a85e440d3977e32432c3aabc1190"
            summary = {
                "schema_version": 1,
                "run_id": "phase51h_redacted",
                "baseline_commit": baseline,
                "gate_status": "HOLD",
                "gate_reason": "phase51h_missing_observed_horizon_features",
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "no_live_flag": True,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "source_telemetry_sha256_list": ["source-sha-phase51i"],
                "excluded_quarantine_count": 4,
                "excluded_quarantine_reason_counts": {
                    "EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL": 4,
                },
                "observed_only_pack_warning": "diagnostic only; excluded censored groups are not negative outcomes",
                "bucket_count": 2,
                "label_count": 2,
                "filled_count": 1,
                "not_filled_count": 1,
                "train_count": 1,
                "holdout_count": 1,
                "observed_horizon_available_count": 1,
                "observed_horizon_missing_count": 1,
                "queue_churn_joined_all_count": 2,
                "queue_churn_joined_partial_count": 0,
                "queue_churn_missing_count": 0,
                "queue_reset_proxy_present_count": 1,
                "native_limit_observed_count": 0,
                "native_limit_partial_count": 1,
                "native_limit_unknown_count": 0,
                "native_limit_not_applicable_count": 1,
                "maker_taker_observed_count": 0,
                "maker_taker_partial_or_unknown_count": 1,
                "maker_taker_missing_count": 0,
                "maker_taker_not_applicable_count": 1,
                "markout_source_available_count": 2,
                "markout_source_missing_count": 0,
                "raw_identifier_input_present_count": 0,
                "horizon_recovery_run": "/tmp/phase51j_recovery",
                "horizon_recovery_status_counts": {
                    "RECOVERED_TERMINAL_SOURCE_TICKS": 1,
                    "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE": 1,
                },
                "horizon_recovery_applied_count": 1,
                "horizon_recovered_terminal_count": 1,
                "horizon_recovery_preserved_existing_count": 0,
                "horizon_recovery_fill_timebase_remaining_count": 1,
                "filled_horizon_recovery_run": "/tmp/phase51k_recovery",
                "filled_horizon_recovery_status_counts": {
                    "MISSING_JOIN": 1,
                    "PRESERVED_EXISTING_SOURCE_TICKS": 1,
                },
                "filled_horizon_recovery_applied_count": 0,
                "filled_horizon_recovered_source_tick_count": 0,
                "filled_horizon_exchange_ms_only_count": 0,
                "filled_horizon_unrecovered_count": 1,
                "filled_horizon_source_key_recovery_run": "/tmp/phase51l_recovery",
                "filled_horizon_source_key_recovery_status_counts": {
                    "NO_OBSERVED_FILL_HASH_MATCH": 1,
                    "PRESERVED_EXISTING_SOURCE_TICKS": 1,
                },
                "filled_horizon_source_key_recovery_applied_count": 0,
                "filled_horizon_source_key_recovered_source_tick_count": 0,
                "filled_horizon_source_key_pfill_horizon_recovered_count": 0,
                "filled_horizon_source_key_observed_hash_recovered_count": 0,
                "filled_horizon_source_key_unrecovered_count": 1,
                "missing_feature_total": 3,
            }
            (feature_run / "pfill_feature_audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            labels = [
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                    "label_seq": 1,
                    "baseline_commit": baseline,
                    "gate_status": "HOLD",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "canonical_group_id": "group-filled",
                    "venue_id": "lighter",
                    "side": "BID",
                    "p_fill_outcome": 1.0,
                    "raw_identifier_input_present": False,
                },
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
                    "label_seq": 2,
                    "baseline_commit": baseline,
                    "gate_status": "HOLD",
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "canonical_group_id": "group-not-filled",
                    "venue_id": "aster",
                    "side": "ASK",
                    "p_fill_outcome": 0.0,
                    "raw_identifier_input_present": False,
                },
            ]
            with (feature_run / "pfill_feature_coverage_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            buckets = [
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_BUCKET_READINESS",
                    "bucket_seq": 1,
                    "baseline_commit": baseline,
                    "bucket_id": "GLOBAL",
                    "bucket_dimensions": {"scope": "GLOBAL"},
                    "gate_status": "HOLD",
                    "gate_reasons": [
                        "missing_observed_horizon_features",
                        "filled_horizon_source_key_still_missing",
                        "filled_horizon_source_tick_still_missing",
                        "lighter_native_limit_pressure_not_fully_observed",
                        "maker_taker_not_fully_observed_for_filled_orders",
                        "requires_feature_rich_pfill_model_and_board_review",
                    ],
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "label_count": 2,
                    "filled_count": 1,
                    "not_filled_count": 1,
                    "train_count": 1,
                    "holdout_count": 1,
                    "observed_horizon_available_count": 1,
                    "observed_horizon_missing_count": 1,
                    "queue_churn_joined_all_count": 2,
                    "queue_churn_joined_partial_count": 0,
                    "queue_churn_missing_count": 0,
                    "queue_reset_proxy_present_count": 1,
                    "native_limit_observed_count": 0,
                    "native_limit_partial_count": 1,
                    "native_limit_unknown_count": 0,
                    "native_limit_not_applicable_count": 1,
                    "maker_taker_observed_count": 0,
                    "maker_taker_partial_or_unknown_count": 1,
                    "maker_taker_missing_count": 0,
                    "maker_taker_not_applicable_count": 1,
                    "markout_source_available_count": 2,
                    "markout_source_missing_count": 0,
                    "raw_identifier_input_present_count": 0,
                    "horizon_recovery_applied_count": 1,
                    "horizon_recovered_terminal_count": 1,
                    "horizon_recovery_preserved_existing_count": 0,
                    "horizon_recovery_fill_timebase_remaining_count": 1,
                    "filled_horizon_recovery_applied_count": 0,
                    "filled_horizon_recovered_source_tick_count": 0,
                    "filled_horizon_exchange_ms_only_count": 0,
                    "filled_horizon_unrecovered_count": 1,
                    "filled_horizon_source_key_recovery_applied_count": 0,
                    "filled_horizon_source_key_recovered_source_tick_count": 0,
                    "filled_horizon_source_key_pfill_horizon_recovered_count": 0,
                    "filled_horizon_source_key_observed_hash_recovered_count": 0,
                    "filled_horizon_source_key_unrecovered_count": 1,
                    "missing_feature_total": 3,
                },
                {
                    "schema_version": 1,
                    "label_type": "PHASE51H_PFILL_FEATURE_BUCKET_READINESS",
                    "bucket_seq": 2,
                    "baseline_commit": baseline,
                    "bucket_id": "VENUE:aster",
                    "bucket_dimensions": {"venue_id": "aster"},
                    "gate_status": "HOLD",
                    "gate_reasons": [
                        "sparse_pfill_feature_bucket",
                        "requires_feature_rich_pfill_model_and_board_review",
                    ],
                    "approved_for_model_training": False,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                    "approved_for_capital_escalation": False,
                    "admissible_for_financial_claim": False,
                    "admissible_for_ev_admission": False,
                    "no_live_flag": True,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                    "label_count": 1,
                    "filled_count": 0,
                    "not_filled_count": 1,
                    "train_count": 0,
                    "holdout_count": 1,
                    "observed_horizon_available_count": 1,
                    "observed_horizon_missing_count": 0,
                    "queue_churn_joined_all_count": 1,
                    "queue_churn_joined_partial_count": 0,
                    "queue_churn_missing_count": 0,
                    "queue_reset_proxy_present_count": 0,
                    "native_limit_observed_count": 0,
                    "native_limit_partial_count": 0,
                    "native_limit_unknown_count": 0,
                    "native_limit_not_applicable_count": 1,
                    "maker_taker_observed_count": 0,
                    "maker_taker_partial_or_unknown_count": 0,
                    "maker_taker_missing_count": 0,
                    "maker_taker_not_applicable_count": 1,
                    "markout_source_available_count": 1,
                    "markout_source_missing_count": 0,
                    "raw_identifier_input_present_count": 0,
                    "horizon_recovery_applied_count": 1,
                    "horizon_recovered_terminal_count": 1,
                    "horizon_recovery_preserved_existing_count": 0,
                    "horizon_recovery_fill_timebase_remaining_count": 0,
                    "filled_horizon_recovery_applied_count": 0,
                    "filled_horizon_recovered_source_tick_count": 0,
                    "filled_horizon_exchange_ms_only_count": 0,
                    "filled_horizon_unrecovered_count": 0,
                    "filled_horizon_source_key_recovery_applied_count": 0,
                    "filled_horizon_source_key_recovered_source_tick_count": 0,
                    "filled_horizon_source_key_pfill_horizon_recovered_count": 0,
                    "filled_horizon_source_key_observed_hash_recovered_count": 0,
                    "filled_horizon_source_key_unrecovered_count": 0,
                    "missing_feature_total": 0,
                },
            ]
            with (feature_run / "pfill_feature_bucket_readiness.jsonl").open("w", encoding="utf-8") as f:
                for bucket in buckets:
                    f.write(json.dumps(bucket) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51i_pfill_feature_matrix_admissibility_path()),
                    "--feature-audit-run",
                    str(feature_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51i_matrix_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51i_matrix_test"
            matrix_summary = json.loads((run_dir / "pfill_feature_matrix_admissibility_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix_summary["gate_status"], "HOLD")
            self.assertEqual(matrix_summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(matrix_summary["label_count"], 2)
            self.assertEqual(matrix_summary["observed_horizon_missing_count"], 1)
            self.assertEqual(matrix_summary["gate_reason"], "phase51i_filled_horizon_source_key_still_missing")
            self.assertEqual(matrix_summary["horizon_recovery_applied_count"], 1)
            self.assertEqual(matrix_summary["horizon_recovered_terminal_count"], 1)
            self.assertEqual(matrix_summary["filled_horizon_unrecovered_count"], 1)
            self.assertEqual(matrix_summary["filled_horizon_source_key_unrecovered_count"], 1)
            self.assertIn("filled_horizon_source_key_still_missing", matrix_summary["matrix_blocker_ids"])
            self.assertIn("filled_horizon_source_tick_still_missing", matrix_summary["matrix_blocker_ids"])
            self.assertIn("missing_observed_horizon_features", matrix_summary["matrix_blocker_ids"])
            self.assertIn("observed_only_selection_bias_not_resolved", matrix_summary["matrix_blocker_ids"])
            self.assertEqual(matrix_summary["excluded_quarantine_count"], 4)
            self.assertFalse(matrix_summary["approved_for_model_training"])
            self.assertFalse(matrix_summary["admissible_for_ev_admission"])

            blockers = [
                json.loads(line)
                for line in (run_dir / "pfill_feature_matrix_blockers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(blockers[0]["blocker_id"], "raw_identifier_redaction_passed")
            self.assertEqual(blockers[0]["gate_status"], "PASS")
            self.assertTrue(all(blocker["approved_for_model_training"] is False for blocker in blockers))
            self.assertNotIn("decision_id", json.dumps(blockers))

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            labels[0]["decision_id"] = "raw-id-blocks-input"
            with (feature_run / "pfill_feature_coverage_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            raw_id = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51i_pfill_feature_matrix_admissibility_path()),
                    "--feature-audit-run",
                    str(feature_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51i_raw_id_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw_id.returncode, 2)
            self.assertIn("raw identifier field", raw_id.stderr)

    def test_phase51c_markout_calibration_readiness_preserves_fill_splits_and_stats(self):
        """Markout readiness should inherit join splits and stay HOLD-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed"
            join_run = tmp_path / "join"
            output_root = tmp_path / "markout_readiness"
            observed_run.mkdir()
            join_run.mkdir()
            source_sha = "source-sha-markout"
            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "run_id": "observed_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "observed_label_pack_partial_maker_taker_attribution",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "run_id": "join_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "deterministic_join_partial_maker_taker_attribution",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            joined_labels = [
                {
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "fill_id": "fill-train",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "maker_taker_role": "MAKER",
                    "order_join_status": "JOINED",
                    "candidate_join_status": "JOINED",
                    "join_status": "COMPLETE_FOR_NONLIVE_REVIEW",
                    "holdout_split": "TRAIN",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "fill_id": "fill-holdout",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "maker_taker_role": "UNKNOWN",
                    "order_join_status": "JOINED",
                    "candidate_join_status": "MISSING",
                    "join_status": "PARTIAL",
                    "holdout_split": "HOLDOUT",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (join_run / "joined_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in joined_labels:
                    f.write(json.dumps(label) + "\n")
            observed_labels = [
                {
                    "label_type": "OBSERVED_MARKOUT_LABEL",
                    "fill_id": "fill-train",
                    "fill_time_ms": 1000,
                    "markout_horizon_ms": 100,
                    "markout_pnl": 1.5,
                    "future_reference_price_source": "fair_value",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "OBSERVED_MARKOUT_LABEL",
                    "fill_id": "fill-train",
                    "fill_time_ms": 1000,
                    "markout_horizon_ms": 500,
                    "markout_pnl": -0.5,
                    "future_reference_price_source": "fair_value",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "OBSERVED_MARKOUT_LABEL",
                    "fill_id": "fill-holdout",
                    "fill_time_ms": 2000,
                    "markout_horizon_ms": 100,
                    "markout_pnl": -2.0,
                    "future_reference_price_source": "fair_value",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in observed_labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_markout_calibration_readiness_path()),
                    "--observed-run",
                    str(observed_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_markout_readiness_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--min-fills-per-bucket",
                    "2",
                    "--min-holdout-fills-per-bucket",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_markout_readiness_test"
            summary = json.loads((run_dir / "markout_calibration_readiness_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "markout_readiness_sparse_buckets")
            self.assertEqual(summary["markout_row_count"], 3)
            self.assertEqual(summary["unique_fill_count"], 2)
            self.assertEqual(summary["train_fill_count"], 1)
            self.assertEqual(summary["holdout_fill_count"], 1)
            self.assertEqual(summary["adverse_count"], 2)
            self.assertEqual(summary["maker_taker_role_counts_by_fill"], {"MAKER": 1, "UNKNOWN": 1})
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            buckets = [
                json.loads(line)
                for line in (run_dir / "markout_calibration_buckets.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            global_bucket = next(bucket for bucket in buckets if bucket["bucket_id"] == "GLOBAL")
            self.assertEqual(global_bucket["mean_markout_pnl"], -1.0 / 3.0)
            self.assertEqual(global_bucket["adverse_rate"], 2 / 3)
            self.assertIn("maker_taker_unknown_present", global_bucket["gate_reasons"])
            split_manifest = [
                json.loads(line)
                for line in (run_dir / "markout_fill_split_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["fill_id"] for row in split_manifest}, {"fill-train", "fill-holdout"})
            self.assertEqual(
                {row["fill_id"]: row["holdout_split"] for row in split_manifest},
                {"fill-train": "TRAIN", "fill-holdout": "HOLDOUT"},
            )
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

            observed_labels[0]["approved_for_live"] = True
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in observed_labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_markout_calibration_readiness_path()),
                    "--observed-run",
                    str(observed_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_markout_readiness_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

            observed_labels[0]["approved_for_live"] = False
            observed_labels[0]["fill_id"] = "missing-fill"
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in observed_labels:
                    f.write(json.dumps(label) + "\n")
            missing_join = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_markout_calibration_readiness_path()),
                    "--observed-run",
                    str(observed_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_markout_readiness_missing_join_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing_join.returncode, 2)
            self.assertIn("has no deterministic join label", missing_join.stderr)

    def test_phase51c_lighter_attribution_gap_audit_explains_unknowns_hold_only(self):
        """Lighter attribution gap audit should explain unknown roles without inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed"
            join_run = tmp_path / "join"
            backfill_run = tmp_path / "backfill"
            phase51b_run = tmp_path / "phase51b"
            output_root = tmp_path / "gap_audit"
            observed_run.mkdir()
            join_run.mkdir()
            (backfill_run / "source_snapshots").mkdir(parents=True)
            phase51b_run.mkdir()
            source_sha = "source-sha-lighter-gap"
            (observed_run / "observed_label_summary.json").write_text(json.dumps({
                "run_id": "observed_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            fills = [
                {
                    "label_type": "OBSERVED_FILL_LABEL",
                    "fill_id": "known-fill",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "price": 100.0,
                    "size": 0.01,
                    "fill_time_ms": 1000,
                    "maker_taker_role": "MAKER",
                    "order_id_hash": "known-order",
                    "client_order_id_hash": "known-client",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "OBSERVED_FILL_LABEL",
                    "fill_id": "unknown-fill",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "price": 101.0,
                    "size": 0.02,
                    "fill_time_ms": 2000,
                    "maker_taker_role": "UNKNOWN",
                    "order_id_hash": "missing-order",
                    "client_order_id_hash": "missing-client",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for fill in fills:
                    f.write(json.dumps(fill) + "\n")
            (join_run / "join_holdout_summary.json").write_text(json.dumps({
                "run_id": "join_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            join_labels = [
                {
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "fill_id": "known-fill",
                    "candidate_join_status": "JOINED",
                    "join_status": "COMPLETE_FOR_NONLIVE_REVIEW",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "DETERMINISTIC_JOIN_LABEL",
                    "fill_id": "unknown-fill",
                    "candidate_join_status": "JOINED",
                    "join_status": "COMPLETE_FOR_NONLIVE_REVIEW",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (join_run / "joined_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in join_labels:
                    f.write(json.dumps(label) + "\n")
            trades_payload = {
                "schema_version": 1,
                "account_index": 7,
                "trades": [
                    {
                        "trade_id": 1,
                        "timestamp": 1000,
                        "price": "100.0",
                        "size": "0.01",
                        "ask_account_id": 7,
                        "bid_account_id": 8,
                        "ask_id": "known-order",
                        "ask_client_id": "known-client",
                        "bid_id": "other-bid",
                        "bid_client_id": "other-bid-client",
                        "is_maker_ask": True,
                    },
                    {
                        "trade_id": 2,
                        "timestamp": 2000,
                        "price": "101.0",
                        "size": "0.02",
                        "ask_account_id": 7,
                        "bid_account_id": 8,
                        "ask_id": "native-ask",
                        "ask_client_id": "native-client",
                        "bid_id": "other-bid-2",
                        "bid_client_id": "other-bid-client-2",
                        "is_maker_ask": False,
                    },
                ],
            }
            trades_path = backfill_run / "source_snapshots" / "trades_backfill.sanitized.json"
            trades_path.write_text(json.dumps(trades_payload), encoding="utf-8")
            (backfill_run / "lighter_trade_backfill_summary.json").write_text(json.dumps({
                "run_id": "backfill_source",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "account_index": 7,
                "trade_count": 2,
                "trades_path": str(trades_path),
                "trades_sha256": hashlib.sha256(trades_path.read_bytes()).hexdigest(),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (phase51b_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51b_source",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_lighter_attribution_gap_audit_path()),
                    "--observed-run",
                    str(observed_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--lighter-trade-backfill-run",
                    str(backfill_run),
                    "--phase51b-native-run",
                    str(phase51b_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_lighter_gap_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_lighter_gap_test"
            summary = json.loads((run_dir / "lighter_attribution_gap_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["lighter_fill_count"], 2)
            self.assertEqual(summary["observed_role_counts"], {"MAKER": 1, "UNKNOWN": 1})
            self.assertEqual(summary["gap_reason_counts"]["ATTRIBUTED_NATIVE_ROLE"], 1)
            self.assertEqual(summary["gap_reason_counts"]["ORDER_ID_MISMATCH"], 1)
            self.assertFalse(summary["approved_for_live"])
            self.assertFalse(summary["admissible_for_ev_admission"])
            labels = [
                json.loads(line)
                for line in (run_dir / "lighter_attribution_gap_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({label["gap_reason"] for label in labels}, {"ATTRIBUTED_NATIVE_ROLE", "ORDER_ID_MISMATCH"})
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), file_info["sha256"])

            fills[0]["approved_for_live"] = True
            with (observed_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for fill in fills:
                    f.write(json.dumps(fill) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_lighter_attribution_gap_audit_path()),
                    "--observed-run",
                    str(observed_run),
                    "--join-holdout-run",
                    str(join_run),
                    "--lighter-trade-backfill-run",
                    str(backfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_lighter_gap_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51c_queue_churn_labels_emit_hold_only_proxy_fields(self):
        """Queue/churn labels should join lifecycle proxies and keep native pressure unknown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            label_lake_run = tmp_path / "label_lake"
            pfill_run = tmp_path / "pfill_outcome"
            native_run = tmp_path / "phase51b_native"
            output_root = tmp_path / "queue_churn"
            label_lake_run.mkdir()
            pfill_run.mkdir()
            native_run.mkdir()
            source_sha = "source-sha-queue"
            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "label_lake_queue_test",
                "source_telemetry_sha256": source_sha,
                "order_lifecycle_labels": 3,
            }), encoding="utf-8")
            lifecycle = [
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 1,
                    "action": "place",
                    "source_t": 10,
                    "source_line": 10,
                    "source_order_index": 0,
                    "order_id_hash": "order-1",
                    "client_order_id_hash": "client-1",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 2,
                    "action": "replace",
                    "source_t": 11,
                    "source_line": 11,
                    "source_order_index": 0,
                    "order_id_hash": "order-1",
                    "client_order_id_hash": "client-1",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 3,
                    "action": "cancel",
                    "source_t": 12,
                    "source_line": 12,
                    "source_order_index": 0,
                    "order_id_hash": "order-1",
                    "client_order_id_hash": "client-1",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                for label in lifecycle:
                    f.write(json.dumps(label) + "\n")
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_queue_test",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            pfill_labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_queue_test",
                    "order_key": "order-key-1",
                    "order_holdout_split": "TRAIN",
                    "order_label_seq": 1,
                    "order_source_line": 10,
                    "order_source_t": 10,
                    "order_source_order_index": 0,
                    "order_id_hash": "order-1",
                    "client_order_id_hash": "client-1",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "price": 100.0,
                    "size": 0.01,
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "fill_count": 0,
                    "filled_size_total": None,
                    "terminal_action_first": "cancel",
                    "terminal_event_count": 1,
                    "observed_horizon_source_ticks": 2,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_queue_test",
                    "order_key": "order-key-2",
                    "order_holdout_split": "HOLDOUT",
                    "order_label_seq": 99,
                    "order_id_hash": "missing-order",
                    "client_order_id_hash": "missing-client",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "outcome_status": "CENSORED_OR_UNOBSERVED",
                    "p_fill_outcome": None,
                    "fill_count": 0,
                    "filled_size_total": None,
                    "terminal_action_first": None,
                    "terminal_event_count": 0,
                    "observed_horizon_source_ticks": None,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")
            (native_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51b_native_test",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": ["lighter_open_order_limit_headroom_unknown"],
            }), encoding="utf-8")
            with (native_run / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACCOUNT_LIMITS",
                    "run_id": "phase51b_native_test",
                    "venue_id": "lighter",
                    "sendtx_per_minute_remaining": None,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACTIVE_ORDERS",
                    "run_id": "phase51b_native_test",
                    "venue_id": "lighter",
                    "active_orders_count_total": 0,
                    "active_orders_count_market": 0,
                    "active_order_headroom_account": None,
                    "active_order_headroom_market": None,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_queue_churn_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_queue_churn_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--lighter-native-limits-run",
                    str(native_run),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51c_queue_churn_test"
            summary = json.loads((run_dir / "queue_churn_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "queue_churn_partial_lifecycle_join")
            self.assertEqual(summary["queue_churn_label_count"], 2)
            self.assertEqual(summary["matched_lifecycle_count"], 1)
            self.assertEqual(summary["unmatched_lifecycle_count"], 1)
            self.assertEqual(summary["orders_with_churn_count"], 1)
            self.assertEqual(summary["native_limit_pressure_unknown_count"], 0)
            self.assertEqual(summary["native_limit_pressure_partial_count"], 2)
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            labels = [
                json.loads(line)
                for line in (run_dir / "queue_churn_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            joined = next(label for label in labels if label["order_key"] == "order-key-1")
            self.assertEqual(joined["lifecycle_join_status"], "JOINED")
            self.assertEqual(joined["replace_event_count"], 1)
            self.assertEqual(joined["cancel_event_count"], 1)
            self.assertEqual(joined["churn_event_count"], 2)
            self.assertEqual(joined["queue_reset_proxy_event_count"], 1)
            self.assertEqual(joined["observed_lifecycle_source_ticks"], 2)
            self.assertEqual(joined["native_limit_pressure_status"], "PARTIAL_ACTIVE_ORDER_COUNT_OBSERVED_LIMIT_UNKNOWN")
            self.assertEqual(joined["native_active_orders_count_total"], 0)
            self.assertIsNone(joined["native_active_order_limit_source"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

            native_doc_run = tmp_path / "native_doc_run"
            native_doc_run.mkdir()
            (native_doc_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51m_native_doc_test",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": ["lighter_sendtx_remaining_not_observed"],
            }), encoding="utf-8")
            with (native_doc_run / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACCOUNT_LIMITS",
                    "run_id": "phase51m_native_doc_test",
                    "venue_id": "lighter",
                    "sendtx_per_minute_remaining": None,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACTIVE_ORDERS",
                    "run_id": "phase51m_native_doc_test",
                    "venue_id": "lighter",
                    "active_orders_count_total": 2,
                    "active_orders_count_market": 1,
                    "active_order_headroom_account": 1498,
                    "active_order_headroom_market": 999,
                    "active_order_limit_source": "OFFICIAL_DOC_CAP",
                    "active_order_limit_conflicts": [],
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
            observed_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_queue_churn_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_queue_churn_doc_cap_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--lighter-native-limits-run",
                    str(native_doc_run),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(observed_result.returncode, 0, f"stdout: {observed_result.stdout}\nstderr: {observed_result.stderr}")
            observed_run_dir = output_root / "phase51c_queue_churn_doc_cap_test"
            observed_summary = json.loads((observed_run_dir / "queue_churn_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(observed_summary["native_limit_pressure_unknown_count"], 0)
            self.assertEqual(observed_summary["native_limit_pressure_partial_count"], 2)
            observed_labels = [
                json.loads(line)
                for line in (observed_run_dir / "queue_churn_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            observed_joined = next(label for label in observed_labels if label["order_key"] == "order-key-1")
            self.assertEqual(observed_joined["native_limit_pressure_status"], "PARTIAL_NATIVE_HEADROOM_NOT_EVENT_TIME_ALIGNED")
            self.assertEqual(observed_joined["native_active_order_headroom_account"], 1498)
            self.assertEqual(observed_joined["native_active_order_limit_source"], "OFFICIAL_DOC_CAP")
            self.assertEqual(observed_joined["native_active_order_limit_conflicts"], [])
            self.assertIsNone(observed_joined["native_limit_time_alignment_status"])

            native_aligned_run = tmp_path / "native_aligned_run"
            native_aligned_run.mkdir()
            (native_aligned_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51m_native_aligned_test",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": ["lighter_sendtx_remaining_not_observed"],
            }), encoding="utf-8")
            with (native_aligned_run / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACCOUNT_LIMITS",
                    "run_id": "phase51m_native_aligned_test",
                    "venue_id": "lighter",
                    "sendtx_per_minute_remaining": None,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACTIVE_ORDERS",
                    "run_id": "phase51m_native_aligned_test",
                    "venue_id": "lighter",
                    "active_orders_count_total": 2,
                    "active_orders_count_market": 1,
                    "active_order_headroom_account": 1498,
                    "active_order_headroom_market": 999,
                    "active_order_limit_source": "OFFICIAL_DOC_CAP",
                    "active_order_limit_conflicts": [],
                    "native_limit_time_alignment_status": "EVENT_TIME_ALIGNED",
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
            aligned_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_queue_churn_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_queue_churn_aligned_cap_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--lighter-native-limits-run",
                    str(native_aligned_run),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(aligned_result.returncode, 0, f"stdout: {aligned_result.stdout}\nstderr: {aligned_result.stderr}")
            aligned_run_dir = output_root / "phase51c_queue_churn_aligned_cap_test"
            aligned_summary = json.loads((aligned_run_dir / "queue_churn_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(aligned_summary["native_limit_pressure_unknown_count"], 0)
            self.assertEqual(aligned_summary["native_limit_pressure_partial_count"], 0)
            aligned_labels = [
                json.loads(line)
                for line in (aligned_run_dir / "queue_churn_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            aligned_joined = next(label for label in aligned_labels if label["order_key"] == "order-key-1")
            self.assertEqual(aligned_joined["native_limit_pressure_status"], "OBSERVED_NATIVE_LIMIT_HEADROOM")
            self.assertEqual(aligned_joined["native_limit_time_alignment_status"], "EVENT_TIME_ALIGNED")

            pfill_labels[0]["approved_for_live"] = True
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in pfill_labels:
                    f.write(json.dumps(label) + "\n")
            unsafe = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_queue_churn_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51c_queue_churn_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", unsafe.stderr)

    def test_phase51n_lighter_native_limit_alignment_feeds_queue_churn_without_false_clearance(self):
        """Event-time Lighter snapshots should not clear sendTx/REST-native pressure by inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "telemetry.jsonl"
            snapshot_log = tmp_path / "paraphina_live.err.segment"
            pfill_run = tmp_path / "pfill_outcome"
            label_lake_run = tmp_path / "label_lake"
            native_run = tmp_path / "phase51b_native"
            alignment_root = tmp_path / "phase51n_alignment"
            queue_root = tmp_path / "queue_churn"
            pfill_run.mkdir()
            label_lake_run.mkdir()
            native_run.mkdir()
            source_path.write_text(
                json.dumps({"schema_version": 1, "t": 10, "kf_last_update_ms": 1700000000000}) + "\n",
                encoding="utf-8",
            )
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            snapshot_log.write_text(
                "INFO: Lighter account snapshot seq=1700000000000 ts=1700000000100 "
                "positions=flat collateral_usd=1 available_usd=1 total_order_count=4 pending_order_count=0\n",
                encoding="utf-8",
            )
            (native_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51b_native_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "approved_for_calibration_label_ingestion": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": ["lighter_sendtx_remaining_not_observed"],
            }), encoding="utf-8")
            with (native_run / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACCOUNT_LIMITS",
                    "run_id": "phase51b_native_test",
                    "venue_id": "lighter",
                    "sendtx_per_minute_remaining": None,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACTIVE_ORDERS",
                    "run_id": "phase51b_native_test",
                    "venue_id": "lighter",
                    "active_orders_count_total": 0,
                    "active_order_headroom_account": 1500,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_phase51n_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            pfill_label = {
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "run_id": "pfill_phase51n_test",
                "order_key": "order-key-lighter",
                "order_holdout_split": "TRAIN",
                "order_label_seq": 1,
                "order_source_line": 1,
                "order_source_t": 10,
                "order_source_order_index": 0,
                "order_id_hash": "order-hash",
                "client_order_id_hash": "client-hash",
                "venue_id": "lighter",
                "side": "Buy",
                "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                "p_fill_outcome": 0.0,
                "fill_count": 0,
                "filled_size_total": None,
                "terminal_action_first": "cancel",
                "terminal_event_count": 1,
                "observed_horizon_source_ticks": 2,
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(pfill_label) + "\n")

            alignment_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_lighter_native_limit_time_alignment_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--source-telemetry",
                    str(source_path),
                    "--lighter-snapshot-log",
                    str(snapshot_log),
                    "--phase51b-native-run",
                    str(native_run),
                    "--output-root",
                    str(alignment_root),
                    "--run-id",
                    "phase51n_alignment_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(alignment_result.returncode, 0, f"stdout: {alignment_result.stdout}\nstderr: {alignment_result.stderr}")
            alignment_dir = alignment_root / "phase51n_alignment_test"
            alignment_summary = json.loads(
                (alignment_dir / "lighter_native_limit_time_alignment_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(alignment_summary["native_limit_event_time_aligned_count"], 1)
            self.assertEqual(alignment_summary["native_limit_all_pressure_dimensions_observed_count"], 0)
            alignment_label = json.loads(
                (alignment_dir / "lighter_native_limit_time_alignment_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(alignment_label["native_limit_time_alignment_status"], "EVENT_TIME_ALIGNED")
            self.assertEqual(alignment_label["native_active_order_headroom_account"], 1496)
            self.assertFalse(alignment_label["native_limit_all_pressure_dimensions_observed"])
            self.assertEqual(alignment_summary["forward_native_limit_pressure_source_count"], 0)
            forward_source = alignment_dir / "lighter_forward_native_limit_pressure_snapshot.jsonl"
            self.assertTrue(forward_source.is_file())
            self.assertEqual(forward_source.read_text(encoding="utf-8"), "")

            (label_lake_run / "label_lake_summary.json").write_text(json.dumps({
                "run_id": "label_lake_phase51n_test",
                "source_telemetry_sha256": source_sha,
                "order_lifecycle_labels": 1,
            }), encoding="utf-8")
            with (label_lake_run / "labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label_type": "ORDER_LIFECYCLE_LABEL",
                    "label_seq": 1,
                    "action": "cancel",
                    "source_t": 12,
                    "source_line": 12,
                    "source_order_index": 0,
                    "order_id_hash": "order-hash",
                    "client_order_id_hash": "client-hash",
                    "venue_id": "lighter",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")
            queue_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51c_queue_churn_path()),
                    "--label-lake-run",
                    str(label_lake_run),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--output-root",
                    str(queue_root),
                    "--run-id",
                    "phase51n_queue_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--lighter-native-limits-run",
                    str(native_run),
                    "--lighter-native-limit-alignment-run",
                    str(alignment_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(queue_result.returncode, 0, f"stdout: {queue_result.stdout}\nstderr: {queue_result.stderr}")
            queue_dir = queue_root / "phase51n_queue_test"
            queue_summary = json.loads((queue_dir / "queue_churn_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(queue_summary["native_limit_pressure_partial_count"], 1)
            self.assertEqual(queue_summary["native_limit_pressure_observed_count"], 0)
            queue_label = json.loads((queue_dir / "queue_churn_labels.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(
                queue_label["native_limit_pressure_status"],
                "PARTIAL_NATIVE_HEADROOM_EVENT_TIME_ALIGNED_LIMITS_INCOMPLETE",
            )
            self.assertEqual(queue_label["native_limit_alignment_source"], "phase51n_lighter_native_limit_time_alignment")
            self.assertIsNone(queue_label["native_rest_requests_per_minute_remaining"])
            self.assertIsNone(queue_label["native_weighted_requests_per_minute_remaining"])

    def test_phase51n_complete_lighter_limit_alignment_feeds_phase51v(self):
        """Complete event-time Lighter limit pressure should become a 5.1v-ready source row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "telemetry.jsonl"
            snapshot_log = tmp_path / "paraphina_live.err.segment"
            pfill_run = tmp_path / "pfill_outcome"
            native_run = tmp_path / "phase51b_native"
            target_run = tmp_path / "phase51u_targets"
            alignment_root = tmp_path / "phase51n_alignment"
            readiness_root = tmp_path / "phase51v_readiness"
            pfill_run.mkdir()
            native_run.mkdir()
            target_run.mkdir()
            source_path.write_text(
                json.dumps({"schema_version": 1, "t": 10, "kf_last_update_ms": 1700000000000}) + "\n",
                encoding="utf-8",
            )
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            snapshot_log.write_text(
                "INFO: Lighter account snapshot seq=1700000000000 ts=1700000000100 "
                "positions=flat collateral_usd=1 available_usd=1 total_order_count=4 pending_order_count=0\n",
                encoding="utf-8",
            )
            (native_run / "phase51b_acceptance.json").write_text(json.dumps({
                "run_id": "phase51b_native_complete_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "approved_for_financial_claim": False,
                "limitations": [],
            }), encoding="utf-8")
            with (native_run / "telemetry.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACCOUNT_LIMITS",
                    "run_id": "phase51b_native_complete_test",
                    "venue_id": "lighter",
                    "sendtx_per_minute_limit": 120,
                    "sendtx_per_minute_remaining": 119,
                    "weighted_requests_per_minute_limit": 300,
                    "weighted_requests_per_minute_remaining": 299,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
                f.write(json.dumps({
                    "event_type": "V2_LIGHTER_ACTIVE_ORDERS",
                    "run_id": "phase51b_native_complete_test",
                    "venue_id": "lighter",
                    "active_orders_count_total": 0,
                    "active_orders_count_market": 0,
                    "active_order_headroom_account": 1500,
                    "active_order_headroom_market": 100,
                    "approved_for_live": False,
                    "approved_for_canary": False,
                }) + "\n")
            (pfill_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "pfill_phase51n_complete_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "source_telemetry_sha256": source_sha,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (pfill_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "run_id": "pfill_phase51n_complete_test",
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "order_label_seq": 1,
                    "order_source_line": 1,
                    "order_source_t": 10,
                    "venue_id": "lighter",
                    "source_telemetry_sha256": source_sha,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "lighter-group",
                "order_key": "lighter-order",
                "venue_id": "lighter",
                "required_native_limit_source": "LIGHTER_LIMITS_AT_DECISION_TIME",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            alignment_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_lighter_native_limit_time_alignment_path()),
                    "--pfill-outcome-run",
                    str(pfill_run),
                    "--source-telemetry",
                    str(source_path),
                    "--lighter-snapshot-log",
                    str(snapshot_log),
                    "--phase51b-native-run",
                    str(native_run),
                    "--output-root",
                    str(alignment_root),
                    "--run-id",
                    "phase51n_complete_alignment_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(alignment_result.returncode, 0, f"stdout: {alignment_result.stdout}\nstderr: {alignment_result.stderr}")
            alignment_dir = alignment_root / "phase51n_complete_alignment_test"
            alignment_summary = json.loads(
                (alignment_dir / "lighter_native_limit_time_alignment_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(alignment_summary["native_limit_all_pressure_dimensions_observed_count"], 1)
            self.assertEqual(alignment_summary["forward_native_limit_pressure_source_count"], 1)
            self.assertTrue(alignment_summary["phase51v_lighter_native_limit_manifest_ready"])
            alignment_label = json.loads(
                (alignment_dir / "lighter_native_limit_time_alignment_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(alignment_label["native_weighted_requests_per_minute_remaining"], 299)
            forward_source_text = (alignment_dir / "lighter_forward_native_limit_pressure_snapshot.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("sendtx_per_minute_limit", forward_source_text)
            self.assertNotIn("client_order_id", forward_source_text)
            self.assertNotIn("trade_id", forward_source_text)

            readiness_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    alignment_summary["phase51v_lighter_native_limit_manifest_path"],
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_lighter_limit_from_phase51n_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(readiness_result.returncode, 0, f"stdout: {readiness_result.stdout}\nstderr: {readiness_result.stderr}")
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_lighter_limit_from_phase51n_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_ready_count"], 1)
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_missing_count"], 0)
            self.assertFalse(readiness_summary["approved_for_live"])
            self.assertFalse(readiness_summary["live_orders_allowed"])

    def test_phase51n_maker_taker_recovery_uses_only_venue_native_evidence(self):
        """Maker/taker recovery should recover only from explicit venue-native role evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            output_root = tmp_path / "maker_taker_recovery"
            native_roles = tmp_path / "native_roles.jsonl"
            observed_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51n_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 3,
                "censored_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-1",
                    "order_key": "order-1",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "hyperliquid",
                    "side": "Buy",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-2",
                    "order_key": "order-2",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "aster",
                    "side": "Sell",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-3",
                    "order_key": "order-3",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "paradex",
                    "side": "Buy",
                    "fill_count": 0,
                    "outcome_status": "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
                    "p_fill_outcome": 0.0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            native_roles.write_text(json.dumps({
                "canonical_group_id": "group-1",
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 1, "UNKNOWN": 0},
                "maker_taker_attribution_source": "HYPERLIQUID_CROSSED",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_maker_taker_attribution_recovery_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(native_roles),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51n_maker_taker_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51n_maker_taker_test"
            summary = json.loads((run_dir / "maker_taker_attribution_recovery_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["maker_taker_observed_or_recovered_count"], 2)
            self.assertEqual(summary["maker_taker_partial_or_missing_count"], 0)
            recovery_labels = [
                json.loads(line)
                for line in (run_dir / "maker_taker_attribution_recovery_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_group = {label["canonical_group_id"]: label for label in recovery_labels}
            self.assertEqual(by_group["group-1"]["maker_taker_recovery_status"], "RECOVERED_VENUE_NATIVE_ROLE")
            self.assertEqual(by_group["group-1"]["maker_taker_attribution_source"], "HYPERLIQUID_CROSSED")
            self.assertEqual(by_group["group-2"]["maker_taker_recovery_status"], "OBSERVED_PRESERVED")
            self.assertEqual(by_group["group-3"]["maker_taker_recovery_status"], "NO_FILL_NOT_APPLICABLE")

    def test_phase51o_native_role_inventory_feeds_recovery_without_inference(self):
        """5.1o should emit only exact canonical venue-native role evidence for 5.1n recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_root = tmp_path / "source_root"
            inventory_root = tmp_path / "native_role_inventory"
            recovery_root = tmp_path / "maker_taker_recovery"
            native_roles = tmp_path / "native_roles.jsonl"
            observed_run.mkdir()
            source_root.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51o_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 4,
                "censored_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (source_root / "trades_backfill.sanitized.json").write_text(json.dumps({
                "schema_version": 1,
                "source_mode": "synthetic_test_metadata_only",
                "trade_count": 0,
                "trades": [],
            }), encoding="utf-8")
            labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-observed",
                    "order_key": "order-observed",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "hyperliquid",
                    "side": "Buy",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-recovered",
                    "order_key": "order-recovered",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "aster",
                    "side": "Sell",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-source-available",
                    "order_key": "order-source-available",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-missing",
                    "order_key": "order-missing",
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "extended",
                    "side": "Buy",
                    "fill_count": 1,
                    "outcome_status": "OBSERVED_FILLED",
                    "p_fill_outcome": 1.0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")
            native_roles.write_text(json.dumps({
                "canonical_group_id": "group-recovered",
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 1, "UNKNOWN": 0},
                "maker_taker_attribution_source": "ASTER_ORDER_TRADE_UPDATE_M",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            inventory_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51o_native_role_source_inventory_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-root",
                    str(source_root),
                    "--native-role-jsonl",
                    str(native_roles),
                    "--output-root",
                    str(inventory_root),
                    "--run-id",
                    "phase51o_inventory_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                inventory_result.returncode,
                0,
                f"stdout: {inventory_result.stdout}\nstderr: {inventory_result.stderr}",
            )
            inventory_dir = inventory_root / "phase51o_inventory_test"
            inventory_summary = json.loads(
                (inventory_dir / "native_role_source_inventory_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(inventory_summary["input_observed_preserved_count"], 1)
            self.assertEqual(inventory_summary["recovered_native_role_count"], 1)
            self.assertEqual(inventory_summary["missing_native_role_source_count"], 1)
            self.assertEqual(inventory_summary["source_available_no_canonical_join_count"], 1)
            self.assertEqual(inventory_summary["native_role_evidence_record_count"], 1)
            self.assertEqual(inventory_summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(inventory_summary["source_artifact_venues"], ["lighter"])
            evidence_rows = [
                json.loads(line)
                for line in (inventory_dir / "native_role_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(evidence_rows), 1)
            self.assertEqual(evidence_rows[0]["canonical_group_id"], "group-recovered")
            self.assertEqual(evidence_rows[0]["maker_taker_attribution_source"], "ASTER_ORDER_TRADE_UPDATE_M")
            self.assertFalse(evidence_rows[0]["approved_for_live"])
            self.assertFalse(evidence_rows[0]["approved_for_model_training"])

            recovery_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_maker_taker_attribution_recovery_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(inventory_dir / "native_role_evidence.jsonl"),
                    "--output-root",
                    str(recovery_root),
                    "--run-id",
                    "phase51o_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                recovery_result.returncode,
                0,
                f"stdout: {recovery_result.stdout}\nstderr: {recovery_result.stderr}",
            )
            recovery_summary = json.loads(
                (recovery_root / "phase51o_recovery_test" / "maker_taker_attribution_recovery_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(recovery_summary["maker_taker_observed_or_recovered_count"], 2)
            self.assertEqual(recovery_summary["maker_taker_partial_or_missing_count"], 2)
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["RECOVERED_VENUE_NATIVE_ROLE"],
                1,
            )
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["OBSERVED_PRESERVED"],
                1,
            )
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["MISSING_VENUE_NATIVE_ROLE_SOURCE"],
                2,
            )

    def test_phase51p_lighter_native_role_join_feeds_recovery_without_raw_ids(self):
        """5.1p should recover Lighter roles from hashed native IDs without emitting raw IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            inventory_run = tmp_path / "native_role_inventory"
            canonical_run = tmp_path / "canonical_pfill"
            source_run = tmp_path / "source_pfill"
            trade_run = tmp_path / "lighter_trade_backfill"
            join_root = tmp_path / "lighter_native_join"
            recovery_root = tmp_path / "maker_taker_recovery"
            for path in (inventory_run, canonical_run, source_run, trade_run):
                path.mkdir(parents=True)

            def phase51_hash(value):
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                return hashlib.sha256(encoded).hexdigest()

            source_labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "order_key": "source-order-recovered",
                    "canonical_group_id": "source-group-recovered",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "fill_count": 1,
                    "client_order_id_hash": phase51_hash("client-recovered"),
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "order_key": "source-order-missing",
                    "canonical_group_id": "source-group-missing",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "fill_count": 1,
                    "client_order_id_hash": phase51_hash("client-missing"),
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (source_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "source_phase51p_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(source_labels),
                "source_telemetry_sha256": "source-sha",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (source_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in source_labels:
                    f.write(json.dumps(label) + "\n")

            canonical_labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-recovered",
                    "order_key": "canonical-order-recovered",
                    "source_order_keys": ["source-order-recovered"],
                    "source_pfill_run_paths": [str(source_run)],
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "lighter",
                    "side": "Buy",
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "group-missing",
                    "order_key": "canonical-order-missing",
                    "source_order_keys": ["source-order-missing"],
                    "source_pfill_run_paths": [str(source_run)],
                    "source_telemetry_sha256": "source-sha",
                    "venue_id": "lighter",
                    "side": "Sell",
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (canonical_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "canonical_phase51p_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "observed_label_pack_partial_maker_taker_attribution",
                "order_label_count": len(canonical_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (canonical_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in canonical_labels:
                    f.write(json.dumps(label) + "\n")

            inventory_labels = [
                {
                    "label_type": "PHASE51O_NATIVE_ROLE_SOURCE_INVENTORY_LABEL",
                    "canonical_group_id": "group-recovered",
                    "native_role_source_status": "SOURCE_AVAILABLE_NO_CANONICAL_JOIN",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51O_NATIVE_ROLE_SOURCE_INVENTORY_LABEL",
                    "canonical_group_id": "group-missing",
                    "native_role_source_status": "SOURCE_AVAILABLE_NO_CANONICAL_JOIN",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (inventory_run / "native_role_source_inventory_summary.json").write_text(json.dumps({
                "run_id": "inventory_phase51p_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "gate_reason": "phase51o_native_role_source_inventory_incomplete",
                "label_count": len(inventory_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (inventory_run / "native_role_source_inventory_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in inventory_labels:
                    f.write(json.dumps(label) + "\n")

            source_snapshots = trade_run / "source_snapshots"
            source_snapshots.mkdir()
            trades_path = source_snapshots / "trades_backfill.sanitized.json"
            trades_path.write_text(json.dumps({
                "schema_version": 1,
                "account_index": 123,
                "trades": [
                    {
                        "trade_id": "trade-raw-redacted-by-tool",
                        "market_id": 1,
                        "timestamp": 1700000000000,
                        "price": "100.0",
                        "size": "0.1",
                        "is_maker_ask": False,
                        "ask_account_id": 456,
                        "bid_account_id": 123,
                        "ask_client_id": "other-client",
                        "bid_client_id": "client-recovered",
                    }
                ],
            }), encoding="utf-8")
            (trade_run / "lighter_trade_backfill_summary.json").write_text(json.dumps({
                "run_id": "trade_phase51p_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "trades_path": str(trades_path),
                "account_index": 123,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")

            join_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51p_lighter_native_role_canonical_join_path()),
                    "--native-role-inventory-run",
                    str(inventory_run),
                    "--canonical-pfill-run",
                    str(canonical_run),
                    "--source-pfill-run",
                    str(source_run),
                    "--lighter-trade-backfill-run",
                    str(trade_run),
                    "--output-root",
                    str(join_root),
                    "--run-id",
                    "phase51p_join_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(join_result.returncode, 0, f"stdout: {join_result.stdout}\nstderr: {join_result.stderr}")
            join_dir = join_root / "phase51p_join_test"
            join_summary = json.loads(
                (join_dir / "lighter_native_role_canonical_join_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(join_summary["gate_status"], "HOLD")
            self.assertEqual(join_summary["gate_reason"], "phase51p_lighter_native_role_join_incomplete")
            self.assertEqual(join_summary["lighter_source_available_target_count"], 2)
            self.assertEqual(join_summary["recovered_lighter_native_role_count"], 1)
            self.assertEqual(join_summary["unrecovered_lighter_native_role_count"], 1)
            self.assertEqual(join_summary["native_role_evidence_record_count"], 1)
            self.assertEqual(join_summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(
                join_summary["lighter_native_role_join_status_counts"]["RECOVERED_LIGHTER_NATIVE_ROLE"],
                1,
            )
            self.assertEqual(
                join_summary["lighter_native_role_join_status_counts"]["NATIVE_ID_HASH_NO_MATCH"],
                1,
            )

            evidence_rows = [
                json.loads(line)
                for line in (join_dir / "lighter_native_role_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(evidence_rows), 1)
            self.assertEqual(evidence_rows[0]["canonical_group_id"], "group-recovered")
            self.assertEqual(evidence_rows[0]["maker_taker_role_counts"], {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0})
            self.assertEqual(evidence_rows[0]["maker_taker_attribution_source"], "LIGHTER_TRADES_JSON")
            self.assertFalse(evidence_rows[0]["approved_for_live"])
            raw_fields = {
                "order_id",
                "client_order_id",
                "ask_id",
                "bid_id",
                "ask_client_id",
                "bid_client_id",
                "trade_id",
            }
            self.assertFalse(raw_fields & set(evidence_rows[0]))

            recovery_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_maker_taker_attribution_recovery_path()),
                    "--observed-pfill-run",
                    str(canonical_run),
                    "--native-role-jsonl",
                    str(join_dir / "lighter_native_role_evidence.jsonl"),
                    "--output-root",
                    str(recovery_root),
                    "--run-id",
                    "phase51p_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                recovery_result.returncode,
                0,
                f"stdout: {recovery_result.stdout}\nstderr: {recovery_result.stderr}",
            )
            recovery_summary = json.loads(
                (recovery_root / "phase51p_recovery_test" / "maker_taker_attribution_recovery_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(recovery_summary["maker_taker_observed_or_recovered_count"], 1)
            self.assertEqual(recovery_summary["maker_taker_partial_or_missing_count"], 1)
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["RECOVERED_VENUE_NATIVE_ROLE"],
                1,
            )
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["MISSING_VENUE_NATIVE_ROLE_SOURCE"],
                1,
            )

    def test_phase51q_forward_native_evidence_feeds_all_five_venues_without_raw_ids(self):
        """5.1q should emit all-five venue native role evidence and Lighter limit labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            capture_root = tmp_path / "forward_native_capture"
            recovery_root = tmp_path / "maker_taker_recovery"
            native_roles = tmp_path / "native_roles.jsonl"
            native_limits = tmp_path / "native_limits.jsonl"
            observed_run.mkdir()

            observed_labels = [
                ("group-hl", "order-hl", "hyperliquid", "Buy"),
                ("group-paradex", "order-paradex", "paradex", "Sell"),
                ("group-aster", "order-aster", "aster", "Buy"),
                ("group-extended", "order-extended", "extended", "Sell"),
                ("group-lighter", "order-lighter", "lighter", "Buy"),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51q_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": "source-sha",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": 1,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            native_role_rows = [
                ("group-hl", "hyperliquid", "HYPERLIQUID_CROSSED", "MAKER"),
                ("group-paradex", "paradex", "PARADEX_LIQUIDITY", "TAKER"),
                ("group-aster", "aster", "ASTER_ORDER_TRADE_UPDATE_M", "MAKER"),
                ("group-extended", "extended", "EXTENDED_ISTAKER", "TAKER"),
                ("group-lighter", "lighter", "LIGHTER_TRADES_JSON", "MAKER"),
            ]
            with native_roles.open("w", encoding="utf-8") as f:
                for group, venue_id, source, role in native_role_rows:
                    f.write(json.dumps({
                        "canonical_group_id": group,
                        "venue_id": venue_id,
                        "native_role": role,
                        "maker_taker_attribution_source": source,
                        "source_record_sha256": f"sha-{group}",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")
            native_limits.write_text(json.dumps({
                "canonical_group_id": "group-lighter",
                "venue_id": "lighter",
                "active_order_headroom_account": 100,
                "active_order_headroom_market": 10,
                "sendtx_per_minute_limit": 1000,
                "sendtx_per_minute_remaining": 990,
                "rest_requests_per_minute_limit": 1200,
                "rest_requests_per_minute_remaining": 1180,
                "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                "native_limit_staleness_ms": 5.0,
                "source_record_sha256": "limit-sha",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(native_roles),
                    "--native-limit-jsonl",
                    str(native_limits),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51q_capture_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_dir = capture_root / "phase51q_capture_test"
            summary = json.loads(
                (capture_dir / "phase51q_forward_native_evidence_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "phase51q_forward_native_evidence_complete_nonlive_hold")
            self.assertEqual(summary["native_role_evidence_record_count"], 5)
            self.assertEqual(summary["recovered_forward_native_role_count"], 5)
            self.assertEqual(summary["native_limit_pressure_status_counts"]["OBSERVED_NATIVE_LIMIT_PRESSURE"], 1)
            self.assertEqual(summary["native_limit_pressure_status_counts"]["NOT_APPLICABLE_NON_LIGHTER"], 4)
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")

            evidence_rows = [
                json.loads(line)
                for line in (capture_dir / "native_role_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(evidence_rows), 5)
            raw_fields = {
                "order_id",
                "client_order_id",
                "venue_order_id",
                "trade_id",
                "fill_id",
                "ask_id",
                "bid_id",
                "ask_client_id",
                "bid_client_id",
            }
            for row in evidence_rows:
                self.assertFalse(raw_fields & set(row))
                self.assertFalse(row["approved_for_live"])
                self.assertFalse(row["approved_for_model_training"])

            recovery_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_maker_taker_attribution_recovery_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(capture_dir / "native_role_evidence.jsonl"),
                    "--output-root",
                    str(recovery_root),
                    "--run-id",
                    "phase51q_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                recovery_result.returncode,
                0,
                f"stdout: {recovery_result.stdout}\nstderr: {recovery_result.stderr}",
            )
            recovery_summary = json.loads(
                (recovery_root / "phase51q_recovery_test" / "maker_taker_attribution_recovery_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(recovery_summary["maker_taker_observed_or_recovered_count"], 5)
            self.assertEqual(recovery_summary["maker_taker_partial_or_missing_count"], 0)
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["RECOVERED_VENUE_NATIVE_ROLE"],
                5,
            )

    def test_phase51q_forward_native_evidence_rejects_raw_identifiers(self):
        """5.1q should reject source rows that leak raw venue identifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            output_root = tmp_path / "forward_native_capture"
            native_roles = tmp_path / "native_roles.jsonl"
            observed_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51q_raw_reject_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-raw",
                "order_key": "order-raw",
                "source_telemetry_sha256": "source-sha",
                "venue_id": "hyperliquid",
                "side": "Buy",
                "fill_count": 1,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            native_roles.write_text(json.dumps({
                "canonical_group_id": "group-raw",
                "venue_id": "hyperliquid",
                "order_id": "raw-order-id-not-allowed",
                "native_role": "MAKER",
                "maker_taker_attribution_source": "HYPERLIQUID_CROSSED",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(native_roles),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51q_raw_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("raw native role source identifier fields", result.stderr)

    def test_phase51s_local_source_acquisition_feeds_phase51r_without_raw_ids(self):
        """5.1s should stage explicit local sources for 5.1r without leaking raw IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            manifest_path = tmp_path / "manifest.json"
            staging_root = tmp_path / "local_source_acquisition"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            observed_run.mkdir()

            observed_labels = [
                ("group-hl", "order-hl", "hyperliquid", "Buy"),
                ("group-lighter", "order-lighter", "lighter", "Sell"),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51s_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": 1,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            source_rows = [
                {
                    "canonical_group_id": "group-hl",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "oid": "raw-hl-order",
                    "cloid": "raw-hl-client-order",
                    "tid": "raw-hl-trade",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-lighter",
                    "venue_id": "lighter",
                    "account_index": 123,
                    "is_maker_ask": True,
                    "ask_account_id": 123,
                    "bid_account_id": 456,
                    "trade_id": "raw-lighter-trade-id",
                    "bid_client_id": "raw-lighter-client-id",
                    "active_order_headroom_account": 100,
                    "active_order_headroom_market": 8,
                    "sendtx_per_minute_limit": 1000,
                    "sendtx_per_minute_remaining": 990,
                    "rest_requests_per_minute_limit": 1200,
                    "rest_requests_per_minute_remaining": 1180,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "native_limit_staleness_ms": 5.0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [{"source_id": "test_source", "venue_id": "lighter", "path": str(source_path)}],
            }), encoding="utf-8")

            staging_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(staging_root),
                    "--run-id",
                    "phase51s_local_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                staging_result.returncode,
                0,
                f"stdout: {staging_result.stdout}\nstderr: {staging_result.stderr}",
            )
            staging_dir = staging_root / "phase51s_local_test"
            staging_summary = json.loads(
                (staging_dir / "phase51s_local_native_source_acquisition_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(staging_summary["gate_status"], "HOLD")
            self.assertEqual(staging_summary["gate_reason"], "phase51s_local_native_source_acquisition_complete_nonlive_hold")
            self.assertEqual(staging_summary["source_row_count"], 2)
            self.assertEqual(staging_summary["staged_source_row_count"], 2)
            self.assertEqual(staging_summary["join_key_source_row_count"], 2)
            self.assertEqual(staging_summary["complete_lighter_native_limit_source_row_count"], 1)
            self.assertEqual(staging_summary["raw_identifier_redaction_status"], "PASS")
            self.assertFalse(staging_summary["clears_phase51_blockers"])
            self.assertNotIn("path", staging_summary["source_artifacts"][0])
            self.assertIn("path_hash", staging_summary["source_artifacts"][0])

            raw_fields = {
                "order_id",
                "client_order_id",
                "venue_order_id",
                "raw_order_id",
                "raw_client_order_id",
                "ask_id",
                "bid_id",
                "ask_client_id",
                "bid_client_id",
                "trade_id",
                "fill_id",
                "id",
                "oid",
                "cloid",
                "tid",
            }
            for line in (staging_dir / "local_native_source.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertFalse(raw_fields & set(row))
                self.assertFalse(row["approved_for_live"])
                self.assertFalse(row["approved_for_model_training"])

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(staging_dir / "local_native_source.jsonl"),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51s_to_phase51r_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_summary = json.loads(
                (
                    acquisition_root
                    / "phase51s_to_phase51r_test"
                    / "phase51r_forward_native_source_acquisition_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 2)
            self.assertEqual(acquisition_summary["lighter_native_limit_target_recovered_count"], 1)

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_root / "phase51s_to_phase51r_test" / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_root / "phase51s_to_phase51r_test" / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51s_to_phase51q_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_summary = json.loads(
                (
                    capture_root
                    / "phase51s_to_phase51q_test"
                    / "phase51q_forward_native_evidence_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 2)
            self.assertEqual(capture_summary["native_limit_pressure_status_counts"]["OBSERVED_NATIVE_LIMIT_PRESSURE"], 1)

    def test_phase51s_local_source_acquisition_rejects_secrets_and_network_sources(self):
        """5.1s should reject unsafe local-source acquisition surfaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "native_source.jsonl"
            manifest_path = tmp_path / "manifest.json"
            output_root = tmp_path / "local_source_acquisition"

            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "sources": [{"source_id": "network", "path": "https://example.invalid/native.json"}],
            }), encoding="utf-8")
            network_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51s_network_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(network_result.returncode, 2)
            self.assertIn("network source paths are prohibited", network_result.stderr)

            source_path.write_text(json.dumps({
                "canonical_group_id": "group-secret",
                "venue_id": "hyperliquid",
                "crossed": False,
                "api_key": "should-never-be-present",
                "approved_for_live": False,
            }) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "sources": [{"source_id": "secret", "path": str(source_path)}],
            }), encoding="utf-8")
            secret_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51s_secret_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(secret_result.returncode, 2)
            self.assertIn("secret-shaped source row field", secret_result.stderr)

            env_source = tmp_path / "native.env"
            env_source.write_text("{}", encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "sources": [{"source_id": "env", "path": str(env_source)}],
            }), encoding="utf-8")
            env_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51s_env_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(env_result.returncode, 2)
            self.assertIn("env files are prohibited", env_result.stderr)

            target_path = tmp_path / "target.jsonl"
            link_path = tmp_path / "linked.jsonl"
            target_path.write_text(json.dumps({
                "canonical_group_id": "group-link",
                "venue_id": "hyperliquid",
                "crossed": False,
            }) + "\n", encoding="utf-8")
            link_path.symlink_to(target_path)
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "sources": [{"source_id": "link", "path": str(link_path)}],
            }), encoding="utf-8")
            symlink_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51s_symlink_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(symlink_result.returncode, 2)
            self.assertIn("symlink source path is prohibited", symlink_result.stderr)

    def test_phase51s_local_source_link_sidecar_feeds_phase51r_deterministically(self):
        """5.1s should stage redacted source-link sidecars for 5.1r without inferring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            source_link_path = tmp_path / "source_links.jsonl"
            manifest_path = tmp_path / "manifest.json"
            staging_root = tmp_path / "local_source_acquisition"
            acquisition_root = tmp_path / "source_acquisition"
            observed_run.mkdir()

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51s_source_link_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-hl-linked",
                "order_key": "order-hl-linked",
                "source_telemetry_sha256": "source-group-hl-linked",
                "venue_id": "hyperliquid",
                "side": "Buy",
                "fill_count": 1,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_path.write_text(json.dumps({
                "venue_id": "hyperliquid",
                "crossed": False,
                "source_record_sha256": "forward-hl-source-hash",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_link_path.write_text(json.dumps({
                "source_record_sha256": "forward-hl-source-hash",
                "canonical_group_id": "group-hl-linked",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [{"source_id": "native", "path": str(source_path)}],
                "source_links": [{"source_link_id": "native_link", "path": str(source_link_path)}],
            }), encoding="utf-8")

            staged_dirs = []
            for run_id in ("phase51s_source_link_test_a", "phase51s_source_link_test_b"):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(self._get_phase51s_local_native_source_acquisition_path()),
                        "--manifest",
                        str(manifest_path),
                        "--output-root",
                        str(staging_root),
                        "--run-id",
                        run_id,
                        "--timestamp-ns",
                        "1700000000000000000",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"stdout: {result.stdout}\nstderr: {result.stderr}",
                )
                staged_dirs.append(staging_root / run_id)

            summary = json.loads(
                (staged_dirs[0] / "phase51s_local_native_source_acquisition_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_link_file_count"], 1)
            self.assertEqual(summary["source_link_row_count"], 1)
            self.assertEqual(summary["staged_source_link_row_count"], 1)
            self.assertEqual(summary["source_link_hash_count"], 1)
            self.assertEqual(summary["local_source_link_stage_status_counts"]["STAGED_LOCAL_SOURCE_LINK_ROW"], 1)
            self.assertEqual(summary["downstream_source_link_argument"], "--source-link-jsonl local_source_link_sidecar.jsonl")
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertEqual(
                (staged_dirs[0] / "local_source_link_sidecar.jsonl").read_text(encoding="utf-8"),
                (staged_dirs[1] / "local_source_link_sidecar.jsonl").read_text(encoding="utf-8"),
            )

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(staged_dirs[0] / "local_native_source.jsonl"),
                    "--source-link-jsonl",
                    str(staged_dirs[0] / "local_source_link_sidecar.jsonl"),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51s_source_link_to_phase51r_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_summary = json.loads(
                (
                    acquisition_root
                    / "phase51s_source_link_to_phase51r_test"
                    / "phase51r_forward_native_source_acquisition_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["source_link_record_count"], 1)
            self.assertEqual(acquisition_summary["source_link_applied_count"], 1)
            self.assertEqual(acquisition_summary["canonical_group_link_source_counts"]["SOURCE_LINK_SIDECAR"], 1)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 1)

            sidecar_only_manifest = tmp_path / "sidecar_only_manifest.json"
            sidecar_only_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [],
                "source_links": [{"source_link_id": "native_link", "path": str(source_link_path)}],
            }), encoding="utf-8")
            sidecar_only_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(sidecar_only_manifest),
                    "--output-root",
                    str(staging_root),
                    "--run-id",
                    "phase51s_source_link_only_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                sidecar_only_result.returncode,
                0,
                f"stdout: {sidecar_only_result.stdout}\nstderr: {sidecar_only_result.stderr}",
            )
            sidecar_only_summary = json.loads(
                (
                    staging_root
                    / "phase51s_source_link_only_test"
                    / "phase51s_local_native_source_acquisition_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                sidecar_only_summary["gate_reason"],
                "phase51s_local_native_source_acquisition_incomplete_source_links_only",
            )
            self.assertEqual(sidecar_only_summary["staged_source_row_count"], 0)
            self.assertEqual(sidecar_only_summary["staged_source_link_row_count"], 1)
            self.assertFalse(sidecar_only_summary["clears_phase51_blockers"])

    def test_phase51t_builder_emits_phase51s_compatible_source_link_sidecar(self):
        """5.1t should build only redacted sidecars from existing order/client hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            builder_root = tmp_path / "source_link_builder"
            staging_root = tmp_path / "local_source_acquisition"
            acquisition_root = tmp_path / "source_acquisition"
            manifest_path = tmp_path / "manifest.json"
            observed_run.mkdir()

            def stable_hash(value):
                encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
                return hashlib.sha256(encoded).hexdigest()

            client_order_id = "client-order-1"
            source_row = {
                "venue_id": "lighter",
                "ask_client_id": client_order_id,
                "is_maker_ask": True,
                "ask_account_id": 123,
                "bid_account_id": 456,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            expanded_source_row = {"account_index": 123, **source_row}
            source_path.write_text(json.dumps({
                "account_index": 123,
                "not_ev_admission_authorization": True,
                "trades": [source_row],
            }), encoding="utf-8")

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51t_source_link_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-from-identity-hash",
                "order_key": "order-from-identity-hash",
                "source_telemetry_sha256": "source-phase51t",
                "venue_id": "lighter",
                "side": "Sell",
                "fill_count": 1,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "client_order_id_hash": stable_hash(client_order_id),
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            builder_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51t_source_link_sidecar_builder_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(builder_root),
                    "--run-id",
                    "phase51t_source_link_builder_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                builder_result.returncode,
                0,
                f"stdout: {builder_result.stdout}\nstderr: {builder_result.stderr}",
            )
            builder_dir = builder_root / "phase51t_source_link_builder_test"
            builder_summary = json.loads(
                (builder_dir / "phase51t_source_link_sidecar_builder_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(builder_summary["gate_status"], "HOLD")
            self.assertEqual(builder_summary["source_link_record_count"], 1)
            self.assertEqual(builder_summary["source_link_status_counts"]["SOURCE_LINK_EMITTED"], 1)
            self.assertFalse(builder_summary["clears_phase51_blockers"])
            sidecar_row = json.loads(
                (builder_dir / "source_links.sanitized.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(sidecar_row["phase51s_source_record_sha256"], stable_hash(expanded_source_row))
            self.assertEqual(sidecar_row["canonical_group_id"], "group-from-identity-hash")
            self.assertEqual(sidecar_row["order_key"], "order-from-identity-hash")
            self.assertNotIn("client_order_id", sidecar_row)

            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [{"source_id": "native", "path": str(source_path)}],
                "source_links": [{"source_link_id": "phase51t_link", "path": str(builder_dir / "source_links.sanitized.jsonl")}],
            }), encoding="utf-8")
            staging_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(staging_root),
                    "--run-id",
                    "phase51t_to_phase51s_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                staging_result.returncode,
                0,
                f"stdout: {staging_result.stdout}\nstderr: {staging_result.stderr}",
            )
            staging_dir = staging_root / "phase51t_to_phase51s_test"
            staged_source = json.loads(
                (staging_dir / "local_native_source.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(staged_source["phase51s_source_record_sha256"], stable_hash(expanded_source_row))
            self.assertNotIn("ask_client_id", staged_source)

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(staging_dir / "local_native_source.jsonl"),
                    "--source-link-jsonl",
                    str(staging_dir / "local_source_link_sidecar.jsonl"),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51t_to_phase51r_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_summary = json.loads(
                (
                    acquisition_root
                    / "phase51t_to_phase51r_test"
                    / "phase51r_forward_native_source_acquisition_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["source_link_applied_count"], 1)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 1)
            self.assertEqual(acquisition_summary["raw_identifier_redaction_status"], "PASS")

    def test_phase51t_builder_rejects_secret_shaped_source_fields(self):
        """5.1t should not stage source-link evidence from secret-bearing inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            observed_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51t_secret_reject_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text("", encoding="utf-8")
            source_path.write_text(json.dumps({
                "venue_id": "hyperliquid",
                "client_order_id": "client-order-1",
                "private_key": "not-allowed",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51t_source_link_sidecar_builder_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(tmp_path / "source_link_builder"),
                    "--run-id",
                    "phase51t_secret_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("secret-shaped source row field", result.stderr)

    def test_phase51u_forward_capture_target_manifest_emits_exact_targets(self):
        """5.1u should emit redacted all-five role targets and Lighter limit targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            output_root = tmp_path / "forward_capture_targets"
            observed_run.mkdir()
            labels = [
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "lighter-filled",
                    "order_key": "lighter-order",
                    "order_id_hash": "lighter-order-id-hash",
                    "client_order_id_hash": "lighter-client-order-id-hash",
                    "decision_id_hash": "lighter-decision-id-hash",
                    "source_telemetry_sha256": "source-lighter",
                    "venue_id": "lighter",
                    "side": "BID",
                    "price": 100.0,
                    "size": 0.01,
                    "fill_count": 2,
                    "first_fill_time_ms": 1700000000100,
                    "last_fill_time_ms": 1700000000200,
                    "order_source_t": 10,
                    "order_source_line": 20,
                    "source_order_keys": ["source-lighter-a", "source-lighter-b"],
                    "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "hyperliquid-filled",
                    "order_key": "hyper-order",
                    "source_telemetry_sha256": "source-hyper",
                    "venue_id": "hyperliquid",
                    "side": "ASK",
                    "price": 101.0,
                    "size": 0.02,
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "paradex-complete",
                    "order_key": "paradex-order",
                    "source_telemetry_sha256": "source-paradex",
                    "venue_id": "paradex",
                    "side": "BID",
                    "price": 102.0,
                    "size": 0.03,
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 1, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "canonical_group_id": "lighter-not-filled",
                    "order_key": "lighter-empty-order",
                    "source_telemetry_sha256": "source-lighter-empty",
                    "venue_id": "lighter",
                    "side": "ASK",
                    "price": 103.0,
                    "size": 0.04,
                    "fill_count": 0,
                    "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51u_target_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for label in labels:
                    f.write(json.dumps(label) + "\n")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51u_forward_capture_target_manifest_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51u_target_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            run_dir = output_root / "phase51u_target_test"
            summary = json.loads(
                (run_dir / "phase51u_forward_capture_target_manifest_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["native_role_capture_target_count"], 2)
            self.assertEqual(summary["lighter_native_limit_capture_target_count"], 2)
            self.assertEqual(summary["native_role_capture_target_counts_by_venue"]["hyperliquid"], 1)
            self.assertEqual(summary["native_role_capture_target_counts_by_venue"]["lighter"], 1)
            self.assertEqual(summary["native_role_required_source_counts"]["HYPERLIQUID_CROSSED"], 1)
            self.assertEqual(summary["native_role_required_source_counts"]["LIGHTER_TRADES_JSON"], 1)
            self.assertFalse(summary["clears_phase51_blockers"])
            role_targets = [
                json.loads(line)
                for line in (run_dir / "native_role_capture_targets.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            lighter_target = next(target for target in role_targets if target["venue_id"] == "lighter")
            self.assertEqual(lighter_target["order_id_hash"], "lighter-order-id-hash")
            self.assertEqual(lighter_target["client_order_id_hash"], "lighter-client-order-id-hash")
            self.assertEqual(lighter_target["decision_id_hash"], "lighter-decision-id-hash")
            self.assertEqual(lighter_target["first_fill_time_ms"], 1700000000100)
            self.assertEqual(lighter_target["last_fill_time_ms"], 1700000000200)
            self.assertEqual(lighter_target["missing_native_role_count"], 1)
            self.assertEqual(lighter_target["required_native_role_source"], "LIGHTER_TRADES_JSON")
            self.assertFalse(lighter_target["role_inference_allowed"])
            output_text = (run_dir / "native_role_capture_targets.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"client_order_id":', output_text)
            self.assertNotIn('"trade_id":', output_text)
            template = json.loads((run_dir / "capture_bundle_manifest_template.json").read_text(encoding="utf-8"))
            self.assertEqual(template["native_role_capture_target_count"], 2)
            self.assertEqual(template["lighter_native_limit_capture_target_count"], 2)
            self.assertFalse(template["live_orders_allowed"])

    def test_phase51u_forward_capture_target_manifest_rejects_unsafe_labels(self):
        """5.1u should reject P_fill labels that attempt to authorize live use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            observed_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51u_unsafe_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "unsafe-group",
                "venue_id": "extended",
                "fill_count": 1,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
                "approved_for_live": True,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51u_forward_capture_target_manifest_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--output-root",
                    str(tmp_path / "forward_capture_targets"),
                    "--run-id",
                    "phase51u_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe label flag approved_for_live=true", result.stderr)

    def test_phase51v_forward_capture_bundle_readiness_accepts_local_bundle(self):
        """5.1v should verify local source rows and emit a Phase 5.1s-ready manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 2,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            role_targets = [
                {
                    "canonical_group_id": "hyper-group",
                    "order_key": "hyper-order",
                    "venue_id": "hyperliquid",
                    "required_native_role_source": "HYPERLIQUID_CROSSED",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_role_source": "LIGHTER_TRADES_JSON",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            limit_targets = [
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_limit_source": "LIGHTER_LIMITS_AT_DECISION_TIME",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }
            ]
            (target_run / "native_role_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in role_targets),
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in limit_targets),
                encoding="utf-8",
            )
            source_path = tmp_path / "local_native_source.jsonl"
            source_rows = [
                {
                    "venue_id": "hyperliquid",
                    "order_key": "hyper-order",
                    "crossed": False,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "venue_id": "lighter",
                    "canonical_group_id": "lighter-group",
                    "account_index": 7,
                    "is_maker_ask": True,
                    "ask_account_id": 8,
                    "bid_account_id": 9,
                    "active_order_headroom_account": 100,
                    "active_order_headroom_market": 20,
                    "sendtx_per_minute_limit": 120,
                    "sendtx_per_minute_remaining": 119,
                    "weighted_requests_per_minute_limit": 300,
                    "weighted_requests_per_minute_remaining": 299,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "local_all5_sample",
                        "venue_id": "mixed",
                        "path": str(source_path),
                    }
                ],
                "source_links": [],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51v_ready_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            run_dir = output_root / "phase51v_ready_test"
            summary = json.loads(
                (run_dir / "phase51v_forward_capture_bundle_readiness_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["gate_reason"], "phase51v_forward_capture_bundle_ready_for_phase51s_nonlive_hold")
            self.assertEqual(summary["native_role_capture_target_ready_count"], 2)
            self.assertEqual(summary["native_role_capture_target_missing_count"], 0)
            self.assertEqual(summary["lighter_native_limit_capture_target_ready_count"], 1)
            self.assertEqual(summary["lighter_native_limit_capture_target_missing_count"], 0)
            self.assertEqual(summary["source_file_status_counts"], {"LOCAL_FILE_READY": 1})
            self.assertTrue(summary["generated_phase51s_manifest_ready"])
            self.assertEqual(summary["generated_phase51s_source_count"], 1)
            self.assertFalse(summary["clears_phase51_blockers"])
            generated_manifest = json.loads(
                (run_dir / "phase51s_manifest.generated.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generated_manifest["sources"][0]["path"], str(source_path))
            self.assertFalse(generated_manifest["live_orders_allowed"])
            labels_text = (run_dir / "capture_bundle_readiness_labels.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("client_order_id", labels_text)
            self.assertNotIn("trade_id", labels_text)

    def test_phase51ab_lighter_native_limit_pressure_source_feeds_phase51v_without_false_clearance(self):
        """5.1ab should stage sanitized Lighter pressure rows for 5.1v without clearing blockers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            source_root = tmp_path / "phase51ab_pressure_source"
            readiness_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "lighter-limit-group",
                "order_key": "lighter-limit-order",
                "venue_id": "lighter",
                "required_native_limit_source": "LIGHTER_LIMITS_AT_DECISION_TIME",
                "required_native_limit_fields": [
                    "active_order_headroom_account",
                    "active_order_headroom_market",
                    "sendtx_per_minute_limit",
                    "sendtx_per_minute_remaining",
                    "weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining",
                    "native_limit_event_time_status",
                ],
                "accepted_native_limit_event_time_status": ["EVENT_TIME_ALIGNED"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            pressure_row = {
                "venue_id": "lighter",
                "canonical_group_id": "lighter-limit-group",
                "order_key": "lighter-limit-order",
                "active_order_headroom_account": 100,
                "active_order_headroom_market": 50,
                "sendtx_per_minute_limit": 120,
                "sendtx_per_minute_remaining": 119,
                "weighted_requests_per_minute_limit": 24000,
                "weighted_requests_per_minute_remaining": 23999,
                "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                "native_limit_staleness_ms": 5.0,
                "pressure_capture_mode": "NONLIVE_TESTNET_OR_PAPER_CAPTURE",
                "approved_for_live": False,
                "approved_for_financial_claim": False,
                "approved_for_model_training": False,
            }
            pressure_input = tmp_path / "sanitized_lighter_pressure.jsonl"
            pressure_input.write_text(json.dumps(pressure_row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(pressure_input),
                    "--target-run",
                    str(target_run),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            source_dir = source_root / "phase51ab_pressure_test"
            source_summary = json.loads(
                (source_dir / "phase51ab_lighter_native_limit_pressure_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source_summary["gate_status"], "HOLD")
            self.assertEqual(source_summary["complete_lighter_native_limit_source_row_count"], 1)
            self.assertFalse(source_summary["clears_phase51_blockers"])
            self.assertIn("phase51v_forward_capture_bundle_readiness.py", source_summary["phase51v_validation_command"])
            staged_text = Path(source_summary["source_path"]).read_text(encoding="utf-8")
            self.assertNotIn("client_order_id", staged_text)
            self.assertNotIn("trade_id", staged_text)

            readiness_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    source_summary["candidate_manifest_path"],
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_phase51ab_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                readiness_result.returncode,
                0,
                f"stdout: {readiness_result.stdout}\nstderr: {readiness_result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_phase51ab_pressure_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_ready_count"], 1)
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_missing_count"], 0)
            self.assertTrue(readiness_summary["generated_phase51s_manifest_ready"])
            self.assertFalse(readiness_summary["clears_phase51_blockers"])
            labels_text = (
                readiness_root
                / "phase51v_phase51ab_pressure_test"
                / "capture_bundle_readiness_labels.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn('"lighter_limit_target_ready":true', labels_text)
            self.assertNotIn("client_order_id", labels_text)
            self.assertNotIn("trade_id", labels_text)

            unsafe_input = tmp_path / "unsafe_lighter_pressure.jsonl"
            unsafe_row = dict(pressure_row, order_id="raw-order-id")
            unsafe_input.write_text(json.dumps(unsafe_row) + "\n", encoding="utf-8")
            unsafe_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(unsafe_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_unsafe_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_result.returncode, 2)
            self.assertIn("raw identifier", unsafe_result.stderr)

            unsafe_flag_input = tmp_path / "unsafe_flag_lighter_pressure.jsonl"
            unsafe_flag_row = dict(pressure_row, no_live_flag=False)
            unsafe_flag_input.write_text(json.dumps(unsafe_flag_row) + "\n", encoding="utf-8")
            unsafe_flag_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(unsafe_flag_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_unsafe_flag_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_flag_result.returncode, 2)
            self.assertIn("no_live_flag", unsafe_flag_result.stderr)

            unsafe_model_training_input = tmp_path / "unsafe_model_training_lighter_pressure.jsonl"
            unsafe_model_training_row = dict(pressure_row, admissible_for_model_training=True)
            unsafe_model_training_input.write_text(
                json.dumps(unsafe_model_training_row) + "\n",
                encoding="utf-8",
            )
            unsafe_model_training_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(unsafe_model_training_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_unsafe_model_training_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_model_training_result.returncode, 2)
            self.assertIn("admissible_for_model_training", unsafe_model_training_result.stderr)

            nested_raw_input = tmp_path / "nested_raw_lighter_pressure.jsonl"
            nested_raw_row = dict(pressure_row, native_limit_pressure_source={"i": "raw-lighter-id"})
            nested_raw_input.write_text(json.dumps(nested_raw_row) + "\n", encoding="utf-8")
            nested_raw_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(nested_raw_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_nested_raw_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(nested_raw_result.returncode, 2)
            self.assertIn("raw identifier", nested_raw_result.stderr)

            non_scalar_input = tmp_path / "non_scalar_lighter_pressure.jsonl"
            non_scalar_row = dict(pressure_row, native_limit_pressure_source={"safe": "but-nested"})
            non_scalar_input.write_text(json.dumps(non_scalar_row) + "\n", encoding="utf-8")
            non_scalar_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(non_scalar_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_non_scalar_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(non_scalar_result.returncode, 2)
            self.assertIn("must be scalar", non_scalar_result.stderr)

            bool_pressure_input = tmp_path / "bool_pressure_lighter_pressure.jsonl"
            bool_pressure_row = dict(pressure_row, sendtx_per_minute_remaining=True)
            bool_pressure_input.write_text(json.dumps(bool_pressure_row) + "\n", encoding="utf-8")
            bool_pressure_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(bool_pressure_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "phase51ab_bool_pressure_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(bool_pressure_result.returncode, 2)
            self.assertIn("sendtx_per_minute_remaining", bool_pressure_result.stderr)

            unsafe_run_id_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ab_lighter_native_limit_pressure_source_path()),
                    "--input-jsonl",
                    str(pressure_input),
                    "--output-root",
                    str(source_root),
                    "--run-id",
                    "../phase51ab_unsafe_run_id_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_run_id_result.returncode, 2)
            self.assertIn("run_id", unsafe_run_id_result.stderr)

    def test_phase51ac_source_link_reuse_audit_reports_reusable_and_missing_hashes(self):
        """5.1ac should compare request-pack hashes with existing sanitized sidecars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            sidecar_root = tmp_path / "sidecars"
            output_root = tmp_path / "phase51ac"
            request_pack.mkdir()
            sidecar_root.mkdir()

            request_sources = [
                {
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_SOURCE",
                    "source_record_sha256": "hash-reusable",
                    "venue_id": "lighter",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_SOURCE",
                    "source_record_sha256": "hash-missing",
                    "venue_id": "aster",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            request_targets = [
                {
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_TARGET",
                    "canonical_group_id": "group-lighter",
                    "order_key": "order-lighter",
                    "venue_id": "lighter",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_TARGET",
                    "canonical_group_id": "group-aster",
                    "order_key": "order-aster",
                    "venue_id": "aster",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (request_pack / "source_link_request_sources.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in request_sources),
                encoding="utf-8",
            )
            (request_pack / "source_link_request_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in request_targets),
                encoding="utf-8",
            )
            reusable_sidecar_dir = sidecar_root / "existing"
            reusable_sidecar_dir.mkdir()
            (reusable_sidecar_dir / "source_links.sanitized.jsonl").write_text(json.dumps({
                "source_record_sha256": "hash-reusable",
                "canonical_group_id": "group-lighter",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ac_source_link_reuse_audit_path()),
                    "--request-pack",
                    str(request_pack),
                    "--sidecar-root",
                    str(sidecar_root),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ac_reuse_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51ac_reuse_test"
            summary = json.loads(
                (run_dir / "phase51ac_source_link_reuse_audit_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["source_link_request_source_count"], 2)
            self.assertEqual(summary["existing_sidecar_file_count"], 1)
            self.assertEqual(summary["existing_sidecar_row_count"], 1)
            self.assertEqual(summary["reusable_source_link_count"], 1)
            self.assertEqual(summary["missing_source_link_count"], 1)
            self.assertEqual(summary["reusable_source_link_counts_by_venue"], {"lighter": 1})
            self.assertEqual(summary["missing_source_link_counts_by_venue"], {"aster": 1})
            self.assertFalse(summary["candidate_sidecar_complete"])
            self.assertFalse(summary["clears_phase51_blockers"])
            reusable_text = (run_dir / "reusable_source_links.jsonl").read_text(encoding="utf-8")
            missing_text = (run_dir / "missing_source_link_request_sources.jsonl").read_text(encoding="utf-8")
            self.assertIn("hash-reusable", reusable_text)
            self.assertIn("hash-missing", missing_text)
            self.assertNotIn("client_order_id", reusable_text + missing_text)
            self.assertNotIn("trade_id", reusable_text + missing_text)

            unsafe_dir = sidecar_root / "unsafe"
            unsafe_dir.mkdir()
            (unsafe_dir / "source_links.sanitized.jsonl").write_text(json.dumps({
                "source_record_sha256": "hash-reusable",
                "canonical_group_id": "group-lighter",
                "client_order_id": "raw-client-id",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            unsafe_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ac_source_link_reuse_audit_path()),
                    "--request-pack",
                    str(request_pack),
                    "--sidecar-root",
                    str(unsafe_dir),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ac_unsafe_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_result.returncode, 2)
            self.assertIn("raw identifier", unsafe_result.stderr)

    def test_phase51ad_source_link_sidecar_materialize_validates_redacted_mappings(self):
        """5.1ad should materialize only request-pack source hashes and target keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            target_run = tmp_path / "phase51u_targets"
            source_path = tmp_path / "phase51z_sources.jsonl"
            mapping_path = tmp_path / "mapping.jsonl"
            output_root = tmp_path / "phase51ad"
            phase51v_root = tmp_path / "phase51v"
            request_pack.mkdir()
            target_run.mkdir()

            source_hash = hashlib.sha256(b"phase51ad-source").hexdigest()
            canonical_group_id = "group-aster"
            order_key = "order-aster"
            source_row = {
                "label_type": "PHASE51Z_UNLINKED_NATIVE_ROLE_SOURCE",
                "source_record_sha256": source_hash,
                "venue_id": "aster",
                "e": "ORDER_TRADE_UPDATE",
                "o": {"m": True, "l": "0.01"},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            target_row = {
                "label_type": "PHASE51U_NATIVE_ROLE_CAPTURE_TARGET",
                "canonical_group_id": canonical_group_id,
                "order_key": order_key,
                "venue_id": "aster",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            source_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(
                json.dumps(target_row) + "\n",
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            (request_pack / "source_link_request_sources.jsonl").write_text(
                json.dumps({
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_SOURCE",
                    "source_record_sha256": source_hash,
                    "venue_id": "aster",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n",
                encoding="utf-8",
            )
            (request_pack / "source_link_request_targets.jsonl").write_text(
                json.dumps({
                    "label_type": "PHASE51Z_SOURCE_LINK_REQUEST_TARGET",
                    "canonical_group_id": canonical_group_id,
                    "order_key": order_key,
                    "venue_id": "aster",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n",
                encoding="utf-8",
            )
            empty_sidecar_path = request_pack / "source_links.proposed.empty.jsonl"
            empty_sidecar_path.write_text("", encoding="utf-8")
            empty_manifest_path = request_pack / "candidate_manifest_with_empty_sidecar.json"
            empty_manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [{"source_id": "source", "venue_id": "aster", "path": str(source_path)}],
                "source_links": [{"source_link_id": "empty", "path": str(empty_sidecar_path)}],
            }), encoding="utf-8")
            (request_pack / "phase51z_source_link_request_pack_summary.json").write_text(json.dumps({
                "run_id": "phase51z_request_pack",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "target_run": str(target_run),
                "source_path": str(source_path),
                "candidate_manifest_with_empty_sidecar": str(empty_manifest_path),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            mapping_path.write_text(json.dumps({
                "source_record_sha256": source_hash,
                "canonical_group_id": canonical_group_id,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ad_source_link_sidecar_materialize_path()),
                    "--request-pack",
                    str(request_pack),
                    "--mapping",
                    str(mapping_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ad_materialize_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51ad_materialize_test"
            summary = json.loads(
                (run_dir / "phase51ad_source_link_sidecar_materialize_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["source_link_request_source_count"], 1)
            self.assertEqual(summary["materialized_source_link_count"], 1)
            self.assertTrue(summary["candidate_sidecar_complete"])
            self.assertFalse(summary["clears_phase51_blockers"])
            sidecar_text = (run_dir / "source_links.sanitized.jsonl").read_text(encoding="utf-8")
            self.assertIn(source_hash, sidecar_text)
            self.assertIn(canonical_group_id, sidecar_text)
            self.assertNotIn("client_order_id", sidecar_text)
            self.assertNotIn("trade_id", sidecar_text)

            phase51v_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(run_dir / "candidate_manifest_with_materialized_sidecar.json"),
                    "--output-root",
                    str(phase51v_root),
                    "--run-id",
                    "phase51ad_materialize_to_phase51v_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                phase51v_result.returncode,
                0,
                f"stdout: {phase51v_result.stdout}\nstderr: {phase51v_result.stderr}",
            )
            phase51v_summary = json.loads(
                (
                    phase51v_root
                    / "phase51ad_materialize_to_phase51v_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(phase51v_summary["native_role_capture_target_ready_count"], 1)
            self.assertEqual(phase51v_summary["source_link_applied_row_count"], 1)

            rejection_cases = [
                (
                    "unknown_source",
                    {"source_record_sha256": hashlib.sha256(b"unknown").hexdigest(), "canonical_group_id": canonical_group_id},
                    "source hash not found",
                ),
                (
                    "raw_identifier",
                    {"source_record_sha256": source_hash, "canonical_group_id": canonical_group_id, "client_order_id": "raw"},
                    "raw identifier",
                ),
                (
                    "duplicate_source",
                    [
                        {"source_record_sha256": source_hash, "canonical_group_id": canonical_group_id},
                        {"source_record_sha256": source_hash, "canonical_group_id": canonical_group_id},
                    ],
                    "duplicate mapping",
                ),
            ]
            for suffix, payload, expected_error in rejection_cases:
                bad_mapping = tmp_path / f"bad_{suffix}.jsonl"
                rows = payload if isinstance(payload, list) else [payload]
                bad_mapping.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                bad_result = subprocess.run(
                    [
                        sys.executable,
                        str(self._get_phase51ad_source_link_sidecar_materialize_path()),
                        "--request-pack",
                        str(request_pack),
                        "--mapping",
                        str(bad_mapping),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        f"phase51ad_bad_{suffix}",
                        "--timestamp-ns",
                        "1700000000000000000",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(bad_result.returncode, 2)
                self.assertIn(expected_error, bad_result.stderr)

            dirty_manifest_path = request_pack / "candidate_manifest_with_empty_sidecar.json"
            original_manifest_text = dirty_manifest_path.read_text(encoding="utf-8")
            dirty_manifest = json.loads(original_manifest_text)
            dirty_manifest["sources"][0]["client_order_id"] = "raw-client-id"
            dirty_manifest_path.write_text(json.dumps(dirty_manifest), encoding="utf-8")
            dirty_manifest_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ad_source_link_sidecar_materialize_path()),
                    "--request-pack",
                    str(request_pack),
                    "--mapping",
                    str(mapping_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ad_bad_manifest_raw_identifier",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(dirty_manifest_result.returncode, 2)
            self.assertIn("raw identifier", dirty_manifest_result.stderr)
            self.assertFalse(
                (
                    output_root
                    / "phase51ad_bad_manifest_raw_identifier"
                    / "candidate_manifest_with_materialized_sidecar.json"
                ).exists()
            )
            dirty_manifest_path.write_text(original_manifest_text, encoding="utf-8")

    def test_phase51ae_candidate_manifest_compose_validates_all_inputs(self):
        """5.1ae should compose Phase 5.1v manifests without manual stitching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            output_root = tmp_path / "phase51ae"
            phase51v_root = tmp_path / "phase51v"
            aster_source_path = tmp_path / "aster_source.jsonl"
            hyper_source_path = tmp_path / "hyper_source.jsonl"
            source_link_path = tmp_path / "source_links.sanitized.jsonl"
            aster_manifest_path = tmp_path / "aster_manifest.json"
            hyper_manifest_path = tmp_path / "hyper_manifest.json"

            source_hash = hashlib.sha256(b"phase51ae-aster-source").hexdigest()
            aster_group = "group-aster"
            aster_order = "order-aster"
            hyper_group = "group-hyper"
            hyper_order = "order-hyper"
            limit_group = "group-lighter-limit"
            limit_order = "order-lighter-limit"

            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 2,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(
                json.dumps({
                    "canonical_group_id": aster_group,
                    "order_key": aster_order,
                    "venue_id": "aster",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n" + json.dumps({
                    "canonical_group_id": hyper_group,
                    "order_key": hyper_order,
                    "venue_id": "hyperliquid",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n",
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": limit_group,
                "order_key": limit_order,
                "venue_id": "lighter",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            aster_source_path.write_text(json.dumps({
                "venue_id": "aster",
                "source_record_sha256": source_hash,
                "e": "ORDER_TRADE_UPDATE",
                "o": {"m": True, "l": "0.01"},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_link_path.write_text(json.dumps({
                "source_record_sha256": source_hash,
                "canonical_group_id": aster_group,
                "order_key": aster_order,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            hyper_source_path.write_text(json.dumps({
                "venue_id": "hyperliquid",
                "canonical_group_id": hyper_group,
                "order_key": hyper_order,
                "crossed": False,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            aster_manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [{"source_id": "aster_source", "venue_id": "aster", "path": str(aster_source_path)}],
                "source_links": [{"source_link_id": "aster_links", "path": str(source_link_path)}],
            }), encoding="utf-8")
            hyper_manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [{"source_id": "hyper_source", "venue_id": "hyperliquid", "path": str(hyper_source_path)}],
                "source_links": [],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ae_candidate_manifest_compose_path()),
                    "--candidate-manifest",
                    str(aster_manifest_path),
                    "--candidate-manifest",
                    str(hyper_manifest_path),
                    "--target-run",
                    str(target_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ae_compose_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            run_dir = output_root / "phase51ae_compose_test"
            summary = json.loads(
                (run_dir / "phase51ae_candidate_manifest_compose_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["input_candidate_manifest_count"], 2)
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["source_link_count"], 1)
            self.assertFalse(summary["clears_phase51_blockers"])
            composed = json.loads((run_dir / "candidate_manifest.composed.json").read_text(encoding="utf-8"))
            self.assertEqual(len(composed["sources"]), 2)
            self.assertEqual(len(composed["source_links"]), 1)
            self.assertFalse(composed["approved_for_live"])

            phase51v_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(run_dir / "candidate_manifest.composed.json"),
                    "--output-root",
                    str(phase51v_root),
                    "--run-id",
                    "phase51ae_compose_to_phase51v_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                phase51v_result.returncode,
                0,
                f"stdout: {phase51v_result.stdout}\nstderr: {phase51v_result.stderr}",
            )
            phase51v_summary = json.loads(
                (
                    phase51v_root
                    / "phase51ae_compose_to_phase51v_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(phase51v_summary["native_role_capture_target_ready_count"], 2)
            self.assertEqual(phase51v_summary["lighter_native_limit_capture_target_ready_count"], 0)
            self.assertEqual(phase51v_summary["lighter_native_limit_capture_target_missing_count"], 1)
            self.assertEqual(phase51v_summary["source_link_applied_row_count"], 1)
            self.assertFalse(phase51v_summary["generated_phase51s_manifest_ready"])

            unsafe_manifest_path = tmp_path / "unsafe_manifest.json"
            unsafe_payload = json.loads(aster_manifest_path.read_text(encoding="utf-8"))
            unsafe_payload["approved_for_live"] = True
            unsafe_manifest_path.write_text(json.dumps(unsafe_payload), encoding="utf-8")
            unsafe_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51ae_candidate_manifest_compose_path()),
                    "--candidate-manifest",
                    str(unsafe_manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ae_unsafe_manifest_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsafe_result.returncode, 2)
            self.assertIn("unsafe candidate manifest flag approved_for_live=true", unsafe_result.stderr)

    def test_phase51ae_candidate_manifest_compose_rejects_unsafe_inputs(self):
        """5.1ae should fail closed on unsafe or ambiguous composition inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_root = tmp_path / "phase51ae"
            source_path = tmp_path / "source.jsonl"
            source_link_path = tmp_path / "source_links.sanitized.jsonl"
            manifest_path = tmp_path / "candidate_manifest.json"

            source_path.write_text(json.dumps({
                "venue_id": "aster",
                "canonical_group_id": "group-1",
                "order_key": "order-1",
                "m": True,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_link_path.write_text("", encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [{"source_id": "source_a", "venue_id": "aster", "path": str(source_path)}],
                "source_links": [],
            }), encoding="utf-8")

            def run_compose(*args):
                return subprocess.run(
                    [
                        sys.executable,
                        str(self._get_phase51ae_candidate_manifest_compose_path()),
                        "--output-root",
                        str(output_root),
                        *args,
                    ],
                    capture_output=True,
                    text=True,
                )

            source_link_only = run_compose(
                "--source-link",
                f"links={source_link_path}",
                "--run-id",
                "source_link_only",
            )
            self.assertEqual(source_link_only.returncode, 2)
            self.assertIn("source-link-only composition is prohibited", source_link_only.stderr)

            network_source = run_compose(
                "--source",
                "network=aster=https://example.invalid/source.jsonl",
                "--run-id",
                "network_source",
            )
            self.assertEqual(network_source.returncode, 2)
            self.assertIn("network source[0] path is prohibited", network_source.stderr)

            conflicting_duplicate = run_compose(
                "--source",
                f"source_a=aster={source_path}",
                "--source",
                f"source_b=aster={source_path}",
                "--run-id",
                "conflicting_duplicate",
            )
            self.assertEqual(conflicting_duplicate.returncode, 2)
            self.assertIn("conflicting source metadata", conflicting_duplicate.stderr)

            raw_identifier_manifest = tmp_path / "raw_identifier_manifest.json"
            raw_identifier_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_identifier_payload["order_id"] = "raw-order-id"
            raw_identifier_manifest.write_text(json.dumps(raw_identifier_payload), encoding="utf-8")
            raw_identifier = run_compose(
                "--candidate-manifest",
                str(raw_identifier_manifest),
                "--run-id",
                "raw_identifier",
            )
            self.assertEqual(raw_identifier.returncode, 2)
            self.assertIn("candidate manifest leaked raw identifier fields", raw_identifier.stderr)

            secret_manifest = tmp_path / "secret_manifest.json"
            secret_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            secret_payload["api_key"] = "not-a-real-key"
            secret_manifest.write_text(json.dumps(secret_payload), encoding="utf-8")
            secret = run_compose(
                "--candidate-manifest",
                str(secret_manifest),
                "--run-id",
                "secret_manifest",
            )
            self.assertEqual(secret.returncode, 2)
            self.assertIn("secret-shaped candidate manifest field", secret.stderr)

            first = run_compose(
                "--candidate-manifest",
                str(manifest_path),
                "--run-id",
                "deterministic_a",
                "--timestamp-ns",
                "1700000000000000000",
            )
            second = run_compose(
                "--candidate-manifest",
                str(manifest_path),
                "--run-id",
                "deterministic_b",
                "--timestamp-ns",
                "1700000000000000000",
            )
            self.assertEqual(first.returncode, 0, f"stdout: {first.stdout}\nstderr: {first.stderr}")
            self.assertEqual(second.returncode, 0, f"stdout: {second.stdout}\nstderr: {second.stderr}")
            first_summary = json.loads(
                (output_root / "deterministic_a" / "phase51ae_candidate_manifest_compose_summary.json")
                .read_text(encoding="utf-8")
            )
            second_summary = json.loads(
                (output_root / "deterministic_b" / "phase51ae_candidate_manifest_compose_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(first_summary["candidate_manifest_sha256"], second_summary["candidate_manifest_sha256"])

    def test_phase51af_local_source_retrieval_audit_holds_without_join_or_pressure(self):
        """5.1af should prove local raw IDs alone do not clear source-link or pressure blockers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request_pack = tmp_path / "request_pack"
            output_root = tmp_path / "phase51af"
            telemetry_path = tmp_path / "bounded_telemetry.jsonl"
            log_path = tmp_path / "runtime.log"
            request_pack.mkdir()

            source_hash = hashlib.sha256(b"phase51af-extended-source").hexdigest()
            (request_pack / "phase51z_source_link_request_pack_summary.json").write_text(json.dumps({
                "run_id": "phase51z_request_pack",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "source_request_count": 1,
                "target_request_count": 1,
                "next_required_artifact": "validated_redacted_source_link_mapping",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (request_pack / "source_link_request_sources.jsonl").write_text(json.dumps({
                "venue_id": "extended",
                "source_record_sha256": source_hash,
                "isTaker": False,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (request_pack / "source_link_request_targets.jsonl").write_text(json.dumps({
                "venue_id": "extended",
                "canonical_group_id": "group-extended",
                "order_key": "order-extended",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            telemetry_payload = [
                {
                    "orders": [{"client_order_id": "raw-local-id", "order_id": "raw-order-id"}],
                    "fills": [{"decision_id": "raw-decision-id"}],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "tick_timing": {"order_tx_pending": False},
                    "canary_breach_response": {"open_order_count": 1, "max_open_orders": 10},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            telemetry_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in telemetry_payload),
                encoding="utf-8",
            )
            telemetry_hash = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
            log_path.write_text("rate_limit_rps=3 rate_limit_burst=6\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51af_local_source_retrieval_audit_path()),
                    "--request-pack",
                    str(request_pack),
                    "--bounded-telemetry",
                    f"{telemetry_hash}={telemetry_path}",
                    "--log",
                    str(log_path),
                    "--max-log-bytes",
                    "1000000",
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51af_hold_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            summary = json.loads(
                (
                    output_root
                    / "phase51af_hold_test"
                    / "phase51af_local_source_retrieval_audit_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertFalse(summary["local_retrieval_possible_without_inference"])
            self.assertFalse(summary["source_rows_have_join_fields"])
            self.assertEqual(summary["source_link_retrieval_status"], "MISSING_REQUIRED_LINKAGE")
            self.assertEqual(summary["lighter_pressure_retrieval_status"], "MISSING_REQUIRED_PRESSURE_FIELDS")
            self.assertEqual(summary["runtime_log_pattern_status"], "NO_USABLE_PRESSURE_PATTERN")
            self.assertTrue(summary["bounded_telemetry_hashes_match"])
            self.assertGreater(
                summary["bounded_telemetry_audits"][0]["raw_identifier_field_presence_count"],
                0,
            )
            self.assertEqual(
                summary["bounded_telemetry_audits"][0]["lighter_pressure_field_presence_count"],
                0,
            )
            self.assertEqual(summary["log_audits"][0]["pattern_counts"], {"rate_limit": 2})

            network_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51af_local_source_retrieval_audit_path()),
                    "--request-pack",
                    str(request_pack),
                    "--log",
                    "https://example.invalid/log.jsonl",
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51af_network_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(network_result.returncode, 2)
            self.assertIn("network log path is prohibited", network_result.stderr)

    def test_phase51v_forward_capture_bundle_readiness_applies_source_link_sidecar(self):
        """5.1v should use validated source-link sidecars to join source rows to targets."""
        def stable_hash(value):
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "extended-group",
                "order_key": "extended-order",
                "venue_id": "extended",
                "required_native_role_source": "EXTENDED_ISTAKER",
                "required_native_role_fields": ["isTaker"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            source_row = {
                "venue_id": "extended",
                "isTaker": False,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            source_path = tmp_path / "extended_native_source.jsonl"
            source_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
            source_link_path = tmp_path / "source_links.jsonl"
            source_link_path.write_text(json.dumps({
                "source_record_sha256": stable_hash(source_row),
                "canonical_group_id": "extended-group",
                "order_key": "extended-order",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "extended_source_without_direct_join_key",
                        "venue_id": "extended",
                        "path": str(source_path),
                    }
                ],
                "source_links": [
                    {
                        "source_link_id": "phase51t_links",
                        "path": str(source_link_path),
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51v_source_link_ready_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            run_dir = output_root / "phase51v_source_link_ready_test"
            summary = json.loads(
                (run_dir / "phase51v_forward_capture_bundle_readiness_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["native_role_capture_target_ready_count"], 1)
            self.assertEqual(summary["native_role_capture_target_missing_count"], 0)
            self.assertEqual(summary["source_link_file_status_counts"], {"LOCAL_FILE_READY": 1})
            self.assertEqual(summary["source_link_hash_count"], 1)
            self.assertEqual(summary["source_link_applied_row_count"], 1)
            self.assertTrue(summary["generated_phase51s_manifest_ready"])
            labels = (run_dir / "capture_bundle_readiness_labels.jsonl").read_text(encoding="utf-8")
            self.assertIn('"role_target_join_status":"SOURCE_LINK_SIDECAR"', labels)
            self.assertIn('"source_link_applied":true', labels)
            self.assertFalse(summary["clears_phase51_blockers"])

    def test_phase51v_forward_capture_bundle_readiness_source_link_only_holds(self):
        """5.1v should not treat source-link sidecars alone as native source evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "extended-group",
                "order_key": "extended-order",
                "venue_id": "extended",
                "required_native_role_source": "EXTENDED_ISTAKER",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            source_link_path = tmp_path / "source_links.jsonl"
            source_link_path.write_text(json.dumps({
                "source_record_sha256": "a" * 64,
                "canonical_group_id": "extended-group",
                "order_key": "extended-order",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [],
                "source_links": [
                    {
                        "source_link_id": "phase51t_links",
                        "path": str(source_link_path),
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51v_source_link_only_hold_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            summary = json.loads(
                (
                    output_root
                    / "phase51v_source_link_only_hold_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["native_role_capture_target_ready_count"], 0)
            self.assertEqual(summary["native_role_capture_target_missing_count"], 1)
            self.assertEqual(summary["source_link_hash_count"], 1)
            self.assertEqual(summary["source_link_applied_row_count"], 0)
            self.assertFalse(summary["generated_phase51s_manifest_ready"])

    def test_phase51v_forward_capture_bundle_readiness_rejects_duplicate_source_link_hash(self):
        """5.1v should reject ambiguous source-link hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "extended-group",
                "order_key": "extended-order",
                "venue_id": "extended",
                "required_native_role_source": "EXTENDED_ISTAKER",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            source_link_path = tmp_path / "duplicate_source_links.jsonl"
            source_link_path.write_text(
                json.dumps({
                    "source_record_sha256": "b" * 64,
                    "canonical_group_id": "extended-group",
                    "order_key": "extended-order",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n"
                + json.dumps({
                    "source_record_sha256": "b" * 64,
                    "canonical_group_id": "extended-group",
                    "order_key": "extended-order",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n",
                encoding="utf-8",
            )
            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [],
                "source_links": [
                    {
                        "source_link_id": "phase51t_links",
                        "path": str(source_link_path),
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(tmp_path / "phase51v_readiness"),
                    "--run-id",
                    "phase51v_duplicate_source_link_hash_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate source link hash", result.stderr)

    def test_phase51v_forward_capture_bundle_readiness_holds_template_placeholders(self):
        """5.1v should keep placeholder capture manifests in HOLD with no ready targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "hyper-group",
                "order_key": "hyper-order",
                "venue_id": "hyperliquid",
                "required_native_role_source": "HYPERLIQUID_CROSSED",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "lighter-group",
                "order_key": "lighter-order",
                "venue_id": "lighter",
                "required_native_limit_source": "LIGHTER_LIMITS_AT_DECISION_TIME",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            candidate_manifest = tmp_path / "capture_bundle_manifest_template.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "hyperliquid_private_fills",
                        "venue_id": "hyperliquid",
                        "path": "<local_sanitized_hyperliquid_private_fills.jsonl>",
                    }
                ],
                "source_links": [
                    {
                        "source_link_id": "optional_links",
                        "path": "<optional_phase51t_source_link_sidecar.jsonl>",
                    }
                ],
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51v_template_hold_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            summary = json.loads(
                (
                    output_root
                    / "phase51v_template_hold_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_reason"], "phase51v_forward_capture_bundle_incomplete_nonlive_hold")
            self.assertEqual(summary["native_role_capture_target_ready_count"], 0)
            self.assertEqual(summary["lighter_native_limit_capture_target_ready_count"], 0)
            self.assertEqual(summary["source_file_status_counts"], {"PLACEHOLDER_PATH": 1})
            self.assertEqual(summary["source_link_file_status_counts"], {"PLACEHOLDER_PATH": 1})
            self.assertFalse(summary["generated_phase51s_manifest_ready"])
            self.assertEqual(summary["generated_phase51s_source_count"], 0)

    def test_phase51v_forward_capture_bundle_readiness_rejects_unsafe_manifest(self):
        """5.1v should reject network paths and secret-shaped manifest fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            secret_manifest = tmp_path / "secret_manifest.json"
            secret_manifest.write_text(json.dumps({
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "api_key": "redacted",
                "approved_for_live": False,
            }), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(secret_manifest),
                    "--output-root",
                    str(tmp_path / "phase51v_readiness"),
                    "--run-id",
                    "phase51v_unsafe_secret_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("secret-shaped candidate manifest field", result.stderr)

            network_manifest = tmp_path / "network_manifest.json"
            network_manifest.write_text(json.dumps({
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "sources": [
                    {
                        "source_id": "network_source",
                        "venue_id": "hyperliquid",
                        "path": "https://example.com/private_fills.jsonl",
                    }
                ],
            }), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(network_manifest),
                    "--output-root",
                    str(tmp_path / "phase51v_readiness"),
                    "--run-id",
                    "phase51v_unsafe_network_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("network source[0] paths are prohibited", result.stderr)

    def test_phase51x_hyperliquid_native_role_adapter_emits_phase51v_ready_rows(self):
        """5.1x should redact local Hyperliquid fills and feed 5.1v structurally."""
        def stable_hash(value):
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            adapter_root = tmp_path / "phase51x_hyperliquid"
            readiness_root = tmp_path / "phase51v_readiness"
            observed_run.mkdir()
            target_run.mkdir()
            raw_client_order_id = "client-hl-1"
            client_order_id_hash = stable_hash(raw_client_order_id)
            observed_label = {
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "hl-group",
                "order_key": "hl-order",
                "venue_id": "hyperliquid",
                "fill_count": 1,
                "client_order_id_hash": client_order_id_hash,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51x_hl_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(
                json.dumps(observed_label) + "\n",
                encoding="utf-8",
            )
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_hl_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "hl-group",
                "order_key": "hl-order",
                "venue_id": "hyperliquid",
                "required_native_role_source": "HYPERLIQUID_CROSSED",
                "required_native_role_fields": ["crossed"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            source_path = tmp_path / "hyperliquid_user_fills.json"
            source_path.write_text(json.dumps({
                "fills": [
                    {
                        "cloid": raw_client_order_id,
                        "oid": 12345,
                        "tid": 67890,
                        "coin": "ETH",
                        "px": "3200.0",
                        "sz": "0.01",
                        "crossed": False,
                    }
                ]
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51x_hyperliquid_native_role_adapter_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(adapter_root),
                    "--run-id",
                    "phase51x_hl_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            adapter_dir = adapter_root / "phase51x_hl_test"
            summary = json.loads(
                (adapter_dir / "phase51x_hyperliquid_native_role_adapter_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["hyperliquid_target_count"], 1)
            self.assertEqual(summary["hyperliquid_target_recovered_count"], 1)
            self.assertEqual(summary["source_row_emitted_count"], 1)
            self.assertFalse(summary["live_orders_allowed"])
            self.assertFalse(summary["admissible_for_financial_claim"])
            output_text = (
                (adapter_dir / "hyperliquid_forward_native_role_snapshot.jsonl").read_text(encoding="utf-8")
                + (adapter_dir / "hyperliquid_native_role_adapter_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertIn('"crossed":false', output_text)
            self.assertNotIn("client-hl-1", output_text)
            self.assertNotIn('"cloid"', output_text)
            self.assertNotIn('"oid"', output_text)
            self.assertNotIn('"tid"', output_text)

            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "hyperliquid_forward_native_role_snapshot",
                        "venue_id": "hyperliquid",
                        "path": str(adapter_dir / "hyperliquid_forward_native_role_snapshot.jsonl"),
                    }
                ],
                "source_links": [],
            }), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_from_phase51x_hl_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_from_phase51x_hl_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(readiness_summary["generated_phase51s_manifest_ready"])
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 1)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 0)
            self.assertEqual(readiness_summary["source_file_status_counts"], {"LOCAL_FILE_READY": 1})
            self.assertFalse(readiness_summary["clears_phase51_blockers"])

    def test_phase51x_hyperliquid_native_role_adapter_rejects_network_sources(self):
        """5.1x should not fetch private source rows or accept network paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            observed_run.mkdir()
            target_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51x_network_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text("", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_empty_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51x_hyperliquid_native_role_adapter_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    "https://api.hyperliquid.example/fills.json",
                    "--output-root",
                    str(tmp_path / "phase51x_hyperliquid"),
                    "--run-id",
                    "phase51x_hl_network_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("network source paths are prohibited", result.stderr)

    def test_phase51y_all5_native_role_adapter_emits_phase51v_ready_rows(self):
        """5.1y should redact local all-venue native-role rows and feed 5.1v."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            adapter_root = tmp_path / "phase51y_all5"
            readiness_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            targets = [
                {
                    "canonical_group_id": "aster-group",
                    "order_key": "aster-order",
                    "venue_id": "aster",
                    "required_native_role_source": "ASTER_ORDER_TRADE_UPDATE",
                    "required_native_role_fields": ["e=ORDER_TRADE_UPDATE", "o.m", "positive o.l"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "extended-group",
                    "order_key": "extended-order",
                    "venue_id": "extended",
                    "required_native_role_source": "EXTENDED_ISTAKER",
                    "required_native_role_fields": ["isTaker"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "hyper-group",
                    "order_key": "hyper-order",
                    "venue_id": "hyperliquid",
                    "required_native_role_source": "HYPERLIQUID_CROSSED",
                    "required_native_role_fields": ["crossed"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_role_source": "LIGHTER_TRADES_JSON",
                    "required_native_role_fields": [
                        "account_index",
                        "is_maker_ask",
                        "ask_account_id",
                        "bid_account_id",
                    ],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "paradex-group",
                    "order_key": "paradex-order",
                    "venue_id": "paradex",
                    "required_native_role_source": "PARADEX_LIQUIDITY",
                    "required_native_role_fields": ["liquidity"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_all5_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 5,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in targets),
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            source_path = tmp_path / "all5_native_source.jsonl"
            source_rows = [
                {
                    "canonical_group_id": "aster-group",
                    "venue_id": "aster",
                    "e": "ORDER_TRADE_UPDATE",
                    "o": {"m": True, "l": "0.25", "orderId": "raw-aster-order"},
                },
                {
                    "canonical_group_id": "extended-group",
                    "venue_id": "extended",
                    "isTaker": False,
                    "client_order_id": "raw-extended-client",
                },
                {
                    "canonical_group_id": "hyper-group",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "oid": "raw-hyper-oid",
                },
                {
                    "canonical_group_id": "lighter-group",
                    "venue_id": "lighter",
                    "account_index": "42",
                    "is_maker_ask": True,
                    "ask_account_id": "42",
                    "bid_account_id": "7",
                    "trade_id": "raw-lighter-trade",
                },
                {
                    "canonical_group_id": "paradex-group",
                    "venue_id": "paradex",
                    "liquidity": "TAKER",
                    "order_id": "raw-paradex-order",
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51y_all5_native_role_adapter_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(adapter_root),
                    "--run-id",
                    "phase51y_all5_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            adapter_dir = adapter_root / "phase51y_all5_test"
            summary = json.loads(
                (adapter_dir / "phase51y_all5_native_role_adapter_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["native_role_target_count"], 5)
            self.assertEqual(summary["native_role_target_recovered_count"], 5)
            self.assertEqual(summary["source_row_emitted_count"], 5)
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertFalse(summary["live_orders_allowed"])
            self.assertFalse(summary["admissible_for_financial_claim"])
            output_text = (
                (adapter_dir / "all5_forward_native_role_snapshot.jsonl").read_text(encoding="utf-8")
                + (adapter_dir / "all5_native_role_adapter_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertIn('"liquidity":"TAKER"', output_text)
            self.assertIn('"crossed":false', output_text)
            self.assertIn('"isTaker":false', output_text)
            self.assertNotIn("raw-aster-order", output_text)
            self.assertNotIn("raw-extended-client", output_text)
            self.assertNotIn("raw-hyper-oid", output_text)
            self.assertNotIn("raw-lighter-trade", output_text)
            self.assertNotIn("raw-paradex-order", output_text)
            self.assertNotIn('"orderId"', output_text)
            self.assertNotIn('"client_order_id"', output_text)
            self.assertNotIn('"oid"', output_text)
            self.assertNotIn('"trade_id"', output_text)
            self.assertNotIn('"order_id"', output_text)

            candidate_manifest = tmp_path / "capture_bundle_manifest.json"
            candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "all5_forward_native_role_snapshot",
                        "venue_id": "all5",
                        "path": str(adapter_dir / "all5_forward_native_role_snapshot.jsonl"),
                    }
                ],
                "source_links": [],
            }), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_from_phase51y_all5_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_from_phase51y_all5_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(readiness_summary["generated_phase51s_manifest_ready"])
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 5)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 0)
            self.assertEqual(readiness_summary["source_file_status_counts"], {"LOCAL_FILE_READY": 1})
            self.assertFalse(readiness_summary["clears_phase51_blockers"])

    def test_phase51y_all5_native_role_adapter_rejects_network_sources(self):
        """5.1y should not fetch native source rows or accept network paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_empty_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51y_all5_native_role_adapter_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    "https://api.venue.example/private_trades.json",
                    "--output-root",
                    str(tmp_path / "phase51y_all5"),
                    "--run-id",
                    "phase51y_all5_network_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("network source paths are prohibited", result.stderr)

    def test_phase51z_readonly_native_role_capture_maps_raw_rows_to_sanitized_bundle(self):
        """5.1z should map read-only native rows and prehashed Lighter rows without leaking raw IDs."""
        def stable_hash(value):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            capture_root = tmp_path / "phase51z_capture"
            readiness_root = tmp_path / "phase51v_readiness"
            observed_run.mkdir()
            target_run.mkdir()
            raw_ids = {
                "aster": "aster-client-raw",
                "extended": "extended-client-raw",
                "lighter": "123456789",
                "lighter_order": "987654321",
                "paradex": "paradex-client-raw",
            }
            target_rows = []
            pfill_rows = []
            for seq, venue in enumerate(("aster", "extended", "lighter", "paradex"), start=1):
                group = f"{venue}-group"
                order_key = f"{venue}-order-key"
                target_rows.append({
                    "canonical_group_id": group,
                    "order_key": order_key,
                    "venue_id": venue,
                    "required_native_role_source": "VENUE_NATIVE_FILL_FIELD",
                    "required_native_role_fields": ["native role field"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                })
                pfill_rows.append({
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "label_seq": seq,
                    "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                    "canonical_group_id": group,
                    "order_key": order_key,
                    "venue_id": venue,
                    "client_order_id_hash": stable_hash(raw_ids[venue]) if venue != "lighter" else None,
                    "order_id_hash": stable_hash(raw_ids["lighter_order"]) if venue == "lighter" else None,
                    "fill_count": 1,
                    "first_fill_time_ms": 1700000000000 + seq,
                    "last_fill_time_ms": 1700000000000 + seq,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                    "live_orders_allowed": False,
                    "capital_change_allowed": False,
                    "risk_limit_relaxation_allowed": False,
                })
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51z_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(pfill_rows),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in pfill_rows),
                encoding="utf-8",
            )
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51z_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "observed_pfill_run": str(observed_run),
                "native_role_capture_target_count": len(target_rows),
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in target_rows),
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            source_path = tmp_path / "raw_readonly_native_rows.jsonl"
            source_rows = [
                {
                    "venue_id": "aster",
                    "clientOrderId": raw_ids["aster"],
                    "orderId": "raw-aster-order",
                    "maker": True,
                    "qty": "0.01",
                },
                {
                    "venue_id": "extended",
                    "externalId": raw_ids["extended"],
                    "id": "raw-extended-trade",
                    "isTaker": False,
                },
                {
                    "venue_id": "lighter",
                    "account_index": 42,
                    "ask_account_id": 42,
                    "bid_account_id": 7,
                    "ask_id_sha256": stable_hash(int(raw_ids["lighter_order"])),
                    "ask_id_str_sha256": stable_hash(raw_ids["lighter_order"]),
                    "is_maker_ask": True,
                    "trade_id_sha256": stable_hash("raw-lighter-trade"),
                },
                {
                    "venue_id": "paradex",
                    "client_id": raw_ids["paradex"],
                    "order_id": "raw-paradex-order",
                    "liquidity": "TAKER",
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_readonly_native_role_capture_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51z_capture_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            capture_dir = capture_root / "phase51z_capture_test"
            summary = json.loads(
                (capture_dir / "phase51z_readonly_native_role_capture_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["sanitized_source_row_count"], 4)
            self.assertEqual(summary["raw_identifier_redaction_status"], "PASS")
            self.assertFalse(summary["live_orders_allowed"])
            diagnostics = summary["capture_diagnostics_by_venue"]
            self.assertEqual(diagnostics["aster"]["target_count"], 1)
            self.assertEqual(diagnostics["aster"]["target_ready_count"], 1)
            self.assertEqual(diagnostics["aster"]["target_missing_count"], 0)
            self.assertEqual(diagnostics["aster"]["source_row_count"], 1)
            self.assertEqual(diagnostics["aster"]["native_field_ready_count"], 1)
            self.assertEqual(diagnostics["aster"]["target_matched_row_count"], 1)
            self.assertEqual(diagnostics["aster"]["duplicate_matched_row_count"], 0)
            self.assertEqual(diagnostics["aster"]["no_target_match_count"], 0)
            self.assertEqual(diagnostics["aster"]["rows_with_redacted_hash_candidates"], 1)
            self.assertEqual(
                diagnostics["aster"]["capture_status_counts"],
                {"SANITIZED_SOURCE_ROW_EMITTED": 1},
            )
            self.assertEqual(diagnostics["extended"]["target_ready_count"], 1)
            self.assertEqual(diagnostics["lighter"]["target_ready_count"], 1)
            self.assertEqual(diagnostics["paradex"]["target_ready_count"], 1)
            output_text = (
                (capture_dir / "source_snapshots" / "phase51z_forward_native_role_rows.jsonl").read_text(encoding="utf-8")
                + (capture_dir / "phase51z_readonly_native_role_capture_labels.jsonl").read_text(encoding="utf-8")
            )
            for raw in raw_ids.values():
                self.assertNotIn(str(raw), output_text)
            self.assertNotIn("raw-aster-order", output_text)
            self.assertNotIn("raw-extended-trade", output_text)
            self.assertNotIn("raw-lighter-trade", output_text)
            self.assertNotIn("raw-paradex-order", output_text)

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(capture_dir / "phase51z_candidate_manifest.json"),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_from_phase51z_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_from_phase51z_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(readiness_summary["generated_phase51s_manifest_ready"])
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 4)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 0)
            self.assertFalse(readiness_summary["clears_phase51_blockers"])

    def test_phase51aa_lighter_ws_snapshot_feeds_phase51z_without_raw_ids(self):
        """5.1aa should sanitize Lighter WS trades and feed existing 5.1z/5.1v gates."""
        def stable_hash(value):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            ws_root = tmp_path / "phase51aa_ws"
            capture_root = tmp_path / "phase51z_capture"
            readiness_root = tmp_path / "phase51v_readiness"
            observed_run.mkdir()
            target_run.mkdir()
            raw_lighter_order_id = 987654321
            raw_lighter_bid_id = 123456789
            raw_trade_id = 555666777
            raw_tx_hash = "0x" + "a" * 64

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51aa_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "schema_version": 1,
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "label_seq": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "canonical_group_id": "lighter-group",
                "order_key": "lighter-order-key",
                "venue_id": "lighter",
                "order_id_hash": stable_hash(raw_lighter_order_id),
                "client_order_id_hash": None,
                "fill_count": 1,
                "first_fill_time_ms": 1700000000000,
                "last_fill_time_ms": 1700000000000,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
            }) + "\n", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51aa_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "observed_pfill_run": str(observed_run),
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "lighter-group",
                "order_key": "lighter-order-key",
                "venue_id": "lighter",
                "required_native_role_source": "LIGHTER_TRADES_JSON",
                "required_native_role_fields": [
                    "account_index",
                    "is_maker_ask",
                    "ask_account_id",
                    "bid_account_id",
                ],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            message_path = tmp_path / "ws_account_all_trades.json"
            message_path.write_text(json.dumps({
                "channel": "account_all_trades:42",
                "type": "update/account_all_trades",
                "trades": {
                    "0": [
                        {
                            "trade_id": raw_trade_id,
                            "tx_hash": raw_tx_hash,
                            "type": "trade",
                            "market_id": 0,
                            "size": "0.01",
                            "price": "100.0",
                            "ask_id": raw_lighter_order_id,
                            "bid_id": raw_lighter_bid_id,
                            "ask_account_id": 42,
                            "bid_account_id": 7,
                            "is_maker_ask": True,
                            "timestamp": 1700000000,
                        }
                    ],
                },
            }), encoding="utf-8")

            ws_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51aa_lighter_ws_account_trades_snapshot_path()),
                    "--target-run",
                    str(target_run),
                    "--message-json",
                    str(message_path),
                    "--account-index",
                    "42",
                    "--output-root",
                    str(ws_root),
                    "--run-id",
                    "phase51aa_ws_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                ws_result.returncode,
                0,
                f"stdout: {ws_result.stdout}\nstderr: {ws_result.stderr}",
            )
            ws_dir = ws_root / "phase51aa_ws_test"
            ws_summary = json.loads(
                (ws_dir / "phase51aa_lighter_ws_account_trades_snapshot_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ws_summary["gate_status"], "HOLD")
            self.assertEqual(ws_summary["trade_count"], 1)
            self.assertEqual(ws_summary["raw_identifier_redaction_status"], "PASS")
            self.assertEqual(ws_summary["raw_identifier_key_violation_count"], 0)
            source_path = ws_dir / "source_snapshots" / "lighter_ws_account_trades.sanitized.jsonl"
            output_text = source_path.read_text(encoding="utf-8")
            self.assertNotIn(str(raw_lighter_order_id), output_text)
            self.assertNotIn(str(raw_lighter_bid_id), output_text)
            self.assertNotIn(str(raw_trade_id), output_text)
            self.assertNotIn(raw_tx_hash, output_text)
            self.assertIn("ask_id_sha256", output_text)

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_readonly_native_role_capture_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    f"lighter={source_path}",
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51aa_to_phase51z_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_dir = capture_root / "phase51aa_to_phase51z_test"
            capture_summary = json.loads(
                (capture_dir / "phase51z_readonly_native_role_capture_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["capture_diagnostics_by_venue"]["lighter"]["target_ready_count"], 1)

            readiness_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(capture_dir / "phase51z_candidate_manifest.json"),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51aa_to_phase51v_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                readiness_result.returncode,
                0,
                f"stdout: {readiness_result.stdout}\nstderr: {readiness_result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51aa_to_phase51v_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 1)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 0)
            self.assertTrue(readiness_summary["generated_phase51s_manifest_ready"])
            self.assertFalse(readiness_summary["clears_phase51_blockers"])

    def test_phase51aa_lighter_ws_snapshot_rejects_unsafe_messages(self):
        """5.1aa should reject offline WS messages that try to promote unsafe flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51aa_unsafe_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")
            message_path = tmp_path / "unsafe_ws_message.json"
            message_path.write_text(json.dumps({
                "channel": "account_all_trades:42",
                "type": "update/account_all_trades",
                "approved_for_live": True,
                "trades": {},
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51aa_lighter_ws_account_trades_snapshot_path()),
                    "--target-run",
                    str(target_run),
                    "--message-json",
                    str(message_path),
                    "--account-index",
                    "42",
                    "--output-root",
                    str(tmp_path / "phase51aa_ws"),
                    "--run-id",
                    "phase51aa_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe input message flag approved_for_live=true", result.stderr)

    def test_phase51aa_lighter_ws_snapshot_accepts_account_all_source_channel(self):
        """5.1aa should accept account_all snapshots as a read-only Lighter trade source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51aa_account_all_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            message_path = tmp_path / "ws_account_all.json"
            message_path.write_text(json.dumps({
                "channel": "account_all:42",
                "type": "update/account_all",
                "assets": [
                    {
                        "asset_id": 3,
                        "symbol": "USDC",
                    }
                ],
                "trades": {
                    "0": [
                        {
                            "trade_id": 111222333,
                            "market_id": 0,
                            "size": "0.01",
                            "price": "100.0",
                            "ask_id": 444555666,
                            "bid_id": 777888999,
                            "ask_account_id": 42,
                            "bid_account_id": 7,
                            "is_maker_ask": False,
                            "timestamp": 1700000000,
                        }
                    ],
                },
            }), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51aa_lighter_ws_account_trades_snapshot_path()),
                    "--target-run",
                    str(target_run),
                    "--message-json",
                    str(message_path),
                    "--account-index",
                    "42",
                    "--channel",
                    "account_all",
                    "--output-root",
                    str(tmp_path / "phase51aa_ws"),
                    "--run-id",
                    "phase51aa_account_all_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            ws_dir = tmp_path / "phase51aa_ws" / "phase51aa_account_all_test"
            summary = json.loads(
                (ws_dir / "phase51aa_lighter_ws_account_trades_snapshot_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["requested_channels"], ["account_all"])
            self.assertEqual(summary["trade_count"], 1)
            rows = [
                json.loads(line)
                for line in (ws_dir / "source_snapshots" / "lighter_ws_account_trades.sanitized.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(rows[0]["source_channel"], "account_all")
            self.assertIn("ask_id_sha256", rows[0])
            self.assertNotIn("ask_id", rows[0])
            sanitized_message_path = (
                ws_dir / "source_snapshots" / "lighter_ws_messages.sanitized.jsonl"
            )
            sanitized_message_text = sanitized_message_path.read_text(encoding="utf-8")
            message_rows = [json.loads(line) for line in sanitized_message_text.splitlines() if line]
            self.assertEqual(message_rows[0]["source_channel"], "account_all")
            self.assertEqual(message_rows[0]["trade_row_count"], 1)
            self.assertIn("assets", message_rows[0]["top_level_keys"])
            self.assertNotIn('"asset_id":', sanitized_message_text)
            self.assertNotIn('"assets":', sanitized_message_text)

    def test_phase51z_readonly_native_role_capture_rejects_network_sources(self):
        """5.1z should refuse network source paths before reading any source rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            observed_run.mkdir()
            target_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51z_network_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text("", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_empty_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "observed_pfill_run": str(observed_run),
                "native_role_capture_target_count": 0,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text("", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_readonly_native_role_capture_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    "https://api.venue.example/private_fills.json",
                    "--output-root",
                    str(tmp_path / "phase51z_capture"),
                    "--run-id",
                    "phase51z_network_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("network source paths are prohibited", result.stderr)

    def test_phase51z_readonly_native_role_capture_reports_no_target_match_diagnostics(self):
        """5.1z diagnostics should show source coverage without leaking unmatched IDs."""
        def stable_hash(value):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            capture_root = tmp_path / "phase51z_capture"
            observed_run.mkdir()
            target_run.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51z_diagnostics_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "schema_version": 1,
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "label_seq": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "canonical_group_id": "aster-target-group",
                "order_key": "aster-target-order",
                "venue_id": "aster",
                "client_order_id_hash": stable_hash("target-client-id"),
                "order_id_hash": None,
                "fill_count": 1,
                "first_fill_time_ms": 1700000000000,
                "last_fill_time_ms": 1700000000000,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
            }) + "\n", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51z_diagnostics_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "observed_pfill_run": str(observed_run),
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "aster-target-group",
                "order_key": "aster-target-order",
                "venue_id": "aster",
                "required_native_role_source": "ASTER_ORDER_TRADE_UPDATE_M",
                "required_native_role_fields": ["o.m"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            source_path = tmp_path / "raw_unmatched_aster_rows.jsonl"
            source_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [
                        {
                            "venue_id": "aster",
                            "clientOrderId": "unmatched-client-1",
                            "maker": True,
                            "qty": "0.01",
                        },
                        {
                            "venue_id": "aster",
                            "clientOrderId": "unmatched-client-2",
                            "maker": False,
                            "qty": "0.02",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_readonly_native_role_capture_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51z_diagnostics_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            capture_dir = capture_root / "phase51z_diagnostics_test"
            summary = json.loads(
                (capture_dir / "phase51z_readonly_native_role_capture_summary.json").read_text(encoding="utf-8")
            )
            diagnostics = summary["capture_diagnostics_by_venue"]["aster"]
            self.assertEqual(summary["sanitized_source_row_count"], 0)
            self.assertEqual(diagnostics["target_count"], 1)
            self.assertEqual(diagnostics["target_ready_count"], 0)
            self.assertEqual(diagnostics["target_missing_count"], 1)
            self.assertEqual(diagnostics["source_row_count"], 2)
            self.assertEqual(diagnostics["native_field_ready_count"], 2)
            self.assertEqual(diagnostics["target_matched_row_count"], 0)
            self.assertEqual(diagnostics["duplicate_matched_row_count"], 0)
            self.assertEqual(diagnostics["no_target_match_count"], 2)
            self.assertEqual(diagnostics["rows_with_redacted_hash_candidates"], 2)
            self.assertEqual(diagnostics["target_match_status_counts"], {"NO_TARGET_MATCH": 2})
            output_text = (
                (capture_dir / "phase51z_readonly_native_role_capture_summary.json").read_text(encoding="utf-8")
                + (capture_dir / "phase51z_readonly_native_role_capture_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertNotIn("unmatched-client-1", output_text)
            self.assertNotIn("unmatched-client-2", output_text)

    def test_phase51z_unlinked_lighter_rows_require_source_link_sidecar(self):
        """5.1z can preserve sanitized unlinked Lighter rows without clearing 5.1v."""
        def stable_hash(value):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            target_run = tmp_path / "phase51u_targets"
            capture_root = tmp_path / "phase51z_capture"
            readiness_root = tmp_path / "phase51v_readiness"
            observed_run.mkdir()
            target_run.mkdir()
            target_group = "lighter-unlinked-group"
            target_order_key = "lighter-unlinked-order"
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51z_unlinked_lighter_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "schema_version": 1,
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "label_seq": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "canonical_group_id": target_group,
                "order_key": target_order_key,
                "venue_id": "lighter",
                "order_id_hash": stable_hash("target-native-order-id"),
                "client_order_id_hash": stable_hash("target-client-order-id"),
                "fill_count": 1,
                "first_fill_time_ms": 1700000000000,
                "last_fill_time_ms": 1700000000000,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
            }) + "\n", encoding="utf-8")
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_phase51z_unlinked_lighter_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "observed_pfill_run": str(observed_run),
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": target_group,
                "order_key": target_order_key,
                "venue_id": "lighter",
                "required_native_role_source": "LIGHTER_TRADES_JSON",
                "required_native_role_fields": ["is_maker_ask"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            raw_source_row = {
                "venue_id": "lighter",
                "account_index": 42,
                "ask_account_id": 42,
                "bid_account_id": 7,
                "ask_id": "unmatched-native-order-id",
                "ask_client_id": "unmatched-client-order-id",
                "is_maker_ask": True,
                "trade_id": "raw-lighter-trade-id",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }
            source_path = tmp_path / "raw_unmatched_lighter_rows.jsonl"
            source_path.write_text(json.dumps(raw_source_row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_readonly_native_role_capture_path()),
                    "--target-run",
                    str(target_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51z_unlinked_lighter_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--emit-unlinked-native-role-source-rows",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            capture_dir = capture_root / "phase51z_unlinked_lighter_test"
            summary = json.loads(
                (capture_dir / "phase51z_readonly_native_role_capture_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["sanitized_source_row_count"], 1)
            self.assertEqual(summary["sanitized_source_row_counts_by_venue"], {})
            self.assertTrue(summary["emit_unlinked_native_role_source_rows"])
            self.assertEqual(summary["unlinked_sanitized_source_row_count"], 1)
            self.assertEqual(summary["unlinked_sanitized_source_row_counts_by_venue"], {"lighter": 1})
            self.assertEqual(
                summary["capture_status_counts"],
                {"SANITIZED_UNLINKED_SOURCE_ROW_EMITTED": 1},
            )
            diagnostics = summary["capture_diagnostics_by_venue"]["lighter"]
            self.assertEqual(diagnostics["target_ready_count"], 0)
            self.assertEqual(diagnostics["target_missing_count"], 1)
            self.assertEqual(diagnostics["unlinked_sanitized_source_row_count"], 1)

            sanitized_source_path = capture_dir / "source_snapshots" / "phase51z_forward_native_role_rows.jsonl"
            sanitized_rows = [
                json.loads(line)
                for line in sanitized_source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(sanitized_rows), 1)
            sanitized_row = sanitized_rows[0]
            self.assertEqual(sanitized_row["label_type"], "PHASE51Z_UNLINKED_NATIVE_ROLE_SOURCE")
            self.assertNotIn("canonical_group_id", sanitized_row)
            self.assertNotIn("order_key", sanitized_row)
            self.assertEqual(sanitized_row["venue_id"], "lighter")
            self.assertEqual(sanitized_row["account_index"], 42)
            self.assertIsInstance(sanitized_row["source_record_sha256"], str)
            output_text = (
                sanitized_source_path.read_text(encoding="utf-8")
                + (capture_dir / "phase51z_readonly_native_role_capture_labels.jsonl").read_text(encoding="utf-8")
            )
            self.assertNotIn("unmatched-native-order-id", output_text)
            self.assertNotIn("unmatched-client-order-id", output_text)
            self.assertNotIn("raw-lighter-trade-id", output_text)

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(capture_dir / "phase51z_candidate_manifest.json"),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_unlinked_lighter_without_sidecar_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            no_sidecar_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_unlinked_lighter_without_sidecar_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(no_sidecar_summary["native_role_capture_target_ready_count"], 0)
            self.assertEqual(no_sidecar_summary["native_role_capture_target_missing_count"], 1)
            self.assertFalse(no_sidecar_summary["generated_phase51s_manifest_ready"])

            source_link_path = tmp_path / "source_links.jsonl"
            source_link_path.write_text(json.dumps({
                "source_record_sha256": sanitized_row["source_record_sha256"],
                "canonical_group_id": target_group,
                "order_key": target_order_key,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            linked_candidate_manifest = tmp_path / "linked_candidate_manifest.json"
            linked_candidate_manifest.write_text(json.dumps({
                "manifest_version": 1,
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_model_training": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "sources": [
                    {
                        "source_id": "phase51z_unlinked_lighter_rows",
                        "venue_id": "lighter",
                        "path": str(sanitized_source_path),
                    }
                ],
                "source_links": [
                    {
                        "source_link_id": "operator_validated_source_link",
                        "path": str(source_link_path),
                    }
                ],
            }), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(linked_candidate_manifest),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_unlinked_lighter_with_sidecar_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            linked_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_unlinked_lighter_with_sidecar_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(linked_summary["native_role_capture_target_ready_count"], 1)
            self.assertEqual(linked_summary["native_role_capture_target_missing_count"], 0)
            self.assertEqual(linked_summary["source_link_applied_row_count"], 1)
            self.assertTrue(linked_summary["generated_phase51s_manifest_ready"])
            linked_labels = (
                readiness_root
                / "phase51v_unlinked_lighter_with_sidecar_test"
                / "capture_bundle_readiness_labels.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn('"role_target_join_status":"SOURCE_LINK_SIDECAR"', linked_labels)
            self.assertIn('"source_link_applied":true', linked_labels)

    def test_phase51z_source_link_request_pack_emits_hold_only_pack(self):
        """5.1z should package unlinked Lighter rows for sidecar review only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            source_run = tmp_path / "phase51z_unlinked"
            output_root = tmp_path / "request_pack"
            readiness_root = tmp_path / "phase51v_readiness"
            target_run.mkdir()
            source_dir = source_run / "source_snapshots"
            source_dir.mkdir(parents=True)
            target_group = "lighter-source-link-request-group"
            target_order_key = "lighter-source-link-request-order"
            source_hash = "a" * 64
            aster_source_hash = "c" * 64
            raw_should_not_appear = "raw-lighter-trade-id"

            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_source_link_request_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 1,
                "lighter_native_limit_capture_target_count": 0,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "schema_version": 1,
                "label_type": "PHASE51U_NATIVE_ROLE_CAPTURE_TARGET",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "canonical_group_id": target_group,
                "order_key": target_order_key,
                "venue_id": "lighter",
                "required_native_role_source": "LIGHTER_TRADES_JSON",
                "required_native_role_fields": ["account_index", "is_maker_ask", "ask_account_id", "bid_account_id"],
                "first_fill_time_ms": 1700000000000,
                "last_fill_time_ms": 1700000000000,
                "side": "ASK",
                "price": 100.0,
                "size": 1.0,
                "client_order_id_hash": "b" * 64,
                "order_id_hash": None,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
            }) + "\n" + json.dumps({
                "schema_version": 1,
                "label_type": "PHASE51U_NATIVE_ROLE_CAPTURE_TARGET",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "canonical_group_id": "aster-target",
                "order_key": "aster-order",
                "venue_id": "aster",
                "required_native_role_source": "ASTER_ORDER_TRADE_UPDATE_M",
                "required_native_role_fields": ["m"],
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            source_path = source_dir / "phase51z_forward_native_role_rows.jsonl"
            source_path.write_text(json.dumps({
                "schema_version": 1,
                "label_type": "PHASE51Z_UNLINKED_NATIVE_ROLE_SOURCE",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "run_id": "phase51z_unlinked_request_test",
                "gate_status": "HOLD",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
                "venue_id": "lighter",
                "source_record_sha256": source_hash,
                "account_index": 42,
                "ask_account_id": 42,
                "bid_account_id": 7,
                "is_maker_ask": True,
            }) + "\n" + json.dumps({
                "schema_version": 1,
                "label_type": "PHASE51Z_UNLINKED_NATIVE_ROLE_SOURCE",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "run_id": "phase51z_unlinked_request_test",
                "gate_status": "HOLD",
                "no_live_flag": True,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
                "venue_id": "aster",
                "source_record_sha256": aster_source_hash,
                "m": True,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_source_link_request_pack_path()),
                    "--target-run",
                    str(target_run),
                    "--source-run",
                    str(source_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51z_source_link_request_pack_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            pack_dir = output_root / "phase51z_source_link_request_pack_test"
            summary = json.loads(
                (pack_dir / "phase51z_source_link_request_pack_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertEqual(summary["source_link_request_source_count"], 1)
            self.assertEqual(summary["source_link_request_target_count"], 1)
            self.assertEqual(summary["source_link_sidecar_template_row_count"], 0)
            self.assertEqual(summary["next_required_artifact"], "validated_redacted_source_link_sidecar")

            request_sources = (pack_dir / "source_link_request_sources.jsonl").read_text(encoding="utf-8")
            request_targets = (pack_dir / "source_link_request_targets.jsonl").read_text(encoding="utf-8")
            self.assertIn(source_hash, request_sources)
            self.assertIn(target_group, request_targets)
            self.assertIn(target_order_key, request_targets)
            self.assertNotIn("aster-target", request_targets)
            self.assertNotIn(aster_source_hash, request_sources)
            self.assertNotIn(raw_should_not_appear, request_sources + request_targets)
            self.assertEqual((pack_dir / "source_links.proposed.empty.jsonl").read_text(encoding="utf-8"), "")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51z_source_link_request_pack_path()),
                    "--target-run",
                    str(target_run),
                    "--source-run",
                    str(source_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51z_source_link_request_pack_all_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--venue-id",
                    "all",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            all_pack_dir = output_root / "phase51z_source_link_request_pack_all_test"
            all_summary = json.loads(
                (all_pack_dir / "phase51z_source_link_request_pack_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(all_summary["venue_id"], "all")
            self.assertEqual(all_summary["source_link_request_source_count"], 2)
            self.assertEqual(all_summary["source_link_request_target_count"], 2)
            self.assertEqual(all_summary["source_link_request_source_counts_by_venue"], {"aster": 1, "lighter": 1})
            self.assertEqual(all_summary["source_link_request_target_counts_by_venue"], {"aster": 1, "lighter": 1})
            all_request_sources = (all_pack_dir / "source_link_request_sources.jsonl").read_text(encoding="utf-8")
            all_request_targets = (all_pack_dir / "source_link_request_targets.jsonl").read_text(encoding="utf-8")
            self.assertIn(source_hash, all_request_sources)
            self.assertIn(aster_source_hash, all_request_sources)
            self.assertIn("aster-target", all_request_targets)
            self.assertNotIn(raw_should_not_appear, all_request_sources + all_request_targets)

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(pack_dir / "candidate_manifest_with_empty_sidecar.json"),
                    "--output-root",
                    str(readiness_root),
                    "--run-id",
                    "phase51v_source_link_request_empty_sidecar_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            readiness_summary = json.loads(
                (
                    readiness_root
                    / "phase51v_source_link_request_empty_sidecar_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(readiness_summary["gate_status"], "HOLD")
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 0)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 2)
            self.assertFalse(readiness_summary["generated_phase51s_manifest_ready"])

    def test_phase51w_forward_capture_request_pack_emits_operator_pack(self):
        """5.1w should emit an operator-facing request pack from 5.1u targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51w_request_pack"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 3,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            role_targets = [
                {
                    "canonical_group_id": "hyper-group",
                    "order_key": "hyper-order",
                    "venue_id": "hyperliquid",
                    "required_native_role_source": "HYPERLIQUID_CROSSED",
                    "required_native_role_fields": ["crossed"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_role_source": "LIGHTER_TRADES_JSON",
                    "required_native_role_fields": [
                        "account_index",
                        "is_maker_ask",
                        "ask_account_id",
                        "bid_account_id",
                    ],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "lighter-group-2",
                    "order_key": "lighter-order-2",
                    "venue_id": "lighter",
                    "required_native_role_source": "LIGHTER_TRADES_JSON",
                    "required_native_role_fields": [
                        "account_index",
                        "is_maker_ask",
                        "ask_account_id",
                        "bid_account_id",
                    ],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            limit_targets = [
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_limit_fields": [
                        "active_order_headroom_account",
                        "active_order_headroom_market",
                        "sendtx_per_minute_limit",
                        "sendtx_per_minute_remaining",
                        "weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining",
                        "native_limit_event_time_status",
                    ],
                    "accepted_native_limit_event_time_status": ["EVENT_TIME_ALIGNED"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }
            ]
            (target_run / "native_role_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in role_targets),
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in limit_targets),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51w_forward_capture_request_pack_path()),
                    "--target-run",
                    str(target_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51w_request_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            run_dir = output_root / "phase51w_request_test"
            summary = json.loads(
                (run_dir / "phase51w_forward_capture_request_pack_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["native_role_capture_target_count"], 3)
            self.assertEqual(summary["native_role_capture_target_counts_by_venue"], {"hyperliquid": 1, "lighter": 2})
            self.assertEqual(summary["lighter_native_limit_capture_target_count"], 1)
            self.assertEqual(summary["required_local_source_file_count"], 3)
            self.assertFalse(summary["clears_phase51_blockers"])

            request_pack = json.loads((run_dir / "forward_capture_request_pack.json").read_text(encoding="utf-8"))
            self.assertEqual(request_pack["required_local_source_file_count"], 3)
            self.assertEqual(request_pack["native_role_required_source_counts"]["LIGHTER_TRADES_JSON"], 2)
            self.assertIn("phase51v -> phase51s", request_pack["downstream_chain"])
            self.assertFalse(request_pack["live_orders_allowed"])
            skeleton = json.loads((run_dir / "capture_bundle_manifest.skeleton.json").read_text(encoding="utf-8"))
            self.assertEqual(len(skeleton["sources"]), 3)
            self.assertFalse(skeleton["approved_for_live"])
            markdown = (run_dir / "forward_capture_request_pack.md").read_text(encoding="utf-8")
            self.assertIn("HYPERLIQUID_CROSSED", markdown)
            self.assertIn("LIGHTER_TRADES_JSON", markdown)
            self.assertIn("EVENT_TIME_ALIGNED", markdown)
            self.assertNotIn("client_order_id", markdown)
            self.assertNotIn("trade_id", markdown)

    def test_phase51w_forward_capture_request_pack_stages_empty_local_bundle(self):
        """5.1w can stage the six canonical local source files without clearing 5.1v."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            output_root = tmp_path / "phase51w_request_pack"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "native_role_capture_target_count": 5,
                "lighter_native_limit_capture_target_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            role_targets = [
                {
                    "canonical_group_id": "aster-group",
                    "order_key": "aster-order",
                    "venue_id": "aster",
                    "required_native_role_source": "ASTER_ORDER_TRADE_UPDATE",
                    "required_native_role_fields": ["e=ORDER_TRADE_UPDATE", "o.m", "positive o.l"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "extended-group",
                    "order_key": "extended-order",
                    "venue_id": "extended",
                    "required_native_role_source": "EXTENDED_ISTAKER",
                    "required_native_role_fields": ["isTaker"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "hyper-group",
                    "order_key": "hyper-order",
                    "venue_id": "hyperliquid",
                    "required_native_role_source": "HYPERLIQUID_CROSSED",
                    "required_native_role_fields": ["crossed"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_role_source": "LIGHTER_TRADES_JSON",
                    "required_native_role_fields": [
                        "account_index",
                        "is_maker_ask",
                        "ask_account_id",
                        "bid_account_id",
                    ],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "paradex-group",
                    "order_key": "paradex-order",
                    "venue_id": "paradex",
                    "required_native_role_source": "PARADEX_LIQUIDITY",
                    "required_native_role_fields": ["liquidity"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            limit_targets = [
                {
                    "canonical_group_id": "lighter-group",
                    "order_key": "lighter-order",
                    "venue_id": "lighter",
                    "required_native_limit_fields": [
                        "active_order_headroom_account",
                        "active_order_headroom_market",
                        "sendtx_per_minute_limit",
                        "sendtx_per_minute_remaining",
                        "weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining",
                        "native_limit_event_time_status",
                    ],
                    "accepted_native_limit_event_time_status": ["EVENT_TIME_ALIGNED"],
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }
            ]
            (target_run / "native_role_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in role_targets),
                encoding="utf-8",
            )
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in limit_targets),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51w_forward_capture_request_pack_path()),
                    "--target-run",
                    str(target_run),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51w_staging_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--stage-local-source-dir",
                    "local_source_staging",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            run_dir = output_root / "phase51w_staging_test"
            summary = json.loads(
                (run_dir / "phase51w_forward_capture_request_pack_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertTrue(summary["local_source_staging_enabled"])
            self.assertFalse(summary["clears_phase51_blockers"])
            local_manifest_path = Path(summary["local_capture_bundle_manifest_path"])
            local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(local_manifest["sources"]), 6)
            self.assertEqual(local_manifest["source_links"], [])
            self.assertFalse(local_manifest["approved_for_live"])
            self.assertFalse(local_manifest["live_orders_allowed"])
            staged_paths = [Path(source["path"]) for source in local_manifest["sources"]]
            self.assertEqual(len({path.name for path in staged_paths}), 6)
            for path in staged_paths:
                self.assertTrue(path.is_file())
                self.assertEqual(path.suffix, ".jsonl")
                self.assertEqual(path.read_text(encoding="utf-8"), "")
                self.assertTrue(path.resolve().is_relative_to(run_dir.resolve()))
            staged_text = (run_dir / "local_source_field_guide.json").read_text(encoding="utf-8")
            self.assertNotIn("client_order_id", staged_text)
            self.assertNotIn("trade_id", staged_text)

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51v_forward_capture_bundle_readiness_path()),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(local_manifest_path),
                    "--output-root",
                    str(tmp_path / "phase51v_readiness"),
                    "--run-id",
                    "phase51v_empty_staged_bundle_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
            readiness_summary = json.loads(
                (
                    tmp_path
                    / "phase51v_readiness"
                    / "phase51v_empty_staged_bundle_test"
                    / "phase51v_forward_capture_bundle_readiness_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(readiness_summary["gate_status"], "HOLD")
            self.assertEqual(
                readiness_summary["gate_reason"],
                "phase51v_forward_capture_bundle_incomplete_nonlive_hold",
            )
            self.assertEqual(readiness_summary["native_role_capture_target_ready_count"], 0)
            self.assertEqual(readiness_summary["native_role_capture_target_missing_count"], 5)
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_ready_count"], 0)
            self.assertEqual(readiness_summary["lighter_native_limit_capture_target_missing_count"], 1)
            self.assertEqual(readiness_summary["source_file_status_counts"], {"LOCAL_FILE_READY": 6})
            self.assertFalse(readiness_summary["generated_phase51s_manifest_ready"])

    def test_phase51w_forward_capture_request_pack_rejects_unsafe_targets(self):
        """5.1w should reject unsafe target manifests instead of building a request pack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "phase51u_targets"
            target_run.mkdir()
            (target_run / "phase51u_forward_capture_target_manifest_summary.json").write_text(json.dumps({
                "run_id": "phase51u_targets",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (target_run / "native_role_capture_targets.jsonl").write_text(json.dumps({
                "canonical_group_id": "unsafe-group",
                "order_key": "unsafe-order",
                "venue_id": "extended",
                "required_native_role_source": "EXTENDED_ISTAKER",
                "required_native_role_fields": ["isTaker"],
                "approved_for_live": True,
            }) + "\n", encoding="utf-8")
            (target_run / "lighter_native_limit_capture_targets.jsonl").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51w_forward_capture_request_pack_path()),
                    "--target-run",
                    str(target_run),
                    "--output-root",
                    str(tmp_path / "phase51w_request_pack"),
                    "--run-id",
                    "phase51w_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe role target flag approved_for_live=true", result.stderr)

    def test_phase51s_source_link_sidecar_rejects_unsafe_rows(self):
        """5.1s source-link sidecars should be local, redacted, and unambiguous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "native_source.jsonl"
            source_link_path = tmp_path / "source_links.jsonl"
            manifest_path = tmp_path / "manifest.json"
            output_root = tmp_path / "local_source_acquisition"
            source_path.write_text(json.dumps({
                "venue_id": "hyperliquid",
                "crossed": False,
                "source_record_sha256": "forward-hl-source-hash",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            rejection_cases = [
                (
                    "network_link",
                    None,
                    [{"source_link_id": "network", "path": "https://example.invalid/source_links.jsonl"}],
                    "network source_links paths are prohibited",
                ),
                (
                    "raw_link",
                    {
                        "source_record_sha256": "raw-hash",
                        "canonical_group_id": "group-raw",
                        "trade_id": "raw-trade-id",
                        "approved_for_live": False,
                    },
                    None,
                    "source link leaked raw identifier fields",
                ),
                (
                    "secret_link",
                    {
                        "source_record_sha256": "secret-hash",
                        "canonical_group_id": "group-secret",
                        "api_key": "not-allowed",
                        "approved_for_live": False,
                    },
                    None,
                    "secret-shaped source link field",
                ),
                (
                    "unsupported_link",
                    {
                        "source_record_sha256": "unsupported-hash",
                        "canonical_group_id": "group-unsupported",
                        "note": "unsupported",
                        "approved_for_live": False,
                    },
                    None,
                    "source link has unsupported fields",
                ),
                (
                    "unsafe_link",
                    {
                        "source_record_sha256": "unsafe-hash",
                        "canonical_group_id": "group-unsafe",
                        "approved_for_live": True,
                    },
                    None,
                    "unsafe source link flag approved_for_live=true",
                ),
                (
                    "non_scalar_link",
                    {
                        "source_record_sha256": ["non-scalar-hash"],
                        "canonical_group_id": "group-non-scalar",
                        "approved_for_live": False,
                    },
                    None,
                    "source link field source_record_sha256 must be a string",
                ),
            ]
            for run_suffix, link_row, source_links, expected_error in rejection_cases:
                with self.subTest(run_suffix=run_suffix):
                    if link_row is not None:
                        source_link_path.write_text(json.dumps(link_row) + "\n", encoding="utf-8")
                    manifest_path.write_text(json.dumps({
                        "manifest_version": 1,
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "sources": [{"source_id": "native", "path": str(source_path)}],
                        "source_links": source_links or [{"source_link_id": "native_link", "path": str(source_link_path)}],
                    }), encoding="utf-8")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(self._get_phase51s_local_native_source_acquisition_path()),
                            "--manifest",
                            str(manifest_path),
                            "--output-root",
                            str(output_root),
                            "--run-id",
                            f"phase51s_{run_suffix}_test",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected_error, result.stderr)

            source_link_path.write_text(
                json.dumps({
                    "source_record_sha256": "duplicate-hash",
                    "canonical_group_id": "group-a",
                    "approved_for_live": False,
                })
                + "\n"
                + json.dumps({
                    "source_record_sha256": "duplicate-hash",
                    "canonical_group_id": "group-b",
                    "approved_for_live": False,
                })
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [{"source_id": "native", "path": str(source_path)}],
                "source_links": [{"source_link_id": "native_link", "path": str(source_link_path)}],
            }), encoding="utf-8")
            duplicate_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51s_duplicate_source_link_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(duplicate_result.returncode, 2)
            self.assertIn("duplicate source link hash", duplicate_result.stderr)

    def test_phase51s_local_source_acquisition_does_not_false_clear_partial_sources(self):
        """5.1s should not convert staged local rows into blocker-clearing evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            manifest_path = tmp_path / "manifest.json"
            staging_root = tmp_path / "local_source_acquisition"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            observed_run.mkdir()

            observed_labels = [
                ("group-lighter-partial", "order-lighter-partial", "lighter", "Buy", 1),
                ("group-hl-unjoined", "order-hl-unjoined", "hyperliquid", "Sell", 1),
                ("group-generic", "order-generic", "extended", "Buy", 1),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51s_no_false_clear_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side, fill_count in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": fill_count,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            source_rows = [
                {
                    "canonical_group_id": "group-lighter-partial",
                    "venue_id": "lighter",
                    "account_index": 123,
                    "is_maker_ask": True,
                    "ask_account_id": 123,
                    "bid_account_id": 456,
                    "active_order_headroom_account": 10,
                    "active_order_headroom_market": 3,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "tid": "raw-unjoined-hl-trade",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-generic",
                    "venue_id": "extended",
                    "native_role": "MAKER",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
                "sources": [{"source_id": "partial", "path": str(source_path)}],
            }), encoding="utf-8")

            staging_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51s_local_native_source_acquisition_path()),
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(staging_root),
                    "--run-id",
                    "phase51s_no_false_clear_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                staging_result.returncode,
                0,
                f"stdout: {staging_result.stdout}\nstderr: {staging_result.stderr}",
            )
            staging_dir = staging_root / "phase51s_no_false_clear_test"
            staging_summary = json.loads(
                (staging_dir / "phase51s_local_native_source_acquisition_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(staging_summary["source_row_count"], 3)
            self.assertEqual(staging_summary["join_key_source_row_count"], 2)
            self.assertEqual(staging_summary["source_row_without_join_key_count"], 1)
            self.assertEqual(staging_summary["complete_lighter_native_limit_source_row_count"], 0)
            self.assertFalse(staging_summary["clears_phase51_blockers"])

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(staging_dir / "local_native_source.jsonl"),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51s_no_false_clear_to_phase51r_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_summary = json.loads(
                (
                    acquisition_root
                    / "phase51s_no_false_clear_to_phase51r_test"
                    / "phase51r_forward_native_source_acquisition_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["gate_reason"], "phase51r_forward_native_source_acquisition_incomplete")
            self.assertEqual(acquisition_summary["source_row_count"], 3)
            self.assertEqual(acquisition_summary["native_role_target_count"], 3)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 1)
            self.assertEqual(acquisition_summary["native_limit_source_record_count"], 1)
            self.assertEqual(acquisition_summary["native_limit_complete_source_record_count"], 0)
            self.assertEqual(acquisition_summary["lighter_native_limit_target_recovered_count"], 0)

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_root / "phase51s_no_false_clear_to_phase51r_test" / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_root / "phase51s_no_false_clear_to_phase51r_test" / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51s_no_false_clear_to_phase51q_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_summary = json.loads(
                (
                    capture_root
                    / "phase51s_no_false_clear_to_phase51q_test"
                    / "phase51q_forward_native_evidence_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 1)
            self.assertEqual(capture_summary["native_role_capture_status_counts"]["MISSING_FORWARD_NATIVE_ROLE_SOURCE"], 2)
            self.assertEqual(capture_summary["native_limit_pressure_status_counts"]["PARTIAL_NATIVE_LIMIT_PRESSURE_SOURCE"], 1)

    def test_phase51r_source_link_sidecar_recovers_joinable_staged_rows(self):
        """5.1r should use a validated source-link sidecar without inferring native fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            link_path = tmp_path / "source_links.jsonl"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            observed_run.mkdir()

            observed_labels = [
                ("group-hl", "order-hl", "hyperliquid", "Buy"),
                ("group-lighter", "order-lighter", "lighter", "Sell"),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_source_link_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": 1,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            source_rows = [
                {
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "phase51s_source_record_sha256": "native-hl-source-hash",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "venue_id": "lighter",
                    "account_index": 123,
                    "is_maker_ask": False,
                    "ask_account_id": 456,
                    "bid_account_id": 123,
                    "active_order_headroom_account": 100,
                    "active_order_headroom_market": 8,
                    "sendtx_per_minute_limit": 1000,
                    "sendtx_per_minute_remaining": 990,
                    "weighted_requests_per_minute_limit": 24000,
                    "weighted_requests_per_minute_remaining": 23900,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "native_limit_staleness_ms": 5.0,
                    "phase51s_source_record_sha256": "native-lighter-source-hash",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")
            link_rows = [
                {
                    "phase51s_source_record_sha256": "native-hl-source-hash",
                    "canonical_group_id": "group-hl",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "phase51s_source_record_sha256": "native-lighter-source-hash",
                    "order_key": "order-lighter",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with link_path.open("w", encoding="utf-8") as f:
                for row in link_rows:
                    f.write(json.dumps(row) + "\n")

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--source-link-jsonl",
                    str(link_path),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51r_source_link_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_dir = acquisition_root / "phase51r_source_link_test"
            acquisition_summary = json.loads(
                (acquisition_dir / "phase51r_forward_native_source_acquisition_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["source_link_record_count"], 2)
            self.assertEqual(acquisition_summary["source_link_hash_count"], 2)
            self.assertEqual(acquisition_summary["source_link_applied_count"], 2)
            self.assertEqual(acquisition_summary["canonical_group_link_source_counts"]["SOURCE_LINK_SIDECAR"], 2)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 2)
            self.assertEqual(acquisition_summary["lighter_native_limit_target_recovered_count"], 1)
            self.assertEqual(acquisition_summary["raw_identifier_redaction_status"], "PASS")

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_dir / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_dir / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51r_source_link_to_phase51q_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_summary = json.loads(
                (
                    capture_root
                    / "phase51r_source_link_to_phase51q_test"
                    / "phase51q_forward_native_evidence_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 2)
            self.assertEqual(capture_summary["native_limit_pressure_status_counts"]["OBSERVED_NATIVE_LIMIT_PRESSURE"], 1)

    def test_phase51r_source_link_sidecar_rejects_ambiguous_or_raw_links(self):
        """5.1r source-link sidecars must be unambiguous and redacted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            link_path = tmp_path / "source_links.jsonl"
            output_root = tmp_path / "source_acquisition"
            observed_run.mkdir()

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_bad_source_link_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 2,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key in (("group-a", "order-a"), ("group-b", "order-b")):
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": "hyperliquid",
                        "side": "Buy",
                        "fill_count": 1,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")
            source_path.write_text(json.dumps({
                "venue_id": "hyperliquid",
                "crossed": False,
                "phase51s_source_record_sha256": "duplicate-source-hash",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            with link_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "phase51s_source_record_sha256": "duplicate-source-hash",
                    "canonical_group_id": "group-a",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")
                f.write(json.dumps({
                    "phase51s_source_record_sha256": "duplicate-source-hash",
                    "canonical_group_id": "group-b",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                }) + "\n")
            duplicate_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--source-link-jsonl",
                    str(link_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51r_duplicate_source_link_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(duplicate_result.returncode, 2)
            self.assertIn("duplicate source link hash", duplicate_result.stderr)

            link_path.write_text(json.dumps({
                "phase51s_source_record_sha256": "raw-source-hash",
                "canonical_group_id": "group-a",
                "trade_id": "raw-trade-id",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            raw_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--source-link-jsonl",
                    str(link_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51r_raw_source_link_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw_result.returncode, 2)
            self.assertIn("source link leaked raw identifier fields", raw_result.stderr)

            rejection_cases = [
                (
                    "secret_source_link",
                    {
                        "phase51s_source_record_sha256": "secret-source-hash",
                        "canonical_group_id": "group-a",
                        "api_key": "not-allowed",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    },
                    "secret-shaped source link field",
                ),
                (
                    "unsupported_source_link",
                    {
                        "phase51s_source_record_sha256": "unsupported-source-hash",
                        "canonical_group_id": "group-a",
                        "note": "unsupported-sidecar-metadata",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    },
                    "source link has unsupported fields",
                ),
                (
                    "unsafe_source_link",
                    {
                        "phase51s_source_record_sha256": "unsafe-source-hash",
                        "canonical_group_id": "group-a",
                        "approved_for_live": True,
                        "approved_for_model_training": False,
                    },
                    "unsafe source link flag approved_for_live=true",
                ),
                (
                    "conflicting_source_link",
                    {
                        "phase51s_source_record_sha256": "conflict-source-hash",
                        "canonical_group_id": "group-a",
                        "order_key": "order-b",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    },
                    "source link canonical_group_id conflicts with order_key",
                ),
                (
                    "nested_raw_source_link",
                    {
                        "phase51s_source_record_sha256": "nested-raw-source-hash",
                        "canonical_group_id": "group-a",
                        "metadata": {"trade_id": "nested-raw-trade-id"},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    },
                    "source link leaked raw identifier fields",
                ),
            ]
            for run_suffix, link_row, expected_error in rejection_cases:
                with self.subTest(run_suffix=run_suffix):
                    link_path.write_text(json.dumps(link_row) + "\n", encoding="utf-8")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(self._get_phase51r_forward_native_source_acquisition_path()),
                            "--observed-pfill-run",
                            str(observed_run),
                            "--source-json",
                            str(source_path),
                            "--source-link-jsonl",
                            str(link_path),
                            "--output-root",
                            str(output_root),
                            "--run-id",
                            f"phase51r_{run_suffix}_test",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected_error, result.stderr)

    def test_phase51r_source_acquisition_aggregates_multi_fill_native_roles(self):
        """5.1r should aggregate distinct native fill/trade rows for one group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            observed_run.mkdir()

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_aggregate_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-multi-fill",
                "order_key": "order-multi-fill",
                "source_telemetry_sha256": "source-multi-fill",
                "venue_id": "hyperliquid",
                "side": "BID",
                "fill_count": 2,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 2},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_rows = [
                {
                    "canonical_group_id": "group-multi-fill",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "sequence_hash": "native-fill-a",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-multi-fill",
                    "venue_id": "hyperliquid",
                    "crossed": True,
                    "sequence_hash": "native-fill-b",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51r_aggregate_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_dir = acquisition_root / "phase51r_aggregate_test"
            acquisition_summary = json.loads(
                (acquisition_dir / "phase51r_forward_native_source_acquisition_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["native_role_source_record_count"], 1)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 1)
            role_row = json.loads((acquisition_dir / "native_role_source.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(role_row["source_record_count"], 2)
            self.assertEqual(role_row["maker_taker_role_counts"]["MAKER"], 1)
            self.assertEqual(role_row["maker_taker_role_counts"]["TAKER"], 1)

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_dir / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_dir / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51r_aggregate_to_phase51q_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_summary = json.loads(
                (
                    capture_root
                    / "phase51r_aggregate_to_phase51q_test"
                    / "phase51q_forward_native_evidence_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 1)
            self.assertEqual(capture_summary["native_role_capture_status_counts"]["RECOVERED_FORWARD_NATIVE_ROLE"], 1)

    def test_phase51r_source_acquisition_feeds_phase51q_without_raw_ids(self):
        """5.1r should normalize venue-native snapshots into redacted 5.1q inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            recovery_root = tmp_path / "maker_taker_recovery"
            observed_run.mkdir()

            observed_labels = [
                ("group-hl", "order-hl", "hyperliquid", "Buy"),
                ("group-paradex", "order-paradex", "paradex", "Sell"),
                ("group-aster", "order-aster", "aster", "Buy"),
                ("group-extended", "order-extended", "extended", "Sell"),
                ("group-lighter", "order-lighter", "lighter", "Buy"),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": 1,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            source_rows = [
                {
                    "canonical_group_id": "group-hl",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "oid": "raw-hl-order",
                    "cloid": "raw-hl-client-order",
                    "tid": "raw-hl-trade",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-hl",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "oid": "raw-hl-order",
                    "cloid": "raw-hl-client-order",
                    "tid": "raw-hl-trade",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-paradex",
                    "venue_id": "paradex",
                    "liquidity": "TAKER",
                    "id": "raw-paradex-fill-id",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-aster",
                    "venue_id": "aster",
                    "e": "ORDER_TRADE_UPDATE",
                    "o": {"m": True, "l": "0.1", "i": 12345},
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-extended",
                    "venue_id": "extended",
                    "isTaker": True,
                    "trade_id": "raw-extended-trade-id",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-lighter",
                    "venue_id": "lighter",
                    "account_index": 123,
                    "is_maker_ask": False,
                    "ask_account_id": 456,
                    "bid_account_id": 123,
                    "trade_id": "raw-lighter-trade-id",
                    "bid_client_id": "raw-lighter-client-id",
                    "active_order_headroom_account": 100,
                    "active_order_headroom_market": 10,
                    "sendtx_per_minute_limit": 1000,
                    "sendtx_per_minute_remaining": 990,
                    "rest_requests_per_minute_limit": 1200,
                    "rest_requests_per_minute_remaining": 1180,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "native_limit_staleness_ms": 5.0,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51r_acquisition_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_dir = acquisition_root / "phase51r_acquisition_test"
            acquisition_summary = json.loads(
                (acquisition_dir / "phase51r_forward_native_source_acquisition_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["gate_status"], "HOLD")
            self.assertEqual(acquisition_summary["gate_reason"], "phase51r_forward_native_source_acquisition_complete_nonlive_hold")
            self.assertEqual(acquisition_summary["source_row_count"], 6)
            self.assertEqual(acquisition_summary["native_role_source_record_count"], 5)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 5)
            self.assertEqual(acquisition_summary["native_limit_source_record_count"], 1)
            self.assertEqual(acquisition_summary["lighter_native_limit_target_recovered_count"], 1)
            self.assertEqual(acquisition_summary["raw_identifier_redaction_status"], "PASS")

            raw_fields = {
                "decision_id",
                "order_id",
                "client_order_id",
                "venue_order_id",
                "raw_order_id",
                "raw_client_order_id",
                "ask_id",
                "bid_id",
                "ask_client_id",
                "bid_client_id",
                "trade_id",
                "fill_id",
                "id",
                "oid",
                "cloid",
                "tid",
            }
            for artifact in ("native_role_source.jsonl", "native_limit_source.jsonl", "source_acquisition_labels.jsonl"):
                for line in (acquisition_dir / artifact).read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertFalse(raw_fields & set(row), artifact)
                    self.assertFalse(row["approved_for_live"])
                    self.assertFalse(row["approved_for_model_training"])

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_dir / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_dir / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51r_capture_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_dir = capture_root / "phase51r_capture_test"
            capture_summary = json.loads(
                (capture_dir / "phase51q_forward_native_evidence_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 5)
            self.assertEqual(capture_summary["native_limit_pressure_status_counts"]["OBSERVED_NATIVE_LIMIT_PRESSURE"], 1)

            recovery_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51n_maker_taker_attribution_recovery_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(capture_dir / "native_role_evidence.jsonl"),
                    "--output-root",
                    str(recovery_root),
                    "--run-id",
                    "phase51r_recovery_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                recovery_result.returncode,
                0,
                f"stdout: {recovery_result.stdout}\nstderr: {recovery_result.stderr}",
            )
            recovery_summary = json.loads(
                (recovery_root / "phase51r_recovery_test" / "maker_taker_attribution_recovery_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(recovery_summary["maker_taker_observed_or_recovered_count"], 5)
            self.assertEqual(recovery_summary["maker_taker_partial_or_missing_count"], 0)
            self.assertEqual(
                recovery_summary["maker_taker_recovery_status_counts"]["RECOVERED_VENUE_NATIVE_ROLE"],
                5,
            )

    def test_phase51r_source_acquisition_rejects_unsafe_flags(self):
        """5.1r must reject source rows that attempt to authorize live or training use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "unsafe_source.jsonl"
            output_root = tmp_path / "source_acquisition"
            observed_run.mkdir()

            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_unsafe_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-unsafe",
                "order_key": "order-unsafe",
                "source_telemetry_sha256": "source-sha",
                "venue_id": "hyperliquid",
                "side": "Buy",
                "fill_count": 1,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            source_path.write_text(json.dumps({
                "canonical_group_id": "group-unsafe",
                "venue_id": "hyperliquid",
                "crossed": False,
                "approved_for_live": True,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51r_unsafe_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe source row flag approved_for_live=true", result.stderr)

    def test_phase51r_source_acquisition_does_not_false_clear_partial_or_inferred_sources(self):
        """5.1r should not treat partial limits or wrong-field role rows as complete evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_path = tmp_path / "native_source.jsonl"
            acquisition_root = tmp_path / "source_acquisition"
            capture_root = tmp_path / "forward_native_capture"
            observed_run.mkdir()

            observed_labels = [
                ("group-lighter-partial", "order-lighter-partial", "lighter", "Buy", 1),
                ("group-wrong-field", "order-wrong-field", "paradex", "Sell", 1),
                ("group-generic", "order-generic", "extended", "Buy", 1),
                ("group-distinct-duplicate", "order-distinct-duplicate", "hyperliquid", "Buy", 2),
            ]
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51r_no_false_clear_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": len(observed_labels),
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            with (observed_run / "pfill_order_labels.jsonl").open("w", encoding="utf-8") as f:
                for group, order_key, venue_id, side, fill_count in observed_labels:
                    f.write(json.dumps({
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "source_telemetry_sha256": f"source-{group}",
                        "venue_id": venue_id,
                        "side": side,
                        "fill_count": fill_count,
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                    }) + "\n")

            source_rows = [
                {
                    "canonical_group_id": "group-lighter-partial",
                    "venue_id": "lighter",
                    "account_index": 123,
                    "is_maker_ask": True,
                    "ask_account_id": 123,
                    "bid_account_id": 456,
                    "active_order_headroom_account": 10,
                    "active_order_headroom_market": 3,
                    "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-wrong-field",
                    "venue_id": "paradex",
                    "crossed": False,
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-generic",
                    "venue_id": "extended",
                    "native_role": "MAKER",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-distinct-duplicate",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "tid": "raw-distinct-duplicate-fill-1",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
                {
                    "canonical_group_id": "group-distinct-duplicate",
                    "venue_id": "hyperliquid",
                    "crossed": False,
                    "tid": "raw-distinct-duplicate-fill-2",
                    "approved_for_live": False,
                    "approved_for_model_training": False,
                },
            ]
            with source_path.open("w", encoding="utf-8") as f:
                for row in source_rows:
                    f.write(json.dumps(row) + "\n")

            acquisition_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51r_forward_native_source_acquisition_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-json",
                    str(source_path),
                    "--output-root",
                    str(acquisition_root),
                    "--run-id",
                    "phase51r_no_false_clear_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                acquisition_result.returncode,
                0,
                f"stdout: {acquisition_result.stdout}\nstderr: {acquisition_result.stderr}",
            )
            acquisition_dir = acquisition_root / "phase51r_no_false_clear_test"
            acquisition_summary = json.loads(
                (acquisition_dir / "phase51r_forward_native_source_acquisition_summary.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(acquisition_summary["gate_reason"], "phase51r_forward_native_source_acquisition_incomplete")
            self.assertEqual(acquisition_summary["native_role_target_count"], 4)
            self.assertEqual(acquisition_summary["native_role_source_record_count"], 2)
            self.assertEqual(acquisition_summary["native_role_target_recovered_count"], 2)
            self.assertEqual(acquisition_summary["native_limit_source_record_count"], 1)
            self.assertEqual(acquisition_summary["native_limit_complete_source_record_count"], 0)
            self.assertEqual(acquisition_summary["lighter_native_limit_target_recovered_count"], 0)
            self.assertNotIn("path", acquisition_summary["source_artifacts"][0])
            self.assertIn("path_hash", acquisition_summary["source_artifacts"][0])

            capture_result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51q_forward_native_evidence_capture_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-role-jsonl",
                    str(acquisition_dir / "native_role_source.jsonl"),
                    "--native-limit-jsonl",
                    str(acquisition_dir / "native_limit_source.jsonl"),
                    "--output-root",
                    str(capture_root),
                    "--run-id",
                    "phase51r_no_false_clear_capture_test",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capture_result.returncode,
                0,
                f"stdout: {capture_result.stdout}\nstderr: {capture_result.stderr}",
            )
            capture_summary = json.loads(
                (
                    capture_root
                    / "phase51r_no_false_clear_capture_test"
                    / "phase51q_forward_native_evidence_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(capture_summary["recovered_forward_native_role_count"], 2)
            self.assertEqual(capture_summary["native_role_capture_status_counts"]["MISSING_FORWARD_NATIVE_ROLE_SOURCE"], 2)
            self.assertEqual(capture_summary["native_role_capture_status_counts"].get("PARTIAL_FORWARD_NATIVE_ROLE_SOURCE", 0), 0)
            self.assertEqual(capture_summary["native_limit_pressure_status_counts"]["PARTIAL_NATIVE_LIMIT_PRESSURE_SOURCE"], 1)

    def test_phase51o_native_role_inventory_rejects_non_native_source(self):
        """5.1o should reject role evidence from non-native or inferred sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            observed_run = tmp_path / "observed_pfill"
            source_root = tmp_path / "source_root"
            output_root = tmp_path / "native_role_inventory"
            native_roles = tmp_path / "native_roles.jsonl"
            observed_run.mkdir()
            source_root.mkdir()
            (observed_run / "pfill_outcome_summary.json").write_text(json.dumps({
                "run_id": "observed_phase51o_reject_test",
                "baseline_commit": "18dd09512288a85e440d3977e32432c3aabc1190",
                "gate_status": "HOLD",
                "order_label_count": 1,
                "approved_for_live": False,
                "approved_for_model_training": False,
            }), encoding="utf-8")
            (observed_run / "pfill_order_labels.jsonl").write_text(json.dumps({
                "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                "canonical_group_id": "group-unsafe",
                "order_key": "order-unsafe",
                "source_telemetry_sha256": "source-sha",
                "venue_id": "lighter",
                "side": "Buy",
                "fill_count": 1,
                "outcome_status": "OBSERVED_FILLED",
                "p_fill_outcome": 1.0,
                "maker_taker_role_counts": {"MAKER": 0, "TAKER": 0, "UNKNOWN": 1},
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            native_roles.write_text(json.dumps({
                "canonical_group_id": "group-unsafe",
                "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                "maker_taker_attribution_source": "POST_ONLY_INTENT_INFERENCE",
                "approved_for_live": False,
                "approved_for_model_training": False,
            }) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(self._get_phase51o_native_role_source_inventory_path()),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--source-root",
                    str(source_root),
                    "--native-role-jsonl",
                    str(native_roles),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51o_inventory_reject_test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported native role source", result.stderr)
    
    def test_invalid_file_exit_1(self):
        """File with contract violation should exit with code 1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            record = self._make_valid_telemetry_record(tick=0)
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
    
    def test_no_args_with_valid_telemetry_file_exit_0(self):
        """No arguments with valid telemetry file in cwd should exit 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid telemetry file in the temp directory
            telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            record = self._make_valid_telemetry_record(tick=0)
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
    
    def test_valid_mc_runs_file_exit_0(self):
        """Valid mc_runs.jsonl file should exit with code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc_runs_path = Path(tmpdir) / "mc_runs.jsonl"
            with open(mc_runs_path, "w") as f:
                for i in range(3):
                    record = self._make_valid_mc_run_record(run_index=i)
                    f.write(json.dumps(record) + "\n")
            
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(mc_runs_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("OK", result.stdout)
            self.assertIn("3 record(s) validated", result.stdout)
    
    def test_invalid_mc_runs_missing_field_exit_1(self):
        """mc_runs.jsonl with missing required field should exit with code 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc_runs_path = Path(tmpdir) / "mc_runs.jsonl"
            record = self._make_valid_mc_run_record()
            del record["run_index"]  # Remove required field
            with open(mc_runs_path, "w") as f:
                f.write(json.dumps(record) + "\n")
            
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(mc_runs_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("FAILED", result.stdout)
            self.assertIn("run_index", result.stdout)
    
    def test_directory_with_mc_runs_validates_correctly(self):
        """Directory containing mc_runs.jsonl should validate it with correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc_runs_path = Path(tmpdir) / "mc_runs.jsonl"
            with open(mc_runs_path, "w") as f:
                for i in range(3):
                    record = self._make_valid_mc_run_record(run_index=i)
                    f.write(json.dumps(record) + "\n")
            
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), tmpdir],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("mc_runs.jsonl", result.stdout)
    
    def test_unmapped_file_fails_loudly(self):
        """Unmapped file type (e.g. metrics.jsonl) should exit 2 with instructions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            with open(metrics_path, "w") as f:
                f.write('{"some": "data"}\n')
            
            result = subprocess.run(
                [sys.executable, str(self._get_validator_path()), str(metrics_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, f"stdout: {result.stdout}\nstderr: {result.stderr}")
            self.assertIn("No schema defined", result.stderr)
            self.assertIn("metrics.jsonl", result.stderr)


class TestSchemaGeneratedTelemetry(unittest.TestCase):
    """
    Test that telemetry generated from schema definition validates successfully.
    
    This ensures the schema file itself is self-consistent.
    """
    
    def test_schema_driven_telemetry_record_validates(self):
        """A record generated from telemetry schema required fields should validate."""
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
        errors, _ = validate_record(record, schema, 1, None, "t")
        self.assertEqual(len(errors), 0, f"Schema-driven record should validate. Errors: {errors}")
    
    def test_schema_driven_mc_runs_record_validates(self):
        """A record generated from mc_runs schema required fields should validate."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "mc_runs_schema_v1.json"
        schema = load_schema(schema_path)
        self.assertIsNotNone(schema)
        
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
                record[field] = "None"
            elif ftype == "boolean":
                record[field] = False
        
        record["schema_version"] = 1
        
        errors, _ = validate_record(record, schema, 1, None, "run_index")
        self.assertEqual(len(errors), 0, f"Schema-driven mc_runs record should validate. Errors: {errors}")
    
    def test_telemetry_schema_file_has_all_required_keys(self):
        """Telemetry schema file should have all expected top-level keys."""
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
    
    def test_mc_runs_schema_file_has_all_required_keys(self):
        """mc_runs schema file should have all expected top-level keys."""
        script_dir = Path(__file__).parent.parent
        schema_path = script_dir / "schemas" / "mc_runs_schema_v1.json"
        schema = load_schema(schema_path)
        
        self.assertIsNotNone(schema)
        self.assertIn("required_fields", schema)
        self.assertIn("optional_fields", schema)
        self.assertIn("field_types", schema)
        self.assertIn("schema_version", schema)
        self.assertEqual(schema["schema_version"], 1)


if __name__ == '__main__':
    unittest.main()
