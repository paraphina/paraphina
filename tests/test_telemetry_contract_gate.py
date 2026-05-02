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
            input_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")

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
            order_books_path = tmp_path / "order_books.json"
            trades_path = tmp_path / "trades.json"
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
                "active_orders_per_account_limit": 1000,
                "active_orders_per_market_limit": 100,
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
                    "--order-books-json",
                    str(order_books_path),
                    "--trades-json",
                    str(trades_path),
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
            self.assertEqual(limits["active_orders_per_account_limit"], 1000)
            self.assertEqual(limits["active_orders_per_market_limit"], 100)
            self.assertEqual(limits["lighter_user_tier"], "premium")
            self.assertEqual(limits["lighter_user_tier_name"], "premium")
            self.assertEqual(limits["current_maker_fee_tick"], 40)
            self.assertEqual(limits["current_taker_fee_tick"], 280)
            self.assertEqual(limits["effective_lit_stakes"], "0.00000000")

            active_orders = next(r for r in records if r["event_type"] == "V2_LIGHTER_ACTIVE_ORDERS")
            self.assertEqual(active_orders["active_orders_count_total"], 3)
            self.assertEqual(active_orders["active_orders_count_market"], 2)
            self.assertEqual(active_orders["pending_orders_count_total"], 1)
            self.assertEqual(active_orders["active_order_headroom_account"], 997)
            self.assertEqual(active_orders["active_order_headroom_market"], 98)

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
            source_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
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
