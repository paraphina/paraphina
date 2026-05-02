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
            self.assertFalse(summary["approved_for_model_training"])
            self.assertFalse(summary["approved_for_live"])
            trades = json.loads((run_dir / "source_snapshots" / "trades_backfill.sanitized.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trades["trades"]), 2)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

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
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            for file_info in manifest["files"]:
                artifact = run_dir / file_info["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    file_info["sha256"],
                    file_info["path"],
                )

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
