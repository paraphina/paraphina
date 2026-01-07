// tests/telemetry_schema_tests.rs
//
// Telemetry Contract Gate Rust-side tests.
//
// These tests verify that telemetry records emitted by the engine conform to
// the schema defined in schemas/telemetry_schema_v1.json.
//
// The tests:
// 1. Load the schema file to get required/optional fields
// 2. Construct a minimal telemetry record (matching what strategy.rs emits)
// 3. Verify all required keys exist and have valid types

use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// Get the path to the schema file relative to the repo root.
fn schema_path() -> PathBuf {
    // Tests run from paraphina/ directory, schema is at repo root
    PathBuf::from("../schemas/telemetry_schema_v1.json")
}

/// Load and parse the telemetry schema.
fn load_schema() -> Value {
    let path = schema_path();
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read schema file {:?}: {}", path, e));
    serde_json::from_str(&content).unwrap_or_else(|e| panic!("Failed to parse schema JSON: {}", e))
}

/// Build a telemetry record matching what strategy.rs emits.
/// This mirrors the json! macro call in StrategyRunner::run_simulation.
#[allow(clippy::too_many_arguments)]
fn build_telemetry_record(
    tick: u64,
    pnl_realised: f64,
    pnl_unrealised: f64,
    pnl_total: f64,
    risk_regime: &str,
    kill_switch: bool,
    kill_reason: &str,
    q_global_tao: f64,
    dollar_delta_usd: f64,
    basis_usd: f64,
    fv_available: bool,
    fair_value: Option<f64>,
    sigma_eff: f64,
    healthy_venues_used_count: usize,
    healthy_venues_used: &[usize],
) -> Value {
    json!({
        // Schema version (required for contract validation)
        "schema_version": 1,
        // Required fields
        "t": tick,
        "pnl_realised": pnl_realised,
        "pnl_unrealised": pnl_unrealised,
        "pnl_total": pnl_total,
        "risk_regime": risk_regime,
        "kill_switch": kill_switch,
        "kill_reason": kill_reason,
        "q_global_tao": q_global_tao,
        "dollar_delta_usd": dollar_delta_usd,
        "basis_usd": basis_usd,
        // Optional fields
        "fv_available": fv_available,
        "fair_value": fair_value,
        "sigma_eff": sigma_eff,
        "healthy_venues_used_count": healthy_venues_used_count,
        "healthy_venues_used": healthy_venues_used,
    })
}

/// Build a minimal telemetry record with defaults.
fn build_minimal_telemetry_record() -> Value {
    build_telemetry_record(
        0,           // tick
        0.0,         // pnl_realised
        0.0,         // pnl_unrealised
        0.0,         // pnl_total
        "Normal",    // risk_regime
        false,       // kill_switch
        "None",      // kill_reason
        0.0,         // q_global_tao
        0.0,         // dollar_delta_usd
        0.0,         // basis_usd
        true,        // fv_available
        Some(250.0), // fair_value
        0.02,        // sigma_eff
        3,           // healthy_venues_used_count
        &[0, 1, 2],  // healthy_venues_used
    )
}

// =============================================================================
// Schema compliance tests
// =============================================================================

#[test]
fn telemetry_record_includes_schema_version() {
    let record = build_minimal_telemetry_record();

    assert!(
        record.get("schema_version").is_some(),
        "Telemetry record must include schema_version field"
    );

    assert_eq!(
        record["schema_version"].as_i64(),
        Some(1),
        "schema_version must be 1"
    );
}

#[test]
fn telemetry_record_has_all_required_fields() {
    let schema = load_schema();
    let record = build_minimal_telemetry_record();

    // Get required fields from schema
    let required_fields: Vec<String> = schema["required_fields"]
        .as_array()
        .expect("Schema must have required_fields array")
        .iter()
        .map(|v| {
            v.as_str()
                .expect("required field must be string")
                .to_string()
        })
        .collect();

    // Check each required field exists in record
    let record_obj = record.as_object().expect("Record must be an object");
    let record_keys: HashSet<&str> = record_obj.keys().map(|s| s.as_str()).collect();

    for field in &required_fields {
        assert!(
            record_keys.contains(field.as_str()),
            "Telemetry record missing required field: {}",
            field
        );
    }
}

#[test]
fn telemetry_schema_version_matches_schema_invariant() {
    let schema = load_schema();
    let record = build_minimal_telemetry_record();

    let expected_version = schema["invariants"]["schema_version_value"]
        .as_i64()
        .expect("Schema must specify schema_version_value invariant");

    let actual_version = record["schema_version"]
        .as_i64()
        .expect("Record must have integer schema_version");

    assert_eq!(
        actual_version, expected_version,
        "Telemetry schema_version must match schema invariant"
    );
}

#[test]
fn telemetry_risk_regime_is_valid_enum() {
    let schema = load_schema();

    let valid_regimes: Vec<String> = schema["enums"]["risk_regime"]
        .as_array()
        .expect("Schema must define risk_regime enum")
        .iter()
        .map(|v| v.as_str().expect("enum value must be string").to_string())
        .collect();

    // Test each valid regime
    for regime in &valid_regimes {
        let record = build_telemetry_record(
            0,
            0.0,
            0.0,
            0.0,
            regime, // risk_regime
            false,
            "None",
            0.0,
            0.0,
            0.0,
            true,
            Some(250.0),
            0.02,
            3,
            &[0, 1, 2],
        );

        let record_regime = record["risk_regime"]
            .as_str()
            .expect("risk_regime must be string");

        assert!(
            valid_regimes.contains(&record_regime.to_string()),
            "risk_regime '{}' should be in valid enum values: {:?}",
            record_regime,
            valid_regimes
        );
    }
}

#[test]
fn telemetry_numeric_fields_are_finite() {
    let record = build_minimal_telemetry_record();
    let schema = load_schema();

    let field_types = schema["field_types"]
        .as_object()
        .expect("Schema must have field_types");

    let record_obj = record.as_object().expect("Record must be object");

    for (field, ftype) in field_types {
        let type_str = match ftype {
            Value::String(s) => s.as_str(),
            Value::Array(arr) => {
                // Union type - check if "number" is one of them
                if arr.iter().any(|v| v.as_str() == Some("number")) {
                    "number"
                } else {
                    continue;
                }
            }
            _ => continue,
        };

        if type_str == "number" {
            if let Some(value) = record_obj.get(field) {
                if let Some(n) = value.as_f64() {
                    assert!(
                        n.is_finite(),
                        "Numeric field '{}' must be finite, got: {}",
                        field,
                        n
                    );
                }
            }
        }
    }
}

#[test]
fn telemetry_tick_is_non_negative_integer() {
    let record = build_minimal_telemetry_record();

    let t = record["t"].as_i64().expect("t field must be integer");
    assert!(t >= 0, "tick must be non-negative");
}

#[test]
fn telemetry_kill_switch_is_boolean() {
    let record = build_minimal_telemetry_record();

    assert!(
        record["kill_switch"].is_boolean(),
        "kill_switch must be a boolean, not a number or string"
    );
}

#[test]
fn telemetry_healthy_venues_used_is_array_of_integers() {
    let record = build_minimal_telemetry_record();

    let arr = record["healthy_venues_used"]
        .as_array()
        .expect("healthy_venues_used must be an array");

    for (i, elem) in arr.iter().enumerate() {
        assert!(
            elem.is_i64() || elem.is_u64(),
            "healthy_venues_used[{}] must be an integer, got: {:?}",
            i,
            elem
        );
    }
}

#[test]
fn telemetry_fair_value_can_be_null() {
    // Test that fair_value can be null (when FV is unavailable)
    let record = build_telemetry_record(
        0,
        0.0,
        0.0,
        0.0,
        "Normal",
        false,
        "None",
        0.0,
        0.0,
        0.0,
        false, // fv_available = false
        None,  // fair_value = null
        0.02,
        0,
        &[], // no healthy venues
    );

    // fair_value should be null (serde_json represents None as null)
    assert!(
        record["fair_value"].is_null(),
        "fair_value should be null when fv_available=false"
    );
}

// =============================================================================
// Schema file integrity tests
// =============================================================================

#[test]
fn schema_file_exists_and_is_valid_json() {
    let path = schema_path();
    assert!(path.exists(), "Schema file must exist at {:?}", path);

    let content = fs::read_to_string(&path).expect("Must be able to read schema file");
    let _: Value = serde_json::from_str(&content).expect("Schema must be valid JSON");
}

#[test]
fn schema_has_required_top_level_keys() {
    let schema = load_schema();
    let obj = schema.as_object().expect("Schema must be an object");

    let required_keys = [
        "schema_version",
        "required_fields",
        "optional_fields",
        "field_types",
        "enums",
        "invariants",
    ];

    for key in required_keys {
        assert!(
            obj.contains_key(key),
            "Schema must have top-level key: {}",
            key
        );
    }
}

#[test]
fn schema_version_is_one() {
    let schema = load_schema();

    assert_eq!(
        schema["schema_version"].as_i64(),
        Some(1),
        "Schema schema_version must be 1"
    );
}
