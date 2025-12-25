// src/sim_eval/env_override.rs
//
// Scoped environment variable override utilities.
//
// Provides a thread-safe mechanism for temporarily setting environment variables
// during scenario execution, then restoring them afterward.
//
// This is used by sim_eval suite for inline scenarios with env_overrides.

use std::collections::BTreeMap;
use std::env;
use std::sync::Mutex;

/// Global mutex to serialize environment modifications.
///
/// Environment variables are process-global, so concurrent modifications
/// from multiple threads would cause race conditions. This mutex ensures
/// only one thread modifies the environment at a time.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Previous environment state for a single variable.
#[derive(Debug, Clone)]
enum PrevEnvValue {
    /// Variable was set to this value before override.
    Set(String),
    /// Variable was not set before override.
    Unset,
}

/// Run a closure with temporary environment variable overrides.
///
/// This function:
/// 1. Acquires a global mutex to prevent concurrent env modifications
/// 2. Saves the current values of all keys in `overrides`
/// 3. Sets all overrides
/// 4. Runs the closure
/// 5. Restores all previous values (or unsets if they weren't set)
///
/// # Arguments
///
/// * `overrides` - Map of environment variable names to their override values
/// * `f` - Closure to run with the overrides in place
///
/// # Returns
///
/// Returns the result of the closure.
///
/// # Thread Safety
///
/// This function holds a global mutex for the duration of the closure execution.
/// If the closure takes a long time, other threads will block waiting for env access.
///
/// # Panic Safety
///
/// If the closure panics, the environment is NOT restored. This is intentional
/// to avoid masking the panic with cleanup errors. Use with care in code that
/// may panic.
///
/// # Example
///
/// ```ignore
/// use std::collections::BTreeMap;
/// use paraphina::sim_eval::with_env_overrides;
///
/// let mut overrides = BTreeMap::new();
/// overrides.insert("MY_VAR".to_string(), "my_value".to_string());
///
/// let result = with_env_overrides(&overrides, || {
///     std::env::var("MY_VAR").unwrap()
/// });
///
/// assert_eq!(result, "my_value");
/// // MY_VAR is now restored to its previous value (or unset)
/// ```
pub fn with_env_overrides<F, R>(overrides: &BTreeMap<String, String>, f: F) -> R
where
    F: FnOnce() -> R,
{
    // Acquire global lock
    let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    // Save previous values
    let mut prev_values: Vec<(String, PrevEnvValue)> = Vec::with_capacity(overrides.len());
    for key in overrides.keys() {
        let prev = match env::var(key) {
            Ok(val) => PrevEnvValue::Set(val),
            Err(_) => PrevEnvValue::Unset,
        };
        prev_values.push((key.clone(), prev));
    }

    // Set overrides
    for (key, value) in overrides {
        env::set_var(key, value);
    }

    // Run the closure
    let result = f();

    // Restore previous values
    for (key, prev) in prev_values {
        match prev {
            PrevEnvValue::Set(val) => env::set_var(&key, val),
            PrevEnvValue::Unset => env::remove_var(&key),
        }
    }

    result
}

/// Parse env_overrides from either a YAML mapping or a list of "KEY=VALUE" strings.
///
/// This function supports two formats:
///
/// 1. YAML mapping (BTreeMap<String, String>):
///    ```yaml
///    env_overrides:
///      KEY1: "value1"
///      KEY2: "value2"
///    ```
///
/// 2. List of "KEY=VALUE" strings:
///    ```yaml
///    env_overrides:
///      - "KEY1=value1"
///      - "KEY2=value2"
///    ```
///
/// # Arguments
///
/// * `value` - A serde_yaml::Value that is either a mapping or sequence
///
/// # Returns
///
/// Returns a BTreeMap of key-value pairs, or an error if parsing fails.
pub fn parse_env_overrides(value: &serde_yaml::Value) -> Result<BTreeMap<String, String>, String> {
    match value {
        serde_yaml::Value::Mapping(map) => {
            let mut result = BTreeMap::new();
            for (k, v) in map {
                let key = k
                    .as_str()
                    .ok_or_else(|| format!("env_override key must be a string, got: {:?}", k))?
                    .to_string();
                let val = match v {
                    serde_yaml::Value::String(s) => s.clone(),
                    serde_yaml::Value::Number(n) => n.to_string(),
                    serde_yaml::Value::Bool(b) => b.to_string(),
                    serde_yaml::Value::Null => String::new(),
                    _ => {
                        return Err(format!(
                            "env_override value for '{}' must be a scalar, got: {:?}",
                            key, v
                        ))
                    }
                };
                result.insert(key, val);
            }
            Ok(result)
        }
        serde_yaml::Value::Sequence(seq) => {
            let mut result = BTreeMap::new();
            for item in seq {
                let s = item.as_str().ok_or_else(|| {
                    format!("env_override list item must be a string, got: {:?}", item)
                })?;
                let (key, val) = s.split_once('=').ok_or_else(|| {
                    format!("env_override list item must be 'KEY=VALUE', got: '{}'", s)
                })?;
                result.insert(key.to_string(), val.to_string());
            }
            Ok(result)
        }
        serde_yaml::Value::Null => Ok(BTreeMap::new()),
        _ => Err(format!(
            "env_overrides must be a mapping or list, got: {:?}",
            value
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_env_overrides_sets_and_restores() {
        // Set a test variable first
        let test_key = "TEST_WITH_ENV_OVERRIDES_1";
        env::set_var(test_key, "original_value");

        let mut overrides = BTreeMap::new();
        overrides.insert(test_key.to_string(), "overridden_value".to_string());

        let inner_value = with_env_overrides(&overrides, || env::var(test_key).unwrap());

        assert_eq!(inner_value, "overridden_value");
        assert_eq!(env::var(test_key).unwrap(), "original_value");

        // Clean up
        env::remove_var(test_key);
    }

    #[test]
    fn test_with_env_overrides_restores_unset() {
        let test_key = "TEST_WITH_ENV_OVERRIDES_2";
        // Ensure it's not set
        env::remove_var(test_key);

        let mut overrides = BTreeMap::new();
        overrides.insert(test_key.to_string(), "temp_value".to_string());

        let inner_value = with_env_overrides(&overrides, || env::var(test_key).unwrap());

        assert_eq!(inner_value, "temp_value");
        assert!(env::var(test_key).is_err()); // Should be unset after
    }

    #[test]
    fn test_with_env_overrides_multiple_vars() {
        let key1 = "TEST_WITH_ENV_OVERRIDES_3A";
        let key2 = "TEST_WITH_ENV_OVERRIDES_3B";
        let key3 = "TEST_WITH_ENV_OVERRIDES_3C";

        env::set_var(key1, "orig1");
        env::remove_var(key2);
        env::set_var(key3, "orig3");

        let mut overrides = BTreeMap::new();
        overrides.insert(key1.to_string(), "new1".to_string());
        overrides.insert(key2.to_string(), "new2".to_string());
        overrides.insert(key3.to_string(), "new3".to_string());

        with_env_overrides(&overrides, || {
            assert_eq!(env::var(key1).unwrap(), "new1");
            assert_eq!(env::var(key2).unwrap(), "new2");
            assert_eq!(env::var(key3).unwrap(), "new3");
        });

        assert_eq!(env::var(key1).unwrap(), "orig1");
        assert!(env::var(key2).is_err());
        assert_eq!(env::var(key3).unwrap(), "orig3");

        // Clean up
        env::remove_var(key1);
        env::remove_var(key3);
    }

    #[test]
    fn test_with_env_overrides_empty() {
        let overrides = BTreeMap::new();
        let result = with_env_overrides(&overrides, || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_parse_env_overrides_mapping() {
        let yaml = r#"
          KEY1: "value1"
          KEY2: "value2"
          KEY3: 123
          KEY4: true
        "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = parse_env_overrides(&value).unwrap();

        assert_eq!(result.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(result.get("KEY2"), Some(&"value2".to_string()));
        assert_eq!(result.get("KEY3"), Some(&"123".to_string()));
        assert_eq!(result.get("KEY4"), Some(&"true".to_string()));
    }

    #[test]
    fn test_parse_env_overrides_list() {
        let yaml = r#"
          - "KEY1=value1"
          - "KEY2=value2"
          - "KEY3=value=with=equals"
        "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = parse_env_overrides(&value).unwrap();

        assert_eq!(result.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(result.get("KEY2"), Some(&"value2".to_string()));
        assert_eq!(result.get("KEY3"), Some(&"value=with=equals".to_string()));
    }

    #[test]
    fn test_parse_env_overrides_null() {
        let value = serde_yaml::Value::Null;
        let result = parse_env_overrides(&value).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_env_overrides_invalid_list_item() {
        let yaml = r#"
          - "VALID=value"
          - "INVALID_NO_EQUALS"
        "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = parse_env_overrides(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("KEY=VALUE"));
    }

    #[test]
    fn test_parse_env_overrides_invalid_type() {
        let yaml = r#"42"#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = parse_env_overrides(&value);
        assert!(result.is_err());
    }

    #[test]
    fn test_env_restoration_exact() {
        // Test that env is restored exactly, including empty strings
        let key = "TEST_EXACT_RESTORE";
        env::set_var(key, "");

        let mut overrides = BTreeMap::new();
        overrides.insert(key.to_string(), "nonempty".to_string());

        with_env_overrides(&overrides, || {
            assert_eq!(env::var(key).unwrap(), "nonempty");
        });

        // Should be restored to empty string, not unset
        assert_eq!(env::var(key).unwrap(), "");

        env::remove_var(key);
    }
}
