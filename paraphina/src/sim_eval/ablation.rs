// src/sim_eval/ablation.rs
//
// Ablation harness for research experiments.
//
// This module defines the set of supported ablations that can be applied
// to simulation runs to study the impact of individual components.
//
// Each ablation disables a specific feature:
// - disable_vol_floor: bypass sigma_eff floor (do not clamp to sigma_min)
// - disable_fair_value_gating: gating always allows (never blocks quoting)
// - disable_toxicity_gate: toxicity gating never blocks/disables venues
// - disable_risk_regime: always use the default/normal regime (no regime switching)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

/// All valid ablation IDs.
/// These are stable strings that must not change across versions.
pub const VALID_ABLATION_IDS: &[&str] = &[
    "disable_vol_floor",
    "disable_fair_value_gating",
    "disable_toxicity_gate",
    "disable_risk_regime",
];

/// Ablation descriptions for the `ablations` subcommand.
pub const ABLATION_DESCRIPTIONS: &[(&str, &str)] = &[
    (
        "disable_vol_floor",
        "Bypass sigma_eff floor (do not clamp to sigma_min)",
    ),
    (
        "disable_fair_value_gating",
        "Gating always allows (never blocks quoting)",
    ),
    (
        "disable_toxicity_gate",
        "Toxicity gating never blocks/disables venues",
    ),
    (
        "disable_risk_regime",
        "Always use the default/normal regime (no regime switching/scaling)",
    ),
];

/// Active ablation set for a simulation run.
///
/// Ablations are stored as a sorted, deduplicated set of strings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AblationSet {
    /// Sorted list of active ablation IDs.
    #[serde(default)]
    pub ablations: Vec<String>,
}

impl AblationSet {
    /// Create a new empty ablation set (baseline behavior).
    pub fn new() -> Self {
        Self {
            ablations: Vec::new(),
        }
    }

    /// Create an ablation set from a list of ablation IDs.
    ///
    /// Returns an error if any ID is not recognized.
    pub fn from_ids(ids: &[String]) -> Result<Self, AblationError> {
        let mut set = BTreeSet::new();

        for id in ids {
            if !VALID_ABLATION_IDS.contains(&id.as_str()) {
                return Err(AblationError::UnknownAblation {
                    id: id.clone(),
                    valid: VALID_ABLATION_IDS.iter().map(|s| s.to_string()).collect(),
                });
            }
            set.insert(id.clone());
        }

        Ok(Self {
            ablations: set.into_iter().collect(),
        })
    }

    /// Check if the set is empty (baseline behavior).
    pub fn is_empty(&self) -> bool {
        self.ablations.is_empty()
    }

    /// Check if a specific ablation is active.
    pub fn has(&self, id: &str) -> bool {
        self.ablations.iter().any(|s| s == id)
    }

    /// Check if vol floor is disabled.
    pub fn disable_vol_floor(&self) -> bool {
        self.has("disable_vol_floor")
    }

    /// Check if fair value gating is disabled.
    pub fn disable_fair_value_gating(&self) -> bool {
        self.has("disable_fair_value_gating")
    }

    /// Check if toxicity gating is disabled.
    pub fn disable_toxicity_gate(&self) -> bool {
        self.has("disable_toxicity_gate")
    }

    /// Check if risk regime switching is disabled.
    pub fn disable_risk_regime(&self) -> bool {
        self.has("disable_risk_regime")
    }

    /// Compute a short hash of the ablation set for directory naming.
    ///
    /// Returns "baseline" if empty, otherwise a 6-char hex hash.
    pub fn short_hash(&self) -> String {
        if self.ablations.is_empty() {
            return "baseline".to_string();
        }

        let mut hasher = Sha256::new();
        for id in &self.ablations {
            hasher.update(id.as_bytes());
            hasher.update(b"|");
        }
        let hash = hasher.finalize();

        // Take first 3 bytes = 6 hex chars
        format!("{:02x}{:02x}{:02x}", hash[0], hash[1], hash[2])
    }

    /// Compute bytes for inclusion in a determinism checksum.
    ///
    /// The format is deterministic: sorted ablation IDs joined by "|".
    pub fn checksum_bytes(&self) -> Vec<u8> {
        self.ablations.join("|").into_bytes()
    }

    /// Compute a deterministic directory suffix for the ablation set.
    ///
    /// Returns:
    /// - `__baseline` if no ablations are active
    /// - `__<id>` if one ablation is active
    /// - `__<id1>__<id2>` if multiple ablations (sorted alphabetically)
    ///
    /// Examples:
    /// - `[]` → `__baseline`
    /// - `["disable_vol_floor"]` → `__disable_vol_floor`
    /// - `["disable_vol_floor", "disable_toxicity_gate"]` → `__disable_toxicity_gate__disable_vol_floor`
    pub fn dir_suffix(&self) -> String {
        if self.ablations.is_empty() {
            "__baseline".to_string()
        } else {
            // ablations are already sorted (BTreeSet in from_ids)
            format!("__{}", self.ablations.join("__"))
        }
    }
}

impl fmt::Display for AblationSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ablations.is_empty() {
            write!(f, "baseline")
        } else {
            write!(f, "{}", self.ablations.join(","))
        }
    }
}

/// Errors related to ablation handling.
#[derive(Debug, Clone)]
pub enum AblationError {
    /// An unknown ablation ID was specified.
    UnknownAblation { id: String, valid: Vec<String> },
}

impl fmt::Display for AblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AblationError::UnknownAblation { id, valid } => {
                write!(
                    f,
                    "Unknown ablation ID '{}'. Valid ablation IDs are: {}",
                    id,
                    valid.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for AblationError {}

/// Print the list of supported ablations and their descriptions.
pub fn print_ablations() {
    println!("Supported ablation IDs:");
    println!();
    for (id, desc) in ABLATION_DESCRIPTIONS {
        println!("  {:<28} {}", id, desc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_ablation_set() {
        let set = AblationSet::new();
        assert!(set.is_empty());
        assert!(!set.disable_vol_floor());
        assert!(!set.disable_fair_value_gating());
        assert!(!set.disable_toxicity_gate());
        assert!(!set.disable_risk_regime());
        assert_eq!(set.short_hash(), "baseline");
    }

    #[test]
    fn test_valid_ablation_set() {
        let ids = vec![
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();

        assert!(!set.is_empty());
        assert!(set.disable_vol_floor());
        assert!(!set.disable_fair_value_gating());
        assert!(set.disable_toxicity_gate());
        assert!(!set.disable_risk_regime());
    }

    #[test]
    fn test_unknown_ablation_error() {
        let ids = vec!["disable_vol_floor".to_string(), "invalid_id".to_string()];
        let result = AblationSet::from_ids(&ids);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("invalid_id"));
        assert!(msg.contains("disable_vol_floor"));
    }

    #[test]
    fn test_ablations_sorted_and_deduped() {
        let ids = vec![
            "disable_risk_regime".to_string(),
            "disable_vol_floor".to_string(),
            "disable_vol_floor".to_string(), // duplicate
            "disable_toxicity_gate".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();

        // Should be sorted and deduped
        assert_eq!(set.ablations.len(), 3);
        assert_eq!(set.ablations[0], "disable_risk_regime");
        assert_eq!(set.ablations[1], "disable_toxicity_gate");
        assert_eq!(set.ablations[2], "disable_vol_floor");
    }

    #[test]
    fn test_short_hash_determinism() {
        let ids1 = vec![
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let ids2 = vec![
            "disable_toxicity_gate".to_string(),
            "disable_vol_floor".to_string(),
        ];

        let set1 = AblationSet::from_ids(&ids1).unwrap();
        let set2 = AblationSet::from_ids(&ids2).unwrap();

        // Same ablations in different order should produce same hash
        assert_eq!(set1.short_hash(), set2.short_hash());
        assert_ne!(set1.short_hash(), "baseline");
    }

    #[test]
    fn test_different_ablations_different_hash() {
        let ids1 = vec!["disable_vol_floor".to_string()];
        let ids2 = vec!["disable_toxicity_gate".to_string()];

        let set1 = AblationSet::from_ids(&ids1).unwrap();
        let set2 = AblationSet::from_ids(&ids2).unwrap();

        assert_ne!(set1.short_hash(), set2.short_hash());
    }

    #[test]
    fn test_checksum_bytes_determinism() {
        let ids = vec![
            "disable_risk_regime".to_string(),
            "disable_vol_floor".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();

        let bytes = set.checksum_bytes();
        assert_eq!(bytes, b"disable_risk_regime|disable_vol_floor");
    }

    #[test]
    fn test_display() {
        let empty = AblationSet::new();
        assert_eq!(format!("{}", empty), "baseline");

        let ids = vec![
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();
        assert!(format!("{}", set).contains("disable_toxicity_gate"));
        assert!(format!("{}", set).contains("disable_vol_floor"));
    }

    #[test]
    fn test_serde_round_trip() {
        let ids = vec![
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();

        let json = serde_json::to_string(&set).unwrap();
        let parsed: AblationSet = serde_json::from_str(&json).unwrap();

        assert_eq!(set, parsed);
    }

    #[test]
    fn test_serde_empty() {
        let json = "{}";
        let parsed: AblationSet = serde_json::from_str(json).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_dir_suffix_no_ablation() {
        let set = AblationSet::new();
        assert_eq!(set.dir_suffix(), "__baseline");
    }

    #[test]
    fn test_dir_suffix_one_ablation() {
        let ids = vec!["disable_vol_floor".to_string()];
        let set = AblationSet::from_ids(&ids).unwrap();
        assert_eq!(set.dir_suffix(), "__disable_vol_floor");
    }

    #[test]
    fn test_dir_suffix_two_ablations_sorted() {
        // Test that order of input doesn't matter - output is always sorted
        let ids1 = vec![
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let ids2 = vec![
            "disable_toxicity_gate".to_string(),
            "disable_vol_floor".to_string(),
        ];

        let set1 = AblationSet::from_ids(&ids1).unwrap();
        let set2 = AblationSet::from_ids(&ids2).unwrap();

        // Both should produce the same suffix (sorted alphabetically)
        assert_eq!(set1.dir_suffix(), set2.dir_suffix());
        assert_eq!(
            set1.dir_suffix(),
            "__disable_toxicity_gate__disable_vol_floor"
        );
    }

    #[test]
    fn test_dir_suffix_three_ablations_sorted() {
        let ids = vec![
            "disable_risk_regime".to_string(),
            "disable_vol_floor".to_string(),
            "disable_toxicity_gate".to_string(),
        ];
        let set = AblationSet::from_ids(&ids).unwrap();
        assert_eq!(
            set.dir_suffix(),
            "__disable_risk_regime__disable_toxicity_gate__disable_vol_floor"
        );
    }
}
