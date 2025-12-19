// src/rl/action_encoding.rs
//
// RL-2: Versioned action vector encoding for behaviour cloning.
//
// Per ROADMAP.md RL-2, this module provides:
// - Bounded, stable action vector encoding for policy outputs
// - Version-controlled schema for reproducibility
// - Encode/decode helpers with round-trip + clamping semantics
//
// Design:
// - Action vector is a flat f32 array suitable for neural network output
// - All values are normalized to bounded ranges (typically [-1, 1] or [0, 1])
// - PolicyAction fields map deterministically to/from the vector

use serde::{Deserialize, Serialize};

use super::policy::PolicyAction;

/// Current action encoding version.
/// Increment when changing the encoding schema.
pub const ACTION_VERSION: u32 = 1;

/// Action encoding bounds and metadata.
///
/// Per-venue fields (repeated for each venue):
/// - spread_scale: [0.5, 3.0] -> normalized to [0, 1]
/// - size_scale: [0.0, 2.0] -> normalized to [0, 1]
/// - rprice_offset_usd: [-10.0, 10.0] -> normalized to [-1, 1]
///
/// Global fields:
/// - hedge_scale: [0.0, 2.0] -> normalized to [0, 1]
/// - hedge_venue_weights: simplex (sum to 1, each in [0, 1])
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionEncodingSpec {
    /// Schema version.
    pub version: u32,
    /// Number of venues.
    pub num_venues: usize,
    /// Total dimension of the action vector.
    pub action_dim: usize,
}

impl ActionEncodingSpec {
    /// Create a new spec for the given number of venues.
    pub fn new(num_venues: usize) -> Self {
        // Per-venue: spread_scale, size_scale, rprice_offset_usd (3 per venue)
        // Global: hedge_scale (1) + hedge_venue_weights (num_venues)
        let action_dim = num_venues * 3 + 1 + num_venues;

        Self {
            version: ACTION_VERSION,
            num_venues,
            action_dim,
        }
    }

    /// Validate that a vector has the expected dimension.
    pub fn validate_dim(&self, vec: &[f32]) -> bool {
        vec.len() == self.action_dim
    }
}

/// Bounds for action encoding.
pub mod bounds {
    /// Spread scale bounds (raw space).
    pub const SPREAD_SCALE_MIN: f32 = 0.5;
    pub const SPREAD_SCALE_MAX: f32 = 3.0;

    /// Size scale bounds (raw space).
    pub const SIZE_SCALE_MIN: f32 = 0.0;
    pub const SIZE_SCALE_MAX: f32 = 2.0;

    /// Reservation price offset bounds (raw space, USD).
    pub const RPRICE_OFFSET_MIN: f32 = -10.0;
    pub const RPRICE_OFFSET_MAX: f32 = 10.0;

    /// Hedge scale bounds (raw space).
    pub const HEDGE_SCALE_MIN: f32 = 0.0;
    pub const HEDGE_SCALE_MAX: f32 = 2.0;

    /// Hedge venue weight bounds (raw space, before normalization).
    pub const HEDGE_WEIGHT_MIN: f32 = 0.0;
    pub const HEDGE_WEIGHT_MAX: f32 = 1.0;
}

/// Normalize a value from [min, max] to [0, 1].
#[inline]
fn normalize_01(val: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        return 0.5;
    }
    ((val - min) / (max - min)).clamp(0.0, 1.0)
}

/// Denormalize a value from [0, 1] to [min, max].
#[inline]
fn denormalize_01(val: f32, min: f32, max: f32) -> f32 {
    let clamped = val.clamp(0.0, 1.0);
    min + clamped * (max - min)
}

/// Normalize a value from [min, max] to [-1, 1].
#[inline]
fn normalize_symmetric(val: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        return 0.0;
    }
    let mid = (min + max) / 2.0;
    let half_range = (max - min) / 2.0;
    ((val - mid) / half_range).clamp(-1.0, 1.0)
}

/// Denormalize a value from [-1, 1] to [min, max].
#[inline]
fn denormalize_symmetric(val: f32, min: f32, max: f32) -> f32 {
    let clamped = val.clamp(-1.0, 1.0);
    let mid = (min + max) / 2.0;
    let half_range = (max - min) / 2.0;
    mid + clamped * half_range
}

/// Encode a PolicyAction to a normalized f32 vector.
///
/// The encoding is deterministic and version-controlled.
/// Layout: [spread_scale_0..n, size_scale_0..n, rprice_offset_0..n, hedge_scale, hedge_weights_0..n]
pub fn encode_action(action: &PolicyAction, num_venues: usize) -> Vec<f32> {
    let spec = ActionEncodingSpec::new(num_venues);
    let mut vec = Vec::with_capacity(spec.action_dim);

    // Per-venue: spread_scale (normalized to [0, 1])
    for i in 0..num_venues {
        let val = action.spread_scale.get(i).copied().unwrap_or(1.0) as f32;
        vec.push(normalize_01(
            val,
            bounds::SPREAD_SCALE_MIN,
            bounds::SPREAD_SCALE_MAX,
        ));
    }

    // Per-venue: size_scale (normalized to [0, 1])
    for i in 0..num_venues {
        let val = action.size_scale.get(i).copied().unwrap_or(1.0) as f32;
        vec.push(normalize_01(
            val,
            bounds::SIZE_SCALE_MIN,
            bounds::SIZE_SCALE_MAX,
        ));
    }

    // Per-venue: rprice_offset_usd (normalized to [-1, 1])
    for i in 0..num_venues {
        let val = action.rprice_offset_usd.get(i).copied().unwrap_or(0.0) as f32;
        vec.push(normalize_symmetric(
            val,
            bounds::RPRICE_OFFSET_MIN,
            bounds::RPRICE_OFFSET_MAX,
        ));
    }

    // Global: hedge_scale (normalized to [0, 1])
    vec.push(normalize_01(
        action.hedge_scale as f32,
        bounds::HEDGE_SCALE_MIN,
        bounds::HEDGE_SCALE_MAX,
    ));

    // Global: hedge_venue_weights (already in [0, 1], but we normalize the sum)
    for i in 0..num_venues {
        let val = action.hedge_venue_weights.get(i).copied().unwrap_or(0.0) as f32;
        vec.push(val.clamp(bounds::HEDGE_WEIGHT_MIN, bounds::HEDGE_WEIGHT_MAX));
    }

    vec
}

/// Decode a normalized f32 vector to a PolicyAction.
///
/// Values are clamped to valid ranges. The hedge_venue_weights are
/// normalized to sum to 1.0 (simplex projection).
pub fn decode_action(vec: &[f32], num_venues: usize, policy_version: &str) -> PolicyAction {
    let spec = ActionEncodingSpec::new(num_venues);

    // Validate dimension
    if vec.len() != spec.action_dim {
        // Return identity action if dimension mismatch
        return PolicyAction::identity(num_venues, policy_version);
    }

    let mut idx = 0;

    // Per-venue: spread_scale
    let mut spread_scale = Vec::with_capacity(num_venues);
    for _ in 0..num_venues {
        let normalized = vec[idx];
        spread_scale.push(denormalize_01(
            normalized,
            bounds::SPREAD_SCALE_MIN,
            bounds::SPREAD_SCALE_MAX,
        ) as f64);
        idx += 1;
    }

    // Per-venue: size_scale
    let mut size_scale = Vec::with_capacity(num_venues);
    for _ in 0..num_venues {
        let normalized = vec[idx];
        size_scale.push(
            denormalize_01(normalized, bounds::SIZE_SCALE_MIN, bounds::SIZE_SCALE_MAX) as f64,
        );
        idx += 1;
    }

    // Per-venue: rprice_offset_usd
    let mut rprice_offset_usd = Vec::with_capacity(num_venues);
    for _ in 0..num_venues {
        let normalized = vec[idx];
        rprice_offset_usd.push(denormalize_symmetric(
            normalized,
            bounds::RPRICE_OFFSET_MIN,
            bounds::RPRICE_OFFSET_MAX,
        ) as f64);
        idx += 1;
    }

    // Global: hedge_scale
    let hedge_scale = denormalize_01(vec[idx], bounds::HEDGE_SCALE_MIN, bounds::HEDGE_SCALE_MAX);
    idx += 1;

    // Global: hedge_venue_weights (normalize to simplex)
    let mut hedge_venue_weights = Vec::with_capacity(num_venues);
    let mut weight_sum = 0.0_f32;
    for _ in 0..num_venues {
        let w = vec[idx].clamp(0.0, 1.0);
        hedge_venue_weights.push(w as f64);
        weight_sum += w;
        idx += 1;
    }

    // Normalize to simplex
    if weight_sum > 0.0 {
        for w in &mut hedge_venue_weights {
            *w /= weight_sum as f64;
        }
    } else {
        // Uniform if all zeros
        let uniform = 1.0 / num_venues as f64;
        hedge_venue_weights.fill(uniform);
    }

    PolicyAction {
        policy_version: policy_version.to_string(),
        policy_id: None,
        spread_scale,
        size_scale,
        rprice_offset_usd,
        hedge_scale: hedge_scale as f64,
        hedge_venue_weights,
    }
}

/// Compute the L2 distance between two action vectors.
pub fn action_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute the mean absolute error between two action vectors.
pub fn action_mae(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_VENUES: usize = 5;
    const POLICY_VERSION: &str = "test-v1";

    #[test]
    fn test_action_spec() {
        let spec = ActionEncodingSpec::new(5);
        assert_eq!(spec.version, ACTION_VERSION);
        assert_eq!(spec.num_venues, 5);
        // 5*3 (per-venue) + 1 (hedge_scale) + 5 (hedge_weights) = 21
        assert_eq!(spec.action_dim, 21);
    }

    #[test]
    fn test_identity_action_encode_decode() {
        let action = PolicyAction::identity(NUM_VENUES, POLICY_VERSION);
        let encoded = encode_action(&action, NUM_VENUES);
        let decoded = decode_action(&encoded, NUM_VENUES, POLICY_VERSION);

        // Check spread_scale (identity = 1.0 for all)
        for (orig, dec) in action.spread_scale.iter().zip(decoded.spread_scale.iter()) {
            assert!(
                (orig - dec).abs() < 1e-4,
                "spread_scale mismatch: {} vs {}",
                orig,
                dec
            );
        }

        // Check size_scale (identity = 1.0 for all)
        for (orig, dec) in action.size_scale.iter().zip(decoded.size_scale.iter()) {
            assert!(
                (orig - dec).abs() < 1e-4,
                "size_scale mismatch: {} vs {}",
                orig,
                dec
            );
        }

        // Check rprice_offset_usd (identity = 0.0 for all)
        for (orig, dec) in action
            .rprice_offset_usd
            .iter()
            .zip(decoded.rprice_offset_usd.iter())
        {
            assert!(
                (orig - dec).abs() < 1e-4,
                "rprice_offset mismatch: {} vs {}",
                orig,
                dec
            );
        }

        // Check hedge_scale (identity = 1.0)
        assert!(
            (action.hedge_scale - decoded.hedge_scale).abs() < 1e-4,
            "hedge_scale mismatch: {} vs {}",
            action.hedge_scale,
            decoded.hedge_scale
        );

        // Check hedge_venue_weights (identity = uniform)
        for (orig, dec) in action
            .hedge_venue_weights
            .iter()
            .zip(decoded.hedge_venue_weights.iter())
        {
            assert!(
                (orig - dec).abs() < 1e-4,
                "hedge_weight mismatch: {} vs {}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_round_trip_various_values() {
        let action = PolicyAction {
            policy_version: POLICY_VERSION.to_string(),
            policy_id: Some("test".to_string()),
            spread_scale: vec![0.5, 1.0, 1.5, 2.0, 3.0],
            size_scale: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            rprice_offset_usd: vec![-10.0, -5.0, 0.0, 5.0, 10.0],
            hedge_scale: 1.5,
            hedge_venue_weights: vec![0.4, 0.3, 0.2, 0.1, 0.0],
        };

        let encoded = encode_action(&action, NUM_VENUES);
        let decoded = decode_action(&encoded, NUM_VENUES, POLICY_VERSION);

        // Check all fields with tolerance
        for i in 0..NUM_VENUES {
            assert!(
                (action.spread_scale[i] - decoded.spread_scale[i]).abs() < 1e-3,
                "spread_scale[{}] mismatch",
                i
            );
            assert!(
                (action.size_scale[i] - decoded.size_scale[i]).abs() < 1e-3,
                "size_scale[{}] mismatch",
                i
            );
            assert!(
                (action.rprice_offset_usd[i] - decoded.rprice_offset_usd[i]).abs() < 0.1,
                "rprice_offset[{}] mismatch",
                i
            );
        }

        assert!(
            (action.hedge_scale - decoded.hedge_scale).abs() < 1e-3,
            "hedge_scale mismatch"
        );
    }

    #[test]
    fn test_clamping_out_of_bounds() {
        // Create action with out-of-bounds values
        let action = PolicyAction {
            policy_version: POLICY_VERSION.to_string(),
            policy_id: None,
            spread_scale: vec![0.0, 5.0, 1.0, 1.0, 1.0], // 0.0 < 0.5, 5.0 > 3.0
            size_scale: vec![-1.0, 3.0, 1.0, 1.0, 1.0],  // -1.0 < 0.0, 3.0 > 2.0
            rprice_offset_usd: vec![-20.0, 20.0, 0.0, 0.0, 0.0], // Out of [-10, 10]
            hedge_scale: 5.0,                            // > 2.0
            hedge_venue_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
        };

        let encoded = encode_action(&action, NUM_VENUES);
        let decoded = decode_action(&encoded, NUM_VENUES, POLICY_VERSION);

        // Values should be clamped to valid ranges
        assert!(
            decoded.spread_scale[0] >= 0.5,
            "spread_scale should be clamped to min"
        );
        assert!(
            decoded.spread_scale[1] <= 3.0,
            "spread_scale should be clamped to max"
        );
        assert!(
            decoded.size_scale[0] >= 0.0,
            "size_scale should be clamped to min"
        );
        assert!(
            decoded.size_scale[1] <= 2.0,
            "size_scale should be clamped to max"
        );
        assert!(
            decoded.rprice_offset_usd[0] >= -10.0,
            "rprice_offset should be clamped to min"
        );
        assert!(
            decoded.rprice_offset_usd[1] <= 10.0,
            "rprice_offset should be clamped to max"
        );
        assert!(
            decoded.hedge_scale <= 2.0,
            "hedge_scale should be clamped to max"
        );
    }

    #[test]
    fn test_encoding_determinism() {
        let action = PolicyAction::identity(NUM_VENUES, POLICY_VERSION);

        let encoded1 = encode_action(&action, NUM_VENUES);
        let encoded2 = encode_action(&action, NUM_VENUES);

        assert_eq!(
            encoded1, encoded2,
            "Encoding should be deterministic: {:?} vs {:?}",
            encoded1, encoded2
        );
    }

    #[test]
    fn test_decode_wrong_dimension() {
        let too_short = vec![0.5; 10];
        let decoded = decode_action(&too_short, NUM_VENUES, POLICY_VERSION);

        // Should return identity action
        assert!(decoded.is_identity());
    }

    #[test]
    fn test_hedge_weights_normalization() {
        // Create encoded vector with non-normalized weights
        let spec = ActionEncodingSpec::new(NUM_VENUES);
        let mut encoded = vec![0.5_f32; spec.action_dim];

        // Set hedge weights to non-normalized values at the end
        let weight_start = spec.action_dim - NUM_VENUES;
        encoded[weight_start] = 0.5;
        encoded[weight_start + 1] = 0.3;
        encoded[weight_start + 2] = 0.2;
        encoded[weight_start + 3] = 0.0;
        encoded[weight_start + 4] = 0.0;

        let decoded = decode_action(&encoded, NUM_VENUES, POLICY_VERSION);

        // Weights should sum to 1.0
        let sum: f64 = decoded.hedge_venue_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Hedge weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        let dist = action_l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6, "L2 distance should be 5.0");
    }

    #[test]
    fn test_mae() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 6.0, 0.0];

        let mae = action_mae(&a, &b);
        assert!((mae - 3.0).abs() < 1e-6, "MAE should be 3.0");
    }

    #[test]
    fn test_normalize_denormalize_01() {
        let val = 1.75; // middle of [0.5, 3.0]
        let normalized = normalize_01(val, 0.5, 3.0);
        let denormalized = denormalize_01(normalized, 0.5, 3.0);

        assert!(
            (val - denormalized).abs() < 1e-6,
            "Round-trip failed: {} vs {}",
            val,
            denormalized
        );
    }

    #[test]
    fn test_normalize_denormalize_symmetric() {
        let val = 5.0; // middle of [-10, 10]
        let normalized = normalize_symmetric(val, -10.0, 10.0);
        let denormalized = denormalize_symmetric(normalized, -10.0, 10.0);

        assert!(
            (val - denormalized).abs() < 1e-6,
            "Round-trip failed: {} vs {}",
            val,
            denormalized
        );
    }
}
