// src/rl/safety.rs
//
// Deterministic safety layer for policy actions.

use super::policy::PolicyAction;

#[derive(Debug, Clone)]
pub struct SafetyResult {
    pub raw: PolicyAction,
    pub applied: PolicyAction,
    pub rejection_reasons: Vec<String>,
}

pub struct SafetyLayer;

impl SafetyLayer {
    pub fn apply(action: &PolicyAction, num_venues: usize) -> SafetyResult {
        let mut applied = action.clone();
        let mut reasons = Vec::new();

        normalize_len(
            &mut applied.spread_scale,
            num_venues,
            1.0,
            "spread_scale",
            &mut reasons,
        );
        normalize_len(
            &mut applied.size_scale,
            num_venues,
            1.0,
            "size_scale",
            &mut reasons,
        );
        normalize_len(
            &mut applied.rprice_offset_usd,
            num_venues,
            0.0,
            "rprice_offset_usd",
            &mut reasons,
        );
        normalize_len(
            &mut applied.hedge_venue_weights,
            num_venues,
            if num_venues > 0 {
                1.0 / num_venues as f64
            } else {
                0.0
            },
            "hedge_venue_weights",
            &mut reasons,
        );

        clamp_vec(
            &mut applied.spread_scale,
            0.5,
            3.0,
            "spread_scale",
            1.0,
            &mut reasons,
        );
        clamp_vec(
            &mut applied.size_scale,
            0.0,
            2.0,
            "size_scale",
            1.0,
            &mut reasons,
        );
        clamp_vec(
            &mut applied.rprice_offset_usd,
            -10.0,
            10.0,
            "rprice_offset_usd",
            0.0,
            &mut reasons,
        );
        applied.hedge_scale = clamp_scalar(
            applied.hedge_scale,
            0.0,
            2.0,
            "hedge_scale",
            1.0,
            &mut reasons,
        );
        clamp_vec(
            &mut applied.hedge_venue_weights,
            0.0,
            1.0,
            "hedge_venue_weights",
            if num_venues > 0 {
                1.0 / num_venues as f64
            } else {
                0.0
            },
            &mut reasons,
        );

        let sum: f64 = applied.hedge_venue_weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            reasons.push("hedge_venue_weights_normalized_to_uniform".to_string());
            let val = if num_venues > 0 {
                1.0 / num_venues as f64
            } else {
                0.0
            };
            applied.hedge_venue_weights.fill(val);
        } else if (sum - 1.0).abs() > 1e-6 {
            reasons.push("hedge_venue_weights_normalized".to_string());
            for w in &mut applied.hedge_venue_weights {
                *w /= sum;
            }
        }

        if applied.policy_version.is_empty() {
            applied.policy_version = "unknown".to_string();
            reasons.push("policy_version_empty".to_string());
        }

        SafetyResult {
            raw: action.clone(),
            applied,
            rejection_reasons: reasons,
        }
    }
}

#[allow(clippy::ptr_arg)]
fn normalize_len(
    vec: &mut Vec<f64>,
    len: usize,
    default: f64,
    name: &str,
    reasons: &mut Vec<String>,
) {
    if vec.len() == len {
        return;
    }
    reasons.push(format!("{}_len_mismatch:{}->{}", name, vec.len(), len));
    vec.resize(len, default);
}

fn clamp_vec(
    vec: &mut [f64],
    min: f64,
    max: f64,
    name: &str,
    default: f64,
    reasons: &mut Vec<String>,
) {
    for (i, v) in vec.iter_mut().enumerate() {
        if !v.is_finite() {
            *v = default;
            reasons.push(format!("{}_nan_or_inf:{}", name, i));
            continue;
        }
        let clamped = v.clamp(min, max);
        if (*v - clamped).abs() > 1e-12 {
            *v = clamped;
            reasons.push(format!("{}_clamped:{}", name, i));
        }
    }
}

fn clamp_scalar(
    value: f64,
    min: f64,
    max: f64,
    name: &str,
    default: f64,
    reasons: &mut Vec<String>,
) -> f64 {
    if !value.is_finite() {
        reasons.push(format!("{}_nan_or_inf", name));
        return default;
    }
    let clamped = value.clamp(min, max);
    if (value - clamped).abs() > 1e-12 {
        reasons.push(format!("{}_clamped", name));
    }
    clamped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::action_encoding::encode_action;

    #[test]
    fn safety_layer_is_deterministic() {
        let mut action = PolicyAction::identity(3, "p");
        action.spread_scale = vec![10.0, f64::NAN];
        action.size_scale = vec![2.5, -1.0, 1.0];
        action.rprice_offset_usd = vec![100.0, -100.0, 0.0];
        action.hedge_scale = 10.0;
        action.hedge_venue_weights = vec![0.0, 0.0];

        let a = SafetyLayer::apply(&action, 3);
        let b = SafetyLayer::apply(&action, 3);
        assert_eq!(a.applied, b.applied);
        assert_eq!(a.rejection_reasons, b.rejection_reasons);
    }

    #[test]
    fn safety_layer_serialization_stable() {
        let action = PolicyAction::identity(2, "p");
        let result = SafetyLayer::apply(&action, 2);
        let raw_vec = encode_action(&result.raw, 2);
        let applied_vec = encode_action(&result.applied, 2);
        let raw_json = serde_json::to_string(&raw_vec).unwrap();
        let applied_json = serde_json::to_string(&applied_vec).unwrap();
        let raw_json2 = serde_json::to_string(&raw_vec).unwrap();
        let applied_json2 = serde_json::to_string(&applied_vec).unwrap();
        assert_eq!(raw_json, raw_json2);
        assert_eq!(applied_json, applied_json2);
    }
}
