// src/toxicity.rs
//
// Very small toxicity scoring module.
//
// It mirrors the structure of Section 7 of the whitepaper but, for now,
// only uses feature f1 (relative local vol vs global sigma_eff).
// The other features (markouts, imbalance, flow, throughput) are
// placeholders and stay at 0 until we wire real data.

use crate::state::GlobalState;
use crate::types::VenueStatus;

// How aggressive we treat "extra volatility" as toxic.
// Larger VOL_TOX_SCALE means we need a bigger vol jump to hit f1 = 1.
const VOL_TOX_SCALE: f64 = 1.0;

// Regime thresholds.
const TOX_MED_THRESHOLD: f64 = 0.4;
const TOX_HIGH_THRESHOLD: f64 = 0.8;

// Feature weights w1..w5. For now w1 dominates because we only
// implement f1; the others stay 0.0 until we add real data.
const W1: f64 = 0.8;
const W2: f64 = 0.05;
const W3: f64 = 0.05;
const W4: f64 = 0.05;
const W5: f64 = 0.05;

// Clamp to [0,1].
fn clip01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Update per-venue toxicity score and status.
///
/// - toxicity_v ∈ [0,1]
/// - status_v ∈ {Healthy, Degraded, Disabled}
pub fn update_toxicity(state: &mut GlobalState) {
    let sigma_eff = state.sigma_eff;

    for v in state.venues.iter_mut() {
        let mut tox = 0.0;

        if sigma_eff > 0.0 {
            // f1: relative venue vol vs global vol.
            //
            // ratio_v = local_vol_short_v / sigma_eff
            // f1 = clip((ratio_v - 1) / VOL_TOX_SCALE, 0, 1)
            let ratio = (v.local_vol_short / sigma_eff).max(0.0);
            let f1 = clip01((ratio - 1.0) / VOL_TOX_SCALE);

            // Placeholders for now (we will wire these later from real data):
            let f2 = 0.0; // negative markouts fraction
            let f3 = 0.0; // order book imbalance
            let f4 = 0.0; // directional aggressive flow
            let f5 = 0.0; // total throughput

            tox = clip01(W1 * f1 + W2 * f2 + W3 * f3 + W4 * f4 + W5 * f5);
        }

        v.toxicity = tox;

        v.status = if tox < TOX_MED_THRESHOLD {
            VenueStatus::Healthy
        } else if tox < TOX_HIGH_THRESHOLD {
            VenueStatus::Degraded
        } else {
            VenueStatus::Disabled
        };
    }
}
