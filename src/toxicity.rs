// src/toxicity.rs
//
// Per-venue toxicity + venue health, aligned with Section 7:
//
//  - tox_v is driven (for now) only by relative local volatility vs σ_eff.
//  - Later we can add markouts / order-book imbalance / flow features.
//  - VenueStatus is set from toxicity via low/medium/high thresholds.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::VenueStatus;

/// Core toxicity updater.
///
/// Uses:
///   - venue.local_vol_short
///   - global state.sigma_eff
///   - cfg.toxicity.{vol_tox_scale, tox_med_threshold, tox_high_threshold}
pub fn update_toxicity(state: &mut GlobalState, cfg: &Config) {
    let tox_cfg = &cfg.toxicity;

    // Avoid divide-by-zero if sigma_eff is tiny.
    let sigma_eff = if state.sigma_eff > 0.0 {
        state.sigma_eff
    } else {
        1e-6
    };

    for venue in &mut state.venues {
        // ---- f1: relative venue vol vs global σ_eff ----
        let local_vol = venue.local_vol_short.max(0.0);
        let ratio = local_vol / sigma_eff.max(1e-9);

        // Map ratio>1 into [0,1] using vol_tox_scale.
        // If ratio == 1      → feature = 0
        // If ratio >= 1+scale→ feature ≈ 1
        let vol_feature = ((ratio - 1.0) / tox_cfg.vol_tox_scale)
            .max(0.0)
            .min(1.0);

        // TODO: f2..f5 (markouts, imbalance, flow, throughput).
        // For now they are 0, so toxicity is just vol_feature.
        let tox = vol_feature;

        venue.toxicity = tox;

        // ---- Health classification from toxicity ----
        venue.status = if tox < tox_cfg.tox_med_threshold {
            VenueStatus::Healthy
        } else if tox < tox_cfg.tox_high_threshold {
            VenueStatus::Degraded
        } else {
            VenueStatus::Disabled
        };
    }
}

/// Convenience wrapper if you ever want the same signature as before.
pub fn update_toxicity_and_health(state: &mut GlobalState, cfg: &Config) {
    update_toxicity(state, cfg);
}
