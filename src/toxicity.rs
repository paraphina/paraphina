// src/toxicity.rs
//
// Per-venue volatility-based toxicity scoring.
//
// We currently use a single feature:
//
//   f_vol = clip((local_vol_short / sigma_eff - 1) / vol_tox_scale, 0, 1)
//
// and write the final score into `venue.toxicity ∈ [0, 1]`.
//
// Other parts of the engine can interpret this as:
//
//   - tox <  tox_med_threshold   → effectively "Healthy"
//   - tox ∈ [tox_med, tox_high)  → "Warning"
//   - tox ≥  tox_high_threshold  → "Disabled"
//
// We *don't* store a separate health enum/flag here to keep it compatible
// with the existing VenueState struct.

use crate::config::Config;
use crate::state::GlobalState;

/// Update per-venue toxicity scores.
///
/// Assumes:
///   - `state.sigma_eff` has been updated by the engine,
///   - each venue has `local_vol_short` populated.
pub fn update_toxicity_and_health(state: &mut GlobalState, cfg: &Config) {
    let tox_cfg = &cfg.toxicity;
    let vol_cfg = &cfg.volatility;

    // Keep sigma_eff away from zero so ratios are well-defined.
    let sigma_eff: f64 = state.sigma_eff.max(vol_cfg.sigma_min);

    for venue in &mut state.venues {
        // Base toxicity: if we have no mid or no depth, treat as highly toxic.
        let mut tox: f64 = if venue.mid.is_some() && venue.depth_near_mid > 0.0_f64 {
            0.0_f64
        } else {
            1.0_f64
        };

        // Volatility-based feature.
        if sigma_eff > 0.0_f64 && tox_cfg.vol_tox_scale > 0.0_f64 {
            let local: f64 = venue.local_vol_short.max(vol_cfg.sigma_min);
            let ratio: f64 = local / sigma_eff;

            let raw: f64 = (ratio - 1.0_f64) / tox_cfg.vol_tox_scale;
            // Clamp to [0, 1] using the intrinsic clamp() to satisfy clippy.
            let f_vol: f64 = raw.clamp(0.0_f64, 1.0_f64);

            tox = tox.max(f_vol);
        }

        // Store final score in [0, 1] (again via clamp() for clippy).
        venue.toxicity = tox.clamp(0.0_f64, 1.0_f64);
    }
}
