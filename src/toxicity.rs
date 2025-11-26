// src/toxicity.rs
//
// Venue toxicity + health scoring (Section 7 of the whitepaper).
// Right now we fully implement feature f1 (relative venue vol) and
// leave the flow/markout-based features as 0.0 until we wire real data.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::VenueStatus;

/// Update per-venue toxicity score in [0,1] and VenueStatus.
///
/// For now:
///   - f1: relative venue vol vs global sigma_eff
///   - f2..f5: stubs (0.0) until we add flows / markouts / imbalance
pub fn update_toxicity_and_health(cfg: &Config, state: &mut GlobalState) {
    let tox_cfg = &cfg.toxicity;

    // Use sigma_eff as the reference volatility.
    let sigma_eff = state.sigma_eff.max(1e-9);

    for v in &mut state.venues {
        // -------- f1: relative venue vol --------
        let ratio = if sigma_eff > 0.0 {
            v.local_vol_short / sigma_eff
        } else {
            0.0
        };

        // Map (ratio - 1) / VOL_TOX_SCALE into [0,1], clipped.
        let f1 = ((ratio - 1.0) / tox_cfg.vol_tox_scale)
            .max(0.0)
            .min(1.0);

        // -------- Placeholders for f2..f5 --------
        // Until we wire negative markouts, order-book imbalance and flow,
        // we leave these as 0.0.
        let f2 = 0.0; // neg markouts fraction
        let f3 = 0.0; // order book imbalance
        let f4 = 0.0; // directional aggressive flow
        let f5 = 0.0; // throughput

        // Weighted sum -> toxicity in [0,1].
        let tox = (tox_cfg.w1 * f1
            + tox_cfg.w2 * f2
            + tox_cfg.w3 * f3
            + tox_cfg.w4 * f4
            + tox_cfg.w5 * f5)
            .max(0.0)
            .min(1.0);

        v.toxicity = tox;

        // -------- VenueStatus classification --------
        v.status = if tox >= tox_cfg.tox_high_threshold {
            // High toxicity → disable venue (no quoting / hedging).
            VenueStatus::Disabled
        } else if tox >= tox_cfg.tox_med_threshold {
            // Medium toxicity → in a more complete implementation we might
            // mark Degraded and shrink quotes. For now, keep Healthy so that
            // quoting logic can still run, but you can see toxicity values
            // on the console.
            VenueStatus::Healthy
        } else {
            VenueStatus::Healthy
        };
    }
}
