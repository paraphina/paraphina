// src/toxicity.rs
//
// Per-venue toxicity + health classification.
// Approximation of the whitepaper logic:
//  - measure how "noisy" / off-market each venue looks vs fair value,
//  - map that to a toxicity score in [0, ~2],
//  - threshold into {Healthy, Warning, Disabled}.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::VenueStatus;

/// Update per-venue toxicity score and health status.
///
/// In production this would use per-venue realised volatility.
/// Here we approximate it from the deviation of each venue mid
/// from the current global fair value.
pub fn update_toxicity_and_health(state: &mut GlobalState, cfg: &Config) {
    let tox_cfg = &cfg.toxicity;

    let fair = match state.fair_value {
        Some(v) if v > 0.0 => v,
        _ => {
            // If we have no fair value yet, mark everything healthy.
            for v in &mut state.venues {
                v.toxicity = 0.0;
                v.status = VenueStatus::Healthy;
            }
            return;
        }
    };

    let sigma_eff = state.sigma_eff.max(1e-8);

    for v in &mut state.venues {
        let mid = match v.mid {
            Some(m) if m > 0.0 => m,
            _ => {
                // No usable mid: treat venue as toxic / disabled.
                v.toxicity = 1.0;
                v.status = VenueStatus::Disabled;
                continue;
            }
        };

        // Simple proxy for local volatility: instantaneous deviation vs fair.
        let rel_move = ((mid - fair) / fair).abs();

        let vol_ratio_local = (rel_move / sigma_eff).min(10.0);
        let tox = (vol_ratio_local * tox_cfg.vol_tox_scale).min(2.0);

        v.toxicity = tox;

        v.status = if tox >= tox_cfg.tox_high_threshold {
            VenueStatus::Disabled
        } else if tox >= tox_cfg.tox_med_threshold {
            VenueStatus::Warning
        } else {
            VenueStatus::Healthy
        };
    }
}
