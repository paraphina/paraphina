// src/toxicity.rs
//
// Per-venue toxicity + health classification.
// Approximation of the whitepaper logic:
//  - use local per-venue volatility vs global σ_eff as the primary signal,
//  - map that to a toxicity score in [0, 1],
//  - threshold into {Healthy, Warning, Disabled}.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::VenueStatus;

/// Update per-venue toxicity score and health status.
///
/// In production this would also use realised markouts and flow imbalance.
/// Here we approximate the whitepaper's f₁ feature:
///
///   ratio_v = local_vol_short_v / σ_eff
///   f₁     = clip((ratio_v - 1) / VOL_TOX_SCALE, 0, 1)
///
/// and treat f₁ itself as the toxicity score.
///
/// `VOL_TOX_SCALE` controls how far above global volatility a venue has to run
/// before it is considered suspect. With the default config:
///
///   - ratio_v ≲ 1.25  → tox < 0.6  → Healthy
///   - ratio_v ≈ 1.3   → tox ≈ 0.6  → Warning
///   - ratio_v ≳ 1.45  → tox ≳ 0.9  → Disabled
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
                // No usable mid: treat venue as disabled and maximally toxic.
                v.toxicity = tox_cfg.tox_high_threshold.max(1.0);
                v.status = VenueStatus::Disabled;
                continue;
            }
        };

        // Primary signal: local vol vs global σ_eff.
        let local_sigma = v.local_vol_short.max(1e-8);
        let ratio_v = local_sigma / sigma_eff;

        // Excess volatility above 1, scaled by vol_tox_scale and clipped to [0,1].
        let mut tox = ((ratio_v - 1.0) / tox_cfg.vol_tox_scale).max(0.0);
        if tox > 1.0 {
            tox = 1.0;
        }

        // Very large persistent price deviations vs fair value are also suspect.
        // This is a light additional guard so that wildly off-market books
        // are not treated as perfectly healthy.
        let rel_dev = ((mid - fair) / fair).abs();
        let dev_floor = 0.01; // 1% baseline deviation tolerance.
        let dev_thresh = 3.0 * sigma_eff.max(dev_floor);

        if rel_dev > dev_thresh {
            let overshoot = rel_dev - dev_thresh;
            let dev_scale = dev_thresh.max(1e-4);
            let extra = (overshoot / dev_scale).min(1.0); // up to +1
            tox = (tox + extra).min(1.0);
        }

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
