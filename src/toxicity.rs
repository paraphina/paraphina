// src/toxicity.rs
//
// Simple toxicity + venue health classification scaffolding.
//
// In the full whitepaper this would be driven by trade mark-outs and
// queue-position information. For now we:
//  - keep the per-venue toxicity scalar in [0,1],
//  - downgrade health based on staleness / low depth / high toxicity,
//  - use thresholds from Config.toxicity.

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{TimestampMs, VenueStatus};

pub fn update_toxicity_and_health(
    cfg: &Config,
    state: &mut GlobalState,
    now_ms: TimestampMs,
) {
    let tox_cfg = &cfg.toxicity;

    for v in &mut state.venues {
        let t = v.toxicity;

        // How stale is the last mid update?
        let stale_ms = v
            .last_mid_update_ms
            .map(|last| now_ms - last)
            .unwrap_or(i64::MAX);

        let mut status = VenueStatus::Healthy;

        // 1) Data quality / staleness.
        if stale_ms > tox_cfg.stale_ms_disabled {
            status = VenueStatus::Disabled;
        } else if stale_ms > tox_cfg.stale_ms_degraded {
            status = VenueStatus::Degraded;
        }

        // 2) Structural liquidity.
        if v.depth_near_mid < tox_cfg.min_depth_healthy {
            status = match status {
                VenueStatus::Healthy => VenueStatus::Degraded,
                other => other,
            };
        }

        // 3) Toxicity thresholds.
        if t > tox_cfg.max_toxicity_degraded {
            status = VenueStatus::Disabled;
        } else if t > tox_cfg.max_toxicity_healthy {
            status = match status {
                VenueStatus::Healthy => VenueStatus::Degraded,
                other => other,
            };
        }

        v.status = status;
    }
}
