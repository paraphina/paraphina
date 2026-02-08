//! Venue health manager for live trading (feature-gated).

use crate::config::Config;
use crate::state::GlobalState;
use crate::types::VenueStatus;

pub struct VenueHealthManager {
    dev_breaches: Vec<u32>,
    api_errors: Vec<u32>,
    stale_counts: Vec<u32>,
    dev_breach_limit: u32,
    api_error_limit: u32,
    stale_limit: u32,
}

impl VenueHealthManager {
    pub fn new(cfg: &Config) -> Self {
        let count = cfg.venues.len();
        Self {
            dev_breaches: vec![0; count],
            api_errors: vec![0; count],
            stale_counts: vec![0; count],
            dev_breach_limit: 3,
            api_error_limit: 3,
            stale_limit: 2,
        }
    }

    pub fn record_api_error(&mut self, venue_index: usize) {
        if let Some(val) = self.api_errors.get_mut(venue_index) {
            *val = val.saturating_add(1);
        }
    }

    pub fn update_from_snapshot(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        snapshot: &crate::live::state_cache::CanonicalCacheSnapshot,
    ) -> Vec<usize> {
        let mut disabled = Vec::new();
        let dev_limit = cfg.book.max_mid_jump_pct.abs().max(1e-9);
        let fair = state.fair_value.unwrap_or(state.fair_value_prev);

        for venue in &snapshot.market {
            let idx = venue.venue_index;
            let Some(vstate) = state.venues.get_mut(idx) else {
                continue;
            };
            if venue.is_stale {
                if let Some(val) = self.stale_counts.get_mut(idx) {
                    *val = val.saturating_add(1);
                }
            } else {
                // FIX D2: Reset stale_counts AND api_errors when venue is fresh.
                // Without the api_errors reset, once api_errors >= api_error_limit,
                // api_breached stays true forever, locking the venue in Disabled
                // with no recovery path.
                if let Some(val) = self.stale_counts.get_mut(idx) {
                    *val = 0;
                }
                if let Some(val) = self.api_errors.get_mut(idx) {
                    *val = 0;
                }
            }

            if let (Some(mid), true) = (venue.mid, fair.is_finite() && fair > 0.0) {
                let dev = ((mid - fair).abs() / fair).abs();
                if dev > dev_limit {
                    if let Some(val) = self.dev_breaches.get_mut(idx) {
                        *val = val.saturating_add(1);
                    }
                } else if let Some(val) = self.dev_breaches.get_mut(idx) {
                    *val = 0;
                }
            }

            let dev_breached = self
                .dev_breaches
                .get(idx)
                .is_some_and(|v| *v >= self.dev_breach_limit);
            let api_breached = self
                .api_errors
                .get(idx)
                .is_some_and(|v| *v >= self.api_error_limit);
            let stale_breached = self
                .stale_counts
                .get(idx)
                .is_some_and(|v| *v >= self.stale_limit);

            if dev_breached || api_breached {
                if !matches!(vstate.status, VenueStatus::Disabled) {
                    vstate.status = VenueStatus::Disabled;
                    disabled.push(idx);
                }
            } else if stale_breached {
                vstate.status = VenueStatus::Warning;
            } else {
                vstate.status = VenueStatus::Healthy;
            }
        }

        disabled
    }
}
