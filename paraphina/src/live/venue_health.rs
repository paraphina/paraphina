//! Venue health manager for live trading (feature-gated).

use crate::config::Config;
use crate::state::GlobalState;
use crate::telemetry::VenueHealthDiagnostics;
use crate::types::VenueStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VenueHealthErrorSource {
    MarketCache,
    AccountCache,
}

impl VenueHealthErrorSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::MarketCache => "market_cache",
            Self::AccountCache => "account_cache",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VenueHealthDisableReason {
    ApiError,
    DevBreach,
    Stale,
}

impl VenueHealthDisableReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::ApiError => "api_error",
            Self::DevBreach => "dev_breach",
            Self::Stale => "stale",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VenueHealthTransition {
    pub venue_index: usize,
    pub from: VenueStatus,
    pub to: VenueStatus,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VenueHealthUpdate {
    pub disabled: Vec<usize>,
    pub transitions: Vec<VenueHealthTransition>,
}

pub struct VenueHealthManager {
    dev_breaches: Vec<u32>,
    api_errors: Vec<u32>,
    stale_counts: Vec<u32>,
    last_error_source: Vec<Option<VenueHealthErrorSource>>,
    last_error_message: Vec<String>,
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
            last_error_source: vec![None; count],
            last_error_message: vec![String::new(); count],
            dev_breach_limit: 3,
            api_error_limit: 3,
            stale_limit: 2,
        }
    }

    pub fn record_api_error(
        &mut self,
        venue_index: usize,
        source: VenueHealthErrorSource,
        message: &str,
    ) {
        if let Some(val) = self.api_errors.get_mut(venue_index) {
            *val = val.saturating_add(1);
        }
        if let Some(last_source) = self.last_error_source.get_mut(venue_index) {
            *last_source = Some(source);
        }
        if let Some(last_message) = self.last_error_message.get_mut(venue_index) {
            *last_message = truncate_error_message(message);
        }
    }

    pub fn diagnostics(&self) -> Vec<VenueHealthDiagnostics> {
        (0..self.api_errors.len())
            .map(|idx| {
                let disable_reason = self.disable_reason_for(idx).unwrap_or("");
                let last_error_source = self
                    .last_error_source
                    .get(idx)
                    .and_then(|source| source.map(VenueHealthErrorSource::as_str))
                    .unwrap_or("");
                VenueHealthDiagnostics {
                    api_errors: *self.api_errors.get(idx).unwrap_or(&0),
                    stale_count: *self.stale_counts.get(idx).unwrap_or(&0),
                    dev_breaches: *self.dev_breaches.get(idx).unwrap_or(&0),
                    disable_reason: disable_reason.to_string(),
                    last_error_source: last_error_source.to_string(),
                    last_error_message: self
                        .last_error_message
                        .get(idx)
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect()
    }

    pub fn update_from_snapshot(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        snapshot: &crate::live::state_cache::CanonicalCacheSnapshot,
    ) -> VenueHealthUpdate {
        let mut update = VenueHealthUpdate::default();
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
                if let Some(last_source) = self.last_error_source.get_mut(idx) {
                    *last_source = None;
                }
                if let Some(last_message) = self.last_error_message.get_mut(idx) {
                    last_message.clear();
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

            let new_status = if dev_breached || api_breached {
                VenueStatus::Disabled
            } else if stale_breached {
                VenueStatus::Warning
            } else {
                VenueStatus::Healthy
            };
            let prev_status = vstate.status;
            if new_status != prev_status {
                update.transitions.push(VenueHealthTransition {
                    venue_index: idx,
                    from: prev_status,
                    to: new_status,
                });
            }
            if matches!(new_status, VenueStatus::Disabled)
                && !matches!(prev_status, VenueStatus::Disabled)
            {
                update.disabled.push(idx);
            }
            vstate.status = new_status;
        }

        update
    }

    pub fn describe_disable_reason(&self, venue_index: usize) -> &'static str {
        self.disable_reason_for(venue_index).unwrap_or("")
    }

    fn disable_reason_for(&self, venue_index: usize) -> Option<&'static str> {
        let dev_breached = self
            .dev_breaches
            .get(venue_index)
            .is_some_and(|v| *v >= self.dev_breach_limit);
        let api_breached = self
            .api_errors
            .get(venue_index)
            .is_some_and(|v| *v >= self.api_error_limit);
        let stale_breached = self
            .stale_counts
            .get(venue_index)
            .is_some_and(|v| *v >= self.stale_limit);
        let reason = if dev_breached {
            Some(VenueHealthDisableReason::DevBreach)
        } else if api_breached {
            Some(VenueHealthDisableReason::ApiError)
        } else if stale_breached {
            Some(VenueHealthDisableReason::Stale)
        } else {
            None
        };
        reason.map(VenueHealthDisableReason::as_str)
    }
}

fn truncate_error_message(message: &str) -> String {
    const MAX_CHARS: usize = 120;
    message.chars().take(MAX_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::{VenueHealthErrorSource, VenueHealthManager, VenueHealthTransition};
    use crate::config::Config;
    use crate::live::state_cache::{
        CanonicalCacheSnapshot, VenueAccountSnapshot, VenueMarketSnapshot,
    };
    use crate::state::GlobalState;
    use crate::types::VenueStatus;

    fn snapshot_for_status(cfg: &Config, mid: f64, is_stale: bool) -> CanonicalCacheSnapshot {
        CanonicalCacheSnapshot {
            timestamp_ms: 1_000,
            market: vec![VenueMarketSnapshot {
                venue_index: 0,
                venue_id: cfg.venues[0].id_arc.clone(),
                seq: 1,
                timestamp_ms: Some(1_000),
                mid: Some(mid),
                spread: Some(1.0),
                depth_near_mid: 1_000.0,
                is_stale,
            }],
            account: vec![VenueAccountSnapshot {
                venue_index: 0,
                venue_id: cfg.venues[0].id_arc.clone(),
                seq: 1,
                timestamp_ms: Some(1_000),
                position_tao: 0.0,
                avg_entry_price: 0.0,
                funding_8h: None,
                margin_balance_usd: 0.0,
                margin_used_usd: 0.0,
                margin_available_usd: 0.0,
                price_liq: None,
                dist_liq_sigma: None,
                is_stale,
            }],
        }
    }

    #[test]
    fn api_error_disable_records_diagnostics() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        let mut manager = VenueHealthManager::new(&cfg);
        for _ in 0..3 {
            manager.record_api_error(
                0,
                VenueHealthErrorSource::MarketCache,
                "SeqOutOfOrder on market cache",
            );
        }

        let update =
            manager.update_from_snapshot(&cfg, &mut state, &snapshot_for_status(&cfg, 100.0, true));
        assert_eq!(update.disabled, vec![0]);
        assert_eq!(
            update.transitions,
            vec![VenueHealthTransition {
                venue_index: 0,
                from: VenueStatus::Healthy,
                to: VenueStatus::Disabled,
            }]
        );
        let diagnostics = manager.diagnostics();
        assert_eq!(diagnostics[0].api_errors, 3);
        assert_eq!(diagnostics[0].disable_reason, "api_error");
        assert_eq!(diagnostics[0].last_error_source, "market_cache");
        assert!(diagnostics[0].last_error_message.contains("SeqOutOfOrder"));
    }

    #[test]
    fn fresh_snapshot_recovers_from_api_error_disable() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        let mut manager = VenueHealthManager::new(&cfg);
        for _ in 0..3 {
            manager.record_api_error(0, VenueHealthErrorSource::AccountCache, "non-monotonic seq");
        }

        let _ =
            manager.update_from_snapshot(&cfg, &mut state, &snapshot_for_status(&cfg, 100.0, true));
        let update = manager.update_from_snapshot(
            &cfg,
            &mut state,
            &snapshot_for_status(&cfg, 100.0, false),
        );
        assert_eq!(
            update.transitions,
            vec![VenueHealthTransition {
                venue_index: 0,
                from: VenueStatus::Disabled,
                to: VenueStatus::Healthy,
            }]
        );
        let diagnostics = manager.diagnostics();
        assert_eq!(diagnostics[0].api_errors, 0);
        assert!(diagnostics[0].disable_reason.is_empty());
        assert!(diagnostics[0].last_error_source.is_empty());
        assert!(diagnostics[0].last_error_message.is_empty());
        assert_eq!(state.venues[0].status, VenueStatus::Healthy);
    }
}
