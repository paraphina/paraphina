//! Canonical live state cache for market + account data.

use std::sync::Arc;

use crate::config::Config;
use crate::types::TimestampMs;

use super::orderbook_l2::{DepthConfig, OrderBookError, OrderBookL2};
use super::types::{
    AccountEvent, AccountSnapshot, L2Delta, L2Snapshot, LiquidationSnapshot, MarginSnapshot,
    MarketDataEvent,
};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq)]
pub struct CacheError {
    pub message: String,
}

impl CacheError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VenueMarketCache {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub seq: u64,
    pub last_update_ms: Option<TimestampMs>,
    pub orderbook: OrderBookL2,
    pub mid: Option<f64>,
    pub spread: Option<f64>,
    pub depth_near_mid: f64,
    first_book_update_logged: bool,
    first_nonzero_depth_logged: bool,
}

impl VenueMarketCache {
    pub fn new(venue_index: usize, venue_id: Arc<str>) -> Self {
        Self {
            venue_index,
            venue_id,
            seq: 0,
            last_update_ms: None,
            orderbook: OrderBookL2::new(),
            mid: None,
            spread: None,
            depth_near_mid: 0.0,
            first_book_update_logged: false,
            first_nonzero_depth_logged: false,
        }
    }

    pub fn is_stale(&self, now_ms: TimestampMs, max_age_ms: i64) -> bool {
        let Some(last) = self.last_update_ms else {
            return true;
        };
        now_ms.saturating_sub(last) > max_age_ms
    }

    pub fn apply_market_event(&mut self, event: &MarketDataEvent) -> Result<(), CacheError> {
        match event {
            MarketDataEvent::L2Snapshot(snapshot) => self.apply_l2_snapshot(snapshot),
            MarketDataEvent::L2Delta(delta) => self.apply_l2_delta(delta),
            MarketDataEvent::Trade(_) | MarketDataEvent::FundingUpdate(_) => {
                // Trades/funding are recorded in separate caches; no book impact here.
                Ok(())
            }
        }
    }

    fn apply_l2_snapshot(&mut self, snapshot: &L2Snapshot) -> Result<(), CacheError> {
        if snapshot.venue_index != self.venue_index {
            return Err(CacheError::new("venue_index mismatch for L2 snapshot"));
        }
        self.orderbook
            .apply_snapshot(&snapshot.bids, &snapshot.asks, snapshot.seq)
            .map_err(cache_err)?;
        self.seq = self.orderbook.last_seq();
        self.last_update_ms = Some(snapshot.timestamp_ms);
        self.recompute_derived();
        self.maybe_log_first_book_update(&snapshot.venue_id);
        Ok(())
    }

    fn apply_l2_delta(&mut self, delta: &L2Delta) -> Result<(), CacheError> {
        if delta.venue_index != self.venue_index {
            return Err(CacheError::new("venue_index mismatch for L2 delta"));
        }
        self.orderbook
            .apply_delta(&delta.changes, delta.seq)
            .map_err(cache_err)?;
        self.seq = self.orderbook.last_seq();
        self.last_update_ms = Some(delta.timestamp_ms);
        self.recompute_derived();
        self.maybe_log_first_book_update(&delta.venue_id);
        Ok(())
    }

    fn recompute_derived(&mut self) {
        let metrics = self.orderbook.compute_mid_spread_depth(DepthConfig {
            levels: 3,
            include_imbalance: false,
        });
        self.mid = metrics.mid;
        self.spread = metrics.spread;
        self.depth_near_mid = compute_top_notional_depth(&self.orderbook);
    }

    fn maybe_log_first_book_update(&mut self, symbol: &str) {
        if self.first_book_update_logged {
            return;
        }
        if self.depth_near_mid <= 0.0 {
            return;
        }
        let (Some(mid), Some(spread), Some(ts)) = (self.mid, self.spread, self.last_update_ms)
        else {
            return;
        };
        eprintln!(
            "FIRST_BOOK_UPDATE venue={} symbol={} mid={} spread={} ts={}",
            self.venue_id, symbol, mid, spread, ts
        );
        self.first_book_update_logged = true;
        if !self.first_nonzero_depth_logged {
            if let (Some(bid), Some(ask)) = (self.orderbook.best_bid(), self.orderbook.best_ask()) {
                eprintln!(
                    "FIRST_NOTIONAL_DEPTH venue={} depth_usd={} bid_px={} bid_sz={} ask_px={} ask_sz={}",
                    self.venue_id,
                    self.depth_near_mid,
                    bid.price,
                    bid.size,
                    ask.price,
                    ask.size
                );
                self.first_nonzero_depth_logged = true;
            }
        }
    }
}

fn compute_top_notional_depth(book: &OrderBookL2) -> f64 {
    let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask()) else {
        return 0.0;
    };
    let bid_notional = (bid.price * bid.size).abs();
    let ask_notional = (ask.price * ask.size).abs();
    if bid_notional.is_finite() && ask_notional.is_finite() {
        bid_notional + ask_notional
    } else {
        0.0
    }
}

fn cache_err(err: OrderBookError) -> CacheError {
    CacheError::new(format!("{err:?}"))
}

#[derive(Debug, Clone)]
pub struct VenueAccountCache {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub seq: u64,
    pub last_update_ms: Option<TimestampMs>,
    pub positions: Vec<super::types::PositionSnapshot>,
    pub balances: Vec<super::types::BalanceSnapshot>,
    pub position_tao: f64,
    pub avg_entry_price: f64,
    pub funding_8h: Option<f64>,
    pub margin: MarginSnapshot,
    pub liquidation: LiquidationSnapshot,
}

impl VenueAccountCache {
    pub fn new(venue_index: usize, venue_id: Arc<str>) -> Self {
        Self {
            venue_index,
            venue_id,
            seq: 0,
            last_update_ms: None,
            positions: Vec::new(),
            balances: Vec::new(),
            position_tao: 0.0,
            avg_entry_price: 0.0,
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd: 0.0,
                used_usd: 0.0,
                available_usd: 0.0,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        }
    }

    pub fn is_stale(&self, now_ms: TimestampMs, max_age_ms: i64) -> bool {
        let Some(last) = self.last_update_ms else {
            return true;
        };
        now_ms.saturating_sub(last) > max_age_ms
    }

    pub fn apply_account_event(&mut self, event: &AccountEvent) -> Result<(), CacheError> {
        match event {
            AccountEvent::Snapshot(snapshot) => self.apply_snapshot(snapshot),
        }
    }

    fn apply_snapshot(&mut self, snapshot: &AccountSnapshot) -> Result<(), CacheError> {
        if snapshot.venue_index != self.venue_index {
            return Err(CacheError::new("venue_index mismatch for account snapshot"));
        }
        if snapshot.seq < self.seq {
            return Err(CacheError::new("non-monotonic seq for account snapshot"));
        }
        self.seq = snapshot.seq;
        self.last_update_ms = Some(snapshot.timestamp_ms);
        self.positions = snapshot.positions.clone();
        self.balances = snapshot.balances.clone();
        self.funding_8h = snapshot.funding_8h;
        self.margin = snapshot.margin.clone();
        self.liquidation = snapshot.liquidation.clone();
        let (position_tao, avg_entry_price) = derive_position_metrics(&self.positions);
        self.position_tao = position_tao;
        self.avg_entry_price = avg_entry_price;
        Ok(())
    }
}

fn derive_position_metrics(positions: &[super::types::PositionSnapshot]) -> (f64, f64) {
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    let mut total = 0.0;
    let mut weighted = 0.0;
    for pos in positions {
        total += pos.size;
        weighted += pos.size.abs() * pos.entry_price;
    }
    let avg_entry = if total.abs() > 0.0 {
        weighted / total.abs()
    } else {
        0.0
    };
    (total, avg_entry)
}

#[derive(Debug, Clone)]
pub struct VenueMarketSnapshot {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub seq: u64,
    pub timestamp_ms: Option<TimestampMs>,
    pub mid: Option<f64>,
    pub spread: Option<f64>,
    pub depth_near_mid: f64,
    pub is_stale: bool,
}

#[derive(Debug, Clone)]
pub struct VenueAccountSnapshot {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub seq: u64,
    pub timestamp_ms: Option<TimestampMs>,
    pub position_tao: f64,
    pub avg_entry_price: f64,
    pub funding_8h: Option<f64>,
    pub margin_balance_usd: f64,
    pub margin_used_usd: f64,
    pub margin_available_usd: f64,
    pub price_liq: Option<f64>,
    pub dist_liq_sigma: Option<f64>,
    pub is_stale: bool,
}

#[derive(Debug, Clone)]
pub struct CanonicalCacheSnapshot {
    pub timestamp_ms: TimestampMs,
    pub market: Vec<VenueMarketSnapshot>,
    pub account: Vec<VenueAccountSnapshot>,
}

impl CanonicalCacheSnapshot {
    pub fn ready_market_count(&self) -> usize {
        self.market
            .iter()
            .filter(|m| !m.is_stale && m.mid.is_some())
            .count()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReconciliationReport {
    pub market_ok: bool,
    pub account_ok: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountFieldDiff {
    pub field: &'static str,
    pub prev: String,
    pub next: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccountReconcileDiff {
    pub venue_index: usize,
    pub venue_id: String,
    pub prev_seq: u64,
    pub next_seq: u64,
    pub changes: Vec<AccountFieldDiff>,
}

impl ReconciliationReport {
    pub fn new() -> Self {
        Self {
            market_ok: true,
            account_ok: true,
            issues: Vec::new(),
        }
    }

    pub fn record_issue(&mut self, issue: impl Into<String>) {
        self.issues.push(issue.into());
    }
}

#[derive(Debug)]
pub struct LiveStateCache {
    pub market: Vec<VenueMarketCache>,
    pub account: Vec<VenueAccountCache>,
}

impl LiveStateCache {
    pub fn new(cfg: &Config) -> Self {
        let market = cfg
            .venues
            .iter()
            .enumerate()
            .map(|(idx, vcfg)| VenueMarketCache::new(idx, vcfg.id_arc.clone()))
            .collect();
        let account = cfg
            .venues
            .iter()
            .enumerate()
            .map(|(idx, vcfg)| VenueAccountCache::new(idx, vcfg.id_arc.clone()))
            .collect();
        Self { market, account }
    }

    pub fn apply_market_event(&mut self, event: &MarketDataEvent) -> Result<(), CacheError> {
        let venue_index = match event {
            MarketDataEvent::L2Snapshot(s) => s.venue_index,
            MarketDataEvent::L2Delta(d) => d.venue_index,
            MarketDataEvent::Trade(t) => t.venue_index,
            MarketDataEvent::FundingUpdate(f) => f.venue_index,
        };
        if let Some(cache) = self.market.get_mut(venue_index) {
            cache.apply_market_event(event)
        } else {
            Err(CacheError::new("market event for unknown venue_index"))
        }
    }

    pub fn apply_account_event(&mut self, event: &AccountEvent) -> Result<(), CacheError> {
        let venue_index = match event {
            AccountEvent::Snapshot(s) => s.venue_index,
        };
        if let Some(cache) = self.account.get_mut(venue_index) {
            cache.apply_account_event(event)
        } else {
            Err(CacheError::new("account event for unknown venue_index"))
        }
    }

    pub fn snapshot(&self, now_ms: TimestampMs, max_age_ms: i64) -> CanonicalCacheSnapshot {
        let market = self
            .market
            .iter()
            .map(|m| VenueMarketSnapshot {
                venue_index: m.venue_index,
                venue_id: m.venue_id.clone(),
                seq: m.seq,
                timestamp_ms: m.last_update_ms,
                mid: m.mid,
                spread: m.spread,
                depth_near_mid: m.depth_near_mid,
                is_stale: m.is_stale(now_ms, max_age_ms),
            })
            .collect();
        let account = self
            .account
            .iter()
            .map(|a| VenueAccountSnapshot {
                venue_index: a.venue_index,
                venue_id: a.venue_id.clone(),
                seq: a.seq,
                timestamp_ms: a.last_update_ms,
                position_tao: a.position_tao,
                avg_entry_price: a.avg_entry_price,
                funding_8h: a.funding_8h,
                margin_balance_usd: a.margin.balance_usd,
                margin_used_usd: a.margin.used_usd,
                margin_available_usd: a.margin.available_usd,
                price_liq: a.liquidation.price_liq,
                dist_liq_sigma: a.liquidation.dist_liq_sigma,
                is_stale: a.is_stale(now_ms, max_age_ms),
            })
            .collect();
        CanonicalCacheSnapshot {
            timestamp_ms: now_ms,
            market,
            account,
        }
    }

    pub fn reconcile_market_snapshot(&self, snapshot: &L2Snapshot) -> ReconciliationReport {
        let mut report = ReconciliationReport::new();
        let Some(cache) = self.market.get(snapshot.venue_index) else {
            report.market_ok = false;
            report.record_issue("market snapshot refers to unknown venue_index");
            return report;
        };
        if snapshot.seq < cache.seq {
            report.market_ok = false;
            report.record_issue("market snapshot seq is behind cache seq");
        }
        if snapshot.venue_id != cache.venue_id.as_ref() {
            report.market_ok = false;
            report.record_issue("market snapshot venue_id mismatch");
        }
        report
    }

    pub fn reconcile_account_snapshot(&self, snapshot: &AccountSnapshot) -> ReconciliationReport {
        let mut report = ReconciliationReport::new();
        let Some(cache) = self.account.get(snapshot.venue_index) else {
            report.account_ok = false;
            report.record_issue("account snapshot refers to unknown venue_index");
            return report;
        };
        if snapshot.seq < cache.seq {
            report.account_ok = false;
            report.record_issue("account snapshot seq is behind cache seq");
        }
        if snapshot.venue_id != cache.venue_id.as_ref() {
            report.account_ok = false;
            report.record_issue("account snapshot venue_id mismatch");
        }
        report
    }

    pub fn reconcile_account_snapshot_with_diff(
        &mut self,
        snapshot: &AccountSnapshot,
    ) -> (ReconciliationReport, Option<AccountReconcileDiff>) {
        let report = self.reconcile_account_snapshot(snapshot);
        let Some(cache) = self.account.get(snapshot.venue_index) else {
            return (report, None);
        };

        let (next_pos, next_avg) = derive_position_metrics(&snapshot.positions);
        let mut changes = Vec::new();
        collect_diff("position_tao", cache.position_tao, next_pos, &mut changes);
        collect_diff(
            "avg_entry_price",
            cache.avg_entry_price,
            next_avg,
            &mut changes,
        );
        collect_opt_diff(
            "funding_8h",
            cache.funding_8h,
            snapshot.funding_8h,
            &mut changes,
        );
        collect_diff(
            "margin_balance_usd",
            cache.margin.balance_usd,
            snapshot.margin.balance_usd,
            &mut changes,
        );
        collect_diff(
            "margin_used_usd",
            cache.margin.used_usd,
            snapshot.margin.used_usd,
            &mut changes,
        );
        collect_diff(
            "margin_available_usd",
            cache.margin.available_usd,
            snapshot.margin.available_usd,
            &mut changes,
        );
        collect_opt_diff(
            "price_liq",
            cache.liquidation.price_liq,
            snapshot.liquidation.price_liq,
            &mut changes,
        );

        let diff = if changes.is_empty() {
            None
        } else {
            Some(AccountReconcileDiff {
                venue_index: snapshot.venue_index,
                venue_id: snapshot.venue_id.clone(),
                prev_seq: cache.seq,
                next_seq: snapshot.seq,
                changes,
            })
        };

        if report.account_ok {
            let _ = self.apply_account_event(&AccountEvent::Snapshot(snapshot.clone()));
        }

        (report, diff)
    }
}

fn collect_diff(field: &'static str, prev: f64, next: f64, out: &mut Vec<AccountFieldDiff>) {
    if (prev - next).abs() > 1e-9 {
        out.push(AccountFieldDiff {
            field,
            prev: format!("{prev:.10}"),
            next: format!("{next:.10}"),
        });
    }
}

fn collect_opt_diff(
    field: &'static str,
    prev: Option<f64>,
    next: Option<f64>,
    out: &mut Vec<AccountFieldDiff>,
) {
    if prev != next {
        out.push(AccountFieldDiff {
            field,
            prev: prev
                .map(|v| format!("{v:.10}"))
                .unwrap_or_else(|| "None".to_string()),
            next: next
                .map(|v| format!("{v:.10}"))
                .unwrap_or_else(|| "None".to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::live::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};

    #[test]
    fn market_cache_staleness_and_seq() {
        let cfg = Config::default();
        let mut cache = LiveStateCache::new(&cfg);
        let snapshot = L2Snapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.clone(),
            seq: 10,
            timestamp_ms: 1_000,
            bids: vec![BookLevel {
                price: 99.0,
                size: 1.0,
            }],
            asks: vec![BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        cache
            .apply_market_event(&MarketDataEvent::L2Snapshot(snapshot.clone()))
            .expect("apply snapshot");

        let snap = cache.snapshot(1_500, 1_000);
        assert_eq!(snap.market[0].seq, 10);
        assert!(!snap.market[0].is_stale);
        assert!(snap.market[0].mid.unwrap() > 0.0);

        let stale = cache.snapshot(3_500, 1_000);
        assert!(stale.market[0].is_stale);

        let stale_delta = L2Delta {
            venue_index: 0,
            venue_id: cfg.venues[0].id.clone(),
            seq: 10,
            timestamp_ms: 2_000,
            changes: vec![BookLevelDelta {
                side: BookSide::Bid,
                price: 98.0,
                size: 1.0,
            }],
        };
        let err = cache
            .apply_market_event(&MarketDataEvent::L2Delta(stale_delta))
            .unwrap_err();
        assert!(err.message.contains("SeqOutOfOrder"));
    }

    #[test]
    fn account_cache_seq_checks() {
        let cfg = Config::default();
        let mut cache = LiveStateCache::new(&cfg);
        let snapshot = AccountSnapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.clone(),
            seq: 5,
            timestamp_ms: 1_000,
            positions: Vec::new(),
            balances: Vec::new(),
            funding_8h: Some(0.001),
            margin: MarginSnapshot {
                balance_usd: 10_000.0,
                used_usd: 0.0,
                available_usd: 10_000.0,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: Some(10.0),
            },
        };
        cache
            .apply_account_event(&AccountEvent::Snapshot(snapshot.clone()))
            .expect("apply snapshot");

        let stale = AccountSnapshot { seq: 4, ..snapshot };
        let err = cache
            .apply_account_event(&AccountEvent::Snapshot(stale))
            .unwrap_err();
        assert!(err.message.contains("non-monotonic seq"));
    }
}
