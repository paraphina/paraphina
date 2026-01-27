// src/state.rs
//
// Global engine state + per-venue state for Paraphina.
//
// This file is intentionally "spec-shaped":
//  - Per-venue: book-derived mid/spread/depth + local vols + toxicity/health,
//    position + funding + (synthetic) margin/liquidation.
//  - Global: fair value + vol scalars, inventories + basis exposures, PnL,
//    risk regime + kill switch.
//
// Notes:
//  - We keep the current 3-state regime enum {Normal, Warning, HardLimit}.
//    In the whitepaper this maps to {Normal, Warning, Critical}, but in this
//    codebase HardLimit is the "circuit-breaker" state.
//  - Kill-switch is a separate boolean and is intended to be *latched*
//    by the Engine (once set, it stays set until manual reset).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::orderbook_l2::{
    BookLevel, BookLevelDelta, DepthConfig, DerivedBookMetrics, OrderBookError, OrderBookL2,
};
use crate::types::{
    FillEvent, OrderIntent, OrderPurpose, PlaceOrderIntent, Side, TimeInForce, TimestampMs,
    VenueStatus,
};

/// A pending markout evaluation entry.
///
/// Created at fill time; evaluated at `t_eval_ms` to compute realized markout.
#[derive(Debug, Clone)]
pub struct PendingMarkout {
    /// Timestamp when the fill occurred.
    pub t_fill_ms: TimestampMs,
    /// Timestamp when we should evaluate the markout (t_fill_ms + horizon).
    pub t_eval_ms: TimestampMs,
    /// Side of the fill (Buy or Sell).
    pub side: Side,
    /// Size of the fill in TAO.
    pub size_tao: f64,
    /// Fill price in USD/TAO.
    pub price: f64,
    /// Fair value at fill time (for optional analysis).
    pub fair_at_fill: f64,
    /// Venue mid price at fill time (used for markout calculation).
    pub mid_at_fill: f64,
    /// Deterministic fill sequence for linking to recent_fills.
    pub fill_seq: u64,
}

/// Canonical open-order record for per-venue order ledger.
#[derive(Debug, Clone)]
pub struct OpenOrderRecord {
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub remaining: f64,
    pub timestamp_ms: TimestampMs,
    pub purpose: OrderPurpose,
    pub time_in_force: Option<TimeInForce>,
    pub post_only: Option<bool>,
    pub reduce_only: Option<bool>,
}

/// Canonical fill record for recent-fills ring buffer.
#[derive(Debug, Clone)]
pub struct FillRecord {
    pub fill_seq: u64,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub fill_time_ms: TimestampMs,
    pub purpose: OrderPurpose,
    pub fee_bps: f64,
    pub markout_pnl_short: Option<f64>,
    pub pre_position_tao: Option<f64>,
    pub post_position_tao: Option<f64>,
    pub pre_q_global_tao: Option<f64>,
    pub post_q_global_tao: Option<f64>,
    pub realised_pnl_usd: Option<f64>,
}

/// Per-venue state (one per perp venue / subaccount).
#[derive(Debug, Clone)]
pub struct VenueState {
    /// Stable identifier matching Config.venues[i].id.
    /// Uses Arc<str> for cheap cloning in hot paths.
    pub id: Arc<str>,

    // ----- Order book & local vols -----
    /// Current mid price from order book (if any).
    pub mid: Option<f64>,
    /// Current best spread from order book (if any).
    pub spread: Option<f64>,
    /// Depth near mid (sum within top-K levels).
    pub depth_near_mid: f64,
    /// Last time (ms) we updated the mid / spread.
    pub last_mid_update_ms: Option<TimestampMs>,
    /// Last time (ms) we applied a book snapshot or delta.
    pub last_book_update_ms: Option<TimestampMs>,
    /// Core L2 order book (bounded).
    pub orderbook_l2: OrderBookL2,
    /// Short-horizon EWMA local vol of log mid.
    pub local_vol_short: f64,
    /// Long-horizon EWMA local vol of log mid.
    pub local_vol_long: f64,
    /// Cached ln(mid) from the previous tick for log-return computation.
    /// Stored to avoid repeated log() calls in the hot path.
    pub prev_ln_mid: Option<f64>,

    // ----- Health / toxicity -----
    pub status: VenueStatus,
    /// Toxicity score ∈ [0,1] (EWMA of markout-based instantaneous toxicity).
    pub toxicity: f64,
    /// Pending markout evaluations (ring buffer, bounded by config).
    pub pending_markouts: VecDeque<PendingMarkout>,
    /// Opt17: Cached t_eval_ms of pending_markouts.front() to avoid VecDeque access.
    /// Invariant: i64::MAX if pending_markouts is empty, else front().unwrap().t_eval_ms.
    pub pending_markouts_next_eval_ms: TimestampMs,
    /// Running EWMA of markout in USD/TAO for telemetry/debugging.
    pub markout_ewma_usd_per_tao: f64,

    // ----- Order ledger (whitepaper §4.3) -----
    /// Open orders keyed by order_id (BTreeMap for deterministic iteration).
    pub open_orders: BTreeMap<String, OpenOrderRecord>,
    /// Recent fills ring buffer (deterministic FIFO).
    pub recent_fills: VecDeque<FillRecord>,
    /// Maximum number of recent fills retained.
    pub recent_fills_cap: usize,
    /// Monotonic fill sequence counter (per venue).
    pub next_fill_seq: u64,

    // ----- Position & funding -----
    /// Net TAO-equivalent perp position.
    pub position_tao: f64,
    /// Current 8h funding rate (dimensionless).
    pub funding_8h: f64,
    /// Volume-weighted average entry price of the current position (USD per TAO).
    /// Defined only when `position_tao != 0.0`, otherwise 0.0.
    pub avg_entry_price: f64,

    // ----- Margin & liquidation (synthetic for now) -----
    /// Total margin balance in USD (if known).
    pub margin_balance_usd: f64,
    /// Margin currently used in USD.
    pub margin_used_usd: f64,
    /// Margin available for new positions in USD.
    pub margin_available_usd: f64,
    /// Estimated liquidation price (if known).
    pub price_liq: Option<f64>,
    /// Distance to liquidation in sigma units (approx).
    pub dist_liq_sigma: f64,

    // ----- MM order management (Milestone H §11) -----
    /// Current active MM bid order (if any).
    pub mm_open_bid: Option<MmOpenOrder>,
    /// Current active MM ask order (if any).
    pub mm_open_ask: Option<MmOpenOrder>,
}

#[derive(Debug, Clone)]
pub struct SimAccountSnapshot {
    pub venue_index: usize,
    pub timestamp_ms: TimestampMs,
    pub position_tao: f64,
    pub avg_entry_price: f64,
    pub funding_8h: f64,
    pub margin_balance_usd: f64,
    pub margin_used_usd: f64,
    pub margin_available_usd: f64,
    pub price_liq: Option<f64>,
    pub dist_liq_sigma: f64,
}

impl VenueState {
    /// Update local vol EWMAs from a new mid price using the canonical formula.
    pub fn update_local_vol_from_mid(&mut self, mid: f64, alpha_short: f64, alpha_long: f64) {
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        // Cache ln(mid) for log-return computation; guard mid for safety.
        let ln_mid = mid.max(1e-6).ln();

        // Update local vols only when we have a previous log mid.
        if let Some(prev_ln_mid) = self.prev_ln_mid {
            let r = ln_mid - prev_ln_mid;
            let r2 = r * r;

            let var_short_prev = self.local_vol_short * self.local_vol_short;
            let var_long_prev = self.local_vol_long * self.local_vol_long;

            let var_short_new = one_minus_alpha_short * var_short_prev + alpha_short * r2;
            let var_long_new = one_minus_alpha_long * var_long_prev + alpha_long * r2;

            self.local_vol_short = var_short_new.max(0.0).sqrt();
            self.local_vol_long = var_long_new.max(0.0).sqrt();
        }

        // Store current ln(mid) for next tick.
        self.prev_ln_mid = Some(ln_mid);
    }

    pub fn apply_l2_snapshot(
        &mut self,
        bids: &[BookLevel],
        asks: &[BookLevel],
        seq: u64,
        timestamp_ms: TimestampMs,
        max_levels: usize,
        alpha_short: f64,
        alpha_long: f64,
    ) -> Result<DerivedBookMetrics, OrderBookError> {
        self.orderbook_l2.apply_snapshot(bids, asks, seq)?;
        self.orderbook_l2.trim_levels(max_levels);
        let metrics = self.orderbook_l2.compute_mid_spread_depth(DepthConfig {
            levels: max_levels.max(1),
            include_imbalance: false,
        });
        self.last_book_update_ms = Some(timestamp_ms);
        self.depth_near_mid = top_of_book_notional(&self.orderbook_l2);
        if let (Some(mid), Some(spread)) = (metrics.mid, metrics.spread) {
            self.mid = Some(mid);
            self.spread = Some(spread);
            self.last_mid_update_ms = Some(timestamp_ms);
            self.update_local_vol_from_mid(mid, alpha_short, alpha_long);
        }
        Ok(metrics)
    }

    pub fn apply_l2_delta(
        &mut self,
        deltas: &[BookLevelDelta],
        seq: u64,
        timestamp_ms: TimestampMs,
        max_levels: usize,
        alpha_short: f64,
        alpha_long: f64,
    ) -> Result<DerivedBookMetrics, OrderBookError> {
        self.orderbook_l2.apply_delta(deltas, seq)?;
        self.orderbook_l2.trim_levels(max_levels);
        let metrics = self.orderbook_l2.compute_mid_spread_depth(DepthConfig {
            levels: max_levels.max(1),
            include_imbalance: false,
        });
        self.last_book_update_ms = Some(timestamp_ms);
        self.depth_near_mid = top_of_book_notional(&self.orderbook_l2);
        if let (Some(mid), Some(spread)) = (metrics.mid, metrics.spread) {
            self.mid = Some(mid);
            self.spread = Some(spread);
            self.last_mid_update_ms = Some(timestamp_ms);
            self.update_local_vol_from_mid(mid, alpha_short, alpha_long);
        }
        Ok(metrics)
    }
}

/// Minimal MM order tracking for order management (one per side per venue).
#[derive(Debug, Clone)]
pub struct MmOpenOrder {
    pub price: f64,
    pub size: f64,
    pub timestamp_ms: TimestampMs,
    pub order_id: String,
}

impl VenueState {
    pub(crate) fn upsert_open_order(&mut self, record: OpenOrderRecord) {
        self.open_orders.insert(record.order_id.clone(), record);
    }

    pub(crate) fn remove_open_order(&mut self, order_id: &str) {
        self.open_orders.remove(order_id);
        if self
            .mm_open_bid
            .as_ref()
            .is_some_and(|o| o.order_id == order_id)
        {
            self.mm_open_bid = None;
        }
        if self
            .mm_open_ask
            .as_ref()
            .is_some_and(|o| o.order_id == order_id)
        {
            self.mm_open_ask = None;
        }
    }

    pub(crate) fn clear_open_orders(&mut self) {
        self.open_orders.clear();
        self.mm_open_bid = None;
        self.mm_open_ask = None;
    }

    pub(crate) fn apply_markout_to_fill(&mut self, fill_seq: u64, markout: f64) {
        for fill in self.recent_fills.iter_mut().rev() {
            if fill.fill_seq == fill_seq {
                if fill.markout_pnl_short.is_none() {
                    fill.markout_pnl_short = Some(markout);
                }
                break;
            }
        }
    }
}

fn top_of_book_notional(book: &OrderBookL2) -> f64 {
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

/// High-level risk regime (whitepaper Section 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRegime {
    Normal,
    Warning,
    /// Hard limit / circuit-breaker regime (whitepaper "Critical").
    /// When this regime is active, kill_switch MUST be latched true.
    HardLimit,
}

/// Reason code for kill switch activation.
///
/// The kill switch can be triggered by multiple conditions. This enum
/// captures the primary reason for activation. Once latched, the reason
/// is preserved until manual reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillReason {
    /// No kill triggered (default state).
    None,
    /// Daily PnL loss limit breached.
    PnlHardBreach,
    /// Volatility-scaled delta limit breached.
    DeltaHardBreach,
    /// Basis exposure hard limit breached.
    BasisHardBreach,
    /// Liquidation distance too close (below critical sigma threshold).
    LiquidationDistanceBreach,
    /// Reconciliation drift detected between internal and venue state.
    ReconciliationDrift,
    /// Canary limit breach (position/open orders).
    CanaryLimitBreach,
    /// Stale market data in canary mode.
    StaleMarket,
}

/// Audit-grade kill event snapshot emitted on kill transition.
#[derive(Debug, Clone, Serialize)]
pub struct KillEvent {
    /// Tick/timebase identifier (simulation tick or live timebase id).
    pub timebase_id: u64,
    /// Explicit kill reason (string).
    pub kill_reason: String,
    /// Dollar delta at kill (ΔUSD_t).
    pub delta_usd_t: f64,
    /// Basis exposure at kill (B_t).
    pub basis_b_t: f64,
    /// Daily total PnL at kill.
    pub daily_pnl_total: f64,
    /// Per-venue positions (ordered by venue index).
    pub per_venue_q: Vec<f64>,
    /// Per-venue liquidation distance in sigma units (ordered by venue index).
    pub per_venue_dist_liq_sigma: Vec<f64>,
    /// Per-venue status (ordered by venue index).
    pub per_venue_status: Vec<String>,
    /// Current risk regime (string).
    pub risk_regime: String,
    /// Config version identifier.
    pub config_version_id: String,
}

/// Global engine state shared across strategy components.
#[derive(Debug, Clone)]
pub struct GlobalState {
    // ----- Per-venue -----
    pub venues: Vec<VenueState>,

    // ----- Scratch buffers for hot-path reuse -----
    // These buffers are used internally by Engine methods to avoid per-tick
    // heap allocations. They are cleared and reused each tick. The contents
    // are transient and not part of the logical state.
    //
    // Safety: These are only written to by Engine methods during a single
    // tick and cleared before use. They do not affect determinism since
    // they are always cleared/rebuilt from scratch each tick.
    /// Scratch buffer for median mid computation (used by median_mid_from_books).
    pub(crate) scratch_mids: Vec<f64>,
    /// Scratch buffer for Kalman filter observations (used by collect_kf_observations).
    pub(crate) scratch_kf_obs: Vec<(usize, f64, f64)>,

    // ----- Fair value / Kalman filter backbone -----
    /// Last time the fair-value filter was updated.
    pub kf_last_update_ms: TimestampMs,
    /// 1D KF variance term P.
    pub kf_p: f64,
    /// Filtered log-price x_t = log(S_t).
    pub kf_x_hat: f64,
    /// Previous fair value S_{t-1}.
    pub fair_value_prev: f64,
    /// Current fair value S_t (if we have one).
    pub fair_value: Option<f64>,
    /// Short-horizon fair value vol (σ_short).
    pub fv_short_vol: f64,
    /// Long-horizon fair value vol (σ_long).
    pub fv_long_vol: f64,
    /// Effective volatility σ_eff = max(σ_short, σ_min).
    pub sigma_eff: f64,

    // ----- Milestone D: FV gating telemetry -----
    /// Whether fair value was successfully updated this tick.
    ///
    /// True if >= min_healthy_for_kf observations passed gating and the
    /// measurement update was applied. False if we only did a time update
    /// (prediction step) due to insufficient healthy venue data.
    pub fv_available: bool,
    /// Indices of venues whose observations were used in the last KF update.
    ///
    /// Empty if fv_available is false (time update only).
    pub healthy_venues_used: Vec<usize>,
    /// Count of healthy venues used in the last KF update (convenience field).
    pub healthy_venues_used_count: usize,

    // ----- Volatility-driven control scalars (whitepaper Section 6) -----
    /// Vol ratio clipped into [vol_ratio_min, vol_ratio_max].
    pub vol_ratio_clipped: f64,
    pub spread_mult: f64,
    pub size_mult: f64,
    pub band_mult: f64,

    // ----- Inventory & basis (whitepaper Section 8) -----
    /// Global net trading inventory q_t in TAO.
    pub q_global_tao: f64,
    /// Dollar delta q_t * S_t.
    pub dollar_delta_usd: f64,
    /// Net basis exposure B_t in USD (signed): sum_v q_v * (m_v - S_t).
    pub basis_usd: f64,
    /// Gross basis exposure in USD: sum_v |q_v| * |m_v - S_t|.
    pub basis_gross_usd: f64,

    // ----- PnL -----
    pub daily_realised_pnl: f64,
    pub daily_unrealised_pnl: f64,
    pub daily_pnl_total: f64,
    /// Optional PnL anchor fair value (kept for spec / future).
    pub pnl_ref_fair_value: Option<f64>,

    // ----- Risk regime & limits -----
    pub risk_regime: RiskRegime,
    /// Kill switch: once true, stays true until manual reset (latching).
    /// When kill_switch is true, risk_regime MUST be HardLimit.
    pub kill_switch: bool,
    /// Primary reason for kill switch activation (if any).
    /// Preserved once set until manual reset.
    pub kill_reason: KillReason,
    /// Tick at which kill switch transition was handled (one-shot).
    pub kill_handled_at_tick: Option<u64>,
    /// Volatility-scaled delta limit (updated in engine).
    pub delta_limit_usd: f64,
    /// Basis warning limit in USD (soft).
    pub basis_limit_warn_usd: f64,
    /// Basis hard limit in USD.
    pub basis_limit_hard_usd: f64,

    // ----- Live order lifecycle tracking (feature-gated) -----
    #[cfg(feature = "live")]
    pub live_order_state: crate::live::order_state::LiveOrderState,
}

// =============================================================================
// Scratch buffer accessor methods for testing and inspection
// =============================================================================

impl GlobalState {
    /// Returns the current capacity of the scratch_mids buffer.
    ///
    /// This is intended for testing buffer reuse behavior - verifying that
    /// capacity is preserved across tick cycles without reallocation.
    pub fn scratch_mids_capacity(&self) -> usize {
        self.scratch_mids.capacity()
    }

    /// Returns the current length of the scratch_mids buffer.
    ///
    /// After a tick cycle, this reflects how many mids were collected
    /// for median computation.
    pub fn scratch_mids_len(&self) -> usize {
        self.scratch_mids.len()
    }

    /// Returns the current capacity of the scratch_kf_obs buffer.
    ///
    /// This is intended for testing buffer reuse behavior - verifying that
    /// capacity is preserved across tick cycles without reallocation.
    pub fn scratch_kf_obs_capacity(&self) -> usize {
        self.scratch_kf_obs.capacity()
    }

    /// Returns the current length of the scratch_kf_obs buffer.
    ///
    /// After a tick cycle, this reflects how many KF observations were
    /// collected from venues.
    pub fn scratch_kf_obs_len(&self) -> usize {
        self.scratch_kf_obs.len()
    }
}

impl GlobalState {
    /// Returns true exactly once when kill transition should be handled.
    pub fn mark_kill_handled(&mut self, tick: u64) -> bool {
        if !self.kill_switch {
            return false;
        }
        if self.kill_handled_at_tick.is_some() {
            return false;
        }
        self.kill_handled_at_tick = Some(tick);
        true
    }

    /// Build an audit-grade kill event snapshot.
    pub fn build_kill_event(&self, timebase_id: u64, cfg: &Config) -> KillEvent {
        let kill_reason = format!("{:?}", self.kill_reason);
        KillEvent {
            timebase_id,
            kill_reason: if kill_reason.is_empty() {
                "None".to_string()
            } else {
                kill_reason
            },
            delta_usd_t: self.dollar_delta_usd,
            basis_b_t: self.basis_usd,
            daily_pnl_total: self.daily_pnl_total,
            per_venue_q: self.venues.iter().map(|v| v.position_tao).collect(),
            per_venue_dist_liq_sigma: self.venues.iter().map(|v| v.dist_liq_sigma).collect(),
            per_venue_status: self
                .venues
                .iter()
                .map(|v| format!("{:?}", v.status))
                .collect(),
            risk_regime: format!("{:?}", self.risk_regime),
            config_version_id: cfg.version.to_string(),
        }
    }

    /// Build a single best-effort reduce-only IOC exit intent (or None).
    pub fn best_effort_kill_intent_exit_first(
        &self,
        cfg: &Config,
        timebase_id: u64,
    ) -> Option<OrderIntent> {
        let crit_sigma = cfg.risk.liq_crit_sigma;
        let mut selected: Option<(usize, f64)> = None;
        for (idx, venue) in self.venues.iter().enumerate() {
            if venue.status != VenueStatus::Healthy {
                continue;
            }
            if !venue.dist_liq_sigma.is_finite() {
                continue;
            }
            if venue.dist_liq_sigma > crit_sigma {
                continue;
            }
            if venue.position_tao.abs() < 1e-9 {
                continue;
            }
            match selected {
                None => selected = Some((idx, venue.dist_liq_sigma)),
                Some((best_idx, best_dist)) => {
                    if venue.dist_liq_sigma < best_dist
                        || (venue.dist_liq_sigma == best_dist && idx < best_idx)
                    {
                        selected = Some((idx, venue.dist_liq_sigma));
                    }
                }
            }
        }

        let (venue_index, _) = selected?;
        let venue = &self.venues[venue_index];
        let pos = venue.position_tao;
        let side = if pos > 0.0 { Side::Sell } else { Side::Buy };
        let mid = venue.mid.or(self.fair_value).unwrap_or(0.0);
        if !mid.is_finite() || mid <= 0.0 {
            return None;
        }
        let spread = venue.spread.unwrap_or(0.0).abs();
        let guard = cfg.hedge.guard_mult * spread;
        let price = if side == Side::Sell {
            (mid - guard).max(0.0)
        } else {
            mid + guard
        };
        if !price.is_finite() || price <= 0.0 {
            return None;
        }

        Some(OrderIntent::Place(PlaceOrderIntent {
            venue_index,
            venue_id: venue.id.clone(),
            side,
            price,
            size: pos.abs(),
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: Some(format!("kill_exit_{}_{}", venue_index, timebase_id)),
        }))
    }
}

impl GlobalState {
    /// Create a new state with per-venue scaffolding derived from config.
    pub fn new(cfg: &Config) -> Self {
        let mut venues = Vec::with_capacity(cfg.venues.len());

        for vcfg in &cfg.venues {
            venues.push(VenueState {
                id: vcfg.id_arc.clone(),

                mid: None,
                spread: None,
                depth_near_mid: 0.0,
                last_mid_update_ms: None,
                last_book_update_ms: None,
                orderbook_l2: OrderBookL2::new(),
                local_vol_short: 0.0,
                local_vol_long: 0.0,
                prev_ln_mid: None,

                status: VenueStatus::Healthy,
                toxicity: 0.0,
                pending_markouts: VecDeque::new(),
                pending_markouts_next_eval_ms: i64::MAX,
                markout_ewma_usd_per_tao: 0.0,

                open_orders: BTreeMap::new(),
                recent_fills: VecDeque::new(),
                recent_fills_cap: cfg.toxicity.max_pending_per_venue.max(1),
                next_fill_seq: 0,

                position_tao: 0.0,
                funding_8h: 0.0,
                avg_entry_price: 0.0,

                // Synthetic defaults; real implementations poll these.
                margin_balance_usd: 0.0,
                margin_used_usd: 0.0,
                margin_available_usd: 10_000.0,
                price_liq: None,
                dist_liq_sigma: 10.0,

                // MM order management (Milestone H §11)
                mm_open_bid: None,
                mm_open_ask: None,
            });
        }

        GlobalState {
            venues,

            // Scratch buffers for hot-path reuse.
            // Pre-allocate with typical venue count to avoid early reallocations.
            scratch_mids: Vec::with_capacity(cfg.venues.len()),
            scratch_kf_obs: Vec::with_capacity(cfg.venues.len()),

            // Fair value filter
            kf_last_update_ms: 0,
            kf_p: cfg.kalman.p_init,
            kf_x_hat: 0.0,
            fair_value_prev: 0.0,
            fair_value: None,
            fv_short_vol: 0.0,
            fv_long_vol: 0.0,
            sigma_eff: cfg.volatility.sigma_min,

            // Milestone D: FV gating telemetry
            fv_available: false,
            healthy_venues_used: Vec::new(),
            healthy_venues_used_count: 0,

            // Vol control scalars
            vol_ratio_clipped: 1.0,
            spread_mult: 1.0,
            size_mult: 1.0,
            band_mult: 1.0,

            // Inventory & basis
            q_global_tao: 0.0,
            dollar_delta_usd: 0.0,
            basis_usd: 0.0,
            basis_gross_usd: 0.0,

            // PnL
            daily_realised_pnl: 0.0,
            daily_unrealised_pnl: 0.0,
            daily_pnl_total: 0.0,
            pnl_ref_fair_value: None,

            // Risk
            risk_regime: RiskRegime::Normal,
            kill_switch: false,
            kill_reason: KillReason::None,
            kill_handled_at_tick: None,
            delta_limit_usd: cfg.risk.delta_hard_limit_usd_base,
            basis_limit_warn_usd: cfg.risk.basis_warn_frac * cfg.risk.basis_hard_limit_usd,
            basis_limit_hard_usd: cfg.risk.basis_hard_limit_usd,

            #[cfg(feature = "live")]
            live_order_state: crate::live::order_state::LiveOrderState::new(),
        }
    }

    pub fn synthesize_account_snapshots(
        &self,
        _cfg: &Config,
        now_ms: TimestampMs,
    ) -> Vec<SimAccountSnapshot> {
        let mut snapshots = Vec::with_capacity(self.venues.len());
        for (venue_index, venue) in self.venues.iter().enumerate() {
            let price_liq = if let (Some(mid), Some(fair)) = (venue.mid, self.fair_value) {
                if fair.is_finite()
                    && fair > 0.0
                    && self.sigma_eff.is_finite()
                    && self.sigma_eff > 0.0
                {
                    Some(mid - venue.dist_liq_sigma * self.sigma_eff * fair)
                } else {
                    venue.price_liq
                }
            } else {
                venue.price_liq
            };
            snapshots.push(SimAccountSnapshot {
                venue_index,
                timestamp_ms: now_ms,
                position_tao: venue.position_tao,
                avg_entry_price: venue.avg_entry_price,
                funding_8h: venue.funding_8h,
                margin_balance_usd: venue.margin_balance_usd,
                margin_used_usd: venue.margin_used_usd,
                margin_available_usd: venue.margin_available_usd,
                price_liq,
                dist_liq_sigma: venue.dist_liq_sigma,
            });
        }
        snapshots
    }

    pub fn apply_sim_account_snapshots(&mut self, snapshots: &[SimAccountSnapshot]) {
        for snapshot in snapshots {
            let Some(venue) = self.venues.get_mut(snapshot.venue_index) else {
                continue;
            };
            venue.position_tao = snapshot.position_tao;
            venue.avg_entry_price = if snapshot.position_tao.abs() > 0.0 {
                snapshot.avg_entry_price
            } else {
                0.0
            };
            venue.funding_8h = snapshot.funding_8h;
            venue.margin_balance_usd = snapshot.margin_balance_usd;
            venue.margin_used_usd = snapshot.margin_used_usd;
            venue.margin_available_usd = snapshot.margin_available_usd;
            venue.price_liq = snapshot.price_liq;
            venue.dist_liq_sigma = snapshot.dist_liq_sigma;
            venue.last_mid_update_ms = venue.last_mid_update_ms.or(Some(snapshot.timestamp_ms));
        }
    }
}

// --- Perp-fill application + realised PnL accounting ------------------------
//
//  - maintain per-venue position q_v and VWAP entry price,
//  - on each fill, compute realised PnL for the portion that *closes*
//    existing inventory,
//  - subtract trading fees,
//  - accumulate into daily_realised_pnl; total PnL will be updated when
//    we recompute marks/unrealised.

impl GlobalState {
    /// Apply a single perp fill into the state.
    ///
    /// - `venue_index`: index into `self.venues`.
    /// - `side`: Buy or Sell.
    /// - `size_tao`: filled size in TAO (non-negative).
    /// - `price`: fill price in USD per TAO.
    /// - `fee_bps`: net fee in basis points (positive = cost, negative = rebate).
    pub fn apply_perp_fill(
        &mut self,
        venue_index: usize,
        side: Side,
        size_tao: f64,
        price: f64,
        fee_bps: f64,
    ) {
        if size_tao <= 0.0 || !price.is_finite() || price <= 0.0 {
            return;
        }

        let Some(v) = self.venues.get_mut(venue_index) else {
            return;
        };

        // Signed trade size: + for buy, - for sell.
        let trade = match side {
            Side::Buy => size_tao,
            Side::Sell => -size_tao,
        };

        let q_old = v.position_tao;
        let p_old = v.avg_entry_price;
        let p_trade = price;

        // Total fee in USD (positive reduces PnL; negative is a rebate).
        let fee = (fee_bps / 10_000.0) * p_trade * size_tao;

        let mut realised = 0.0_f64;

        if q_old == 0.0 {
            // Opening a fresh position.
            v.position_tao = trade;
            v.avg_entry_price = p_trade;
        } else {
            let same_dir = q_old.signum() == trade.signum();

            if same_dir {
                // Add to existing position, update VWAP entry price.
                let q_new = q_old + trade;
                if q_new != 0.0 {
                    let w_old = q_old.abs();
                    let w_trade = trade.abs();
                    v.avg_entry_price = (p_old * w_old + p_trade * w_trade) / (w_old + w_trade);
                    v.position_tao = q_new;
                } else {
                    // Exactly flat (rare).
                    v.position_tao = 0.0;
                    v.avg_entry_price = 0.0;
                }
            } else {
                // Closing or flipping some/all of the existing position.
                let close_qty = trade.abs().min(q_old.abs());

                if close_qty > 0.0 {
                    if q_old > 0.0 {
                        // Closing a long: sell closes.
                        realised += (p_trade - p_old) * close_qty;
                    } else {
                        // Closing a short: buy closes.
                        realised += (p_old - p_trade) * close_qty;
                    }
                }

                let q_new = q_old + trade;

                if q_old.abs() > trade.abs() {
                    // Partial close; keep original entry price.
                    v.position_tao = q_new;
                    v.avg_entry_price = p_old;
                } else if q_old.abs() < trade.abs() {
                    // Closed and flipped into a new position.
                    v.position_tao = q_new;
                    v.avg_entry_price = p_trade;
                } else {
                    // Exactly flat.
                    v.position_tao = 0.0;
                    v.avg_entry_price = 0.0;
                }
            }
        }

        // Fees always apply to the executed trade.
        realised -= fee;

        self.daily_realised_pnl += realised;
        // Unrealised is marked in `recompute_after_fills`.
        self.daily_pnl_total = self.daily_realised_pnl + self.daily_unrealised_pnl;
    }

    /// Apply a fill event and update order/markout ledgers deterministically.
    ///
    /// This is the canonical path for fill application in simulations and live loops.
    pub fn apply_fill_event(&mut self, fill: &FillEvent, now_ms: TimestampMs, cfg: &Config) {
        if fill.size <= 0.0 || !fill.price.is_finite() || fill.price <= 0.0 {
            return;
        }

        let pre_position = self
            .venues
            .get(fill.venue_index)
            .map(|v| v.position_tao)
            .unwrap_or(0.0);
        let pre_q_global = self.q_global_tao;
        let realised_before = self.daily_realised_pnl;

        self.apply_perp_fill(
            fill.venue_index,
            fill.side,
            fill.size,
            fill.price,
            fill.fee_bps,
        );
        let realised_after = self.daily_realised_pnl;
        let realised_pnl = realised_after - realised_before;

        let post_position = self
            .venues
            .get(fill.venue_index)
            .map(|v| v.position_tao)
            .unwrap_or(pre_position);
        let signed_size = match fill.side {
            Side::Buy => fill.size,
            Side::Sell => -fill.size,
        };
        let post_q_global = pre_q_global + signed_size;

        let fair = self.fair_value.unwrap_or(self.fair_value_prev).max(1.0);
        let mid = self
            .venues
            .get(fill.venue_index)
            .and_then(|v| v.mid)
            .unwrap_or(fair);

        self.record_fill_ledger(
            fill,
            now_ms,
            pre_position,
            post_position,
            pre_q_global,
            post_q_global,
            realised_pnl,
            fair,
            mid,
            cfg.toxicity.markout_horizon_ms,
            cfg.toxicity.max_pending_per_venue,
        );
    }

    /// Recompute inventory, basis and mark-to-market PnL.
    ///
    /// Despite the historical name, this is safe and intended to be called:
    ///  - after a batch of fills, and
    ///  - on every main tick after fair value updates (whitepaper: mark-to-S_t).
    pub fn recompute_after_fills(&mut self, _cfg: &Config) {
        // Prefer current fair; else fall back to previous fair.
        let s_t = self.fair_value.unwrap_or(self.fair_value_prev);

        // If we still don't have a sane fair value, keep everything at 0.
        if !s_t.is_finite() || s_t <= 0.0 {
            self.q_global_tao = 0.0;
            self.dollar_delta_usd = 0.0;
            self.basis_usd = 0.0;
            self.basis_gross_usd = 0.0;
            self.daily_unrealised_pnl = 0.0;
            self.daily_pnl_total = self.daily_realised_pnl;
            return;
        }

        self.q_global_tao = 0.0;
        self.dollar_delta_usd = 0.0;
        self.basis_usd = 0.0;
        self.basis_gross_usd = 0.0;

        let mut unrealised = 0.0_f64;

        for v in &self.venues {
            let q = v.position_tao;

            self.q_global_tao += q;
            self.dollar_delta_usd += q * s_t;

            // Per-venue basis b_v = m_v - S_t (if no mid, assume m_v == S_t => b_v=0).
            let b_v = v.mid.unwrap_or(s_t) - s_t;

            // Net basis exposure: sum q_v * b_v
            self.basis_usd += q * b_v;

            // Gross basis exposure: sum |q_v| * |b_v|
            self.basis_gross_usd += q.abs() * b_v.abs();

            // Mark-to-fair unrealised PnL.
            if q != 0.0 {
                unrealised += q * (s_t - v.avg_entry_price);
            }
        }

        self.daily_unrealised_pnl = unrealised;
        self.daily_pnl_total = self.daily_realised_pnl + self.daily_unrealised_pnl;
    }

    /// Record a pending markout evaluation for a fill.
    ///
    /// Called immediately after a fill is applied. The markout will be
    /// evaluated at `now_ms + horizon_ms` in `update_toxicity_and_health`.
    pub fn record_pending_markout(&mut self, record: PendingMarkoutRecord) {
        if record.size_tao <= 0.0 || !record.price.is_finite() || record.price <= 0.0 {
            return;
        }

        let Some(v) = self.venues.get_mut(record.venue_index) else {
            return;
        };

        let t_eval_ms = record.now_ms + record.horizon_ms;

        // Monotonicity invariant: new t_eval_ms must be >= the last entry's t_eval_ms.
        // This invariant holds because:
        // 1. Fills are recorded in chronological order (now_ms monotonically increases).
        // 2. The markout horizon is a fixed constant from config for all fills.
        // Therefore t_eval_ms = now_ms + horizon_ms is monotonically non-decreasing.
        // This allows O(1) FIFO processing in update_toxicity_and_health_impl instead
        // of O(n) full scans - we simply pop_front while front().t_eval_ms <= now.
        debug_assert!(
            v.pending_markouts
                .back()
                .is_none_or(|last| t_eval_ms >= last.t_eval_ms),
            "Pending markout monotonicity violated: new t_eval_ms={} < last t_eval_ms={}. \
             This breaks the FIFO processing optimization.",
            t_eval_ms,
            v.pending_markouts.back().map_or(0, |last| last.t_eval_ms)
        );

        let entry = PendingMarkout {
            t_fill_ms: record.now_ms,
            t_eval_ms,
            side: record.side,
            size_tao: record.size_tao,
            price: record.price,
            fair_at_fill: record.fair,
            mid_at_fill: record.mid,
            fill_seq: record.fill_seq,
        };

        v.pending_markouts.push_back(entry);

        // Enforce bounded queue: drop oldest entries if over limit.
        while v.pending_markouts.len() > record.max_pending {
            v.pending_markouts.pop_front();
        }

        // Opt17: Refresh cached next eval time after push and any trimming.
        v.pending_markouts_next_eval_ms = v
            .pending_markouts
            .front()
            .map_or(i64::MAX, |pm| pm.t_eval_ms);
    }

    fn record_fill_ledger(
        &mut self,
        fill: &FillEvent,
        now_ms: TimestampMs,
        pre_position_tao: f64,
        post_position_tao: f64,
        pre_q_global_tao: f64,
        post_q_global_tao: f64,
        realised_pnl_usd: f64,
        fair: f64,
        mid: f64,
        markout_horizon_ms: i64,
        max_pending: usize,
    ) {
        let Some(v) = self.venues.get_mut(fill.venue_index) else {
            return;
        };

        if let Some(order_id) = fill.order_id.as_ref() {
            let mut remove = false;
            if let Some(open) = v.open_orders.get_mut(order_id) {
                open.remaining = (open.remaining - fill.size).max(0.0);
                remove = open.remaining <= 0.0;
            }
            if remove {
                v.open_orders.remove(order_id);
                if v.mm_open_bid
                    .as_ref()
                    .is_some_and(|o| o.order_id == order_id.as_str())
                {
                    v.mm_open_bid = None;
                }
                if v.mm_open_ask
                    .as_ref()
                    .is_some_and(|o| o.order_id == order_id.as_str())
                {
                    v.mm_open_ask = None;
                }
            }
        }

        let fill_seq = v.next_fill_seq;
        v.next_fill_seq = v.next_fill_seq.wrapping_add(1);

        v.recent_fills.push_back(FillRecord {
            fill_seq,
            order_id: fill.order_id.clone(),
            client_order_id: fill.client_order_id.clone(),
            side: fill.side,
            price: fill.price,
            size: fill.size,
            fill_time_ms: now_ms,
            purpose: fill.purpose,
            fee_bps: fill.fee_bps,
            markout_pnl_short: None,
            pre_position_tao: Some(pre_position_tao),
            post_position_tao: Some(post_position_tao),
            pre_q_global_tao: Some(pre_q_global_tao),
            post_q_global_tao: Some(post_q_global_tao),
            realised_pnl_usd: Some(realised_pnl_usd),
        });

        while v.recent_fills.len() > v.recent_fills_cap {
            v.recent_fills.pop_front();
        }

        self.record_pending_markout(PendingMarkoutRecord {
            venue_index: fill.venue_index,
            side: fill.side,
            size_tao: fill.size,
            price: fill.price,
            now_ms,
            fair,
            mid,
            horizon_ms: markout_horizon_ms,
            max_pending,
            fill_seq,
        });
    }
}

/// Parameters for recording a pending markout evaluation.
///
/// Bundles all parameters needed by `record_pending_markout` to avoid
/// clippy::too_many_arguments.
pub struct PendingMarkoutRecord {
    /// Index into `GlobalState.venues`.
    pub venue_index: usize,
    /// Buy or Sell.
    pub side: Side,
    /// Filled size in TAO.
    pub size_tao: f64,
    /// Fill price in USD/TAO.
    pub price: f64,
    /// Current timestamp.
    pub now_ms: TimestampMs,
    /// Current fair value (for optional analysis).
    pub fair: f64,
    /// Current venue mid price.
    pub mid: f64,
    /// Time until markout evaluation.
    pub horizon_ms: i64,
    /// Maximum queue size (older entries dropped).
    pub max_pending: usize,
    /// Deterministic fill sequence ID for linking markout to FillRecord.
    pub fill_seq: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::Side;

    fn approx(label: &str, expected: f64, actual: f64) {
        let diff = (expected - actual).abs();
        assert!(
            diff < 1e-6,
            "{}: left={} right={} diff={}",
            label,
            expected,
            actual,
            diff
        );
    }

    #[test]
    fn basis_and_unrealised_pnl_two_venues() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        // Set fair value.
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;

        // Venue 0: long 5 @ 95, mid 102.
        {
            let v0 = &mut state.venues[0];
            v0.position_tao = 5.0;
            v0.avg_entry_price = 95.0;
            v0.mid = Some(102.0);
        }

        // Venue 1: short 3 @ 110, mid 98.
        {
            let v1 = &mut state.venues[1];
            v1.position_tao = -3.0;
            v1.avg_entry_price = 110.0;
            v1.mid = Some(98.0);
        }

        state.recompute_after_fills(&cfg);

        approx("q_global_tao", 2.0, state.q_global_tao);
        approx("dollar_delta_usd", 200.0, state.dollar_delta_usd);

        // Basis contributions:
        // v0: 5 * (102 - 100) = 10
        // v1: -3 * (98 - 100) =  6
        approx("basis_usd", 16.0, state.basis_usd);
        approx("basis_gross_usd", 16.0, state.basis_gross_usd);

        // Unrealised PnL is marked vs fair value:
        // v0: 5 * (100 - 95) = 25
        // v1: -3 * (100 - 110) = 30
        approx("daily_unrealised_pnl", 55.0, state.daily_unrealised_pnl);
        approx("daily_pnl_total", 55.0, state.daily_pnl_total);
    }

    #[test]
    fn long_open_and_close_no_fees() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        let venue_index = 0;
        let size = 10.0;
        let entry_price = 100.0;
        let exit_price = 110.0;

        // Open long.
        state.apply_perp_fill(venue_index, Side::Buy, size, entry_price, 0.0);
        // Close long.
        state.apply_perp_fill(venue_index, Side::Sell, size, exit_price, 0.0);

        // Mark at exit price.
        state.fair_value = Some(exit_price);
        state.fair_value_prev = exit_price;
        state.recompute_after_fills(&cfg);

        approx(
            "realised PnL long round-trip no fees",
            (exit_price - entry_price) * size,
            state.daily_realised_pnl,
        );
        approx("unrealised PnL flat book", 0.0, state.daily_unrealised_pnl);
        approx("q_global_tao flat book", 0.0, state.q_global_tao);
    }

    #[test]
    fn short_open_and_close_with_fees() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);

        let venue_index = 0;
        let size = 10.0;
        let entry_price = 110.0;
        let exit_price = 100.0;
        let fee_bps = 5.0;

        // Open short: sell first.
        state.apply_perp_fill(venue_index, Side::Sell, size, entry_price, fee_bps);
        // Close short: buy back.
        state.apply_perp_fill(venue_index, Side::Buy, size, exit_price, fee_bps);

        // Mark at exit price (flat book).
        state.fair_value = Some(exit_price);
        state.fair_value_prev = exit_price;
        state.recompute_after_fills(&cfg);

        let fee_entry = (fee_bps / 10_000.0) * entry_price * size;
        let fee_exit = (fee_bps / 10_000.0) * exit_price * size;
        let expected_realised = (entry_price - exit_price) * size - fee_entry - fee_exit;

        approx(
            "realised PnL short round-trip with fees",
            expected_realised,
            state.daily_realised_pnl,
        );
        approx("unrealised PnL flat book", 0.0, state.daily_unrealised_pnl);
        approx("q_global_tao flat book", 0.0, state.q_global_tao);
    }
}
