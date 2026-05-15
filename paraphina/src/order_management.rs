// src/order_management.rs
//
// Milestone H: Order management for MM quotes (Whitepaper §11).
// Implements cancel/replace logic with MIN_QUOTE_LIFETIME_MS and tolerance gates.

use std::collections::BTreeMap;

use crate::actions::ActionIdGenerator;
use crate::config::Config;
use crate::mm::{
    compute_venue_utility_decision, evaluate_replace_order,
    venue_utility_conversion_penalties_enabled, ActiveMmOrder, MmLevel, MmQuote, MmReplaceDecision,
    MmReplaceOutcome, MmReplaceReason, ShouldReplaceOrderCtx,
};
use crate::state::{GlobalState, MmOpenOrder, MmOpenTrackingSource};
use crate::types::{
    CancelOrderIntent, OrderIntent, OrderPurpose, PlaceOrderIntent, ReplaceOrderIntent, Side,
    TimeInForce, TimestampMs,
};
use serde::Serialize;

const EXTENDED_NATIVE_REPLACE_ENABLED_ENV: &str = "PARAPHINA_EXTENDED_NATIVE_REPLACE_ENABLED";

/// Output of MM order management planner.
#[derive(Debug, Clone)]
pub struct MmOrderManagementPlan {
    pub intents: Vec<OrderIntent>,
    pub decision_summary: MmOrderDecisionSummary,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MmOrderDecisionSummary {
    pub keep_count: u64,
    pub replace_count: u64,
    pub place_count: u64,
    pub cancel_count: u64,
    pub aster_touch_offside_fastpath_count: u64,
    pub aster_touch_offside_nearmiss_count: u64,
    pub aster_touch_offside_nearmiss_by_reason: BTreeMap<String, u64>,
    pub touch_risk_hysteresis_count: u64,
    pub touch_risk_fastpath_count: u64,
    pub touch_risk_size_band_count: u64,
    pub touch_risk_nearmiss_count: u64,
    pub touch_risk_nearmiss_by_reason: BTreeMap<String, u64>,
    pub compression_edge_hysteresis_count: u64,
    pub compression_edge_nearmiss_count: u64,
    pub compression_edge_nearmiss_by_reason: BTreeMap<String, u64>,
    pub keep_by_reason: BTreeMap<String, u64>,
    pub replace_by_reason: BTreeMap<String, u64>,
    pub keep_by_utility_tier: BTreeMap<String, u64>,
    pub replace_by_utility_tier: BTreeMap<String, u64>,
    pub keep_by_venue_role: BTreeMap<String, u64>,
    pub replace_by_venue_role: BTreeMap<String, u64>,
    pub supported_replace_opportunity_by_venue: BTreeMap<String, u64>,
    pub supported_replace_visibility_miss_by_venue: BTreeMap<String, u64>,
    pub supported_replace_keep_by_reason: BTreeMap<String, u64>,
    pub supported_replace_preempted_by_venue: BTreeMap<String, u64>,
    pub supported_replace_gap_grace_by_venue: BTreeMap<String, u64>,
    pub paradex_pending_place_guard_place_suppressed: u64,
    pub paradex_pending_place_guard_cancel_suppressed: u64,
    pub paradex_pending_place_guard_replace_suppressed: u64,
    pub decision_records: Vec<MmDecisionRecord>,
    pub replace_decisions: Vec<MmReplaceDecisionRecord>,
    pub supported_replace_visibility_records: Vec<SupportedReplaceVisibilityRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MmDecisionRecord {
    pub decision_id: String,
    pub venue_index: usize,
    pub venue_id: String,
    pub side: String,
    pub purpose: String,
    pub outcome: String,
    pub reason: String,
    pub fair_value: Option<f64>,
    pub q_global_tao: f64,
    pub current_order_id: Option<String>,
    pub current_price: Option<f64>,
    pub current_size: Option<f64>,
    pub desired_price: Option<f64>,
    pub desired_size: Option<f64>,
    pub client_order_id: Option<String>,
    pub utility_tier: Option<String>,
    pub utility_reason: Option<String>,
    pub venue_role: Option<String>,
    pub role_cap_applied: Option<bool>,
    pub inventory_reducing: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MmReplaceDecisionRecord {
    pub decision_id: String,
    pub venue_index: usize,
    pub venue_id: String,
    pub side: String,
    pub outcome: String,
    pub reason: String,
    pub fair_value: Option<f64>,
    pub q_global_tao: f64,
    pub current_price: f64,
    pub current_size: f64,
    pub desired_price: f64,
    pub desired_size: f64,
    pub current_order_id: String,
    pub client_order_id: Option<String>,
    pub utility_tier: String,
    pub utility_reason: String,
    pub venue_role: String,
    pub role_cap_applied: bool,
    pub inventory_reducing: bool,
    pub age_ms: TimestampMs,
    pub min_quote_lifetime_ms: TimestampMs,
    pub min_price_reprice_ms: TimestampMs,
    pub price_diff_ticks: f64,
    pub price_tol_ticks: f64,
    pub size_diff_rel: f64,
    pub size_tol_rel: f64,
    pub touch_risk_size_band_applied: bool,
    pub hyperliquid_sell_price_excess_ticks: Option<f64>,
    pub hyperliquid_sell_best_ask_motion_ticks: Option<f64>,
    pub hyperliquid_sell_current_clearance_ticks: Option<f64>,
    pub hyperliquid_sell_desired_clearance_ticks: Option<f64>,
    pub hyperliquid_sell_price_reprice_remaining_ms: Option<TimestampMs>,
    pub aster_touch_offside_nearmiss_reason: Option<String>,
    pub touch_risk_nearmiss_reason: Option<String>,
    pub compression_edge_nearmiss_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportedReplaceIdentityKind {
    ExchangeOrderId,
    ClientOrderId,
}

impl SupportedReplaceIdentityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExchangeOrderId => "exchange_order_id",
            Self::ClientOrderId => "client_order_id",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportedReplaceBlockedBy {
    DesiredSuppressed,
    NoCurrentSameSide,
    IdentityMissing,
    CurrentTooYoung,
    PriceCadence,
    PriceTol,
    SizeTol,
    TouchRisk,
    CompressionEdge,
    StationaryDeadband,
    PreemptedByCancelLayer,
}

impl SupportedReplaceBlockedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DesiredSuppressed => "desired_suppressed",
            Self::NoCurrentSameSide => "no_current_same_side",
            Self::IdentityMissing => "identity_missing",
            Self::CurrentTooYoung => "current_too_young",
            Self::PriceCadence => "price_cadence",
            Self::PriceTol => "price_tol",
            Self::SizeTol => "size_tol",
            Self::TouchRisk => "touch_risk",
            Self::CompressionEdge => "compression_edge",
            Self::StationaryDeadband => "stationary_deadband",
            Self::PreemptedByCancelLayer => "preempted_by_cancel_layer",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportedReplaceVisibilityRecord {
    pub decision_id: String,
    pub venue_index: usize,
    pub venue_id: String,
    pub side: String,
    pub desired_present: bool,
    pub current_present: bool,
    pub current_source: Option<String>,
    pub identity_kind: Option<String>,
    pub current_age_ms: Option<TimestampMs>,
    pub native_replace_supported: bool,
    pub action: String,
    pub blocked_by: Option<String>,
    pub current_order_id: Option<String>,
    pub current_client_order_id: Option<String>,
    pub post_control_absence_reason: Option<String>,
    pub suppression_grace_applied: bool,
}

impl MmOrderDecisionSummary {
    fn record_supported_replace_visibility(
        &mut self,
        venue_id: &str,
        desired_present: bool,
        current_present: bool,
        current_source: Option<MmOpenTrackingSource>,
        blocked_by: Option<SupportedReplaceBlockedBy>,
        record: SupportedReplaceVisibilityRecord,
    ) {
        if desired_present && current_present {
            *self
                .supported_replace_opportunity_by_venue
                .entry(venue_id.to_string())
                .or_insert(0) += 1;
        }
        if blocked_by.is_some() {
            *self
                .supported_replace_visibility_miss_by_venue
                .entry(venue_id.to_string())
                .or_insert(0) += 1;
        }
        if let Some(reason) = blocked_by {
            *self
                .supported_replace_keep_by_reason
                .entry(reason.as_str().to_string())
                .or_insert(0) += 1;
            if matches!(reason, SupportedReplaceBlockedBy::PreemptedByCancelLayer) {
                *self
                    .supported_replace_preempted_by_venue
                    .entry(venue_id.to_string())
                    .or_insert(0) += 1;
            }
        }
        if matches!(current_source, Some(MmOpenTrackingSource::GapGrace)) {
            *self
                .supported_replace_gap_grace_by_venue
                .entry(venue_id.to_string())
                .or_insert(0) += 1;
        }
        self.supported_replace_visibility_records.push(record);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_replace_decision(
        &mut self,
        decision_id: &str,
        venue_index: usize,
        venue_id: &str,
        side: Side,
        fair_value: Option<f64>,
        q_global_tao: f64,
        current: &ActiveMmOrder,
        current_order_id: &str,
        desired: &MmLevel,
        client_order_id: Option<String>,
        decision: MmReplaceDecision,
    ) {
        let tier_key = decision.utility_tier.as_str().to_string();
        let reason_key = decision.reason.as_str().to_string();
        let role_key = decision.venue_role.as_str().to_string();
        let compression_edge_nearmiss_reason = decision
            .compression_edge_nearmiss_reason
            .map(|reason| reason.as_str().to_string());
        let aster_touch_offside_nearmiss_reason = decision
            .aster_touch_offside_nearmiss_reason
            .map(|reason| reason.as_str().to_string());
        let touch_risk_nearmiss_reason = decision
            .touch_risk_nearmiss_reason
            .map(|reason| reason.as_str().to_string());
        if reason_key == "touch_risk_hysteresis" {
            self.touch_risk_hysteresis_count += 1;
        }
        if reason_key == "aster_touch_offside_fastpath" {
            self.aster_touch_offside_fastpath_count += 1;
        }
        if let Some(reason) = aster_touch_offside_nearmiss_reason.as_ref() {
            self.aster_touch_offside_nearmiss_count += 1;
            *self
                .aster_touch_offside_nearmiss_by_reason
                .entry(reason.clone())
                .or_insert(0) += 1;
        }
        if reason_key == "touch_risk_fastpath" {
            self.touch_risk_fastpath_count += 1;
        }
        if decision.touch_risk_size_band_applied {
            self.touch_risk_size_band_count += 1;
        }
        if let Some(reason) = touch_risk_nearmiss_reason.as_ref() {
            self.touch_risk_nearmiss_count += 1;
            *self
                .touch_risk_nearmiss_by_reason
                .entry(reason.clone())
                .or_insert(0) += 1;
        }
        if reason_key == "compression_edge_hysteresis" {
            self.compression_edge_hysteresis_count += 1;
        }
        if let Some(reason) = compression_edge_nearmiss_reason.as_ref() {
            self.compression_edge_nearmiss_count += 1;
            *self
                .compression_edge_nearmiss_by_reason
                .entry(reason.clone())
                .or_insert(0) += 1;
        }
        match decision.outcome {
            MmReplaceOutcome::Keep => {
                self.keep_count += 1;
                *self.keep_by_reason.entry(reason_key.clone()).or_insert(0) += 1;
                *self
                    .keep_by_utility_tier
                    .entry(tier_key.clone())
                    .or_insert(0) += 1;
                *self.keep_by_venue_role.entry(role_key.clone()).or_insert(0) += 1;
            }
            MmReplaceOutcome::Replace => {
                self.replace_count += 1;
                *self
                    .replace_by_reason
                    .entry(reason_key.clone())
                    .or_insert(0) += 1;
                *self
                    .replace_by_utility_tier
                    .entry(tier_key.clone())
                    .or_insert(0) += 1;
                *self
                    .replace_by_venue_role
                    .entry(role_key.clone())
                    .or_insert(0) += 1;
            }
        }
        self.decision_records.push(MmDecisionRecord {
            decision_id: decision_id.to_string(),
            venue_index,
            venue_id: venue_id.to_string(),
            side: format!("{:?}", side),
            purpose: format!("{:?}", OrderPurpose::Mm),
            outcome: decision.outcome.as_str().to_string(),
            reason: reason_key.clone(),
            fair_value,
            q_global_tao,
            current_order_id: Some(current_order_id.to_string()),
            current_price: Some(current.price),
            current_size: Some(current.size),
            desired_price: Some(desired.price),
            desired_size: Some(desired.size),
            client_order_id: client_order_id.clone(),
            utility_tier: Some(tier_key.clone()),
            utility_reason: Some(decision.utility_reason.as_str().to_string()),
            venue_role: Some(role_key.clone()),
            role_cap_applied: Some(decision.role_cap_applied),
            inventory_reducing: Some(decision.inventory_reducing),
        });
        self.replace_decisions.push(MmReplaceDecisionRecord {
            decision_id: decision_id.to_string(),
            venue_index,
            venue_id: venue_id.to_string(),
            side: format!("{:?}", side),
            outcome: decision.outcome.as_str().to_string(),
            reason: reason_key,
            fair_value,
            q_global_tao,
            current_price: current.price,
            current_size: current.size,
            desired_price: desired.price,
            desired_size: desired.size,
            current_order_id: current_order_id.to_string(),
            client_order_id,
            utility_tier: tier_key,
            utility_reason: decision.utility_reason.as_str().to_string(),
            venue_role: role_key,
            role_cap_applied: decision.role_cap_applied,
            inventory_reducing: decision.inventory_reducing,
            age_ms: decision.age_ms,
            min_quote_lifetime_ms: decision.min_quote_lifetime_ms,
            min_price_reprice_ms: decision.min_price_reprice_ms,
            price_diff_ticks: decision.price_diff_ticks,
            price_tol_ticks: decision.price_tol_ticks,
            size_diff_rel: decision.size_diff_rel,
            size_tol_rel: decision.size_tol_rel,
            touch_risk_size_band_applied: decision.touch_risk_size_band_applied,
            hyperliquid_sell_price_excess_ticks: decision.hyperliquid_sell_price_excess_ticks,
            hyperliquid_sell_best_ask_motion_ticks: decision.hyperliquid_sell_best_ask_motion_ticks,
            hyperliquid_sell_current_clearance_ticks: decision
                .hyperliquid_sell_current_clearance_ticks,
            hyperliquid_sell_desired_clearance_ticks: decision
                .hyperliquid_sell_desired_clearance_ticks,
            hyperliquid_sell_price_reprice_remaining_ms: decision
                .hyperliquid_sell_price_reprice_remaining_ms,
            aster_touch_offside_nearmiss_reason,
            touch_risk_nearmiss_reason,
            compression_edge_nearmiss_reason,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn record_place_decision(
        &mut self,
        decision_id: &str,
        venue_index: usize,
        venue_id: &str,
        side: Side,
        fair_value: Option<f64>,
        q_global_tao: f64,
        desired: &MmLevel,
        client_order_id: Option<String>,
        utility_tier: Option<String>,
        utility_reason: Option<String>,
        venue_role: Option<String>,
        role_cap_applied: Option<bool>,
    ) {
        self.place_count += 1;
        self.decision_records.push(MmDecisionRecord {
            decision_id: decision_id.to_string(),
            venue_index,
            venue_id: venue_id.to_string(),
            side: format!("{:?}", side),
            purpose: format!("{:?}", OrderPurpose::Mm),
            outcome: "place".to_string(),
            reason: "new_quote".to_string(),
            fair_value,
            q_global_tao,
            current_order_id: None,
            current_price: None,
            current_size: None,
            desired_price: Some(desired.price),
            desired_size: Some(desired.size),
            client_order_id,
            utility_tier,
            utility_reason,
            venue_role,
            role_cap_applied,
            inventory_reducing: Some(false),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn record_keep_without_desired(
        &mut self,
        decision_id: &str,
        venue_index: usize,
        venue_id: &str,
        side: Side,
        fair_value: Option<f64>,
        q_global_tao: f64,
        current: &ActiveMmOrder,
        current_order_id: &str,
        client_order_id: Option<String>,
        utility_tier: Option<String>,
        utility_reason: Option<String>,
        venue_role: Option<String>,
        role_cap_applied: Option<bool>,
        reason: &str,
    ) {
        self.keep_count += 1;
        *self.keep_by_reason.entry(reason.to_string()).or_insert(0) += 1;
        if let Some(tier) = utility_tier.as_ref() {
            *self.keep_by_utility_tier.entry(tier.clone()).or_insert(0) += 1;
        }
        if let Some(role) = venue_role.as_ref() {
            *self.keep_by_venue_role.entry(role.clone()).or_insert(0) += 1;
        }
        self.decision_records.push(MmDecisionRecord {
            decision_id: decision_id.to_string(),
            venue_index,
            venue_id: venue_id.to_string(),
            side: format!("{:?}", side),
            purpose: format!("{:?}", OrderPurpose::Mm),
            outcome: "keep".to_string(),
            reason: reason.to_string(),
            fair_value,
            q_global_tao,
            current_order_id: Some(current_order_id.to_string()),
            current_price: Some(current.price),
            current_size: Some(current.size),
            desired_price: None,
            desired_size: None,
            client_order_id,
            utility_tier,
            utility_reason,
            venue_role,
            role_cap_applied,
            inventory_reducing: None,
        });
    }

    fn record_cancel(&mut self) {
        self.cancel_count += 1;
    }

    pub fn bind_mm_intent_client_order_ids(&mut self, intents: &[OrderIntent]) {
        let mut by_slot = BTreeMap::new();
        for intent in intents {
            match intent {
                OrderIntent::Place(place) if matches!(place.purpose, OrderPurpose::Mm) => {
                    if let Some(client_order_id) = place.client_order_id.clone() {
                        by_slot.insert(
                            (
                                place.venue_index,
                                format!("{:?}", place.side),
                                "place".to_string(),
                            ),
                            client_order_id,
                        );
                    }
                }
                OrderIntent::Replace(replace) if matches!(replace.purpose, OrderPurpose::Mm) => {
                    if let Some(client_order_id) = replace.client_order_id.clone() {
                        by_slot.insert(
                            (
                                replace.venue_index,
                                format!("{:?}", replace.side),
                                "replace".to_string(),
                            ),
                            client_order_id,
                        );
                    }
                }
                _ => {}
            }
        }

        for record in &mut self.decision_records {
            if !matches!(record.outcome.as_str(), "place" | "replace") {
                continue;
            }
            record.client_order_id = by_slot
                .get(&(
                    record.venue_index,
                    record.side.clone(),
                    record.outcome.clone(),
                ))
                .cloned();
        }
        for record in &mut self.replace_decisions {
            if !matches!(record.outcome.as_str(), "replace") {
                continue;
            }
            record.client_order_id = by_slot
                .get(&(
                    record.venue_index,
                    record.side.clone(),
                    record.outcome.clone(),
                ))
                .cloned();
        }
    }
}

fn is_hyperliquid_cloid(value: &str) -> bool {
    value.len() == 34
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_hyperliquid_oid(value: &str) -> bool {
    value.parse::<u64>().is_ok()
}

fn is_lighter_numeric_identity(value: &str) -> bool {
    value.parse::<u64>().is_ok()
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("y")
        })
        .unwrap_or(false)
}

fn extended_native_replace_enabled() -> bool {
    env_flag_enabled(EXTENDED_NATIVE_REPLACE_ENABLED_ENV)
}

fn is_extended_external_identity(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty() && !value.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_supported_native_replace_venue(venue_id: &str) -> bool {
    venue_id.eq_ignore_ascii_case("hyperliquid")
        || venue_id.eq_ignore_ascii_case("paradex")
        || venue_id.eq_ignore_ascii_case("lighter")
        || (venue_id.eq_ignore_ascii_case("extended") && extended_native_replace_enabled())
}

fn supported_native_replace_identity_kind(
    venue_id: &str,
    current: &MmOpenOrder,
) -> Option<SupportedReplaceIdentityKind> {
    if venue_id.eq_ignore_ascii_case("hyperliquid") {
        if current
            .client_order_id
            .as_deref()
            .is_some_and(is_hyperliquid_cloid)
        {
            return Some(SupportedReplaceIdentityKind::ClientOrderId);
        }
        if is_hyperliquid_cloid(&current.order_id) || is_hyperliquid_oid(&current.order_id) {
            return Some(SupportedReplaceIdentityKind::ExchangeOrderId);
        }
        return None;
    }
    if venue_id.eq_ignore_ascii_case("paradex") {
        return (!current.order_id.starts_with("co_"))
            .then_some(SupportedReplaceIdentityKind::ExchangeOrderId);
    }
    if venue_id.eq_ignore_ascii_case("lighter") {
        if is_lighter_numeric_identity(&current.order_id) {
            return Some(SupportedReplaceIdentityKind::ExchangeOrderId);
        }
        if current
            .client_order_id
            .as_deref()
            .is_some_and(is_lighter_numeric_identity)
        {
            return Some(SupportedReplaceIdentityKind::ClientOrderId);
        }
        return None;
    }
    if venue_id.eq_ignore_ascii_case("extended") {
        if !extended_native_replace_enabled() {
            return None;
        }
        if current
            .client_order_id
            .as_deref()
            .is_some_and(is_extended_external_identity)
        {
            return Some(SupportedReplaceIdentityKind::ClientOrderId);
        }
        return None;
    }
    None
}

fn supported_native_replace_order_id(venue_id: &str, current: &MmOpenOrder) -> Option<String> {
    match supported_native_replace_identity_kind(venue_id, current) {
        Some(SupportedReplaceIdentityKind::ClientOrderId) => current.client_order_id.clone(),
        Some(SupportedReplaceIdentityKind::ExchangeOrderId) => Some(current.order_id.clone()),
        None => None,
    }
}

#[allow(clippy::needless_lifetimes)]
fn quote_side_terminal_reason<'a>(quote: Option<&'a MmQuote>, side: Side) -> Option<&'a str> {
    let quote = quote?;
    Some(match side {
        Side::Buy => quote.bid_terminal_reason,
        Side::Sell => quote.ask_terminal_reason,
    })
}

fn supported_replace_post_control_suppression(reason: Option<&str>) -> bool {
    matches!(reason, Some("projected_mm_budget_suppressed"))
}

fn supported_replace_edge_floor_absence(reason: Option<&str>) -> bool {
    matches!(reason, Some("edge_below_min"))
}

fn active_mm_order_is_dangerous(
    cfg: &Config,
    venue_index: usize,
    current: &ActiveMmOrder,
    best_bid: f64,
    best_ask: f64,
) -> bool {
    let tick = cfg
        .venues
        .get(venue_index)
        .map(|venue| venue.tick_size.max(1e-6))
        .unwrap_or(1e-6);
    match current.side {
        Side::Buy => {
            let non_passive = current.price > best_bid - tick;
            let crosses = current.price >= best_ask - tick;
            non_passive || crosses
        }
        Side::Sell => {
            let non_passive = current.price < best_ask + tick;
            let crosses = current.price <= best_bid + tick;
            non_passive || crosses
        }
    }
}

fn current_mm_order_edge_metrics(
    cfg: &Config,
    venue_index: usize,
    venue_id: &str,
    current: &ActiveMmOrder,
    fair_value: Option<f64>,
) -> Option<(f64, f64, f64)> {
    let fair_value = fair_value.filter(|v| v.is_finite() && *v > 0.0)?;
    let vcfg = cfg.venues.get(venue_index)?;
    let fee_rate = vcfg.maker_fee_bps / 10_000.0;
    let rebate_rate = vcfg.maker_rebate_bps / 10_000.0;
    let maker_cost = (fee_rate - rebate_rate) * fair_value;
    let edge_threshold = match current.side {
        Side::Buy => cfg.mm.edge_local_min_bid_for(venue_id),
        Side::Sell => cfg.mm.edge_local_min_ask_for(venue_id),
    };
    let edge_local = match current.side {
        Side::Buy => fair_value - current.price - maker_cost,
        Side::Sell => current.price - fair_value - maker_cost,
    };
    let deficit = (edge_threshold - edge_local).max(0.0);
    Some((edge_local, edge_threshold, deficit))
}

fn blocked_by_for_within_tolerance(decision: &MmReplaceDecision) -> SupportedReplaceBlockedBy {
    let price_ratio = if decision.price_tol_ticks > 0.0 {
        decision.price_diff_ticks / decision.price_tol_ticks
    } else {
        0.0
    };
    let size_ratio = if decision.size_tol_rel > 0.0 {
        decision.size_diff_rel / decision.size_tol_rel
    } else {
        0.0
    };
    if decision.price_diff_ticks > 0.0 && price_ratio >= size_ratio {
        SupportedReplaceBlockedBy::PriceTol
    } else {
        SupportedReplaceBlockedBy::SizeTol
    }
}

fn blocked_by_for_keep_decision(decision: &MmReplaceDecision) -> SupportedReplaceBlockedBy {
    match decision.reason {
        MmReplaceReason::YoungPassiveHysteresis => SupportedReplaceBlockedBy::CurrentTooYoung,
        MmReplaceReason::TouchRiskHysteresis | MmReplaceReason::TouchRiskFastpath => {
            SupportedReplaceBlockedBy::TouchRisk
        }
        MmReplaceReason::CompressionEdgeHysteresis => SupportedReplaceBlockedBy::CompressionEdge,
        MmReplaceReason::StationaryPriceDeadbandHysteresis => {
            SupportedReplaceBlockedBy::StationaryDeadband
        }
        MmReplaceReason::PriceCadenceHysteresis => SupportedReplaceBlockedBy::PriceCadence,
        MmReplaceReason::WithinTolerance => blocked_by_for_within_tolerance(decision),
        _ => SupportedReplaceBlockedBy::PriceTol,
    }
}

/// Plan MM order actions based on desired quotes and current open orders.
///
/// Whitepaper §11: one order per venue per side, with MIN_QUOTE_LIFETIME gate
/// and price/size tolerance-based replace. Cancels are issued when a side should
/// no longer be quoted. Deterministic order: venue index ascending, then side
/// order Buy then Sell; for a replace, Cancel then Place.
pub fn plan_mm_order_actions(
    cfg: &Config,
    state: &GlobalState,
    desired_quotes: &[MmQuote],
    now_ms: TimestampMs,
    gen: &mut ActionIdGenerator,
) -> MmOrderManagementPlan {
    let mut intents = Vec::new();
    let mut decision_summary = MmOrderDecisionSummary::default();

    // Hard guard: if kill switch is active, allow only cancels (no place/replace).
    // This ensures no new risk after a hard breach while clearing existing quotes.
    if state.kill_switch {
        for (venue_index, vstate) in state.venues.iter().enumerate() {
            if let Some(cur) = &vstate.mm_open_bid {
                decision_summary.record_cancel();
                intents.push(OrderIntent::Cancel(CancelOrderIntent {
                    venue_index,
                    venue_id: vstate.id.clone(),
                    order_id: cur.order_id.clone(),
                }));
            }
            if let Some(cur) = &vstate.mm_open_ask {
                decision_summary.record_cancel();
                intents.push(OrderIntent::Cancel(CancelOrderIntent {
                    venue_index,
                    venue_id: vstate.id.clone(),
                    order_id: cur.order_id.clone(),
                }));
            }
        }
        return MmOrderManagementPlan {
            intents,
            decision_summary,
        };
    }

    let venue_count = state.venues.len();
    let mut desired_by_venue = vec![None; venue_count];
    for quote in desired_quotes {
        if quote.venue_index < venue_count {
            desired_by_venue[quote.venue_index] = Some(quote);
        }
    }

    for (venue_index, vstate) in state.venues.iter().enumerate() {
        let (best_bid, best_ask) = match (vstate.mid, vstate.spread) {
            (Some(mid), Some(spread)) => {
                let half = spread / 2.0;
                (mid - half, mid + half)
            }
            _ => (f64::NAN, f64::NAN),
        };

        let desired = desired_by_venue[venue_index];
        let (desired_bid, desired_ask) = match desired {
            Some(q) => (q.bid.as_ref(), q.ask.as_ref()),
            None => (None, None),
        };

        // Deterministic side order: Buy then Sell.
        plan_side(
            cfg,
            gen,
            venue_index,
            vstate,
            state.fair_value,
            state.q_global_tao,
            desired,
            desired_bid,
            Side::Buy,
            best_bid,
            best_ask,
            now_ms,
            &mut intents,
            &mut decision_summary,
        );
        plan_side(
            cfg,
            gen,
            venue_index,
            vstate,
            state.fair_value,
            state.q_global_tao,
            desired,
            desired_ask,
            Side::Sell,
            best_bid,
            best_ask,
            now_ms,
            &mut intents,
            &mut decision_summary,
        );
    }

    MmOrderManagementPlan {
        intents,
        decision_summary,
    }
}

fn mm_decision_id(gen: &ActionIdGenerator, venue_index: usize, side: Side) -> String {
    format!(
        "d{}_mm_v{}_{}",
        gen.tick_index(),
        venue_index,
        match side {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_side(
    cfg: &Config,
    gen: &mut ActionIdGenerator,
    venue_index: usize,
    vstate: &crate::state::VenueState,
    fair_value: Option<f64>,
    q_global_tao: f64,
    desired_quote: Option<&MmQuote>,
    desired: Option<&MmLevel>,
    side: Side,
    best_bid: f64,
    best_ask: f64,
    now_ms: TimestampMs,
    intents: &mut Vec<OrderIntent>,
    decision_summary: &mut MmOrderDecisionSummary,
) {
    let current = match side {
        Side::Buy => vstate.mm_open_bid.as_ref(),
        Side::Sell => vstate.mm_open_ask.as_ref(),
    };
    let supported_replace = is_supported_native_replace_venue(vstate.id.as_ref());
    let post_control_absence_reason = quote_side_terminal_reason(desired_quote, side);

    if desired.is_none() {
        if let Some(cur) = current {
            let supported_identity_kind = supported_native_replace_identity_kind(&vstate.id, cur);
            let current_active = ActiveMmOrder {
                venue_index,
                side,
                price: cur.price,
                size: cur.size,
                timestamp_ms: cur.timestamp_ms,
            };
            let current_age_ms = now_ms.saturating_sub(cur.timestamp_ms);
            let base_min_quote_lifetime_ms = cfg.mm.min_quote_lifetime_ms_for(&vstate.id);
            let post_control_suppression =
                supported_replace_post_control_suppression(post_control_absence_reason);
            let edge_floor_absence =
                supported_replace_edge_floor_absence(post_control_absence_reason);
            let current_safe_to_keep = best_bid.is_finite()
                && best_ask.is_finite()
                && !active_mm_order_is_dangerous(
                    cfg,
                    venue_index,
                    &current_active,
                    best_bid,
                    best_ask,
                );
            let base_suppression_grace_applied = supported_replace
                && supported_identity_kind.is_some()
                && post_control_suppression
                && current_age_ms < base_min_quote_lifetime_ms
                && current_safe_to_keep;
            let post_control_suppression_grace_ms =
                cfg.mm.post_control_suppression_grace_ms_for(&vstate.id);
            let paradex_queue_preservation_applied = supported_replace
                && vstate.id.eq_ignore_ascii_case("paradex")
                && supported_identity_kind.is_some()
                && post_control_suppression
                && current_age_ms >= base_min_quote_lifetime_ms
                && current_age_ms < post_control_suppression_grace_ms
                && current_safe_to_keep;
            let edge_under_min_grace_ms = cfg.mm.edge_under_min_grace_ms_for(&vstate.id);
            let edge_under_min_band_usd = cfg.mm.edge_under_min_band_usd_for(&vstate.id);
            let edge_floor_queue_grace_keep_reason = supported_replace
                .then_some(())
                .filter(|_| supported_identity_kind.is_some())
                .filter(|_| edge_floor_absence)
                .filter(|_| current_age_ms < edge_under_min_grace_ms)
                .filter(|_| current_safe_to_keep)
                .filter(|_| {
                    edge_under_min_band_usd.is_some_and(|band_usd| {
                        current_mm_order_edge_metrics(
                            cfg,
                            venue_index,
                            &vstate.id,
                            &current_active,
                            fair_value,
                        )
                        .is_some_and(|(_, _, deficit_usd)| deficit_usd <= band_usd + 1e-9)
                    })
                })
                .and_then(|_| {
                    if vstate.id.eq_ignore_ascii_case("paradex") {
                        Some("paradex_edge_floor_queue_grace")
                    } else if vstate.id.eq_ignore_ascii_case("hyperliquid") {
                        Some("hyperliquid_edge_floor_queue_grace")
                    } else if vstate.id.eq_ignore_ascii_case("extended") {
                        Some("extended_edge_floor_queue_grace")
                    } else {
                        None
                    }
                });
            let suppression_grace_applied = base_suppression_grace_applied
                || paradex_queue_preservation_applied
                || edge_floor_queue_grace_keep_reason.is_some();
            if suppression_grace_applied {
                let keep_reason = if paradex_queue_preservation_applied {
                    eprintln!(
                        "PARADEX_QUEUE_PRESERVATION_KEEP age_ms={} grace_ms={} base_lifetime_ms={} order_id={} client_id={}",
                        current_age_ms,
                        post_control_suppression_grace_ms,
                        base_min_quote_lifetime_ms,
                        cur.order_id,
                        cur.client_order_id.as_deref().unwrap_or("-"),
                    );
                    "paradex_queue_preservation_grace"
                } else if let Some(edge_floor_keep_reason) = edge_floor_queue_grace_keep_reason {
                    if let Some((edge_local_usd, edge_threshold_usd, deficit_usd)) =
                        current_mm_order_edge_metrics(
                            cfg,
                            venue_index,
                            &vstate.id,
                            &current_active,
                            fair_value,
                        )
                    {
                        let log_label = if vstate.id.eq_ignore_ascii_case("hyperliquid") {
                            "HYPERLIQUID_EDGE_FLOOR_QUEUE_KEEP"
                        } else if vstate.id.eq_ignore_ascii_case("extended") {
                            "EXTENDED_EDGE_FLOOR_QUEUE_KEEP"
                        } else {
                            "PARADEX_EDGE_FLOOR_QUEUE_KEEP"
                        };
                        eprintln!(
                            "{log_label} age_ms={} deficit_usd={:.6} band_usd={:.6} edge_local_usd={:.6} edge_threshold_usd={:.6} order_id={} client_id={}",
                            current_age_ms,
                            deficit_usd,
                            edge_under_min_band_usd.unwrap_or(0.0),
                            edge_local_usd,
                            edge_threshold_usd,
                            cur.order_id,
                            cur.client_order_id.as_deref().unwrap_or("-"),
                        );
                    }
                    edge_floor_keep_reason
                } else {
                    "supported_replace_suppression_grace"
                };
                let decision_id = mm_decision_id(gen, venue_index, side);
                let utility = compute_venue_utility_decision(
                    &cfg.mm,
                    q_global_tao,
                    &cfg.venues[venue_index],
                    vstate,
                    venue_utility_conversion_penalties_enabled(),
                );
                decision_summary.record_keep_without_desired(
                    &decision_id,
                    venue_index,
                    &vstate.id,
                    side,
                    fair_value,
                    q_global_tao,
                    &current_active,
                    &cur.order_id,
                    cur.client_order_id.clone(),
                    Some(utility.tier.as_str().to_string()),
                    Some(utility.reason.as_str().to_string()),
                    Some(utility.role.as_str().to_string()),
                    Some(utility.role_cap_applied),
                    keep_reason,
                );
                decision_summary.record_supported_replace_visibility(
                    &vstate.id,
                    false,
                    true,
                    Some(cur.tracking_source),
                    None,
                    SupportedReplaceVisibilityRecord {
                        decision_id,
                        venue_index,
                        venue_id: vstate.id.to_string(),
                        side: format!("{:?}", side),
                        desired_present: false,
                        current_present: true,
                        current_source: Some(cur.tracking_source.as_str().to_string()),
                        identity_kind: supported_identity_kind
                            .map(|kind| kind.as_str().to_string()),
                        current_age_ms: Some(current_age_ms),
                        native_replace_supported: true,
                        action: "keep".to_string(),
                        blocked_by: None,
                        current_order_id: Some(cur.order_id.clone()),
                        current_client_order_id: cur.client_order_id.clone(),
                        post_control_absence_reason: post_control_absence_reason
                            .map(str::to_string),
                        suppression_grace_applied: true,
                    },
                );
                return;
            }
            // Cancel when side should not be quoted.
            decision_summary.record_cancel();
            intents.push(OrderIntent::Cancel(CancelOrderIntent {
                venue_index,
                venue_id: vstate.id.clone(),
                order_id: cur.order_id.clone(),
            }));
            if supported_replace {
                let decision_id = mm_decision_id(gen, venue_index, side);
                decision_summary.record_supported_replace_visibility(
                    &vstate.id,
                    false,
                    true,
                    Some(cur.tracking_source),
                    Some(SupportedReplaceBlockedBy::DesiredSuppressed),
                    SupportedReplaceVisibilityRecord {
                        decision_id,
                        venue_index,
                        venue_id: vstate.id.to_string(),
                        side: format!("{:?}", side),
                        desired_present: false,
                        current_present: true,
                        current_source: Some(cur.tracking_source.as_str().to_string()),
                        identity_kind: supported_native_replace_identity_kind(&vstate.id, cur)
                            .map(|kind| kind.as_str().to_string()),
                        current_age_ms: Some(now_ms.saturating_sub(cur.timestamp_ms)),
                        native_replace_supported: true,
                        action: "cancel".to_string(),
                        blocked_by: Some(
                            SupportedReplaceBlockedBy::DesiredSuppressed
                                .as_str()
                                .to_string(),
                        ),
                        current_order_id: Some(cur.order_id.clone()),
                        current_client_order_id: cur.client_order_id.clone(),
                        post_control_absence_reason: post_control_absence_reason
                            .map(str::to_string),
                        suppression_grace_applied: false,
                    },
                );
            }
        }
        return;
    }

    let desired = desired.unwrap();

    // No current order -> place new.
    let Some(cur) = current else {
        let decision_id = mm_decision_id(gen, venue_index, side);
        let client_order_id = Some(gen.client_order_id(venue_index, OrderPurpose::Mm));
        let utility = compute_venue_utility_decision(
            &cfg.mm,
            q_global_tao,
            &cfg.venues[venue_index],
            vstate,
            venue_utility_conversion_penalties_enabled(),
        );
        decision_summary.record_place_decision(
            &decision_id,
            venue_index,
            &vstate.id,
            side,
            fair_value,
            q_global_tao,
            desired,
            client_order_id.clone(),
            Some(utility.tier.as_str().to_string()),
            Some(utility.reason.as_str().to_string()),
            Some(utility.role.as_str().to_string()),
            Some(utility.role_cap_applied),
        );
        let intent = OrderIntent::Place(PlaceOrderIntent {
            venue_index,
            venue_id: vstate.id.clone(),
            side,
            price: desired.price,
            size: desired.size,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id,
            phase51_target_key: None,
        });
        intents.push(intent);
        if supported_replace {
            decision_summary.record_supported_replace_visibility(
                &vstate.id,
                true,
                false,
                None,
                Some(SupportedReplaceBlockedBy::NoCurrentSameSide),
                SupportedReplaceVisibilityRecord {
                    decision_id,
                    venue_index,
                    venue_id: vstate.id.to_string(),
                    side: format!("{:?}", side),
                    desired_present: true,
                    current_present: false,
                    current_source: None,
                    identity_kind: None,
                    current_age_ms: None,
                    native_replace_supported: true,
                    action: "place".to_string(),
                    blocked_by: Some(
                        SupportedReplaceBlockedBy::NoCurrentSameSide
                            .as_str()
                            .to_string(),
                    ),
                    current_order_id: None,
                    current_client_order_id: None,
                    post_control_absence_reason: None,
                    suppression_grace_applied: false,
                },
            );
        }
        return;
    };

    let current_active = ActiveMmOrder {
        venue_index,
        side,
        price: cur.price,
        size: cur.size,
        timestamp_ms: cur.timestamp_ms,
    };
    let ctx = ShouldReplaceOrderCtx {
        cfg,
        vcfg: &cfg.venues[venue_index],
        vstate,
        q_global_tao,
        current: &current_active,
        desired_price: desired.price,
        desired_size: desired.size,
        now_ms,
        best_bid,
        best_ask,
    };
    let decision = evaluate_replace_order(ctx);
    let decision_id = mm_decision_id(gen, venue_index, side);
    let supported_identity_kind = supported_replace
        .then(|| supported_native_replace_identity_kind(&vstate.id, cur))
        .flatten();
    let client_order_id = matches!(decision.outcome, MmReplaceOutcome::Replace)
        .then(|| gen.client_order_id(venue_index, OrderPurpose::Mm));
    decision_summary.record_replace_decision(
        &decision_id,
        venue_index,
        &vstate.id,
        side,
        fair_value,
        q_global_tao,
        &current_active,
        &cur.order_id,
        desired,
        client_order_id.clone(),
        decision,
    );

    if supported_replace {
        let blocked_by = match decision.outcome {
            MmReplaceOutcome::Keep => Some(blocked_by_for_keep_decision(&decision)),
            MmReplaceOutcome::Replace if supported_identity_kind.is_none() => {
                Some(SupportedReplaceBlockedBy::IdentityMissing)
            }
            MmReplaceOutcome::Replace => None,
        };
        decision_summary.record_supported_replace_visibility(
            &vstate.id,
            true,
            true,
            Some(cur.tracking_source),
            blocked_by,
            SupportedReplaceVisibilityRecord {
                decision_id: decision_id.clone(),
                venue_index,
                venue_id: vstate.id.to_string(),
                side: format!("{:?}", side),
                desired_present: true,
                current_present: true,
                current_source: Some(cur.tracking_source.as_str().to_string()),
                identity_kind: supported_identity_kind.map(|kind| kind.as_str().to_string()),
                current_age_ms: Some(now_ms.saturating_sub(cur.timestamp_ms)),
                native_replace_supported: true,
                action: decision.outcome.as_str().to_string(),
                blocked_by: blocked_by.map(|value| value.as_str().to_string()),
                current_order_id: Some(cur.order_id.clone()),
                current_client_order_id: cur.client_order_id.clone(),
                post_control_absence_reason: None,
                suppression_grace_applied: false,
            },
        );
    }

    if matches!(decision.outcome, MmReplaceOutcome::Replace) {
        intents.push(OrderIntent::Replace(ReplaceOrderIntent {
            venue_index,
            venue_id: vstate.id.clone(),
            side,
            price: desired.price,
            size: desired.size,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: supported_native_replace_order_id(&vstate.id, cur)
                .unwrap_or_else(|| cur.order_id.clone()),
            client_order_id,
            phase51_target_key: None,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::GlobalState;
    use crate::state::MmOpenOrder;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        prior: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                prior: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.prior {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    fn mk_state_with_quote(cfg: &Config) -> GlobalState {
        let mut state = GlobalState::new(cfg);
        state.fair_value = Some(300.0);
        state.fair_value_prev = 300.0;
        for v in &mut state.venues {
            v.mid = Some(300.0);
            v.spread = Some(1.0);
        }
        state
    }

    fn mk_quote(venue_index: usize, bid: Option<(f64, f64)>, ask: Option<(f64, f64)>) -> MmQuote {
        MmQuote {
            venue_index,
            venue_id: "test".into(),
            bid: bid.map(|(p, s)| MmLevel {
                price: p,
                size: s,
                canonical_target_identity: None,
            }),
            ask: ask.map(|(p, s)| MmLevel {
                price: p,
                size: s,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }
    }

    #[test]
    fn min_quote_lifetime_respected() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 1_000;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: now_ms - (cfg.mm.min_quote_lifetime_ms - 1),
            order_id: "co_1".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((298.0, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "Should not replace under lifetime when passive"
        );
        assert_eq!(plan.decision_summary.keep_count, 1);
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("young_passive_hysteresis")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn replace_triggers_on_price_or_size_diff() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 10_000;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: now_ms - (cfg.mm.min_quote_lifetime_ms + 1),
            order_id: "co_1".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((295.0, 2.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "Replace expected");
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn plan_mm_order_actions_place_keeps_phase51_target_key_none_when_quote_identity_present() {
        let cfg = Config::default();
        let state = mk_state_with_quote(&cfg);
        let identity =
            crate::types::CanonicalTargetIdentity::from_explicit("canonical-group", "order-key")
                .expect("complete identity");
        let quotes = vec![MmQuote {
            venue_index: 0,
            venue_id: "test".into(),
            bid: Some(MmLevel {
                price: 299.0,
                size: 1.0,
                canonical_target_identity: Some(identity),
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, 1_000, &mut gen);

        assert_eq!(plan.intents.len(), 1);
        match &plan.intents[0] {
            OrderIntent::Place(place) => assert!(place.phase51_target_key.is_none()),
            other => panic!("expected place, got {other:?}"),
        }
    }

    #[test]
    fn plan_mm_order_actions_replace_keeps_phase51_target_key_none_when_quote_identity_present() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 10_000;
        let identity =
            crate::types::CanonicalTargetIdentity::from_explicit("canonical-group", "order-key")
                .expect("complete identity");

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: now_ms - (cfg.mm.min_quote_lifetime_ms + 1),
            order_id: "co_1".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![MmQuote {
            venue_index: 0,
            venue_id: "test".into(),
            bid: Some(MmLevel {
                price: 295.0,
                size: 2.0,
                canonical_target_identity: Some(identity),
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);

        assert_eq!(plan.intents.len(), 1);
        match &plan.intents[0] {
            OrderIntent::Replace(replace) => assert!(replace.phase51_target_key.is_none()),
            other => panic!("expected replace, got {other:?}"),
        }
    }

    #[test]
    fn supported_replace_visibility_records_no_current_same_side() {
        let cfg = Config::default();
        let state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: Some(MmLevel {
                price: 299.0,
                size: 1.0,
                canonical_target_identity: None,
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, 1_000, &mut gen);
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Buy")
            .expect("hyperliquid buy visibility record");
        assert_eq!(record.action, "place");
        assert_eq!(record.blocked_by.as_deref(), Some("no_current_same_side"));
        assert!(!record.current_present);
    }

    #[test]
    fn supported_replace_visibility_tracks_lighter_numeric_identity() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "lighter")
            .expect("lighter venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_bid = Some(MmOpenOrder {
            price: 300.0,
            size: 0.02,
            timestamp_ms: now_ms - 500,
            order_id: "55".to_string(),
            client_order_id: Some("55".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "lighter".into(),
            bid: Some(MmLevel {
                price: 299.0,
                size: 0.02,
                canonical_target_identity: None,
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "lighter" && record.side == "Buy")
            .expect("lighter buy visibility record");
        assert!(record.native_replace_supported);
        assert_eq!(record.identity_kind.as_deref(), Some("exchange_order_id"));
        assert_eq!(
            plan.decision_summary
                .supported_replace_opportunity_by_venue
                .get("lighter")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn extended_native_replace_visibility_is_env_gated() {
        let _guard = env_lock().lock().expect("env mutex");
        let _env_guard = EnvVarGuard::new(EXTENDED_NATIVE_REPLACE_ENABLED_ENV);
        std::env::remove_var(EXTENDED_NATIVE_REPLACE_ENABLED_ENV);

        let cfg = Config::default();
        let state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "extended")
            .expect("extended venue in config");
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "extended".into(),
            bid: Some(MmLevel {
                price: 299.0,
                size: 1.0,
                canonical_target_identity: None,
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, 1_000, &mut gen);

        assert!(plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .all(|record| record.venue_id != "extended"));
    }

    #[test]
    fn extended_native_replace_uses_client_external_id_when_enabled() {
        let _guard = env_lock().lock().expect("env mutex");
        let _env_guard = EnvVarGuard::new(EXTENDED_NATIVE_REPLACE_ENABLED_ENV);
        std::env::set_var(EXTENDED_NATIVE_REPLACE_ENABLED_ENV, "1");

        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "extended")
            .expect("extended venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_bid = Some(MmOpenOrder {
            price: 300.0,
            size: 0.02,
            timestamp_ms: now_ms - 500,
            order_id: "1784963886257016832".to_string(),
            client_order_id: Some("d0_mm_v4_buy".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "extended".into(),
            bid: Some(MmLevel {
                price: 299.0,
                size: 0.02,
                canonical_target_identity: None,
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);

        match plan.intents.first().expect("replace intent") {
            OrderIntent::Replace(replace) => {
                assert_eq!(replace.venue_id.as_ref(), "extended");
                assert_eq!(replace.order_id, "d0_mm_v4_buy");
            }
            other => panic!("expected replace, got {other:?}"),
        }
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "extended" && record.side == "Buy")
            .expect("extended buy visibility record");
        assert!(record.native_replace_supported);
        assert_eq!(record.identity_kind.as_deref(), Some("client_order_id"));
        assert_eq!(
            plan.decision_summary
                .supported_replace_opportunity_by_venue
                .get("extended")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn extended_edge_floor_queue_grace_keeps_safe_same_side_order_within_band() {
        let _guard = env_lock().lock().expect("env mutex");
        let _env_guard = EnvVarGuard::new(EXTENDED_NATIVE_REPLACE_ENABLED_ENV);
        std::env::set_var(EXTENDED_NATIVE_REPLACE_ENABLED_ENV, "1");

        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_bid_by_venue
            .insert("extended".to_string(), 0.61);
        cfg.mm.extended_edge_under_min_grace_ms = Some(3000);
        cfg.mm.extended_edge_under_min_band_usd = Some(0.04);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "extended")
            .expect("extended venue in config");
        state.venues[venue_index].spread = Some(1.0);
        let now_ms = 4_000;
        state.venues[venue_index].mm_open_bid = Some(MmOpenOrder {
            price: 299.35,
            size: 0.01,
            timestamp_ms: now_ms - 2_000,
            order_id: "1784963886257016832".to_string(),
            client_order_id: Some("d0_mm_v4_buy".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "extended".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: true,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: Some("extended_clip"),
            bid_terminal_reason: "edge_below_min",
            ask_terminal_reason: "not_quoted",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);

        assert!(
            plan.intents.is_empty(),
            "safe Extended order should keep queue when edge dip remains within the configured band"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("extended_edge_floor_queue_grace")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "extended" && record.side == "Buy")
            .expect("extended buy visibility record");
        assert_eq!(record.action, "keep");
        assert!(record.suppression_grace_applied);
        assert_eq!(
            record.post_control_absence_reason.as_deref(),
            Some("edge_below_min")
        );
    }

    #[test]
    fn supported_replace_budget_suppression_grace_keeps_paradex_same_side_order() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        let now_ms = 1_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 0.01,
            timestamp_ms: now_ms - 250,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "suppression grace should hold supported same-side order instead of canceling"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("supported_replace_suppression_grace")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "paradex" && record.side == "Sell")
            .expect("paradex sell visibility record");
        assert_eq!(record.action, "keep");
        assert_eq!(
            record.post_control_absence_reason.as_deref(),
            Some("projected_mm_budget_suppressed")
        );
        assert!(record.suppression_grace_applied);
        assert_eq!(record.blocked_by, None);
    }

    #[test]
    fn supported_replace_budget_suppression_grace_expires_after_lifetime() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 0.01,
            timestamp_ms: now_ms - (cfg.mm.min_quote_lifetime_ms_for("paradex") + 1),
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "aged order should cancel once grace expires"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "paradex" && record.side == "Sell")
            .expect("paradex sell visibility record");
        assert_eq!(record.action, "cancel");
        assert!(!record.suppression_grace_applied);
        assert_eq!(
            record.post_control_absence_reason.as_deref(),
            Some("projected_mm_budget_suppressed")
        );
    }

    #[test]
    fn paradex_queue_preservation_grace_keeps_safe_same_side_order_beyond_base_lifetime() {
        let mut cfg = Config::default();
        cfg.mm
            .min_quote_lifetime_ms_by_venue
            .insert("paradex".to_string(), 300);
        cfg.mm.paradex_post_control_suppression_grace_ms = Some(900);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "safe Paradex order should be preserved during the extended queue-preservation grace"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_queue_preservation_grace")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "paradex" && record.side == "Sell")
            .expect("paradex sell visibility record");
        assert_eq!(record.action, "keep");
        assert!(record.suppression_grace_applied);
        assert_eq!(record.blocked_by, None);
    }

    #[test]
    fn paradex_queue_preservation_grace_expires_after_extended_window() {
        let mut cfg = Config::default();
        cfg.mm
            .min_quote_lifetime_ms_by_venue
            .insert("paradex".to_string(), 300);
        cfg.mm.paradex_post_control_suppression_grace_ms = Some(900);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 0.01,
            timestamp_ms: now_ms - 901,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "aged Paradex order should cancel after the extended queue-preservation grace"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_queue_preservation_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn paradex_queue_preservation_grace_does_not_keep_dangerous_order() {
        let mut cfg = Config::default();
        cfg.mm
            .min_quote_lifetime_ms_by_venue
            .insert("paradex".to_string(), 300);
        cfg.mm.paradex_post_control_suppression_grace_ms = Some(900);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.4,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "dangerous Paradex order should cancel immediately"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_queue_preservation_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn paradex_queue_preservation_grace_does_not_change_hyperliquid_suppression_expiry() {
        let mut cfg = Config::default();
        cfg.mm.paradex_post_control_suppression_grace_ms = Some(900);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "12345".to_string(),
            client_order_id: Some("0x1234567890abcdef1234567890abcdef".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "projected_mm_budget_suppressed",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "non-Paradex venue should still cancel once the base suppression grace expires"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_queue_preservation_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn paradex_edge_floor_queue_grace_keeps_safe_same_side_order_within_band() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("paradex".to_string(), 0.06);
        cfg.mm.paradex_edge_under_min_grace_ms = Some(900);
        cfg.mm.paradex_edge_under_min_band_usd = Some(0.02);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.11,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "safe Paradex order should keep queue when edge dip remains within the configured band"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_edge_floor_queue_grace")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "paradex" && record.side == "Sell")
            .expect("paradex sell visibility record");
        assert_eq!(record.action, "keep");
        assert!(record.suppression_grace_applied);
        assert_eq!(
            record.post_control_absence_reason.as_deref(),
            Some("edge_below_min")
        );
    }

    #[test]
    fn paradex_edge_floor_queue_grace_expires_after_grace_window() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("paradex".to_string(), 0.06);
        cfg.mm.paradex_edge_under_min_grace_ms = Some(900);
        cfg.mm.paradex_edge_under_min_band_usd = Some(0.02);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.11,
            size: 0.01,
            timestamp_ms: now_ms - 901,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "aged Paradex order should cancel after the edge-under-min grace"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn paradex_edge_floor_queue_grace_requires_small_deficit() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("paradex".to_string(), 0.06);
        cfg.mm.paradex_edge_under_min_grace_ms = Some(900);
        cfg.mm.paradex_edge_under_min_band_usd = Some(0.02);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.08,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Paradex queue hold must not trigger when the edge deficit exceeds the configured band"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn paradex_edge_floor_queue_grace_does_not_keep_dangerous_order() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("paradex".to_string(), 0.06);
        cfg.mm.paradex_edge_under_min_grace_ms = Some(900);
        cfg.mm.paradex_edge_under_min_band_usd = Some(0.02);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "paradex")
            .expect("paradex venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 2_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.055,
            size: 0.01,
            timestamp_ms: now_ms - 600,
            order_id: "1774989134070201709267840000".to_string(),
            client_order_id: Some("co_test_paradex".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "paradex".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "dangerous Paradex order should cancel immediately even within the under-edge grace window");
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("paradex_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn hyperliquid_edge_floor_queue_grace_keeps_safe_same_side_order_within_band() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("hyperliquid".to_string(), 0.06);
        cfg.mm.hyperliquid_edge_under_min_grace_ms = Some(3000);
        cfg.mm.hyperliquid_edge_under_min_band_usd = Some(0.04);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 4_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.11,
            size: 0.01,
            timestamp_ms: now_ms - 2_000,
            order_id: "12345".to_string(),
            client_order_id: Some("0x1234567890abcdef1234567890abcdef".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: true,
            touch_mode_kind: Some("hyperliquid_clip"),
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "safe Hyperliquid order should keep queue when edge dip remains within the configured band"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("hyperliquid_edge_floor_queue_grace")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Sell")
            .expect("hyperliquid sell visibility record");
        assert_eq!(record.action, "keep");
        assert!(record.suppression_grace_applied);
        assert_eq!(
            record.post_control_absence_reason.as_deref(),
            Some("edge_below_min")
        );
    }

    #[test]
    fn hyperliquid_edge_floor_queue_grace_expires_after_grace_window() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("hyperliquid".to_string(), 0.06);
        cfg.mm.hyperliquid_edge_under_min_grace_ms = Some(3000);
        cfg.mm.hyperliquid_edge_under_min_band_usd = Some(0.04);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 4_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.11,
            size: 0.01,
            timestamp_ms: now_ms - 3_001,
            order_id: "12345".to_string(),
            client_order_id: Some("0x1234567890abcdef1234567890abcdef".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: true,
            touch_mode_kind: Some("hyperliquid_clip"),
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "aged Hyperliquid order should cancel after the edge-under-min grace"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("hyperliquid_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn hyperliquid_edge_floor_queue_grace_requires_small_deficit() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("hyperliquid".to_string(), 0.06);
        cfg.mm.hyperliquid_edge_under_min_grace_ms = Some(3000);
        cfg.mm.hyperliquid_edge_under_min_band_usd = Some(0.02);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 4_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.01,
            size: 0.01,
            timestamp_ms: now_ms - 2_000,
            order_id: "12345".to_string(),
            client_order_id: Some("0x1234567890abcdef1234567890abcdef".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: true,
            touch_mode_kind: Some("hyperliquid_clip"),
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Hyperliquid queue hold must not trigger when the edge deficit exceeds the configured band"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("hyperliquid_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn hyperliquid_edge_floor_queue_grace_does_not_keep_dangerous_order() {
        let mut cfg = Config::default();
        cfg.mm
            .edge_local_min_ask_by_venue
            .insert("hyperliquid".to_string(), 0.06);
        cfg.mm.hyperliquid_edge_under_min_grace_ms = Some(3000);
        cfg.mm.hyperliquid_edge_under_min_band_usd = Some(0.04);
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        state.venues[venue_index].spread = Some(0.1);
        let now_ms = 4_000;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 300.055,
            size: 0.01,
            timestamp_ms: now_ms - 2_000,
            order_id: "12345".to_string(),
            client_order_id: Some("0x1234567890abcdef1234567890abcdef".to_string()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: None,
            ask: None,
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: true,
            touch_mode_kind: Some("hyperliquid_clip"),
            bid_terminal_reason: "not_quoted",
            ask_terminal_reason: "edge_below_min",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "dangerous Hyperliquid order should cancel immediately even within the under-edge grace window"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Cancel(_)));
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("hyperliquid_edge_floor_queue_grace")
                .copied(),
            None
        );
    }

    #[test]
    fn hyperliquid_replace_prefers_cloid_identity_for_native_modify() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 10_000;
        let cloid = "0x1234567890abcdef1234567890abcdef".to_string();
        state.venues[venue_index].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: now_ms - (cfg.mm.min_quote_lifetime_ms + 1),
            order_id: "12345".to_string(),
            client_order_id: Some(cloid.clone()),
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "hyperliquid".into(),
            bid: Some(MmLevel {
                price: 295.0,
                size: 2.0,
                canonical_target_identity: None,
            }),
            ask: None,
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        let replace = match &plan.intents[0] {
            OrderIntent::Replace(replace) => replace,
            other => panic!("expected replace, got {other:?}"),
        };
        assert_eq!(replace.order_id, cloid);
        let record = plan
            .decision_summary
            .supported_replace_visibility_records
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Buy")
            .expect("hyperliquid buy visibility record");
        assert_eq!(record.identity_kind.as_deref(), Some("client_order_id"));
        assert_eq!(record.action, "replace");
        assert_eq!(record.blocked_by, None);
    }

    #[test]
    fn cancel_when_side_not_quoted() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 5_000;

        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 1.0,
            timestamp_ms: now_ms - 10_000,
            order_id: "co_2".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((299.0, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 2, "Cancel ask + place bid expected");
        assert!(plan
            .intents
            .iter()
            .any(|i| matches!(i, OrderIntent::Cancel(_))));
    }

    #[test]
    fn kill_switch_blocks_mm_order_management() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.kill_switch = true;
        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 99.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_ks_bid".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 101.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_ks_ask".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        // Provide a desired quote to ensure the guard is what blocks actions.
        let quotes = vec![MmQuote {
            venue_index: 0,
            venue_id: "test".into(),
            bid: Some(MmLevel {
                price: 100.0,
                size: 1.0,
                canonical_target_identity: None,
            }),
            ask: Some(MmLevel {
                price: 101.0,
                size: 1.0,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: false,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];

        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, 0, &mut gen);

        assert!(
            plan.intents
                .iter()
                .all(|i| matches!(i, OrderIntent::Cancel(_))),
            "kill_switch should only allow cancels"
        );
        assert!(
            !plan.intents.is_empty(),
            "kill_switch should cancel open orders"
        );
    }

    #[test]
    fn dangerous_offside_bypasses_lifetime() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 1_000;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 300.0, // non-passive (>= best_bid - tick)
            size: 1.0,
            timestamp_ms: now_ms - 1,
            order_id: "co_3".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((298.0, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Dangerously offside should allow replace"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn aster_touch_offside_fastpath_keeps_young_one_tick_touch_quote() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "aster")
            .expect("aster venue in config");
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
        let now_ms = base_lifetime.saturating_sub(1);
        let mid = 2_131.30;
        let tick = cfg.venues[venue_index].tick_size.max(1e-6);
        let best_bid = mid - (1.5 * tick);
        let best_ask = mid + (1.5 * tick);

        state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
        state.venues[venue_index].spread = Some(best_ask - best_bid);
        state.venues[venue_index].toxicity = 0.0;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: best_ask + (0.5 * tick),
            size: 0.01,
            timestamp_ms: 0,
            order_id: "co_aster_touch_ask".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "aster".into(),
            bid: None,
            ask: Some(MmLevel {
                price: best_ask + tick,
                size: 0.01,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "Aster one-tick touch-local ask should keep queue instead of replacing for dangerous_offside"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("aster_touch_offside_fastpath")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.aster_touch_offside_fastpath_count, 1);
        assert_eq!(plan.decision_summary.replace_count, 0);
    }

    #[test]
    fn aster_touch_offside_fastpath_records_quote_too_old_nearmiss() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "aster")
            .expect("aster venue in config");
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
        let now_ms = base_lifetime + 100;
        let mid = 2_131.30;
        let tick = cfg.venues[venue_index].tick_size.max(1e-6);
        let best_bid = mid - (1.5 * tick);
        let best_ask = mid + (1.5 * tick);

        state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
        state.venues[venue_index].spread = Some(best_ask - best_bid);
        state.venues[venue_index].toxicity = 0.0;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: best_ask + (0.5 * tick),
            size: 0.01,
            timestamp_ms: 0,
            order_id: "co_aster_touch_ask_old".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "aster".into(),
            bid: None,
            ask: Some(MmLevel {
                price: best_ask + tick,
                size: 0.01,
                canonical_target_identity: None,
            }),
            generated_spread_cap_applied: true,
            generated_spread_cap_bid_suppressed: false,
            generated_spread_cap_ask_suppressed: false,
            touch_mode_kind: None,
            bid_terminal_reason: "active",
            ask_terminal_reason: "active",
        }];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents
                .iter()
                .any(|intent| matches!(intent, OrderIntent::Replace(_))),
            "older Aster touch-local ask should fall back to dangerous_offside replace"
        );
        assert_eq!(
            plan.decision_summary
                .aster_touch_offside_nearmiss_by_reason
                .get("quote_too_old")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.aster_touch_offside_fastpath_count, 0);
    }

    #[test]
    fn reduced_utility_extends_quote_hysteresis_for_non_reducing_side() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.venues[0].utility.mm_fill_credit_ewma = 0.0;
        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_reduced_bid".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((299.015, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "reduced-utility worsening-side quote should keep queue position under extended hysteresis"
        );
    }

    #[test]
    fn probationary_role_records_small_fill_rehab_mode() {
        let mut cfg = Config::default();
        cfg.mm.venue_role_by_venue.insert(
            cfg.venues[0].id.clone(),
            crate::config::MmVenueRole::Probationary,
        );
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_probationary_bid".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((299.015, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "probationary venue should keep queue under reduced-style hysteresis"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_venue_role
                .get("probationary")
                .copied(),
            Some(1)
        );
        let record = &plan.decision_summary.replace_decisions[0];
        assert_eq!(record.venue_role, "probationary");
        assert!(record.role_cap_applied);
    }

    #[test]
    fn reducing_side_bypasses_extended_hysteresis() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.q_global_tao = 5.0;
        state.venues[0].utility.mm_fill_credit_ewma = 0.0;
        state.venues[0].utility.mm_fillless_ack_pressure = 25.0;
        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 301.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_reduced_ask".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, None, Some((300.985, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "inventory-reducing ask should still refresh"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn healthy_worsening_side_growth_uses_slower_hysteresis() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_full_bid_growth".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((299.015, 1.15)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "healthy worsening-side quote growth should keep queue position under slower hysteresis"
        );
        assert_eq!(plan.decision_summary.keep_count, 1);
    }

    #[test]
    fn healthy_worsening_side_derisking_stays_fast() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_full_bid_derisk".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((298.985, 0.85)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "healthy worsening-side de-risking should still refresh promptly"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn passive_price_only_move_can_be_held_by_price_cadence() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 700;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.00,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_price_cadence".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, Some((299.05, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "price-only passive reprice should be held by cadence on a wide-spread venue"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("price_cadence_hysteresis")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn inventory_reducing_price_move_bypasses_price_cadence() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 200;

        state.q_global_tao = 5.0;
        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 301.00,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_price_cadence_bypass".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(0, None, Some((300.95, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "inventory-reducing price reprice should bypass cadence"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn hyperliquid_sell_tighten_into_compressed_spread_uses_compression_guard() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 1_000;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(1.0);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 101.02,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_compression".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.91, 1.08)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "Hyperliquid sell tightening into a compressed spread should be held"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("compression_edge_hysteresis")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.compression_edge_hysteresis_count, 1);
        assert_eq!(plan.decision_summary.compression_edge_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_compressed_spread_holds_small_derisk_size_change() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 1_000;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(1.0);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 101.02,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_derisk_compression".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.91, 0.86)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "Hyperliquid sell compression guard should hold a small derisk size change"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("compression_edge_hysteresis")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.compression_edge_hysteresis_count, 1);
        assert_eq!(plan.decision_summary.compression_edge_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_tighten_without_compression_still_reprices() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 1_600;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 101.02,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_no_compression".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.91, 1.08)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Hyperliquid sell without spread compression should still reprice"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
        assert_eq!(plan.decision_summary.compression_edge_hysteresis_count, 0);
        assert_eq!(plan.decision_summary.compression_edge_nearmiss_count, 1);
        assert_eq!(
            plan.decision_summary
                .compression_edge_nearmiss_by_reason
                .get("not_compressed")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn hyperliquid_sell_tighten_into_thin_touch_uses_touch_risk_hysteresis() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("hyperliquid");
        let now_ms = base_lifetime.saturating_mul(2) + 200;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.50);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.70,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_touch_hold".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.51, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "Hyperliquid sell should hold instead of tightening into a thin touch"
        );
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("touch_risk_hysteresis")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.touch_risk_hysteresis_count, 1);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_thin_quote_with_rising_touch_bypasses_young_hysteresis() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 250;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.48);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.52,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_touch_fastpath".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.60, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Hyperliquid sell should reprice early when touch rises into a thin quote"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
        assert_eq!(
            plan.decision_summary
                .replace_by_reason
                .get("touch_risk_fastpath")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.touch_risk_fastpath_count, 1);
        assert_eq!(plan.decision_summary.touch_risk_size_band_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_small_size_drift_can_use_touch_risk_fastpath() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 250;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.48);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.52,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_touch_fastpath_size_band".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.60, 1.15)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "small Hyperliquid sell size drift should still allow touch-risk fastpath"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
        assert_eq!(
            plan.decision_summary
                .replace_by_reason
                .get("touch_risk_fastpath")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.touch_risk_fastpath_count, 1);
        assert_eq!(plan.decision_summary.touch_risk_size_band_count, 1);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_three_tick_clearance_now_bypasses_young_hysteresis() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 250;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.48);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.53,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_touch_fastpath_3tick".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.61, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "Hyperliquid sell with three-tick clearance should now reprice early"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
        assert_eq!(
            plan.decision_summary
                .replace_by_reason
                .get("touch_risk_fastpath")
                .copied(),
            Some(1)
        );
        assert_eq!(plan.decision_summary.touch_risk_fastpath_count, 1);
        assert_eq!(plan.decision_summary.touch_risk_size_band_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 0);
    }

    #[test]
    fn hyperliquid_sell_touch_risk_nearmiss_is_recorded() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("hyperliquid");
        let now_ms = base_lifetime.saturating_mul(2) + 200;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = None;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.52,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_touch_nearmiss".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.60, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(
            plan.intents.len(),
            1,
            "expected replace when touch fastpath cannot fire"
        );
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
        assert_eq!(plan.decision_summary.touch_risk_fastpath_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_size_band_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 1);
        assert_eq!(
            plan.decision_summary
                .touch_risk_nearmiss_by_reason
                .get("no_prev_best_ask")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn hyperliquid_sell_large_size_drift_still_blocks_touch_risk_path() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 250;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.48);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.52,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_touch_size_block".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.60, 1.5)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.decision_summary.touch_risk_fastpath_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_size_band_count, 0);
        assert_eq!(plan.decision_summary.touch_risk_nearmiss_count, 1);
        assert_eq!(
            plan.decision_summary
                .touch_risk_nearmiss_by_reason
                .get("size_exceeds_block")
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn hyperliquid_sell_price_move_forensics_are_recorded_on_decision_records() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 2_000;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.49);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.70,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_price_forensics".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.72, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.decision_summary.replace_count, 1);
        let record = plan
            .decision_summary
            .replace_decisions
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Sell")
            .expect("hyperliquid sell decision record");
        assert_eq!(record.reason, "price_move");
        assert!(record
            .hyperliquid_sell_price_excess_ticks
            .is_some_and(|value| (value - 1.0).abs() < 1e-9));
        assert!(record
            .hyperliquid_sell_best_ask_motion_ticks
            .is_some_and(|value| (value - 1.0).abs() < 1e-9));
        assert!(record
            .hyperliquid_sell_current_clearance_ticks
            .is_some_and(|value| (value - 20.0).abs() < 1e-9));
        assert!(record
            .hyperliquid_sell_desired_clearance_ticks
            .is_some_and(|value| (value - 22.0).abs() < 1e-9));
        assert_eq!(record.hyperliquid_sell_price_reprice_remaining_ms, Some(0));
    }

    #[test]
    fn hyperliquid_sell_stationary_touch_small_old_price_move_is_held() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 2_000;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.50);
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.70,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_stationary_deadband".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.72, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "small stationary Hyperliquid sell widens should be held by the extra deadband"
        );
        assert_eq!(plan.decision_summary.replace_count, 0);
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("stationary_price_deadband_hysteresis")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .replace_decisions
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Sell")
            .expect("hyperliquid sell decision record");
        assert_eq!(record.outcome, "keep");
        assert_eq!(record.reason, "stationary_price_deadband_hysteresis");
        assert!(record
            .hyperliquid_sell_best_ask_motion_ticks
            .is_some_and(|value| value.abs() < 1e-9));
        assert!(record
            .hyperliquid_sell_price_excess_ticks
            .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    }

    #[test]
    fn hyperliquid_sell_reduced_adverse_selection_still_uses_stationary_deadband() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .expect("hyperliquid venue in config");
        let now_ms = 2_000;

        state.venues[venue_index].mid = Some(100.25);
        state.venues[venue_index].spread = Some(0.5);
        state.venues[venue_index].prev_spread = Some(0.5);
        state.venues[venue_index].prev_best_ask = Some(100.50);
        state.venues[venue_index].toxicity = 0.7;
        state.venues[venue_index].mm_open_ask = Some(MmOpenOrder {
            price: 100.70,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_hl_sell_stationary_deadband_reduced".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });

        let quotes = vec![mk_quote(venue_index, None, Some((100.72, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(plan.intents.is_empty());
        assert_eq!(
            plan.decision_summary
                .keep_by_reason
                .get("stationary_price_deadband_hysteresis")
                .copied(),
            Some(1)
        );
        let record = plan
            .decision_summary
            .replace_decisions
            .iter()
            .find(|record| record.venue_id == "hyperliquid" && record.side == "Sell")
            .expect("hyperliquid sell decision record");
        assert_eq!(record.outcome, "keep");
        assert_eq!(record.reason, "stationary_price_deadband_hysteresis");
        assert_eq!(record.utility_tier, "reduced");
        assert_eq!(record.utility_reason, "adverse_selection");
    }

    #[test]
    fn deterministic_ordering_cancels_before_places() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let now_ms = 10_000;

        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: now_ms - 10_000,
            order_id: "co_4".to_string(),
            client_order_id: None,
            tracking_source: crate::state::MmOpenTrackingSource::OpenSnapshot,
        });
        let quotes = vec![mk_quote(0, Some((295.0, 2.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "Replace intent expected");
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }
}
