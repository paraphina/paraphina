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
    MmReplaceOutcome, ShouldReplaceOrderCtx,
};
use crate::state::GlobalState;
use crate::types::{
    CancelOrderIntent, OrderIntent, OrderPurpose, PlaceOrderIntent, ReplaceOrderIntent, Side,
    TimeInForce, TimestampMs,
};
use serde::Serialize;

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
    pub decision_records: Vec<MmDecisionRecord>,
    pub replace_decisions: Vec<MmReplaceDecisionRecord>,
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

impl MmOrderDecisionSummary {
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
            if let Some(client_order_id) = by_slot
                .get(&(
                    record.venue_index,
                    record.side.clone(),
                    record.outcome.clone(),
                ))
                .cloned()
            {
                record.client_order_id = Some(client_order_id);
            }
        }
        for record in &mut self.replace_decisions {
            if let Some(client_order_id) = by_slot
                .get(&(
                    record.venue_index,
                    record.side.clone(),
                    record.outcome.clone(),
                ))
                .cloned()
            {
                record.client_order_id = Some(client_order_id);
            }
        }
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

    if desired.is_none() {
        if let Some(cur) = current {
            // Cancel when side should not be quoted.
            decision_summary.record_cancel();
            intents.push(OrderIntent::Cancel(CancelOrderIntent {
                venue_index,
                venue_id: vstate.id.clone(),
                order_id: cur.order_id.clone(),
            }));
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
        });
        intents.push(intent);
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
            order_id: cur.order_id.clone(),
            client_order_id,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::GlobalState;
    use crate::state::MmOpenOrder;

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
            bid: bid.map(|(p, s)| MmLevel { price: p, size: s }),
            ask: ask.map(|(p, s)| MmLevel { price: p, size: s }),
            generated_spread_cap_applied: false,
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
        });

        let quotes = vec![mk_quote(0, Some((295.0, 2.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "Replace expected");
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
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
        });
        state.venues[0].mm_open_ask = Some(MmOpenOrder {
            price: 101.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_ks_ask".to_string(),
        });

        // Provide a desired quote to ensure the guard is what blocks actions.
        let quotes = vec![MmQuote {
            venue_index: 0,
            venue_id: "test".into(),
            bid: Some(MmLevel {
                price: 100.0,
                size: 1.0,
            }),
            ask: Some(MmLevel {
                price: 101.0,
                size: 1.0,
            }),
            generated_spread_cap_applied: false,
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
        });

        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "aster".into(),
            bid: None,
            ask: Some(MmLevel {
                price: best_ask + tick,
                size: 0.01,
            }),
            generated_spread_cap_applied: true,
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
        });

        let quotes = vec![MmQuote {
            venue_index,
            venue_id: "aster".into(),
            bid: None,
            ask: Some(MmLevel {
                price: best_ask + tick,
                size: 0.01,
            }),
            generated_spread_cap_applied: true,
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
        });
        let quotes = vec![mk_quote(0, Some((295.0, 2.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "Replace intent expected");
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
    }
}
