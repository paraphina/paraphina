// src/order_management.rs
//
// Milestone H: Order management for MM quotes (Whitepaper §11).
// Implements cancel/replace logic with MIN_QUOTE_LIFETIME_MS and tolerance gates.

use std::collections::BTreeMap;

use crate::actions::ActionIdGenerator;
use crate::config::Config;
use crate::mm::{
    evaluate_replace_order, ActiveMmOrder, MmLevel, MmQuote, MmReplaceDecision,
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
    pub keep_by_reason: BTreeMap<String, u64>,
    pub replace_by_reason: BTreeMap<String, u64>,
    pub keep_by_utility_tier: BTreeMap<String, u64>,
    pub replace_by_utility_tier: BTreeMap<String, u64>,
    pub replace_decisions: Vec<MmReplaceDecisionRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MmReplaceDecisionRecord {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: String,
    pub outcome: String,
    pub reason: String,
    pub utility_tier: String,
    pub utility_reason: String,
    pub inventory_reducing: bool,
    pub age_ms: TimestampMs,
    pub min_quote_lifetime_ms: TimestampMs,
    pub price_diff_ticks: f64,
    pub price_tol_ticks: f64,
    pub size_diff_rel: f64,
    pub size_tol_rel: f64,
}

impl MmOrderDecisionSummary {
    fn record_replace_decision(
        &mut self,
        venue_index: usize,
        venue_id: &str,
        side: Side,
        decision: MmReplaceDecision,
    ) {
        let tier_key = decision.utility_tier.as_str().to_string();
        let reason_key = decision.reason.as_str().to_string();
        match decision.outcome {
            MmReplaceOutcome::Keep => {
                self.keep_count += 1;
                *self.keep_by_reason.entry(reason_key.clone()).or_insert(0) += 1;
                *self.keep_by_utility_tier.entry(tier_key.clone()).or_insert(0) += 1;
            }
            MmReplaceOutcome::Replace => {
                self.replace_count += 1;
                *self.replace_by_reason.entry(reason_key.clone()).or_insert(0) += 1;
                *self
                    .replace_by_utility_tier
                    .entry(tier_key.clone())
                    .or_insert(0) += 1;
            }
        }
        self.replace_decisions.push(MmReplaceDecisionRecord {
            venue_index,
            venue_id: venue_id.to_string(),
            side: format!("{:?}", side),
            outcome: decision.outcome.as_str().to_string(),
            reason: reason_key,
            utility_tier: tier_key,
            utility_reason: decision.utility_reason.as_str().to_string(),
            inventory_reducing: decision.inventory_reducing,
            age_ms: decision.age_ms,
            min_quote_lifetime_ms: decision.min_quote_lifetime_ms,
            price_diff_ticks: decision.price_diff_ticks,
            price_tol_ticks: decision.price_tol_ticks,
            size_diff_rel: decision.size_diff_rel,
            size_tol_rel: decision.size_tol_rel,
        });
    }

    fn record_place(&mut self) {
        self.place_count += 1;
    }

    fn record_cancel(&mut self) {
        self.cancel_count += 1;
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

#[allow(clippy::too_many_arguments)]
fn plan_side(
    cfg: &Config,
    gen: &mut ActionIdGenerator,
    venue_index: usize,
    vstate: &crate::state::VenueState,
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
        decision_summary.record_place();
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
            client_order_id: Some(gen.client_order_id(venue_index, OrderPurpose::Mm)),
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
    decision_summary.record_replace_decision(venue_index, &vstate.id, side, decision);

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
            client_order_id: Some(gen.client_order_id(venue_index, OrderPurpose::Mm)),
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
    fn reduced_utility_extends_quote_hysteresis_for_non_reducing_side() {
        let cfg = Config::default();
        let mut state = mk_state_with_quote(&cfg);
        let base_lifetime = cfg.mm.min_quote_lifetime_ms_for(&state.venues[0].id);
        let now_ms = base_lifetime + 100;

        state.venues[0].utility.mm_fill_credit_ewma = 0.0;
        state.venues[0].utility.mm_fillless_ack_pressure = 25.0;
        state.venues[0].mm_open_bid = Some(MmOpenOrder {
            price: 299.0,
            size: 1.0,
            timestamp_ms: 0,
            order_id: "co_reduced_bid".to_string(),
        });

        let quotes = vec![mk_quote(0, Some((298.985, 1.0)), None)];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert!(
            plan.intents.is_empty(),
            "reduced-utility worsening-side quote should keep queue position under extended hysteresis"
        );
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

        let quotes = vec![mk_quote(0, None, Some((301.015, 1.0)))];
        let mut gen = ActionIdGenerator::new(0);
        let plan = plan_mm_order_actions(&cfg, &state, &quotes, now_ms, &mut gen);
        assert_eq!(plan.intents.len(), 1, "inventory-reducing ask should still refresh");
        assert!(matches!(plan.intents[0], OrderIntent::Replace(_)));
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
        assert_eq!(plan.decision_summary.replace_count, 1);
        assert_eq!(
            plan.decision_summary
                .replace_by_reason
                .get("price_and_size_move")
                .copied(),
            Some(1)
        );
    }
}
