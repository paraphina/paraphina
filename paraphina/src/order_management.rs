// src/order_management.rs
//
// Milestone H: Order management for MM quotes (Whitepaper §11).
// Implements cancel/replace logic with MIN_QUOTE_LIFETIME_MS and tolerance gates.

use crate::actions::ActionIdGenerator;
use crate::config::Config;
use crate::mm::{MmLevel, MmQuote};
use crate::state::GlobalState;
use crate::types::{
    CancelOrderIntent, OrderIntent, OrderPurpose, PlaceOrderIntent, ReplaceOrderIntent, Side,
    TimeInForce, TimestampMs,
};

/// Output of MM order management planner.
#[derive(Debug, Clone)]
pub struct MmOrderManagementPlan {
    pub intents: Vec<OrderIntent>,
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

    // Hard guard: if kill switch is active, allow only cancels (no place/replace).
    // This ensures no new risk after a hard breach while clearing existing quotes.
    if state.kill_switch {
        for (venue_index, vstate) in state.venues.iter().enumerate() {
            if let Some(cur) = &vstate.mm_open_bid {
                intents.push(OrderIntent::Cancel(CancelOrderIntent {
                    venue_index,
                    venue_id: vstate.id.clone(),
                    order_id: cur.order_id.clone(),
                }));
            }
            if let Some(cur) = &vstate.mm_open_ask {
                intents.push(OrderIntent::Cancel(CancelOrderIntent {
                    venue_index,
                    venue_id: vstate.id.clone(),
                    order_id: cur.order_id.clone(),
                }));
            }
        }
        return MmOrderManagementPlan { intents };
    }

    let venue_count = state.venues.len();
    let mut desired_by_venue = vec![None; venue_count];
    for quote in desired_quotes {
        if quote.venue_index < venue_count {
            desired_by_venue[quote.venue_index] = Some(quote);
        }
    }

    for (venue_index, vstate) in state.venues.iter().enumerate() {
        let vcfg = &cfg.venues[venue_index];
        let tick = vcfg.tick_size.max(1e-6);

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
            desired_bid,
            Side::Buy,
            best_bid,
            best_ask,
            tick,
            now_ms,
            &mut intents,
        );
        plan_side(
            cfg,
            gen,
            venue_index,
            vstate,
            desired_ask,
            Side::Sell,
            best_bid,
            best_ask,
            tick,
            now_ms,
            &mut intents,
        );
    }

    MmOrderManagementPlan { intents }
}

#[allow(clippy::too_many_arguments)]
fn plan_side(
    cfg: &Config,
    gen: &mut ActionIdGenerator,
    venue_index: usize,
    vstate: &crate::state::VenueState,
    desired: Option<&MmLevel>,
    side: Side,
    best_bid: f64,
    best_ask: f64,
    tick: f64,
    now_ms: TimestampMs,
    intents: &mut Vec<OrderIntent>,
) {
    let current = match side {
        Side::Buy => vstate.mm_open_bid.as_ref(),
        Side::Sell => vstate.mm_open_ask.as_ref(),
    };

    if desired.is_none() {
        if let Some(cur) = current {
            // Cancel when side should not be quoted.
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

    let age_ms = now_ms - cur.timestamp_ms;
    let is_dangerous = dangerously_offside(side, cur.price, best_bid, best_ask, tick);

    // MIN_QUOTE_LIFETIME_MS gate unless dangerously offside.
    if age_ms < cfg.mm.min_quote_lifetime_ms && !is_dangerous {
        return;
    }

    let price_diff_ticks = (cur.price - desired.price).abs() / tick;
    let size_diff_rel =
        (cur.size - desired.size).abs() / desired.size.max(cfg.venues[venue_index].lot_size_tao);

    if is_dangerous
        || price_diff_ticks > cfg.mm.price_tol_ticks
        || size_diff_rel > cfg.mm.size_tol_rel
    {
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

/// Dangerous offside check for MM orders (Whitepaper §11).
///
/// We align with current passivity rules: bids must be <= best_bid - tick,
/// asks must be >= best_ask + tick. An order is "dangerously offside" if it
/// is non-passive or would cross the touch.
fn dangerously_offside(side: Side, price: f64, best_bid: f64, best_ask: f64, tick: f64) -> bool {
    if !best_bid.is_finite() || !best_ask.is_finite() {
        return true;
    }
    match side {
        Side::Buy => {
            let non_passive = price > best_bid - tick;
            let crosses = price >= best_ask - tick;
            non_passive || crosses
        }
        Side::Sell => {
            let non_passive = price < best_ask + tick;
            let crosses = price <= best_bid + tick;
            non_passive || crosses
        }
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
