// src/execution_events.rs
//
// Execution event applier: the single place where execution events mutate state.

use crate::state::{GlobalState, MmOpenOrder, OpenOrderRecord};
use crate::types::{ExecutionEvent, OrderPurpose, Side, TimestampMs};

/// Apply execution events to state and return any fills for downstream batching.
pub fn apply_execution_events(
    state: &mut GlobalState,
    events: &[ExecutionEvent],
    now_ms: TimestampMs,
) -> Vec<crate::types::FillEvent> {
    let mut fills = Vec::new();

    for event in events {
        match event {
            ExecutionEvent::BookUpdate(book) => {
                if let Some(v) = state.venues.get_mut(book.venue_index) {
                    v.mid = Some(book.mid);
                    v.spread = Some(book.spread);
                    v.depth_near_mid = book.depth_near_mid;
                    v.last_mid_update_ms = Some(book.timestamp_ms);
                }
            }
            ExecutionEvent::FundingUpdate(update) => {
                if let Some(v) = state.venues.get_mut(update.venue_index) {
                    v.funding_8h = update.funding_8h;
                }
            }
            ExecutionEvent::BalanceUpdate(update) => {
                if let Some(v) = state.venues.get_mut(update.venue_index) {
                    v.margin_balance_usd = update.margin_balance_usd;
                    v.margin_used_usd = update.margin_used_usd;
                    v.margin_available_usd = update.margin_available_usd;
                }
            }
            ExecutionEvent::OrderAck(ack) => {
                let is_cancel_all = ack.side.is_none()
                    && ack.price.is_none()
                    && ack.size.is_none()
                    && ack.purpose.is_none()
                    && ack.order_id == "cancel_all";
                if is_cancel_all && ack.venue_id.as_ref() == "all" {
                    for venue in &mut state.venues {
                        venue.clear_open_orders();
                    }
                    continue;
                }

                if let Some(v) = state.venues.get_mut(ack.venue_index) {
                    // Place ack if side/price/size/purpose are present.
                    if let (Some(side), Some(price), Some(size), Some(purpose)) =
                        (ack.side, ack.price, ack.size, ack.purpose)
                    {
                        v.upsert_open_order(OpenOrderRecord {
                            order_id: ack.order_id.clone(),
                            client_order_id: ack.client_order_id.clone(),
                            side,
                            price,
                            size,
                            remaining: size,
                            timestamp_ms: now_ms,
                            purpose,
                            time_in_force: None,
                            post_only: None,
                            reduce_only: None,
                        });
                        if purpose == OrderPurpose::Mm {
                            let order = MmOpenOrder {
                                price,
                                size,
                                timestamp_ms: now_ms,
                                order_id: ack.order_id.clone(),
                            };
                            match side {
                                Side::Buy => v.mm_open_bid = Some(order),
                                Side::Sell => v.mm_open_ask = Some(order),
                            }
                        }
                    } else if is_cancel_all {
                        v.clear_open_orders();
                    } else {
                        // Cancel ack: clear matching open MM orders.
                        v.remove_open_order(&ack.order_id);
                    }
                }
            }
            ExecutionEvent::OrderReject(reject) => {
                if let Some(order_id) = &reject.order_id {
                    if let Some(v) = state.venues.get_mut(reject.venue_index) {
                        v.remove_open_order(order_id);
                    }
                }
            }
            ExecutionEvent::Fill(fill) => {
                fills.push(fill.clone());
            }
        }

        #[cfg(feature = "live")]
        {
            state.live_order_state.apply_execution_event(event, now_ms);
        }
    }

    fills
}
