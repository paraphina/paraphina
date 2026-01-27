//! Deterministic shadow execution adapter (feature-gated).

use std::collections::HashMap;

use crate::config::Config;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

use super::instrument::InstrumentSpec;
use super::ops::config_hash;
use super::types::{
    CancelAccepted, CancelAllAccepted, CancelRejected, ExecutionEvent, OrderAccepted, OrderRejected,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ShadowOrder {
    venue_index: usize,
    venue_id: String,
    side: Side,
    price: f64,
    size: f64,
    purpose: OrderPurpose,
    time_in_force: TimeInForce,
    post_only: bool,
    reduce_only: bool,
    client_order_id: Option<String>,
}

#[derive(Debug)]
pub struct ShadowAckAdapter {
    cfg_hash: u64,
    nonce: u64,
    seq: u64,
    open_orders: HashMap<String, ShadowOrder>,
    specs: Vec<InstrumentSpec>,
    best_bid_ask: HashMap<usize, (Option<f64>, Option<f64>)>,
}

impl ShadowAckAdapter {
    pub fn new(cfg: &Config) -> Self {
        Self {
            cfg_hash: config_hash(cfg),
            nonce: 0,
            seq: 0,
            open_orders: HashMap::new(),
            specs: InstrumentSpec::from_config(cfg),
            best_bid_ask: HashMap::new(),
        }
    }

    /// Update best bid/ask for optional post-only crossing checks.
    pub fn update_best_bid_ask(
        &mut self,
        venue_index: usize,
        best_bid: Option<f64>,
        best_ask: Option<f64>,
    ) {
        self.best_bid_ask.insert(venue_index, (best_bid, best_ask));
    }

    pub fn handle_intents(
        &mut self,
        intents: Vec<OrderIntent>,
        tick: u64,
        now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent> {
        let mut events = Vec::new();
        for intent in intents {
            match intent {
                OrderIntent::Place(place) => {
                    let place_events = self.handle_place(place, tick, now_ms);
                    events.extend(place_events);
                }
                OrderIntent::Cancel(cancel) => {
                    let event = self.handle_cancel(cancel, now_ms);
                    events.push(event);
                }
                OrderIntent::Replace(replace) => {
                    let cancel_event = self.handle_cancel(
                        crate::types::CancelOrderIntent {
                            venue_index: replace.venue_index,
                            venue_id: replace.venue_id.clone(),
                            order_id: replace.order_id.clone(),
                        },
                        now_ms,
                    );
                    events.push(cancel_event);
                    let place_events = self.handle_place(
                        crate::types::PlaceOrderIntent {
                            venue_index: replace.venue_index,
                            venue_id: replace.venue_id.clone(),
                            side: replace.side,
                            price: replace.price,
                            size: replace.size,
                            purpose: replace.purpose,
                            time_in_force: replace.time_in_force,
                            post_only: replace.post_only,
                            reduce_only: replace.reduce_only,
                            client_order_id: replace.client_order_id.clone(),
                        },
                        tick,
                        now_ms,
                    );
                    events.extend(place_events);
                }
                OrderIntent::CancelAll(cancel_all) => {
                    let event = self.handle_cancel_all(cancel_all, now_ms);
                    events.push(event);
                }
            }
        }
        events
    }

    fn handle_place(
        &mut self,
        mut place: crate::types::PlaceOrderIntent,
        tick: u64,
        now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent> {
        let mut events = Vec::new();
        if let Some(spec) = self.specs.get(place.venue_index) {
            place.price = spec.round_price(place.price);
            place.size = spec.round_size(place.size);
            if !spec.meets_min_notional(place.size, place.price) {
                self.seq = self.seq.wrapping_add(1);
                events.push(ExecutionEvent::OrderRejected(OrderRejected {
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    seq: self.seq,
                    timestamp_ms: now_ms,
                    order_id: place.client_order_id.clone(),
                    reason: "min_notional_usd".to_string(),
                }));
                return events;
            }
        }

        if place.post_only && self.crosses_book(&place) {
            self.seq = self.seq.wrapping_add(1);
            events.push(ExecutionEvent::OrderRejected(OrderRejected {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: place.client_order_id.clone(),
                reason: "post_only_cross".to_string(),
            }));
            return events;
        }

        let order_id = self.next_order_id(
            place.venue_id.as_ref(),
            tick,
            place.side,
            place.purpose,
            place.client_order_id.clone(),
        );
        let client_order_id = Some(order_id.clone());
        self.seq = self.seq.wrapping_add(1);
        events.push(ExecutionEvent::OrderAccepted(OrderAccepted {
            venue_index: place.venue_index,
            venue_id: place.venue_id.to_string(),
            seq: self.seq,
            timestamp_ms: now_ms,
            order_id: order_id.clone(),
            client_order_id: client_order_id.clone(),
            side: place.side,
            price: place.price,
            size: place.size,
            purpose: place.purpose,
        }));

        if place.time_in_force == TimeInForce::Ioc {
            self.seq = self.seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelAccepted(CancelAccepted {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: order_id.clone(),
            }));
        } else {
            self.open_orders.insert(
                order_id.clone(),
                ShadowOrder {
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    side: place.side,
                    price: place.price,
                    size: place.size,
                    purpose: place.purpose,
                    time_in_force: place.time_in_force,
                    post_only: place.post_only,
                    reduce_only: place.reduce_only,
                    client_order_id,
                },
            );
        }
        events
    }

    fn handle_cancel(
        &mut self,
        cancel: crate::types::CancelOrderIntent,
        now_ms: TimestampMs,
    ) -> ExecutionEvent {
        let venue_id = cancel.venue_id.to_string();
        if self.open_orders.remove(&cancel.order_id).is_some() {
            self.seq = self.seq.wrapping_add(1);
            ExecutionEvent::CancelAccepted(CancelAccepted {
                venue_index: cancel.venue_index,
                venue_id,
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: cancel.order_id,
            })
        } else {
            self.seq = self.seq.wrapping_add(1);
            ExecutionEvent::CancelRejected(CancelRejected {
                venue_index: cancel.venue_index,
                venue_id,
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: Some(cancel.order_id),
                reason: "unknown_order_id".to_string(),
            })
        }
    }

    fn handle_cancel_all(
        &mut self,
        cancel_all: crate::types::CancelAllOrderIntent,
        now_ms: TimestampMs,
    ) -> ExecutionEvent {
        let mut cancelled = Vec::new();
        for (order_id, order) in self.open_orders.iter() {
            if cancel_all
                .venue_index
                .map_or(true, |idx| idx == order.venue_index)
            {
                cancelled.push(order_id.clone());
            }
        }
        for order_id in &cancelled {
            self.open_orders.remove(order_id);
        }
        let venue_id = cancel_all
            .venue_id
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "all".to_string());
        self.seq = self.seq.wrapping_add(1);
        ExecutionEvent::CancelAllAccepted(CancelAllAccepted {
            venue_index: cancel_all.venue_index.unwrap_or(0),
            venue_id,
            seq: self.seq,
            timestamp_ms: now_ms,
            count: cancelled.len(),
        })
    }

    fn next_order_id(
        &mut self,
        venue_id: &str,
        tick: u64,
        side: Side,
        purpose: OrderPurpose,
        provided: Option<String>,
    ) -> String {
        if let Some(id) = provided {
            return id;
        }
        let nonce = self.nonce;
        self.nonce = self.nonce.wrapping_add(1);
        format!(
            "shadow_{:016x}_{}_{}_{}_{}_{}",
            self.cfg_hash,
            venue_id,
            tick,
            format!("{:?}", side).to_lowercase(),
            format!("{:?}", purpose).to_lowercase(),
            nonce
        )
    }

    fn crosses_book(&self, place: &crate::types::PlaceOrderIntent) -> bool {
        let Some((best_bid, best_ask)) = self.best_bid_ask.get(&place.venue_index) else {
            return false;
        };
        match place.side {
            Side::Buy => best_ask.map_or(false, |ask| place.price >= ask),
            Side::Sell => best_bid.map_or(false, |bid| place.price <= bid),
        }
    }
}
