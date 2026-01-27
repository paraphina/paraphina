// src/gateway.rs
//
// Execution gateway abstraction + a simple synthetic gateway used by
// the simulation harness. In production, this will be replaced by
// real exchange connectors implementing `ExecutionGateway`.

use crate::config::Config;
use crate::types::TimestampMs;
use crate::types::{ExecutionEvent, FillEvent, OrderAck, OrderIntent, OrderPurpose, OrderReject};

/// Abstract execution gateway.
///
/// Given a set of `OrderIntent`s, the gateway is responsible for turning
/// them into real orders / fills and returning execution events.
/// State updates happen in a separate event applier to keep strategy pure.
pub trait ExecutionGateway {
    fn process_intents(
        &mut self,
        cfg: &Config,
        intents: &[OrderIntent],
        now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent>;
}

/// Synthetic gateway used in the simulation:
///  - every intent is assumed to fill fully at its limit price,
///  - maker / taker fees are taken from `VenueConfig`,
///  - state updates happen via the execution event applier (not here).
#[derive(Debug, Default)]
pub struct SimGateway {
    seq: u64,
}

impl SimGateway {
    pub fn new() -> Self {
        Self { seq: 0 }
    }

    fn next_order_id(&mut self, venue_index: usize, purpose: OrderPurpose) -> String {
        let purpose_str = match purpose {
            OrderPurpose::Mm => "mm",
            OrderPurpose::Exit => "exit",
            OrderPurpose::Hedge => "hedge",
        };
        let id = format!("sim_{}_{}_{}", venue_index, purpose_str, self.seq);
        self.seq += 1;
        id
    }
}

impl ExecutionGateway for SimGateway {
    fn process_intents(
        &mut self,
        cfg: &Config,
        intents: &[OrderIntent],
        _now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent> {
        let mut events = Vec::with_capacity(intents.len() * 2);

        for intent in intents {
            let (venue_index, venue_id) = match intent {
                OrderIntent::Place(pi) => (pi.venue_index, pi.venue_id.clone()),
                OrderIntent::Cancel(ci) => (ci.venue_index, ci.venue_id.clone()),
                OrderIntent::Replace(ri) => (ri.venue_index, ri.venue_id.clone()),
                OrderIntent::CancelAll(ci) => (
                    ci.venue_index.unwrap_or(0),
                    ci.venue_id.clone().unwrap_or_else(|| "all".into()),
                ),
            };

            let vcfg = match cfg.venues.get(venue_index) {
                Some(v) => v,
                None => {
                    eprintln!(
                        "SimGateway: invalid venue_index {} for intent {:?}",
                        venue_index, intent
                    );
                    continue;
                }
            };

            match intent {
                OrderIntent::Place(pi) => {
                    let order_id = pi
                        .client_order_id
                        .clone()
                        .unwrap_or_else(|| self.next_order_id(venue_index, pi.purpose));
                    if pi.size <= 0.0 || !pi.price.is_finite() || pi.price <= 0.0 {
                        events.push(ExecutionEvent::OrderReject(OrderReject {
                            venue_index,
                            venue_id: venue_id.clone(),
                            order_id: Some(order_id),
                            client_order_id: pi.client_order_id.clone(),
                            seq: None,
                            reason: "Invalid price/size".to_string(),
                        }));
                        continue;
                    }

                    events.push(ExecutionEvent::OrderAck(OrderAck {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: order_id.clone(),
                        client_order_id: pi.client_order_id.clone(),
                        seq: None,
                        side: Some(pi.side),
                        price: Some(pi.price),
                        size: Some(pi.size),
                        purpose: Some(pi.purpose),
                    }));

                    let fee_bps = match pi.purpose {
                        OrderPurpose::Mm => vcfg.maker_fee_bps - vcfg.maker_rebate_bps,
                        OrderPurpose::Exit | OrderPurpose::Hedge => vcfg.taker_fee_bps,
                    };

                    events.push(ExecutionEvent::Fill(FillEvent {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: Some(order_id.clone()),
                        client_order_id: pi.client_order_id.clone(),
                        seq: None,
                        side: pi.side,
                        price: pi.price,
                        size: pi.size,
                        purpose: pi.purpose,
                        fee_bps,
                    }));
                }
                OrderIntent::Cancel(ci) => {
                    events.push(ExecutionEvent::OrderAck(OrderAck {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: ci.order_id.clone(),
                        client_order_id: None,
                        seq: None,
                        side: None,
                        price: None,
                        size: None,
                        purpose: None,
                    }));
                }
                OrderIntent::Replace(ri) => {
                    // Cancel old order
                    events.push(ExecutionEvent::OrderAck(OrderAck {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: ri.order_id.clone(),
                        client_order_id: None,
                        seq: None,
                        side: None,
                        price: None,
                        size: None,
                        purpose: None,
                    }));

                    let new_order_id = ri
                        .client_order_id
                        .clone()
                        .unwrap_or_else(|| self.next_order_id(venue_index, ri.purpose));
                    if ri.size <= 0.0 || !ri.price.is_finite() || ri.price <= 0.0 {
                        events.push(ExecutionEvent::OrderReject(OrderReject {
                            venue_index,
                            venue_id: venue_id.clone(),
                            order_id: Some(new_order_id),
                            client_order_id: ri.client_order_id.clone(),
                            seq: None,
                            reason: "Invalid price/size".to_string(),
                        }));
                        continue;
                    }

                    events.push(ExecutionEvent::OrderAck(OrderAck {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: new_order_id.clone(),
                        client_order_id: ri.client_order_id.clone(),
                        seq: None,
                        side: Some(ri.side),
                        price: Some(ri.price),
                        size: Some(ri.size),
                        purpose: Some(ri.purpose),
                    }));

                    let fee_bps = match ri.purpose {
                        OrderPurpose::Mm => vcfg.maker_fee_bps - vcfg.maker_rebate_bps,
                        OrderPurpose::Exit | OrderPurpose::Hedge => vcfg.taker_fee_bps,
                    };

                    events.push(ExecutionEvent::Fill(FillEvent {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: Some(new_order_id.clone()),
                        client_order_id: ri.client_order_id.clone(),
                        seq: None,
                        side: ri.side,
                        price: ri.price,
                        size: ri.size,
                        purpose: ri.purpose,
                        fee_bps,
                    }));
                }
                OrderIntent::CancelAll(_) => {
                    events.push(ExecutionEvent::OrderAck(OrderAck {
                        venue_index,
                        venue_id: venue_id.clone(),
                        order_id: "cancel_all".to_string(),
                        client_order_id: None,
                        seq: None,
                        side: None,
                        price: None,
                        size: None,
                        purpose: None,
                    }));
                }
            }
        }

        events
    }
}
