//! Live order lifecycle state (feature-gated).

use std::collections::HashMap;

use crate::types::{ExecutionEvent, OrderPurpose, Side, TimestampMs};

use super::types::OrderSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    Accepted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct LiveOrder {
    pub client_order_id: Option<String>,
    pub exchange_order_id: Option<String>,
    pub venue_index: usize,
    pub side: Option<Side>,
    pub price: Option<f64>,
    pub total_qty: Option<f64>,
    pub remaining_qty: Option<f64>,
    pub purpose: Option<OrderPurpose>,
    pub status: OrderStatus,
    pub created_ms: TimestampMs,
    pub updated_ms: TimestampMs,
    pub last_update_seq: Option<u64>,
}

impl LiveOrder {}

#[derive(Debug, Default, Clone)]
pub struct LiveOrderState {
    orders: HashMap<String, LiveOrder>,
    exchange_to_key: HashMap<String, String>,
}

impl LiveOrderState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open_orders(&self) -> Vec<&LiveOrder> {
        self.orders
            .values()
            .filter(|o| {
                matches!(
                    o.status,
                    OrderStatus::Accepted | OrderStatus::PartiallyFilled
                )
            })
            .collect()
    }

    pub fn open_order_ids_by_venue(&self, venue_index: usize) -> Vec<String> {
        let mut out = Vec::new();
        for (key, order) in &self.orders {
            if order.venue_index != venue_index {
                continue;
            }
            if !matches!(
                order.status,
                OrderStatus::Accepted | OrderStatus::PartiallyFilled
            ) {
                continue;
            }
            if let Some(id) = order.exchange_order_id.as_ref() {
                out.push(id.clone());
            } else {
                out.push(key.clone());
            }
        }
        out.sort();
        out
    }

    pub fn apply_execution_event(&mut self, event: &ExecutionEvent, now_ms: TimestampMs) {
        match event {
            ExecutionEvent::OrderAck(ack) => {
                if ack.side.is_none() && ack.price.is_none() && ack.size.is_none() {
                    self.apply_cancel_ack(
                        ack.order_id.clone(),
                        ack.client_order_id.clone(),
                        now_ms,
                        ack.seq,
                    );
                } else {
                    self.apply_order_ack(
                        ack.order_id.clone(),
                        ack.client_order_id.clone(),
                        ack.venue_index,
                        ack.side,
                        ack.price,
                        ack.size,
                        ack.purpose,
                        now_ms,
                        ack.seq,
                    );
                }
            }
            ExecutionEvent::OrderReject(rej) => {
                self.apply_reject(
                    rej.order_id.clone(),
                    rej.client_order_id.clone(),
                    rej.venue_index,
                    now_ms,
                    rej.seq,
                );
            }
            ExecutionEvent::Fill(fill) => {
                self.apply_fill(
                    fill.order_id.clone(),
                    fill.client_order_id.clone(),
                    fill.venue_index,
                    fill.size,
                    now_ms,
                    fill.seq,
                );
            }
            _ => {}
        }
    }

    pub fn reconcile(&mut self, snapshot: &OrderSnapshot, now_ms: TimestampMs) {
        let mut seen = HashMap::new();
        for order in &snapshot.open_orders {
            let key = order.order_id.clone();
            let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
                client_order_id: None,
                exchange_order_id: Some(order.order_id.clone()),
                venue_index: snapshot.venue_index,
                side: Some(order.side),
                price: Some(order.price),
                total_qty: Some(order.size),
                remaining_qty: Some(order.size),
                purpose: Some(order.purpose),
                status: OrderStatus::Accepted,
                created_ms: now_ms,
                updated_ms: now_ms,
                last_update_seq: Some(snapshot.seq),
            });
            entry.exchange_order_id = Some(order.order_id.clone());
            entry.side = Some(order.side);
            entry.price = Some(order.price);
            entry.total_qty = Some(order.size);
            entry.remaining_qty = Some(order.size);
            entry.purpose = Some(order.purpose);
            entry.status = OrderStatus::Accepted;
            entry.updated_ms = now_ms;
            entry.last_update_seq = Some(snapshot.seq);
            seen.insert(key, true);
        }

        for (key, order) in self.orders.iter_mut() {
            if order.venue_index == snapshot.venue_index && !seen.contains_key(key) {
                order.status = OrderStatus::Cancelled;
                order.remaining_qty = Some(0.0);
                order.updated_ms = now_ms;
                order.last_update_seq = Some(snapshot.seq);
            }
        }
    }

    pub fn cancel_all(
        &mut self,
        venue_index: Option<usize>,
        now_ms: TimestampMs,
        seq: Option<u64>,
    ) {
        for order in self.orders.values_mut() {
            if venue_index.is_some_and(|idx| order.venue_index != idx) {
                continue;
            }
            if let Some(prev) = order.last_update_seq {
                if seq.is_some_and(|s| s <= prev) {
                    continue;
                }
            }
            order.status = OrderStatus::Cancelled;
            order.remaining_qty = Some(0.0);
            order.updated_ms = now_ms;
            order.last_update_seq = seq;
        }
    }

    fn apply_order_ack(
        &mut self,
        exchange_order_id: String,
        client_order_id: Option<String>,
        venue_index: usize,
        side: Option<Side>,
        price: Option<f64>,
        size: Option<f64>,
        purpose: Option<OrderPurpose>,
        now_ms: TimestampMs,
        seq: Option<u64>,
    ) {
        let key = client_order_id
            .clone()
            .unwrap_or_else(|| exchange_order_id.clone());
        let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
            client_order_id: client_order_id.clone(),
            exchange_order_id: Some(exchange_order_id.clone()),
            venue_index,
            side,
            price,
            total_qty: size,
            remaining_qty: size,
            purpose,
            status: OrderStatus::Accepted,
            created_ms: now_ms,
            updated_ms: now_ms,
            last_update_seq: seq,
        });
        if let Some(prev) = entry.last_update_seq {
            if seq.is_some_and(|s| s <= prev) {
                return;
            }
        }
        entry.exchange_order_id = Some(exchange_order_id.clone());
        entry.client_order_id = client_order_id.clone();
        entry.side = side.or(entry.side);
        entry.price = price.or(entry.price);
        entry.total_qty = size.or(entry.total_qty);
        entry.remaining_qty = entry.remaining_qty.or(size);
        entry.purpose = purpose.or(entry.purpose);
        entry.status = OrderStatus::Accepted;
        entry.updated_ms = now_ms;
        entry.last_update_seq = seq;
        self.exchange_to_key.insert(exchange_order_id, key);
    }

    fn apply_cancel_ack(
        &mut self,
        exchange_order_id: String,
        client_order_id: Option<String>,
        now_ms: TimestampMs,
        seq: Option<u64>,
    ) {
        if let Some(key) = client_order_id
            .clone()
            .or_else(|| self.exchange_to_key.get(&exchange_order_id).cloned())
        {
            if let Some(entry) = self.orders.get_mut(&key) {
                if let Some(prev) = entry.last_update_seq {
                    if seq.is_some_and(|s| s <= prev) {
                        return;
                    }
                }
                entry.status = OrderStatus::Cancelled;
                entry.remaining_qty = Some(0.0);
                entry.updated_ms = now_ms;
                entry.last_update_seq = seq;
            }
        }
    }

    fn apply_reject(
        &mut self,
        exchange_order_id: Option<String>,
        client_order_id: Option<String>,
        venue_index: usize,
        now_ms: TimestampMs,
        seq: Option<u64>,
    ) {
        let key = client_order_id
            .clone()
            .or_else(|| exchange_order_id.clone())
            .unwrap_or_else(|| format!("rejected_{}_{}", venue_index, now_ms));
        let entry = self.orders.entry(key).or_insert_with(|| LiveOrder {
            client_order_id,
            exchange_order_id,
            venue_index,
            side: None,
            price: None,
            total_qty: None,
            remaining_qty: Some(0.0),
            purpose: None,
            status: OrderStatus::Rejected,
            created_ms: now_ms,
            updated_ms: now_ms,
            last_update_seq: seq,
        });
        if let Some(prev) = entry.last_update_seq {
            if seq.is_some_and(|s| s <= prev) {
                return;
            }
        }
        entry.status = OrderStatus::Rejected;
        entry.remaining_qty = Some(0.0);
        entry.updated_ms = now_ms;
        entry.last_update_seq = seq;
    }

    fn apply_fill(
        &mut self,
        exchange_order_id: Option<String>,
        client_order_id: Option<String>,
        venue_index: usize,
        fill_qty: f64,
        now_ms: TimestampMs,
        seq: Option<u64>,
    ) {
        let key = client_order_id
            .clone()
            .or_else(|| exchange_order_id.clone())
            .or_else(|| {
                exchange_order_id
                    .as_ref()
                    .and_then(|id| self.exchange_to_key.get(id).cloned())
            });
        let Some(key) = key else {
            return;
        };
        let Some(entry) = self.orders.get_mut(&key) else {
            return;
        };
        if entry.venue_index != venue_index {
            return;
        }
        if let Some(prev) = entry.last_update_seq {
            if seq.is_some_and(|s| s <= prev) {
                return;
            }
        }
        let remaining = entry
            .remaining_qty
            .unwrap_or(entry.total_qty.unwrap_or(0.0));
        let new_remaining = (remaining - fill_qty).max(0.0);
        entry.remaining_qty = Some(new_remaining);
        entry.status = if new_remaining <= 0.0 {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        entry.updated_ms = now_ms;
        entry.last_update_seq = seq;
    }
}
