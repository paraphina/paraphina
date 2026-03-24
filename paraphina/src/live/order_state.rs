//! Live order lifecycle state (feature-gated).

use std::collections::HashMap;

use crate::types::{ExecutionEvent, OrderPurpose, Side, TimestampMs};

use super::types::{OpenOrderSnapshot, OrderSnapshot};

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

#[derive(Debug, Clone, PartialEq)]
pub struct InferredFill {
    pub venue_index: usize,
    pub venue_id: String,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub seq: u64,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
}

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
        const EPS: f64 = 1e-9;
        let mut seen = HashMap::new();
        for order in &snapshot.open_orders {
            let key = self.snapshot_order_key(order);
            let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
                client_order_id: order.client_order_id.clone(),
                exchange_order_id: Some(order.order_id.clone()),
                venue_index: snapshot.venue_index,
                side: Some(order.side),
                price: Some(order.price),
                total_qty: Some(order.size),
                remaining_qty: Some(order.size),
                purpose: order.purpose,
                status: OrderStatus::Accepted,
                created_ms: now_ms,
                updated_ms: now_ms,
                last_update_seq: Some(snapshot.seq),
            });
            if let Some(prev) = entry.last_update_seq {
                if snapshot.seq <= prev {
                    seen.insert(key, true);
                    continue;
                }
            }
            entry.exchange_order_id = Some(order.order_id.clone());
            entry.client_order_id = order
                .client_order_id
                .clone()
                .or_else(|| entry.client_order_id.clone());
            entry.side = Some(order.side);
            entry.price = Some(order.price);
            entry.total_qty = Some(entry.total_qty.unwrap_or(order.size).max(order.size));
            entry.remaining_qty = Some(order.size);
            entry.purpose = order.purpose.or(entry.purpose);
            entry.status = if entry.total_qty.unwrap_or(order.size) > order.size + EPS {
                OrderStatus::PartiallyFilled
            } else {
                OrderStatus::Accepted
            };
            entry.updated_ms = now_ms;
            entry.last_update_seq = Some(snapshot.seq);
            self.exchange_to_key
                .insert(order.order_id.clone(), key.clone());
            seen.insert(key, true);
        }

        for (key, order) in self.orders.iter_mut() {
            if order.venue_index == snapshot.venue_index && !seen.contains_key(key) {
                if let Some(prev) = order.last_update_seq {
                    if snapshot.seq <= prev {
                        continue;
                    }
                }
                if !matches!(
                    order.status,
                    OrderStatus::Accepted | OrderStatus::PartiallyFilled
                ) {
                    continue;
                }
                order.status = OrderStatus::Cancelled;
                order.remaining_qty = Some(0.0);
                order.updated_ms = now_ms;
                order.last_update_seq = Some(snapshot.seq);
            }
        }
    }

    pub fn reconcile_with_fill_inference(&self, snapshot: &OrderSnapshot) -> Vec<InferredFill> {
        const EPS: f64 = 1e-9;
        let mut seen = HashMap::new();
        let mut inferred = Vec::new();
        for order in &snapshot.open_orders {
            let key = self.snapshot_order_key(order);
            let Some(entry) = self.orders.get(&key) else {
                seen.insert(key, true);
                continue;
            };
            if let Some(prev) = entry.last_update_seq {
                if snapshot.seq <= prev {
                    seen.insert(key, true);
                    continue;
                }
            }
            let previous_remaining = entry
                .remaining_qty
                .unwrap_or(entry.total_qty.unwrap_or(order.size));
            if matches!(
                entry.status,
                OrderStatus::Accepted | OrderStatus::PartiallyFilled
            ) && previous_remaining > order.size + EPS
            {
                if let Some(fill) = Self::build_inferred_fill(
                    snapshot,
                    entry,
                    previous_remaining - order.size,
                    Some(order.order_id.clone()),
                    order.client_order_id.clone(),
                ) {
                    inferred.push(fill);
                }
            }
            seen.insert(key, true);
        }
        // A missing order in a venue snapshot is not enough evidence of a fill.
        // It can be caused by cancel/replace churn or a stale/missing open-order
        // snapshot. We only infer fills here from observed size reductions; full
        // disappearance is attributed later from account-position deltas.
        inferred
    }

    pub fn infer_fills_from_position_delta(
        &self,
        venue_index: usize,
        venue_id: &str,
        position_delta_tao: f64,
        snapshot_seq: u64,
        now_ms: TimestampMs,
    ) -> Vec<InferredFill> {
        const EPS: f64 = 1e-9;
        const CANCEL_FILL_ATTRIBUTION_GRACE_MS: TimestampMs = 15_000;

        if position_delta_tao.abs() <= EPS {
            return Vec::new();
        }

        let side = if position_delta_tao > 0.0 {
            Side::Buy
        } else {
            Side::Sell
        };
        let mut remaining = position_delta_tao.abs();

        let mut candidates = self
            .orders
            .values()
            .filter_map(|order| {
                if order.venue_index != venue_index {
                    return None;
                }
                if order.side != Some(side) {
                    return None;
                }
                let available = order
                    .remaining_qty
                    .unwrap_or(order.total_qty.unwrap_or(0.0));
                if available <= EPS {
                    return None;
                }
                if order.price.is_none() || order.purpose.is_none() {
                    return None;
                }
                match order.status {
                    OrderStatus::Accepted | OrderStatus::PartiallyFilled => Some(order),
                    OrderStatus::Cancelled
                        if now_ms.saturating_sub(order.updated_ms)
                            <= CANCEL_FILL_ATTRIBUTION_GRACE_MS =>
                    {
                        Some(order)
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|lhs, rhs| {
            rhs.updated_ms
                .cmp(&lhs.updated_ms)
                .then_with(|| rhs.created_ms.cmp(&lhs.created_ms))
        });

        let mut inferred = Vec::new();
        for order in candidates {
            if remaining <= EPS {
                break;
            }
            let available = order
                .remaining_qty
                .unwrap_or(order.total_qty.unwrap_or(0.0))
                .max(0.0);
            if available <= EPS {
                continue;
            }
            let fill_size = remaining.min(available);
            if let Some(fill) = Self::build_inferred_fill_from_live_order(
                venue_index,
                venue_id,
                snapshot_seq,
                order,
                fill_size,
            ) {
                inferred.push(fill);
                remaining -= fill_size;
            }
        }

        inferred
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

    fn snapshot_order_key(&self, order: &OpenOrderSnapshot) -> String {
        order
            .client_order_id
            .clone()
            .or_else(|| self.exchange_to_key.get(&order.order_id).cloned())
            .unwrap_or_else(|| order.order_id.clone())
    }

    fn build_inferred_fill(
        snapshot: &OrderSnapshot,
        order: &LiveOrder,
        size: f64,
        order_id: Option<String>,
        client_order_id: Option<String>,
    ) -> Option<InferredFill> {
        if size <= 0.0 {
            return None;
        }
        Some(InferredFill {
            venue_index: snapshot.venue_index,
            venue_id: snapshot.venue_id.clone(),
            order_id: order_id.or_else(|| order.exchange_order_id.clone()),
            client_order_id: client_order_id.or_else(|| order.client_order_id.clone()),
            seq: snapshot.seq,
            side: order.side?,
            price: order.price?,
            size,
            purpose: order.purpose?,
        })
    }

    fn build_inferred_fill_from_live_order(
        venue_index: usize,
        venue_id: &str,
        snapshot_seq: u64,
        order: &LiveOrder,
        size: f64,
    ) -> Option<InferredFill> {
        if size <= 0.0 {
            return None;
        }
        Some(InferredFill {
            venue_index,
            venue_id: venue_id.to_string(),
            order_id: order.exchange_order_id.clone(),
            client_order_id: order.client_order_id.clone(),
            seq: snapshot_seq,
            side: order.side?,
            price: order.price?,
            size,
            purpose: order.purpose?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionEvent, OrderAck};

    fn ack(
        venue_index: usize,
        order_id: &str,
        client_order_id: &str,
        side: Side,
        price: f64,
        size: f64,
        purpose: OrderPurpose,
        seq: u64,
    ) -> ExecutionEvent {
        ExecutionEvent::OrderAck(OrderAck {
            venue_index,
            venue_id: "test".into(),
            order_id: order_id.to_string(),
            client_order_id: Some(client_order_id.to_string()),
            seq: Some(seq),
            side: Some(side),
            price: Some(price),
            size: Some(size),
            purpose: Some(purpose),
        })
    }

    fn snapshot(seq: u64, open_orders: Vec<OpenOrderSnapshot>) -> OrderSnapshot {
        OrderSnapshot {
            venue_index: 0,
            venue_id: "test".to_string(),
            seq,
            timestamp_ms: 2_000,
            open_orders,
        }
    }

    #[test]
    fn reconcile_matches_exchange_ids_and_infers_partial_fill() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "oid_1",
                "co_1",
                Side::Buy,
                100.0,
                1.0,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );

        let inferred = state.reconcile_with_fill_inference(&snapshot(
            2,
            vec![OpenOrderSnapshot {
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                side: Side::Buy,
                price: 100.0,
                size: 0.4,
                purpose: None,
            }],
        ));

        assert_eq!(inferred.len(), 1);
        assert_eq!(inferred[0].order_id.as_deref(), Some("oid_1"));
        assert_eq!(inferred[0].client_order_id.as_deref(), Some("co_1"));
        assert!((inferred[0].size - 0.6).abs() < 1e-9);
        state.reconcile(
            &snapshot(
                2,
                vec![OpenOrderSnapshot {
                    order_id: "oid_1".to_string(),
                    client_order_id: Some("co_1".to_string()),
                    side: Side::Buy,
                    price: 100.0,
                    size: 0.4,
                    purpose: None,
                }],
            ),
            2_000,
        );
        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.remaining_qty, Some(0.4));
    }

    #[test]
    fn reconcile_does_not_infer_fill_when_live_order_disappears() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "oid_1",
                "co_1",
                Side::Sell,
                101.0,
                0.5,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );

        let inferred = state.reconcile_with_fill_inference(&snapshot(2, vec![]));

        assert!(inferred.is_empty());
        state.reconcile(&snapshot(2, vec![]), 2_000);
        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
    }

    #[test]
    fn reconcile_does_not_infer_fill_after_cancel_ack() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "oid_1",
                "co_1",
                Side::Sell,
                101.0,
                0.5,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );
        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 0,
                venue_id: "test".into(),
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(2),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_500,
        );

        let inferred = state.reconcile_with_fill_inference(&snapshot(3, vec![]));

        assert!(inferred.is_empty());
        state.reconcile(&snapshot(3, vec![]), 2_000);
        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
    }

    #[test]
    fn account_position_delta_can_attribute_recent_cancelled_order_fill() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "oid_1",
                "co_1",
                Side::Sell,
                101.0,
                0.5,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );
        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 0,
                venue_id: "test".into(),
                order_id: "oid_1".to_string(),
                client_order_id: Some("co_1".to_string()),
                seq: Some(2),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_100,
        );

        let inferred = state.infer_fills_from_position_delta(0, "test", -0.2, 3, 1_200);

        assert_eq!(inferred.len(), 1);
        assert_eq!(inferred[0].order_id.as_deref(), Some("oid_1"));
        assert_eq!(inferred[0].client_order_id.as_deref(), Some("co_1"));
        assert_eq!(inferred[0].side, Side::Sell);
        assert!((inferred[0].size - 0.2).abs() < 1e-9);
    }
}
