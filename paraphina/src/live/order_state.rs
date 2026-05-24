//! Live order lifecycle state (feature-gated).

use std::collections::HashMap;

use crate::types::{ExecutionEvent, OrderPurpose, Side, TimestampMs};

use super::types::{OpenOrderSnapshot, OrderSnapshot};

pub const SUPPORTED_REPLACE_GAP_GRACE_MS: TimestampMs = 2_000;
const CANCEL_FILL_ATTRIBUTION_GRACE_MS: TimestampMs = 15_000;
const PARADEX_CANCEL_FILL_ATTRIBUTION_GRACE_MS: TimestampMs = 180_000;

fn cancel_fill_attribution_grace_ms(venue_id: &str) -> TimestampMs {
    if venue_id.eq_ignore_ascii_case("paradex") {
        PARADEX_CANCEL_FILL_ATTRIBUTION_GRACE_MS
    } else {
        CANCEL_FILL_ATTRIBUTION_GRACE_MS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    Accepted,
    PartiallyFilled,
    SnapshotGapGrace,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct LiveOrder {
    pub decision_id: Option<String>,
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
    pub gap_grace_started_ms: Option<TimestampMs>,
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

    pub fn active_orders(&self) -> Vec<&LiveOrder> {
        self.orders
            .values()
            .filter(|o| {
                matches!(
                    o.status,
                    OrderStatus::Pending
                        | OrderStatus::Accepted
                        | OrderStatus::PartiallyFilled
                        | OrderStatus::SnapshotGapGrace
                )
            })
            .collect()
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

    pub fn register_mm_decision_lineage(
        &mut self,
        venue_index: usize,
        client_order_id: &str,
        side: Side,
        price: f64,
        size: f64,
        purpose: OrderPurpose,
        decision_id: &str,
        now_ms: TimestampMs,
    ) {
        let key = client_order_id.to_string();
        let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
            decision_id: Some(decision_id.to_string()),
            client_order_id: Some(key.clone()),
            exchange_order_id: None,
            venue_index,
            side: Some(side),
            price: Some(price),
            total_qty: Some(size),
            remaining_qty: Some(size),
            purpose: Some(purpose),
            status: OrderStatus::Pending,
            created_ms: now_ms,
            updated_ms: now_ms,
            gap_grace_started_ms: None,
            last_update_seq: None,
        });
        entry.decision_id = Some(decision_id.to_string());
        entry.client_order_id = Some(key);
        entry.venue_index = venue_index;
        entry.side = Some(side);
        entry.price = Some(price);
        entry.total_qty = Some(size);
        entry.remaining_qty = entry.remaining_qty.or(Some(size));
        entry.purpose = Some(purpose);
        entry.updated_ms = now_ms;
    }

    pub fn decision_id_for_order(
        &self,
        order_id: Option<&str>,
        client_order_id: Option<&str>,
    ) -> Option<&str> {
        let key = client_order_id
            .map(ToOwned::to_owned)
            .or_else(|| order_id.and_then(|id| self.exchange_to_key.get(id).cloned()))
            .or_else(|| order_id.map(ToOwned::to_owned))?;
        self.orders.get(&key)?.decision_id.as_deref()
    }

    pub fn apply_execution_event(&mut self, event: &ExecutionEvent, now_ms: TimestampMs) {
        match event {
            ExecutionEvent::OrderAck(ack) => {
                let is_cancel_all = ack.side.is_none()
                    && ack.price.is_none()
                    && ack.size.is_none()
                    && ack.purpose.is_none()
                    && ack.order_id == "cancel_all";
                if is_cancel_all {
                    let venue_index = if ack.venue_id.as_ref() == "all" {
                        None
                    } else {
                        Some(ack.venue_index)
                    };
                    self.cancel_all(venue_index, now_ms, ack.seq);
                } else if ack.side.is_none() && ack.price.is_none() && ack.size.is_none() {
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
                    rej.purpose.is_none() && rej.reduce_only.is_none(),
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
        self.reconcile_with_supported_replace_gap_grace_ms(
            snapshot,
            now_ms,
            SUPPORTED_REPLACE_GAP_GRACE_MS,
        );
    }

    pub fn reconcile_with_supported_replace_gap_grace_ms(
        &mut self,
        snapshot: &OrderSnapshot,
        now_ms: TimestampMs,
        supported_replace_gap_grace_ms: TimestampMs,
    ) {
        const EPS: f64 = 1e-9;
        let supported_replace_gap_grace_ms = supported_replace_gap_grace_ms.max(1);
        let mut seen = HashMap::new();
        for order in &snapshot.open_orders {
            let key = self.snapshot_order_key(order);
            let exchange_order_id = snapshot_exchange_order_id(order);
            let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
                decision_id: None,
                client_order_id: order.client_order_id.clone(),
                exchange_order_id: exchange_order_id.clone(),
                venue_index: snapshot.venue_index,
                side: Some(order.side),
                price: Some(order.price),
                total_qty: Some(order.size),
                remaining_qty: Some(order.size),
                purpose: order.purpose,
                status: OrderStatus::Accepted,
                created_ms: now_ms,
                updated_ms: now_ms,
                gap_grace_started_ms: None,
                last_update_seq: Some(snapshot.seq),
            });
            if let Some(prev) = entry.last_update_seq {
                if snapshot.seq <= prev {
                    if supports_present_snapshot_lower_seq_reconcile(&snapshot.venue_id, entry)
                        && matches!(
                            entry.status,
                            OrderStatus::SnapshotGapGrace | OrderStatus::Cancelled
                        )
                    {
                        let was_snapshot_gap_grace =
                            matches!(entry.status, OrderStatus::SnapshotGapGrace);
                        entry.exchange_order_id = exchange_order_id.clone();
                        entry.client_order_id = order
                            .client_order_id
                            .clone()
                            .or_else(|| entry.client_order_id.clone());
                        entry.side = Some(order.side);
                        entry.price = Some(order.price);
                        entry.total_qty =
                            Some(entry.total_qty.unwrap_or(order.size).max(order.size));
                        entry.remaining_qty = Some(order.size);
                        entry.purpose = order.purpose.or(entry.purpose);
                        let total_qty = entry.total_qty.unwrap_or(order.size);
                        let remaining_qty = entry.remaining_qty.unwrap_or(total_qty);
                        entry.status = if total_qty > remaining_qty + EPS {
                            OrderStatus::PartiallyFilled
                        } else {
                            OrderStatus::Accepted
                        };
                        entry.updated_ms = now_ms;
                        if was_snapshot_gap_grace {
                            emit_supported_replace_snapshot_gap_cleared(
                                &snapshot.venue_id,
                                entry.side,
                                entry
                                    .exchange_order_id
                                    .as_deref()
                                    .unwrap_or(&order.order_id),
                                entry.client_order_id.as_deref(),
                            );
                        }
                        entry.gap_grace_started_ms = None;
                        entry.last_update_seq = Some(prev.max(snapshot.seq));
                        if let Some(exchange_order_id) = exchange_order_id {
                            self.exchange_to_key.insert(exchange_order_id, key.clone());
                        }
                    }
                    seen.insert(key, true);
                    continue;
                }
            }
            let was_snapshot_gap_grace = matches!(entry.status, OrderStatus::SnapshotGapGrace);
            entry.exchange_order_id = exchange_order_id.clone();
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
            if was_snapshot_gap_grace {
                emit_supported_replace_snapshot_gap_cleared(
                    &snapshot.venue_id,
                    entry.side,
                    entry
                        .exchange_order_id
                        .as_deref()
                        .unwrap_or(&order.order_id),
                    entry.client_order_id.as_deref(),
                );
            }
            entry.gap_grace_started_ms = None;
            entry.last_update_seq = Some(snapshot.seq);
            if let Some(exchange_order_id) = exchange_order_id {
                self.exchange_to_key.insert(exchange_order_id, key.clone());
            }
            seen.insert(key, true);
        }

        for (key, order) in self.orders.iter_mut() {
            if order.venue_index == snapshot.venue_index && !seen.contains_key(key) {
                if let Some(prev) = order.last_update_seq {
                    if snapshot.seq <= prev
                        && !supports_missing_snapshot_lower_seq_reconcile(&snapshot.venue_id, order)
                    {
                        continue;
                    }
                }
                if !matches!(
                    order.status,
                    OrderStatus::Accepted
                        | OrderStatus::PartiallyFilled
                        | OrderStatus::SnapshotGapGrace
                ) {
                    continue;
                }
                if supports_supported_replace_snapshot_gap_grace(&snapshot.venue_id, order) {
                    let gap_start_ms = order.gap_grace_started_ms.unwrap_or(now_ms);
                    if order.gap_grace_started_ms.is_none() {
                        emit_supported_replace_snapshot_gap_grace(
                            &snapshot.venue_id,
                            order.side,
                            order.exchange_order_id.as_deref().unwrap_or(key),
                            order.client_order_id.as_deref(),
                        );
                    }
                    if now_ms.saturating_sub(gap_start_ms) <= supported_replace_gap_grace_ms {
                        order.status = OrderStatus::SnapshotGapGrace;
                        order.updated_ms = now_ms;
                        order.gap_grace_started_ms = Some(gap_start_ms);
                        order.last_update_seq = match order.last_update_seq {
                            Some(prev) => Some(prev.max(snapshot.seq)),
                            None => Some(snapshot.seq),
                        };
                        continue;
                    }
                    emit_supported_replace_snapshot_gap_expired(
                        &snapshot.venue_id,
                        order.side,
                        order.exchange_order_id.as_deref().unwrap_or(key),
                        order.client_order_id.as_deref(),
                    );
                }
                order.status = OrderStatus::Cancelled;
                order.remaining_qty = Some(0.0);
                order.updated_ms = now_ms;
                order.gap_grace_started_ms = None;
                order.last_update_seq = match order.last_update_seq {
                    Some(prev) => Some(prev.max(snapshot.seq)),
                    None => Some(snapshot.seq),
                };
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
                OrderStatus::Accepted
                    | OrderStatus::PartiallyFilled
                    | OrderStatus::SnapshotGapGrace
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

        if position_delta_tao.abs() <= EPS {
            return Vec::new();
        }

        let side = if position_delta_tao > 0.0 {
            Side::Buy
        } else {
            Side::Sell
        };
        let mut remaining = position_delta_tao.abs();
        let cancelled_order_grace_ms = cancel_fill_attribution_grace_ms(venue_id);

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
                    OrderStatus::SnapshotGapGrace => Some(order),
                    OrderStatus::Cancelled
                        if now_ms.saturating_sub(order.updated_ms) <= cancelled_order_grace_ms =>
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
            // REST cancel-all acknowledgements can arrive on a sequence stream
            // independent from private order updates. Treat them as terminal
            // just like individual cancel acknowledgements.
            order.status = OrderStatus::Cancelled;
            order.updated_ms = now_ms;
            order.gap_grace_started_ms = None;
            order.last_update_seq = match (order.last_update_seq, seq) {
                (Some(prev), Some(next)) => Some(prev.max(next)),
                (Some(prev), None) => Some(prev),
                (None, next) => next,
            };
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
        let existing_order = self.orders.contains_key(&key);
        let entry = self.orders.entry(key.clone()).or_insert_with(|| LiveOrder {
            decision_id: None,
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
            gap_grace_started_ms: None,
            last_update_seq: seq,
        });
        if existing_order {
            if let Some(prev) = entry.last_update_seq {
                if seq.is_some_and(|s| s <= prev) {
                    return;
                }
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
        entry.gap_grace_started_ms = None;
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
            .or_else(|| {
                self.orders
                    .contains_key(&exchange_order_id)
                    .then_some(exchange_order_id)
            })
        {
            if let Some(entry) = self.orders.get_mut(&key) {
                // Some venues, notably ParaDex, document REST and WebSocket seq_no
                // values as independent streams. A terminal REST cancel ack must not
                // be ignored just because a prior private-order update used a larger
                // venue seq_no.
                entry.status = OrderStatus::Cancelled;
                entry.updated_ms = now_ms;
                entry.gap_grace_started_ms = None;
                entry.last_update_seq = match (entry.last_update_seq, seq) {
                    (Some(prev), Some(next)) => Some(prev.max(next)),
                    (Some(prev), None) => Some(prev),
                    (None, next) => next,
                };
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
        force_terminal_reject: bool,
    ) {
        let key = client_order_id
            .clone()
            .or_else(|| {
                exchange_order_id
                    .as_ref()
                    .and_then(|id| self.exchange_to_key.get(id).cloned())
            })
            .or_else(|| exchange_order_id.clone())
            .unwrap_or_else(|| format!("rejected_{}_{}", venue_index, now_ms));
        let entry = self.orders.entry(key).or_insert_with(|| LiveOrder {
            decision_id: None,
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
            gap_grace_started_ms: None,
            last_update_seq: seq,
        });
        if !force_terminal_reject {
            if let Some(prev) = entry.last_update_seq {
                if seq.is_some_and(|s| s <= prev) {
                    return;
                }
            }
        }
        entry.status = OrderStatus::Rejected;
        entry.remaining_qty = Some(0.0);
        entry.updated_ms = now_ms;
        entry.gap_grace_started_ms = None;
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
            .or_else(|| {
                exchange_order_id
                    .as_ref()
                    .and_then(|id| self.exchange_to_key.get(id).cloned())
            })
            .or_else(|| exchange_order_id.clone());
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
        entry.gap_grace_started_ms = None;
        entry.last_update_seq = seq;
    }

    fn snapshot_order_key(&self, order: &OpenOrderSnapshot) -> String {
        order
            .client_order_id
            .clone()
            .or_else(|| {
                order
                    .exchange_order_id
                    .as_ref()
                    .and_then(|id| self.exchange_to_key.get(id).cloned())
            })
            .or_else(|| self.exchange_to_key.get(&order.order_id).cloned())
            .or_else(|| order.exchange_order_id.clone())
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
        // Account-position syncs and private order updates can use independent
        // sequence domains. Attribute the inferred fill after the tracked order
        // update so live-order cleanup is not rejected as an older venue seq.
        let seq = order
            .last_update_seq
            .and_then(|prev| prev.checked_add(1))
            .unwrap_or(snapshot_seq)
            .max(snapshot_seq);
        Some(InferredFill {
            venue_index,
            venue_id: venue_id.to_string(),
            order_id: order.exchange_order_id.clone(),
            client_order_id: order.client_order_id.clone(),
            seq,
            side: order.side?,
            price: order.price?,
            size,
            purpose: order.purpose?,
        })
    }
}

fn snapshot_exchange_order_id(order: &OpenOrderSnapshot) -> Option<String> {
    order
        .exchange_order_id
        .clone()
        .or_else(|| (!order.order_id.starts_with("co_")).then_some(order.order_id.clone()))
}

fn supports_supported_replace_snapshot_gap_grace(venue_id: &str, order: &LiveOrder) -> bool {
    (venue_id.eq_ignore_ascii_case("hyperliquid")
        || venue_id.eq_ignore_ascii_case("lighter")
        || venue_id.eq_ignore_ascii_case("paradex")
        || venue_id.eq_ignore_ascii_case("extended"))
        && order.purpose == Some(OrderPurpose::Mm)
        && order.side.is_some()
}

fn supports_missing_snapshot_lower_seq_reconcile(venue_id: &str, order: &LiveOrder) -> bool {
    venue_id.eq_ignore_ascii_case("lighter")
        && supports_supported_replace_snapshot_gap_grace(venue_id, order)
}

fn supports_present_snapshot_lower_seq_reconcile(venue_id: &str, order: &LiveOrder) -> bool {
    venue_id.eq_ignore_ascii_case("lighter")
        && supports_supported_replace_snapshot_gap_grace(venue_id, order)
}

fn supported_replace_snapshot_gap_id_state(raw: Option<&str>) -> &'static str {
    match raw {
        Some(value) if !value.trim().is_empty() => "present_redacted",
        _ => "absent",
    }
}

fn emit_supported_replace_snapshot_gap_grace(
    venue_id: &str,
    side: Option<Side>,
    order_id: &str,
    client_order_id: Option<&str>,
) {
    eprintln!(
        "SUPPORTED_REPLACE_SNAPSHOT_GAP_GRACE venue={} side={} order_id_state={} client_id_state={}",
        venue_id,
        side.map(|value| format!("{value:?}"))
            .unwrap_or_else(|| "unknown".to_string()),
        supported_replace_snapshot_gap_id_state(Some(order_id)),
        supported_replace_snapshot_gap_id_state(client_order_id),
    );
}

fn emit_supported_replace_snapshot_gap_cleared(
    venue_id: &str,
    side: Option<Side>,
    order_id: &str,
    client_order_id: Option<&str>,
) {
    eprintln!(
        "SUPPORTED_REPLACE_SNAPSHOT_GAP_CLEARED venue={} side={} order_id_state={} client_id_state={}",
        venue_id,
        side.map(|value| format!("{value:?}"))
            .unwrap_or_else(|| "unknown".to_string()),
        supported_replace_snapshot_gap_id_state(Some(order_id)),
        supported_replace_snapshot_gap_id_state(client_order_id),
    );
}

fn emit_supported_replace_snapshot_gap_expired(
    venue_id: &str,
    side: Option<Side>,
    order_id: &str,
    client_order_id: Option<&str>,
) {
    eprintln!(
        "SUPPORTED_REPLACE_SNAPSHOT_GAP_EXPIRED venue={} side={} order_id_state={} client_id_state={}",
        venue_id,
        side.map(|value| format!("{value:?}"))
            .unwrap_or_else(|| "unknown".to_string()),
        supported_replace_snapshot_gap_id_state(Some(order_id)),
        supported_replace_snapshot_gap_id_state(client_order_id),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionEvent, FillEvent, OrderAck, OrderReject};

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
    fn supported_replace_snapshot_gap_log_id_state_redacts_raw_identifiers() {
        assert_eq!(
            supported_replace_snapshot_gap_id_state(Some("raw-order-id")),
            "present_redacted"
        );
        assert_eq!(
            supported_replace_snapshot_gap_id_state(Some("raw-client-id")),
            "present_redacted"
        );
        assert_eq!(supported_replace_snapshot_gap_id_state(Some("")), "absent");
        assert_eq!(
            supported_replace_snapshot_gap_id_state(Some("  ")),
            "absent"
        );
        assert_eq!(supported_replace_snapshot_gap_id_state(None), "absent");
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
                exchange_order_id: None,
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
                    exchange_order_id: None,
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
    fn reconcile_supported_mm_order_enters_snapshot_gap_grace_before_cancel() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                4,
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

        let empty_paradex_snapshot = OrderSnapshot {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            seq: 2,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile(&empty_paradex_snapshot, 2_000);

        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);
        assert_eq!(order.gap_grace_started_ms, Some(2_000));
        assert!(state.open_orders().is_empty());
        assert_eq!(state.active_orders().len(), 1);

        state.reconcile(
            &OrderSnapshot {
                seq: 3,
                timestamp_ms: 4_001,
                ..empty_paradex_snapshot
            },
            4_001,
        );
        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.gap_grace_started_ms, None);
    }

    #[test]
    fn reconcile_extended_mm_order_enters_snapshot_gap_grace_before_cancel() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "extended_oid_1",
                "co_extended_1",
                Side::Sell,
                101.0,
                0.01,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );

        let empty_extended_snapshot = OrderSnapshot {
            venue_index: 0,
            venue_id: "extended".to_string(),
            seq: 2,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile(&empty_extended_snapshot, 2_000);

        let order = state.orders.get("co_extended_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);
        assert_eq!(order.gap_grace_started_ms, Some(2_000));
        assert!(state.open_orders().is_empty());
        assert_eq!(state.active_orders().len(), 1);
    }

    #[test]
    fn lower_seq_lighter_missing_snapshot_reconciles_after_gap_grace() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_lighter",
                "co_lighter",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        let empty_lighter_snapshot = OrderSnapshot {
            venue_index: 1,
            venue_id: "lighter".to_string(),
            seq: 3,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile(&empty_lighter_snapshot, 2_000);

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);
        assert_eq!(order.gap_grace_started_ms, Some(2_000));
        assert_eq!(order.last_update_seq, Some(20_000));
        assert!(state.open_orders().is_empty());
        assert_eq!(state.active_orders().len(), 1);

        state.reconcile(
            &OrderSnapshot {
                timestamp_ms: 4_001,
                ..empty_lighter_snapshot
            },
            4_001,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.gap_grace_started_ms, None);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert!(state.open_order_ids_by_venue(1).is_empty());
    }

    #[test]
    fn lower_seq_lighter_present_snapshot_clears_gap_grace() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_lighter",
                "co_lighter",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.reconcile(
            &OrderSnapshot {
                venue_index: 1,
                venue_id: "lighter".to_string(),
                seq: 3,
                timestamp_ms: 2_000,
                open_orders: Vec::new(),
            },
            2_000,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);
        assert_eq!(order.gap_grace_started_ms, Some(2_000));
        assert!(state.open_orders().is_empty());

        state.reconcile(
            &OrderSnapshot {
                venue_index: 1,
                venue_id: "lighter".to_string(),
                seq: 3,
                timestamp_ms: 2_500,
                open_orders: vec![OpenOrderSnapshot {
                    order_id: "oid_lighter".to_string(),
                    client_order_id: Some("co_lighter".to_string()),
                    exchange_order_id: None,
                    side: Side::Buy,
                    price: 100.0,
                    size: 0.01,
                    purpose: Some(OrderPurpose::Mm),
                }],
            },
            2_500,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Accepted);
        assert_eq!(order.gap_grace_started_ms, None);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert_eq!(
            state.open_order_ids_by_venue(1),
            vec!["oid_lighter".to_string()]
        );
    }

    #[test]
    fn lower_seq_lighter_present_snapshot_restores_after_gap_expiry() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_lighter",
                "co_lighter",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        let empty_lighter_snapshot = OrderSnapshot {
            venue_index: 1,
            venue_id: "lighter".to_string(),
            seq: 3,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile_with_supported_replace_gap_grace_ms(&empty_lighter_snapshot, 2_000, 500);
        state.reconcile_with_supported_replace_gap_grace_ms(
            &OrderSnapshot {
                timestamp_ms: 2_501,
                ..empty_lighter_snapshot
            },
            2_501,
            500,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.remaining_qty, Some(0.0));
        assert_eq!(order.last_update_seq, Some(20_000));
        assert!(state.open_order_ids_by_venue(1).is_empty());

        state.reconcile_with_supported_replace_gap_grace_ms(
            &OrderSnapshot {
                venue_index: 1,
                venue_id: "lighter".to_string(),
                seq: 3,
                timestamp_ms: 3_000,
                open_orders: vec![OpenOrderSnapshot {
                    order_id: "oid_lighter".to_string(),
                    client_order_id: Some("co_lighter".to_string()),
                    exchange_order_id: None,
                    side: Side::Buy,
                    price: 100.0,
                    size: 0.01,
                    purpose: Some(OrderPurpose::Mm),
                }],
            },
            3_000,
            500,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Accepted);
        assert_eq!(order.remaining_qty, Some(0.01));
        assert_eq!(order.gap_grace_started_ms, None);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert_eq!(
            state.open_order_ids_by_venue(1),
            vec!["oid_lighter".to_string()]
        );
    }

    #[test]
    fn lower_seq_unsupported_missing_snapshot_does_not_clear_live_order() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                2,
                "oid_unknown",
                "co_unknown",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.reconcile(
            &OrderSnapshot {
                venue_index: 2,
                venue_id: "unknown_venue".to_string(),
                seq: 3,
                timestamp_ms: 2_000,
                open_orders: Vec::new(),
            },
            2_000,
        );

        let order = state.orders.get("co_unknown").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Accepted);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert_eq!(
            state.open_order_ids_by_venue(2),
            vec!["oid_unknown".to_string()]
        );
    }

    #[test]
    fn reconcile_supported_mm_order_honors_custom_snapshot_gap_grace() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                4,
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

        let empty_paradex_snapshot = OrderSnapshot {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            seq: 2,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile_with_supported_replace_gap_grace_ms(&empty_paradex_snapshot, 2_000, 4_000);
        state.reconcile_with_supported_replace_gap_grace_ms(
            &OrderSnapshot {
                seq: 3,
                timestamp_ms: 5_500,
                ..empty_paradex_snapshot
            },
            5_500,
            4_000,
        );

        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);
        assert_eq!(order.gap_grace_started_ms, Some(2_000));
    }

    #[test]
    fn lower_seq_cancel_ack_clears_supported_snapshot_gap_order() {
        let mut state = LiveOrderState::new();
        state.register_mm_decision_lineage(
            4,
            "co_1",
            Side::Sell,
            101.0,
            0.5,
            OrderPurpose::Mm,
            "d1_mm_v4_sell",
            500,
        );
        state.apply_execution_event(
            &ack(
                4,
                "oid_1",
                "co_1",
                Side::Sell,
                101.0,
                0.5,
                OrderPurpose::Mm,
                10_000,
            ),
            1_000,
        );

        let empty_paradex_snapshot = OrderSnapshot {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            seq: 20_000,
            timestamp_ms: 2_000,
            open_orders: Vec::new(),
        };
        state.reconcile_with_supported_replace_gap_grace_ms(&empty_paradex_snapshot, 2_000, 4_000);
        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::SnapshotGapGrace);

        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 4,
                venue_id: "paradex".into(),
                order_id: "oid_1".to_string(),
                client_order_id: None,
                seq: Some(1),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            2_500,
        );

        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.gap_grace_started_ms, None);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert!(state.active_orders().is_empty());
    }

    #[test]
    fn lower_seq_cancel_reject_clears_exchange_keyed_live_order() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_hl_1",
                "co_hl_1",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::OrderReject(OrderReject {
                venue_index: 1,
                venue_id: "hyperliquid".into(),
                order_id: Some("oid_hl_1".to_string()),
                client_order_id: None,
                seq: Some(2),
                purpose: None,
                reduce_only: None,
                reason: "sanitized_cancel_reject".to_string(),
            }),
            1_100,
        );

        let order = state.orders.get("co_hl_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Rejected);
        assert_eq!(order.remaining_qty, Some(0.0));
        assert_eq!(order.last_update_seq, Some(2));
        assert!(state.active_orders().is_empty());
    }

    #[test]
    fn exchange_order_id_fill_updates_client_keyed_live_order() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_hl_1",
                "co_hl_1",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::Fill(FillEvent {
                venue_index: 1,
                venue_id: "hyperliquid".into(),
                order_id: Some("oid_hl_1".to_string()),
                client_order_id: None,
                seq: Some(20_001),
                side: Side::Buy,
                price: 100.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                fee_bps: 0.0,
            }),
            1_100,
        );

        let order = state.orders.get("co_hl_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Filled);
        assert_eq!(order.remaining_qty, Some(0.0));
        assert_eq!(order.last_update_seq, Some(20_001));
        assert!(state.active_orders().is_empty());
    }

    #[test]
    fn lower_seq_place_reject_does_not_clear_accepted_live_order() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_hl_1",
                "co_hl_1",
                Side::Buy,
                100.0,
                0.01,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::OrderReject(OrderReject {
                venue_index: 1,
                venue_id: "hyperliquid".into(),
                order_id: Some("oid_hl_1".to_string()),
                client_order_id: None,
                seq: Some(2),
                purpose: Some(OrderPurpose::Mm),
                reduce_only: Some(false),
                reason: "sanitized_place_reject".to_string(),
            }),
            1_100,
        );

        let order = state.orders.get("co_hl_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Accepted);
        assert_eq!(order.remaining_qty, Some(0.01));
        assert_eq!(order.last_update_seq, Some(20_000));
        assert_eq!(state.active_orders().len(), 1);
    }

    #[test]
    fn venue_scoped_cancel_all_ack_clears_live_orders_for_that_venue() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                0,
                "oid_ext",
                "co_ext",
                Side::Sell,
                100.0,
                1.0,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );
        state.apply_execution_event(
            &ack(
                1,
                "oid_hl",
                "co_hl",
                Side::Buy,
                101.0,
                1.0,
                OrderPurpose::Mm,
                2,
            ),
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 0,
                venue_id: "extended".into(),
                order_id: "cancel_all".to_string(),
                client_order_id: None,
                seq: Some(3),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_100,
        );

        assert!(state.open_order_ids_by_venue(0).is_empty());
        assert_eq!(state.open_order_ids_by_venue(1), vec!["oid_hl".to_string()]);
    }

    #[test]
    fn lower_seq_cancel_all_ack_clears_live_orders_for_that_venue() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                1,
                "oid_lighter",
                "co_lighter",
                Side::Buy,
                100.0,
                1.0,
                OrderPurpose::Mm,
                20_000,
            ),
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 1,
                venue_id: "lighter".into(),
                order_id: "cancel_all".to_string(),
                client_order_id: None,
                seq: Some(3),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_100,
        );

        let order = state.orders.get("co_lighter").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.last_update_seq, Some(20_000));
        assert!(state.open_order_ids_by_venue(1).is_empty());
    }

    #[test]
    fn reconcile_supported_mm_order_clears_snapshot_gap_grace_when_order_returns() {
        let mut state = LiveOrderState::new();
        state.apply_execution_event(
            &ack(
                4,
                "oid_1",
                "co_1",
                Side::Buy,
                100.0,
                0.5,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );

        state.reconcile(
            &OrderSnapshot {
                venue_index: 4,
                venue_id: "paradex".to_string(),
                seq: 2,
                timestamp_ms: 2_000,
                open_orders: Vec::new(),
            },
            2_000,
        );
        state.reconcile(
            &OrderSnapshot {
                venue_index: 4,
                venue_id: "paradex".to_string(),
                seq: 3,
                timestamp_ms: 2_500,
                open_orders: vec![OpenOrderSnapshot {
                    order_id: "oid_1".to_string(),
                    client_order_id: Some("co_1".to_string()),
                    exchange_order_id: None,
                    side: Side::Buy,
                    price: 100.0,
                    size: 0.5,
                    purpose: Some(OrderPurpose::Mm),
                }],
            },
            2_500,
        );

        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Accepted);
        assert_eq!(order.gap_grace_started_ms, None);
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
    fn cancel_ack_with_only_client_key_clears_pending_order() {
        let mut state = LiveOrderState::new();
        state.register_mm_decision_lineage(
            0,
            "co_pending_1",
            Side::Buy,
            100.0,
            0.5,
            OrderPurpose::Mm,
            "d1_mm_v0_buy",
            1_000,
        );

        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 0,
                venue_id: "test".into(),
                order_id: "co_pending_1".to_string(),
                client_order_id: None,
                seq: Some(2),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_500,
        );

        let order = state.orders.get("co_pending_1").expect("tracked order");
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.remaining_qty, Some(0.5));
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

    #[test]
    fn account_position_delta_can_attribute_delayed_paradex_cancelled_order_fill() {
        let mut state = LiveOrderState::new();
        state.register_mm_decision_lineage(
            4,
            "co_pdx",
            Side::Buy,
            100.0,
            0.5,
            OrderPurpose::Mm,
            "d_pdx_mm_buy",
            1_000,
        );
        state.apply_execution_event(
            &ack(
                4,
                "oid_pdx",
                "co_pdx",
                Side::Buy,
                100.0,
                0.5,
                OrderPurpose::Mm,
                1,
            ),
            1_000,
        );
        state.apply_execution_event(
            &ExecutionEvent::OrderAck(OrderAck {
                venue_index: 4,
                venue_id: "paradex".into(),
                order_id: "oid_pdx".to_string(),
                client_order_id: Some("co_pdx".to_string()),
                seq: Some(2),
                side: None,
                price: None,
                size: None,
                purpose: None,
            }),
            1_100,
        );

        let inferred = state.infer_fills_from_position_delta(4, "paradex", 0.01, 3, 1_100 + 60_000);

        assert_eq!(inferred.len(), 1);
        assert_eq!(inferred[0].order_id.as_deref(), Some("oid_pdx"));
        assert_eq!(inferred[0].client_order_id.as_deref(), Some("co_pdx"));
        assert_eq!(inferred[0].side, Side::Buy);
        assert!((inferred[0].size - 0.01).abs() < 1e-9);
        assert_eq!(
            state.decision_id_for_order(Some("oid_pdx"), Some("co_pdx")),
            Some("d_pdx_mm_buy")
        );
    }

    #[test]
    fn registered_decision_lineage_survives_ack_and_fill_lookup() {
        let mut state = LiveOrderState::new();
        state.register_mm_decision_lineage(
            0,
            "co_1",
            Side::Buy,
            100.0,
            1.0,
            OrderPurpose::Mm,
            "d1_mm_v0_buy",
            900,
        );

        assert_eq!(
            state.decision_id_for_order(None, Some("co_1")),
            Some("d1_mm_v0_buy")
        );

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

        assert_eq!(
            state.decision_id_for_order(Some("oid_1"), Some("co_1")),
            Some("d1_mm_v0_buy")
        );

        state.apply_execution_event(
            &ExecutionEvent::Fill(crate::types::FillEvent {
                venue_index: 0,
                venue_id: "test".into(),
                order_id: Some("oid_1".to_string()),
                client_order_id: Some("co_1".to_string()),
                seq: Some(2),
                side: Side::Buy,
                price: 100.0,
                size: 0.5,
                purpose: OrderPurpose::Mm,
                fee_bps: 0.0,
            }),
            1_100,
        );

        assert_eq!(
            state.decision_id_for_order(Some("oid_1"), Some("co_1")),
            Some("d1_mm_v0_buy")
        );
    }

    #[test]
    fn reconcile_prefers_snapshot_exchange_order_id_when_client_id_backed_snapshot_arrives() {
        let mut state = LiveOrderState::new();

        state.reconcile(
            &OrderSnapshot {
                venue_index: 4,
                venue_id: "paradex".to_string(),
                seq: 1,
                timestamp_ms: 1_000,
                open_orders: vec![OpenOrderSnapshot {
                    order_id: "co_1".to_string(),
                    client_order_id: Some("co_1".to_string()),
                    exchange_order_id: Some("pdx_1".to_string()),
                    side: Side::Buy,
                    price: 100.0,
                    size: 0.5,
                    purpose: Some(OrderPurpose::Mm),
                }],
            },
            1_000,
        );

        let order = state.orders.get("co_1").expect("tracked order");
        assert_eq!(order.exchange_order_id.as_deref(), Some("pdx_1"));
        assert_eq!(state.open_order_ids_by_venue(4), vec!["pdx_1".to_string()]);
    }
}
