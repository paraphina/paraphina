#[cfg(feature = "live")]
mod tests {
    use paraphina::actions::ActionBatch;
    use paraphina::config::Config;
    use paraphina::execution_events::apply_execution_events;
    use paraphina::live::mock_exchange::{spawn_mock_exchange, MockExchangeConfig};
    use paraphina::live::order_state::OrderStatus;
    use paraphina::live::runner::LiveOrderRequest;
    use paraphina::types::{ExecutionEvent, OrderIntent, OrderPurpose, Side, TimeInForce};

    #[tokio::test]
    async fn reconciliation_recovers_open_orders_after_ws_drop() {
        let cfg = Config::default();
        let mock_cfg = MockExchangeConfig::default();
        let handle = spawn_mock_exchange(&cfg, mock_cfg).await;

        let mut state = paraphina::state::GlobalState::new(&cfg);
        let now_ms = 1_000;

        // Place a GTC order via mock exchange.
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 99.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("mm_drop_1".to_string()),
        });
        let request = LiveOrderRequest {
            intents: vec![intent],
            action_batch: ActionBatch::new(now_ms, 0, &cfg.version),
            now_ms,
            response: response_tx,
        };
        handle.order_tx.send(request).await.expect("send order");
        let mut events = None;
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(val) => {
                    events = Some(val);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await;
                }
            }
        }
        let events = events.expect("order response");
        let core_events = map_live_to_core(&events);
        let _ = apply_execution_events(&mut state, &core_events, now_ms);

        // Simulate WS drop: cancel-all on exchange but do not apply cancel event.
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        handle
            .command_tx
            .send(
                paraphina::live::mock_exchange::MockExchangeCommand::CancelAll {
                    venue_index: Some(0),
                    now_ms: now_ms + 100,
                    response: cancel_tx,
                },
            )
            .await
            .expect("cancel-all send");
        let _ = cancel_rx.await;

        // Reconcile from snapshot.
        let snapshot = handle
            .snapshot_open_orders(0, cfg.venues[0].id.to_string(), now_ms + 200, 5)
            .await;
        state.live_order_state.reconcile(&snapshot, now_ms + 200);

        // Open orders should be empty after reconciliation.
        assert!(state.live_order_state.open_orders().is_empty());

        // No fills were applied, positions should be unchanged.
        assert_eq!(state.q_global_tao, 0.0);

        // Ensure any tracked order is marked cancelled.
        for order in state.live_order_state.open_orders() {
            assert_ne!(order.status, OrderStatus::Accepted);
        }
    }

    fn map_live_to_core(events: &[paraphina::live::types::ExecutionEvent]) -> Vec<ExecutionEvent> {
        let mut out = Vec::new();
        for event in events {
            match event {
                paraphina::live::types::ExecutionEvent::OrderAccepted(ack) => {
                    out.push(ExecutionEvent::OrderAck(paraphina::types::OrderAck {
                        venue_index: ack.venue_index,
                        venue_id: ack.venue_id.as_str().into(),
                        order_id: ack.order_id.clone(),
                        client_order_id: ack.client_order_id.clone(),
                        seq: Some(ack.seq),
                        side: Some(ack.side),
                        price: Some(ack.price),
                        size: Some(ack.size),
                        purpose: Some(ack.purpose),
                    }));
                }
                paraphina::live::types::ExecutionEvent::OrderRejected(rej) => {
                    out.push(ExecutionEvent::OrderReject(paraphina::types::OrderReject {
                        venue_index: rej.venue_index,
                        venue_id: rej.venue_id.as_str().into(),
                        order_id: rej.order_id.clone(),
                        client_order_id: None,
                        seq: Some(rej.seq),
                        reason: rej.reason.clone(),
                    }));
                }
                paraphina::live::types::ExecutionEvent::Filled(fill) => {
                    out.push(ExecutionEvent::Fill(paraphina::types::FillEvent {
                        venue_index: fill.venue_index,
                        venue_id: fill.venue_id.as_str().into(),
                        order_id: fill.order_id.clone(),
                        client_order_id: fill.client_order_id.clone(),
                        seq: Some(fill.seq),
                        side: fill.side,
                        price: fill.price,
                        size: fill.size,
                        purpose: fill.purpose,
                        fee_bps: fill.fee_bps,
                    }));
                }
                paraphina::live::types::ExecutionEvent::CancelAccepted(cancel) => {
                    out.push(ExecutionEvent::OrderAck(paraphina::types::OrderAck {
                        venue_index: cancel.venue_index,
                        venue_id: cancel.venue_id.as_str().into(),
                        order_id: cancel.order_id.clone(),
                        client_order_id: None,
                        seq: Some(cancel.seq),
                        side: None,
                        price: None,
                        size: None,
                        purpose: None,
                    }));
                }
                paraphina::live::types::ExecutionEvent::CancelRejected(rej) => {
                    out.push(ExecutionEvent::OrderReject(paraphina::types::OrderReject {
                        venue_index: rej.venue_index,
                        venue_id: rej.venue_id.as_str().into(),
                        order_id: rej.order_id.clone(),
                        client_order_id: None,
                        seq: Some(rej.seq),
                        reason: rej.reason.clone(),
                    }));
                }
                _ => {}
            }
        }
        out
    }
}
