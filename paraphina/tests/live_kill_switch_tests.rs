#[cfg(feature = "live")]
mod tests {
    use paraphina::actions::ActionBatch;
    use paraphina::config::Config;
    use paraphina::execution_events::apply_execution_events;
    use paraphina::live::mock_exchange::{spawn_mock_exchange, MockExchangeConfig};
    use paraphina::live::runner::{handle_kill_switch, LiveOrderRequest};
    use paraphina::types::{ExecutionEvent, OrderIntent, OrderPurpose, Side, TimeInForce};

    #[tokio::test]
    async fn kill_switch_cancel_all_only() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let mut state = paraphina::state::GlobalState::new(&cfg);

        // Create an open MM order.
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
            client_order_id: Some("mm_kill_1".to_string()),
        });
        handle
            .order_tx
            .send(LiveOrderRequest {
                intents: vec![intent],
                action_batch: ActionBatch::new(1_000, 0, &cfg.version),
                now_ms: 1_000,
                response: response_tx,
            })
            .await
            .expect("order send");
        let mut events = None;
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(val) => {
                    events = Some(val);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await
                }
            }
        }
        let events = events.expect("order response");
        let core_events = map_live_to_core(&events);
        let fills = apply_execution_events(&mut state, &core_events, 1_000);
        for fill in &fills {
            state.apply_fill_event(fill, 1_000, &cfg);
        }
        state.recompute_after_fills(&cfg);

        state.kill_switch = true;
        state.kill_reason = paraphina::state::KillReason::DeltaHardBreach;
        let audit_dir = std::path::PathBuf::from("/tmp/paraphina_kill_audit_test_missing");
        handle_kill_switch(
            &cfg,
            &mut state,
            &handle.order_tx,
            1_100,
            1,
            false,
            None,
            &audit_dir,
        )
        .await;
        assert_eq!(handle.open_orders_len().await, 0);
    }

    #[tokio::test]
    async fn kill_switch_best_effort_flatten_reduces_position() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let mut state = paraphina::state::GlobalState::new(&cfg);

        // Create a long position via IOC buy.
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 101.0,
            size: 2.0,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: false,
            client_order_id: Some("kill_open_long".to_string()),
        });
        handle
            .order_tx
            .send(LiveOrderRequest {
                intents: vec![intent],
                action_batch: ActionBatch::new(2_000, 0, &cfg.version),
                now_ms: 2_000,
                response: response_tx,
            })
            .await
            .expect("order send");
        let mut events = None;
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(val) => {
                    events = Some(val);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await
                }
            }
        }
        let events = events.expect("order response");
        let core_events = map_live_to_core(&events);
        let fills = apply_execution_events(&mut state, &core_events, 2_000);
        for fill in &fills {
            state.apply_fill_event(fill, 2_000, &cfg);
        }
        state.recompute_after_fills(&cfg);

        state.kill_switch = true;
        state.kill_reason = paraphina::state::KillReason::LiquidationDistanceBreach;
        state.venues[0].status = paraphina::types::VenueStatus::Healthy;
        state.venues[0].dist_liq_sigma = cfg.risk.liq_crit_sigma - 0.1;
        state.venues[0].mid = Some(100.0);
        state.venues[0].spread = Some(0.5);

        let before = handle.position(0).await.abs();
        let audit_dir = std::path::PathBuf::from("/tmp/paraphina_kill_audit_test_missing");
        handle_kill_switch(
            &cfg,
            &mut state,
            &handle.order_tx,
            2_100,
            2,
            true,
            None,
            &audit_dir,
        )
        .await;
        let after = handle.position(0).await.abs();
        assert!(
            after <= before,
            "position should not increase under reduce-only flatten"
        );
    }

    #[tokio::test]
    async fn kill_switch_cancel_all_clears_ledger_hl() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let mut state = paraphina::state::GlobalState::new(&cfg);
        let venue_index = cfg
            .venues
            .iter()
            .position(|venue| venue.id == "hyperliquid")
            .unwrap_or(0);

        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index,
            venue_id: cfg.venues[venue_index].id_arc.clone(),
            side: Side::Buy,
            price: 101.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("hl_kill_1".to_string()),
        });
        handle
            .order_tx
            .send(LiveOrderRequest {
                intents: vec![intent],
                action_batch: ActionBatch::new(3_000, 0, &cfg.version),
                now_ms: 3_000,
                response: response_tx,
            })
            .await
            .expect("order send");
        let mut events = None;
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(val) => {
                    events = Some(val);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await
                }
            }
        }
        let events = events.expect("order response");
        let core_events = map_live_to_core(&events);
        let fills = apply_execution_events(&mut state, &core_events, 3_000);
        for fill in &fills {
            state.apply_fill_event(fill, 3_000, &cfg);
        }
        state.recompute_after_fills(&cfg);
        assert_eq!(handle.open_orders_len().await, 1);

        state.kill_switch = true;
        state.kill_reason = paraphina::state::KillReason::DeltaHardBreach;
        let audit_dir = std::path::PathBuf::from("/tmp/paraphina_kill_audit_test_missing");
        handle_kill_switch(
            &cfg,
            &mut state,
            &handle.order_tx,
            3_100,
            3,
            false,
            None,
            &audit_dir,
        )
        .await;
        assert_eq!(handle.open_orders_len().await, 0);
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
                _ => {}
            }
        }
        out
    }
}
