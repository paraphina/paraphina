#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::execution_events::apply_execution_events;
    use paraphina::live::ops::{HealthState, LiveMetrics};
    use paraphina::live::runner::{run_live_loop, LiveChannels, LiveRunMode, LiveRuntimeHooks};
    use paraphina::live::shadow_adapter::ShadowAckAdapter;
    use paraphina::live::types::ExecutionEvent as LiveExecutionEvent;
    use paraphina::state::GlobalState;
    use paraphina::types::{ExecutionEvent, OrderIntent, OrderPurpose, Side, TimeInForce};
    use tokio::sync::mpsc;

    fn core_event_from_live(event: &LiveExecutionEvent) -> Option<ExecutionEvent> {
        match event {
            LiveExecutionEvent::OrderAccepted(ack) => {
                Some(ExecutionEvent::OrderAck(paraphina::types::OrderAck {
                    venue_index: ack.venue_index,
                    venue_id: ack.venue_id.as_str().into(),
                    order_id: ack.order_id.clone(),
                    client_order_id: ack.client_order_id.clone(),
                    seq: Some(ack.seq),
                    side: Some(ack.side),
                    price: Some(ack.price),
                    size: Some(ack.size),
                    purpose: Some(ack.purpose),
                }))
            }
            LiveExecutionEvent::CancelAccepted(cancel) => {
                Some(ExecutionEvent::OrderAck(paraphina::types::OrderAck {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.as_str().into(),
                    order_id: cancel.order_id.clone(),
                    client_order_id: None,
                    seq: Some(cancel.seq),
                    side: None,
                    price: None,
                    size: None,
                    purpose: None,
                }))
            }
            LiveExecutionEvent::OrderRejected(rej) => {
                Some(ExecutionEvent::OrderReject(paraphina::types::OrderReject {
                    venue_index: rej.venue_index,
                    venue_id: rej.venue_id.as_str().into(),
                    order_id: rej.order_id.clone(),
                    client_order_id: None,
                    seq: Some(rej.seq),
                    reason: rej.reason.clone(),
                }))
            }
            LiveExecutionEvent::CancelRejected(rej) => {
                Some(ExecutionEvent::OrderReject(paraphina::types::OrderReject {
                    venue_index: rej.venue_index,
                    venue_id: rej.venue_id.as_str().into(),
                    order_id: rej.order_id.clone(),
                    client_order_id: None,
                    seq: Some(rej.seq),
                    reason: rej.reason.clone(),
                }))
            }
            _ => None,
        }
    }

    #[test]
    fn shadow_adapter_deterministic_ids_and_open_order_stability() {
        let cfg = Config::default();
        let mut adapter = ShadowAckAdapter::new(&cfg);
        let mut adapter_clone = ShadowAckAdapter::new(&cfg);

        let place = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: None,
        });

        let events = adapter.handle_intents(vec![place.clone()], 42, 1_000);
        let events_clone = adapter_clone.handle_intents(vec![place.clone()], 42, 1_000);

        let id = events
            .iter()
            .find_map(|event| match event {
                LiveExecutionEvent::OrderAccepted(ack) => Some(ack.order_id.clone()),
                _ => None,
            })
            .expect("order id");
        let id_clone = events_clone
            .iter()
            .find_map(|event| match event {
                LiveExecutionEvent::OrderAccepted(ack) => Some(ack.order_id.clone()),
                _ => None,
            })
            .expect("order id clone");
        assert_eq!(id, id_clone);

        let mut state = GlobalState::new(&cfg);
        let core_events: Vec<_> = events.iter().filter_map(core_event_from_live).collect();
        let _ = apply_execution_events(&mut state, &core_events, 1_000);
        assert_eq!(state.venues[0].open_orders.len(), 1);

        let ioc = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Sell,
            price: 101.0,
            size: 1.0,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: false,
            client_order_id: None,
        });
        let ioc_events = adapter.handle_intents(vec![ioc], 43, 1_001);
        let core_ioc_events: Vec<_> = ioc_events.iter().filter_map(core_event_from_live).collect();
        let _ = apply_execution_events(&mut state, &core_ioc_events, 1_001);
        assert_eq!(state.venues[0].open_orders.len(), 1);
    }

    #[tokio::test]
    async fn live_runner_tick_metrics_monotonic() {
        let cfg = Config::default();
        let metrics = LiveMetrics::new();
        let hooks = LiveRuntimeHooks {
            metrics: metrics.clone(),
            health: HealthState::new(),
            telemetry: None,
        };

        let (_market_tx, market_rx) = mpsc::channel(8);
        let (_account_tx, account_rx) = mpsc::channel(8);
        let (_exec_tx, exec_rx) = mpsc::channel(8);
        let (order_tx, _order_rx) = mpsc::channel(8);
        let (_snapshot_tx, order_snapshot_rx) = mpsc::channel(8);

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            order_tx,
            order_snapshot_rx: Some(order_snapshot_rx),
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms: 1_000,
                step_ms: 250,
                ticks: 5,
            },
            Some(hooks),
        )
        .await;

        assert_eq!(summary.ticks_run, 5);
        let metrics_text = metrics.gather();
        let mut last_tick_ms = None;
        let mut tick_total = None;
        for line in metrics_text.lines() {
            if line.starts_with("paraphina_live_last_tick_ms") {
                last_tick_ms = line
                    .split_whitespace()
                    .last()
                    .and_then(|v| v.parse::<i64>().ok());
            }
            if line.starts_with("paraphina_live_ticks") {
                tick_total = line
                    .split_whitespace()
                    .last()
                    .and_then(|v| v.parse::<i64>().ok());
            }
        }
        assert_eq!(tick_total, Some(5));
        assert_eq!(last_tick_ms, Some(1_000 + 250 * 4));
    }
}
