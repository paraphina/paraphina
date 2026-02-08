#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::live::mock_exchange::{
        spawn_mock_exchange, MockExchangeCommand, MockExchangeConfig,
    };
    use paraphina::live::runner::{run_live_loop, LiveChannels, LiveRunMode};
    use paraphina::live::types::ExecutionEvent;
    use paraphina::types::{OrderIntent, OrderPurpose, Side, TimeInForce};
    use std::collections::HashMap;

    #[tokio::test]
    async fn live_mock_exchange_semantics() {
        let cfg = Config::default();
        let mock_cfg = MockExchangeConfig {
            levels: 3,
            level_size: 2.0,
            mid_price: 100.0,
            spread: 1.0,
            tick_size: 0.5,
            taker_fee_bps: 5.0,
            account_balance_usd: 10_000.0,
            account_used_usd: 0.0,
            account_available_usd: 10_000.0,
            account_liq_price_offset: -10.0,
            account_funding_8h: 0.0,
        };
        let handle = spawn_mock_exchange(&cfg, mock_cfg).await;
        let mut market_sub = handle.market_tx.subscribe();
        let mut account_sub = handle.account_tx.subscribe();
        let (market_tx, market_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::MarketDataEvent>(256);
        tokio::spawn(async move {
            while let Ok(event) = market_sub.recv().await {
                let _ = market_tx.send(event).await;
            }
        });
        let (account_tx, account_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::AccountEvent>(64);
        tokio::spawn(async move {
            while let Ok(event) = account_sub.recv().await {
                let _ = account_tx.send(event).await;
            }
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            order_tx: handle.order_tx.clone(),
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let ticks = 6_u64;
        let start_ms = 1_000_i64;
        let step_ms = 100_i64;
        let tick_tx = handle.tick_tx.clone();
        tokio::spawn(async move {
            for i in 0..ticks {
                let now_ms = start_ms + step_ms * i as i64;
                let _ = tick_tx.send(now_ms).await;
                tokio::task::yield_now().await;
            }
        });

        let cfg_clone = cfg.clone();
        let live_task = tokio::spawn(async move {
            run_live_loop(
                &cfg_clone,
                channels,
                LiveRunMode::Step {
                    start_ms,
                    step_ms,
                    ticks,
                },
                None,
            )
            .await
        });
        let mut finished = false;
        for _ in 0..1_000 {
            if live_task.is_finished() {
                finished = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        if !finished {
            live_task.abort();
            panic!("live loop did not finish deterministically");
        }
        let _ = live_task.await.expect("live loop join");

        // Inject explicit IOC orders to validate execution semantics.
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let ioc_intents = vec![
            OrderIntent::Place(paraphina::types::PlaceOrderIntent {
                venue_index: 0,
                venue_id: cfg.venues[0].id_arc.clone(),
                side: Side::Buy,
                price: 101.0,
                size: 10.0,
                purpose: OrderPurpose::Exit,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("exit_ioc_1".to_string()),
            }),
            OrderIntent::Place(paraphina::types::PlaceOrderIntent {
                venue_index: 0,
                venue_id: cfg.venues[0].id_arc.clone(),
                side: Side::Sell,
                price: 99.0,
                size: 10.0,
                purpose: OrderPurpose::Hedge,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("hedge_ioc_1".to_string()),
            }),
        ];
        let request = paraphina::live::runner::LiveOrderRequest {
            intents: ioc_intents,
            action_batch: paraphina::actions::ActionBatch::new(
                start_ms + step_ms * ticks as i64,
                0,
                &cfg.version,
            ),
            now_ms: start_ms + step_ms * ticks as i64,
            response: paraphina::live::ResponseMode::Oneshot(response_tx),
        };
        let _ = handle.order_tx.send(request).await;
        let _ = handle
            .tick_tx
            .send(start_ms + step_ms * (ticks as i64 + 1))
            .await;
        let mut events: Option<Vec<ExecutionEvent>> = None;
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(val) => {
                    events = Some(val);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await;
                }
            }
        }
        let events = events.expect("ioc response");

        // Post-only MM orders should never produce taker fills.
        {
            let log = handle.execution_log.lock().await;
            assert!(
                !log.iter().any(|e| matches!(e, ExecutionEvent::Filled(fill) if fill.purpose == OrderPurpose::Mm)),
                "MM orders should not fill as takers under post-only semantics"
            );
        }

        // IOC orders should either fill immediately or cancel remainder.
        let mut order_sizes: HashMap<String, f64> = HashMap::new();
        let mut order_fills: HashMap<String, f64> = HashMap::new();
        let mut order_cancelled: HashMap<String, bool> = HashMap::new();
        let mut order_accepted: HashMap<String, bool> = HashMap::new();
        for event in &events {
            match event {
                ExecutionEvent::OrderAccepted(ack) => {
                    order_sizes.insert(ack.order_id.clone(), ack.size);
                    order_accepted.insert(ack.order_id.clone(), true);
                }
                ExecutionEvent::Filled(fill) => {
                    if let Some(order_id) = &fill.order_id {
                        *order_fills.entry(order_id.clone()).or_insert(0.0) += fill.size;
                    }
                }
                ExecutionEvent::CancelAccepted(cancel) => {
                    order_cancelled.insert(cancel.order_id.clone(), true);
                }
                _ => {}
            }
        }
        for (order_id, size) in order_sizes {
            if !order_accepted.get(&order_id).copied().unwrap_or(false) {
                continue;
            }
            let filled = order_fills.get(&order_id).copied().unwrap_or(0.0);
            if filled < size {
                assert!(
                    order_cancelled.get(&order_id).copied().unwrap_or(false),
                    "IOC order remainder should be cancelled"
                );
            }
        }

        // Cancel-all should clear open orders.
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        handle
            .command_tx
            .send(MockExchangeCommand::CancelAll {
                venue_index: Some(0),
                now_ms: start_ms + step_ms * (ticks as i64 + 2),
                response: cancel_tx,
            })
            .await
            .expect("cancel-all send");
        let _ = handle
            .tick_tx
            .send(start_ms + step_ms * (ticks as i64 + 3))
            .await;
        for _ in 0..1_000 {
            match cancel_rx.try_recv() {
                Ok(_) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await;
                }
            }
        }
        assert_eq!(handle.open_orders_len().await, 0);
    }
}
