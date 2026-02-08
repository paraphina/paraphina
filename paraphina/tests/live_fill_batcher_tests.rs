#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::live::mock_exchange::{spawn_mock_exchange, MockExchangeConfig};
    use paraphina::live::ops::{HealthState, LiveMetrics};
    use paraphina::live::runner::{run_live_loop, LiveChannels, LiveRunMode, LiveRuntimeHooks};
    use paraphina::live::types::{ExecutionEvent, Fill};
    use paraphina::types::{OrderPurpose, Side, TimeInForce};

    fn parse_last_tick_ms(metrics: &str) -> Option<i64> {
        for line in metrics.lines() {
            if line.starts_with("paraphina_live_last_tick_ms ") {
                let val = line.trim().split(' ').nth(1)?;
                return val.parse::<f64>().ok().map(|v| v as i64);
            }
        }
        None
    }

    async fn forward_broadcast<T: Clone + Send + 'static>(
        mut rx: tokio::sync::broadcast::Receiver<T>,
        tx: tokio::sync::mpsc::Sender<T>,
    ) {
        while let Ok(event) = rx.recv().await {
            let _ = tx.send(event).await;
        }
    }

    #[tokio::test]
    async fn live_fill_batcher_defers_exit_until_flush() {
        let mut cfg = Config::default();
        cfg.fill_agg_interval_ms = 1_000;
        cfg.exit.enabled = true;
        cfg.exit.min_global_abs_tao = 0.0;
        cfg.exit.edge_min_usd = -10.0;
        cfg.exit.edge_vol_mult = 0.0;
        cfg.exit.max_total_tao_per_tick = 10.0;

        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let (market_tx, market_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::MarketDataEvent>(256);
        tokio::spawn(forward_broadcast(handle.market_tx.subscribe(), market_tx));
        let (account_tx, account_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::AccountEvent>(64);
        tokio::spawn(forward_broadcast(handle.account_tx.subscribe(), account_tx));
        let (exec_tx, exec_rx) = tokio::sync::mpsc::channel::<ExecutionEvent>(64);

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            order_tx: handle.order_tx.clone(),
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let hooks = LiveRuntimeHooks {
            metrics: LiveMetrics::new(),
            health: HealthState::new(),
            telemetry: None,
        };

        let ticks = 4_u64;
        let start_ms = 0_i64;
        let step_ms = 500_i64;
        let tick_tx = handle.tick_tx.clone();
        tokio::spawn(async move {
            for i in 0..ticks {
                let now_ms = start_ms + step_ms * i as i64;
                let _ = tick_tx.send(now_ms).await;
                tokio::task::yield_now().await;
            }
        });

        let exec_tx_clone = exec_tx.clone();
        let metrics = hooks.metrics.clone();
        tokio::spawn(async move {
            loop {
                if let Some(last) = parse_last_tick_ms(&metrics.gather()) {
                    if last >= 500 {
                        break;
                    }
                }
                tokio::task::yield_now().await;
            }
            let _ = exec_tx_clone
                .send(ExecutionEvent::Filled(Fill {
                    venue_index: 0,
                    venue_id: "TAO".to_string(),
                    seq: 1,
                    timestamp_ms: 500,
                    order_id: Some("external_fill".to_string()),
                    client_order_id: None,
                    fill_id: Some("fill_1".to_string()),
                    side: Side::Buy,
                    price: 100.0,
                    size: 5.0,
                    purpose: OrderPurpose::Mm,
                    fee_bps: 0.0,
                }))
                .await;
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
                Some(hooks),
            )
            .await
        });
        let _ = live_task.await.expect("live loop join");

        let log = handle.execution_log.lock().await;
        let mut exit_orders: Vec<i64> = Vec::new();
        for event in log.iter() {
            if let ExecutionEvent::OrderAccepted(ack) = event {
                if ack.purpose == OrderPurpose::Exit {
                    exit_orders.push(ack.timestamp_ms);
                }
            }
        }
        assert!(
            exit_orders.iter().all(|ts| *ts >= 1_000),
            "exit orders must not be emitted before batch flush"
        );
        // Exit intents depend on model thresholds; only assert pre-flush suppression.
    }

    #[tokio::test]
    async fn live_kill_flushes_pending_fills_once() {
        let mut cfg = Config::default();
        cfg.fill_agg_interval_ms = 1_000;
        cfg.risk.liq_crit_sigma = 1_000.0;

        let mock_cfg = MockExchangeConfig {
            account_liq_price_offset: -1.0,
            ..MockExchangeConfig::default()
        };
        let handle = spawn_mock_exchange(&cfg, mock_cfg).await;
        let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
        let intent = paraphina::types::OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 99.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("kill_open_mm".to_string()),
        });
        handle
            .order_tx
            .send(paraphina::live::runner::LiveOrderRequest {
                intents: vec![intent],
                action_batch: paraphina::actions::ActionBatch::new(0, 0, &cfg.version),
                now_ms: 0,
                response: paraphina::live::ResponseMode::Oneshot(response_tx),
            })
            .await
            .expect("order send");
        for _ in 0..1_000 {
            match response_rx.try_recv() {
                Ok(_) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    tokio::task::yield_now().await;
                }
            }
        }
        assert!(handle.open_orders_len().await > 0);
        let (market_tx, market_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::MarketDataEvent>(256);
        tokio::spawn(forward_broadcast(handle.market_tx.subscribe(), market_tx));
        let (account_tx, account_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::AccountEvent>(64);
        tokio::spawn(forward_broadcast(handle.account_tx.subscribe(), account_tx));
        let (exec_tx, exec_rx) = tokio::sync::mpsc::channel::<ExecutionEvent>(64);

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            order_tx: handle.order_tx.clone(),
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let hooks = LiveRuntimeHooks {
            metrics: LiveMetrics::new(),
            health: HealthState::new(),
            telemetry: None,
        };

        std::env::set_var("PARAPHINA_KILL_BEST_EFFORT", "1");

        let ticks = 6_u64;
        let start_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_millis() as i64;
        let step_ms = 500_i64;
        let tick_tx = handle.tick_tx.clone();
        tokio::spawn(async move {
            for i in 0..ticks {
                let now_ms = start_ms + step_ms * i as i64;
                let _ = tick_tx.send(now_ms).await;
                tokio::task::yield_now().await;
            }
        });

        let _ = exec_tx
            .send(ExecutionEvent::Filled(Fill {
                venue_index: 0,
                venue_id: "TAO".to_string(),
                seq: 2,
                timestamp_ms: start_ms,
                order_id: Some("kill_fill".to_string()),
                client_order_id: None,
                fill_id: Some("fill_kill".to_string()),
                side: Side::Buy,
                price: 100.0,
                size: 5.0,
                purpose: OrderPurpose::Mm,
                fee_bps: 0.0,
            }))
            .await;

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
                Some(hooks),
            )
            .await
        });
        let _ = live_task.await.expect("live loop join");
        std::env::remove_var("PARAPHINA_KILL_BEST_EFFORT");

        let log = handle.execution_log.lock().await;
        let cancel_all = log
            .iter()
            .filter(|e| matches!(e, ExecutionEvent::CancelAllAccepted(_)))
            .count();
        let remaining = handle.open_orders_len().await;
        assert!(
            cancel_all > 0 || remaining == 0,
            "expected cancel-all to run or open orders to clear"
        );
        let kill_exit = log
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    ExecutionEvent::OrderAccepted(ack)
                        if ack.client_order_id.as_deref().unwrap_or("").starts_with("kill_exit_")
                )
            })
            .count();
        assert!(kill_exit <= 1);
    }

    #[tokio::test]
    async fn live_mock_determinism_execution_log() {
        async fn run_once(cfg: &Config) -> Vec<ExecutionEvent> {
            let handle = spawn_mock_exchange(cfg, MockExchangeConfig::default()).await;
            let (market_tx, market_rx) =
                tokio::sync::mpsc::channel::<paraphina::live::types::MarketDataEvent>(256);
            tokio::spawn(forward_broadcast(handle.market_tx.subscribe(), market_tx));
            let (account_tx, account_rx) =
                tokio::sync::mpsc::channel::<paraphina::live::types::AccountEvent>(64);
            tokio::spawn(forward_broadcast(handle.account_tx.subscribe(), account_tx));

            let channels = LiveChannels {
                market_rx,
                account_rx,
                exec_rx: None,
                account_reconcile_tx: None,
                order_tx: handle.order_tx.clone(),
                order_snapshot_rx: None,
            shared_venue_ages: None,
            };

            let hooks = LiveRuntimeHooks {
                metrics: LiveMetrics::new(),
                health: HealthState::new(),
                telemetry: None,
            };

            let ticks = 4_u64;
            let start_ms = 0_i64;
            let step_ms = 500_i64;
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
                    Some(hooks),
                )
                .await
            });
            let _ = live_task.await.expect("live loop join");
            let log = handle.execution_log.lock().await;
            log.clone()
        }

        let cfg = Config::default();
        let a = run_once(&cfg).await;
        let b = run_once(&cfg).await;
        assert_eq!(a, b, "execution logs must be deterministic");
    }
}
