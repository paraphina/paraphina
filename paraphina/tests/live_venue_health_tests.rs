#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::live::mock_exchange::{spawn_mock_exchange, MockExchangeConfig};
    use paraphina::live::orderbook_l2::BookSide;
    use paraphina::live::runner::{run_live_loop, LiveChannels, LiveRunMode};
    use paraphina::live::types::MarketDataEvent;
    use paraphina::live::BookLevelDelta;
    use paraphina::types::{OrderIntent, OrderPurpose, Side, TimeInForce};

    #[tokio::test]
    async fn disable_triggers_cancel_all() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;

        // Open an MM order.
        let (response_tx, _response_rx) = tokio::sync::oneshot::channel();
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
            client_order_id: Some("health_mm_1".to_string()),
        });
        let request = paraphina::live::runner::LiveOrderRequest {
            intents: vec![intent],
            action_batch: paraphina::actions::ActionBatch::new(0, 0, "test"),
            now_ms: 1_000,
            response: response_tx,
        };
        handle.order_tx.send(request).await.expect("send order");
        for _ in 0..100 {
            if handle.open_orders_len().await > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }

        let (market_tx, market_rx) = tokio::sync::mpsc::channel::<MarketDataEvent>(32);
        let (_account_tx, account_rx) =
            tokio::sync::mpsc::channel::<paraphina::live::types::AccountEvent>(8);
        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            order_tx: handle.order_tx.clone(),
            order_snapshot_rx: None,
        };

        let live_cfg = cfg.clone();
        let live_task = tokio::spawn(async move {
            run_live_loop(
                &live_cfg,
                channels,
                LiveRunMode::Step {
                    start_ms: 1_000,
                    step_ms: 100,
                    ticks: 1,
                },
                None,
            )
            .await;
        });

        // Inject invalid deltas to trip API error threshold.
        for _ in 0..3 {
            let delta = paraphina::live::types::L2Delta {
                venue_index: 0,
                venue_id: cfg.venues[0].id.clone(),
                seq: 0,
                timestamp_ms: 1_000,
                changes: vec![BookLevelDelta {
                    side: BookSide::Bid,
                    price: 98.0,
                    size: 1.0,
                }],
            };
            let _ = market_tx.send(MarketDataEvent::L2Delta(delta)).await;
        }

        let _ = live_task.await;
        assert_eq!(handle.open_orders_len().await, 0);
    }
}
