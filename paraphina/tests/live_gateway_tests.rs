#[cfg(feature = "live")]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use paraphina::actions::ActionBatch;
    use paraphina::config::Config;
    use paraphina::io::GatewayPolicy;
    use paraphina::live::gateway::{
        LiveGateway, LiveGatewayError, LiveRestClient, LiveRestResponse,
    };
    use paraphina::live::mock_exchange::{
        spawn_mock_exchange, MockExchangeConfig, MockExchangeHandle,
    };
    use paraphina::live::ops::LiveMetrics;
    use paraphina::live::TradeMode;
    use paraphina::types::{OrderIntent, OrderPurpose, Side, TimeInForce};

    #[derive(Clone)]
    struct MockRestClient {
        handle: MockExchangeHandle,
        failures: Arc<Mutex<VecDeque<LiveGatewayError>>>,
        place_calls: Arc<Mutex<usize>>,
        cancel_calls: Arc<Mutex<usize>>,
        cancel_all_calls: Arc<Mutex<usize>>,
        last_client_order_id: Arc<Mutex<Option<String>>>,
    }

    impl MockRestClient {
        fn new(handle: MockExchangeHandle) -> Self {
            Self {
                handle,
                failures: Arc::new(Mutex::new(VecDeque::new())),
                place_calls: Arc::new(Mutex::new(0)),
                cancel_calls: Arc::new(Mutex::new(0)),
                cancel_all_calls: Arc::new(Mutex::new(0)),
                last_client_order_id: Arc::new(Mutex::new(None)),
            }
        }

        fn push_failure(&self, err: LiveGatewayError) {
            self.failures.lock().unwrap().push_back(err);
        }

        fn take_failure(&self) -> Option<LiveGatewayError> {
            self.failures.lock().unwrap().pop_front()
        }

        fn place_call_count(&self) -> usize {
            *self.place_calls.lock().unwrap()
        }

        fn cancel_call_count(&self) -> usize {
            *self.cancel_calls.lock().unwrap()
        }

        fn cancel_all_call_count(&self) -> usize {
            *self.cancel_all_calls.lock().unwrap()
        }

        fn last_client_order_id(&self) -> Option<String> {
            self.last_client_order_id.lock().unwrap().clone()
        }
    }

    impl LiveRestClient for MockRestClient {
        fn place_order(
            &self,
            req: paraphina::live::gateway::LiveRestPlaceRequest,
        ) -> paraphina::live::gateway::BoxFuture<
            '_,
            paraphina::live::gateway::LiveResult<LiveRestResponse>,
        > {
            let handle = self.handle.clone();
            let failures = self.take_failure();
            let calls = self.place_calls.clone();
            let last_client_order_id = self.last_client_order_id.clone();
            Box::pin(async move {
                *calls.lock().unwrap() += 1;
                if let Some(err) = failures {
                    return Err(err);
                }
                *last_client_order_id.lock().unwrap() = Some(req.client_order_id.clone());
                let (response_tx, mut response_rx) = tokio::sync::oneshot::channel();
                let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    side: req.side,
                    price: req.price,
                    size: req.size,
                    purpose: req.purpose,
                    time_in_force: req.time_in_force,
                    post_only: req.post_only,
                    reduce_only: req.reduce_only,
                    client_order_id: Some(req.client_order_id.clone()),
                });
                let request = paraphina::live::runner::LiveOrderRequest {
                    intents: vec![intent],
                    action_batch: ActionBatch::new(0, 0, "test"),
                    now_ms: 1_000,
                    response: response_tx,
                };
                let _ = handle.order_tx.send(request).await;
                let mut order_id = None;
                for _ in 0..1_000 {
                    match response_rx.try_recv() {
                        Ok(events) => {
                            for event in events {
                                if let paraphina::live::types::ExecutionEvent::OrderAccepted(ack) =
                                    event
                                {
                                    order_id = Some(ack.order_id);
                                }
                            }
                            break;
                        }
                        Err(tokio::sync::oneshot::error::TryRecvError::Closed) => break,
                        Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                            tokio::task::yield_now().await
                        }
                    }
                }
                Ok(LiveRestResponse { order_id })
            })
        }

        fn cancel_order(
            &self,
            req: paraphina::live::gateway::LiveRestCancelRequest,
        ) -> paraphina::live::gateway::BoxFuture<
            '_,
            paraphina::live::gateway::LiveResult<LiveRestResponse>,
        > {
            let handle = self.handle.clone();
            let failures = self.take_failure();
            let calls = self.cancel_calls.clone();
            Box::pin(async move {
                *calls.lock().unwrap() += 1;
                if let Some(err) = failures {
                    return Err(err);
                }
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                let intent = OrderIntent::Cancel(paraphina::types::CancelOrderIntent {
                    venue_index: req.venue_index,
                    venue_id: req.venue_id.as_str().into(),
                    order_id: req.order_id.clone(),
                });
                let request = paraphina::live::runner::LiveOrderRequest {
                    intents: vec![intent],
                    action_batch: ActionBatch::new(0, 0, "test"),
                    now_ms: 1_000,
                    response: response_tx,
                };
                let _ = handle.order_tx.send(request).await;
                let _ = response_rx.await;
                Ok(LiveRestResponse { order_id: None })
            })
        }

        fn cancel_all(
            &self,
            req: paraphina::live::gateway::LiveRestCancelAllRequest,
        ) -> paraphina::live::gateway::BoxFuture<
            '_,
            paraphina::live::gateway::LiveResult<LiveRestResponse>,
        > {
            let handle = self.handle.clone();
            let failures = self.take_failure();
            let calls = self.cancel_all_calls.clone();
            Box::pin(async move {
                *calls.lock().unwrap() += 1;
                if let Some(err) = failures {
                    return Err(err);
                }
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                let intent = OrderIntent::CancelAll(paraphina::types::CancelAllOrderIntent {
                    venue_index: Some(req.venue_index),
                    venue_id: Some(req.venue_id.as_str().into()),
                });
                let request = paraphina::live::runner::LiveOrderRequest {
                    intents: vec![intent],
                    action_batch: ActionBatch::new(0, 0, "test"),
                    now_ms: 1_000,
                    response: response_tx,
                };
                let _ = handle.order_tx.send(request).await;
                let _ = response_rx.await;
                Ok(LiveRestResponse { order_id: None })
            })
        }
    }

    fn metrics_value(metrics: &LiveMetrics, name: &str) -> i64 {
        let text = metrics.gather();
        for line in text.lines() {
            if line.starts_with(name) {
                if let Some(value) = line.split_whitespace().last() {
                    return value.parse::<f64>().unwrap_or(0.0) as i64;
                }
            }
        }
        0
    }

    #[tokio::test]
    async fn retries_retryable_failures() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle);
        client.push_failure(LiveGatewayError::retryable("transient"));
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            Some(metrics.clone()),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
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
        let res = gateway.submit_intent(&intent, 1, 1_000).await;
        assert!(res.is_ok());
        assert_eq!(client.place_call_count(), 2);
        assert_eq!(metrics_value(&metrics, "paraphina_live_retry_count"), 1);
        assert_eq!(metrics_value(&metrics, "paraphina_live_order_submit_ok"), 1);
    }

    #[tokio::test]
    async fn post_only_reject_is_fatal() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle);
        client.push_failure(LiveGatewayError::post_only_reject("post-only"));
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            Some(metrics.clone()),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 101.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: None,
        });
        let res = gateway.submit_intent(&intent, 2, 1_000).await;
        assert!(res.is_err());
        assert_eq!(client.place_call_count(), 1);
        assert_eq!(
            metrics_value(&metrics, "paraphina_live_order_submit_fail"),
            1
        );
    }

    #[tokio::test]
    async fn reduce_only_reject_is_fatal() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle);
        client.push_failure(LiveGatewayError::reduce_only_violation("reduce-only"));
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            Some(metrics.clone()),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Sell,
            price: 99.0,
            size: 1.0,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
        });
        let res = gateway.submit_intent(&intent, 3, 1_000).await;
        assert!(res.is_err());
        assert_eq!(client.place_call_count(), 1);
        assert_eq!(
            metrics_value(&metrics, "paraphina_live_order_submit_fail"),
            1
        );
    }

    #[tokio::test]
    async fn rate_limit_sleeps_are_counted() {
        let mut policy = GatewayPolicy::for_live();
        policy.rate_limit.enabled = true;
        policy.rate_limit.max_requests_per_second = 1.0;
        policy.rate_limit.burst_capacity = 1;
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle);
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            policy,
            Some(metrics.clone()),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
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
        let _ = gateway.submit_intent(&intent, 4, 1_000).await;
        let _ = gateway.submit_intent(&intent, 4, 1_000).await;
        assert!(metrics_value(&metrics, "paraphina_live_rate_limit_sleep_total_ms") > 0);
    }

    #[tokio::test]
    async fn cancel_paths_increment_metrics() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle.clone());
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            Some(metrics.clone()),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
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
        let placed = gateway.submit_intent(&intent, 5, 1_000).await.unwrap();
        let order_id = placed.order_id.expect("order id");

        let cancel_intent = OrderIntent::Cancel(paraphina::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            order_id,
        });
        let _ = gateway.submit_intent(&cancel_intent, 5, 1_000).await;

        let cancel_all_intent = OrderIntent::CancelAll(paraphina::types::CancelAllOrderIntent {
            venue_index: Some(0),
            venue_id: Some(cfg.venues[0].id_arc.clone()),
        });
        let _ = gateway.submit_intent(&cancel_all_intent, 5, 1_000).await;

        assert_eq!(metrics_value(&metrics, "paraphina_live_cancel_ok"), 2);
    }

    #[tokio::test]
    async fn cancel_all_removes_open_orders() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle.clone());
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            None,
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("co_cancel_all_1".to_string()),
        });
        let _ = gateway.submit_intent(&intent, 10, 1_000).await.unwrap();
        assert_eq!(handle.open_orders_len().await, 1);

        let cancel_all_intent = OrderIntent::CancelAll(paraphina::types::CancelAllOrderIntent {
            venue_index: Some(0),
            venue_id: Some(cfg.venues[0].id_arc.clone()),
        });
        let _ = gateway.submit_intent(&cancel_all_intent, 10, 1_000).await;
        assert_eq!(handle.open_orders_len().await, 0);
    }

    #[tokio::test]
    async fn idempotent_replay_keeps_single_open_order() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle.clone());
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            None,
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: cfg.venues[0].id_arc.clone(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            client_order_id: Some("co_replay_1".to_string()),
        });
        let _ = gateway.submit_intent(&intent, 11, 1_000).await.unwrap();
        let _ = gateway.submit_intent(&intent, 11, 1_000).await.unwrap();
        assert_eq!(handle.open_orders_len().await, 1);
    }

    #[tokio::test]
    async fn deterministic_client_order_id_generation() {
        let cfg = Config::default();
        let handle = spawn_mock_exchange(&cfg, MockExchangeConfig::default()).await;
        let client = MockRestClient::new(handle);
        let metrics = LiveMetrics::new();
        let mut gateway = LiveGateway::new(
            &cfg,
            client.clone(),
            GatewayPolicy::for_live(),
            Some(metrics),
            TradeMode::Live,
        )
        .expect("live gateway")
        .with_sleep_fn(Arc::new(|_d| Box::pin(async {})));

        let intent = OrderIntent::Place(paraphina::types::PlaceOrderIntent {
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
        let _ = gateway.submit_intent(&intent, 42, 1_000).await;
        let id = client.last_client_order_id().expect("client id");
        assert!(id.contains(cfg.venues[0].id.as_str()));
        assert!(id.contains("42"));
        assert!(id.contains("mm"));
    }

    #[tokio::test]
    async fn shadow_mode_blocks_rest_gateway() {
        #[derive(Clone)]
        struct PanicClient;

        impl LiveRestClient for PanicClient {
            fn place_order(
                &self,
                _req: paraphina::live::gateway::LiveRestPlaceRequest,
            ) -> paraphina::live::gateway::BoxFuture<
                '_,
                paraphina::live::gateway::LiveResult<LiveRestResponse>,
            > {
                panic!("rest should not be called in shadow mode")
            }

            fn cancel_order(
                &self,
                _req: paraphina::live::gateway::LiveRestCancelRequest,
            ) -> paraphina::live::gateway::BoxFuture<
                '_,
                paraphina::live::gateway::LiveResult<LiveRestResponse>,
            > {
                panic!("rest should not be called in shadow mode")
            }

            fn cancel_all(
                &self,
                _req: paraphina::live::gateway::LiveRestCancelAllRequest,
            ) -> paraphina::live::gateway::BoxFuture<
                '_,
                paraphina::live::gateway::LiveResult<LiveRestResponse>,
            > {
                panic!("rest should not be called in shadow mode")
            }
        }

        let cfg = Config::default();
        let client = PanicClient;
        let result = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_live(),
            None,
            TradeMode::Shadow,
        );
        assert!(result.is_err());
    }
}
