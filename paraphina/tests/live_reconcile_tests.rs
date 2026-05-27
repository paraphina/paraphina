#[cfg(feature = "live")]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use paraphina::config::Config;
    use paraphina::live::runner::{
        run_live_loop, LiveAccountRequest, LiveChannels, LiveRunMode, LiveTelemetry,
        LiveTelemetryStats,
    };
    use paraphina::live::types::{
        AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, Fill, L2Snapshot,
        LiquidationSnapshot, MarginSnapshot, MarketDataEvent, PositionSnapshot,
    };
    use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink};
    use paraphina::types::{OrderPurpose, Side, TimestampMs};
    use tempfile::tempdir;
    use tokio::sync::mpsc;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn build_account_snapshot(
        venue_id: &str,
        venue_index: usize,
        position_tao: f64,
        balance_usd: f64,
        available_usd: f64,
        now_ms: TimestampMs,
        seq: u64,
    ) -> AccountSnapshot {
        AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq,
            timestamp_ms: now_ms,
            open_order_count: None,
            positions: vec![PositionSnapshot {
                symbol: "TAO".to_string(),
                size: position_tao,
                entry_price: 100.0,
            }],
            balances: vec![BalanceSnapshot {
                asset: "USD".to_string(),
                total: balance_usd,
                available: available_usd,
            }],
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd,
                used_usd: 0.0,
                available_usd,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        }
    }

    fn build_unavailable_account_snapshot(venue_id: &str, venue_index: usize) -> AccountSnapshot {
        AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: 0,
            timestamp_ms: 0,
            open_order_count: None,
            positions: Vec::new(),
            balances: Vec::new(),
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd: 0.0,
                used_usd: 0.0,
                available_usd: 0.0,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        }
    }

    #[tokio::test]
    async fn first_fresh_snapshot_hydrates_state_without_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, mut order_rx) =
            mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let start_ms = 1_000;
        let step_ms = 100;
        let ticks = 2_u64;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = L2Snapshot {
            venue_index: 0,
            venue_id,
            seq: 1,
            timestamp_ms: start_ms,
            bids: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        let _ = market_tx.send(MarketDataEvent::L2Snapshot(snapshot)).await;

        let venue_id = cfg.venues[0].id.clone();
        let snapshot = build_account_snapshot(&venue_id, 0, 0.0, 0.0, 0.0, start_ms, 1);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "first fresh account snapshot should hydrate live state before drift enforcement"
        );

        let mut cancel_all_count = 0;
        while let Ok(req) = order_rx.try_recv() {
            for intent in req.intents {
                if matches!(intent, paraphina::types::OrderIntent::CancelAll(_)) {
                    cancel_all_count += 1;
                }
            }
        }
        assert_eq!(
            cancel_all_count, 0,
            "hydration should not trigger cancel-all"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
    }

    #[tokio::test]
    async fn reconcile_match_does_not_trigger() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let start_ms = 1_000;
        let step_ms = 100;
        let ticks = 2_u64;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = L2Snapshot {
            venue_index: 0,
            venue_id,
            seq: 1,
            timestamp_ms: start_ms,
            bids: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        let _ = market_tx.send(MarketDataEvent::L2Snapshot(snapshot)).await;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = build_account_snapshot(&venue_id, 0, 0.0, 0.0, 10_000.0, start_ms, 1);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            None,
        )
        .await;
        assert!(!summary.kill_switch, "did not expect kill switch");

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
    }

    #[tokio::test]
    async fn first_fresh_snapshot_does_not_emit_drift_telemetry() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let temp = tempdir().expect("tempdir");
        let telemetry_path = temp.path().join("telemetry.jsonl");
        let telemetry = LiveTelemetry {
            sink: paraphina::telemetry::TelemetrySinkHandle::Sync(std::sync::Arc::new(
                std::sync::Mutex::new(TelemetrySink::from_config(TelemetryConfig {
                    mode: TelemetryMode::Jsonl,
                    path: Some(telemetry_path.clone()),
                    append: false,
                })),
            )),
            shadow_mode: false,
            execution_mode: "live",
            max_orders_per_tick: 200,
            stats: std::sync::Arc::new(LiveTelemetryStats::default()),
        };

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let start_ms = 1_000;
        let step_ms = 100;
        let ticks = 2_u64;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = L2Snapshot {
            venue_index: 0,
            venue_id,
            seq: 1,
            timestamp_ms: start_ms,
            bids: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        let _ = market_tx.send(MarketDataEvent::L2Snapshot(snapshot)).await;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = build_account_snapshot(&venue_id, 0, 2.0, 9_000.0, 0.0, start_ms, 1);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let hooks = paraphina::live::runner::LiveRuntimeHooks {
            metrics: paraphina::live::ops::LiveMetrics::new(),
            health: paraphina::live::ops::HealthState::new(),
            telemetry: Some(telemetry),
        };

        let _ = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            Some(hooks),
        )
        .await;

        let lines = std::fs::read_to_string(&telemetry_path).expect("telemetry");
        let mut drift_lines = Vec::new();
        for line in lines.lines() {
            if line.contains("reconcile_drift") {
                drift_lines.push(line.to_string());
            }
        }
        assert!(
            drift_lines.is_empty(),
            "first fresh snapshot should not emit reconcile drift telemetry"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
    }

    #[tokio::test]
    async fn margin_snapshot_updates_do_not_trigger_reconcile_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];
        cfg.main_loop_interval_ms = 20;

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let venue_id = cfg.venues[0].id.clone();
        let first_now_ms = paraphina::types::now_ms();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: first_now_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id,
                0,
                0.0,
                100.0,
                90.0,
                first_now_ms,
                1,
            )))
            .await;

        let delayed_account_tx = account_tx.clone();
        let delayed_venue_id = venue_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.0,
                    100.0,
                    87.5,
                    now_ms,
                    2,
                )))
                .await;
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Realtime {
                interval_ms: cfg.main_loop_interval_ms as u64,
                max_ticks: Some(4),
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "fresh account margin updates should not trigger reconcile drift kill"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
    }

    #[tokio::test]
    async fn single_small_position_drift_is_corrected_without_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK", "2");
        std::env::set_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT", "5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];
        cfg.main_loop_interval_ms = 20;

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let venue_id = cfg.venues[0].id.clone();
        let first_now_ms = paraphina::types::now_ms();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: first_now_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id,
                0,
                0.0,
                100.0,
                90.0,
                first_now_ms,
                1,
            )))
            .await;

        let delayed_account_tx = account_tx.clone();
        let delayed_venue_id = venue_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.02,
                    100.0,
                    89.0,
                    now_ms,
                    2,
                )))
                .await;
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Realtime {
                interval_ms: cfg.main_loop_interval_ms as u64,
                max_ticks: Some(5),
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "single small drift should be corrected by the account snapshot without killing"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT");
    }

    #[tokio::test]
    async fn fresh_large_position_change_is_corrected_without_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK", "2");
        std::env::set_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT", "5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];
        cfg.main_loop_interval_ms = 20;

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (_exec_tx, exec_rx) = mpsc::channel::<ExecutionEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let venue_id = cfg.venues[0].id.clone();
        let first_now_ms = paraphina::types::now_ms();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: first_now_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id,
                0,
                0.0,
                100.0,
                90.0,
                first_now_ms,
                1,
            )))
            .await;

        let delayed_account_tx = account_tx.clone();
        let delayed_venue_id = venue_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.06,
                    100.0,
                    87.0,
                    now_ms,
                    2,
                )))
                .await;
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Realtime {
                interval_ms: cfg.main_loop_interval_ms as u64,
                max_ticks: Some(5),
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "a fresh venue-side position change should synchronize before reconcile kills"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT");
    }

    #[tokio::test]
    async fn stable_large_position_mismatch_after_apply_still_triggers_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK", "2");
        std::env::set_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT", "5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];
        cfg.main_loop_interval_ms = 20;

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (exec_tx, exec_rx) = mpsc::channel::<ExecutionEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let venue_id = cfg.venues[0].id.clone();
        let first_now_ms = paraphina::types::now_ms();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: first_now_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id,
                0,
                0.06,
                100.0,
                87.0,
                first_now_ms,
                1,
            )))
            .await;

        let delayed_exec_tx = exec_tx.clone();
        let delayed_account_tx = account_tx.clone();
        let delayed_venue_id = venue_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_exec_tx
                .send(ExecutionEvent::Filled(Fill {
                    venue_index: 0,
                    venue_id: delayed_venue_id.clone(),
                    seq: 2,
                    timestamp_ms: now_ms,
                    order_id: Some("fill_order".to_string()),
                    client_order_id: Some("fill_client".to_string()),
                    fill_id: Some("fill_1".to_string()),
                    phase51_target_key: None,
                    phase51_native_role: None,
                    phase51_lighter_native_limit: None,
                    side: Side::Sell,
                    price: 100.0,
                    size: 0.06,
                    purpose: OrderPurpose::Mm,
                    fee_bps: 0.0,
                }))
                .await;
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.06,
                    100.0,
                    87.0,
                    now_ms,
                    3,
                )))
                .await;
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Realtime {
                interval_ms: cfg.main_loop_interval_ms as u64,
                max_ticks: Some(6),
            },
            None,
        )
        .await;
        assert!(
            summary.kill_switch,
            "a stable venue position mismatch after the last applied snapshot should still kill"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT");
    }

    #[tokio::test]
    async fn exempt_position_drift_venue_does_not_kill() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK", "2");
        std::env::set_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT", "5");
        std::env::set_var("PARAPHINA_RECONCILE_POS_KILL_EXEMPT_VENUES", "lighter");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];
        cfg.venues[0].id = "lighter".to_string();
        cfg.main_loop_interval_ms = 20;

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let venue_id = cfg.venues[0].id.clone();
        let first_now_ms = paraphina::types::now_ms();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: first_now_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id,
                0,
                0.0,
                100.0,
                90.0,
                first_now_ms,
                1,
            )))
            .await;

        let delayed_account_tx = account_tx.clone();
        let delayed_venue_id = venue_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.06,
                    100.0,
                    87.0,
                    now_ms,
                    2,
                )))
                .await;
            tokio::time::sleep(Duration::from_millis(25)).await;
            let now_ms = paraphina::types::now_ms();
            let _ = delayed_account_tx
                .send(AccountEvent::Snapshot(build_account_snapshot(
                    &delayed_venue_id,
                    0,
                    0.06,
                    100.0,
                    87.0,
                    now_ms,
                    3,
                )))
                .await;
        });

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Realtime {
                interval_ms: cfg.main_loop_interval_ms as u64,
                max_ticks: Some(6),
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "exempt venues should audit repeated position drift without killing"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_SOFT_KILL_STREAK");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_HARD_KILL_MULT");
        std::env::remove_var("PARAPHINA_RECONCILE_POS_KILL_EXEMPT_VENUES");
    }

    #[tokio::test]
    async fn reconcile_unavailable_snapshot_does_not_kill_or_drift() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let temp = tempdir().expect("tempdir");
        let telemetry_path = temp.path().join("telemetry.jsonl");
        let telemetry = LiveTelemetry {
            sink: paraphina::telemetry::TelemetrySinkHandle::Sync(std::sync::Arc::new(
                std::sync::Mutex::new(TelemetrySink::from_config(TelemetryConfig {
                    mode: TelemetryMode::Jsonl,
                    path: Some(telemetry_path.clone()),
                    append: false,
                })),
            )),
            shadow_mode: true,
            execution_mode: "shadow",
            max_orders_per_tick: 200,
            stats: std::sync::Arc::new(LiveTelemetryStats::default()),
        };

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let start_ms = 1_000;
        let step_ms = 100;
        let ticks = 1_u64;
        let venue_id = cfg.venues[0].id.clone();
        let snapshot = L2Snapshot {
            venue_index: 0,
            venue_id: venue_id.clone(),
            seq: 1,
            timestamp_ms: start_ms,
            bids: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![paraphina::live::orderbook_l2::BookLevel {
                price: 101.0,
                size: 1.0,
            }],
        };
        let _ = market_tx.send(MarketDataEvent::L2Snapshot(snapshot)).await;
        let snapshot = build_unavailable_account_snapshot(&venue_id, 0);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let hooks = paraphina::live::runner::LiveRuntimeHooks {
            metrics: paraphina::live::ops::LiveMetrics::new(),
            health: paraphina::live::ops::HealthState::new(),
            telemetry: Some(telemetry),
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            Some(hooks),
        )
        .await;
        assert!(
            !summary.kill_switch,
            "did not expect kill switch for unavailable snapshot"
        );

        let lines = std::fs::read_to_string(&telemetry_path).expect("telemetry");
        let drift_lines: Vec<_> = lines
            .lines()
            .filter(|line| line.contains("reconcile_drift"))
            .collect();
        assert!(
            drift_lines.is_empty(),
            "did not expect reconcile drift in shadow/unavailable snapshot"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
    }

    #[tokio::test]
    async fn fresh_account_snapshots_do_not_emit_account_unavailable_drift() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS", "1");
        std::env::set_var("PARAPHINA_LIVE_ACCOUNT_POLL_MS", "5000");

        let temp = tempdir().expect("tempdir");
        std::env::set_var("PARAPHINA_LIVE_AUDIT_DIR", temp.path());

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);

        let start_ms = 1_000;
        let step_ms = 1_000;
        let ticks = 3_u64;
        let venue_id = cfg.venues[0].id.clone();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: start_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id, 0, 0.0, 100.0, 90.0, start_ms, 1,
            )))
            .await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "fresh account snapshots should not trigger kill"
        );

        let drift_path = temp.path().join("reconcile_drift.jsonl");
        let drift_lines = std::fs::read_to_string(&drift_path).unwrap_or_default();
        assert!(
            !drift_lines.contains("\"kind\":\"account_unavailable\""),
            "fresh account snapshots should not emit account_unavailable drift"
        );

        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS");
        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_POLL_MS");
        std::env::remove_var("PARAPHINA_LIVE_AUDIT_DIR");
    }

    #[tokio::test]
    async fn account_reconcile_request_tx_is_not_used_without_explicit_opt_in() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS", "1");
        std::env::set_var("PARAPHINA_LIVE_ACCOUNT_POLL_MS", "5000");
        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_REQUEST_TX");

        let temp = tempdir().expect("tempdir");
        std::env::set_var("PARAPHINA_LIVE_AUDIT_DIR", temp.path());

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let (market_tx, market_rx) = mpsc::channel::<MarketDataEvent>(32);
        let (account_tx, account_rx) = mpsc::channel::<AccountEvent>(32);
        let (order_tx, _order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(32);
        let (account_request_tx, mut account_request_rx) = mpsc::channel::<LiveAccountRequest>(32);

        let start_ms = 1_000;
        let step_ms = 1_000;
        let ticks = 3_u64;
        let venue_id = cfg.venues[0].id.clone();
        let _ = market_tx
            .send(MarketDataEvent::L2Snapshot(L2Snapshot {
                venue_index: 0,
                venue_id: venue_id.clone(),
                seq: 1,
                timestamp_ms: start_ms,
                bids: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 100.0,
                    size: 1.0,
                }],
                asks: vec![paraphina::live::orderbook_l2::BookLevel {
                    price: 101.0,
                    size: 1.0,
                }],
            }))
            .await;
        let _ = account_tx
            .send(AccountEvent::Snapshot(build_account_snapshot(
                &venue_id, 0, 0.0, 100.0, 90.0, start_ms, 1,
            )))
            .await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: Some(account_request_tx),
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: None,
            shared_venue_ages: None,
        };

        let summary = run_live_loop(
            &cfg,
            channels,
            LiveRunMode::Step {
                start_ms,
                step_ms,
                ticks,
            },
            None,
        )
        .await;
        assert!(
            !summary.kill_switch,
            "request channel should not change reconcile health without opt-in"
        );
        assert!(
            account_request_rx.try_recv().is_err(),
            "account request channel must remain unused unless PARAPHINA_LIVE_ACCOUNT_RECONCILE_REQUEST_TX is enabled"
        );

        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS");
        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_POLL_MS");
        std::env::remove_var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_REQUEST_TX");
        std::env::remove_var("PARAPHINA_LIVE_AUDIT_DIR");
    }
}
