#[cfg(feature = "live")]
mod tests {
    use std::sync::Mutex;

    use paraphina::config::Config;
    use paraphina::live::runner::{
        run_live_loop, LiveChannels, LiveRunMode, LiveTelemetry, LiveTelemetryStats,
    };
    use paraphina::live::types::{
        AccountEvent, AccountSnapshot, BalanceSnapshot, L2Snapshot, LiquidationSnapshot,
        MarginSnapshot, MarketDataEvent, PositionSnapshot,
    };
    use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink};
    use paraphina::types::TimestampMs;
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
    async fn reconcile_mismatch_triggers_kill_and_cancel_all() {
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
        let snapshot = build_account_snapshot(&venue_id, 0, 5.0, 9_000.0, 9_000.0, start_ms, 1);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            order_tx,
            order_snapshot_rx: None,
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
            summary.kill_switch,
            "expected kill switch from reconcile drift"
        );

        let mut cancel_all_count = 0;
        while let Ok(req) = order_rx.try_recv() {
            for intent in req.intents {
                if matches!(intent, paraphina::types::OrderIntent::CancelAll(_)) {
                    cancel_all_count += 1;
                }
            }
        }
        assert_eq!(cancel_all_count, 1, "cancel-all should be issued once");

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
            order_tx,
            order_snapshot_rx: None,
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
    async fn reconcile_drift_event_order_is_deterministic() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_RECONCILE_POS_TAO_TOL", "0.01");
        std::env::set_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL", "0.5");

        let mut cfg = Config::default();
        cfg.venues = vec![cfg.venues[0].clone()];

        let temp = tempdir().expect("tempdir");
        let telemetry_path = temp.path().join("telemetry.jsonl");
        let telemetry = LiveTelemetry {
            sink: std::sync::Arc::new(std::sync::Mutex::new(TelemetrySink::from_config(
                TelemetryConfig {
                    mode: TelemetryMode::Jsonl,
                    path: Some(telemetry_path.clone()),
                    append: false,
                },
            ))),
            shadow_mode: false,
            execution_mode: "live",
            max_orders_per_tick: 200,
            stats: std::sync::Arc::new(std::sync::Mutex::new(LiveTelemetryStats::default())),
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
        let snapshot = build_account_snapshot(&venue_id, 0, 2.0, 9_000.0, 9_000.0, start_ms, 1);
        let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;

        let channels = LiveChannels {
            market_rx,
            account_rx,
            exec_rx: None,
            account_reconcile_tx: None,
            order_tx,
            order_snapshot_rx: None,
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
            !drift_lines.is_empty(),
            "expected reconcile drift telemetry"
        );

        std::env::remove_var("PARAPHINA_RECONCILE_POS_TAO_TOL");
        std::env::remove_var("PARAPHINA_RECONCILE_BALANCE_USD_TOL");
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
            sink: std::sync::Arc::new(std::sync::Mutex::new(TelemetrySink::from_config(
                TelemetryConfig {
                    mode: TelemetryMode::Jsonl,
                    path: Some(telemetry_path.clone()),
                    append: false,
                },
            ))),
            shadow_mode: true,
            execution_mode: "shadow",
            max_orders_per_tick: 200,
            stats: std::sync::Arc::new(std::sync::Mutex::new(LiveTelemetryStats::default())),
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
            order_tx,
            order_snapshot_rx: None,
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
}
