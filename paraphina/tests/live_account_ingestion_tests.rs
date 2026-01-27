#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::engine::Engine;
    use paraphina::live::runner::apply_account_snapshot_to_state;
    use paraphina::live::state_cache::{
        CanonicalCacheSnapshot, LiveStateCache, VenueAccountSnapshot, VenueMarketSnapshot,
    };
    use paraphina::state::GlobalState;

    fn build_market_snapshots(cfg: &Config, mid: f64) -> Vec<VenueMarketSnapshot> {
        cfg.venues
            .iter()
            .enumerate()
            .map(|(idx, v)| VenueMarketSnapshot {
                venue_index: idx,
                venue_id: v.id_arc.clone(),
                seq: 1,
                timestamp_ms: Some(1_000),
                mid: Some(mid),
                spread: Some(1.0),
                depth_near_mid: 10_000.0,
                is_stale: false,
            })
            .collect()
    }

    fn build_account_snapshot(cfg: &Config, price_liq: f64, seq: u64) -> Vec<VenueAccountSnapshot> {
        cfg.venues
            .iter()
            .enumerate()
            .map(|(idx, v)| VenueAccountSnapshot {
                venue_index: idx,
                venue_id: v.id_arc.clone(),
                seq,
                timestamp_ms: Some(1_000),
                position_tao: if idx == 0 { 2.0 } else { 0.0 },
                avg_entry_price: 100.0,
                funding_8h: Some(0.001),
                margin_balance_usd: 10_000.0,
                margin_used_usd: 500.0,
                margin_available_usd: 9_500.0,
                price_liq: if idx == 0 { Some(price_liq) } else { None },
                dist_liq_sigma: None,
                is_stale: false,
            })
            .collect()
    }

    #[test]
    fn dist_liq_sigma_updates_from_account_snapshot() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.sigma_eff = 1.0;
        state.venues[0].mid = Some(100.0);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: 1_000,
            market: build_market_snapshots(&cfg, 100.0),
            account: build_account_snapshot(&cfg, 95.0, 10),
        };
        apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, 1_000);
        let dist = state.venues[0].dist_liq_sigma;
        assert!((dist - 0.05).abs() < 1e-6, "dist_liq_sigma={dist}");
        assert_eq!(state.venues[0].position_tao, 2.0);
        assert_eq!(state.venues[0].price_liq, Some(95.0));
        assert!((state.venues[0].margin_balance_usd - 10_000.0).abs() < 1e-6);
        assert_eq!(state.venues[0].funding_8h, 0.001);
    }

    #[test]
    fn kill_switch_triggers_on_liq_distance() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.sigma_eff = 1.0;
        state.venues[0].mid = Some(100.0);

        let snapshot = CanonicalCacheSnapshot {
            timestamp_ms: 1_000,
            market: build_market_snapshots(&cfg, 100.0),
            account: build_account_snapshot(&cfg, 99.5, 11),
        };
        apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, 1_000);

        let engine = Engine::new(&cfg);
        engine.update_risk_limits_and_regime(&mut state);
        assert!(state.kill_switch);
    }

    #[test]
    fn reconciliation_diff_is_idempotent() {
        let cfg = Config::default();
        let mut cache = LiveStateCache::new(&cfg);
        let mut snapshot = paraphina::live::types::AccountSnapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.clone(),
            seq: 1,
            timestamp_ms: 1_000,
            positions: Vec::new(),
            balances: Vec::new(),
            funding_8h: Some(0.001),
            margin: paraphina::live::types::MarginSnapshot {
                balance_usd: 10_000.0,
                used_usd: 0.0,
                available_usd: 10_000.0,
            },
            liquidation: paraphina::live::types::LiquidationSnapshot {
                price_liq: Some(80.0),
                dist_liq_sigma: None,
            },
        };
        cache
            .apply_account_event(&paraphina::live::types::AccountEvent::Snapshot(
                snapshot.clone(),
            ))
            .expect("apply snapshot");

        snapshot.seq = 2;
        snapshot.margin.balance_usd = 9_000.0;
        let (_report, diff) = cache.reconcile_account_snapshot_with_diff(&snapshot);
        assert!(diff.is_some(), "expected diff on drift");

        snapshot.seq = 3;
        let (_report2, diff2) = cache.reconcile_account_snapshot_with_diff(&snapshot);
        assert!(diff2.is_none(), "expected idempotent reconcile");
    }

    #[test]
    fn account_snapshot_staleness_tracks_unavailable() {
        let cfg = Config::default();
        let mut cache = LiveStateCache::new(&cfg);
        let snapshot = paraphina::live::types::AccountSnapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.clone(),
            seq: 0,
            timestamp_ms: 0,
            positions: Vec::new(),
            balances: Vec::new(),
            funding_8h: None,
            margin: paraphina::live::types::MarginSnapshot {
                balance_usd: 0.0,
                used_usd: 0.0,
                available_usd: 0.0,
            },
            liquidation: paraphina::live::types::LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        };
        cache
            .apply_account_event(&paraphina::live::types::AccountEvent::Snapshot(snapshot))
            .expect("apply snapshot");
        let snapshot = cache.snapshot(2_000, 1_000);
        assert!(
            snapshot.account[0].is_stale,
            "account should be stale when unavailable"
        );
    }
}
