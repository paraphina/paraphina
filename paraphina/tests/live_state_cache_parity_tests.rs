#[cfg(feature = "live")]
mod tests {
    use paraphina::config::Config;
    use paraphina::live::runner::apply_account_snapshot_to_state;
    use paraphina::live::state_cache::LiveStateCache;
    use paraphina::live::types::{
        AccountEvent, AccountSnapshot, BalanceSnapshot, L2Snapshot, LiquidationSnapshot,
        MarginSnapshot, MarketDataEvent, PositionSnapshot,
    };
    use paraphina::orderbook_l2::{BookLevel, DepthConfig};
    use paraphina::state::GlobalState;

    fn assert_close(a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "expected {a} ~= {b} (diff {diff})");
    }

    #[test]
    fn cache_and_core_l2_parity_with_account_snapshot() {
        let mut cfg = Config::default();
        cfg.book.depth_levels = 3;
        let max_levels = cfg.book.depth_levels.max(1) as usize;

        let mut cache = LiveStateCache::new(&cfg);
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(100.0);
        state.fair_value_prev = 100.0;
        state.sigma_eff = 1.0;

        let now_ms = 1_000;
        let l2_snapshot = L2Snapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.to_string(),
            seq: 1,
            timestamp_ms: now_ms,
            bids: vec![
                BookLevel {
                    price: 100.0,
                    size: 1.0,
                },
                BookLevel {
                    price: 99.5,
                    size: 1.2,
                },
                BookLevel {
                    price: 99.0,
                    size: 1.4,
                },
                BookLevel {
                    price: 98.5,
                    size: 1.6,
                },
            ],
            asks: vec![
                BookLevel {
                    price: 101.0,
                    size: 2.0,
                },
                BookLevel {
                    price: 101.5,
                    size: 2.2,
                },
                BookLevel {
                    price: 102.0,
                    size: 2.4,
                },
                BookLevel {
                    price: 102.5,
                    size: 2.6,
                },
            ],
        };
        let market_event = MarketDataEvent::L2Snapshot(l2_snapshot.clone());
        cache
            .apply_market_event(&market_event)
            .expect("cache apply");

        let alpha_short = cfg.volatility.fv_vol_alpha_short;
        let alpha_long = cfg.volatility.fv_vol_alpha_long;
        let venue = state.venues.get_mut(0).expect("venue");
        venue
            .apply_l2_snapshot(
                &l2_snapshot.bids,
                &l2_snapshot.asks,
                l2_snapshot.seq,
                l2_snapshot.timestamp_ms,
                max_levels,
                alpha_short,
                alpha_long,
            )
            .expect("core apply");

        let cache_book = &cache.market[0].orderbook;
        assert_eq!(cache_book.bids().len(), l2_snapshot.bids.len());
        assert_eq!(cache_book.asks().len(), l2_snapshot.asks.len());
        assert!(venue.orderbook_l2.bids().len() <= max_levels);
        assert!(venue.orderbook_l2.asks().len() <= max_levels);

        let expected = cache_book.compute_mid_spread_depth(DepthConfig {
            levels: max_levels,
            include_imbalance: false,
        });
        assert_close(venue.mid.unwrap(), expected.mid.unwrap(), 1e-9);
        assert_close(venue.spread.unwrap(), expected.spread.unwrap(), 1e-9);
        let bid = cache_book.bids().first().expect("best bid");
        let ask = cache_book.asks().first().expect("best ask");
        let expected_depth = (bid.price * bid.size).abs() + (ask.price * ask.size).abs();
        assert_close(venue.depth_near_mid, expected_depth, 1e-9);

        let account_snapshot = AccountSnapshot {
            venue_index: 0,
            venue_id: cfg.venues[0].id.to_string(),
            seq: 1,
            timestamp_ms: now_ms,
            positions: vec![PositionSnapshot {
                symbol: "TAO-PERP".to_string(),
                size: 2.0,
                entry_price: 100.0,
            }],
            balances: vec![BalanceSnapshot {
                asset: "USD".to_string(),
                total: 10_000.0,
                available: 9_500.0,
            }],
            funding_8h: Some(0.001),
            margin: MarginSnapshot {
                balance_usd: 10_000.0,
                used_usd: 500.0,
                available_usd: 9_500.0,
            },
            liquidation: LiquidationSnapshot {
                price_liq: Some(95.0),
                dist_liq_sigma: None,
            },
        };
        cache
            .apply_account_event(&AccountEvent::Snapshot(account_snapshot))
            .expect("account apply");

        let snapshot = cache.snapshot(now_ms, cfg.main_loop_interval_ms * 2);
        apply_account_snapshot_to_state(&cfg, &snapshot, &mut state, now_ms);

        let account_cache = &cache.account[0];
        let venue_state = &state.venues[0];
        assert_close(account_cache.margin.balance_usd, 10_000.0, 1e-9);
        assert_close(venue_state.margin_balance_usd, 10_000.0, 1e-9);
        assert_close(venue_state.margin_used_usd, 500.0, 1e-9);
        assert_close(venue_state.margin_available_usd, 9_500.0, 1e-9);
        assert_eq!(venue_state.funding_8h, 0.001);
        assert_eq!(venue_state.price_liq, Some(95.0));

        let expected_dist =
            (venue_state.mid.unwrap() - 95.0).abs() / (state.sigma_eff * state.fair_value.unwrap());
        assert_close(venue_state.dist_liq_sigma, expected_dist, 1e-9);
        assert_close(venue_state.position_tao, 2.0, 1e-9);
    }
}
