#![cfg(feature = "live_hyperliquid")]

use std::path::PathBuf;

use paraphina::config::Config;
use paraphina::live::connectors::hyperliquid::HyperliquidFixtureFeed;
use paraphina::live::types::MarketDataEvent;
use paraphina::orderbook_l2::BookLevel;
use paraphina::state::GlobalState;
use tokio::sync::mpsc;

fn apply_market_event_to_state(state: &mut GlobalState, cfg: &Config, event: &MarketDataEvent) {
    let alpha_short = cfg.volatility.fv_vol_alpha_short;
    let alpha_long = cfg.volatility.fv_vol_alpha_long;
    let max_levels = cfg.book.depth_levels.max(1) as usize;
    match event {
        MarketDataEvent::L2Snapshot(snapshot) => {
            let venue = state.venues.get_mut(snapshot.venue_index).expect("venue");
            let _ = venue.apply_l2_snapshot(
                &snapshot.bids,
                &snapshot.asks,
                snapshot.seq,
                snapshot.timestamp_ms,
                max_levels,
                alpha_short,
                alpha_long,
            );
        }
        MarketDataEvent::L2Delta(delta) => {
            let venue = state.venues.get_mut(delta.venue_index).expect("venue");
            let _ = venue.apply_l2_delta(
                &delta.changes,
                delta.seq,
                delta.timestamp_ms,
                max_levels,
                alpha_short,
                alpha_long,
            );
        }
        MarketDataEvent::Trade(_) | MarketDataEvent::FundingUpdate(_) => {}
    }
}

#[tokio::test]
async fn hyperliquid_fixture_populates_core_l2_book() {
    let cfg = Config::default();
    let mut state = GlobalState::new(&cfg);

    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/hyperliquid");
    let feed = HyperliquidFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
    let (market_tx, mut market_rx) = mpsc::channel(16);
    let venue_index = 1;
    let start_ms = 1_000;
    let step_ms = 250;
    let ticks = 1;

    feed.run_ticks(market_tx, venue_index, start_ms, step_ms, ticks)
        .await;
    while let Some(event) = market_rx.recv().await {
        apply_market_event_to_state(&mut state, &cfg, &event);
    }

    let venue = state.venues.get(venue_index).expect("venue");
    let best_bid = venue
        .orderbook_l2
        .bids()
        .first()
        .copied()
        .unwrap_or(BookLevel {
            price: 0.0,
            size: 0.0,
        });
    let best_ask = venue
        .orderbook_l2
        .asks()
        .first()
        .copied()
        .unwrap_or(BookLevel {
            price: 0.0,
            size: 0.0,
        });

    assert_eq!(best_bid.price, 100.0);
    assert_eq!(best_bid.size, 1.2);
    assert_eq!(best_ask.price, 101.0);
    assert_eq!(best_ask.size, 2.5);
    assert_eq!(venue.last_book_update_ms, Some(start_ms));
    assert_eq!(venue.last_mid_update_ms, Some(start_ms));
}
