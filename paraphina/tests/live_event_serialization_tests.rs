#[cfg(feature = "live")]
mod tests {
    use paraphina::live::types::*;
    use paraphina::live::BookLevel;

    #[test]
    fn serialize_market_data_event_snapshot() {
        let event = MarketDataEvent::L2Snapshot(L2Snapshot {
            venue_index: 0,
            venue_id: "venue_a".to_string(),
            seq: 42,
            timestamp_ms: 1_000,
            bids: vec![BookLevel {
                price: 100.0,
                size: 1.0,
            }],
            asks: vec![BookLevel {
                price: 101.0,
                size: 2.0,
            }],
        });

        let json = serde_json::to_string(&event).unwrap();
        let expected = r#"{"L2Snapshot":{"venue_index":0,"venue_id":"venue_a","seq":42,"timestamp_ms":1000,"bids":[{"price":100.0,"size":1.0}],"asks":[{"price":101.0,"size":2.0}]}}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn serialize_account_snapshot() {
        let event = AccountEvent::Snapshot(AccountSnapshot {
            venue_index: 1,
            venue_id: "venue_b".to_string(),
            seq: 7,
            timestamp_ms: 2_000,
            positions: vec![PositionSnapshot {
                symbol: "TAO-PERP".to_string(),
                size: 5.0,
                entry_price: 95.0,
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
                price_liq: Some(70.0),
                dist_liq_sigma: Some(8.0),
            },
        });

        let json = serde_json::to_string(&event).unwrap();
        let expected = r#"{"Snapshot":{"venue_index":1,"venue_id":"venue_b","seq":7,"timestamp_ms":2000,"positions":[{"symbol":"TAO-PERP","size":5.0,"entry_price":95.0}],"balances":[{"asset":"USD","total":10000.0,"available":9500.0}],"funding_8h":0.001,"margin":{"balance_usd":10000.0,"used_usd":500.0,"available_usd":9500.0},"liquidation":{"price_liq":70.0,"dist_liq_sigma":8.0}}}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn serialize_execution_event_order_accepted() {
        let event = ExecutionEvent::OrderAccepted(OrderAccepted {
            venue_index: 0,
            venue_id: "venue_a".to_string(),
            seq: 99,
            timestamp_ms: 3_000,
            order_id: "order_1".to_string(),
            client_order_id: Some("client_1".to_string()),
            side: paraphina::Side::Buy,
            price: 100.0,
            size: 0.5,
            purpose: paraphina::OrderPurpose::Mm,
        });

        let json = serde_json::to_string(&event).unwrap();
        let expected = r#"{"OrderAccepted":{"venue_index":0,"venue_id":"venue_a","seq":99,"timestamp_ms":3000,"order_id":"order_1","client_order_id":"client_1","side":"Buy","price":100.0,"size":0.5,"purpose":"Mm"}}"#;
        assert_eq!(json, expected);
    }
}
