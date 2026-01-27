#[cfg(feature = "live_extended")]
mod extended_ws_defaults_tests {
    use paraphina::live::connectors::extended::ExtendedConfig;

    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn clear(keys: &[&str]) -> Self {
            let mut saved = Vec::with_capacity(keys.len());
            for &key in keys {
                let existing = std::env::var(key).ok();
                saved.push((key.to_string(), existing));
                std::env::remove_var(key);
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.saved.drain(..) {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }

    #[test]
    fn default_orderbook_ws_url_uses_starknet_mainnet() {
        let _guard = EnvGuard::clear(&["EXTENDED_WS_URL"]);
        let cfg = ExtendedConfig::from_env();
        let ws_url = cfg.orderbook_ws_url();
        assert_eq!(
            ws_url,
            "wss://api.starknet.extended.exchange/stream.extended.exchange/v1/orderbooks/BTC-USD?depth=1"
        );
    }
}
