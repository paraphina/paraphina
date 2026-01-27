#[cfg(feature = "live")]
mod tests {
    use paraphina::live::ops::{config_hash, format_startup_log};
    use paraphina::live::TradeMode;
    use paraphina::{BuildInfo, Config};

    #[test]
    fn config_hash_is_stable_for_same_config() {
        let cfg = Config::default();
        let h1 = config_hash(&cfg);
        let h2 = config_hash(&cfg);
        assert_eq!(h1, h2);
    }

    #[test]
    fn startup_log_is_deterministic() {
        let cfg = Config::default();
        let build = BuildInfo::for_test();
        let log_a = format_startup_log(&cfg, &build, TradeMode::Shadow, "mock", "127.0.0.1:9898");
        let log_b = format_startup_log(&cfg, &build, TradeMode::Shadow, "mock", "127.0.0.1:9898");
        assert_eq!(log_a, log_b);
        assert!(log_a.contains("cfg_hash=0x"));
        assert!(log_a.contains("build_id="));
    }
}
