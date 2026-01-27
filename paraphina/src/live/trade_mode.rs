//! Trade mode resolution for live trading (feature-gated).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeMode {
    Shadow,
    Paper,
    Testnet,
    Live,
}

impl TradeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TradeMode::Shadow => "shadow",
            TradeMode::Paper => "paper",
            TradeMode::Testnet => "testnet",
            TradeMode::Live => "live",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "shadow" | "safe" | "s" => Some(TradeMode::Shadow),
            "paper" | "p" => Some(TradeMode::Paper),
            "testnet" | "tn" | "t" => Some(TradeMode::Testnet),
            "live" | "real" | "l" => Some(TradeMode::Live),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeModeSource {
    Cli,
    Env,
    Default,
}

impl TradeModeSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            TradeModeSource::Cli => "cli",
            TradeModeSource::Env => "env",
            TradeModeSource::Default => "default",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EffectiveTradeMode {
    pub trade_mode: TradeMode,
    pub source: TradeModeSource,
}

impl EffectiveTradeMode {
    pub fn log_startup(&self) {
        eprintln!(
            "effective_trade_mode={} source={}",
            self.trade_mode.as_str(),
            self.source.as_str()
        );
    }
}

/// Resolve trade mode with precedence:
/// 1) CLI (--trade-mode)
/// 2) Environment (PARAPHINA_TRADE_MODE)
/// 3) Default (Shadow)
pub fn resolve_effective_trade_mode(cli_mode: Option<TradeMode>) -> EffectiveTradeMode {
    if let Some(mode) = cli_mode {
        return EffectiveTradeMode {
            trade_mode: mode,
            source: TradeModeSource::Cli,
        };
    }

    if let Ok(env_val) = std::env::var("PARAPHINA_TRADE_MODE") {
        if !env_val.is_empty() {
            if let Some(mode) = TradeMode::parse(&env_val) {
                return EffectiveTradeMode {
                    trade_mode: mode,
                    source: TradeModeSource::Env,
                };
            }
            eprintln!(
                "[config] WARN: invalid PARAPHINA_TRADE_MODE={:?}; using default shadow",
                env_val
            );
        }
    }

    EffectiveTradeMode {
        trade_mode: TradeMode::Shadow,
        source: TradeModeSource::Default,
    }
}
