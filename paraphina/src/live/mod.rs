//! Live trading scaffolding (feature-gated).
//!
//! This module is intentionally minimal: it defines the canonical cache
//! and event model for live trading without introducing any network I/O.

pub mod orderbook_l2;
pub mod order_state;
pub mod runner;
pub mod mock_exchange;
pub mod ops;
pub mod trade_mode;
pub mod shadow_adapter;
pub mod gateway;
pub mod instrument;
pub mod venue_health;
pub mod connectors;
pub mod state_cache;
pub mod types;
pub mod venues;

pub use orderbook_l2::{
    BookLevel, BookLevelDelta, BookSide, DepthConfig, DerivedBookMetrics, OrderBookError,
    OrderBookL2,
};
pub use runner::{
    LiveChannels, LiveOrderRequest, LiveRunMode, LiveRunSummary, LiveTelemetry, LiveTelemetryStats,
    run_live_loop,
};
pub use gateway::{LiveGateway, LiveGatewayError, LiveGatewayErrorKind, LiveRestClient};
pub use state_cache::{
    CanonicalCacheSnapshot, ReconciliationReport, VenueAccountCache, VenueMarketCache,
};
pub use types::{AccountEvent, ExecutionEvent, MarketDataEvent};
pub use trade_mode::{EffectiveTradeMode, TradeMode, TradeModeSource, resolve_effective_trade_mode};
pub use shadow_adapter::ShadowAckAdapter;
