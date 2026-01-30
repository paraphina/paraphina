//! Live trading scaffolding (feature-gated).
//!
//! This module is intentionally minimal: it defines the canonical cache
//! and event model for live trading without introducing any network I/O.

pub mod connectors;
pub mod gateway;
pub mod instrument;
mod market_publisher;
pub mod mock_exchange;
pub mod ops;
pub mod order_state;
pub mod orderbook_l2;
pub mod paper_adapter;
pub mod runner;
pub mod shadow_adapter;
pub mod state_cache;
pub mod trade_mode;
pub mod types;
pub mod venue_health;
pub mod venues;

pub use gateway::{LiveGateway, LiveGatewayError, LiveGatewayErrorKind, LiveRestClient};
pub use orderbook_l2::{
    BookLevel, BookLevelDelta, BookSide, DepthConfig, DerivedBookMetrics, OrderBookError,
    OrderBookL2,
};
pub use runner::{
    run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRunSummary, LiveTelemetry,
    LiveTelemetryStats,
};
pub(crate) use market_publisher::MarketPublisher;
pub use shadow_adapter::ShadowAckAdapter;
pub use state_cache::{
    CanonicalCacheSnapshot, ReconciliationReport, VenueAccountCache, VenueMarketCache,
};
pub use trade_mode::{
    resolve_effective_trade_mode, EffectiveTradeMode, TradeMode, TradeModeSource,
};
pub use types::{AccountEvent, ExecutionEvent, MarketDataEvent};
