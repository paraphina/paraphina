//! Live trading scaffolding (feature-gated).
//!
//! This module is intentionally minimal: it defines the canonical cache
//! and event model for live trading without introducing any network I/O.

pub mod connector_registry;
pub mod connectors;
pub mod gateway;
pub mod instrument;
mod market_publisher;
pub mod mock_exchange;
pub mod ops;
pub mod order_state;
pub mod orderbook_l2;
pub mod paper_adapter;
pub mod phase51_forward_refresh_capture;
pub mod phase51_target_key_registry;
pub mod rest_health_monitor;
pub mod runner;
pub mod shadow_adapter;
pub mod shared_venue_ages;
pub mod state_cache;
pub mod supervision;
pub mod trade_mode;
pub mod types;
pub mod venue_health;
pub mod venue_health_enforcer;
pub mod venues;

pub use gateway::{LiveGateway, LiveGatewayError, LiveGatewayErrorKind, LiveRestClient};
pub(crate) use market_publisher::{
    live_market_pub_drain_max, live_market_pub_queue_cap, MarketPublisher,
};
pub use orderbook_l2::{
    BookLevel, BookLevelDelta, BookSide, DepthConfig, DerivedBookMetrics, OrderBookError,
    OrderBookL2,
};
pub use phase51_forward_refresh_capture::{
    Phase51CaptureExecutionMode, Phase51CaptureTargetKey, Phase51ForwardRefreshCapture,
    Phase51LighterNativeLimitPressure, Phase51LiveNativeRoleCanaryContext, Phase51VenueNativeRole,
};
pub use runner::{
    run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRunSummary, LiveTelemetry,
    LiveTelemetryStats, ResponseMode,
};
pub use shadow_adapter::ShadowAckAdapter;
pub use shared_venue_ages::SharedVenueAges;
pub use state_cache::{
    CanonicalCacheSnapshot, ReconciliationReport, VenueAccountCache, VenueMarketCache,
};
pub use supervision::{spawn_supervised, spawn_supervised_with_threshold};
pub use trade_mode::{
    resolve_effective_trade_mode, EffectiveTradeMode, TradeMode, TradeModeSource,
};
pub use types::{AccountEvent, ExecutionEvent, MarketDataEvent};
