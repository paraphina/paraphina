//! Paradex connector (public WS market data + fixtures, feature-gated).

#[cfg(feature = "live_paradex")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_paradex")]
pub const SUPPORTS_EXECUTION: bool = true;

const PARADEX_STALE_MS_DEFAULT: u64 = 10_000;
const PARADEX_WATCHDOG_TICK_MS: u64 = 200;
const PARADEX_STALE_GUARDBAND_MS: u64 = 400;
const PARADEX_ORDERBOOK_STALE_GUARDBAND_MS: u64 = 1_200;
const PARADEX_MARKET_PUB_QUEUE_CAP: usize = 256;
const PARADEX_MARKET_PUB_DRAIN_MAX: usize = 64;
const PARADEX_TOKEN_REFRESH_DEFAULT_SECS: u64 = 240;
const PARADEX_RATE_LIMIT_BASE_BACKOFF_MS: u64 = 2_000;
const PARADEX_RATE_LIMIT_MAX_BACKOFF_MS: u64 = 15_000;
const PARADEX_UI_BOOK_TRUTH_POLL_MS_DEFAULT: u64 = 500;
const PARADEX_UI_BOOK_TRUTH_STALE_MS_DEFAULT: u64 = 1_500;
const PARADEX_BBO_QUIET_REFRESH_MS_DEFAULT: u64 = 1_000;
const PARADEX_BBO_TRANSPORT_STALE_MS_DEFAULT: u64 = 5_000;
const PARADEX_PRIVATE_WS_READ_TIMEOUT_MS_DEFAULT: u64 = 30_000;
const PARADEX_BATCH_MAX_ITEMS: usize = 50;
const PARADEX_REPLACE_IDENTITY_CACHE_MAX_ITEMS: usize = 4_096;
const PARADEX_SIGN_ORDER_CMD_DEFAULT: &str =
    "/opt/paraphina/.venv_paradex/bin/python3 /opt/paraphina/tools/paradex_sign_order.py";

static MONO_START: OnceLock<Instant> = OnceLock::new();
static PARADEX_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static PARADEX_PING_SENT_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_PING_SEND_FAIL_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_INTERACTIVE_TOP_AUDIT_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_INTERACTIVE_PUBLIC_TOP_AUDIT_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_PROFILE_USAGE_REUSE_AUDIT_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_BBO_QUIET_REFRESH_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_PRIVATE_ORDER_TRUTH_AUDIT_COUNT: AtomicU64 = AtomicU64::new(0);
static PARADEX_RECONNECT_COUNTS: OnceLock<StdMutex<BTreeMap<&'static str, u64>>> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn paradex_stale_ms() -> u64 {
    if let Some(explicit) = std::env::var("PARAPHINA_PARADEX_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return explicit;
    }
    if let Some(state_override_ms) = std::env::var("PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
    {
        let guardband_ms = match ParadexPublicFeedMode::from_process_env() {
            ParadexPublicFeedMode::Bbo => PARADEX_STALE_GUARDBAND_MS,
            ParadexPublicFeedMode::Orderbook { .. } => PARADEX_ORDERBOOK_STALE_GUARDBAND_MS,
        };
        // Keep the connector watchdog slightly ahead of the runtime stale budget
        // so Paradex reconnects before the shared canary logic kills the surface.
        return state_override_ms.saturating_sub(guardband_ms).max(1_000);
    }
    PARADEX_STALE_MS_DEFAULT
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParadexPublicFeedMode {
    Bbo,
    Orderbook {
        feed_type: String,
        refresh_rate: Option<String>,
    },
}

impl ParadexPublicFeedMode {
    fn from_env(raw: &str) -> Self {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "" | "bbo" => Self::Bbo,
            "orderbook" | "orderbook_snapshot" => Self::Orderbook {
                feed_type: "snapshot".to_string(),
                refresh_rate: Some("100ms".to_string()),
            },
            "orderbook_deltas" | "deltas" => Self::Orderbook {
                feed_type: "deltas".to_string(),
                refresh_rate: None,
            },
            "orderbook_interactive" | "interactive" => Self::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("100ms".to_string()),
            },
            "orderbook_interactive_50ms" | "interactive_50ms" => Self::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("50ms".to_string()),
            },
            other if other.starts_with("orderbook_") => Self::Orderbook {
                feed_type: other.trim_start_matches("orderbook_").to_string(),
                refresh_rate: Some("100ms".to_string()),
            },
            _ => Self::Bbo,
        }
    }

    fn from_process_env() -> Self {
        let raw =
            std::env::var("PARAPHINA_PARADEX_PUBLIC_FEED").unwrap_or_else(|_| "bbo".to_string());
        Self::from_env(&raw)
    }

    fn channel(&self, market: &str) -> String {
        match self {
            Self::Bbo => format!("bbo.{market}"),
            Self::Orderbook {
                feed_type,
                refresh_rate,
            } => {
                if feed_type.eq_ignore_ascii_case("deltas") {
                    format!("order_book.{market}.deltas")
                } else {
                    format!(
                        "order_book.{market}.{feed_type}@15@{}",
                        refresh_rate.as_deref().unwrap_or("100ms")
                    )
                }
            }
        }
    }
}

fn paradex_public_feed_uses_ui_book_truth_poller(mode: &ParadexPublicFeedMode) -> bool {
    matches!(mode, ParadexPublicFeedMode::Bbo)
        || matches!(
            mode,
            ParadexPublicFeedMode::Orderbook { feed_type, .. }
                if feed_type.eq_ignore_ascii_case("interactive")
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParadexTokenUsage {
    Pro,
    Interactive,
}

impl ParadexTokenUsage {
    fn from_env(raw: &str) -> Self {
        if raw.trim().eq_ignore_ascii_case("interactive") {
            Self::Interactive
        } else {
            Self::Pro
        }
    }

    fn from_process_env() -> Self {
        let raw = std::env::var("PARADEX_TOKEN_USAGE").unwrap_or_else(|_| "pro".to_string());
        Self::from_env(&raw)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pro => "pro",
            Self::Interactive => "interactive",
        }
    }
}

fn paradex_ws_audit_enabled() -> bool {
    *PARADEX_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn paradex_ui_book_truth_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_UI_BOOK_TRUTH_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn paradex_ui_book_truth_poll_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_UI_BOOK_TRUTH_POLL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARADEX_UI_BOOK_TRUTH_POLL_MS_DEFAULT)
        .max(250)
}

fn paradex_ui_book_truth_stale_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_UI_BOOK_TRUTH_STALE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARADEX_UI_BOOK_TRUTH_STALE_MS_DEFAULT)
        .max(paradex_ui_book_truth_poll_ms())
}

fn paradex_ui_touch_reference_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_UI_TOUCH_REFERENCE_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn paradex_bbo_quiet_refresh_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_BBO_QUIET_REFRESH_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn paradex_bbo_backstop_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_BBO_BACKSTOP_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn paradex_bbo_quiet_refresh_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_BBO_QUIET_REFRESH_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARADEX_BBO_QUIET_REFRESH_MS_DEFAULT)
        .max(250)
}

fn paradex_bbo_transport_stale_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_BBO_TRANSPORT_STALE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARADEX_BBO_TRANSPORT_STALE_MS_DEFAULT)
        .max(paradex_bbo_quiet_refresh_ms())
}

fn paradex_private_ws_read_timeout_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_PRIVATE_WS_READ_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(PARADEX_PRIVATE_WS_READ_TIMEOUT_MS_DEFAULT)
        .max(5_000)
}

fn paradex_audit_reconnect(reason: &'static str) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    let mut counts = PARADEX_RECONNECT_COUNTS
        .get_or_init(|| StdMutex::new(BTreeMap::new()))
        .lock()
        .expect("paradex reconnect audit mutex poisoned");
    let count = counts
        .entry(reason)
        .and_modify(|value| *value += 1)
        .or_insert(1);
    eprintln!(
        "WS_AUDIT venue=paradex reconnect_reason={} count={}",
        reason, *count
    );
}

fn paradex_emit_bbo_quiet_refresh_audit(quiet_refresh_ms: u64, transport_stale_ms: u64) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    let count = PARADEX_BBO_QUIET_REFRESH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 3 || count % 100 == 0 {
        eprintln!(
            "WS_AUDIT venue=paradex bbo_quiet_refresh_count={} quiet_refresh_ms={} transport_stale_ms={}",
            count, quiet_refresh_ms, transport_stale_ms
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct ParadexBboTopState {
    bid_px: f64,
    bid_sz: f64,
    ask_px: f64,
    ask_sz: f64,
}

fn should_emit_bbo_quiet_refresh(
    now_ns: u64,
    last_ws_rx_ns: u64,
    last_book_event_ns: u64,
    quiet_refresh_ms: u64,
    transport_stale_ms: u64,
) -> bool {
    if last_ws_rx_ns == 0 || last_book_event_ns == 0 {
        return false;
    }
    age_ms(now_ns, last_ws_rx_ns) <= transport_stale_ms
        && age_ms(now_ns, last_book_event_ns) >= quiet_refresh_ms
}

#[derive(Debug, Default, Clone, Copy)]
struct ParadexTopEntry {
    price: Option<f64>,
    size: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ParadexInteractiveTop {
    best_bid_api: ParadexTopEntry,
    best_bid_interactive: ParadexTopEntry,
    best_ask_api: ParadexTopEntry,
    best_ask_interactive: ParadexTopEntry,
    seq_no: Option<u64>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ParadexUiBookTruthSnapshot {
    bid: ParadexTopEntry,
    ask: ParadexTopEntry,
    best_bid_api: ParadexTopEntry,
    best_bid_interactive: ParadexTopEntry,
    best_ask_api: ParadexTopEntry,
    best_ask_interactive: ParadexTopEntry,
    best_bid_interactive_from_top_level: bool,
    best_ask_interactive_from_top_level: bool,
    seq_no: Option<u64>,
    last_updated_at_ms: Option<i64>,
}

impl ParadexUiBookTruthSnapshot {
    fn has_any_truth(&self) -> bool {
        [
            self.bid.price,
            self.bid.size,
            self.ask.price,
            self.ask.size,
            self.best_bid_api.price,
            self.best_bid_api.size,
            self.best_bid_interactive.price,
            self.best_bid_interactive.size,
            self.best_ask_api.price,
            self.best_ask_api.size,
            self.best_ask_interactive.price,
            self.best_ask_interactive.size,
        ]
        .into_iter()
        .any(|value| value.is_some())
    }

    fn normalized_for_touch_reference(mut self, source: ParadexUiBookTruthSource) -> Self {
        if source != ParadexUiBookTruthSource::Interactive {
            return self;
        }

        if self.best_bid_interactive.price.is_none() {
            if let Some(price) = positive_finite(self.bid.price) {
                self.best_bid_interactive.price = Some(price);
                self.best_bid_interactive_from_top_level = true;
            }
        }
        if self.best_bid_interactive.size.is_none() {
            self.best_bid_interactive.size = positive_finite(self.bid.size);
        }

        if self.best_ask_interactive.price.is_none() {
            if let Some(price) = positive_finite(self.ask.price) {
                self.best_ask_interactive.price = Some(price);
                self.best_ask_interactive_from_top_level = true;
            }
        }
        if self.best_ask_interactive.size.is_none() {
            self.best_ask_interactive.size = positive_finite(self.ask.size);
        }

        self
    }

    fn ui_touch_source_kind(self) -> &'static str {
        if self.best_bid_interactive_from_top_level || self.best_ask_interactive_from_top_level {
            "top_level_fallback"
        } else {
            "split"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParadexUiBookTruthSource {
    Api,
    Interactive,
}

impl ParadexUiBookTruthSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Api => "api",
            Self::Interactive => "interactive",
        }
    }

    fn path(self, market: &str) -> String {
        match self {
            Self::Api => format!("/orderbook/{market}?depth=1"),
            Self::Interactive => format!("/orderbook/{market}/interactive?depth=1"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParadexUiTouchReferenceSample {
    snapshot: ParadexUiBookTruthSnapshot,
    received_at_ns: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct ParadexUiTouchReferenceCache {
    api: Option<ParadexUiTouchReferenceSample>,
    interactive: Option<ParadexUiTouchReferenceSample>,
}

#[derive(Debug, Default)]
struct ParadexUiTouchReferenceState {
    cache: StdMutex<ParadexUiTouchReferenceCache>,
}

impl ParadexUiTouchReferenceState {
    fn update(&self, source: ParadexUiBookTruthSource, snapshot: ParadexUiBookTruthSnapshot) {
        let sample = ParadexUiTouchReferenceSample {
            snapshot: snapshot.normalized_for_touch_reference(source),
            received_at_ns: mono_now_ns(),
        };
        let mut cache = self
            .cache
            .lock()
            .expect("paradex ui touch reference mutex poisoned");
        match source {
            ParadexUiBookTruthSource::Api => cache.api = Some(sample),
            ParadexUiBookTruthSource::Interactive => cache.interactive = Some(sample),
        }
    }

    fn apply_to_bbo(
        &self,
        bid_px: f64,
        bid_sz: f64,
        ask_px: f64,
        ask_sz: f64,
    ) -> (f64, f64, f64, f64) {
        self.apply_to_bbo_with_source(bid_px, bid_sz, ask_px, ask_sz)
            .0
    }

    fn apply_to_bbo_with_source(
        &self,
        bid_px: f64,
        bid_sz: f64,
        ask_px: f64,
        ask_sz: f64,
    ) -> ((f64, f64, f64, f64), Option<&'static str>) {
        let now_ns = mono_now_ns();
        let stale_ms = paradex_ui_book_truth_stale_ms();
        let cache = *self
            .cache
            .lock()
            .expect("paradex ui touch reference mutex poisoned");
        let sample = [cache.interactive, cache.api]
            .into_iter()
            .flatten()
            .filter(|entry| paradex_ui_touch_reference_sample_fresh(entry, now_ns, stale_ms))
            .max_by_key(|entry| entry.received_at_ns);
        let original = (bid_px, bid_sz, ask_px, ask_sz);
        let Some(entry) = sample else {
            return (original, None);
        };
        let adjusted =
            paradex_apply_ui_touch_reference(bid_px, bid_sz, ask_px, ask_sz, entry.snapshot);
        if adjusted != original {
            let source_kind = entry.snapshot.ui_touch_source_kind();
            paradex_emit_ui_touch_reference_applied_audit(source_kind, original, adjusted);
            return (adjusted, Some(source_kind));
        }
        (adjusted, None)
    }
}

fn paradex_ui_touch_reference_sample_fresh(
    sample: &ParadexUiTouchReferenceSample,
    now_ns: u64,
    stale_ms: u64,
) -> bool {
    if sample.received_at_ns == 0 || age_ms(now_ns, sample.received_at_ns) > stale_ms {
        return false;
    }
    match sample.snapshot.last_updated_at_ms {
        Some(last_updated_at_ms) => now_ms().saturating_sub(last_updated_at_ms) <= stale_ms as i64,
        None => true,
    }
}

fn positive_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|entry| entry.is_finite() && *entry > 0.0)
}

fn max_optional_f64(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    match (lhs, rhs) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn paradex_effective_bid_from_ui_truth(
    snapshot: ParadexUiBookTruthSnapshot,
) -> Option<ParadexTopEntry> {
    let api_px = positive_finite(snapshot.best_bid_api.price)?;
    let interactive_px = positive_finite(snapshot.best_bid_interactive.price)?;
    let api_sz = positive_finite(snapshot.best_bid_api.size);
    let interactive_sz = positive_finite(snapshot.best_bid_interactive.size);
    let entry = if interactive_px > api_px {
        ParadexTopEntry {
            price: Some(interactive_px),
            size: interactive_sz.or(api_sz),
        }
    } else if api_px > interactive_px {
        ParadexTopEntry {
            price: Some(api_px),
            size: api_sz.or(interactive_sz),
        }
    } else {
        ParadexTopEntry {
            price: Some(api_px),
            size: max_optional_f64(api_sz, interactive_sz),
        }
    };
    Some(entry)
}

fn paradex_effective_ask_from_ui_truth(
    snapshot: ParadexUiBookTruthSnapshot,
) -> Option<ParadexTopEntry> {
    let api_px = positive_finite(snapshot.best_ask_api.price)?;
    let interactive_px = positive_finite(snapshot.best_ask_interactive.price)?;
    let api_sz = positive_finite(snapshot.best_ask_api.size);
    let interactive_sz = positive_finite(snapshot.best_ask_interactive.size);
    let entry = if interactive_px < api_px {
        ParadexTopEntry {
            price: Some(interactive_px),
            size: interactive_sz.or(api_sz),
        }
    } else if api_px < interactive_px {
        ParadexTopEntry {
            price: Some(api_px),
            size: api_sz.or(interactive_sz),
        }
    } else {
        ParadexTopEntry {
            price: Some(api_px),
            size: max_optional_f64(api_sz, interactive_sz),
        }
    };
    Some(entry)
}

fn paradex_apply_ui_touch_reference(
    bid_px: f64,
    bid_sz: f64,
    ask_px: f64,
    ask_sz: f64,
    snapshot: ParadexUiBookTruthSnapshot,
) -> (f64, f64, f64, f64) {
    let original = (bid_px, bid_sz, ask_px, ask_sz);
    let mut adjusted = original;

    if let Some(entry) = paradex_effective_bid_from_ui_truth(snapshot) {
        let candidate_bid_px = entry.price.expect("effective bid price");
        let candidate_bid_sz = positive_finite(entry.size).unwrap_or(adjusted.1);
        if candidate_bid_px < adjusted.2 {
            adjusted.0 = candidate_bid_px;
            adjusted.1 = candidate_bid_sz;
        }
    }

    if let Some(entry) = paradex_effective_ask_from_ui_truth(snapshot) {
        let candidate_ask_px = entry.price.expect("effective ask price");
        let candidate_ask_sz = positive_finite(entry.size).unwrap_or(adjusted.3);
        if adjusted.0 < candidate_ask_px {
            adjusted.2 = candidate_ask_px;
            adjusted.3 = candidate_ask_sz;
        }
    }

    if !adjusted.0.is_finite()
        || !adjusted.1.is_finite()
        || !adjusted.2.is_finite()
        || !adjusted.3.is_finite()
        || adjusted.1 <= 0.0
        || adjusted.3 <= 0.0
        || adjusted.0 >= adjusted.2
    {
        return original;
    }
    adjusted
}

fn paradex_emit_ui_touch_reference_applied_audit(
    source_kind: &str,
    original: (f64, f64, f64, f64),
    adjusted: (f64, f64, f64, f64),
) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    eprintln!(
        "WS_AUDIT venue=paradex component=ui_touch_reference action=applied source_kind={} orig_bid={} orig_bid_sz={} orig_ask={} orig_ask_sz={} adj_bid={} adj_bid_sz={} adj_ask={} adj_ask_sz={}",
        source_kind,
        format_decimal(original.0),
        format_decimal(original.1),
        format_decimal(original.2),
        format_decimal(original.3),
        format_decimal(adjusted.0),
        format_decimal(adjusted.1),
        format_decimal(adjusted.2),
        format_decimal(adjusted.3),
    );
}

fn paradex_emit_profile_usage_audit(
    action: &str,
    token_usage: ParadexTokenUsage,
    auth_source: &str,
) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    if action == "reuse" {
        let count = PARADEX_PROFILE_USAGE_REUSE_AUDIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count > 5 && count % 100 != 0 {
            return;
        }
    }
    eprintln!(
        "WS_AUDIT venue=paradex component=profile_usage action={} token_usage={} auth_source={}",
        action,
        token_usage.as_str(),
        auth_source
    );
}

fn paradex_emit_order_flags_audit(action: &str, token_usage: ParadexTokenUsage, payload: &Value) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    let instruction = payload
        .get("instruction")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let flags = payload
        .get("flags")
        .and_then(|value| value.as_array())
        .map(|values| {
            let parts: Vec<&str> = values.iter().filter_map(|item| item.as_str()).collect();
            if parts.is_empty() {
                "none".to_string()
            } else {
                parts.join(",")
            }
        })
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "WS_AUDIT venue=paradex component=order_flags action={} token_usage={} instruction={} flags={}",
        action,
        token_usage.as_str(),
        instruction,
        flags
    );
}

fn paradex_emit_fill_flags_audit(token_usage: ParadexTokenUsage, flags: &[String]) {
    if !paradex_ws_audit_enabled() || flags.is_empty() {
        return;
    }
    eprintln!(
        "WS_AUDIT venue=paradex component=fill_flags token_usage={} flags={}",
        token_usage.as_str(),
        flags.join(",")
    );
}

fn paradex_emit_replace_identity_resolve_audit(source: &str, client_id: &str, order_id: &str) {
    eprintln!(
        "PARADEX_REPLACE_IDENTITY_RESOLVE source={} client_id_state={} order_id_state={}",
        paradex_sanitized_audit_status(source),
        paradex_audit_id_state(Some(client_id)),
        paradex_audit_id_state(Some(order_id))
    );
}

fn paradex_emit_replace_identity_resolve_failed_audit(reason: &str, client_id: &str) {
    eprintln!(
        "PARADEX_REPLACE_IDENTITY_RESOLVE_FAILED reason={} client_id_state={}",
        paradex_sanitized_audit_status(reason),
        paradex_audit_id_state(Some(client_id))
    );
}

fn paradex_emit_open_identity_normalized_audit(source: &str, client_id: &str, order_id: &str) {
    eprintln!(
        "PARADEX_OPEN_IDENTITY_NORMALIZED source={} client_id_state={} order_id_state={}",
        paradex_sanitized_audit_status(source),
        paradex_audit_id_state(Some(client_id)),
        paradex_audit_id_state(Some(order_id))
    );
}

fn paradex_audit_id_state(raw: Option<&str>) -> &'static str {
    match raw {
        Some(value) if !value.trim().is_empty() => "present_redacted",
        _ => "absent",
    }
}

fn paradex_sanitized_audit_status(raw: &str) -> &str {
    if raw.is_empty()
        || raw.len() > 32
        || raw
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'))
    {
        "other"
    } else {
        raw
    }
}

fn paradex_open_identity_unresolved_audit_line(reason: &str, client_id: &str) -> String {
    format!(
        "PARADEX_OPEN_IDENTITY_UNRESOLVED reason={} client_id_state={}",
        paradex_sanitized_audit_status(reason),
        paradex_audit_id_state(Some(client_id)),
    )
}

fn paradex_emit_open_identity_unresolved_audit(reason: &str, client_id: &str) {
    eprintln!(
        "{}",
        paradex_open_identity_unresolved_audit_line(reason, client_id)
    );
}

fn paradex_native_replace_audit_line(
    submit_source: &str,
    order_id: &str,
    client_order_id: Option<&str>,
) -> String {
    format!(
        "PARADEX_NATIVE_REPLACE submit_source={} order_id_state={} client_id_state={}",
        paradex_sanitized_audit_status(submit_source),
        paradex_audit_id_state(Some(order_id)),
        paradex_audit_id_state(client_order_id),
    )
}

fn paradex_emit_native_replace_audit(
    submit_source: &str,
    order_id: &str,
    client_order_id: Option<&str>,
) {
    eprintln!(
        "{}",
        paradex_native_replace_audit_line(submit_source, order_id, client_order_id)
    );
}

fn paradex_private_order_truth_audit_line(
    count: u64,
    status: &str,
    order_id: &str,
    client_order_id: Option<&str>,
    _seq_no: u64,
) -> String {
    format!(
        "PARADEX_PRIVATE_ORDER_TRUTH count={} status={} order_id_state={} client_id_state={} seq_no_state=present_redacted",
        count,
        paradex_sanitized_audit_status(status),
        paradex_audit_id_state(Some(order_id)),
        paradex_audit_id_state(client_order_id),
    )
}

fn paradex_emit_private_order_truth_audit(
    status: &str,
    order_id: &str,
    client_order_id: Option<&str>,
    seq_no: u64,
) {
    let count = PARADEX_PRIVATE_ORDER_TRUTH_AUDIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count > 5 && count % 100 != 0 {
        return;
    }
    eprintln!(
        "{}",
        paradex_private_order_truth_audit_line(count, status, order_id, client_order_id, seq_no)
    );
}

fn paradex_emit_interactive_top_audit(feed_type: &str, top: ParadexInteractiveTop) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    let count = PARADEX_INTERACTIVE_TOP_AUDIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count > 5 && count % 250 != 0 {
        return;
    }
    eprintln!(
        concat!(
            "WS_AUDIT venue=paradex component=interactive_top count={} feed_type={} seq_no={} ",
            "best_bid_api_price={} best_bid_api_size={} ",
            "best_bid_interactive_price={} best_bid_interactive_size={} ",
            "best_ask_api_price={} best_ask_api_size={} ",
            "best_ask_interactive_price={} best_ask_interactive_size={}"
        ),
        count,
        feed_type,
        top.seq_no
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_string()),
        format_optional_decimal(top.best_bid_api.price),
        format_optional_decimal(top.best_bid_api.size),
        format_optional_decimal(top.best_bid_interactive.price),
        format_optional_decimal(top.best_bid_interactive.size),
        format_optional_decimal(top.best_ask_api.price),
        format_optional_decimal(top.best_ask_api.size),
        format_optional_decimal(top.best_ask_interactive.price),
        format_optional_decimal(top.best_ask_interactive.size),
    );
}

fn paradex_emit_interactive_public_top_audit(
    top_source: &str,
    bid_px: f64,
    bid_sz: f64,
    ask_px: f64,
    ask_sz: f64,
) {
    let count = PARADEX_INTERACTIVE_PUBLIC_TOP_AUDIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count > 5 && count % 250 != 0 {
        return;
    }
    eprintln!(
        concat!(
            "PARADEX_INTERACTIVE_PUBLIC_TOP count={} source=interactive_orderbook ",
            "top_source={} bid={} bid_sz={} ask={} ask_sz={}"
        ),
        count, top_source, bid_px, bid_sz, ask_px, ask_sz,
    );
}

fn paradex_emit_ui_book_truth_audit(
    source: ParadexUiBookTruthSource,
    status: &str,
    token_usage: ParadexTokenUsage,
    snapshot: Option<&ParadexUiBookTruthSnapshot>,
    error_class: Option<&str>,
) {
    if !paradex_ws_audit_enabled() {
        return;
    }
    let snapshot = snapshot.copied().unwrap_or_default();
    eprintln!(
        concat!(
            "WS_AUDIT venue=paradex component=ui_book_truth source={} status={} token_usage={} error_class={} ",
            "seq_no={} last_updated_at_ms={} bid_px={} bid_sz={} ask_px={} ask_sz={} ",
            "best_bid_api_px={} best_bid_api_sz={} best_bid_interactive_px={} best_bid_interactive_sz={} ",
            "best_ask_api_px={} best_ask_api_sz={} best_ask_interactive_px={} best_ask_interactive_sz={}"
        ),
        source.as_str(),
        status,
        token_usage.as_str(),
        error_class.unwrap_or("none"),
        snapshot
            .seq_no
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_string()),
        snapshot
            .last_updated_at_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_string()),
        format_optional_decimal(snapshot.bid.price),
        format_optional_decimal(snapshot.bid.size),
        format_optional_decimal(snapshot.ask.price),
        format_optional_decimal(snapshot.ask.size),
        format_optional_decimal(snapshot.best_bid_api.price),
        format_optional_decimal(snapshot.best_bid_api.size),
        format_optional_decimal(snapshot.best_bid_interactive.price),
        format_optional_decimal(snapshot.best_bid_interactive.size),
        format_optional_decimal(snapshot.best_ask_api.price),
        format_optional_decimal(snapshot.best_ask_api.size),
        format_optional_decimal(snapshot.best_ask_interactive.price),
        format_optional_decimal(snapshot.best_ask_interactive.size),
    );
}

fn format_optional_decimal(value: Option<f64>) -> String {
    value
        .map(format_decimal)
        .unwrap_or_else(|| "na".to_string())
}

fn paradex_is_rate_limit_error(message: &str) -> bool {
    message.to_ascii_lowercase().contains("rate limit")
}

fn paradex_next_rate_limit_backoff_ms(current_ms: u64, poll_ms: u64) -> u64 {
    if current_ms == 0 {
        poll_ms.max(PARADEX_RATE_LIMIT_BASE_BACKOFF_MS)
    } else {
        (current_ms * 2).min(PARADEX_RATE_LIMIT_MAX_BACKOFF_MS)
    }
}

#[allow(dead_code)]
fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
    /// Tracks the last time a book event (snapshot or delta) was decoded into a
    /// publishable MarketDataEvent. Used by the watchdog to detect "WS alive but
    /// no book data" scenarios where non-book messages keep last_ws_rx_ns fresh.
    last_book_event_ns: AtomicU64,
}

impl Freshness {
    fn reset_for_new_connection(&self) {
        self.last_ws_rx_ns.store(0, Ordering::Relaxed);
        self.last_data_rx_ns.store(0, Ordering::Relaxed);
        self.last_parsed_ns.store(0, Ordering::Relaxed);
        self.last_published_ns.store(0, Ordering::Relaxed);
        self.last_book_event_ns.store(0, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
        let last_book = self.last_book_event_ns.load(Ordering::Relaxed);
        let last_pub = self.last_published_ns.load(Ordering::Relaxed);
        let anchor = last_book.max(last_pub);
        if anchor == 0 {
            connect_start_ns
        } else {
            anchor
        }
    }
}

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::super::gateway::{
    BoxFuture, LiveGatewayError, LiveGatewayErrorKind, LiveRestCancelAllRequest,
    LiveRestCancelRequest, LiveRestClient, LiveRestPlaceRequest, LiveRestReplaceRequest,
    LiveRestResponse, LiveResult,
};
use super::super::orderbook_l2::{BookLevel, BookLevelDelta, BookSide};
use super::super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, FundingUpdate,
    LiquidationSnapshot, MarginSnapshot, MarketDataEvent, OpenOrderSnapshot, OrderSnapshot,
    Phase51ForwardRefreshNativeRole, Phase51ForwardRefreshSourceOwnerFill, PositionSnapshot,
    TopOfBook,
};
use crate::live::{live_market_pub_drain_max, live_market_pub_queue_cap, MarketPublisher};
use crate::types::{FundingSource, SettlementPriceKind, Side, TimeInForce, TimestampMs};

#[derive(Debug, Clone)]
pub struct ParadexConfig {
    pub ws_url: String,
    pub rest_url: String,
    pub auth_url: String,
    pub token_usage: ParadexTokenUsage,
    pub market: String,
    pub account_path: String,
    pub order_path: String,
    pub venue_index: usize,
    pub jwt: Option<String>,
    pub jwt_cmd: Option<String>,
    pub sign_order_cmd: Option<String>,
    pub auth_payload_json: Option<Value>,
    pub token_refresh_secs: u64,
    pub record_dir: Option<PathBuf>,
}

impl ParadexConfig {
    pub fn from_env() -> Self {
        let ws_url = std::env::var("PARADEX_WS_URL")
            .unwrap_or_else(|_| "wss://ws.api.prod.paradex.trade/v1".to_string());
        let rest_url = std::env::var("PARADEX_REST_URL")
            .unwrap_or_else(|_| "https://api.prod.paradex.trade/v1".to_string());
        let token_usage = ParadexTokenUsage::from_process_env();
        let auth_url = std::env::var("PARADEX_AUTH_URL")
            .unwrap_or_else(|_| "https://api.prod.paradex.trade/v1/auth".to_string());
        let market = std::env::var("PARADEX_MARKET").unwrap_or_else(|_| "BTC-USD-PERP".to_string());
        let account_path =
            std::env::var("PARADEX_ACCOUNT_PATH").unwrap_or_else(|_| "/account".to_string());
        let order_path =
            std::env::var("PARADEX_ORDER_PATH").unwrap_or_else(|_| "/orders".to_string());
        let jwt = std::env::var("PARADEX_JWT").ok();
        let jwt_cmd = std::env::var("PARADEX_JWT_CMD")
            .ok()
            .map(|raw| raw.trim().to_string())
            .filter(|raw| !raw.is_empty());
        let sign_order_cmd = std::env::var("PARADEX_SIGN_ORDER_CMD")
            .ok()
            .map(|raw| raw.trim().to_string())
            .filter(|raw| !raw.is_empty())
            .or_else(|| {
                let l2_address = std::env::var("PARADEX_L2_ADDRESS").ok()?;
                let l2_private_key = std::env::var("PARADEX_L2_PRIVATE_KEY").ok()?;
                if l2_address.trim().is_empty() || l2_private_key.trim().is_empty() {
                    return None;
                }
                Some(PARADEX_SIGN_ORDER_CMD_DEFAULT.to_string())
            });
        let auth_payload_json = std::env::var("PARADEX_AUTH_PAYLOAD_JSON")
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
        let token_refresh_secs = std::env::var("PARADEX_TOKEN_REFRESH_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|raw| *raw > 0)
            .unwrap_or(PARADEX_TOKEN_REFRESH_DEFAULT_SECS);
        Self {
            ws_url,
            rest_url,
            auth_url,
            token_usage,
            market,
            account_path,
            order_path,
            venue_index: 0,
            jwt,
            jwt_cmd,
            sign_order_cmd,
            auth_payload_json,
            token_refresh_secs,
            record_dir: None,
        }
    }

    pub fn with_record_dir(mut self, dir: PathBuf) -> Self {
        self.record_dir = Some(dir);
        self
    }

    pub fn has_auth(&self) -> bool {
        self.jwt.is_some() || self.jwt_cmd.is_some() || self.auth_payload_json.is_some()
    }

    pub fn has_refreshable_auth(&self) -> bool {
        self.jwt_cmd.is_some() || self.auth_payload_json.is_some()
    }

    fn auth_url_with_token_usage(&self) -> String {
        if self.token_usage != ParadexTokenUsage::Interactive {
            return self.auth_url.clone();
        }
        if self.auth_url.contains("token_usage=") {
            return self.auth_url.clone();
        }
        let separator = if self.auth_url.contains('?') {
            '&'
        } else {
            '?'
        };
        format!("{}{separator}token_usage=interactive", self.auth_url)
    }
}

#[derive(Debug)]
pub struct ParadexConnector {
    cfg: ParadexConfig,
    http: Client,
    market_publisher: MarketPublisher,
    recorder: Option<Mutex<ParadexRecorder>>,
    freshness: Arc<Freshness>,
    ui_truth_client: Option<ParadexRestClient>,
    ui_touch_reference_state: Option<Arc<ParadexUiTouchReferenceState>>,
    is_fixture: bool,
}

impl ParadexConnector {
    pub fn new(cfg: ParadexConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| ParadexRecorder::new(dir).ok())
            .map(Mutex::new);
        let is_fixture = std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("PARADEX_FIXTURE_MODE").is_ok();
        let freshness = Arc::new(Freshness::default());
        let publish_freshness = freshness.clone();
        let on_published = Arc::new(move || {
            publish_freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        });
        let market_publisher = MarketPublisher::new(
            live_market_pub_queue_cap(PARADEX_MARKET_PUB_QUEUE_CAP),
            live_market_pub_drain_max(PARADEX_MARKET_PUB_DRAIN_MAX),
            "paradex",
            market_tx.clone(),
            Some(Arc::new(move || is_fixture || Self::fixture_mode_now())),
            Arc::new(|event: &MarketDataEvent| {
                matches!(
                    event,
                    MarketDataEvent::L2Delta(_) | MarketDataEvent::L2Snapshot(_)
                )
            }),
            Some(on_published),
            "paradex market_tx closed",
            "paradex market publish queue closed",
        );
        let ui_truth_client = (paradex_ui_book_truth_enabled() && cfg.has_auth())
            .then(|| ParadexRestClient::new(cfg.clone()));
        let ui_touch_reference_state = (paradex_ui_touch_reference_enabled()
            && ui_truth_client.is_some())
        .then(|| Arc::new(ParadexUiTouchReferenceState::default()));
        let connector = Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("paradex http client build"),
            market_publisher,
            recorder,
            freshness,
            ui_truth_client,
            ui_touch_reference_state,
            is_fixture,
        };
        connector
    }

    fn fixture_mode_now() -> bool {
        std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("PARADEX_FIXTURE_MODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures: u32 = 0;

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = std::time::Instant::now();

            // Layer C: session-level timeout catches ALL hang scenarios.
            let max_session = Duration::from_secs(
                std::env::var("PARAPHINA_WS_MAX_SESSION_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(86_400), // 24h — Layer A enforcer handles stuck connections
            );
            let result = tokio::time::timeout(max_session, self.public_ws_once()).await;
            match result {
                Ok(Err(err)) => {
                    consecutive_failures += 1;
                    let level = if consecutive_failures >= 20 {
                        "ERROR"
                    } else if consecutive_failures >= 5 {
                        "WARN"
                    } else {
                        "INFO"
                    };
                    eprintln!(
                        "{level}: Paradex public WS error (consecutive_failures={consecutive_failures}): {err}"
                    );
                }
                Err(_timeout) => {
                    paradex_audit_reconnect("session_timeout");
                    eprintln!(
                        "ERROR: Paradex public WS session timeout ({}s) — force reconnect",
                        max_session.as_secs()
                    );
                    consecutive_failures += 1;
                }
                Ok(Ok(())) => {}
            }

            // FIX: Reset backoff and failure counter if connection was healthy for long enough
            let session_duration = session_start.elapsed();
            if session_duration >= healthy_threshold {
                if consecutive_failures > 0 {
                    eprintln!(
                        "INFO: Paradex WS session was healthy for {:?}; \
                         resetting backoff and failure counter (was {})",
                        session_duration, consecutive_failures
                    );
                }
                consecutive_failures = 0;
                backoff = Duration::from_secs(1);
            }

            // Escalating backoff caps: give upstream more time to recover
            let max_backoff = match consecutive_failures {
                0..=10 => Duration::from_secs(30),
                11..=20 => Duration::from_secs(60),
                _ => Duration::from_secs(120),
            };

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(max_backoff);
        }
    }

    pub async fn run_funding_polling(&self, interval_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut seq: u64 = 0;
        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg).await {
                Ok(mut update) => {
                    seq = seq.wrapping_add(1);
                    update.seq = seq;
                    if let Err(err) = self
                        .publish_market(MarketDataEvent::FundingUpdate(update))
                        .await
                    {
                        eprintln!("Paradex funding publish error: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Paradex funding polling error: {err}");
                }
            }
        }
    }

    async fn public_ws_once(&self) -> anyhow::Result<()> {
        eprintln!("INFO: Paradex public WS connecting url={}", self.cfg.ws_url);
        let (ws_stream, _) = tokio::time::timeout(
            Duration::from_secs(15),
            connect_async(self.cfg.ws_url.as_str()),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Paradex public WS connect timeout (15s)"))?
        .map_err(|e| anyhow::anyhow!("Paradex public WS connect error: {e}"))?;
        eprintln!("INFO: Paradex public WS connected url={}", self.cfg.ws_url);
        let (mut write, mut read) = ws_stream.split();
        let mut subscribed = false;
        let public_feed_mode = ParadexPublicFeedMode::from_process_env();
        let is_orderbook_mode = matches!(public_feed_mode, ParadexPublicFeedMode::Orderbook { .. });
        let interactive_public_top_mode = matches!(
            &public_feed_mode,
            ParadexPublicFeedMode::Orderbook { feed_type, .. }
                if feed_type.eq_ignore_ascii_case("interactive")
        );
        let channel = public_feed_mode.channel(&self.cfg.market);
        let subscribe = ParadexSubscribeCandidate::new(
            "subscribe",
            serde_json::json!({ "channel": channel.clone() }),
        );
        send_paradex_subscribe(&mut write, &subscribe).await?;
        eprintln!("INFO: Paradex subscribed channel={channel}");
        let bbo_backstop_enabled = is_orderbook_mode && paradex_bbo_backstop_enabled();
        if bbo_backstop_enabled {
            let bbo_channel = ParadexPublicFeedMode::Bbo.channel(&self.cfg.market);
            let bbo_subscribe = ParadexSubscribeCandidate::new(
                "subscribe",
                serde_json::json!({ "channel": bbo_channel.clone() }),
            );
            send_paradex_subscribe_with_id(&mut write, &bbo_subscribe, 2).await?;
            eprintln!(
                "INFO: Paradex subscribed secondary channel={bbo_channel} purpose=bbo_backstop"
            );
        }

        let mut tracker = ParadexSeqState::new(self.cfg.venue_index);
        let mut first_book_update_logged = false;
        let mut first_message_logged = false;
        let mut first_message_keys_logged = false;
        let mut logged_non_utf8_binary = false;
        let mut first_decoded_top_logged = false;
        let mut decode_miss_count = 0usize;
        let mut bbo_seq: u64 = 0;
        let mut last_bbo_top: Option<ParadexBboTopState> = None;
        let bbo_quiet_refresh_enabled = (matches!(public_feed_mode, ParadexPublicFeedMode::Bbo)
            || bbo_backstop_enabled)
            && paradex_bbo_quiet_refresh_enabled();
        let ui_truth_task = if paradex_public_feed_uses_ui_book_truth_poller(&public_feed_mode) {
            if let Some(client) = self.ui_truth_client.clone() {
                let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
                let handle = tokio::spawn(run_paradex_ui_book_truth_poller(
                    client,
                    stop_rx,
                    self.ui_touch_reference_state.clone(),
                ));
                Some((stop_tx, handle))
            } else {
                None
            }
        } else {
            None
        };
        let result = async {
            let ping_interval_ms: u64 = std::env::var("PARAPHINA_PARADEX_PING_INTERVAL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30_000);
            let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
            ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ping_timer.tick().await;
            let quiet_refresh_ms = paradex_bbo_quiet_refresh_ms();
            let transport_stale_ms = paradex_bbo_transport_stale_ms();
            let mut quiet_refresh_timer =
                tokio::time::interval(Duration::from_millis(quiet_refresh_ms));
            quiet_refresh_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            quiet_refresh_timer.tick().await;
            let connect_start_ns = mono_now_ns();
            self.freshness.reset_for_new_connection();
            let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
            let fixture_mode = std::env::var_os("PARADEX_FIXTURE_DIR").is_some()
                || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
                || std::env::var("PARADEX_FIXTURE_MODE")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
            let stale_ms = paradex_stale_ms();
            if fixture_mode {
                eprintln!("INFO: Paradex fixture mode detected; freshness watchdog disabled");
            } else {
                let watchdog_stale_ms = stale_ms;
                let watchdog_freshness = self.freshness.clone();
                tokio::spawn(async move {
                    let mut iv =
                        tokio::time::interval(Duration::from_millis(PARADEX_WATCHDOG_TICK_MS));
                    iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        iv.tick().await;
                        let now = mono_now_ns();
                        let anchor = watchdog_freshness.anchor_with_connect_start(connect_start_ns);
                        if anchor != 0 && age_ms(now, anchor) > watchdog_stale_ms {
                            let _ = stale_tx.send(());
                            break;
                        }
                    }
                });
            }
            loop {
                let msg = tokio::select! {
                    biased;
                    _ = &mut stale_rx => {
                        paradex_audit_reconnect("stale_watchdog");
                        anyhow::bail!("Paradex public WS stale: freshness exceeded {stale_ms}ms");
                    }
                    _ = quiet_refresh_timer.tick(), if bbo_quiet_refresh_enabled => {
                        let now_ns = mono_now_ns();
                        let last_ws_rx_ns = self.freshness.last_ws_rx_ns.load(Ordering::Relaxed);
                        let last_book_event_ns =
                            self.freshness.last_book_event_ns.load(Ordering::Relaxed);
                        if should_emit_bbo_quiet_refresh(
                            now_ns,
                            last_ws_rx_ns,
                            last_book_event_ns,
                            quiet_refresh_ms,
                            transport_stale_ms,
                        ) {
                            if let Some(top) = last_bbo_top {
                                if let Some((_top, snapshot)) = build_bbo_snapshot(
                                    self.cfg.venue_index,
                                    &self.cfg.market,
                                    &mut bbo_seq,
                                    top.bid_px,
                                    top.bid_sz,
                                    top.ask_px,
                                    top.ask_sz,
                                ) {
                                    self.freshness
                                        .last_parsed_ns
                                        .store(now_ns, Ordering::Relaxed);
                                    self.freshness
                                        .last_book_event_ns
                                        .store(now_ns, Ordering::Relaxed);
                                    paradex_emit_bbo_quiet_refresh_audit(
                                        quiet_refresh_ms,
                                        transport_stale_ms,
                                    );
                                    if let Err(err) = self.publish_market(snapshot).await {
                                        eprintln!("Paradex BBO quiet refresh market send failed: {err}");
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    _ = ping_timer.tick() => {
                        match write.send(Message::Ping(vec![b'p'].into())).await {
                            Ok(()) => {
                                if paradex_ws_audit_enabled() {
                                    let sent = PARADEX_PING_SENT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                                    if sent <= 3 || sent % 100 == 0 {
                                        eprintln!(
                                            "WS_AUDIT venue=paradex paradex_ping_sent_count={} interval_ms={}",
                                            sent, ping_interval_ms
                                        );
                                    }
                                }
                            }
                            Err(err) => {
                                paradex_audit_reconnect("ping_send_fail");
                                if paradex_ws_audit_enabled() {
                                    let fail =
                                        PARADEX_PING_SEND_FAIL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                                    eprintln!(
                                        "WS_AUDIT venue=paradex paradex_ping_send_fail_count={} err={}",
                                        fail, err
                                    );
                                }
                                anyhow::bail!("Paradex public WS ping send failed: {err}");
                            }
                        }
                        continue;
                    }
                    read_result = tokio::time::timeout(Duration::from_secs(30), read.next()) => {
                        let maybe = match read_result {
                            Ok(m) => m,
                            Err(_) => {
                                paradex_audit_reconnect("read_timeout");
                                eprintln!(
                                    "WARN: Paradex public WS read timeout (30s) — \
                                     no frame received, reconnecting"
                                );
                                anyhow::bail!("Paradex public WS read timeout after 30s");
                            }
                        };
                        let Some(msg) = maybe else { break; };
                        msg?
                    }
                };
                self.freshness
                    .last_ws_rx_ns
                    .store(mono_now_ns(), Ordering::Relaxed);
                let payload = match msg {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => match String::from_utf8(bytes) {
                        Ok(text) => text,
                        Err(_) => {
                            if !logged_non_utf8_binary {
                                eprintln!(
                                    "WARN: Paradex public WS non-utf8 binary frame url={}",
                                    self.cfg.ws_url
                                );
                                logged_non_utf8_binary = true;
                            }
                            continue;
                        }
                    },
                    Message::Ping(payload) => {
                        let _ = write.send(Message::Pong(payload)).await;
                        continue;
                    }
                    _ => continue,
                };
                self.freshness
                    .last_data_rx_ns
                    .store(mono_now_ns(), Ordering::Relaxed);
                if !first_message_logged {
                    eprintln!("INFO: Paradex public WS first message received");
                    first_message_logged = true;
                }
                if let Some(recorder) = self.recorder.as_ref() {
                    let mut guard = recorder.lock().await;
                    let _ = guard.record_ws_frame(&payload);
                }
                let value = match serde_json::from_str::<Value>(&payload) {
                    Ok(value) => value,
                    Err(err) => {
                        let snippet: String = payload.chars().take(160).collect();
                        eprintln!(
                            "WARN: Paradex public WS parse error: {err} url={} snippet={}",
                            self.cfg.ws_url, snippet
                        );
                        continue;
                    }
                };
                if !first_message_keys_logged {
                    let keys = value
                        .as_object()
                        .map(|obj| {
                            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                            keys.sort();
                            format!("[{}]", keys.join(","))
                        })
                        .unwrap_or_else(|| "[non-object]".to_string());
                    let snippet: String = payload.chars().take(160).collect();
                    eprintln!("INFO: Paradex WS first msg keys={keys} snippet={snippet}");
                    first_message_keys_logged = true;
                }
                let mut interactive_public_top_published = false;
                if let ParadexPublicFeedMode::Orderbook { feed_type, .. } = &public_feed_mode {
                    if let Some(interactive_top) = decode_interactive_top_value(&value) {
                        paradex_emit_interactive_top_audit(feed_type.as_str(), interactive_top);
                    }
                    if interactive_public_top_mode {
                        if let Some((top, snapshot, top_source)) = decode_interactive_public_top_and_snapshot(
                            &value,
                            self.cfg.venue_index,
                            &self.cfg.market,
                            &mut bbo_seq,
                            self.ui_touch_reference_state.as_deref(),
                        ) {
                            if !first_decoded_top_logged {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=paradex bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                            if !first_book_update_logged {
                                eprintln!("INFO: Paradex public WS first book update");
                                first_book_update_logged = true;
                            }
                            last_bbo_top = Some(ParadexBboTopState {
                                bid_px: top.best_bid_px,
                                bid_sz: top.best_bid_sz,
                                ask_px: top.best_ask_px,
                                ask_sz: top.best_ask_sz,
                            });
                            {
                                let now_ns = mono_now_ns();
                                self.freshness
                                    .last_parsed_ns
                                    .store(now_ns, Ordering::Relaxed);
                                self.freshness
                                    .last_book_event_ns
                                    .store(now_ns, Ordering::Relaxed);
                            }
                            paradex_emit_interactive_public_top_audit(
                                top_source,
                                top.best_bid_px,
                                top.best_bid_sz,
                                top.best_ask_px,
                                top.best_ask_sz,
                            );
                            if let Err(err) = self.publish_market(snapshot).await {
                                eprintln!("Paradex interactive public top market send failed: {err}");
                            }
                            interactive_public_top_published = true;
                        }
                    }
                }
                if !subscribed {
                    if paradex_subscribe_error(&value) {
                        paradex_audit_reconnect("subscribe_error");
                        if let Some(err) = value.get("error") {
                            eprintln!("WARN: Paradex subscribe error: {err}");
                        }
                        anyhow::bail!("Paradex subscribe failed: invalid channel");
                    }
                    if value.get("id").and_then(|v| v.as_i64()) == Some(1)
                        && value.get("result").is_some()
                    {
                        subscribed = true;
                    }
                }
                if let Some((top, snapshot)) = decode_bbo_top_and_snapshot(
                    &value,
                    self.cfg.venue_index,
                    &self.cfg.market,
                    &mut bbo_seq,
                    self.ui_touch_reference_state.as_deref(),
                ) {
                    if !first_decoded_top_logged {
                        eprintln!(
                            "FIRST_DECODED_TOP venue=paradex bid_px={} bid_sz={} ask_px={} ask_sz={}",
                            top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                        );
                        first_decoded_top_logged = true;
                    }
                    if !first_book_update_logged {
                        eprintln!("INFO: Paradex public WS first book update");
                        first_book_update_logged = true;
                    }
                    last_bbo_top = Some(ParadexBboTopState {
                        bid_px: top.best_bid_px,
                        bid_sz: top.best_bid_sz,
                        ask_px: top.best_ask_px,
                        ask_sz: top.best_ask_sz,
                    });
                    {
                        let now_ns = mono_now_ns();
                        self.freshness
                            .last_parsed_ns
                            .store(now_ns, Ordering::Relaxed);
                        self.freshness
                            .last_book_event_ns
                            .store(now_ns, Ordering::Relaxed);
                    }
                    if let Err(err) = self.publish_market(snapshot).await {
                        eprintln!("Paradex public WS market send failed: {err}");
                    }
                }
                if subscribed && !is_orderbook_mode {
                    if !has_paradex_book_fields(&value) {
                        continue;
                    }
                    if let Some(top) = decode_top_of_book_value(&value) {
                        if !first_decoded_top_logged {
                            eprintln!(
                                "FIRST_DECODED_TOP venue=paradex bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                            );
                            first_decoded_top_logged = true;
                        }
                    } else if decode_miss_count < 3 {
                        decode_miss_count += 1;
                        log_decode_miss(
                            "Paradex",
                            &value,
                            &payload,
                            decode_miss_count,
                            self.cfg.ws_url.as_str(),
                        );
                    }
                }
                let event = match parse_orderbook_message_value(&value, &mut tracker) {
                    Ok(event) => event,
                    Err(err) => {
                        let msg = err.to_string();
                        if msg.contains("seq gap") {
                            paradex_audit_reconnect("seq_gap");
                        } else if msg.contains("seq mismatch") {
                            paradex_audit_reconnect("seq_mismatch");
                        } else {
                            paradex_audit_reconnect("parse_error");
                        }
                        return Err(err);
                    }
                };
                if let Some(event) = event {
                    if !first_book_update_logged {
                        eprintln!("INFO: Paradex public WS first book update");
                        first_book_update_logged = true;
                    }
                    self.freshness
                        .last_parsed_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if interactive_public_top_published {
                        continue;
                    }
                    let _ = self.publish_market(event).await;
                }
            }
            Ok(())
        }
        .await;
        if let Some((stop_tx, handle)) = ui_truth_task {
            let _ = stop_tx.send(());
            let mut handle = handle;
            if tokio::time::timeout(Duration::from_secs(1), &mut handle)
                .await
                .is_err()
            {
                handle.abort();
            }
        }
        result
    }
}

async fn run_paradex_ui_book_truth_poller(
    client: ParadexRestClient,
    mut stop_rx: tokio::sync::oneshot::Receiver<()>,
    ui_touch_reference_state: Option<Arc<ParadexUiTouchReferenceState>>,
) {
    let poll_ms = paradex_ui_book_truth_poll_ms();
    let stale_ms = paradex_ui_book_truth_stale_ms();
    let mut interval = tokio::time::interval(Duration::from_millis(poll_ms));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await;
    loop {
        tokio::select! {
            _ = &mut stop_rx => break,
            _ = interval.tick() => {
                for source in [ParadexUiBookTruthSource::Api, ParadexUiBookTruthSource::Interactive] {
                    match fetch_paradex_ui_book_truth_snapshot(&client, source).await {
                        Ok(snapshot) => {
                            if let Some(state) = ui_touch_reference_state.as_ref() {
                                state.update(source, snapshot);
                            }
                            let status = match snapshot.last_updated_at_ms {
                                Some(last_updated_at_ms)
                                    if now_ms().saturating_sub(last_updated_at_ms) > stale_ms as i64 =>
                                {
                                    "stale"
                                }
                                _ => "ok",
                            };
                            paradex_emit_ui_book_truth_audit(
                                source,
                                status,
                                client.cfg.token_usage,
                                Some(&snapshot),
                                None,
                            );
                        }
                        Err(err) => {
                            let status = if err.message.contains("parse error") {
                                "parse_error"
                            } else {
                                "http_error"
                            };
                            paradex_emit_ui_book_truth_audit(
                                source,
                                status,
                                client.cfg.token_usage,
                                None,
                                Some(err.reason_label()),
                            );
                        }
                    }
                }
            }
        }
    }
}

async fn fetch_paradex_ui_book_truth_snapshot(
    client: &ParadexRestClient,
    source: ParadexUiBookTruthSource,
) -> LiveResult<ParadexUiBookTruthSnapshot> {
    let path = source.path(&client.cfg.market);
    let resp = client.send_authed_request(Method::GET, &path, None).await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(map_rest_error(status.as_u16(), &body));
    }
    let value: Value = serde_json::from_str(&body).map_err(|err| {
        LiveGatewayError::fatal(format!("paradex ui book truth parse error: {err}"))
    })?;
    decode_ui_book_truth_snapshot(&value)
        .ok_or_else(|| LiveGatewayError::fatal("paradex ui book truth missing required fields"))
}

fn has_paradex_book_fields(value: &Value) -> bool {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    payload.get("bids").is_some()
        || payload.get("asks").is_some()
        || payload.get("bid").is_some()
        || payload.get("ask").is_some()
}

#[derive(Debug, Clone)]
pub struct ParadexRestClient {
    cfg: ParadexConfig,
    http: Client,
    token_cache: Arc<Mutex<Option<CachedParadexToken>>>,
    poll_seq: Arc<AtomicU64>,
    replace_identity_cache: Arc<StdMutex<BTreeMap<String, String>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ParadexAuthToken {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    token: String,
    #[serde(default)]
    jwt: String,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    expires_at: Option<u64>,
}

#[derive(Debug, Clone)]
struct CachedParadexToken {
    token: ParadexAuthToken,
    fetched_unix_s: u64,
}

impl ParadexRestClient {
    pub fn new(cfg: ParadexConfig) -> Self {
        Self {
            cfg,
            http: Client::builder()
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("paradex rest http client build"),
            token_cache: Arc::new(Mutex::new(None)),
            poll_seq: Arc::new(AtomicU64::new(1)),
            replace_identity_cache: Arc::new(StdMutex::new(BTreeMap::new())),
        }
    }

    pub fn has_auth(&self) -> bool {
        self.cfg.has_auth()
    }

    async fn ensure_token(&self) -> LiveResult<String> {
        if let Some(jwt) = self.cfg.jwt.as_ref() {
            paradex_emit_profile_usage_audit("static_token", self.cfg.token_usage, "jwt_env");
            return Ok(jwt.clone());
        }
        let mut guard = self.token_cache.lock().await;
        if let Some(cached) = guard.as_ref() {
            if !cached_token_stale(cached, self.cfg.token_refresh_secs) {
                if let Some(jwt) = token_token(&cached.token) {
                    paradex_emit_profile_usage_audit("reuse", self.cfg.token_usage, "cache");
                    return Ok(jwt);
                }
            }
        }
        let token = if self.cfg.jwt_cmd.is_some() {
            self.fetch_token_from_cmd().await?
        } else {
            self.fetch_token_from_payload().await?
        };
        let jwt = token_token(&token)
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth token missing access_token"))?;
        *guard = Some(CachedParadexToken {
            token,
            fetched_unix_s: unix_now_secs(),
        });
        let auth_source = if self.cfg.jwt_cmd.is_some() {
            "jwt_cmd"
        } else {
            "auth_payload"
        };
        paradex_emit_profile_usage_audit("fetched", self.cfg.token_usage, auth_source);
        Ok(jwt)
    }

    async fn fetch_token_from_cmd(&self) -> LiveResult<ParadexAuthToken> {
        let cmd = self
            .cfg
            .jwt_cmd
            .as_ref()
            .ok_or_else(|| LiveGatewayError::fatal("paradex jwt command missing"))?;
        let output = StdCommand::new("/bin/bash")
            .arg("-lc")
            .arg(cmd)
            .env_remove("PARADEX_JWT")
            .env("PARADEX_TOKEN_USAGE", self.cfg.token_usage.as_str())
            .output()
            .map_err(|err| {
                LiveGatewayError::retryable(format!("paradex jwt command error: {err}"))
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let detail = if stderr.is_empty() {
                format!("exit_status={}", output.status)
            } else {
                format!("exit_status={} stderr={stderr}", output.status)
            };
            return Err(LiveGatewayError::fatal(format!(
                "paradex jwt command failed: {detail}"
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_token_output(stdout.trim())
            .ok_or_else(|| LiveGatewayError::fatal("paradex jwt command returned no token"))
    }

    async fn fetch_token_from_payload(&self) -> LiveResult<ParadexAuthToken> {
        let payload = self
            .cfg
            .auth_payload_json
            .clone()
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth payload missing"))?;
        let resp = self
            .http
            .request(Method::POST, self.cfg.auth_url_with_token_usage())
            .json(&payload)
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("auth_error: {err}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        parse_auth_token(&body)
            .ok_or_else(|| LiveGatewayError::fatal("paradex auth token parse error"))
    }

    async fn invalidate_cached_token(&self) {
        let mut guard = self.token_cache.lock().await;
        *guard = None;
    }

    async fn send_authed_request_once(
        &self,
        method: Method,
        path: &str,
        payload: Option<Value>,
    ) -> LiveResult<reqwest::Response> {
        let token = self.ensure_token().await?;
        let url = format!("{}{}", self.cfg.rest_url, path);
        let mut builder = self
            .http
            .request(method, url)
            .header("Authorization", format!("Bearer {token}"));
        if let Some(payload) = payload {
            builder = builder.json(&payload);
        }
        builder
            .send()
            .await
            .map_err(|err| LiveGatewayError::retryable(format!("rest_error: {err}")))
    }

    async fn send_authed_request(
        &self,
        method: Method,
        path: &str,
        payload: Option<Value>,
    ) -> LiveResult<reqwest::Response> {
        let response = self
            .send_authed_request_once(method.clone(), path, payload.clone())
            .await?;
        if response.status().as_u16() == 401 && self.cfg.has_refreshable_auth() {
            self.invalidate_cached_token().await;
            return self.send_authed_request_once(method, path, payload).await;
        }
        Ok(response)
    }

    async fn sign_order_payload(&self, payload: &Value) -> LiveResult<Value> {
        let cmd = self
            .cfg
            .sign_order_cmd
            .as_ref()
            .ok_or_else(|| LiveGatewayError::fatal("paradex sign order command missing"))?;
        let output = StdCommand::new("/bin/bash")
            .arg("-lc")
            .arg(cmd)
            .env("PARADEX_ORDER_PAYLOAD", payload.to_string())
            .output()
            .map_err(|err| {
                LiveGatewayError::retryable(format!("paradex sign order command error: {err}"))
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let detail = if stderr.is_empty() {
                format!("exit_status={}", output.status)
            } else {
                format!("exit_status={} stderr={stderr}", output.status)
            };
            return Err(LiveGatewayError::fatal(format!(
                "paradex sign order command failed: {detail}"
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str::<Value>(stdout.trim()).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex sign order payload parse error: {err}"))
        })
    }

    pub async fn fetch_account_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<AccountSnapshot> {
        let resp = self
            .send_authed_request(Method::GET, &self.cfg.account_path, None)
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex account parse error: {err}"))
        })?;
        let mut snapshot =
            parse_account_snapshot(&value, venue_id, venue_index).ok_or_else(|| {
                LiveGatewayError::fatal("paradex account snapshot missing required fields")
            })?;
        if snapshot.positions.is_empty() {
            if let Ok(positions) = self.fetch_position_snapshots().await {
                snapshot.positions = positions;
            }
        }
        Ok(snapshot)
    }

    async fn fetch_open_order_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<OrderSnapshot> {
        let resp = self
            .send_authed_request(Method::GET, &self.cfg.order_path, None)
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex open orders parse error: {err}"))
        })?;
        let mut open_orders = parse_open_orders(&value, &self.cfg.market);
        self.prime_replace_identity_cache_from_open_orders(&open_orders);
        self.normalize_open_snapshot_replace_identities(&mut open_orders)
            .await;
        Ok(OrderSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: self.poll_seq.fetch_add(1, Ordering::Relaxed),
            timestamp_ms: now_ms(),
            open_orders,
        })
    }

    pub async fn run_account_polling(
        self: Arc<Self>,
        account_tx: mpsc::Sender<AccountEvent>,
        venue_id: String,
        venue_index: usize,
        poll_ms: u64,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(250)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut rate_limit_backoff_ms: u64 = 0;
        loop {
            interval.tick().await;
            if rate_limit_backoff_ms > 0 {
                tokio::time::sleep(Duration::from_millis(rate_limit_backoff_ms)).await;
            }
            match self.fetch_account_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    rate_limit_backoff_ms = 0;
                    let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;
                }
                Err(err) => {
                    if paradex_is_rate_limit_error(&err.message) {
                        rate_limit_backoff_ms =
                            paradex_next_rate_limit_backoff_ms(rate_limit_backoff_ms, poll_ms);
                    } else {
                        rate_limit_backoff_ms = 0;
                    }
                    eprintln!("Paradex account snapshot error: {}", err.message);
                }
            }
        }
    }

    pub async fn run_order_polling(
        self: Arc<Self>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
        venue_id: String,
        venue_index: usize,
        poll_ms: u64,
    ) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(500)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut rate_limit_backoff_ms: u64 = 0;
        loop {
            interval.tick().await;
            if rate_limit_backoff_ms > 0 {
                tokio::time::sleep(Duration::from_millis(rate_limit_backoff_ms)).await;
            }
            match self.fetch_open_order_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    rate_limit_backoff_ms = 0;
                    let _ = exec_tx.send(ExecutionEvent::OrderSnapshot(snapshot)).await;
                }
                Err(err) => {
                    if paradex_is_rate_limit_error(&err.message) {
                        rate_limit_backoff_ms =
                            paradex_next_rate_limit_backoff_ms(rate_limit_backoff_ms, poll_ms);
                    } else {
                        rate_limit_backoff_ms = 0;
                    }
                    eprintln!("Paradex open order snapshot error: {}", err.message);
                }
            }
        }
    }

    pub async fn run_private_order_ws(
        self: Arc<Self>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
        venue_id: String,
        venue_index: usize,
    ) {
        let mut backoff = Duration::from_secs(1);
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();
            if let Err(err) = self
                .private_order_ws_once(exec_tx.clone(), &venue_id, venue_index)
                .await
            {
                eprintln!("Paradex private order WS error: {err}");
            }

            if session_start.elapsed() >= healthy_threshold {
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_order_ws_once(
        &self,
        exec_tx: mpsc::Sender<ExecutionEvent>,
        venue_id: &str,
        venue_index: usize,
    ) -> anyhow::Result<()> {
        let mut order_state =
            ParadexPrivateOrderState::new(&self.cfg.market, venue_id, venue_index);
        match self.fetch_open_order_snapshot(venue_id, venue_index).await {
            Ok(snapshot) => {
                order_state =
                    ParadexPrivateOrderState::from_snapshot(&self.cfg.market, snapshot.clone());
                exec_tx
                    .send(ExecutionEvent::OrderSnapshot(snapshot))
                    .await
                    .map_err(|_| anyhow::anyhow!("paradex exec_tx closed during bootstrap"))?;
            }
            Err(err) => {
                eprintln!(
                    "WARN: Paradex bootstrap order snapshot skipped: {}",
                    err.message
                );
            }
        }

        let read_timeout = Duration::from_millis(paradex_private_ws_read_timeout_ms());
        eprintln!(
            "INFO: Paradex private order WS connecting url={}",
            self.cfg.ws_url
        );
        let (ws_stream, _) = tokio::time::timeout(
            Duration::from_secs(15),
            connect_async(self.cfg.ws_url.as_str()),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Paradex private order WS connect timeout (15s)"))?
        .map_err(|err| anyhow::anyhow!("Paradex private order WS connect error: {err}"))?;
        eprintln!(
            "INFO: Paradex private order WS connected url={}",
            self.cfg.ws_url
        );

        let (mut write, mut read) = ws_stream.split();
        let bearer = self.ensure_token().await.map_err(|err| {
            anyhow::anyhow!("Paradex private order WS auth token error: {}", err.message)
        })?;
        let auth = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "auth",
            "params": { "bearer": bearer },
        });
        write
            .send(Message::Text(auth.to_string().into()))
            .await
            .map_err(|err| anyhow::anyhow!("Paradex private order WS auth send error: {err}"))?;
        let order_channel = format!("orders.{}", self.cfg.market);
        let order_subscribe = ParadexSubscribeCandidate::new(
            "subscribe",
            serde_json::json!({ "channel": order_channel.clone() }),
        );
        let fill_channel = format!("fills.{}", self.cfg.market);
        let fill_subscribe = ParadexSubscribeCandidate::new(
            "subscribe",
            serde_json::json!({ "channel": fill_channel.clone() }),
        );
        let mut authenticated = false;
        let mut orders_subscribed = false;
        let mut fills_subscribed = false;
        let mut seq_state = ParadexPrivateSeqState::default();

        loop {
            let maybe = tokio::time::timeout(read_timeout, read.next())
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Paradex private order WS read timeout after {}ms",
                        read_timeout.as_millis()
                    )
                })?;
            let Some(message) = maybe else {
                break;
            };
            let payload = match message? {
                Message::Text(text) => text.to_string(),
                Message::Binary(bytes) => String::from_utf8(bytes.to_vec()).map_err(|_| {
                    anyhow::anyhow!("Paradex private order WS non-utf8 binary frame")
                })?,
                Message::Ping(payload) => {
                    write.send(Message::Pong(payload)).await?;
                    continue;
                }
                Message::Close(_) => break,
                _ => continue,
            };
            let value: Value = serde_json::from_str(&payload)
                .map_err(|err| anyhow::anyhow!("Paradex private order WS parse error: {err}"))?;
            if let Some(error_message) = paradex_private_ws_error_message(&value) {
                if value.get("id").and_then(|raw| raw.as_i64()) == Some(2) {
                    eprintln!(
                        "WARN: Paradex private fills WS subscribe failed; Phase 5.1 source-owner capture disabled for this session"
                    );
                    fills_subscribed = false;
                    continue;
                }
                if paradex_private_ws_error_requires_token_refresh(&value) {
                    self.invalidate_cached_token().await;
                }
                anyhow::bail!("Paradex private order WS error: {error_message}");
            }
            if !authenticated {
                if value.get("id").and_then(|raw| raw.as_i64()) == Some(0)
                    && value.get("result").is_some()
                {
                    authenticated = true;
                    send_paradex_subscribe_with_id(&mut write, &order_subscribe, 1).await?;
                    send_paradex_subscribe_with_id(&mut write, &fill_subscribe, 2).await?;
                }
                continue;
            }
            if value.get("result").is_some() {
                match value.get("id").and_then(|raw| raw.as_i64()) {
                    Some(1) => orders_subscribed = true,
                    Some(2) => fills_subscribed = true,
                    _ => {}
                }
                continue;
            }
            if fills_subscribed {
                for source_owner_fill in
                    phase51_paradex_source_owner_fills_from_subscription_message(
                        &value,
                        order_state.snapshot.venue_index,
                        &order_state.snapshot.venue_id,
                        &self.cfg.market,
                    )
                {
                    exec_tx
                        .send(ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
                            source_owner_fill,
                        ))
                        .await
                        .map_err(|_| anyhow::anyhow!("paradex exec_tx closed"))?;
                }
            }
            if orders_subscribed {
                if let Some(snapshot) =
                    order_state.apply_subscription_message(&value, &mut seq_state, self)
                {
                    exec_tx
                        .send(ExecutionEvent::OrderSnapshot(snapshot))
                        .await
                        .map_err(|_| anyhow::anyhow!("paradex exec_tx closed"))?;
                }
            }
        }

        Ok(())
    }
}

impl LiveRestClient for ParadexRestClient {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let unsigned_payload = build_order_payload(&self.cfg.market, &req)?;
            paradex_emit_order_flags_audit("place", self.cfg.token_usage, &unsigned_payload);
            let payload = if self.cfg.sign_order_cmd.is_some() {
                self.sign_order_payload(&unsigned_payload).await?
            } else {
                unsigned_payload
            };
            let resp = self
                .send_authed_request(Method::POST, &self.cfg.order_path, Some(payload))
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
            let fill_flags = parse_fill_flags_from_body(&body);
            paradex_emit_fill_flags_audit(self.cfg.token_usage, &fill_flags);
            let order_id = parse_order_id(&body).or(Some(req.client_order_id.clone()));
            let client_order_id = parse_client_order_id(&body).or(Some(req.client_order_id));
            self.cache_replace_identity(client_order_id.as_deref(), order_id.as_deref());
            Ok(LiveRestResponse {
                client_order_id,
                order_id,
            })
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let path = if is_client_order_id(&req.order_id) {
                format!(
                    "{}/by_client_id/{}?market={}",
                    self.cfg.order_path, req.order_id, self.cfg.market
                )
            } else {
                format!("{}/{}", self.cfg.order_path, req.order_id)
            };
            let resp = self
                .send_authed_request(Method::DELETE, &path, None)
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                if is_paradex_cancel_not_found(&body) {
                    return Ok(LiveRestResponse {
                        order_id: Some(req.order_id),
                        client_order_id: None,
                    });
                }
                return Err(map_rest_error(status.as_u16(), &body));
            }
            Ok(LiveRestResponse {
                order_id: None,
                client_order_id: None,
            })
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let path = format!("{}/cancel_all", self.cfg.order_path);
            let payload = serde_json::json!({ "market": self.cfg.market });
            let resp = self
                .send_authed_request(Method::POST, &path, Some(payload))
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 404 {
                return self.cancel_all_via_batch().await;
            }
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
            Ok(LiveRestResponse {
                order_id: None,
                client_order_id: None,
            })
        })
    }

    fn cancel_batch(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.len() <= 1 {
                let mut results = Vec::with_capacity(reqs.len());
                for req in reqs {
                    results.push(self.cancel_order(req).await);
                }
                return results;
            }

            let batch_path = format!("{}/batch", self.cfg.order_path);
            let mut all_results = Vec::with_capacity(reqs.len());

            for chunk in reqs.chunks(PARADEX_BATCH_MAX_ITEMS) {
                let mut canonical = Vec::with_capacity(chunk.len());
                for (index, req) in chunk.iter().enumerate() {
                    canonical.push(self.canonicalize_cancel_request(index, req).await);
                }
                let mut batch_entries = Vec::new();
                let mut serial_entries = Vec::new();
                for entry in canonical {
                    if entry.batch_order_id.is_some() {
                        batch_entries.push(entry);
                    } else {
                        serial_entries.push(entry);
                    }
                }
                let resolved_client_count = batch_entries
                    .iter()
                    .filter(|entry| entry.resolved_from_cache)
                    .count();
                paradex_emit_cancel_batch_canonicalize_audit(
                    chunk.len(),
                    batch_entries.len(),
                    resolved_client_count,
                    serial_entries
                        .iter()
                        .filter(|entry| is_client_order_id(&entry.original_order_id))
                        .count(),
                );

                let mut chunk_results: Vec<Option<LiveResult<LiveRestResponse>>> =
                    vec![None; chunk.len()];

                if batch_entries.len() < 2 {
                    serial_entries.extend(batch_entries.into_iter());
                } else {
                    let batch_request_count = batch_entries.len();
                    let order_ids = batch_entries
                        .iter()
                        .filter_map(|entry| entry.batch_order_id.clone())
                        .collect::<Vec<_>>();
                    let payload = serde_json::json!({ "order_ids": order_ids });
                    let response = self
                        .send_authed_request(Method::DELETE, &batch_path, Some(payload))
                        .await;
                    let response = match response {
                        Ok(response) => response,
                        Err(err) => {
                            let mut status_counts = BTreeMap::new();
                            status_counts
                                .insert(format!("request_error:{}", err.reason_label()), 1);
                            paradex_emit_cancel_batch_audit(
                                batch_request_count,
                                batch_request_count,
                                0,
                                0,
                                false,
                                &status_counts,
                            );
                            for entry in batch_entries {
                                chunk_results[entry.original_index] = Some(Err(err.clone()));
                            }
                            for entry in serial_entries {
                                chunk_results[entry.original_index] =
                                    Some(self.cancel_order(entry.canonical_request.clone()).await);
                            }
                            all_results.extend(chunk_results.into_iter().map(|result| {
                                result.unwrap_or_else(|| {
                                    Err(LiveGatewayError::fatal(
                                        "paradex cancel batch request result missing",
                                    ))
                                })
                            }));
                            continue;
                        }
                    };
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();

                    if !status.is_success() {
                        let mut status_counts = BTreeMap::new();
                        status_counts.insert(format!("http_{}", status.as_u16()), 1);
                        let serial_batch_results = self
                            .cancel_requests_serially(
                                &batch_entries
                                    .iter()
                                    .map(|entry| entry.canonical_request.clone())
                                    .collect::<Vec<_>>(),
                            )
                            .await;
                        let success_count = serial_batch_results
                            .iter()
                            .filter(|result| result.is_ok())
                            .count();
                        paradex_emit_cancel_batch_audit(
                            batch_request_count,
                            batch_request_count,
                            0,
                            success_count,
                            true,
                            &status_counts,
                        );
                        for (entry, result) in batch_entries
                            .into_iter()
                            .zip(serial_batch_results.into_iter())
                        {
                            chunk_results[entry.original_index] = Some(result);
                        }
                    } else {
                        let parsed_items = match parse_paradex_batch_cancel_items(&body) {
                            Ok(items) => items,
                            Err(err) => {
                                let mut status_counts = BTreeMap::new();
                                status_counts.insert("parse_error".to_string(), 1);
                                paradex_emit_cancel_batch_audit(
                                    batch_request_count,
                                    batch_request_count,
                                    0,
                                    0,
                                    false,
                                    &status_counts,
                                );
                                for entry in batch_entries {
                                    chunk_results[entry.original_index] = Some(Err(err.clone()));
                                }
                                for entry in serial_entries {
                                    chunk_results[entry.original_index] = Some(
                                        self.cancel_order(entry.canonical_request.clone()).await,
                                    );
                                }
                                all_results.extend(chunk_results.into_iter().map(|result| {
                                    result.unwrap_or_else(|| {
                                        Err(LiveGatewayError::fatal(
                                            "paradex cancel batch parse result missing",
                                        ))
                                    })
                                }));
                                continue;
                            }
                        };

                        let mut status_counts = BTreeMap::new();
                        let mut by_order_id = BTreeMap::new();
                        for item in parsed_items {
                            let label = paradex_batch_cancel_status_label(&item);
                            *status_counts.entry(label).or_insert(0) += 1;
                            if let Some(order_id) = item.order_id.clone() {
                                by_order_id.insert(order_id, item);
                            }
                        }

                        let mut success_count = 0usize;
                        for entry in batch_entries {
                            let batch_order_id = entry
                                .batch_order_id
                                .clone()
                                .expect("paradex batch entry missing canonical order id");
                            let result = match by_order_id.get(&batch_order_id).cloned() {
                                Some(item) if paradex_batch_cancel_is_success(&item) => {
                                    success_count += 1;
                                    Ok(LiveRestResponse {
                                        order_id: item.order_id.or(Some(batch_order_id)),
                                        client_order_id: is_client_order_id(&entry.original_order_id)
                                            .then(|| entry.original_order_id.clone()),
                                    })
                                }
                                Some(item) => Err(LiveGatewayError::retryable(format!(
                                    "paradex cancel batch unresolved status={} order_id_state={} detail_state={}",
                                    paradex_batch_cancel_status_label(&item),
                                    paradex_audit_id_state(Some(entry.original_order_id.as_str())),
                                    paradex_audit_id_state(item.detail.as_deref()),
                                ))),
                                None => {
                                    *status_counts.entry("missing_result".to_string()).or_insert(0) += 1;
                                    Err(LiveGatewayError::retryable(format!(
                                        "paradex cancel batch missing result order_id_state={}",
                                        paradex_audit_id_state(Some(entry.original_order_id.as_str()))
                                    )))
                                }
                            };
                            chunk_results[entry.original_index] = Some(result);
                        }

                        paradex_emit_cancel_batch_audit(
                            batch_request_count,
                            batch_request_count,
                            0,
                            success_count,
                            false,
                            &status_counts,
                        );
                    }
                }

                for entry in serial_entries {
                    chunk_results[entry.original_index] =
                        Some(self.cancel_order(entry.canonical_request.clone()).await);
                }
                all_results.extend(chunk_results.into_iter().map(|result| {
                    result.unwrap_or_else(|| {
                        Err(LiveGatewayError::fatal(
                            "paradex cancel batch chunk result missing",
                        ))
                    })
                }));
            }

            all_results
        })
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let (resolved_order_id, submit_source) = if is_client_order_id(&req.order_id) {
                (
                    self.resolve_exchange_order_id_for_replace(&req.order_id)
                        .await?,
                    "resolved_client_id",
                )
            } else {
                (req.order_id.clone(), "exchange_id")
            };
            paradex_emit_native_replace_audit(
                submit_source,
                &resolved_order_id,
                Some(req.client_order_id.as_str()),
            );
            let replace_req = LiveRestReplaceRequest {
                order_id: resolved_order_id.clone(),
                ..req.clone()
            };
            let mut unsigned_payload = build_modify_order_payload(&self.cfg.market, &replace_req)?;
            paradex_emit_order_flags_audit("replace", self.cfg.token_usage, &unsigned_payload);
            let payload = if self.cfg.sign_order_cmd.is_some() {
                let signed_base = build_order_payload(
                    &self.cfg.market,
                    &LiveRestPlaceRequest {
                        venue_index: replace_req.venue_index,
                        venue_id: replace_req.venue_id.clone(),
                        side: replace_req.side,
                        price: replace_req.price,
                        size: replace_req.size,
                        purpose: replace_req.purpose,
                        time_in_force: replace_req.time_in_force,
                        post_only: replace_req.post_only,
                        reduce_only: replace_req.reduce_only,
                        client_order_id: replace_req.client_order_id.clone(),
                    },
                )?;
                let mut signed_payload = self.sign_order_payload(&signed_base).await?;
                signed_payload["id"] = serde_json::json!(resolved_order_id);
                signed_payload
            } else {
                unsigned_payload.take()
            };
            let path = format!("{}/{}", self.cfg.order_path, resolved_order_id);
            let resp = self
                .send_authed_request(Method::PUT, &path, Some(payload))
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
            let fill_flags = parse_fill_flags_from_body(&body);
            paradex_emit_fill_flags_audit(self.cfg.token_usage, &fill_flags);
            let order_id = parse_order_id(&body).or(Some(resolved_order_id));
            let client_order_id = parse_client_order_id(&body).or(Some(req.client_order_id));
            self.cache_replace_identity(client_order_id.as_deref(), order_id.as_deref());
            Ok(LiveRestResponse {
                order_id,
                client_order_id,
            })
        })
    }
}

impl ParadexRestClient {
    async fn canonicalize_cancel_request(
        &self,
        original_index: usize,
        req: &LiveRestCancelRequest,
    ) -> ParadexCanonicalCancelRequest {
        if is_client_order_id(&req.order_id) {
            if let Some(order_id) = self
                .resolve_exchange_order_id_for_batch_cancel(&req.order_id)
                .await
            {
                return ParadexCanonicalCancelRequest {
                    original_index,
                    original_order_id: req.order_id.clone(),
                    canonical_request: LiveRestCancelRequest {
                        order_id: order_id.clone(),
                        ..req.clone()
                    },
                    batch_order_id: Some(order_id),
                    resolved_from_cache: true,
                };
            }
        }
        ParadexCanonicalCancelRequest {
            original_index,
            original_order_id: req.order_id.clone(),
            canonical_request: req.clone(),
            batch_order_id: (!is_client_order_id(&req.order_id)).then(|| req.order_id.clone()),
            resolved_from_cache: false,
        }
    }

    fn cache_replace_identity(&self, client_order_id: Option<&str>, order_id: Option<&str>) {
        let Some(client_order_id) = client_order_id.filter(|value| is_client_order_id(value))
        else {
            return;
        };
        let Some(order_id) = order_id.filter(|value| !is_client_order_id(value)) else {
            return;
        };
        let mut cache = self
            .replace_identity_cache
            .lock()
            .expect("paradex replace identity cache mutex poisoned");
        if cache.len() >= PARADEX_REPLACE_IDENTITY_CACHE_MAX_ITEMS
            && !cache.contains_key(client_order_id)
        {
            cache.clear();
        }
        cache.insert(client_order_id.to_string(), order_id.to_string());
    }

    fn cached_replace_identity(&self, client_order_id: &str) -> Option<String> {
        self.replace_identity_cache
            .lock()
            .expect("paradex replace identity cache mutex poisoned")
            .get(client_order_id)
            .cloned()
    }

    fn prime_replace_identity_cache_from_open_orders(&self, open_orders: &[OpenOrderSnapshot]) {
        for order in open_orders {
            self.cache_replace_identity(
                order.client_order_id.as_deref(),
                order.exchange_order_id.as_deref().or_else(|| {
                    (!is_client_order_id(&order.order_id)).then_some(order.order_id.as_str())
                }),
            );
        }
    }

    async fn resolve_exchange_order_id_for_open_snapshot(
        &self,
        client_order_id: &str,
    ) -> Option<String> {
        if let Some(order_id) = self.cached_replace_identity(client_order_id) {
            paradex_emit_open_identity_normalized_audit("cache", client_order_id, &order_id);
            return Some(order_id);
        }
        if std::env::var("PARAPHINA_TRADE_MODE")
            .map(|value| value.eq_ignore_ascii_case("shadow"))
            .unwrap_or(false)
        {
            paradex_emit_open_identity_unresolved_audit("shadow_mode", client_order_id);
            return None;
        }

        let path = format!("{}/by_client_id/{}", self.cfg.order_path, client_order_id);
        let response = match self.send_authed_request(Method::GET, &path, None).await {
            Ok(response) => response,
            Err(err) => {
                paradex_emit_open_identity_unresolved_audit(err.reason_label(), client_order_id);
                return None;
            }
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let reason = if status.as_u16() == 404 || paradex_order_lookup_not_found(&body) {
                "not_found"
            } else {
                map_rest_error(status.as_u16(), &body).reason_label()
            };
            paradex_emit_open_identity_unresolved_audit(reason, client_order_id);
            return None;
        }

        let resolved = match parse_paradex_replace_resolved_order(&body) {
            Ok(resolved) => resolved,
            Err(err) => {
                paradex_emit_open_identity_unresolved_audit(err.reason_label(), client_order_id);
                return None;
            }
        };
        if resolved.market.as_deref() != Some(self.cfg.market.as_str()) {
            paradex_emit_open_identity_unresolved_audit("wrong_market", client_order_id);
            return None;
        }
        if is_client_order_id(&resolved.order_id) {
            paradex_emit_open_identity_unresolved_audit("unresolved_identity", client_order_id);
            return None;
        }

        self.cache_replace_identity(
            resolved.client_order_id.as_deref(),
            Some(resolved.order_id.as_str()),
        );
        paradex_emit_open_identity_normalized_audit("rest", client_order_id, &resolved.order_id);
        Some(resolved.order_id)
    }

    async fn normalize_open_snapshot_replace_identities(
        &self,
        open_orders: &mut [OpenOrderSnapshot],
    ) {
        for order in open_orders {
            if order.exchange_order_id.is_some() {
                continue;
            }
            if !is_client_order_id(&order.order_id) {
                order.exchange_order_id = Some(order.order_id.clone());
                continue;
            }
            let client_order_id = order
                .client_order_id
                .as_deref()
                .filter(|value| is_client_order_id(value))
                .or_else(|| is_client_order_id(&order.order_id).then_some(order.order_id.as_str()));
            let Some(client_order_id) = client_order_id else {
                continue;
            };
            order.exchange_order_id = self
                .resolve_exchange_order_id_for_open_snapshot(client_order_id)
                .await;
        }
    }

    async fn resolve_exchange_order_id_for_batch_cancel(
        &self,
        client_order_id: &str,
    ) -> Option<String> {
        if let Some(order_id) = self.cached_replace_identity(client_order_id) {
            paradex_emit_cancel_batch_resolve_audit("cache", client_order_id, &order_id);
            return Some(order_id);
        }

        let path = format!("{}/by_client_id/{}", self.cfg.order_path, client_order_id);
        let response = match self.send_authed_request(Method::GET, &path, None).await {
            Ok(response) => response,
            Err(err) => {
                paradex_emit_cancel_batch_resolve_failed_audit(err.reason_label(), client_order_id);
                return None;
            }
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let reason = if status.as_u16() == 404 || paradex_order_lookup_not_found(&body) {
                "not_found"
            } else {
                map_rest_error(status.as_u16(), &body).reason_label()
            };
            paradex_emit_cancel_batch_resolve_failed_audit(reason, client_order_id);
            return None;
        }

        let resolved = match parse_paradex_replace_resolved_order(&body) {
            Ok(resolved) => resolved,
            Err(err) => {
                paradex_emit_cancel_batch_resolve_failed_audit(err.reason_label(), client_order_id);
                return None;
            }
        };
        if resolved.market.as_deref() != Some(self.cfg.market.as_str()) {
            paradex_emit_cancel_batch_resolve_failed_audit("wrong_market", client_order_id);
            return None;
        }
        if is_client_order_id(&resolved.order_id) {
            paradex_emit_cancel_batch_resolve_failed_audit("unresolved_identity", client_order_id);
            return None;
        }

        self.cache_replace_identity(
            resolved.client_order_id.as_deref(),
            Some(resolved.order_id.as_str()),
        );
        paradex_emit_cancel_batch_resolve_audit("rest", client_order_id, &resolved.order_id);
        Some(resolved.order_id)
    }

    async fn resolve_exchange_order_id_for_replace(
        &self,
        client_order_id: &str,
    ) -> LiveResult<String> {
        if let Some(order_id) = self.cached_replace_identity(client_order_id) {
            paradex_emit_replace_identity_resolve_audit("cache", client_order_id, &order_id);
            return Ok(order_id);
        }

        let path = format!("{}/by_client_id/{}", self.cfg.order_path, client_order_id);
        let resp = self.send_authed_request(Method::GET, &path, None).await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            let err = map_rest_error(status.as_u16(), &body);
            let reason = if status.as_u16() == 404 || paradex_order_lookup_not_found(&body) {
                "not_found"
            } else {
                err.reason_label()
            };
            paradex_emit_replace_identity_resolve_failed_audit(reason, client_order_id);
            if reason == "not_found" {
                return Err(LiveGatewayError::retryable(format!(
                    "paradex replace identity unresolved client_id_state={} reason=not_found",
                    paradex_audit_id_state(Some(client_order_id))
                )));
            }
            return Err(err);
        }

        let resolved = parse_paradex_replace_resolved_order(&body)?;
        if resolved.market.as_deref() != Some(self.cfg.market.as_str()) {
            paradex_emit_replace_identity_resolve_failed_audit("wrong_market", client_order_id);
            return Err(LiveGatewayError::retryable(format!(
                "paradex replace identity unresolved client_id_state={} reason=wrong_market market={}",
                paradex_audit_id_state(Some(client_order_id)),
                resolved.market.unwrap_or_else(|| "unknown".to_string()),
            )));
        }
        if !paradex_status_allows_replace(resolved.status.as_deref()) {
            paradex_emit_replace_identity_resolve_failed_audit("closed", client_order_id);
            return Err(LiveGatewayError::retryable(format!(
                "paradex replace identity unresolved client_id_state={} reason=closed status={}",
                paradex_audit_id_state(Some(client_order_id)),
                paradex_sanitized_audit_status(resolved.status.as_deref().unwrap_or("unknown")),
            )));
        }
        if is_client_order_id(&resolved.order_id) {
            paradex_emit_replace_identity_resolve_failed_audit(
                "unresolved_identity",
                client_order_id,
            );
            return Err(LiveGatewayError::retryable(format!(
                "paradex replace identity unresolved client_id_state={} reason=unresolved_identity",
                paradex_audit_id_state(Some(client_order_id))
            )));
        }

        self.cache_replace_identity(
            resolved.client_order_id.as_deref(),
            Some(resolved.order_id.as_str()),
        );
        paradex_emit_replace_identity_resolve_audit("rest", client_order_id, &resolved.order_id);
        Ok(resolved.order_id)
    }

    async fn fetch_position_snapshots(&self) -> LiveResult<Vec<PositionSnapshot>> {
        let path = format!("/positions?market={}", self.cfg.market);
        let resp = self.send_authed_request(Method::GET, &path, None).await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex positions parse error: {err}"))
        })?;
        Ok(parse_position_snapshots(&value, &self.cfg.market))
    }

    async fn cancel_all_via_batch(&self) -> LiveResult<LiveRestResponse> {
        let resp = self
            .send_authed_request(Method::GET, &self.cfg.order_path, None)
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("paradex open orders parse error: {err}"))
        })?;
        let order_ids = parse_open_order_ids(&value, &self.cfg.market);
        for chunk in order_ids.chunks(PARADEX_BATCH_MAX_ITEMS) {
            let payload = serde_json::json!({ "order_ids": chunk });
            let batch_path = format!("{}/batch", self.cfg.order_path);
            let resp = self
                .send_authed_request(Method::DELETE, &batch_path, Some(payload))
                .await?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(map_rest_error(status.as_u16(), &body));
            }
        }
        Ok(LiveRestResponse {
            order_id: None,
            client_order_id: None,
        })
    }

    async fn cancel_requests_serially(
        &self,
        reqs: &[LiveRestCancelRequest],
    ) -> Vec<LiveResult<LiveRestResponse>> {
        let mut results = Vec::with_capacity(reqs.len());
        for req in reqs {
            results.push(self.cancel_order(req.clone()).await);
        }
        results
    }
}

#[derive(Debug, Clone)]
struct ParadexBatchCancelItem {
    order_id: Option<String>,
    client_order_id: Option<String>,
    status: Option<String>,
    success: Option<bool>,
    detail: Option<String>,
}

#[derive(Debug, Clone)]
struct ParadexCanonicalCancelRequest {
    original_index: usize,
    original_order_id: String,
    canonical_request: LiveRestCancelRequest,
    batch_order_id: Option<String>,
    resolved_from_cache: bool,
}

#[derive(Debug, Clone)]
struct ParadexReplaceResolvedOrder {
    order_id: String,
    client_order_id: Option<String>,
    market: Option<String>,
    status: Option<String>,
}

fn parse_stringish_value(value: Option<&Value>) -> Option<String> {
    let value = value?;
    value
        .as_str()
        .map(|raw| raw.to_string())
        .or_else(|| value.as_i64().map(|raw| raw.to_string()))
        .or_else(|| value.as_u64().map(|raw| raw.to_string()))
}

fn parse_paradex_batch_cancel_detail(item: &Value) -> Option<String> {
    if let Some(detail) = parse_stringish_value(item.get("message")) {
        return Some(detail);
    }
    if let Some(detail) = parse_stringish_value(item.get("reason")) {
        return Some(detail);
    }
    if let Some(error) = item.get("error") {
        if let Some(detail) = parse_stringish_value(Some(error)) {
            return Some(detail);
        }
        if let Some(detail) = parse_stringish_value(error.get("message")) {
            return Some(detail);
        }
        if let Some(detail) = parse_stringish_value(error.get("code")) {
            return Some(detail);
        }
        if let Some(detail) = parse_stringish_value(error.get("reason")) {
            return Some(detail);
        }
    }
    None
}

fn parse_paradex_batch_cancel_items(body: &str) -> LiveResult<Vec<ParadexBatchCancelItem>> {
    let value: Value = serde_json::from_str(body).map_err(|err| {
        LiveGatewayError::retryable(format!("paradex cancel batch parse error: {err}"))
    })?;
    let results = value
        .get("results")
        .and_then(|raw| raw.as_array())
        .or_else(|| value.get("orders").and_then(|raw| raw.as_array()))
        .or_else(|| value.as_array())
        .ok_or_else(|| {
            LiveGatewayError::retryable("paradex cancel batch missing results".to_string())
        })?;

    Ok(results
        .iter()
        .filter_map(|item| {
            item.as_object().map(|_| ParadexBatchCancelItem {
                order_id: parse_stringish_value(
                    item.get("id")
                        .or_else(|| item.get("order_id"))
                        .or_else(|| item.get("orderId")),
                ),
                client_order_id: parse_stringish_value(
                    item.get("client_id")
                        .or_else(|| item.get("client_order_id"))
                        .or_else(|| item.get("clientOrderId")),
                ),
                status: parse_stringish_value(item.get("status")),
                success: item.get("success").and_then(|raw| raw.as_bool()),
                detail: parse_paradex_batch_cancel_detail(item),
            })
        })
        .collect())
}

fn parse_paradex_replace_resolved_order(body: &str) -> LiveResult<ParadexReplaceResolvedOrder> {
    let value: Value = serde_json::from_str(body).map_err(|err| {
        LiveGatewayError::fatal(format!("paradex replace identity parse error: {err}"))
    })?;
    let order_id = parse_stringish_value(
        value
            .get("id")
            .or_else(|| value.get("order_id"))
            .or_else(|| value.get("orderId")),
    )
    .ok_or_else(|| LiveGatewayError::fatal("paradex replace identity missing order id"))?;
    Ok(ParadexReplaceResolvedOrder {
        order_id,
        client_order_id: parse_stringish_value(
            value
                .get("client_id")
                .or_else(|| value.get("client_order_id"))
                .or_else(|| value.get("clientOrderId")),
        ),
        market: parse_stringish_value(value.get("market").or_else(|| value.get("symbol"))),
        status: parse_stringish_value(value.get("status")),
    })
}

fn paradex_status_allows_replace(status: Option<&str>) -> bool {
    match status.map(|value| value.to_ascii_uppercase()) {
        None => true,
        Some(value) => value == "OPEN" || value == "NEW" || value == "PARTIALLY_FILLED",
    }
}

fn paradex_order_lookup_not_found(body: &str) -> bool {
    let upper = body.to_ascii_uppercase();
    upper.contains("CLIENT_ORDER_ID_NOT_FOUND")
        || upper.contains("ORDER_ID_NOT_FOUND")
        || upper.contains("NOT_FOUND")
}

fn paradex_batch_cancel_is_success(item: &ParadexBatchCancelItem) -> bool {
    if item.success == Some(true) {
        return true;
    }
    let mut normalized = item.status.clone().unwrap_or_default();
    if let Some(detail) = item.detail.as_ref() {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(detail);
    }
    let normalized = normalized.to_ascii_uppercase();
    normalized.contains("CANCELLED")
        || normalized.contains("CANCELED")
        || normalized.contains("QUEUED_FOR_CANCELLATION")
        || normalized.contains("ALREADY_CLOSED")
        || normalized.contains("NOT_FOUND")
}

fn paradex_batch_cancel_status_label(item: &ParadexBatchCancelItem) -> String {
    if let Some(status) = item.status.as_ref() {
        return status.to_ascii_uppercase();
    }
    if item.success == Some(true) {
        return "SUCCESS".to_string();
    }
    if let Some(detail) = item.detail.as_ref() {
        return detail.to_ascii_uppercase().replace(' ', "_");
    }
    "UNKNOWN".to_string()
}

fn paradex_emit_cancel_batch_audit(
    request_count: usize,
    order_ids_count: usize,
    client_order_ids_count: usize,
    success_count: usize,
    fallback_used: bool,
    status_counts: &BTreeMap<String, u64>,
) {
    let status_summary = if status_counts.is_empty() {
        "none".to_string()
    } else {
        status_counts
            .iter()
            .map(|(label, count)| format!("{label}:{count}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    eprintln!(
        "PARADEX_CANCEL_BATCH request_count={} order_ids={} client_order_ids={} success_count={} fallback_used={} statuses={}",
        request_count,
        order_ids_count,
        client_order_ids_count,
        success_count,
        if fallback_used { 1 } else { 0 },
        status_summary,
    );
}

fn paradex_emit_cancel_batch_canonicalize_audit(
    request_count: usize,
    batch_order_ids_count: usize,
    resolved_client_order_ids_count: usize,
    unresolved_client_order_ids_count: usize,
) {
    eprintln!(
        "PARADEX_CANCEL_BATCH_CANONICALIZE request_count={} batch_order_ids={} resolved_client_order_ids={} unresolved_client_order_ids={}",
        request_count,
        batch_order_ids_count,
        resolved_client_order_ids_count,
        unresolved_client_order_ids_count,
    );
}

fn paradex_emit_cancel_batch_resolve_audit(source: &str, client_order_id: &str, order_id: &str) {
    eprintln!(
        "PARADEX_CANCEL_BATCH_RESOLVE source={} client_id_state={} order_id_state={}",
        paradex_sanitized_audit_status(source),
        paradex_audit_id_state(Some(client_order_id)),
        paradex_audit_id_state(Some(order_id))
    );
}

fn paradex_emit_cancel_batch_resolve_failed_audit(reason: &str, client_order_id: &str) {
    eprintln!(
        "PARADEX_CANCEL_BATCH_RESOLVE_FAILED reason={} client_id_state={}",
        paradex_sanitized_audit_status(reason),
        paradex_audit_id_state(Some(client_order_id))
    );
}

#[derive(Debug, Default, Clone, Copy)]
struct ParadexPrivateSeqState {
    last_seq: Option<u64>,
}

impl ParadexPrivateSeqState {
    fn accept(&mut self, seq_no: u64) -> bool {
        if self
            .last_seq
            .map(|last_seq| seq_no <= last_seq)
            .unwrap_or(false)
        {
            return false;
        }
        self.last_seq = Some(seq_no);
        true
    }
}

#[derive(Debug, Clone)]
struct ParadexPrivateOrderState {
    market: String,
    snapshot: OrderSnapshot,
    open_orders: BTreeMap<String, OpenOrderSnapshot>,
}

impl ParadexPrivateOrderState {
    fn new(market: &str, venue_id: &str, venue_index: usize) -> Self {
        Self {
            market: market.to_string(),
            snapshot: OrderSnapshot {
                venue_index,
                venue_id: venue_id.to_string(),
                seq: 0,
                timestamp_ms: now_ms(),
                open_orders: Vec::new(),
            },
            open_orders: BTreeMap::new(),
        }
    }

    fn from_snapshot(market: &str, snapshot: OrderSnapshot) -> Self {
        let open_orders = snapshot
            .open_orders
            .iter()
            .cloned()
            .map(|order| (order.order_id.clone(), order))
            .collect();
        Self {
            market: market.to_string(),
            snapshot,
            open_orders,
        }
    }

    fn apply_subscription_message(
        &mut self,
        value: &Value,
        seq_state: &mut ParadexPrivateSeqState,
        client: &ParadexRestClient,
    ) -> Option<OrderSnapshot> {
        let payload = value.get("params")?;
        let channel = payload.get("channel").and_then(|raw| raw.as_str())?;
        if channel != format!("orders.{}", self.market) {
            return None;
        }
        let data = payload.get("data")?;
        let updates: Vec<&Value> = match data.as_array() {
            Some(items) => items.iter().collect(),
            None => vec![data],
        };
        let mut changed = false;
        let mut latest_seq = self.snapshot.seq;
        let mut latest_timestamp_ms = self.snapshot.timestamp_ms;
        for update in updates {
            let seq_no = update
                .get("seq_no")
                .and_then(parse_u64_value)
                .unwrap_or(latest_seq);
            if !seq_state.accept(seq_no) {
                continue;
            }
            latest_seq = seq_no;
            latest_timestamp_ms = update
                .get("timestamp")
                .or_else(|| update.get("last_updated_at"))
                .or_else(|| update.get("published_at"))
                .or_else(|| update.get("received_at"))
                .and_then(parse_i64_value)
                .unwrap_or_else(now_ms);
            if self.apply_order_update(update, latest_seq, latest_timestamp_ms, client) {
                changed = true;
            }
        }
        if !changed {
            return None;
        }
        self.snapshot.seq = latest_seq;
        self.snapshot.timestamp_ms = latest_timestamp_ms;
        self.snapshot.open_orders = self.open_orders.values().cloned().collect();
        Some(self.snapshot.clone())
    }

    fn apply_order_update(
        &mut self,
        update: &Value,
        seq_no: u64,
        timestamp_ms: TimestampMs,
        client: &ParadexRestClient,
    ) -> bool {
        let order_market =
            parse_stringish_value(update.get("market").or_else(|| update.get("symbol")));
        if order_market.as_deref() != Some(self.market.as_str()) {
            return false;
        }
        let Some(order_id) = parse_stringish_value(
            update
                .get("id")
                .or_else(|| update.get("order_id"))
                .or_else(|| update.get("orderId")),
        ) else {
            return false;
        };
        let client_order_id = parse_stringish_value(
            update
                .get("client_id")
                .or_else(|| update.get("client_order_id"))
                .or_else(|| update.get("clientOrderId")),
        );
        client.cache_replace_identity(client_order_id.as_deref(), Some(order_id.as_str()));
        let status =
            parse_stringish_value(update.get("status")).unwrap_or_else(|| "UNKNOWN".to_string());
        paradex_emit_private_order_truth_audit(
            &status,
            &order_id,
            client_order_id.as_deref(),
            seq_no,
        );
        if !paradex_status_allows_replace(Some(status.as_str())) {
            return self.open_orders.remove(&order_id).is_some();
        }
        let side = match update
            .get("side")
            .and_then(|raw| raw.as_str())
            .map(|raw| raw.to_ascii_uppercase())
            .as_deref()
        {
            Some("BUY") => Side::Buy,
            Some("SELL") => Side::Sell,
            _ => return false,
        };
        let existing = self.open_orders.get(&order_id);
        let price = update
            .get("price")
            .or_else(|| update.get("limit_price"))
            .and_then(parse_f64)
            .or_else(|| existing.map(|order| order.price))
            .unwrap_or(0.0);
        let remaining_size = update
            .get("remaining_size")
            .or_else(|| update.get("remainingQuantity"))
            .or_else(|| update.get("remaining_qty"))
            .and_then(parse_f64)
            .or_else(|| {
                let size = update.get("size").and_then(parse_f64)?;
                let filled = update
                    .get("filled")
                    .or_else(|| update.get("filled_size"))
                    .or_else(|| update.get("filledQuantity"))
                    .and_then(parse_f64)
                    .unwrap_or(0.0);
                Some((size - filled).max(0.0))
            })
            .or_else(|| existing.map(|order| order.size))
            .unwrap_or(0.0);
        if !(price.is_finite() && price > 0.0 && remaining_size.is_finite() && remaining_size > 0.0)
        {
            return self.open_orders.remove(&order_id).is_some();
        }
        let exchange_order_id = (!is_client_order_id(&order_id)).then_some(order_id.clone());
        self.open_orders.insert(
            order_id.clone(),
            OpenOrderSnapshot {
                order_id,
                client_order_id,
                exchange_order_id,
                side,
                price,
                size: remaining_size,
                purpose: None,
            },
        );
        let _ = timestamp_ms;
        true
    }
}

fn phase51_paradex_source_owner_fills_from_subscription_message(
    value: &Value,
    venue_index: usize,
    venue_id: &str,
    market: &str,
) -> Vec<Phase51ForwardRefreshSourceOwnerFill> {
    let Some(payload) = value.get("params") else {
        return Vec::new();
    };
    let Some(channel) = payload.get("channel").and_then(|raw| raw.as_str()) else {
        return Vec::new();
    };
    if channel != format!("fills.{market}") {
        return Vec::new();
    }
    let Some(data) = payload.get("data") else {
        return Vec::new();
    };
    let fills: Vec<&Value> = match data.as_array() {
        Some(items) => items.iter().collect(),
        None => vec![data],
    };
    fills
        .into_iter()
        .enumerate()
        .filter_map(|(fill_index, fill)| {
            let fill_market =
                parse_stringish_value(fill.get("market").or_else(|| fill.get("symbol")));
            if fill_market.as_deref() != Some(market) {
                return None;
            }
            let fill_type = parse_stringish_value(fill.get("fill_type"))?;
            if !fill_type.trim().eq_ignore_ascii_case("FILL") {
                return None;
            }
            let liquidity = parse_stringish_value(fill.get("liquidity"))?;
            let normalized_liquidity = liquidity.trim().to_ascii_uppercase();
            if normalized_liquidity != "MAKER" && normalized_liquidity != "TAKER" {
                return None;
            }
            let size = fill.get("size").and_then(parse_f64)?;
            if !size.is_finite() || size <= 0.0 {
                return None;
            }
            let timestamp_ms = fill.get("created_at").and_then(parse_i64_value)?;
            let order_id =
                parse_nonempty_stringish(fill.get("order_id").or_else(|| fill.get("orderId")));
            let client_order_id = parse_nonempty_stringish(
                fill.get("client_id")
                    .or_else(|| fill.get("client_order_id"))
                    .or_else(|| fill.get("clientOrderId")),
            );
            if order_id.is_none() && client_order_id.is_none() {
                return None;
            }

            Some(Phase51ForwardRefreshSourceOwnerFill::new(
                venue_index,
                venue_id,
                phase51_paradex_source_owner_seq(timestamp_ms, fill_index),
                timestamp_ms,
                order_id,
                client_order_id,
                Some(Phase51ForwardRefreshNativeRole::Paradex {
                    liquidity: normalized_liquidity,
                }),
            ))
        })
        .collect()
}

fn phase51_paradex_source_owner_seq(timestamp_ms: TimestampMs, fill_index: usize) -> u64 {
    (timestamp_ms.max(0) as u64)
        .saturating_mul(1_000_000)
        .saturating_add(fill_index as u64)
}

fn parse_nonempty_stringish(value: Option<&Value>) -> Option<String> {
    parse_stringish_value(value).and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn paradex_private_ws_error_message(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    parse_stringish_value(
        error
            .get("message")
            .or_else(|| error.get("data"))
            .or_else(|| Some(error)),
    )
}

fn paradex_private_ws_error_requires_token_refresh(value: &Value) -> bool {
    let error = value.get("error");
    let code = error
        .and_then(|raw| raw.get("code"))
        .and_then(parse_i64_value);
    matches!(code, Some(40110 | 40111))
}

#[derive(Debug, Clone)]
enum ParadexBookMessage {
    Snapshot(ParadexSnapshot),
    Delta(ParadexDelta),
}

#[derive(Debug, Clone)]
struct ParadexSnapshot {
    market: String,
    seq: u64,
    timestamp_ms: TimestampMs,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

#[derive(Debug, Clone)]
struct ParadexDelta {
    market: String,
    seq: u64,
    prev_seq: Option<u64>,
    timestamp_ms: TimestampMs,
    bids: Vec<BookLevelDelta>,
    asks: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone)]
struct ParadexSeqState {
    last_seq: Option<u64>,
    has_snapshot: bool,
    venue_index: usize,
}

impl ParadexSeqState {
    fn new(venue_index: usize) -> Self {
        Self {
            last_seq: None,
            has_snapshot: false,
            venue_index,
        }
    }

    fn apply(&mut self, message: ParadexBookMessage) -> anyhow::Result<Option<MarketDataEvent>> {
        match message {
            ParadexBookMessage::Snapshot(snapshot) => {
                self.last_seq = Some(snapshot.seq);
                self.has_snapshot = true;
                Ok(Some(MarketDataEvent::L2Snapshot(
                    super::super::types::L2Snapshot {
                        venue_index: self.venue_index,
                        venue_id: snapshot.market,
                        seq: snapshot.seq,
                        timestamp_ms: snapshot.timestamp_ms,
                        bids: snapshot.bids,
                        asks: snapshot.asks,
                    },
                )))
            }
            ParadexBookMessage::Delta(delta) => {
                if !self.has_snapshot {
                    return Ok(None);
                }
                if let Some(prev) = delta.prev_seq {
                    if Some(prev) != self.last_seq {
                        return Err(anyhow::anyhow!(
                            "paradex seq mismatch prev_seq={:?} last_seq={:?}",
                            prev,
                            self.last_seq
                        ));
                    }
                }
                if let Some(last) = self.last_seq {
                    if delta.seq <= last {
                        return Ok(None);
                    }
                }
                self.last_seq = Some(delta.seq);
                let mut changes = Vec::with_capacity(delta.bids.len() + delta.asks.len());
                changes.extend(delta.bids.iter().cloned());
                changes.extend(delta.asks.iter().cloned());
                Ok(Some(MarketDataEvent::L2Delta(
                    super::super::types::L2Delta {
                        venue_index: self.venue_index,
                        venue_id: delta.market,
                        seq: delta.seq,
                        timestamp_ms: delta.timestamp_ms,
                        changes,
                    },
                )))
            }
        }
    }
}

fn token_token(token: &ParadexAuthToken) -> Option<String> {
    if !token.access_token.is_empty() {
        return Some(token.access_token.clone());
    }
    if !token.token.is_empty() {
        return Some(token.token.clone());
    }
    if !token.jwt.is_empty() {
        return Some(token.jwt.clone());
    }
    None
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn cached_token_stale(cached: &CachedParadexToken, refresh_secs: u64) -> bool {
    let now = unix_now_secs();
    if let Some(expires_at) = cached.token.expires_at {
        if now.saturating_add(30) >= expires_at {
            return true;
        }
    }
    if let Some(expires_in) = cached.token.expires_in {
        let early_refresh = expires_in.saturating_sub(30).max(1);
        let refresh_after = refresh_secs.max(1).min(early_refresh);
        return now >= cached.fetched_unix_s.saturating_add(refresh_after);
    }
    now >= cached.fetched_unix_s.saturating_add(refresh_secs.max(1))
}

fn parse_token_output(raw: &str) -> Option<ParadexAuthToken> {
    if raw.is_empty() {
        return None;
    }
    if let Some(token) = parse_auth_token(raw) {
        return Some(token);
    }
    let jwt = raw
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .to_string();
    Some(ParadexAuthToken {
        access_token: jwt,
        token: String::new(),
        jwt: String::new(),
        expires_in: None,
        expires_at: None,
    })
}

fn parse_auth_token(body: &str) -> Option<ParadexAuthToken> {
    serde_json::from_str::<ParadexAuthToken>(body).ok()
}

fn parse_order_id(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    if let Some(order_id) = value
        .get("id")
        .or_else(|| value.get("order_id"))
        .or_else(|| value.get("orderId"))
    {
        if let Some(raw) = order_id.as_str() {
            return Some(raw.to_string());
        }
        if let Some(raw) = order_id.as_i64() {
            return Some(raw.to_string());
        }
    }
    value
        .get("client_id")
        .or_else(|| value.get("client_order_id"))
        .or_else(|| value.get("clientOrderId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn is_client_order_id(value: &str) -> bool {
    value.starts_with("co_")
}

fn is_paradex_cancel_not_found(body: &str) -> bool {
    body.contains("ORDER_ID_NOT_FOUND") || body.contains("CLIENT_ORDER_ID_NOT_FOUND")
}

fn map_time_in_force(time_in_force: TimeInForce, post_only: bool) -> &'static str {
    if post_only {
        return "POST_ONLY";
    }
    match time_in_force {
        TimeInForce::Ioc => "IOC",
        TimeInForce::Gtc => "GTC",
    }
}

fn map_side(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

fn build_order_payload(market: &str, req: &LiveRestPlaceRequest) -> LiveResult<Value> {
    if req.post_only && req.time_in_force == TimeInForce::Ioc {
        return Err(LiveGatewayError::post_only_reject(
            "paradex: post_only + IOC not allowed",
        ));
    }
    let instruction = map_time_in_force(req.time_in_force, req.post_only);
    let mut payload = serde_json::json!({
        "market": market,
        "side": map_side(req.side),
        "type": "LIMIT",
        "instruction": instruction,
        "price": format_decimal(req.price),
        "size": format_decimal(req.size),
        "client_id": req.client_order_id,
    });
    if req.reduce_only {
        payload["flags"] = serde_json::json!(["REDUCE_ONLY"]);
    }
    Ok(payload)
}

fn build_modify_order_payload(market: &str, req: &LiveRestReplaceRequest) -> LiveResult<Value> {
    let place = LiveRestPlaceRequest {
        venue_index: req.venue_index,
        venue_id: req.venue_id.clone(),
        side: req.side,
        price: req.price,
        size: req.size,
        purpose: req.purpose,
        time_in_force: req.time_in_force,
        post_only: req.post_only,
        reduce_only: req.reduce_only,
        client_order_id: req.client_order_id.clone(),
    };
    let mut payload = build_order_payload(market, &place)?;
    payload["id"] = serde_json::json!(req.order_id);
    Ok(payload)
}

fn parse_client_order_id(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .get("client_id")
        .or_else(|| value.get("client_order_id"))
        .or_else(|| value.get("clientOrderId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn format_decimal(value: f64) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    let mut formatted = format!("{value:.12}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    formatted
}

fn parse_account_snapshot(
    value: &Value,
    venue_id: &str,
    venue_index: usize,
) -> Option<AccountSnapshot> {
    if value.get("positions").is_some() && value.get("balances").is_some() {
        let positions = value
            .get("positions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|pos| {
                        let symbol = pos.get("symbol")?.as_str()?.to_string();
                        let size = pos.get("size")?.as_f64()?;
                        let entry_price = pos
                            .get("entry_price")
                            .or_else(|| pos.get("entryPrice"))?
                            .as_f64()?;
                        Some(PositionSnapshot {
                            symbol,
                            size,
                            entry_price,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let balances = value
            .get("balances")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|bal| {
                        let asset = bal.get("asset")?.as_str()?.to_string();
                        let total = bal.get("total")?.as_f64()?;
                        let available = bal.get("available")?.as_f64()?;
                        Some(BalanceSnapshot {
                            asset,
                            total,
                            available,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let margin = value.get("margin")?;
        let margin = MarginSnapshot {
            balance_usd: margin.get("balance_usd")?.as_f64()?,
            used_usd: margin.get("used_usd")?.as_f64()?,
            available_usd: margin.get("available_usd")?.as_f64()?,
        };
        let liquidation = value.get("liquidation")?;
        let liquidation = LiquidationSnapshot {
            price_liq: liquidation.get("price_liq").and_then(|v| v.as_f64()),
            dist_liq_sigma: liquidation.get("dist_liq_sigma").and_then(|v| v.as_f64()),
        };
        return Some(AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: value.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
            timestamp_ms: value
                .get("timestamp_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            open_order_count: None,
            positions,
            balances,
            funding_8h: value.get("funding_8h").and_then(|v| v.as_f64()),
            margin,
            liquidation,
        });
    }
    if value.get("account_value").is_some() && value.get("free_collateral").is_some() {
        let balance_usd = value.get("account_value").and_then(parse_f64)?;
        let available_usd = value.get("free_collateral").and_then(parse_f64)?;
        let used_usd = value
            .get("initial_margin_requirement")
            .and_then(parse_f64)
            .unwrap_or_else(|| (balance_usd - available_usd).max(0.0));
        let total_collateral = value
            .get("total_collateral")
            .and_then(parse_f64)
            .unwrap_or(balance_usd);
        let settlement_asset = value
            .get("settlement_asset")
            .and_then(|v| v.as_str())
            .unwrap_or("USDC");
        return Some(AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: value.get("seq_no").and_then(parse_u64_value).unwrap_or(0),
            timestamp_ms: value
                .get("updated_at")
                .and_then(parse_i64_value)
                .unwrap_or(0),
            open_order_count: None,
            positions: Vec::new(),
            balances: vec![BalanceSnapshot {
                asset: settlement_asset.to_string(),
                total: total_collateral,
                available: available_usd,
            }],
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd,
                used_usd,
                available_usd,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        });
    }
    None
}

fn parse_position_snapshots(value: &Value, market: &str) -> Vec<PositionSnapshot> {
    let list = value
        .get("results")
        .and_then(|v| v.as_array())
        .or_else(|| value.as_array())
        .cloned()
        .unwrap_or_default();
    list.into_iter()
        .filter_map(|pos| {
            let symbol = pos
                .get("market")
                .or_else(|| pos.get("symbol"))
                .and_then(|v| v.as_str())?
                .to_string();
            if !market.is_empty() && symbol != market {
                return None;
            }
            let mut size = pos.get("size").and_then(parse_f64)?;
            if let Some(side) = pos.get("side").and_then(|v| v.as_str()) {
                if side.eq_ignore_ascii_case("short") && size > 0.0 {
                    size = -size;
                } else if side.eq_ignore_ascii_case("long") && size < 0.0 {
                    size = size.abs();
                }
            }
            if size.abs() <= f64::EPSILON {
                return None;
            }
            let entry_price = pos
                .get("average_entry_price")
                .or_else(|| pos.get("average_entry_price_usd"))
                .and_then(parse_f64)?;
            Some(PositionSnapshot {
                symbol,
                size,
                entry_price,
            })
        })
        .collect()
}

fn parse_open_order_ids(value: &Value, market: &str) -> Vec<String> {
    let list = value
        .get("results")
        .and_then(|v| v.as_array())
        .or_else(|| value.as_array())
        .cloned()
        .unwrap_or_default();
    list.into_iter()
        .filter_map(|order| {
            let order_market = order.get("market").and_then(|v| v.as_str()).unwrap_or("");
            let status = order.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if order_market != market || !status.eq_ignore_ascii_case("OPEN") {
                return None;
            }
            order
                .get("id")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string())
        })
        .collect()
}

fn parse_open_orders(value: &Value, market: &str) -> Vec<OpenOrderSnapshot> {
    let list = value
        .get("results")
        .and_then(|v| v.as_array())
        .or_else(|| value.as_array())
        .cloned()
        .unwrap_or_default();
    list.into_iter()
        .filter_map(|order| {
            let order_market = order
                .get("market")
                .or_else(|| order.get("symbol"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if order_market != market {
                return None;
            }
            let status = order.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if !(status.eq_ignore_ascii_case("OPEN")
                || status.eq_ignore_ascii_case("NEW")
                || status.eq_ignore_ascii_case("PARTIALLY_FILLED"))
            {
                return None;
            }
            let order_id = order
                .get("id")
                .or_else(|| order.get("order_id"))
                .or_else(|| order.get("orderId"))
                .and_then(|v| {
                    v.as_str()
                        .map(|raw| raw.to_string())
                        .or_else(|| v.as_i64().map(|raw| raw.to_string()))
                })?;
            let client_order_id = order
                .get("client_id")
                .or_else(|| order.get("client_order_id"))
                .or_else(|| order.get("clientOrderId"))
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            let side = match order.get("side").and_then(|v| v.as_str())? {
                side if side.eq_ignore_ascii_case("BUY") => Side::Buy,
                side if side.eq_ignore_ascii_case("SELL") => Side::Sell,
                _ => return None,
            };
            let price = order
                .get("price")
                .or_else(|| order.get("limit_price"))
                .and_then(parse_f64)?;
            let size = order
                .get("remaining_size")
                .or_else(|| order.get("remainingQuantity"))
                .or_else(|| order.get("remaining_qty"))
                .or_else(|| order.get("size"))
                .and_then(parse_f64)?;
            let exchange_order_id = (!is_client_order_id(&order_id)).then_some(order_id.clone());
            (size > 0.0).then_some(OpenOrderSnapshot {
                order_id,
                client_order_id,
                exchange_order_id,
                side,
                price,
                size,
                purpose: None,
            })
        })
        .collect()
}

async fn fetch_public_funding(
    client: &Client,
    cfg: &ParadexConfig,
) -> anyhow::Result<FundingUpdate> {
    // Default path is "/markets/summary" (not "/v1/markets/summary") because
    // cfg.rest_url already includes the /v1 prefix (e.g., "https://api.prod.paradex.trade/v1").
    let path =
        std::env::var("PARADEX_FUNDING_PATH").unwrap_or_else(|_| "/markets/summary".to_string());
    let url = format!("{}{}", cfg.rest_url.trim_end_matches('/'), path);
    let resp = client
        .get(url)
        .query(&[("market", cfg.market.clone())])
        .send()
        .await?;
    let value: Value = resp.json().await?;
    parse_public_funding(&value, cfg)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response"))
}

fn parse_public_funding(value: &Value, cfg: &ParadexConfig) -> Option<FundingUpdate> {
    // Support multiple array extraction paths:
    // - value["data"] (generic)
    // - value["results"] (Paradex /v1/markets/summary response format)
    // - value itself as array
    let list = value
        .get("data")
        .and_then(|v| v.as_array())
        .or_else(|| value.get("results").and_then(|v| v.as_array()))
        .or_else(|| value.as_array());

    let entry = if let Some(list) = list {
        // Match by "market" or "symbol" (case-insensitive)
        list.iter()
            .find(|item| {
                item.get("market")
                    .or_else(|| item.get("symbol"))
                    .and_then(|v| v.as_str())
                    .map(|m| m.eq_ignore_ascii_case(&cfg.market))
                    .unwrap_or(false)
            })
            .unwrap_or(value)
    } else {
        value
    };

    let rate_native = entry
        .get("funding_rate_8h")
        .or_else(|| entry.get("funding_rate"))
        .or_else(|| entry.get("fundingRate"))
        .or_else(|| entry.get("fundingRate8h"))
        .and_then(parse_f64);

    let interval_sec = entry
        .get("funding_interval_sec")
        .or_else(|| entry.get("fundingIntervalSec"))
        .or_else(|| entry.get("funding_interval"))
        .or_else(|| entry.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or_else(|| {
            // Paradex quotes funding rates as 8h-equivalent (28800s).
            // Hardcode so downstream consumers see an explicit interval, not null.
            // Ref: https://docs.paradex.trade/docs/risk/funding-mechanism
            if rate_native.is_some() {
                Some(28_800)
            } else {
                None
            }
        });

    let next_funding_ms = entry
        .get("next_funding_time")
        .or_else(|| entry.get("nextFundingTime"))
        .or_else(|| entry.get("nextFundingTimestamp"))
        .or_else(|| entry.get("next_funding_ms"))
        .and_then(parse_i64_value);

    // Timestamp extraction: try standard fields, then "created_at" (Paradex markets/summary uses this)
    let as_of_ms = entry
        .get("timestamp")
        .or_else(|| entry.get("ts"))
        .or_else(|| entry.get("time"))
        .or_else(|| entry.get("created_at"))
        .and_then(parse_i64_value)
        .unwrap_or_else(now_ms);

    // Rate conversion to canonical 8h:
    // - If interval_sec is provided, scale: rate_8h = rate_native * (8h / interval_sec)
    // - If interval_sec is absent (e.g., Paradex /v1/markets/summary), the funding_rate
    //   field is already an 8h-equivalent rate per Paradex documentation, so we use it directly.
    let rate_8h = match (rate_native, interval_sec) {
        (Some(rate), Some(sec)) if sec > 0 => Some(rate * (8.0 * 60.0 * 60.0 / sec as f64)),
        (Some(rate), None) => Some(rate), // Already 8h-equivalent for Paradex
        _ => None,
    };

    Some(FundingUpdate {
        venue_index: cfg.venue_index,
        venue_id: cfg.market.clone(),
        seq: 0,
        timestamp_ms: as_of_ms,
        received_ms: Some(now_ms()),
        funding_rate_8h: rate_8h,
        funding_rate_native: rate_native,
        interval_sec,
        next_funding_ms,
        // Paradex: Funding Premium = (Mark - Spot Oracle) / USDC Oracle.
        // Settlement in USDC, oracle-adjusted. Ref: https://docs.paradex.trade/docs/risk/funding-mechanism
        settlement_price_kind: Some(SettlementPriceKind::UsdcOracleAdjusted),
        source: FundingSource::MarketDataRest,
    })
}

fn map_rest_error(status: u16, body: &str) -> LiveGatewayError {
    let lower = body.to_lowercase();
    if status == 401 || status == 403 {
        return LiveGatewayError {
            kind: LiveGatewayErrorKind::Fatal,
            message: format!("auth_error: {body}"),
        };
    }
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(body);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(body);
    }
    if status == 429 || lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(body);
    }
    if status >= 500 || lower.contains("timeout") {
        return LiveGatewayError::retryable(body);
    }
    LiveGatewayError {
        kind: LiveGatewayErrorKind::Fatal,
        message: body.to_string(),
    }
}

#[derive(Debug)]
struct ParadexRecorder {
    dir: PathBuf,
}

impl ParadexRecorder {
    fn new(dir: &PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self { dir: dir.clone() })
    }

    fn record_ws_frame(&mut self, raw: &str) -> std::io::Result<()> {
        let path = self.dir.join("ws_frames.jsonl");
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        use std::io::Write;
        file.write_all(raw.as_bytes())?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

fn parse_orderbook_message_value(
    value: &Value,
    tracker: &mut ParadexSeqState,
) -> anyhow::Result<Option<MarketDataEvent>> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    let is_legacy_channel = channel == "orderbook" || channel == "order_book";
    let is_new_channel = channel.starts_with("order_book.");
    if !is_legacy_channel && !is_new_channel {
        return Ok(None);
    }
    if is_new_channel {
        if let Some(message) = parse_orderbook_structured_message(payload) {
            return tracker.apply(message);
        }
    }
    let message_type = payload
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("action").and_then(|v| v.as_str()))
        .unwrap_or("");
    let message = if message_type == "snapshot" {
        parse_snapshot(payload).map(ParadexBookMessage::Snapshot)
    } else if message_type == "delta" || message_type == "update" {
        parse_delta(payload).map(ParadexBookMessage::Delta)
    } else {
        None
    };
    if let Some(message) = message {
        return tracker.apply(message);
    }
    Ok(None)
}

fn decode_interactive_top_value(value: &Value) -> Option<ParadexInteractiveTop> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    if !channel.starts_with("order_book.") {
        return None;
    }
    let data = payload.get("data").unwrap_or(payload);
    let best_bid_api = parse_top_entry(data.get("best_bid_api")?)?;
    let best_bid_interactive = parse_top_entry(data.get("best_bid_interactive")?)?;
    let best_ask_api = parse_top_entry(data.get("best_ask_api")?)?;
    let best_ask_interactive = parse_top_entry(data.get("best_ask_interactive")?)?;
    Some(ParadexInteractiveTop {
        best_bid_api,
        best_bid_interactive,
        best_ask_api,
        best_ask_interactive,
        seq_no: data.get("seq_no").and_then(parse_u64_value),
    })
}

fn decode_interactive_public_top_and_snapshot(
    value: &Value,
    venue_index: usize,
    venue_id: &str,
    seq: &mut u64,
    ui_touch_reference_state: Option<&ParadexUiTouchReferenceState>,
) -> Option<(TopOfBook, MarketDataEvent, &'static str)> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    if !channel.contains(".interactive") || !channel.starts_with("order_book.") {
        return None;
    }
    let data = payload.get("data").unwrap_or(payload);
    let book_bid = parse_first_level_entry(data.get("bids"));
    let book_ask = parse_first_level_entry(data.get("asks"));
    let api_bid = data.get("best_bid_api").and_then(parse_top_entry);
    let api_ask = data.get("best_ask_api").and_then(parse_top_entry);
    let interactive_bid = data.get("best_bid_interactive").and_then(parse_top_entry);
    let interactive_ask = data.get("best_ask_interactive").and_then(parse_top_entry);

    let candidates = [
        (
            "interactive_best",
            interactive_bid,
            interactive_ask,
            api_bid,
            api_ask,
            book_bid,
            book_ask,
        ),
        (
            "api_best",
            api_bid,
            api_ask,
            interactive_bid,
            interactive_ask,
            book_bid,
            book_ask,
        ),
        (
            "book_levels",
            book_bid,
            book_ask,
            interactive_bid,
            interactive_ask,
            api_bid,
            api_ask,
        ),
    ];

    for (
        top_source,
        primary_bid,
        primary_ask,
        secondary_bid,
        secondary_ask,
        tertiary_bid,
        tertiary_ask,
    ) in candidates
    {
        let Some(bid_px) = positive_finite(primary_bid.and_then(|entry| entry.price)) else {
            continue;
        };
        let Some(ask_px) = positive_finite(primary_ask.and_then(|entry| entry.price)) else {
            continue;
        };
        let bid_sz = positive_finite(primary_bid.and_then(|entry| entry.size))
            .or_else(|| positive_finite(secondary_bid.and_then(|entry| entry.size)))
            .or_else(|| positive_finite(tertiary_bid.and_then(|entry| entry.size)));
        let ask_sz = positive_finite(primary_ask.and_then(|entry| entry.size))
            .or_else(|| positive_finite(secondary_ask.and_then(|entry| entry.size)))
            .or_else(|| positive_finite(tertiary_ask.and_then(|entry| entry.size)));
        let (Some(bid_sz), Some(ask_sz)) = (bid_sz, ask_sz) else {
            continue;
        };
        if bid_px >= ask_px {
            continue;
        }
        let ((bid_px, bid_sz, ask_px, ask_sz), touch_source_kind) = ui_touch_reference_state
            .map(|state| state.apply_to_bbo_with_source(bid_px, bid_sz, ask_px, ask_sz))
            .unwrap_or(((bid_px, bid_sz, ask_px, ask_sz), None));
        let top_source = if touch_source_kind == Some("top_level_fallback") {
            "interactive_top_level_fallback"
        } else {
            top_source
        };
        if let Some((top, snapshot)) =
            build_bbo_snapshot(venue_index, venue_id, seq, bid_px, bid_sz, ask_px, ask_sz)
        {
            return Some((top, snapshot, top_source));
        }
    }
    None
}

fn decode_ui_book_truth_snapshot(value: &Value) -> Option<ParadexUiBookTruthSnapshot> {
    let payload = value
        .get("result")
        .or_else(|| value.get("results"))
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let snapshot = ParadexUiBookTruthSnapshot {
        bid: parse_first_level_entry(payload.get("bids")).unwrap_or_default(),
        ask: parse_first_level_entry(payload.get("asks")).unwrap_or_default(),
        best_bid_api: payload
            .get("best_bid_api")
            .and_then(parse_top_entry)
            .unwrap_or_default(),
        best_bid_interactive: payload
            .get("best_bid_interactive")
            .and_then(parse_top_entry)
            .unwrap_or_default(),
        best_ask_api: payload
            .get("best_ask_api")
            .and_then(parse_top_entry)
            .unwrap_or_default(),
        best_ask_interactive: payload
            .get("best_ask_interactive")
            .and_then(parse_top_entry)
            .unwrap_or_default(),
        seq_no: payload.get("seq_no").and_then(parse_u64_value),
        last_updated_at_ms: payload.get("last_updated_at").and_then(parse_i64_value),
        ..ParadexUiBookTruthSnapshot::default()
    };
    snapshot.has_any_truth().then_some(snapshot)
}

fn parse_first_level_entry(value: Option<&Value>) -> Option<ParadexTopEntry> {
    let levels = value?.as_array()?;
    parse_top_entry(levels.first()?)
}

fn parse_top_entry(value: &Value) -> Option<ParadexTopEntry> {
    if let Some(items) = value.as_array() {
        return Some(ParadexTopEntry {
            price: items.first().and_then(parse_f64),
            size: items.get(1).and_then(parse_f64),
        });
    }
    if value.is_object() {
        return Some(ParadexTopEntry {
            price: value
                .get("price")
                .or_else(|| value.get("px"))
                .or_else(|| value.get("bid"))
                .or_else(|| value.get("ask"))
                .and_then(parse_f64),
            size: value
                .get("size")
                .or_else(|| value.get("qty"))
                .or_else(|| value.get("amount"))
                .and_then(parse_f64),
        });
    }
    parse_f64(value).map(|price| ParadexTopEntry {
        price: Some(price),
        size: None,
    })
}

fn parse_fill_flags_from_body(body: &str) -> Vec<String> {
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    let mut flags: Vec<String> = Vec::new();
    collect_fill_flags(&value, &mut flags);
    flags.sort();
    flags.dedup();
    flags
}

fn collect_fill_flags(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_fill_flags(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(flag_values) = map.get("flags").and_then(|entry| entry.as_array()) {
                for flag in flag_values.iter().filter_map(|entry| entry.as_str()) {
                    out.push(flag.to_ascii_lowercase());
                }
            }
            if let Some(fills) = map.get("fills").and_then(|entry| entry.as_array()) {
                for fill in fills {
                    collect_fill_flags(fill, out);
                }
            }
        }
        _ => {}
    }
}

fn parse_orderbook_structured_message(payload: &Value) -> Option<ParadexBookMessage> {
    let data = payload.get("data")?;
    let updates = data.get("updates").and_then(|v| v.as_array());
    let inserts = data.get("inserts").and_then(|v| v.as_array());
    let deletes = data.get("deletes").and_then(|v| v.as_array());
    if updates.is_none() && inserts.is_none() && deletes.is_none() {
        return None;
    }

    let market = data.get("market").and_then(|v| v.as_str())?.to_string();
    let seq = data.get("seq_no").and_then(parse_u64_value)?;
    let update_type = data.get("update_type").and_then(|v| v.as_str())?;
    let timestamp_ms = now_ms();

    let mut upserts = parse_orderbook_entries(updates)?;
    upserts.extend(parse_orderbook_entries(inserts)?);

    let update_type_lc = update_type.to_ascii_lowercase();
    let is_snapshot_like = update_type_lc == "snapshot" || update_type_lc.starts_with('s');
    if is_snapshot_like {
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        for (side, price, size) in upserts {
            match side {
                BookSide::Bid => bids.push(BookLevel { price, size }),
                BookSide::Ask => asks.push(BookLevel { price, size }),
            }
        }
        return Some(ParadexBookMessage::Snapshot(ParadexSnapshot {
            market,
            seq,
            timestamp_ms,
            bids,
            asks,
        }));
    }

    let mut bids = Vec::new();
    let mut asks = Vec::new();
    for (side, price, size) in upserts {
        let level = BookLevelDelta { side, price, size };
        match level.side {
            BookSide::Bid => bids.push(level),
            BookSide::Ask => asks.push(level),
        }
    }
    for (side, price, _size) in parse_orderbook_entries(deletes)? {
        let level = BookLevelDelta {
            side,
            price,
            size: 0.0,
        };
        match level.side {
            BookSide::Bid => bids.push(level),
            BookSide::Ask => asks.push(level),
        }
    }

    Some(ParadexBookMessage::Delta(ParadexDelta {
        market,
        seq,
        prev_seq: None,
        timestamp_ms,
        bids,
        asks,
    }))
}

fn parse_orderbook_entries(entries: Option<&Vec<Value>>) -> Option<Vec<(BookSide, f64, f64)>> {
    let Some(entries) = entries else {
        return Some(Vec::new());
    };
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        out.push(parse_orderbook_entry(entry)?);
    }
    Some(out)
}

fn parse_orderbook_entry(entry: &Value) -> Option<(BookSide, f64, f64)> {
    let side = parse_orderbook_side(entry.get("side").and_then(|v| v.as_str())?)?;
    let price = entry.get("price").and_then(parse_f64)?;
    let size = entry.get("size").and_then(parse_f64)?;
    Some((side, price, size))
}

fn parse_orderbook_side(side: &str) -> Option<BookSide> {
    if side.eq_ignore_ascii_case("buy") || side.eq_ignore_ascii_case("bid") {
        return Some(BookSide::Bid);
    }
    if side.eq_ignore_ascii_case("sell") || side.eq_ignore_ascii_case("ask") {
        return Some(BookSide::Ask);
    }
    None
}

fn parse_snapshot(payload: &Value) -> Option<ParadexSnapshot> {
    let market = payload
        .get("market")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())?
        .to_string();
    let seq = payload
        .get("seq")
        .or_else(|| payload.get("sequence"))
        .and_then(|v| v.as_u64())?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs)
        .unwrap_or_else(now_ms);
    let bids = parse_levels_from_value(payload.get("bids")?)?;
    let asks = parse_levels_from_value(payload.get("asks")?)?;
    Some(ParadexSnapshot {
        market,
        seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn parse_delta(payload: &Value) -> Option<ParadexDelta> {
    let market = payload
        .get("market")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())?
        .to_string();
    let seq = payload
        .get("seq")
        .or_else(|| payload.get("sequence"))
        .and_then(|v| v.as_u64())?;
    let prev_seq = payload
        .get("prev_seq")
        .or_else(|| payload.get("prevSequence"))
        .and_then(|v| v.as_u64());
    let timestamp_ms = now_ms();
    let bids = parse_deltas_from_value(payload.get("bids")?, BookSide::Bid)?;
    let asks = parse_deltas_from_value(payload.get("asks")?, BookSide::Ask)?;
    Some(ParadexDelta {
        market,
        seq,
        prev_seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn decode_top_of_book(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let bids = parse_levels_from_value(payload.get("bids")?)?;
    let asks = parse_levels_from_value(payload.get("asks")?)?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(&bids, &asks, timestamp_ms)
}

fn decode_top_of_book_value(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let bids = parse_levels_any(payload.get("bids")?)?;
    let asks = parse_levels_any(payload.get("asks")?)?;
    let timestamp_ms = payload
        .get("ts")
        .or_else(|| payload.get("timestamp"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(&bids, &asks, timestamp_ms)
}

fn build_bbo_snapshot(
    venue_index: usize,
    venue_id: &str,
    seq: &mut u64,
    bid_px: f64,
    bid_sz: f64,
    ask_px: f64,
    ask_sz: f64,
) -> Option<(TopOfBook, MarketDataEvent)> {
    if bid_sz <= 0.0 || ask_sz <= 0.0 {
        return None;
    }
    let timestamp_ms = now_ms();
    *seq = seq.wrapping_add(1);
    let bids = vec![BookLevel {
        price: bid_px,
        size: bid_sz,
    }];
    let asks = vec![BookLevel {
        price: ask_px,
        size: ask_sz,
    }];
    let top = TopOfBook::from_levels(&bids, &asks, Some(timestamp_ms))?;
    let snapshot = MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq: *seq,
        timestamp_ms,
        bids,
        asks,
    });
    Some((top, snapshot))
}

fn decode_bbo_top_and_snapshot(
    value: &Value,
    venue_index: usize,
    venue_id: &str,
    seq: &mut u64,
    ui_touch_reference_state: Option<&ParadexUiTouchReferenceState>,
) -> Option<(TopOfBook, MarketDataEvent)> {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    if !channel.starts_with("bbo.") {
        return None;
    }
    let data = payload.get("data").unwrap_or(payload);
    let decoded_bid_px = data
        .get("bid")
        .or_else(|| data.get("bid_price"))
        .and_then(parse_f64)?;
    let decoded_bid_sz = data
        .get("bid_size")
        .or_else(|| data.get("bidSize"))
        .and_then(parse_f64)?;
    let decoded_ask_px = data
        .get("ask")
        .or_else(|| data.get("ask_price"))
        .and_then(parse_f64)?;
    let decoded_ask_sz = data
        .get("ask_size")
        .or_else(|| data.get("askSize"))
        .and_then(parse_f64)?;
    let (bid_px, bid_sz, ask_px, ask_sz) = ui_touch_reference_state
        .map(|state| {
            state.apply_to_bbo(
                decoded_bid_px,
                decoded_bid_sz,
                decoded_ask_px,
                decoded_ask_sz,
            )
        })
        .unwrap_or((
            decoded_bid_px,
            decoded_bid_sz,
            decoded_ask_px,
            decoded_ask_sz,
        ));
    build_bbo_snapshot(venue_index, venue_id, seq, bid_px, bid_sz, ask_px, ask_sz)
}

fn log_decode_miss(venue: &str, value: &Value, payload: &str, count: usize, url: &str) {
    let keys = value
        .as_object()
        .map(|obj| {
            let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            keys.sort();
            format!("[{}]", keys.join(","))
        })
        .unwrap_or_else(|| "[non-object]".to_string());
    let snippet: String = payload.chars().take(160).collect();
    eprintln!(
        "WARN: {venue} WS decode miss keys={keys} snippet={snippet} (count={count}) url={url}",
    );
}

#[derive(Clone)]
struct ParadexSubscribeCandidate {
    method: String,
    params: Value,
}

impl ParadexSubscribeCandidate {
    fn new(method: &str, params: Value) -> Self {
        Self {
            method: method.to_string(),
            params,
        }
    }
}

async fn send_paradex_subscribe(
    write: &mut (impl futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error>
              + Unpin),
    candidate: &ParadexSubscribeCandidate,
) -> anyhow::Result<()> {
    send_paradex_subscribe_with_id(write, candidate, 1).await
}

async fn send_paradex_subscribe_with_id(
    write: &mut (impl futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error>
              + Unpin),
    candidate: &ParadexSubscribeCandidate,
    id: u64,
) -> anyhow::Result<()> {
    let sub = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": candidate.method,
        "params": candidate.params,
    });
    write.send(Message::Text(sub.to_string())).await?;
    Ok(())
}

fn paradex_subscribe_error(value: &Value) -> bool {
    let err = value.get("error").and_then(|v| v.as_object());
    let Some(err) = err else {
        return false;
    };
    let message = err
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    let data = err
        .get("data")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    message.contains("invalid") || data.contains("invalid")
}

fn is_paradex_orderbook_message(value: &Value) -> bool {
    let payload = value
        .get("params")
        .or_else(|| value.get("data"))
        .unwrap_or(value);
    let channel = payload
        .get("channel")
        .and_then(|v| v.as_str())
        .or_else(|| value.get("channel").and_then(|v| v.as_str()))
        .unwrap_or("");
    channel == "orderbook" || channel == "order_book" || channel.starts_with("order_book.")
}

fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        out.push(BookLevel { price, size });
    }
    Some(out)
}

fn parse_levels_any(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some((price, size)) = parse_level_pair(entry) {
            out.push(BookLevel { price, size });
            continue;
        }
        if let Some(obj) = entry.as_object() {
            let price = obj
                .get("px")
                .or_else(|| obj.get("price"))
                .and_then(parse_f64)?;
            let size = obj
                .get("sz")
                .or_else(|| obj.get("size"))
                .and_then(parse_f64)?;
            out.push(BookLevel { price, size });
        }
    }
    Some(out)
}

fn parse_deltas_from_value(value: &Value, side: BookSide) -> Option<Vec<BookLevelDelta>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        out.push(BookLevelDelta { side, price, size });
    }
    Some(out)
}

fn parse_level_pair(value: &Value) -> Option<(f64, f64)> {
    let items = value.as_array()?;
    if items.len() < 2 {
        return None;
    }
    let price = parse_f64(&items[0])?;
    let size = parse_f64(&items[1])?;
    Some((price, size))
}

fn parse_f64(value: &Value) -> Option<f64> {
    if let Some(v) = value.as_f64() {
        return Some(v);
    }
    if let Some(s) = value.as_str() {
        return s.parse::<f64>().ok();
    }
    None
}

fn parse_i64_value(value: &Value) -> Option<i64> {
    if let Some(v) = value.as_i64() {
        return Some(v);
    }
    if let Some(v) = value.as_f64() {
        return Some(v as i64);
    }
    if let Some(s) = value.as_str() {
        return s.parse::<i64>().ok();
    }
    None
}

fn parse_u64_value(value: &Value) -> Option<u64> {
    if let Some(v) = value.as_u64() {
        return Some(v);
    }
    if let Some(v) = value.as_i64() {
        return (v >= 0).then_some(v as u64);
    }
    if let Some(s) = value.as_str() {
        return s.parse::<u64>().ok();
    }
    None
}

fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as TimestampMs
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FixtureSnapshot {
    seq: u64,
    timestamp_ms: TimestampMs,
    bids: Vec<[f64; 2]>,
    asks: Vec<[f64; 2]>,
    venue_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureDelta {
    seq: u64,
    timestamp_ms: TimestampMs,
    side: String,
    price: f64,
    size: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixturePosition {
    symbol: String,
    size: f64,
    entry_price: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureBalance {
    asset: String,
    total: f64,
    available: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureMargin {
    balance_usd: f64,
    used_usd: f64,
    available_usd: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureLiquidation {
    price_liq: Option<f64>,
    dist_liq_sigma: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureAccountSnapshot {
    seq: u64,
    timestamp_ms: TimestampMs,
    positions: Vec<FixturePosition>,
    balances: Vec<FixtureBalance>,
    funding_8h: Option<f64>,
    margin: FixtureMargin,
    liquidation: FixtureLiquidation,
}

#[derive(Debug, Clone)]
pub struct ParadexFixtureFeed {
    snapshot: FixtureSnapshot,
    deltas: Vec<FixtureDelta>,
    account: FixtureAccountSnapshot,
}

impl ParadexFixtureFeed {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let snapshot_path = dir.join("snapshot.json");
        let deltas_path = dir.join("deltas.jsonl");
        let account_path = dir.join("account_snapshot.json");

        let snapshot = read_json::<FixtureSnapshot>(&snapshot_path)?;
        let deltas = read_json_lines::<FixtureDelta>(&deltas_path)?;
        let account = read_json::<FixtureAccountSnapshot>(&account_path)?;
        Ok(Self {
            snapshot,
            deltas,
            account,
        })
    }

    pub async fn run_ticks(
        &self,
        market_tx: mpsc::Sender<MarketDataEvent>,
        account_tx: mpsc::Sender<AccountEvent>,
        venue_id: &str,
        venue_index: usize,
        start_ms: TimestampMs,
        step_ms: i64,
        ticks: u64,
    ) {
        let pace_ticks = std::env::var("PARAPHINA_PAPER_USE_WALLCLOCK_TS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let sleep_duration = Duration::from_millis(step_ms.max(1) as u64);
        let mut seq: u64 = 1;
        for tick in 0..ticks {
            let now_ms = start_ms + step_ms.saturating_mul(tick as i64);
            let snapshot = snapshot_event(&self.snapshot, venue_id, venue_index, seq, now_ms);
            seq = seq.wrapping_add(1);
            let _ = market_tx.send(snapshot).await;
            for delta in &self.deltas {
                let delta_event = delta_event(delta, venue_id, venue_index, seq, now_ms);
                seq = seq.wrapping_add(1);
                let _ = market_tx.send(delta_event).await;
            }
            let account = account_event(&self.account, venue_id, venue_index, seq, now_ms);
            seq = seq.wrapping_add(1);
            let _ = account_tx.send(account).await;
            if pace_ticks {
                tokio::time::sleep(sleep_duration).await;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }
}

fn snapshot_event(
    snapshot: &FixtureSnapshot,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    let bids = parse_levels(&snapshot.bids);
    let asks = parse_levels(&snapshot.asks);
    MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    })
}

fn delta_event(
    delta: &FixtureDelta,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> MarketDataEvent {
    let side = match delta.side.as_str() {
        "bid" | "Bid" | "BID" => BookSide::Bid,
        "ask" | "Ask" | "ASK" => BookSide::Ask,
        _ => BookSide::Bid,
    };
    MarketDataEvent::L2Delta(super::super::types::L2Delta {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        changes: vec![BookLevelDelta {
            side,
            price: delta.price,
            size: delta.size,
        }],
    })
}

fn account_event(
    account: &FixtureAccountSnapshot,
    venue_id: &str,
    venue_index: usize,
    seq: u64,
    timestamp_ms: TimestampMs,
) -> AccountEvent {
    let positions = account
        .positions
        .iter()
        .map(|pos| PositionSnapshot {
            symbol: pos.symbol.clone(),
            size: pos.size,
            entry_price: pos.entry_price,
        })
        .collect();
    let balances = account
        .balances
        .iter()
        .map(|bal| BalanceSnapshot {
            asset: bal.asset.clone(),
            total: bal.total,
            available: bal.available,
        })
        .collect();
    AccountEvent::Snapshot(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        open_order_count: None,
        positions,
        balances,
        funding_8h: account.funding_8h,
        margin: MarginSnapshot {
            balance_usd: account.margin.balance_usd,
            used_usd: account.margin.used_usd,
            available_usd: account.margin.available_usd,
        },
        liquidation: LiquidationSnapshot {
            price_liq: account.liquidation.price_liq,
            dist_liq_sigma: account.liquidation.dist_liq_sigma,
        },
    })
}

fn parse_levels(levels: &[[f64; 2]]) -> Vec<BookLevel> {
    levels
        .iter()
        .map(|level| BookLevel {
            price: level[0],
            size: level[1],
        })
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("fixture_read_error path={} err={}", path.display(), err))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("fixture_parse_error path={} err={}", path.display(), err))
}

fn read_json_lines<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("fixture_read_error path={} err={}", path.display(), err))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let item: T = serde_json::from_str(line)
            .map_err(|err| format!("fixture_parse_error path={} err={}", path.display(), err))?;
        out.push(item);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_eval::env_override::with_env_overrides;
    use httpmock::Method::{DELETE, GET, POST, PUT};
    use httpmock::MockServer;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn private_order_truth_audit_line_redacts_raw_identifiers() {
        let line = paradex_private_order_truth_audit_line(
            1,
            "OPEN",
            "raw-order-id-123",
            Some("raw-client-id-456"),
            9_999,
        );

        assert!(line.contains("PARADEX_PRIVATE_ORDER_TRUTH"));
        assert!(line.contains("status=OPEN"));
        assert!(line.contains("order_id_state=present_redacted"));
        assert!(line.contains("client_id_state=present_redacted"));
        assert!(line.contains("seq_no_state=present_redacted"));
        assert!(!line.contains("raw-order-id-123"));
        assert!(!line.contains("raw-client-id-456"));
        assert!(!line.contains("9999"));

        let sanitized_status =
            paradex_private_order_truth_audit_line(2, "OPEN raw payload", "", None, 1);
        assert!(sanitized_status.contains("status=other"));
        assert!(sanitized_status.contains("order_id_state=absent"));
        assert!(sanitized_status.contains("client_id_state=absent"));
    }

    #[test]
    fn paradex_identity_audit_lines_redact_raw_identifiers() {
        let unresolved =
            paradex_open_identity_unresolved_audit_line("missing_native_id", "raw-client-id-789");
        assert!(unresolved.contains("PARADEX_OPEN_IDENTITY_UNRESOLVED"));
        assert!(unresolved.contains("reason=missing_native_id"));
        assert!(unresolved.contains("client_id_state=present_redacted"));
        assert!(!unresolved.contains("raw-client-id-789"));

        let native_replace = paradex_native_replace_audit_line(
            "private_ws",
            "raw-order-id-abc",
            Some("raw-client-id-def"),
        );
        assert!(native_replace.contains("PARADEX_NATIVE_REPLACE"));
        assert!(native_replace.contains("submit_source=private_ws"));
        assert!(native_replace.contains("order_id_state=present_redacted"));
        assert!(native_replace.contains("client_id_state=present_redacted"));
        assert!(!native_replace.contains("raw-order-id-abc"));
        assert!(!native_replace.contains("raw-client-id-def"));

        let sanitized_source =
            paradex_native_replace_audit_line("private ws raw", "raw-order-id", None);
        assert!(sanitized_source.contains("submit_source=other"));
        assert!(sanitized_source.contains("client_id_state=absent"));
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let mut bids = feed.snapshot.bids.clone();
        let delta = feed.deltas.first().expect("delta");
        let side = delta.side.to_lowercase();
        if side == "bid" {
            if let Some(level) = bids.iter_mut().find(|level| level[0] == delta.price) {
                level[1] = delta.size;
            } else {
                bids.push([delta.price, delta.size]);
            }
            assert!(bids.iter().any(|level| level[0] == delta.price));
        }
    }

    #[test]
    fn seq_gap_triggers_refresh_marker() {
        let gap = FixtureDelta {
            seq: 10,
            timestamp_ms: 1_000,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let next = FixtureDelta {
            seq: 12,
            timestamp_ms: 1_010,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let mut last_seq = gap.seq;
        let gap_detected = next.seq > last_seq + 1;
        last_seq = next.seq;
        assert!(gap_detected);
        assert_eq!(last_seq, 12);
    }

    #[test]
    fn deterministic_serialization_roundtrip() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let feed = ParadexFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let raw = serde_json::to_string(&feed.snapshot).expect("serialize");
        let reparsed: FixtureSnapshot = serde_json::from_str(&raw).expect("reparse");
        assert_eq!(feed.snapshot.seq, reparsed.seq);
        assert_eq!(feed.snapshot.bids.len(), reparsed.bids.len());
    }

    #[test]
    fn live_ws_snapshot_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex_live_recording");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");
        let first = frames
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("frame");
        let value: Value = serde_json::from_str(first).expect("snapshot json");
        let payload = value.get("params").unwrap_or(&value);
        let snapshot = parse_snapshot(payload).expect("parse snapshot");
        assert!(snapshot.seq > 0);
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[test]
    fn live_ws_replay_is_deterministic_and_monotonic() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/paradex_live_recording");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");
        let mut tracker = ParadexSeqState::new(0);
        let mut events = Vec::new();
        for line in frames.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed).expect("parse json");
            if let Ok(Some(event)) = parse_orderbook_message_value(&value, &mut tracker) {
                events.push(event);
            }
        }
        assert!(!events.is_empty());
        let mut last_ts: Option<TimestampMs> = None;
        for event in events {
            let ts = match event {
                MarketDataEvent::L2Delta(delta) => delta.timestamp_ms,
                MarketDataEvent::L2Snapshot(snapshot) => snapshot.timestamp_ms,
                MarketDataEvent::Trade(trade) => trade.timestamp_ms,
                MarketDataEvent::FundingUpdate(update) => update.timestamp_ms,
            };
            if let Some(prev) = last_ts {
                assert!(ts >= prev);
            }
            last_ts = Some(ts);
        }
    }

    #[test]
    fn auth_token_fixture_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/paradex");
        let raw = std::fs::read_to_string(fixture_dir.join("auth_token.json")).expect("token");
        let token = parse_auth_token(&raw).expect("parse token");
        assert_eq!(token_token(&token).unwrap(), "test.jwt");
    }

    #[tokio::test]
    async fn jwt_command_returns_token() {
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            auth_url: "https://example.invalid/auth".to_string(),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: None,
            jwt_cmd: Some("printf 'cmd.jwt\\n'".to_string()),
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        let token = client.ensure_token().await.expect("jwt from command");
        assert_eq!(token, "cmd.jwt");
    }

    #[test]
    fn bbo_decode_emits_top() {
        let mut seq = 0u64;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": "bbo.BTC-USD-PERP",
                "data": {
                    "bid": "30000",
                    "bid_size": "1.2",
                    "ask": "30010",
                    "ask_size": "0.9",
                    "ts": 1700000000000i64
                }
            }
        });
        let (top, _snapshot) =
            decode_bbo_top_and_snapshot(&msg, 0, "BTC-USD-PERP", &mut seq, None).expect("bbo");
        assert_eq!(top.best_bid_px, 30000.0);
        assert_eq!(top.best_bid_sz, 1.2);
        assert_eq!(top.best_ask_px, 30010.0);
        assert_eq!(top.best_ask_sz, 0.9);
    }

    #[test]
    fn public_feed_mode_defaults_and_variants_are_stable() {
        assert_eq!(
            ParadexPublicFeedMode::from_env(""),
            ParadexPublicFeedMode::Bbo
        );
        assert_eq!(
            ParadexPublicFeedMode::from_env("bbo"),
            ParadexPublicFeedMode::Bbo
        );
        assert_eq!(
            ParadexPublicFeedMode::from_env("orderbook"),
            ParadexPublicFeedMode::Orderbook {
                feed_type: "snapshot".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
        );
        assert_eq!(
            ParadexPublicFeedMode::from_env("orderbook_deltas"),
            ParadexPublicFeedMode::Orderbook {
                feed_type: "deltas".to_string(),
                refresh_rate: None,
            }
        );
        assert_eq!(
            ParadexPublicFeedMode::from_env("interactive"),
            ParadexPublicFeedMode::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
        );
        assert_eq!(
            ParadexPublicFeedMode::from_env("interactive_50ms"),
            ParadexPublicFeedMode::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("50ms".to_string()),
            }
        );
    }

    #[test]
    fn public_feed_mode_builds_expected_channels() {
        assert_eq!(
            ParadexPublicFeedMode::Bbo.channel("ETH-USD-PERP"),
            "bbo.ETH-USD-PERP"
        );
        assert_eq!(
            ParadexPublicFeedMode::Orderbook {
                feed_type: "snapshot".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
            .channel("ETH-USD-PERP"),
            "order_book.ETH-USD-PERP.snapshot@15@100ms"
        );
        assert_eq!(
            ParadexPublicFeedMode::Orderbook {
                feed_type: "deltas".to_string(),
                refresh_rate: None,
            }
            .channel("ETH-USD-PERP"),
            "order_book.ETH-USD-PERP.deltas"
        );
        assert_eq!(
            ParadexPublicFeedMode::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
            .channel("ETH-USD-PERP"),
            "order_book.ETH-USD-PERP.interactive@15@100ms"
        );
        assert_eq!(
            ParadexPublicFeedMode::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("50ms".to_string()),
            }
            .channel("ETH-USD-PERP"),
            "order_book.ETH-USD-PERP.interactive@15@50ms"
        );
    }

    #[test]
    fn paradex_bbo_backstop_enabled_reads_boolean_env() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_PARADEX_BBO_BACKSTOP_ENABLED".to_string(),
            "1".to_string(),
        );
        assert!(with_env_overrides(&overrides, paradex_bbo_backstop_enabled));

        overrides.insert(
            "PARAPHINA_PARADEX_BBO_BACKSTOP_ENABLED".to_string(),
            "false".to_string(),
        );
        assert!(!with_env_overrides(
            &overrides,
            paradex_bbo_backstop_enabled
        ));
    }

    #[test]
    fn paradex_token_usage_defaults_and_variants_are_stable() {
        assert_eq!(ParadexTokenUsage::from_env(""), ParadexTokenUsage::Pro);
        assert_eq!(ParadexTokenUsage::from_env("pro"), ParadexTokenUsage::Pro);
        assert_eq!(
            ParadexTokenUsage::from_env("interactive"),
            ParadexTokenUsage::Interactive
        );
    }

    #[test]
    fn auth_url_with_token_usage_adds_interactive_query_only_when_needed() {
        let base_cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            auth_url: "https://example.invalid/v1/auth".to_string(),
            token_usage: ParadexTokenUsage::Interactive,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: None,
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        assert_eq!(
            base_cfg.auth_url_with_token_usage(),
            "https://example.invalid/v1/auth?token_usage=interactive"
        );
        let already_query = ParadexConfig {
            auth_url: "https://example.invalid/v1/auth?token_usage=interactive".to_string(),
            ..base_cfg.clone()
        };
        assert_eq!(
            already_query.auth_url_with_token_usage(),
            "https://example.invalid/v1/auth?token_usage=interactive"
        );
        let pro_cfg = ParadexConfig {
            token_usage: ParadexTokenUsage::Pro,
            ..base_cfg
        };
        assert_eq!(
            pro_cfg.auth_url_with_token_usage(),
            "https://example.invalid/v1/auth"
        );
    }

    #[test]
    fn decode_interactive_top_value_accepts_array_and_object_shapes() {
        let array_value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@100ms",
                "data": {
                    "seq_no": 42,
                    "best_bid_api": ["3000.5", "1.2"],
                    "best_bid_interactive": {"price": "3000.4", "size": "0.8"},
                    "best_ask_api": ["3001.1", "1.5"],
                    "best_ask_interactive": {"price": "3001.2", "size": "0.6"}
                }
            }
        });
        let top = decode_interactive_top_value(&array_value).expect("interactive top");
        assert_eq!(top.seq_no, Some(42));
        assert_eq!(top.best_bid_api.price, Some(3000.5));
        assert_eq!(top.best_bid_api.size, Some(1.2));
        assert_eq!(top.best_bid_interactive.price, Some(3000.4));
        assert_eq!(top.best_bid_interactive.size, Some(0.8));
        assert_eq!(top.best_ask_api.price, Some(3001.1));
        assert_eq!(top.best_ask_interactive.size, Some(0.6));
    }

    #[test]
    fn decode_interactive_public_top_and_snapshot_prefers_interactive_fields() {
        let value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@50ms",
                "data": {
                    "best_bid_api": ["3000.1", "1.1"],
                    "best_bid_interactive": ["3000.2", "0.7"],
                    "best_ask_api": ["3000.4", "1.3"],
                    "best_ask_interactive": ["3000.3", "0.5"],
                    "bids": [["3000.1", "1.1"]],
                    "asks": [["3000.4", "1.3"]]
                }
            }
        });

        let mut seq = 0u64;
        let (top, _, source) =
            decode_interactive_public_top_and_snapshot(&value, 0, "ETH-USD-PERP", &mut seq, None)
                .expect("interactive top snapshot");
        assert_eq!(source, "interactive_best");
        assert_eq!(top.best_bid_px, 3000.2);
        assert_eq!(top.best_bid_sz, 0.7);
        assert_eq!(top.best_ask_px, 3000.3);
        assert_eq!(top.best_ask_sz, 0.5);
    }

    #[test]
    fn decode_interactive_public_top_and_snapshot_falls_back_to_api_best() {
        let value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@50ms",
                "data": {
                    "best_bid_api": ["3000.1", "1.1"],
                    "best_ask_api": ["3000.4", "1.3"],
                    "bids": [["3000.0", "0.9"]],
                    "asks": [["3000.5", "1.0"]]
                }
            }
        });

        let mut seq = 0u64;
        let (top, _, source) =
            decode_interactive_public_top_and_snapshot(&value, 0, "ETH-USD-PERP", &mut seq, None)
                .expect("api fallback snapshot");
        assert_eq!(source, "api_best");
        assert_eq!(top.best_bid_px, 3000.1);
        assert_eq!(top.best_ask_px, 3000.4);
    }

    #[test]
    fn decode_interactive_public_top_and_snapshot_falls_back_to_book_levels() {
        let value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@50ms",
                "data": {
                    "bids": [["3000.0", "0.9"]],
                    "asks": [["3000.5", "1.0"]]
                }
            }
        });

        let mut seq = 0u64;
        let (top, _, source) =
            decode_interactive_public_top_and_snapshot(&value, 0, "ETH-USD-PERP", &mut seq, None)
                .expect("book fallback snapshot");
        assert_eq!(source, "book_levels");
        assert_eq!(top.best_bid_px, 3000.0);
        assert_eq!(top.best_ask_px, 3000.5);
    }

    #[test]
    fn decode_interactive_public_top_and_snapshot_applies_ui_touch_reference() {
        let value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@50ms",
                "data": {
                    "best_bid_api": ["3000.1", "1.1"],
                    "best_ask_api": ["3000.4", "1.3"],
                    "bids": [["3000.0", "0.9"]],
                    "asks": [["3000.5", "1.0"]]
                }
            }
        });
        let state = ParadexUiTouchReferenceState::default();
        state.update(
            ParadexUiBookTruthSource::Interactive,
            ParadexUiBookTruthSnapshot {
                best_bid_api: ParadexTopEntry {
                    price: Some(3000.1),
                    size: Some(1.1),
                },
                best_bid_interactive: ParadexTopEntry {
                    price: Some(3000.2),
                    size: Some(0.7),
                },
                best_ask_api: ParadexTopEntry {
                    price: Some(3000.4),
                    size: Some(1.3),
                },
                best_ask_interactive: ParadexTopEntry {
                    price: Some(3000.3),
                    size: Some(0.5),
                },
                ..ParadexUiBookTruthSnapshot::default()
            },
        );

        let mut seq = 0u64;
        let (top, _, source) = decode_interactive_public_top_and_snapshot(
            &value,
            0,
            "ETH-USD-PERP",
            &mut seq,
            Some(&state),
        )
        .expect("touch-adjusted interactive snapshot");
        assert_eq!(source, "api_best");
        assert_eq!(top.best_bid_px, 3000.2);
        assert_eq!(top.best_bid_sz, 0.7);
        assert_eq!(top.best_ask_px, 3000.3);
        assert_eq!(top.best_ask_sz, 0.5);
    }

    #[test]
    fn decode_interactive_public_top_and_snapshot_labels_top_level_fallback_touch() {
        let value = serde_json::json!({
            "params": {
                "channel": "order_book.ETH-USD-PERP.interactive@15@50ms",
                "data": {
                    "best_bid_api": ["3000.1", "1.1"],
                    "best_ask_api": ["3000.4", "1.3"],
                    "bids": [["3000.0", "0.9"]],
                    "asks": [["3000.5", "1.0"]]
                }
            }
        });
        let state = ParadexUiTouchReferenceState::default();
        state.update(
            ParadexUiBookTruthSource::Interactive,
            ParadexUiBookTruthSnapshot {
                bid: ParadexTopEntry {
                    price: Some(3000.2),
                    size: Some(0.7),
                },
                ask: ParadexTopEntry {
                    price: Some(3000.3),
                    size: Some(0.5),
                },
                best_bid_api: ParadexTopEntry {
                    price: Some(3000.1),
                    size: Some(1.1),
                },
                best_ask_api: ParadexTopEntry {
                    price: Some(3000.4),
                    size: Some(1.3),
                },
                ..ParadexUiBookTruthSnapshot::default()
            },
        );

        let mut seq = 0u64;
        let (top, _, source) = decode_interactive_public_top_and_snapshot(
            &value,
            0,
            "ETH-USD-PERP",
            &mut seq,
            Some(&state),
        )
        .expect("top-level touch-adjusted interactive snapshot");
        assert_eq!(source, "interactive_top_level_fallback");
        assert_eq!(top.best_bid_px, 3000.2);
        assert_eq!(top.best_bid_sz, 0.7);
        assert_eq!(top.best_ask_px, 3000.3);
        assert_eq!(top.best_ask_sz, 0.5);
    }

    #[test]
    fn paradex_public_feed_uses_ui_book_truth_poller_in_bbo_and_interactive_modes() {
        assert!(paradex_public_feed_uses_ui_book_truth_poller(
            &ParadexPublicFeedMode::Bbo
        ));
        assert!(paradex_public_feed_uses_ui_book_truth_poller(
            &ParadexPublicFeedMode::Orderbook {
                feed_type: "interactive".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
        ));
        assert!(!paradex_public_feed_uses_ui_book_truth_poller(
            &ParadexPublicFeedMode::Orderbook {
                feed_type: "snapshot".to_string(),
                refresh_rate: Some("100ms".to_string()),
            }
        ));
    }

    #[test]
    fn decode_ui_book_truth_snapshot_parses_api_and_interactive_fields() {
        let value = serde_json::json!({
            "results": {
                "seq_no": 77,
                "last_updated_at": 1_775_315_000_123i64,
                "bids": [["3000.1", "1.5"]],
                "asks": [["3000.3", "1.1"]],
                "best_bid_api": ["3000.1", "1.5"],
                "best_bid_interactive": ["3000.2", "0.7"],
                "best_ask_api": ["3000.3", "1.1"],
                "best_ask_interactive": ["3000.25", "0.5"]
            }
        });

        let snapshot = decode_ui_book_truth_snapshot(&value).expect("ui book truth");
        assert_eq!(snapshot.seq_no, Some(77));
        assert_eq!(snapshot.last_updated_at_ms, Some(1_775_315_000_123));
        assert_eq!(snapshot.bid.price, Some(3000.1));
        assert_eq!(snapshot.ask.size, Some(1.1));
        assert_eq!(snapshot.best_bid_interactive.price, Some(3000.2));
        assert_eq!(snapshot.best_ask_interactive.size, Some(0.5));
    }

    #[test]
    fn parse_fill_flags_from_body_collects_nested_fill_flags() {
        let flags = parse_fill_flags_from_body(
            r#"{"fills":[{"flags":["interactive","fastfill"]},{"flags":["rpi"]}]}"#,
        );
        assert_eq!(flags, vec!["fastfill", "interactive", "rpi"]);
    }

    #[test]
    fn paradex_stale_ms_falls_back_to_state_override_with_guardband() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, paradex_stale_ms);
        assert_eq!(stale_ms, 2600);
    }

    #[test]
    fn paradex_stale_ms_uses_larger_guardband_for_orderbook_feeds() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        overrides.insert(
            "PARAPHINA_PARADEX_PUBLIC_FEED".to_string(),
            "interactive".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, paradex_stale_ms);
        assert_eq!(stale_ms, 1800);
    }

    #[test]
    fn paradex_stale_ms_keeps_guardband_for_interactive_50ms_feed() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        overrides.insert(
            "PARAPHINA_PARADEX_PUBLIC_FEED".to_string(),
            "interactive_50ms".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, paradex_stale_ms);
        assert_eq!(stale_ms, 1800);
    }

    #[test]
    fn paradex_stale_ms_prefers_explicit_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert("PARAPHINA_PARADEX_STALE_MS".to_string(), "1800".to_string());
        overrides.insert(
            "PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, paradex_stale_ms);
        assert_eq!(stale_ms, 1800);
    }

    #[test]
    fn build_bbo_snapshot_emits_snapshot_with_fresh_timestamp() {
        let mut seq = 41u64;
        let (top, snapshot) =
            build_bbo_snapshot(4, "ETH-USD-PERP", &mut seq, 3000.1, 0.7, 3000.2, 0.8)
                .expect("snapshot");
        assert_eq!(seq, 42);
        assert_eq!(top.best_bid_px, 3000.1);
        assert_eq!(top.best_ask_px, 3000.2);
        match snapshot {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.seq, 42);
                assert_eq!(snapshot.venue_index, 4);
                assert_eq!(snapshot.venue_id, "ETH-USD-PERP");
                assert!(snapshot.timestamp_ms > 0);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn should_emit_bbo_quiet_refresh_requires_fresh_transport_and_quiet_book() {
        let now_ns = 10_000_000_000u64;
        assert!(should_emit_bbo_quiet_refresh(
            now_ns,
            6_000_000_000,
            8_500_000_000,
            1_000,
            5_000,
        ));
        assert!(!should_emit_bbo_quiet_refresh(
            now_ns,
            4_900_000_000,
            8_500_000_000,
            1_000,
            5_000,
        ));
        assert!(!should_emit_bbo_quiet_refresh(
            now_ns,
            6_000_000_000,
            9_400_000_000,
            1_000,
            5_000,
        ));
        assert!(!should_emit_bbo_quiet_refresh(
            now_ns,
            0,
            8_500_000_000,
            1_000,
            5_000,
        ));
    }

    #[test]
    fn paradex_apply_ui_touch_reference_uses_more_competitive_split_touch() {
        let snapshot = ParadexUiBookTruthSnapshot {
            best_bid_api: ParadexTopEntry {
                price: Some(3000.1),
                size: Some(1.0),
            },
            best_bid_interactive: ParadexTopEntry {
                price: Some(3000.2),
                size: Some(0.8),
            },
            best_ask_api: ParadexTopEntry {
                price: Some(3000.4),
                size: Some(1.2),
            },
            best_ask_interactive: ParadexTopEntry {
                price: Some(3000.35),
                size: Some(0.7),
            },
            ..ParadexUiBookTruthSnapshot::default()
        };

        let adjusted = paradex_apply_ui_touch_reference(3000.1, 0.5, 3000.4, 0.6, snapshot);

        assert_eq!(adjusted, (3000.2, 0.8, 3000.35, 0.7));
    }

    #[test]
    fn paradex_apply_ui_touch_reference_falls_back_when_split_fields_missing() {
        let snapshot = ParadexUiBookTruthSnapshot {
            best_bid_api: ParadexTopEntry {
                price: Some(3000.1),
                size: Some(1.0),
            },
            best_ask_api: ParadexTopEntry {
                price: Some(3000.4),
                size: Some(1.2),
            },
            ..ParadexUiBookTruthSnapshot::default()
        };

        let adjusted = paradex_apply_ui_touch_reference(3000.1, 0.5, 3000.4, 0.6, snapshot);

        assert_eq!(adjusted, (3000.1, 0.5, 3000.4, 0.6));
    }

    #[test]
    fn paradex_ui_book_truth_snapshot_normalizes_interactive_top_level_when_split_fields_missing() {
        let snapshot = ParadexUiBookTruthSnapshot {
            bid: ParadexTopEntry {
                price: Some(3000.2),
                size: Some(0.8),
            },
            ask: ParadexTopEntry {
                price: Some(3000.35),
                size: Some(0.7),
            },
            best_bid_api: ParadexTopEntry {
                price: Some(3000.1),
                size: Some(1.0),
            },
            best_ask_api: ParadexTopEntry {
                price: Some(3000.4),
                size: Some(1.2),
            },
            ..ParadexUiBookTruthSnapshot::default()
        };

        let normalized =
            snapshot.normalized_for_touch_reference(ParadexUiBookTruthSource::Interactive);

        assert_eq!(normalized.best_bid_interactive.price, Some(3000.2));
        assert_eq!(normalized.best_bid_interactive.size, Some(0.8));
        assert!(normalized.best_bid_interactive_from_top_level);
        assert_eq!(normalized.best_ask_interactive.price, Some(3000.35));
        assert_eq!(normalized.best_ask_interactive.size, Some(0.7));
        assert!(normalized.best_ask_interactive_from_top_level);
    }

    #[test]
    fn paradex_apply_ui_touch_reference_uses_normalized_top_level_fallback() {
        let snapshot = ParadexUiBookTruthSnapshot {
            bid: ParadexTopEntry {
                price: Some(3000.2),
                size: Some(0.8),
            },
            ask: ParadexTopEntry {
                price: Some(3000.35),
                size: Some(0.7),
            },
            best_bid_api: ParadexTopEntry {
                price: Some(3000.1),
                size: Some(1.0),
            },
            best_ask_api: ParadexTopEntry {
                price: Some(3000.4),
                size: Some(1.2),
            },
            ..ParadexUiBookTruthSnapshot::default()
        }
        .normalized_for_touch_reference(ParadexUiBookTruthSource::Interactive);

        let adjusted = paradex_apply_ui_touch_reference(3000.1, 0.5, 3000.4, 0.6, snapshot);

        assert_eq!(adjusted, (3000.2, 0.8, 3000.35, 0.7));
    }

    #[test]
    fn paradex_apply_ui_touch_reference_rejects_crossed_override() {
        let snapshot = ParadexUiBookTruthSnapshot {
            best_bid_api: ParadexTopEntry {
                price: Some(3000.1),
                size: Some(1.0),
            },
            best_bid_interactive: ParadexTopEntry {
                price: Some(3000.45),
                size: Some(0.8),
            },
            best_ask_api: ParadexTopEntry {
                price: Some(3000.4),
                size: Some(1.2),
            },
            best_ask_interactive: ParadexTopEntry {
                price: Some(3000.05),
                size: Some(0.7),
            },
            ..ParadexUiBookTruthSnapshot::default()
        };

        let adjusted = paradex_apply_ui_touch_reference(3000.1, 0.5, 3000.4, 0.6, snapshot);

        assert_eq!(adjusted, (3000.1, 0.5, 3000.4, 0.6));
    }

    #[tokio::test]
    async fn rest_place_order_builds_payload_and_auth_header() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: Some("python3 -c 'import json, os; payload = json.loads(os.environ[\"PARADEX_ORDER_PAYLOAD\"]); payload[\"signature\"] = \"[1,2]\"; payload[\"signature_timestamp\"] = 1700000000000; print(json.dumps(payload, separators=(\",\", \":\")))'".to_string()),
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/orders")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "market": "BTC-USD-PERP",
                        "side": "BUY",
                        "type": "LIMIT",
                        "instruction": "POST_ONLY",
                        "price": "100",
                        "size": "0.1",
                        "client_id": "co_post_only",
                        "signature": "[1,2]",
                        "signature_timestamp": 1700000000000i64,
                    }));
                then.status(200).body("{\"order_id\": \"oid_1\"}");
            })
            .await;

        let _ = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                side: Side::Buy,
                price: 100.0,
                size: 0.1,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_post_only".to_string(),
            })
            .await
            .expect("place order");

        mock.assert_async().await;
    }

    #[test]
    fn build_order_payload_serializes_decimal_fields_as_strings() {
        let payload = build_order_payload(
            "ETH-USD-PERP",
            &LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                side: Side::Buy,
                price: 2073.7899999999995,
                size: 0.010000000000000002,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_decimal".to_string(),
            },
        )
        .expect("payload");

        assert_eq!(
            payload.get("price").and_then(|v| v.as_str()),
            Some("2073.79")
        );
        assert_eq!(payload.get("size").and_then(|v| v.as_str()), Some("0.01"));
        assert_eq!(
            payload.get("instruction").and_then(|v| v.as_str()),
            Some("POST_ONLY")
        );
        assert_eq!(
            payload.get("client_id").and_then(|v| v.as_str()),
            Some("co_decimal")
        );
        assert!(
            payload.get("flags").is_none(),
            "reduce-only flags should be omitted when false"
        );
    }

    #[tokio::test]
    async fn rest_cancel_order_uses_client_id_endpoint_for_client_ids() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/by_client_id/co_123")
                    .query_param("market", "BTC-USD-PERP")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;

        let _ = client
            .cancel_order(LiveRestCancelRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "co_123".to_string(),
            })
            .await
            .expect("cancel order");

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn rest_cancel_batch_uses_native_batch_endpoint_for_multiple_requests() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "order_ids": ["pdx_123", "pdx_456"],
                    }));
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "pdx_123", "status": "CANCELLED"},
                        {"id": "pdx_456", "status": "QUEUED_FOR_CANCELLATION"},
                    ]
                }));
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_456".to_string(),
                },
            ])
            .await;

        batch_mock.assert_async().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.is_ok()));
    }

    #[tokio::test]
    async fn rest_cancel_batch_canonicalizes_cached_client_ids_to_exchange_ids() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        client.cache_replace_identity(Some("co_456"), Some("pdx_456"));

        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "order_ids": ["pdx_123", "pdx_456"],
                    }));
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "pdx_123", "status": "CANCELLED"},
                        {"id": "pdx_456", "status": "QUEUED_FOR_CANCELLATION"},
                    ]
                }));
            })
            .await;
        let client_id_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/by_client_id/co_456")
                    .query_param("market", "BTC-USD-PERP");
                then.status(500);
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_456".to_string(),
                },
            ])
            .await;

        batch_mock.assert_async().await;
        client_id_mock.assert_hits_async(0).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.is_ok()));
        assert_eq!(
            results[1]
                .as_ref()
                .expect("canonicalized client id result")
                .client_order_id
                .as_deref(),
            Some("co_456")
        );
    }

    #[tokio::test]
    async fn rest_cancel_batch_resolves_uncached_client_ids_via_rest_for_native_batch() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let resolve_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_456")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "id": "pdx_456",
                    "client_id": "co_456",
                    "market": "BTC-USD-PERP",
                    "status": "OPEN",
                }));
            })
            .await;
        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "order_ids": ["pdx_123", "pdx_456"],
                    }));
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "pdx_123", "status": "CANCELLED"},
                        {"id": "pdx_456", "status": "QUEUED_FOR_CANCELLATION"},
                    ]
                }));
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_456".to_string(),
                },
            ])
            .await;

        resolve_mock.assert_async().await;
        batch_mock.assert_async().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.is_ok()));
    }

    #[tokio::test]
    async fn rest_cancel_batch_unresolved_client_ids_stay_on_serial_path() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let resolve_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_456")
                    .header("Authorization", "Bearer test.jwt");
                then.status(404)
                    .body("{\"message\":\"CLIENT_ORDER_ID_NOT_FOUND\"}");
            })
            .await;
        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE).path("/orders/batch");
                then.status(500);
            })
            .await;
        let order_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/pdx_123")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;
        let client_id_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/by_client_id/co_456")
                    .query_param("market", "BTC-USD-PERP")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_456".to_string(),
                },
            ])
            .await;

        resolve_mock.assert_async().await;
        batch_mock.assert_hits_async(0).await;
        order_mock.assert_async().await;
        client_id_mock.assert_async().await;
        assert!(results.iter().all(|result| result.is_ok()));
    }

    #[tokio::test]
    async fn rest_cancel_batch_keeps_single_request_on_single_cancel_path() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let single_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/by_client_id/co_single")
                    .query_param("market", "BTC-USD-PERP")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;
        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE).path("/orders/batch");
                then.status(500);
            })
            .await;

        let results = client
            .cancel_batch(vec![LiveRestCancelRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "co_single".to_string(),
            }])
            .await;

        single_mock.assert_async().await;
        batch_mock.assert_hits_async(0).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[tokio::test]
    async fn rest_cancel_batch_treats_success_equivalent_statuses_as_ok() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let resolve_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_c")
                    .header("Authorization", "Bearer test.jwt");
                then.status(404)
                    .body("{\"message\":\"CLIENT_ORDER_ID_NOT_FOUND\"}");
            })
            .await;
        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "pdx_a", "status": "QUEUED_FOR_CANCELLATION"},
                        {"id": "pdx_b", "status": "ALREADY_CLOSED"},
                    ]
                }));
            })
            .await;
        let client_id_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/by_client_id/co_c")
                    .query_param("market", "BTC-USD-PERP")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_a".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_b".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_c".to_string(),
                },
            ])
            .await;

        resolve_mock.assert_async().await;
        batch_mock.assert_async().await;
        client_id_mock.assert_async().await;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|result| result.is_ok()));
    }

    #[tokio::test]
    async fn rest_cancel_batch_missing_result_is_retryable_error() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        client.cache_replace_identity(Some("co_missing"), Some("pdx_missing"));

        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "pdx_123", "status": "CANCELLED"}
                    ]
                }));
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_missing".to_string(),
                },
            ])
            .await;

        batch_mock.assert_async().await;
        assert!(results[0].is_ok());
        let err = results[1]
            .as_ref()
            .expect_err("missing result should error");
        assert_eq!(err.kind, LiveGatewayErrorKind::Retryable);
        assert!(err.message.contains("missing result"));
    }

    #[tokio::test]
    async fn rest_cancel_batch_falls_back_to_serial_on_http_error() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        client.cache_replace_identity(Some("co_456"), Some("pdx_456"));

        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt");
                then.status(500).body("{\"message\":\"temporary\"}");
            })
            .await;
        let order_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/pdx_123")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;
        let client_id_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/pdx_456")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).body("{}");
            })
            .await;

        let results = client
            .cancel_batch(vec![
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "pdx_123".to_string(),
                },
                LiveRestCancelRequest {
                    venue_index: 0,
                    venue_id: "paradex".to_string(),
                    order_id: "co_456".to_string(),
                },
            ])
            .await;

        batch_mock.assert_async().await;
        order_mock.assert_async().await;
        client_id_mock.assert_async().await;
        assert!(results.iter().all(|result| result.is_ok()));
    }

    #[tokio::test]
    async fn rest_replace_order_uses_modify_endpoint_and_preserves_ids() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let mock = server
            .mock_async(|when, then| {
                when.method(PUT)
                    .path("/orders/pdx_123")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "id": "pdx_123",
                        "market": "BTC-USD-PERP",
                        "side": "BUY",
                        "type": "LIMIT",
                        "instruction": "POST_ONLY",
                        "price": "101",
                        "size": "0.2",
                        "client_id": "co_pdx_new",
                    }));
                then.status(200)
                    .body("{\"id\":\"pdx_123\",\"client_id\":\"co_pdx_new\"}");
            })
            .await;

        let response = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "pdx_123".to_string(),
                side: Side::Buy,
                price: 101.0,
                size: 0.2,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_pdx_new".to_string(),
            })
            .await
            .expect("replace order");

        mock.assert_async().await;
        assert_eq!(response.order_id.as_deref(), Some("pdx_123"));
        assert_eq!(response.client_order_id.as_deref(), Some("co_pdx_new"));
    }

    #[tokio::test]
    async fn rest_replace_order_resolves_client_id_via_cache() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        client.cache_replace_identity(Some("co_cached"), Some("pdx_cached"));

        let modify_mock = server
            .mock_async(|when, then| {
                when.method(PUT)
                    .path("/orders/pdx_cached")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "id": "pdx_cached",
                        "market": "BTC-USD-PERP",
                        "side": "BUY",
                        "type": "LIMIT",
                        "instruction": "POST_ONLY",
                        "price": "101",
                        "size": "0.2",
                        "client_id": "co_pdx_new",
                    }));
                then.status(200)
                    .body("{\"id\":\"pdx_cached\",\"client_id\":\"co_pdx_new\"}");
            })
            .await;
        let lookup_mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/orders/by_client_id/co_cached");
                then.status(500);
            })
            .await;

        let response = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "co_cached".to_string(),
                side: Side::Buy,
                price: 101.0,
                size: 0.2,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_pdx_new".to_string(),
            })
            .await
            .expect("replace order");

        modify_mock.assert_async().await;
        lookup_mock.assert_hits_async(0).await;
        assert_eq!(response.order_id.as_deref(), Some("pdx_cached"));
        assert_eq!(response.client_order_id.as_deref(), Some("co_pdx_new"));
    }

    #[tokio::test]
    async fn rest_replace_order_resolves_client_id_via_get_by_client_id() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let lookup_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_lookup")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "id": "pdx_lookup",
                    "client_id": "co_lookup",
                    "market": "BTC-USD-PERP",
                    "status": "OPEN"
                }));
            })
            .await;
        let modify_mock = server
            .mock_async(|when, then| {
                when.method(PUT)
                    .path("/orders/pdx_lookup")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "id": "pdx_lookup",
                        "market": "BTC-USD-PERP",
                        "side": "BUY",
                        "type": "LIMIT",
                        "instruction": "POST_ONLY",
                        "price": "101",
                        "size": "0.2",
                        "client_id": "co_pdx_new",
                    }));
                then.status(200)
                    .body("{\"id\":\"pdx_lookup\",\"client_id\":\"co_pdx_new\"}");
            })
            .await;

        let response = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "co_lookup".to_string(),
                side: Side::Buy,
                price: 101.0,
                size: 0.2,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_pdx_new".to_string(),
            })
            .await
            .expect("replace order");

        lookup_mock.assert_async().await;
        modify_mock.assert_async().await;
        assert_eq!(response.order_id.as_deref(), Some("pdx_lookup"));
        assert_eq!(response.client_order_id.as_deref(), Some("co_pdx_new"));
        assert_eq!(
            client.cached_replace_identity("co_lookup").as_deref(),
            Some("pdx_lookup")
        );
    }

    #[tokio::test]
    async fn rest_replace_order_returns_retryable_when_client_id_cannot_be_resolved() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let lookup_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_missing")
                    .header("Authorization", "Bearer test.jwt");
                then.status(404)
                    .body("{\"error\":\"CLIENT_ORDER_ID_NOT_FOUND\"}");
            })
            .await;

        let err = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "paradex".to_string(),
                order_id: "co_missing".to_string(),
                side: Side::Buy,
                price: 101.0,
                size: 0.2,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_pdx_new".to_string(),
            })
            .await
            .expect_err("missing client id should fail");

        lookup_mock.assert_async().await;
        assert_eq!(err.kind, LiveGatewayErrorKind::Retryable);
        assert!(err.message.contains("reason=not_found"));
    }

    #[tokio::test]
    async fn fetch_open_order_snapshot_primes_replace_identity_cache() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "BTC-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 0,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let snapshot_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {
                            "id": "pdx_open_1",
                            "client_id": "co_open_1",
                            "market": "BTC-USD-PERP",
                            "status": "OPEN",
                            "side": "BUY",
                            "price": "100",
                            "remaining_size": "0.1"
                        }
                    ]
                }));
            })
            .await;

        let snapshot = client
            .fetch_open_order_snapshot("paradex", 0)
            .await
            .expect("open orders snapshot");

        snapshot_mock.assert_async().await;
        assert_eq!(snapshot.open_orders.len(), 1);
        assert_eq!(
            client.cached_replace_identity("co_open_1").as_deref(),
            Some("pdx_open_1")
        );
    }

    #[tokio::test]
    async fn fetch_ui_book_truth_snapshot_uses_authed_orderbook_paths() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Interactive,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let api_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orderbook/ETH-USD-PERP")
                    .query_param("depth", "1")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": {
                        "seq_no": 90,
                        "last_updated_at": 1_700_000_001_000i64,
                        "bids": [["3000.1", "1.5"]],
                        "asks": [["3000.3", "1.1"]]
                    }
                }));
            })
            .await;
        let interactive_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orderbook/ETH-USD-PERP/interactive")
                    .query_param("depth", "1")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": {
                        "seq_no": 91,
                        "last_updated_at": 1_700_000_001_500i64,
                        "bids": [["3000.2", "0.8"]],
                        "asks": [["3000.25", "0.5"]],
                        "best_bid_api": ["3000.1", "1.5"],
                        "best_bid_interactive": ["3000.2", "0.8"],
                        "best_ask_api": ["3000.3", "1.1"],
                        "best_ask_interactive": ["3000.25", "0.5"]
                    }
                }));
            })
            .await;

        let api_snapshot =
            fetch_paradex_ui_book_truth_snapshot(&client, ParadexUiBookTruthSource::Api)
                .await
                .expect("api ui truth");
        let interactive_snapshot =
            fetch_paradex_ui_book_truth_snapshot(&client, ParadexUiBookTruthSource::Interactive)
                .await
                .expect("interactive ui truth");

        api_mock.assert_async().await;
        interactive_mock.assert_async().await;
        assert_eq!(api_snapshot.bid.price, Some(3000.1));
        assert_eq!(
            interactive_snapshot.best_bid_interactive.price,
            Some(3000.2)
        );
        assert_eq!(interactive_snapshot.best_ask_api.size, Some(1.1));
    }

    #[tokio::test]
    async fn fetch_account_snapshot_enriches_positions_from_positions_endpoint() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let account_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/account")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "account": "0xabc",
                    "initial_margin_requirement": "45.2",
                    "maintenance_margin_requirement": "22.8",
                    "account_value": "79.79",
                    "total_collateral": "83.31",
                    "free_collateral": "34.59",
                    "settlement_asset": "USDC",
                    "updated_at": 1_700_000_000_123i64,
                    "seq_no": 42
                }));
            })
            .await;
        let positions_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/positions")
                    .query_param("market", "ETH-USD-PERP")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": [{
                        "market": "ETH-USD-PERP",
                        "side": "SHORT",
                        "size": "-1.07",
                        "average_entry_price": "2088.21759223"
                    }]
                }));
            })
            .await;

        let snapshot = client
            .fetch_account_snapshot("PARADEX", 4)
            .await
            .expect("snapshot");

        account_mock.assert_async().await;
        positions_mock.assert_async().await;
        assert_eq!(snapshot.positions.len(), 1);
        assert_eq!(snapshot.positions[0].symbol, "ETH-USD-PERP");
        assert!((snapshot.positions[0].size + 1.07).abs() < 1e-9);
        assert!((snapshot.positions[0].entry_price - 2088.21759223).abs() < 1e-9);
    }

    #[tokio::test]
    async fn rest_cancel_all_falls_back_to_batch_when_legacy_endpoint_404s() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Pro,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);

        let legacy_mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/orders/cancel_all")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({"market": "ETH-USD-PERP"}));
                then.status(404).body("{\"message\":\"Not Found\"}");
            })
            .await;
        let list_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        { "id": "oid_1", "market": "ETH-USD-PERP", "status": "OPEN" },
                        { "id": "oid_2", "market": "ETH-USD-PERP", "status": "OPEN" },
                        { "id": "oid_3", "market": "BTC-USD-PERP", "status": "OPEN" },
                        { "id": "oid_4", "market": "ETH-USD-PERP", "status": "FILLED" }
                    ]
                }));
            })
            .await;
        let batch_mock = server
            .mock_async(|when, then| {
                when.method(DELETE)
                    .path("/orders/batch")
                    .header("Authorization", "Bearer test.jwt")
                    .json_body(serde_json::json!({
                        "order_ids": ["oid_1", "oid_2"]
                    }));
                then.status(200).json_body(serde_json::json!({
                    "results": [
                        {"id": "oid_1", "status": "CANCELLED"},
                        {"id": "oid_2", "status": "CANCELLED"}
                    ]
                }));
            })
            .await;

        let _ = client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 4,
                venue_id: "paradex".to_string(),
            })
            .await
            .expect("cancel all");

        legacy_mock.assert_async().await;
        list_mock.assert_async().await;
        batch_mock.assert_async().await;
    }

    #[test]
    fn parse_order_id_prefers_exchange_id_before_client_id() {
        let body = serde_json::json!({
            "id": "pdx_order_123",
            "client_id": "co_123",
        })
        .to_string();
        assert_eq!(parse_order_id(&body).as_deref(), Some("pdx_order_123"));
    }

    #[test]
    fn parse_open_orders_keeps_remaining_size_and_client_order_id() {
        let value = serde_json::json!({
            "results": [
                {
                    "id": "pdx_order_123",
                    "client_id": "co_pdx_1",
                    "market": "ETH-USD-PERP",
                    "status": "OPEN",
                    "side": "SELL",
                    "price": "2101.25",
                    "remaining_size": "0.07"
                },
                {
                    "id": "ignore_me",
                    "market": "BTC-USD-PERP",
                    "status": "OPEN",
                    "side": "BUY",
                    "price": "60000",
                    "remaining_size": "1.0"
                }
            ]
        });

        let open_orders = parse_open_orders(&value, "ETH-USD-PERP");

        assert_eq!(open_orders.len(), 1);
        assert_eq!(open_orders[0].order_id, "pdx_order_123");
        assert_eq!(open_orders[0].client_order_id.as_deref(), Some("co_pdx_1"));
        assert_eq!(
            open_orders[0].exchange_order_id.as_deref(),
            Some("pdx_order_123")
        );
        assert_eq!(open_orders[0].side, Side::Sell);
        assert!((open_orders[0].price - 2101.25).abs() < 1e-9);
        assert!((open_orders[0].size - 0.07).abs() < 1e-9);
        assert_eq!(open_orders[0].purpose, None);
    }

    #[tokio::test]
    async fn normalize_open_snapshot_replace_identities_uses_cache() {
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            auth_url: "https://example.invalid/auth".to_string(),
            token_usage: ParadexTokenUsage::Interactive,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        client.cache_replace_identity(Some("co_cached"), Some("pdx_cached"));
        let mut open_orders = vec![OpenOrderSnapshot {
            order_id: "co_cached".to_string(),
            client_order_id: Some("co_cached".to_string()),
            exchange_order_id: None,
            side: Side::Buy,
            price: 2000.0,
            size: 0.05,
            purpose: None,
        }];

        client
            .normalize_open_snapshot_replace_identities(&mut open_orders)
            .await;

        assert_eq!(
            open_orders[0].exchange_order_id.as_deref(),
            Some("pdx_cached")
        );
    }

    #[tokio::test]
    async fn normalize_open_snapshot_replace_identities_resolves_via_rest() {
        let server = MockServer::start_async().await;
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: server.base_url(),
            auth_url: format!("{}/auth", server.base_url()),
            token_usage: ParadexTokenUsage::Interactive,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        let resolve_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/orders/by_client_id/co_rest")
                    .header("Authorization", "Bearer test.jwt");
                then.status(200).json_body(serde_json::json!({
                    "id": "pdx_rest",
                    "client_id": "co_rest",
                    "market": "ETH-USD-PERP",
                    "status": "OPEN",
                }));
            })
            .await;
        let mut open_orders = vec![OpenOrderSnapshot {
            order_id: "co_rest".to_string(),
            client_order_id: Some("co_rest".to_string()),
            exchange_order_id: None,
            side: Side::Buy,
            price: 2001.0,
            size: 0.03,
            purpose: None,
        }];

        client
            .normalize_open_snapshot_replace_identities(&mut open_orders)
            .await;

        resolve_mock.assert_async().await;
        assert_eq!(
            open_orders[0].exchange_order_id.as_deref(),
            Some("pdx_rest")
        );
    }

    #[test]
    fn private_order_state_applies_ws_updates_and_removes_closed_orders() {
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            auth_url: "https://example.invalid/auth".to_string(),
            token_usage: ParadexTokenUsage::Interactive,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: Some("test.jwt".to_string()),
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };
        let client = ParadexRestClient::new(cfg);
        let snapshot = OrderSnapshot {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            seq: 10,
            timestamp_ms: 1_700_000_000_000,
            open_orders: vec![OpenOrderSnapshot {
                order_id: "pdx_order_1".to_string(),
                client_order_id: Some("co_pdx_1".to_string()),
                exchange_order_id: Some("pdx_order_1".to_string()),
                side: Side::Buy,
                price: 2000.0,
                size: 0.05,
                purpose: None,
            }],
        };
        let mut state = ParadexPrivateOrderState::from_snapshot("ETH-USD-PERP", snapshot);
        let mut seq_state = ParadexPrivateSeqState::default();

        let open_message = serde_json::json!({
            "params": {
                "channel": "orders.ETH-USD-PERP",
                "data": {
                    "id": "pdx_order_1",
                    "client_id": "co_pdx_1",
                    "market": "ETH-USD-PERP",
                    "status": "OPEN",
                    "side": "BUY",
                    "price": "2001.5",
                    "remaining_size": "0.04",
                    "seq_no": 11,
                    "timestamp": 1_700_000_000_111i64
                }
            }
        });
        let snapshot = state
            .apply_subscription_message(&open_message, &mut seq_state, &client)
            .expect("updated snapshot");
        assert_eq!(snapshot.seq, 11);
        assert_eq!(snapshot.open_orders.len(), 1);
        assert!((snapshot.open_orders[0].price - 2001.5).abs() < 1e-9);
        assert!((snapshot.open_orders[0].size - 0.04).abs() < 1e-9);
        assert_eq!(
            client.cached_replace_identity("co_pdx_1").as_deref(),
            Some("pdx_order_1")
        );

        let closed_message = serde_json::json!({
            "params": {
                "channel": "orders.ETH-USD-PERP",
                "data": {
                    "id": "pdx_order_1",
                    "client_id": "co_pdx_1",
                    "market": "ETH-USD-PERP",
                    "status": "CLOSED",
                    "seq_no": 12,
                    "timestamp": 1_700_000_000_222i64
                }
            }
        });
        let snapshot = state
            .apply_subscription_message(&closed_message, &mut seq_state, &client)
            .expect("closed snapshot");
        assert_eq!(snapshot.seq, 12);
        assert!(snapshot.open_orders.is_empty());
    }

    #[test]
    fn private_fill_messages_exact_liquidity_create_source_owner_fills() {
        let message = serde_json::json!({
            "params": {
                "channel": "fills.ETH-USD-PERP",
                "data": [
                    {
                        "market": "ETH-USD-PERP",
                        "fill_type": "FILL",
                        "liquidity": "MAKER",
                        "size": "0.01",
                        "created_at": 1_700_000_000_333i64,
                        "order_id": "pdx_order_1",
                        "client_id": "co_pdx_1"
                    },
                    {
                        "market": "ETH-USD-PERP",
                        "fill_type": "FILL",
                        "liquidity": "taker",
                        "size": "0.02",
                        "created_at": 1_700_000_000_333i64,
                        "orderId": "pdx_order_2",
                        "clientOrderId": "co_pdx_2"
                    }
                ]
            }
        });

        let fills = phase51_paradex_source_owner_fills_from_subscription_message(
            &message,
            4,
            "paradex",
            "ETH-USD-PERP",
        );

        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].venue_id, "paradex");
        assert_eq!(fills[0].order_id(), Some("pdx_order_1"));
        assert_eq!(fills[0].client_order_id(), Some("co_pdx_1"));
        assert_eq!(fills[0].seq, 1_700_000_000_333_000_000);
        assert_eq!(fills[0].timestamp_ms, 1_700_000_000_333i64);
        assert_eq!(fills[0].phase51_lighter_native_limit, None);
        assert_eq!(
            fills[0].phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Paradex {
                liquidity: "MAKER".to_string()
            })
        );
        assert_eq!(fills[1].order_id(), Some("pdx_order_2"));
        assert_eq!(fills[1].client_order_id(), Some("co_pdx_2"));
        assert_eq!(fills[1].seq, 1_700_000_000_333_000_001);
        assert_eq!(
            fills[1].phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Paradex {
                liquidity: "TAKER".to_string()
            })
        );
    }

    #[test]
    fn private_fill_message_missing_or_invalid_liquidity_creates_no_source_owner_fill() {
        for fill in [
            serde_json::json!({
                "market": "ETH-USD-PERP",
                "fill_type": "FILL",
                "size": "0.01",
                "created_at": 1_700_000_000_333i64,
                "order_id": "pdx_order_1"
            }),
            serde_json::json!({
                "market": "ETH-USD-PERP",
                "fill_type": "FILL",
                "liquidity": "UNKNOWN",
                "size": "0.01",
                "created_at": 1_700_000_000_333i64,
                "order_id": "pdx_order_1"
            }),
        ] {
            let message = serde_json::json!({
                "params": {
                    "channel": "fills.ETH-USD-PERP",
                    "data": fill
                }
            });

            assert!(
                phase51_paradex_source_owner_fills_from_subscription_message(
                    &message,
                    4,
                    "paradex",
                    "ETH-USD-PERP",
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn private_fill_message_wrong_market_non_fill_or_without_handle_creates_no_source_owner_fill() {
        for fill in [
            serde_json::json!({
                "market": "BTC-USD-PERP",
                "fill_type": "FILL",
                "liquidity": "MAKER",
                "size": "0.01",
                "created_at": 1_700_000_000_333i64,
                "order_id": "pdx_order_1"
            }),
            serde_json::json!({
                "market": "ETH-USD-PERP",
                "fill_type": "LIQUIDATION",
                "liquidity": "MAKER",
                "size": "0.01",
                "created_at": 1_700_000_000_333i64,
                "order_id": "pdx_order_1"
            }),
            serde_json::json!({
                "market": "ETH-USD-PERP",
                "fill_type": "FILL",
                "liquidity": "MAKER",
                "size": "0",
                "created_at": 1_700_000_000_333i64,
                "order_id": "pdx_order_1"
            }),
            serde_json::json!({
                "market": "ETH-USD-PERP",
                "fill_type": "FILL",
                "liquidity": "MAKER",
                "size": "0.01",
                "created_at": 1_700_000_000_333i64,
                "id": "fill_id_not_order_id"
            }),
        ] {
            let message = serde_json::json!({
                "params": {
                    "channel": "fills.ETH-USD-PERP",
                    "data": fill
                }
            });

            assert!(
                phase51_paradex_source_owner_fills_from_subscription_message(
                    &message,
                    4,
                    "paradex",
                    "ETH-USD-PERP",
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn private_order_messages_do_not_create_source_owner_fills() {
        let message = serde_json::json!({
            "params": {
                "channel": "orders.ETH-USD-PERP",
                "data": {
                    "market": "ETH-USD-PERP",
                    "status": "FILLED",
                    "id": "pdx_order_1",
                    "client_id": "co_pdx_1",
                    "liquidity": "MAKER"
                }
            }
        });

        assert!(
            phase51_paradex_source_owner_fills_from_subscription_message(
                &message,
                4,
                "paradex",
                "ETH-USD-PERP",
            )
            .is_empty()
        );
    }

    #[test]
    fn private_seq_state_ignores_non_monotonic_updates() {
        let mut seq_state = ParadexPrivateSeqState::default();
        assert!(seq_state.accept(10));
        assert!(!seq_state.accept(10));
        assert!(!seq_state.accept(9));
        assert!(seq_state.accept(11));
    }

    #[test]
    fn freshness_reset_and_anchor_behavior() {
        let freshness = Freshness::default();
        freshness.last_parsed_ns.store(123, Ordering::Relaxed);
        freshness.last_published_ns.store(456, Ordering::Relaxed);
        freshness.last_book_event_ns.store(789, Ordering::Relaxed);
        freshness.reset_for_new_connection();
        assert_eq!(freshness.last_parsed_ns.load(Ordering::Relaxed), 0);
        assert_eq!(freshness.last_published_ns.load(Ordering::Relaxed), 0);
        assert_eq!(
            freshness.last_book_event_ns.load(Ordering::Relaxed),
            0,
            "last_book_event_ns must be reset on new connection"
        );

        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        // Non-book parsed events must NOT advance watchdog anchor
        freshness.last_parsed_ns.store(2_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(
            anchor, connect_start_ns,
            "non-book parsed events must not advance watchdog anchor"
        );

        freshness.last_book_event_ns.store(3_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 3_000);

        freshness.last_published_ns.store(4_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 4_000);
    }

    #[test]
    fn parse_public_funding_markets_summary_fixture() {
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/paradex/public_markets_summary.json"
        );
        let raw = std::fs::read_to_string(fixture_path).expect("fixture exists");
        let value: Value = serde_json::from_str(&raw).expect("valid json");

        // Test parsing ETH-USD-PERP from "results" array
        let cfg = ParadexConfig {
            ws_url: "wss://example.invalid".to_string(),
            rest_url: "https://example.invalid".to_string(),
            auth_url: "https://example.invalid/auth".to_string(),
            token_usage: ParadexTokenUsage::Pro,
            market: "ETH-USD-PERP".to_string(),
            account_path: "/account".to_string(),
            order_path: "/orders".to_string(),
            venue_index: 4,
            jwt: None,
            jwt_cmd: None,
            sign_order_cmd: None,
            auth_payload_json: None,
            token_refresh_secs: 240,
            record_dir: None,
        };

        let update = parse_public_funding(&value, &cfg).expect("parse funding ok");
        assert_eq!(update.venue_index, 4);

        // funding_rate = "-0.00011283359389" (string) should parse correctly
        let rate_native = update.funding_rate_native.expect("rate_native present");
        assert!(
            (rate_native - (-0.00011283359389)).abs() < 1e-14,
            "rate_native mismatch"
        );

        // Sign must be preserved (negative)
        assert!(
            rate_native < 0.0,
            "negative funding rate sign must be preserved"
        );

        // rate_8h should equal rate_native (Paradex quotes 8h-equivalent rates, interval=28800s)
        let rate_8h = update.funding_rate_8h.expect("rate_8h present");
        assert!(
            (rate_8h - rate_native).abs() < 1e-14,
            "rate_8h should equal rate_native"
        );

        // Fix: Paradex interval_sec must be 28800 (8h-equivalent)
        assert_eq!(
            update.interval_sec,
            Some(28_800),
            "Paradex interval_sec must be 28800"
        );

        // Fix C: Paradex settles via USDC-oracle-adjusted mechanism.
        assert_eq!(
            update.settlement_price_kind,
            Some(SettlementPriceKind::UsdcOracleAdjusted),
            "Paradex settlement must be UsdcOracleAdjusted"
        );

        // Timestamp should come from "created_at"
        assert_eq!(update.timestamp_ms, 1770309699810);

        // Test BTC-USD-PERP (positive rate)
        let cfg_btc = ParadexConfig {
            market: "BTC-USD-PERP".to_string(),
            ..cfg.clone()
        };
        let update_btc = parse_public_funding(&value, &cfg_btc).expect("parse btc ok");
        let rate_btc = update_btc.funding_rate_native.expect("btc rate present");
        assert!(
            rate_btc > 0.0,
            "positive funding rate sign must be preserved"
        );
        assert!((rate_btc - 0.00005672).abs() < 1e-14);

        // Test non-existent market returns None (should fall back to root value which has no funding_rate)
        let cfg_nonexistent = ParadexConfig {
            market: "NOSUCH-PERP".to_string(),
            ..cfg
        };
        let result = parse_public_funding(&value, &cfg_nonexistent);
        // Since no match found, it falls back to `value` itself which has no funding_rate
        // So funding_rate_native and funding_rate_8h should be None
        let update_none = result.expect("returns Some but with None rates");
        assert!(update_none.funding_rate_native.is_none());
        assert!(update_none.funding_rate_8h.is_none());
    }

    #[test]
    fn account_summary_snapshot_parses() {
        let value = serde_json::json!({
            "account": "0xabc",
            "initial_margin_requirement": "12.5",
            "maintenance_margin_requirement": "6.0",
            "account_value": "100.0",
            "total_collateral": "100.0",
            "free_collateral": "87.5",
            "margin_cushion": "87.5",
            "settlement_asset": "USDC",
            "updated_at": 1_700_000_000_123i64,
            "status": "ACTIVE",
            "seq_no": 42
        });
        let snapshot = parse_account_snapshot(&value, "PARADEX", 4).expect("snapshot");
        assert_eq!(snapshot.venue_index, 4);
        assert_eq!(snapshot.venue_id, "PARADEX");
        assert_eq!(snapshot.seq, 42);
        assert_eq!(snapshot.timestamp_ms, 1_700_000_000_123);
        assert!(snapshot.positions.is_empty());
        assert_eq!(snapshot.balances.len(), 1);
        assert_eq!(snapshot.balances[0].asset, "USDC");
        assert!((snapshot.margin.balance_usd - 100.0).abs() < 1e-9);
        assert!((snapshot.margin.used_usd - 12.5).abs() < 1e-9);
        assert!((snapshot.margin.available_usd - 87.5).abs() < 1e-9);
    }
}
