//! Extended connector (public WS market data + fixtures, feature-gated).

#[cfg(feature = "live_extended")]
pub const STUB_CONNECTOR: bool = false;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_MARKET: bool = true;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_ACCOUNT: bool = true;
#[cfg(feature = "live_extended")]
pub const SUPPORTS_EXECUTION: bool = true;

const EXTENDED_STALE_MS_DEFAULT: u64 = 10_000;
const EXTENDED_TRANSPORT_STALE_MS_DEFAULT: u64 = 25_000;
const EXTENDED_WATCHDOG_TICK_MS: u64 = 200;
const EXTENDED_STATE_STALE_GUARDBAND_MS: u64 = 500;
const EXTENDED_FIRST_DATA_TIMEOUT_GUARDBAND_MS: u64 = 400;
const EXTENDED_CONTROL_FRAME_ONLY_TIMEOUT_GUARDBAND_MS: u64 = 50;
const EXTENDED_CONTROL_FRAME_ONLY_HEDGE_START_GUARDBAND_MS: u64 = 100;
const EXTENDED_CONTROL_FRAME_ONLY_HEDGE_START_FLOOR_MS: u64 = 750;
const EXTENDED_POST_PUBLISH_FALLBACK_AFTER_GUARDBAND_MS: u64 = 300;
const EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_GUARDBAND_MS: u64 = 50;
const EXTENDED_WS_READ_TIMEOUT_MS_DEFAULT: u64 = 10_000;
const EXTENDED_PRIVATE_WS_READ_TIMEOUT_MS_DEFAULT: u64 = 35_000;
const EXTENDED_CONNECT_FIRST_FRAME_TIMEOUT_MS_DEFAULT: u64 = 1_500;
const EXTENDED_CONNECT_CONTROL_FRAME_ONLY_TIMEOUT_MS_DEFAULT: u64 = 1_450;
const EXTENDED_CONNECT_BOOK_TIMEOUT_MS_DEFAULT: u64 = 750;
const EXTENDED_FAST_RECONNECT_SLEEP_MS: u64 = 100;
const EXTENDED_STALE_CHURN_WINDOW_MS_DEFAULT: u64 = 120_000;
const EXTENDED_STALE_CHURN_LIMIT_DEFAULT: usize = 2;
const EXTENDED_STALE_CHURN_HEALTHY_RESET_MS_DEFAULT: u64 = 30_000;
const EXTENDED_BOOTSTRAP_CHURN_WINDOW_MS_DEFAULT: u64 = 120_000;
const EXTENDED_BOOTSTRAP_CHURN_LIMIT_DEFAULT: usize = 2;
const EXTENDED_BOOTSTRAP_CHURN_HEALTHY_RESET_MS_DEFAULT: u64 = 30_000;
const EXTENDED_MARKET_PUB_QUEUE_CAP_LIVE: usize = 256;
const EXTENDED_MARKET_PUB_QUEUE_CAP_FIXTURE: usize = 4096;
const EXTENDED_MARKET_PUB_DRAIN_MAX: usize = 64;

static MONO_START: OnceLock<Instant> = OnceLock::new();
static EXTENDED_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static EXTENDED_RECONNECT_COUNTS: OnceLock<StdMutex<BTreeMap<&'static str, u64>>> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn extended_state_stale_ms_override() -> Option<u64> {
    std::env::var("PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
}

fn extended_stale_ms() -> u64 {
    if let Some(explicit) = std::env::var("PARAPHINA_EXTENDED_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return explicit;
    }
    if let Some(state_override_ms) = extended_state_stale_ms_override() {
        return state_override_ms
            .saturating_sub(EXTENDED_STATE_STALE_GUARDBAND_MS)
            .max(1_000);
    }
    EXTENDED_STALE_MS_DEFAULT
}

fn extended_runtime_state_stale_ms() -> u64 {
    extended_state_stale_ms_override().unwrap_or_else(extended_stale_ms)
}

fn extended_transport_stale_ms() -> u64 {
    std::env::var("PARAPHINA_EXTENDED_TRANSPORT_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(EXTENDED_TRANSPORT_STALE_MS_DEFAULT)
}

fn extended_ws_read_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_WS_READ_TIMEOUT_MS_DEFAULT),
    )
}

fn extended_connect_book_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_CONNECT_BOOK_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_CONNECT_BOOK_TIMEOUT_MS_DEFAULT),
    )
}

fn extended_connect_first_frame_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_CONNECT_FIRST_FRAME_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                extended_state_stale_ms_override()
                    .map(|state_ms| {
                        state_ms
                            .saturating_sub(EXTENDED_FIRST_DATA_TIMEOUT_GUARDBAND_MS)
                            .max(1_000)
                    })
                    .unwrap_or(EXTENDED_CONNECT_FIRST_FRAME_TIMEOUT_MS_DEFAULT)
            }),
    )
}

fn extended_connect_control_frame_only_timeout() -> Duration {
    let first_data_timeout_ms = extended_connect_first_frame_timeout().as_millis() as u64;
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_CONNECT_CONTROL_FRAME_ONLY_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                extended_state_stale_ms_override()
                    .map(|state_ms| {
                        first_data_timeout_ms.max(
                            state_ms
                                .saturating_sub(EXTENDED_CONTROL_FRAME_ONLY_TIMEOUT_GUARDBAND_MS),
                        )
                    })
                    .unwrap_or_else(|| {
                        first_data_timeout_ms
                            .max(EXTENDED_CONNECT_CONTROL_FRAME_ONLY_TIMEOUT_MS_DEFAULT)
                    })
            }),
    )
}

fn extended_connect_control_frame_only_hedge_start_after() -> Duration {
    let first_data_timeout_ms = extended_connect_first_frame_timeout().as_millis() as u64;
    Duration::from_millis(
        first_data_timeout_ms
            .saturating_sub(EXTENDED_CONTROL_FRAME_ONLY_HEDGE_START_GUARDBAND_MS)
            .max(EXTENDED_CONTROL_FRAME_ONLY_HEDGE_START_FLOOR_MS),
    )
}

fn extended_post_publish_fallback_after() -> Duration {
    let state_stale_ms = extended_runtime_state_stale_ms();
    let safe_max_ms = state_stale_ms
        .saturating_sub(EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_GUARDBAND_MS)
        .max(1_000);
    if let Some(explicit_ms) = std::env::var("PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_AFTER_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
    {
        return Duration::from_millis(explicit_ms.clamp(1_000, safe_max_ms));
    }
    Duration::from_millis(
        state_stale_ms
            .saturating_sub(EXTENDED_POST_PUBLISH_FALLBACK_AFTER_GUARDBAND_MS)
            .max(1_000),
    )
}

fn extended_post_publish_fallback_deadline() -> Duration {
    let state_stale_ms = extended_runtime_state_stale_ms();
    let after_ms = extended_post_publish_fallback_after().as_millis() as u64;
    let safe_max_ms = state_stale_ms
        .saturating_sub(EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_GUARDBAND_MS)
        .max(after_ms);
    if let Some(explicit_ms) = std::env::var("PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
    {
        return Duration::from_millis(explicit_ms.clamp(after_ms, safe_max_ms));
    }
    Duration::from_millis(
        after_ms.max(
            state_stale_ms.saturating_sub(EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_GUARDBAND_MS),
        ),
    )
}

fn extended_private_ws_read_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_PRIVATE_WS_READ_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_PRIVATE_WS_READ_TIMEOUT_MS_DEFAULT),
    )
}

fn extended_stale_churn_window() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_STALE_CHURN_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_STALE_CHURN_WINDOW_MS_DEFAULT),
    )
}

fn extended_stale_churn_limit() -> usize {
    std::env::var("PARAPHINA_EXTENDED_STALE_CHURN_LIMIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(EXTENDED_STALE_CHURN_LIMIT_DEFAULT)
        .max(1)
}

fn extended_stale_churn_healthy_reset() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_STALE_CHURN_HEALTHY_RESET_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_STALE_CHURN_HEALTHY_RESET_MS_DEFAULT),
    )
}

fn extended_degraded_rebootstrap_max_sleep() -> Option<Duration> {
    std::env::var("PARAPHINA_EXTENDED_DEGRADED_REBOOTSTRAP_MAX_SLEEP_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(Duration::from_millis)
}

fn extended_bootstrap_churn_window() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_BOOTSTRAP_CHURN_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_BOOTSTRAP_CHURN_WINDOW_MS_DEFAULT),
    )
}

fn extended_bootstrap_churn_limit() -> usize {
    std::env::var("PARAPHINA_EXTENDED_BOOTSTRAP_CHURN_LIMIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(EXTENDED_BOOTSTRAP_CHURN_LIMIT_DEFAULT)
        .max(1)
}

fn extended_bootstrap_churn_healthy_reset() -> Duration {
    Duration::from_millis(
        std::env::var("PARAPHINA_EXTENDED_BOOTSTRAP_HEALTHY_RESET_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTENDED_BOOTSTRAP_CHURN_HEALTHY_RESET_MS_DEFAULT),
    )
}

fn extended_ws_depth_levels() -> u32 {
    std::env::var("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(2)
        .max(1)
}

fn extended_ws_audit_enabled() -> bool {
    *EXTENDED_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendedPublicReconnectReason {
    StaleWatchdog,
    BootstrapNoFirstFrame,
    BootstrapControlFrameOnlySessionEstablishment,
    BootstrapControlFrameOnlyBackendAttach,
    BootstrapFrameNoBook,
    BootstrapBookNoPublish,
    PostPublishTransportGap,
    DegradedStreamRebootstrapGap,
    ReadTimeout,
    PingSendFail,
    ParseError,
    SeqGap,
    SeqMismatch,
    SessionTimeout,
    StreamClosed,
    ConnectTimeout,
    ConnectError,
}

impl ExtendedPublicReconnectReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::StaleWatchdog => "stale_watchdog",
            Self::BootstrapNoFirstFrame => "bootstrap_no_first_frame",
            Self::BootstrapControlFrameOnlySessionEstablishment => {
                "bootstrap_control_frame_only_session_establishment"
            }
            Self::BootstrapControlFrameOnlyBackendAttach => {
                "bootstrap_control_frame_only_backend_attach"
            }
            Self::BootstrapFrameNoBook => "bootstrap_frame_no_book",
            Self::BootstrapBookNoPublish => "bootstrap_book_no_publish",
            Self::PostPublishTransportGap => "post_publish_transport_gap",
            Self::DegradedStreamRebootstrapGap => "degraded_stream_rebootstrap_gap",
            Self::ReadTimeout => "read_timeout",
            Self::PingSendFail => "ping_send_fail",
            Self::ParseError => "parse_error",
            Self::SeqGap => "seq_gap",
            Self::SeqMismatch => "seq_mismatch",
            Self::SessionTimeout => "session_timeout",
            Self::StreamClosed => "stream_closed",
            Self::ConnectTimeout => "connect_timeout",
            Self::ConnectError => "connect_error",
        }
    }

    fn is_bootstrap(self) -> bool {
        matches!(
            self,
            Self::BootstrapNoFirstFrame
                | Self::BootstrapControlFrameOnlySessionEstablishment
                | Self::BootstrapControlFrameOnlyBackendAttach
                | Self::BootstrapFrameNoBook
                | Self::BootstrapBookNoPublish
        )
    }

    fn reason_family(self) -> &'static str {
        if self.is_bootstrap() {
            "bootstrap_no_data"
        } else if matches!(
            self,
            Self::StaleWatchdog
                | Self::PostPublishTransportGap
                | Self::DegradedStreamRebootstrapGap
        ) {
            "stale_watchdog"
        } else {
            self.as_str()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendedBootstrapSocketRole {
    Primary,
    Hedge,
}

impl ExtendedBootstrapSocketRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Hedge => "hedge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendedBootstrapStreamKind {
    Depth1,
    FullOrderbook,
}

impl ExtendedBootstrapStreamKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Depth1 => "depth1",
            Self::FullOrderbook => "full_orderbook",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendedHedgeMode {
    BackendAttach,
    PostPublishStreamFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendedStreamPreference {
    Depth1,
    FullOrderbookDegraded,
}

impl ExtendedStreamPreference {
    fn as_str(self) -> &'static str {
        match self {
            Self::Depth1 => "depth1",
            Self::FullOrderbookDegraded => "full_orderbook_degraded",
        }
    }

    fn preferred_stream_kind(self) -> ExtendedBootstrapStreamKind {
        match self {
            Self::Depth1 => ExtendedBootstrapStreamKind::Depth1,
            Self::FullOrderbookDegraded => ExtendedBootstrapStreamKind::FullOrderbook,
        }
    }
}

#[derive(Debug, Clone)]
struct ExtendedPublicWsExit {
    reason: ExtendedPublicReconnectReason,
    message: String,
}

impl ExtendedPublicWsExit {
    fn new(reason: ExtendedPublicReconnectReason, message: impl Into<String>) -> Self {
        Self {
            reason,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ExtendedPublicWsExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ExtendedPublicWsExit {}

async fn connect_extended_public_ws_stream(
    ws_url: &str,
    socket_role: ExtendedBootstrapSocketRole,
    stream_kind: ExtendedBootstrapStreamKind,
) -> Result<ExtendedWsStream, ExtendedPublicWsExit> {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
    eprintln!(
        "INFO: Extended public WS connecting url={} socket_role={}",
        ws_url,
        socket_role.as_str()
    );
    let mut request = ws_url.into_client_request().map_err(|err| {
        ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectError,
            format!(
                "Extended public WS {} request build error: {err}",
                socket_role.as_str()
            ),
        )
    })?;
    request
        .headers_mut()
        .insert(USER_AGENT, HeaderValue::from_static("paraphina"));
    let host = request
        .uri()
        .host()
        .map(str::to_string)
        .unwrap_or_else(|| "unknown".to_string());
    let path = request
        .uri()
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let port = request
        .uri()
        .port_u16()
        .or_else(|| match request.uri().scheme_str() {
            Some("wss") => Some(443),
            Some("ws") => Some(80),
            _ => None,
        })
        .ok_or_else(|| {
            ExtendedPublicWsExit::new(
                ExtendedPublicReconnectReason::ConnectError,
                format!(
                    "Extended public WS {} missing port in URL",
                    socket_role.as_str()
                ),
            )
        })?;
    let connect_started_at = Instant::now();
    let tcp_started_at = Instant::now();
    let socket = tokio::time::timeout(
        CONNECT_TIMEOUT,
        TcpStream::connect(format!("{host}:{port}")),
    )
    .await
    .map_err(|_| {
        emit_extended_socket_establishment_audit(
            "failed",
            socket_role,
            stream_kind,
            host.as_str(),
            path.as_str(),
            true,
            None,
            None,
            connect_started_at.elapsed().as_millis() as u64,
            Some("tcp_connect"),
            Some("timeout"),
        );
        ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectTimeout,
            format!(
                "Extended public WS {} TCP connect timeout (15s)",
                socket_role.as_str()
            ),
        )
    })?
    .map_err(|err| {
        emit_extended_socket_establishment_audit(
            "failed",
            socket_role,
            stream_kind,
            host.as_str(),
            path.as_str(),
            true,
            None,
            None,
            connect_started_at.elapsed().as_millis() as u64,
            Some("tcp_connect"),
            Some("io_error"),
        );
        ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectError,
            format!(
                "Extended public WS {} TCP connect error: {err}",
                socket_role.as_str()
            ),
        )
    })?;
    let tcp_connect_ms = tcp_started_at.elapsed().as_millis() as u64;
    if let Err(err) = socket.set_nodelay(true) {
        emit_extended_socket_establishment_audit(
            "failed",
            socket_role,
            stream_kind,
            host.as_str(),
            path.as_str(),
            true,
            Some(tcp_connect_ms),
            None,
            connect_started_at.elapsed().as_millis() as u64,
            Some("socket_tune"),
            Some("set_nodelay_error"),
        );
        return Err(ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectError,
            format!(
                "Extended public WS {} set_nodelay error: {err}",
                socket_role.as_str()
            ),
        ));
    }
    emit_extended_socket_establishment_audit(
        "tcp_connected",
        socket_role,
        stream_kind,
        host.as_str(),
        path.as_str(),
        true,
        Some(tcp_connect_ms),
        None,
        connect_started_at.elapsed().as_millis() as u64,
        None,
        None,
    );
    let ws_upgrade_started_at = Instant::now();
    let (ws_stream, _) = tokio::time::timeout(
        CONNECT_TIMEOUT,
        client_async_tls_with_config(request, socket, None, None),
    )
    .await
    .map_err(|_| {
        emit_extended_socket_establishment_audit(
            "failed",
            socket_role,
            stream_kind,
            host.as_str(),
            path.as_str(),
            true,
            Some(tcp_connect_ms),
            None,
            connect_started_at.elapsed().as_millis() as u64,
            Some("ws_upgrade"),
            Some("timeout"),
        );
        ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectTimeout,
            format!(
                "Extended public WS {} TLS/WS upgrade timeout (15s)",
                socket_role.as_str()
            ),
        )
    })?
    .map_err(|err| {
        emit_extended_socket_establishment_audit(
            "failed",
            socket_role,
            stream_kind,
            host.as_str(),
            path.as_str(),
            true,
            Some(tcp_connect_ms),
            None,
            connect_started_at.elapsed().as_millis() as u64,
            Some("ws_upgrade"),
            Some("upgrade_error"),
        );
        ExtendedPublicWsExit::new(
            ExtendedPublicReconnectReason::ConnectError,
            format!(
                "Extended public WS {} connect error: {err}",
                socket_role.as_str()
            ),
        )
    })?;
    emit_extended_socket_establishment_audit(
        "ws_upgraded",
        socket_role,
        stream_kind,
        host.as_str(),
        path.as_str(),
        true,
        Some(tcp_connect_ms),
        Some(ws_upgrade_started_at.elapsed().as_millis() as u64),
        connect_started_at.elapsed().as_millis() as u64,
        None,
        None,
    );
    eprintln!(
        "INFO: Extended public WS connected url={} socket_role={}",
        ws_url,
        socket_role.as_str()
    );
    Ok(ws_stream)
}

fn extended_public_reconnect_sleep(
    reason: ExtendedPublicReconnectReason,
    failure_escalation_suppressed: bool,
    backoff: Duration,
) -> Duration {
    if failure_escalation_suppressed {
        Duration::from_millis(EXTENDED_FAST_RECONNECT_SLEEP_MS)
    } else if reason == ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap {
        extended_degraded_rebootstrap_max_sleep()
            .map(|cap| backoff.min(cap))
            .unwrap_or(backoff)
    } else {
        backoff
    }
}

fn emit_extended_reconnect_policy_audit(
    reason: ExtendedPublicReconnectReason,
    sleep: Duration,
    consecutive_failures: u32,
    failure_escalation_suppressed: bool,
    stale_watchdog_count_window: usize,
    stale_watchdog_window: Duration,
    stale_watchdog_limit: usize,
    stale_watchdog_churn_escalated: bool,
    bootstrap_count_window: usize,
    bootstrap_window: Duration,
    bootstrap_limit: usize,
    bootstrap_churn_escalated: bool,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    eprintln!(
        "WS_AUDIT venue=extended component=reconnect_policy reason={} reason_family={} sleep_ms={} failure_escalation_suppressed={} consecutive_failures={} stale_watchdog_count_window={} stale_watchdog_window_ms={} stale_watchdog_limit={} stale_watchdog_churn_escalated={} bootstrap_count_window={} bootstrap_window_ms={} bootstrap_limit={} bootstrap_churn_escalated={}",
        reason.as_str(),
        reason.reason_family(),
        sleep.as_millis(),
        if failure_escalation_suppressed { 1 } else { 0 },
        consecutive_failures,
        stale_watchdog_count_window,
        stale_watchdog_window.as_millis(),
        stale_watchdog_limit,
        if stale_watchdog_churn_escalated { 1 } else { 0 },
        bootstrap_count_window,
        bootstrap_window.as_millis(),
        bootstrap_limit,
        if bootstrap_churn_escalated { 1 } else { 0 },
    );
}

fn emit_extended_stale_watchdog_churn_audit(
    action: &'static str,
    stale_watchdog_count_window: usize,
    stale_watchdog_window: Duration,
    stale_watchdog_limit: usize,
    stale_watchdog_fast_reconnect_allowed: bool,
    stale_watchdog_churn_escalated: bool,
    healthy_session_ms_before_reset: Option<u64>,
    session_duration_ms: Option<u64>,
    previous_stale_watchdog_count_window: Option<usize>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=stale_watchdog_churn action={} stale_watchdog_count_window={} stale_watchdog_window_ms={} stale_watchdog_limit={} stale_watchdog_fast_reconnect_allowed={} stale_watchdog_churn_escalated={}",
        action,
        stale_watchdog_count_window,
        stale_watchdog_window.as_millis(),
        stale_watchdog_limit,
        if stale_watchdog_fast_reconnect_allowed { 1 } else { 0 },
        if stale_watchdog_churn_escalated { 1 } else { 0 },
    );
    if let Some(value) = healthy_session_ms_before_reset {
        line.push_str(&format!(" healthy_session_ms_before_reset={value}"));
    }
    if let Some(value) = session_duration_ms {
        line.push_str(&format!(" session_duration_ms={value}"));
    }
    if let Some(value) = previous_stale_watchdog_count_window {
        line.push_str(&format!(" previous_stale_watchdog_count_window={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_bootstrap_churn_audit(
    action: &'static str,
    bootstrap_reason: ExtendedPublicReconnectReason,
    bootstrap_count_window: usize,
    bootstrap_window: Duration,
    bootstrap_limit: usize,
    bootstrap_fast_reconnect_allowed: bool,
    bootstrap_churn_escalated: bool,
    healthy_session_ms_before_reset: Option<u64>,
    session_duration_ms: Option<u64>,
    previous_bootstrap_count_window: Option<usize>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=bootstrap_churn action={} bootstrap_reason={} bootstrap_reason_family={} bootstrap_count_window={} bootstrap_window_ms={} bootstrap_limit={} bootstrap_fast_reconnect_allowed={} bootstrap_churn_escalated={}",
        action,
        bootstrap_reason.as_str(),
        bootstrap_reason.reason_family(),
        bootstrap_count_window,
        bootstrap_window.as_millis(),
        bootstrap_limit,
        if bootstrap_fast_reconnect_allowed { 1 } else { 0 },
        if bootstrap_churn_escalated { 1 } else { 0 },
    );
    if let Some(value) = healthy_session_ms_before_reset {
        line.push_str(&format!(" healthy_session_ms_before_reset={value}"));
    }
    if let Some(value) = session_duration_ms {
        line.push_str(&format!(" session_duration_ms={value}"));
    }
    if let Some(value) = previous_bootstrap_count_window {
        line.push_str(&format!(" previous_bootstrap_count_window={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_rest_snapshot_seed_audit(
    status: &str,
    http_status: Option<u16>,
    latency_ms: u64,
    seeded: bool,
    bid_levels: usize,
    ask_levels: usize,
    market: &str,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=rest_snapshot_seed status={} latency_ms={} endpoint_kind=official_orderbook seeded={} bid_levels={} ask_levels={} market={}",
        status,
        latency_ms,
        if seeded { 1 } else { 0 },
        bid_levels,
        ask_levels,
        market,
    );
    if let Some(value) = http_status {
        line.push_str(&format!(" http_status={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_bootstrap_seed_bridge_audit(
    action: &str,
    rest_snapshot_seeded: bool,
    seed_age_ms: u64,
    venue_state_stale_ms: u64,
    connect_first_frame_timeout: Duration,
    clear_reason: Option<&str>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=bootstrap_seed_bridge action={} rest_snapshot_seeded={} rest_seed_bridge_active={} seed_age_ms={} venue_state_stale_ms={} connect_first_frame_timeout_ms={}",
        action,
        if rest_snapshot_seeded { 1 } else { 0 },
        if rest_snapshot_seeded { 1 } else { 0 },
        seed_age_ms,
        venue_state_stale_ms,
        connect_first_frame_timeout.as_millis(),
    );
    if let Some(reason) = clear_reason {
        line.push_str(&format!(" clear_reason={reason}"));
    }
    eprintln!("{line}");
}

#[allow(clippy::too_many_arguments)]
fn emit_extended_bootstrap_timeout_audit(
    reason: ExtendedPublicReconnectReason,
    bootstrap_timeout_stage: &'static str,
    connect_first_frame_timeout: Duration,
    connect_book_timeout: Duration,
    control_frame_only_timeout: Option<Duration>,
    rest_snapshot_seeded: bool,
    rest_seed_bridge_active: bool,
    rest_snapshot_seq: Option<u64>,
    rest_snapshot_latency_ms: Option<u64>,
    rest_snapshot_bid_levels: Option<u64>,
    rest_snapshot_ask_levels: Option<u64>,
    seed_age_ms: Option<u64>,
    time_to_first_control_frame_ms: Option<u64>,
    first_control_frame_kind: &'static str,
    time_to_first_message_ms: Option<u64>,
    time_to_first_book_ms: Option<u64>,
    time_to_first_publish_ms: Option<u64>,
    last_frame_kind: &'static str,
    last_data_kind: &'static str,
    last_seq: u64,
    last_snapshot_seq: u64,
    last_book_seq: u64,
    last_publish_seq: u64,
    stale_watchdog_armed: bool,
    stale_watchdog_deferred_until_first_publish: bool,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=bootstrap_timeout reason={} reason_family={} bootstrap_timeout_stage={} connect_first_frame_timeout_ms={} connect_book_timeout_ms={} rest_snapshot_seeded={} rest_seed_bridge_active={} first_control_frame_seen={} first_control_frame_kind={} first_data_frame_seen={} first_message_seen={} first_book_seen={} first_publish_seen={} stale_watchdog_armed={} stale_watchdog_deferred_until_first_publish={} last_frame_kind={} last_data_kind={} last_seq={} last_snapshot_seq={} last_book_seq={} last_publish_seq={}",
        reason.as_str(),
        reason.reason_family(),
        bootstrap_timeout_stage,
        connect_first_frame_timeout.as_millis(),
        connect_book_timeout.as_millis(),
        if rest_snapshot_seeded { 1 } else { 0 },
        if rest_seed_bridge_active { 1 } else { 0 },
        if time_to_first_control_frame_ms.is_some() {
            1
        } else {
            0
        },
        first_control_frame_kind,
        if time_to_first_message_ms.is_some() { 1 } else { 0 },
        if time_to_first_message_ms.is_some() { 1 } else { 0 },
        if time_to_first_book_ms.is_some() { 1 } else { 0 },
        if time_to_first_publish_ms.is_some() { 1 } else { 0 },
        if stale_watchdog_armed { 1 } else { 0 },
        if stale_watchdog_deferred_until_first_publish {
            1
        } else {
            0
        },
        last_frame_kind,
        last_data_kind,
        last_seq,
        last_snapshot_seq,
        last_book_seq,
        last_publish_seq,
    );
    if let Some(value) = rest_snapshot_seq {
        line.push_str(&format!(" rest_snapshot_seq={value}"));
    }
    if let Some(value) = rest_snapshot_latency_ms {
        line.push_str(&format!(" rest_snapshot_latency_ms={value}"));
    }
    if let Some(value) = rest_snapshot_bid_levels {
        line.push_str(&format!(" rest_snapshot_bid_levels={value}"));
    }
    if let Some(value) = rest_snapshot_ask_levels {
        line.push_str(&format!(" rest_snapshot_ask_levels={value}"));
    }
    if let Some(value) = control_frame_only_timeout {
        line.push_str(&format!(
            " control_frame_only_timeout_ms={}",
            value.as_millis()
        ));
    }
    if let Some(value) = seed_age_ms {
        line.push_str(&format!(" seed_age_ms={value}"));
    }
    if let Some(value) = time_to_first_control_frame_ms {
        line.push_str(&format!(" time_to_first_control_frame_ms={value}"));
    }
    if let Some(value) = time_to_first_message_ms {
        line.push_str(&format!(" time_to_first_message_ms={value}"));
    }
    if let Some(value) = time_to_first_book_ms {
        line.push_str(&format!(" time_to_first_book_ms={value}"));
    }
    if let Some(value) = time_to_first_publish_ms {
        line.push_str(&format!(" time_to_first_publish_ms={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_bootstrap_session_hedge_audit(
    action: &str,
    winner: Option<ExtendedBootstrapSocketRole>,
    loser: Option<ExtendedBootstrapSocketRole>,
    hedge_started_at_ms: Option<u64>,
    connect_first_frame_timeout: Duration,
    control_frame_only_timeout: Duration,
    seed_age_ms: u64,
    venue_state_stale_ms: u64,
    first_control_frame_kind: &'static str,
    rest_seed_bridge_active: bool,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=bootstrap_session_hedge action={} connect_first_frame_timeout_ms={} control_frame_only_timeout_ms={} seed_age_ms={} venue_state_stale_ms={} first_control_frame_kind={} rest_seed_bridge_active={}",
        action,
        connect_first_frame_timeout.as_millis(),
        control_frame_only_timeout.as_millis(),
        seed_age_ms,
        venue_state_stale_ms,
        first_control_frame_kind,
        if rest_seed_bridge_active { 1 } else { 0 },
    );
    if let Some(value) = hedge_started_at_ms {
        line.push_str(&format!(" hedge_started_at_ms={value}"));
    }
    if let Some(value) = winner {
        line.push_str(&format!(" winner={}", value.as_str()));
    }
    if let Some(value) = loser {
        line.push_str(&format!(" loser={}", value.as_str()));
    }
    eprintln!("{line}");
}

fn emit_extended_socket_establishment_audit(
    action: &str,
    socket_role: ExtendedBootstrapSocketRole,
    stream_kind: ExtendedBootstrapStreamKind,
    host: &str,
    path: &str,
    disable_nagle: bool,
    tcp_connect_ms: Option<u64>,
    ws_upgrade_ms: Option<u64>,
    elapsed_ms: u64,
    failure_stage: Option<&str>,
    failure_class: Option<&str>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=socket_establishment action={} socket_role={} stream_kind={} host={} path={} disable_nagle={} elapsed_ms={}",
        action,
        socket_role.as_str(),
        stream_kind.as_str(),
        host,
        path,
        if disable_nagle { 1 } else { 0 },
        elapsed_ms,
    );
    if let Some(value) = tcp_connect_ms {
        line.push_str(&format!(" tcp_connect_ms={value}"));
    }
    if let Some(value) = ws_upgrade_ms {
        line.push_str(&format!(" ws_upgrade_ms={value}"));
    }
    if let Some(value) = failure_stage {
        line.push_str(&format!(" failure_stage={value}"));
    }
    if let Some(value) = failure_class {
        line.push_str(&format!(" failure_class={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_session_progress_audit(
    stage: &'static str,
    socket_role: ExtendedBootstrapSocketRole,
    stream_kind: ExtendedBootstrapStreamKind,
    time_to_first_control_frame_ms: Option<u64>,
    time_to_first_message_ms: Option<u64>,
    time_to_first_book_ms: Option<u64>,
    time_to_first_publish_ms: Option<u64>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=session_progress stage={} socket_role={} stream_kind={} ws_upgrade_completed=1",
        stage,
        socket_role.as_str(),
        stream_kind.as_str(),
    );
    if let Some(value) = time_to_first_control_frame_ms {
        line.push_str(&format!(" time_to_first_control_frame_ms={value}"));
    }
    if let Some(value) = time_to_first_message_ms {
        line.push_str(&format!(" time_to_first_message_ms={value}"));
    }
    if let Some(value) = time_to_first_book_ms {
        line.push_str(&format!(" time_to_first_book_ms={value}"));
    }
    if let Some(value) = time_to_first_publish_ms {
        line.push_str(&format!(" time_to_first_publish_ms={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_backend_attach_fallback_audit(
    action: &str,
    winner: Option<ExtendedBootstrapSocketRole>,
    winner_stream_kind: Option<ExtendedBootstrapStreamKind>,
    hedge_started_at_ms: Option<u64>,
    connect_first_frame_timeout: Duration,
    control_frame_only_timeout: Duration,
    seed_age_ms: u64,
    rest_seed_bridge_active: bool,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=backend_attach_fallback action={} primary_stream_kind=depth1 fallback_stream_kind=full_orderbook connect_first_frame_timeout_ms={} control_frame_only_timeout_ms={} seed_age_ms={} rest_seed_bridge_active={}",
        action,
        connect_first_frame_timeout.as_millis(),
        control_frame_only_timeout.as_millis(),
        seed_age_ms,
        if rest_seed_bridge_active { 1 } else { 0 },
    );
    if let Some(value) = hedge_started_at_ms {
        line.push_str(&format!(" hedge_started_at_ms={value}"));
    }
    if let Some(value) = winner {
        line.push_str(&format!(" winner_socket_role={}", value.as_str()));
    }
    if let Some(value) = winner_stream_kind {
        line.push_str(&format!(" winner_stream_kind={}", value.as_str()));
    }
    eprintln!("{line}");
}

fn emit_extended_post_publish_stream_fallback_audit(
    action: &str,
    active_stream_kind: ExtendedBootstrapStreamKind,
    winner_stream_kind: Option<ExtendedBootstrapStreamKind>,
    attempt_index: Option<u64>,
    started_at_ms: Option<u64>,
    fallback_after: Duration,
    fallback_deadline: Duration,
    age_ws_rx_ms: Option<u64>,
    age_data_rx_ms: Option<u64>,
    age_book_event_ms: Option<u64>,
    age_published_ms: Option<u64>,
    last_frame_kind: &'static str,
    last_data_kind: &'static str,
    stream_preference: ExtendedStreamPreference,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=post_publish_stream_fallback action={} active_stream_kind={} fallback_stream_kind=full_orderbook post_publish_fallback_after_ms={} post_publish_fallback_deadline_ms={} last_frame_kind={} last_data_kind={} stream_preference={}",
        action,
        active_stream_kind.as_str(),
        fallback_after.as_millis(),
        fallback_deadline.as_millis(),
        last_frame_kind,
        last_data_kind,
        stream_preference.as_str(),
    );
    if let Some(value) = started_at_ms {
        line.push_str(&format!(" started_at_ms={value}"));
    }
    if let Some(value) = attempt_index {
        line.push_str(&format!(" attempt_index={value}"));
    }
    if let Some(value) = winner_stream_kind {
        line.push_str(&format!(" winner_stream_kind={}", value.as_str()));
    }
    if let Some(value) = age_ws_rx_ms {
        line.push_str(&format!(" age_ws_rx_ms={value}"));
    }
    if let Some(value) = age_data_rx_ms {
        line.push_str(&format!(" age_data_rx_ms={value}"));
    }
    if let Some(value) = age_book_event_ms {
        line.push_str(&format!(" age_book_event_ms={value}"));
    }
    if let Some(value) = age_published_ms {
        line.push_str(&format!(" age_published_ms={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_degraded_stream_watchdog_audit(
    action: &str,
    fallback_after: Duration,
    age_ws_rx_ms: Option<u64>,
    age_data_rx_ms: Option<u64>,
    age_book_event_ms: Option<u64>,
    age_published_ms: Option<u64>,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=degraded_stream_watchdog action={} post_publish_fallback_after_ms={}",
        action,
        fallback_after.as_millis(),
    );
    if let Some(value) = age_ws_rx_ms {
        line.push_str(&format!(" age_ws_rx_ms={value}"));
    }
    if let Some(value) = age_data_rx_ms {
        line.push_str(&format!(" age_data_rx_ms={value}"));
    }
    if let Some(value) = age_book_event_ms {
        line.push_str(&format!(" age_book_event_ms={value}"));
    }
    if let Some(value) = age_published_ms {
        line.push_str(&format!(" age_published_ms={value}"));
    }
    eprintln!("{line}");
}

fn emit_extended_watchdog_bootstrap_transition_audit(
    time_to_first_publish_ms: Option<u64>,
    stale_watchdog_deferred_until_first_publish: bool,
) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut line = format!(
        "WS_AUDIT venue=extended component=watchdog_bootstrap_transition first_publish_observed=1 watchdog_armed_now=1 stale_watchdog_deferred_until_first_publish={}",
        if stale_watchdog_deferred_until_first_publish {
            1
        } else {
            0
        }
    );
    if let Some(value) = time_to_first_publish_ms {
        line.push_str(&format!(" time_to_first_publish_ms={value}"));
    }
    eprintln!("{line}");
}

fn extended_bootstrap_timeout_reason(
    first_message_seen: bool,
    first_book_seen: bool,
    first_publish_seen: bool,
) -> ExtendedPublicReconnectReason {
    if !first_message_seen {
        ExtendedPublicReconnectReason::BootstrapNoFirstFrame
    } else if !first_book_seen {
        ExtendedPublicReconnectReason::BootstrapFrameNoBook
    } else if !first_publish_seen {
        ExtendedPublicReconnectReason::BootstrapBookNoPublish
    } else {
        ExtendedPublicReconnectReason::BootstrapBookNoPublish
    }
}

fn extended_bootstrap_timeout_stage(first_message_seen: bool) -> &'static str {
    if first_message_seen {
        "post_first_frame"
    } else {
        "first_frame"
    }
}

fn extended_should_start_control_frame_only_session_hedge(
    primary_stream_kind: ExtendedBootstrapStreamKind,
    first_control_frame_seen: bool,
    first_data_frame_seen: bool,
    rest_seed_bridge_active: bool,
    session_hedge_started: bool,
) -> bool {
    primary_stream_kind == ExtendedBootstrapStreamKind::Depth1
        && first_control_frame_seen
        && !first_data_frame_seen
        && rest_seed_bridge_active
        && !session_hedge_started
}

fn extended_should_start_post_publish_stream_fallback(
    active_stream_kind: ExtendedBootstrapStreamKind,
    first_publish_observed: bool,
    fallback_started: bool,
    age_ws_rx_ms: u64,
    age_data_rx_ms: u64,
    age_book_event_ms: u64,
    age_published_ms: u64,
    fallback_after_ms: u64,
) -> bool {
    active_stream_kind == ExtendedBootstrapStreamKind::Depth1
        && first_publish_observed
        && !fallback_started
        && age_ws_rx_ms < fallback_after_ms
        && age_data_rx_ms >= fallback_after_ms
        && age_book_event_ms >= fallback_after_ms
        && age_published_ms >= fallback_after_ms
}

fn extended_should_start_degraded_stream_rebootstrap(
    active_stream_kind: ExtendedBootstrapStreamKind,
    first_publish_observed: bool,
    fallback_started: bool,
    age_data_rx_ms: u64,
    age_book_event_ms: u64,
    age_published_ms: u64,
    fallback_after_ms: u64,
) -> bool {
    active_stream_kind == ExtendedBootstrapStreamKind::FullOrderbook
        && first_publish_observed
        && !fallback_started
        && extended_should_fire_degraded_stream_rebootstrap_watchdog(
            age_data_rx_ms,
            age_book_event_ms,
            age_published_ms,
            fallback_after_ms,
        )
}

fn extended_should_arm_degraded_stream_rebootstrap_watchdog(
    active_stream_kind: ExtendedBootstrapStreamKind,
    first_publish_observed: bool,
    reconnect_prefers_full_orderbook: bool,
    hedge_session_started: bool,
    fallback_started: bool,
) -> bool {
    active_stream_kind == ExtendedBootstrapStreamKind::FullOrderbook
        && first_publish_observed
        && reconnect_prefers_full_orderbook
        && !hedge_session_started
        && !fallback_started
}

fn extended_should_fire_degraded_stream_rebootstrap_watchdog(
    age_data_rx_ms: u64,
    age_book_event_ms: u64,
    age_published_ms: u64,
    fallback_after_ms: u64,
) -> bool {
    age_data_rx_ms >= fallback_after_ms
        && age_book_event_ms >= fallback_after_ms
        && (age_published_ms >= fallback_after_ms || age_published_ms <= age_book_event_ms)
}

fn extended_should_rearm_post_publish_stream_fallback(
    hedge_mode: Option<ExtendedHedgeMode>,
) -> bool {
    hedge_mode == Some(ExtendedHedgeMode::PostPublishStreamFallback)
}

fn extended_failure_escalation_suppressed(
    reason: ExtendedPublicReconnectReason,
    stale_watchdog_count_window: usize,
    stale_watchdog_limit: usize,
    bootstrap_count_window: usize,
    bootstrap_limit: usize,
) -> bool {
    match reason {
        ExtendedPublicReconnectReason::StaleWatchdog
        | ExtendedPublicReconnectReason::PostPublishTransportGap
        | ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap => {
            stale_watchdog_count_window <= stale_watchdog_limit
        }
        _ if reason.is_bootstrap() => bootstrap_count_window <= bootstrap_limit,
        _ => false,
    }
}

fn extended_seq_error_reason(message: &str) -> ExtendedPublicReconnectReason {
    if message.contains("seq gap") {
        ExtendedPublicReconnectReason::SeqGap
    } else if message.contains("seq mismatch") {
        ExtendedPublicReconnectReason::SeqMismatch
    } else {
        ExtendedPublicReconnectReason::ParseError
    }
}

#[derive(Debug, Default)]
struct ExtendedReconnectChurnState {
    reconnects: VecDeque<Instant>,
}

impl ExtendedReconnectChurnState {
    fn observe(&mut self, now: Instant, window: Duration) -> usize {
        self.prune(now, window);
        self.reconnects.push_back(now);
        self.reconnects.len()
    }

    fn reset_after_healthy_session(
        &mut self,
        session_duration: Duration,
        healthy_reset: Duration,
    ) -> Option<usize> {
        if session_duration >= healthy_reset && !self.reconnects.is_empty() {
            let previous = self.reconnects.len();
            self.reconnects.clear();
            return Some(previous);
        }
        None
    }

    fn prune(&mut self, now: Instant, window: Duration) {
        while self
            .reconnects
            .front()
            .copied()
            .map(|ts| now.saturating_duration_since(ts) > window)
            .unwrap_or(false)
        {
            self.reconnects.pop_front();
        }
    }
}

fn extended_audit_reconnect(reason: &'static str) {
    if !extended_ws_audit_enabled() {
        return;
    }
    let mut counts = EXTENDED_RECONNECT_COUNTS
        .get_or_init(|| StdMutex::new(BTreeMap::new()))
        .lock()
        .expect("extended reconnect audit mutex poisoned");
    let count = counts
        .entry(reason)
        .and_modify(|value| *value += 1)
        .or_insert(1);
    eprintln!(
        "WS_AUDIT venue=extended reconnect_reason={} count={}",
        reason, *count
    );
}

#[allow(dead_code)]
fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

fn freshness_ages_ms(freshness: &Freshness) -> (u64, u64, u64, u64) {
    let now_ns = mono_now_ns();
    (
        age_ms(now_ns, freshness.last_ws_rx_ns.load(Ordering::Relaxed)),
        age_ms(now_ns, freshness.last_data_rx_ns.load(Ordering::Relaxed)),
        age_ms(now_ns, freshness.last_book_event_ns.load(Ordering::Relaxed)),
        age_ms(now_ns, freshness.last_published_ns.load(Ordering::Relaxed)),
    )
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

    fn activate_rest_seed_bridge(&self, anchor_ns: u64) {
        self.last_parsed_ns.store(anchor_ns, Ordering::Relaxed);
        self.last_book_event_ns.store(anchor_ns, Ordering::Relaxed);
        self.last_published_ns.store(anchor_ns, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
        // Use last_book_event_ns as the primary watchdog anchor.
        // This ensures the watchdog fires when book data stops flowing,
        // even if non-book WS messages (heartbeats) keep last_ws_rx_ns fresh.
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

fn extended_watchdog_should_fire(
    freshness: &Freshness,
    connect_start_ns: u64,
    first_publish_observed: bool,
    stale_ms: u64,
    transport_stale_ms: u64,
    now_ns: u64,
) -> bool {
    if first_publish_observed {
        let anchor = freshness.last_ws_rx_ns.load(Ordering::Relaxed);
        let anchor = if anchor == 0 {
            connect_start_ns
        } else {
            anchor
        };
        return anchor != 0 && age_ms(now_ns, anchor) > transport_stale_ms;
    }
    let anchor = freshness.anchor_with_connect_start(connect_start_ns);
    anchor != 0 && age_ms(now_ns, anchor) > stale_ms
}

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::USER_AGENT;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::{
    client_async_tls_with_config, connect_async, tungstenite::Message, MaybeTlsStream,
    WebSocketStream,
};

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

type ExtendedWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
type ExtendedWsWrite = futures_util::stream::SplitSink<ExtendedWsStream, Message>;
type ExtendedWsRead = futures_util::stream::SplitStream<ExtendedWsStream>;

#[derive(Debug, Clone)]
pub struct ExtendedConfig {
    pub ws_url: String,
    pub private_ws_url: String,
    pub rest_url: String,
    pub market: String,
    pub depth_limit: usize,
    pub venue_index: usize,
    pub api_key: Option<String>,
    pub trader_cmd: Option<String>,
    pub record_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ExtendedRestSnapshotSeedAttempt {
    raw: Option<String>,
    snapshot: Option<ExtendedDepthSnapshot>,
    status: &'static str,
    http_status: Option<u16>,
    latency_ms: u64,
    bid_levels: usize,
    ask_levels: usize,
}

impl ExtendedConfig {
    pub fn from_env() -> Self {
        let ws_url = std::env::var("EXTENDED_WS_URL").unwrap_or_else(|_| {
            "wss://api.starknet.extended.exchange/stream.extended.exchange/v1".to_string()
        });
        let private_ws_url = std::env::var("EXTENDED_PRIVATE_WS_URL")
            .unwrap_or_else(|_| format!("{}/account", ws_url.trim_end_matches('/')));
        // Default to Starknet Extended API (the original api.extended.exchange returns 404)
        let rest_url = std::env::var("EXTENDED_REST_URL")
            .unwrap_or_else(|_| "https://api.starknet.extended.exchange".to_string());
        let market = std::env::var("EXTENDED_MARKET").unwrap_or_else(|_| "BTCUSDT".to_string());
        let market = normalize_extended_market(&market);
        let depth_limit = std::env::var("EXTENDED_DEPTH_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100);
        let api_key = std::env::var("EXTENDED_API_KEY").ok();
        let trader_cmd = std::env::var("EXTENDED_TRADER_CMD")
            .ok()
            .map(|raw| raw.trim().to_string())
            .filter(|raw| !raw.is_empty());
        Self {
            ws_url,
            private_ws_url,
            rest_url,
            market,
            depth_limit,
            venue_index: 0,
            api_key,
            trader_cmd,
            record_dir: None,
        }
    }

    pub fn with_record_dir(mut self, dir: PathBuf) -> Self {
        self.record_dir = Some(dir);
        self
    }

    pub fn has_account_auth(&self) -> bool {
        self.api_key.is_some() && self.trader_cmd.is_some()
    }

    pub fn has_private_read_auth(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn has_execution_auth(&self) -> bool {
        self.api_key.is_some() && self.trader_cmd.is_some()
    }

    pub fn orderbook_ws_url(&self) -> String {
        let depth_levels = extended_ws_depth_levels();
        self.orderbook_ws_url_for_stream_kind(if depth_levels <= 1 {
            ExtendedBootstrapStreamKind::Depth1
        } else {
            ExtendedBootstrapStreamKind::FullOrderbook
        })
    }

    fn orderbook_ws_url_for_stream_kind(&self, stream_kind: ExtendedBootstrapStreamKind) -> String {
        match stream_kind {
            ExtendedBootstrapStreamKind::Depth1 => format!(
                "{}/orderbooks/{}?depth=1",
                self.ws_url.trim_end_matches('/'),
                self.market
            ),
            ExtendedBootstrapStreamKind::FullOrderbook => format!(
                "{}/orderbooks/{}",
                self.ws_url.trim_end_matches('/'),
                self.market
            ),
        }
    }
}

#[derive(Debug)]
pub struct ExtendedConnector {
    cfg: ExtendedConfig,
    http: Client,
    market_publisher: MarketPublisher,
    recorder: Option<Mutex<ExtendedRecorder>>,
    freshness: Arc<Freshness>,
    prefer_full_orderbook_on_reconnect: Arc<AtomicBool>,
    session_post_publish_fallback_used: Arc<AtomicBool>,
    is_fixture: bool,
}

impl ExtendedConnector {
    pub fn new(cfg: ExtendedConfig, market_tx: mpsc::Sender<MarketDataEvent>) -> Self {
        let recorder = cfg
            .record_dir
            .as_ref()
            .and_then(|dir| ExtendedRecorder::new(dir).ok())
            .map(Mutex::new);
        let http = Client::builder()
            .user_agent("paraphina")
            .timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(5)
            .build()
            .expect("extended http client build");
        let freshness = Arc::new(Freshness::default());
        let prefer_full_orderbook_on_reconnect = Arc::new(AtomicBool::new(false));
        let session_post_publish_fallback_used = Arc::new(AtomicBool::new(false));
        let is_fixture = std::env::var_os("EXTENDED_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var_os("EXTENDED_FIXTURE_MODE").is_some();
        let cap = if is_fixture {
            EXTENDED_MARKET_PUB_QUEUE_CAP_FIXTURE
        } else {
            live_market_pub_queue_cap(EXTENDED_MARKET_PUB_QUEUE_CAP_LIVE)
        };
        let drain_max = live_market_pub_drain_max(EXTENDED_MARKET_PUB_DRAIN_MAX);
        let publish_freshness = freshness.clone();
        let on_published = Arc::new(move || {
            publish_freshness
                .last_published_ns
                .store(mono_now_ns(), Ordering::Relaxed);
        });
        let market_publisher = MarketPublisher::new(
            cap,
            drain_max,
            "extended",
            market_tx.clone(),
            Some(Arc::new(move || is_fixture)),
            Arc::new(|event: &MarketDataEvent| {
                matches!(
                    event,
                    MarketDataEvent::L2Delta(_) | MarketDataEvent::L2Snapshot(_)
                )
            }),
            Some(on_published),
            "extended market_tx closed",
            "extended market publish queue closed",
        );
        let connector = Self {
            cfg,
            http,
            market_publisher,
            recorder,
            freshness,
            prefer_full_orderbook_on_reconnect,
            session_post_publish_fallback_used,
            is_fixture,
        };
        connector
    }

    async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        self.market_publisher.publish_market(event).await
    }

    pub async fn run_public_ws(&self) {
        let mut backoff = Duration::from_secs(1);
        let mut consecutive_failures: u32 = 0;
        let mut last_snapshot_warn: Option<Instant> = None;
        let stale_watchdog_window = extended_stale_churn_window();
        let stale_watchdog_limit = extended_stale_churn_limit();
        let stale_watchdog_healthy_reset = extended_stale_churn_healthy_reset();
        let bootstrap_window = extended_bootstrap_churn_window();
        let bootstrap_limit = extended_bootstrap_churn_limit();
        let bootstrap_healthy_reset = extended_bootstrap_churn_healthy_reset();
        let mut stale_watchdog_churn = ExtendedReconnectChurnState::default();
        let mut bootstrap_churn = ExtendedReconnectChurnState::default();

        // FIX: Configurable healthy connection threshold for backoff reset
        let healthy_threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60_000),
        );

        loop {
            let session_start = Instant::now();
            self.session_post_publish_fallback_used
                .store(false, Ordering::Relaxed);
            let stream_preference = if self
                .prefer_full_orderbook_on_reconnect
                .load(Ordering::Relaxed)
            {
                ExtendedStreamPreference::FullOrderbookDegraded
            } else {
                ExtendedStreamPreference::Depth1
            };

            // Layer C: session-level timeout catches ALL hang scenarios.
            let max_session = Duration::from_secs(
                std::env::var("PARAPHINA_WS_MAX_SESSION_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(86_400), // 24h — Layer A enforcer handles stuck connections
            );
            let outcome = match tokio::time::timeout(
                max_session,
                self.public_ws_once(&mut last_snapshot_warn, stream_preference),
            )
            .await
            {
                Ok(Err(exit)) => exit,
                Ok(Ok(())) => ExtendedPublicWsExit::new(
                    ExtendedPublicReconnectReason::StreamClosed,
                    "Extended public WS exited without explicit reconnect reason",
                ),
                Err(_timeout) => ExtendedPublicWsExit::new(
                    ExtendedPublicReconnectReason::SessionTimeout,
                    format!(
                        "Extended public WS session timeout ({}s) — force reconnect",
                        max_session.as_secs()
                    ),
                ),
            };
            extended_audit_reconnect(outcome.reason.as_str());
            let session_duration = session_start.elapsed();
            let post_publish_fallback_used = self
                .session_post_publish_fallback_used
                .swap(false, Ordering::Relaxed);
            if let Some(previous_count) = stale_watchdog_churn
                .reset_after_healthy_session(session_duration, stale_watchdog_healthy_reset)
            {
                emit_extended_stale_watchdog_churn_audit(
                    "reset",
                    0,
                    stale_watchdog_window,
                    stale_watchdog_limit,
                    true,
                    false,
                    Some(session_duration.as_millis() as u64),
                    None,
                    Some(previous_count),
                );
            }
            if let Some(previous_count) = bootstrap_churn
                .reset_after_healthy_session(session_duration, bootstrap_healthy_reset)
            {
                emit_extended_bootstrap_churn_audit(
                    "reset",
                    ExtendedPublicReconnectReason::BootstrapNoFirstFrame,
                    0,
                    bootstrap_window,
                    bootstrap_limit,
                    true,
                    false,
                    Some(session_duration.as_millis() as u64),
                    None,
                    Some(previous_count),
                );
            }
            if self
                .prefer_full_orderbook_on_reconnect
                .load(Ordering::Relaxed)
                && session_duration >= stale_watchdog_healthy_reset
                && outcome.reason != ExtendedPublicReconnectReason::StaleWatchdog
                && outcome.reason != ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap
                && !post_publish_fallback_used
            {
                self.prefer_full_orderbook_on_reconnect
                    .store(false, Ordering::Relaxed);
                emit_extended_post_publish_stream_fallback_audit(
                    "preference_reset",
                    ExtendedBootstrapStreamKind::FullOrderbook,
                    None,
                    None,
                    None,
                    extended_post_publish_fallback_after(),
                    extended_post_publish_fallback_deadline(),
                    None,
                    None,
                    None,
                    None,
                    "none",
                    "none",
                    ExtendedStreamPreference::Depth1,
                );
            }

            let mut stale_watchdog_count_window = 0usize;
            let mut stale_watchdog_churn_escalated = false;
            let mut bootstrap_count_window = 0usize;
            let mut bootstrap_churn_escalated = false;
            if matches!(
                outcome.reason,
                ExtendedPublicReconnectReason::StaleWatchdog
                    | ExtendedPublicReconnectReason::PostPublishTransportGap
                    | ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap
            ) {
                stale_watchdog_count_window =
                    stale_watchdog_churn.observe(Instant::now(), stale_watchdog_window);
                stale_watchdog_churn_escalated = stale_watchdog_count_window > stale_watchdog_limit;
            } else if outcome.reason.is_bootstrap() {
                bootstrap_count_window = bootstrap_churn.observe(Instant::now(), bootstrap_window);
                bootstrap_churn_escalated = bootstrap_count_window > bootstrap_limit;
            }
            let failure_escalation_suppressed = extended_failure_escalation_suppressed(
                outcome.reason,
                stale_watchdog_count_window,
                stale_watchdog_limit,
                bootstrap_count_window,
                bootstrap_limit,
            );
            if outcome.reason == ExtendedPublicReconnectReason::StaleWatchdog {
                emit_extended_stale_watchdog_churn_audit(
                    "record",
                    stale_watchdog_count_window,
                    stale_watchdog_window,
                    stale_watchdog_limit,
                    failure_escalation_suppressed,
                    stale_watchdog_churn_escalated,
                    None,
                    Some(session_duration.as_millis() as u64),
                    None,
                );
            } else if outcome.reason.is_bootstrap() {
                emit_extended_bootstrap_churn_audit(
                    "record",
                    outcome.reason,
                    bootstrap_count_window,
                    bootstrap_window,
                    bootstrap_limit,
                    failure_escalation_suppressed,
                    bootstrap_churn_escalated,
                    None,
                    Some(session_duration.as_millis() as u64),
                    None,
                );
            }

            if failure_escalation_suppressed {
                eprintln!("INFO: {}", outcome.message);
            } else {
                consecutive_failures += 1;
                let level = if consecutive_failures >= 20 {
                    "ERROR"
                } else if consecutive_failures >= 5 {
                    "WARN"
                } else {
                    "INFO"
                };
                eprintln!(
                    "{level}: Extended public WS error (consecutive_failures={consecutive_failures}): {}",
                    outcome.message
                );
            }

            // FIX: Reset backoff and failure counter if connection was healthy for long enough
            if session_duration >= healthy_threshold {
                if consecutive_failures > 0 {
                    eprintln!(
                        "INFO: Extended WS session was healthy for {:?}; \
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

            let sleep = extended_public_reconnect_sleep(
                outcome.reason,
                failure_escalation_suppressed,
                backoff,
            );
            emit_extended_reconnect_policy_audit(
                outcome.reason,
                sleep,
                consecutive_failures,
                failure_escalation_suppressed,
                stale_watchdog_count_window,
                stale_watchdog_window,
                stale_watchdog_limit,
                stale_watchdog_churn_escalated,
                bootstrap_count_window,
                bootstrap_window,
                bootstrap_limit,
                bootstrap_churn_escalated,
            );
            tokio::time::sleep(sleep).await;
            if !failure_escalation_suppressed {
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }

    pub async fn run_funding_polling(&self, poll_ms: u64) {
        let mut interval = tokio::time::interval(Duration::from_millis(poll_ms.max(250)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut seq: u64 = 0;
        loop {
            interval.tick().await;
            match fetch_public_funding(&self.http, &self.cfg).await {
                Ok(mut update) => {
                    seq = seq.wrapping_add(1);
                    update.seq = seq;
                    if let Err(err) = self
                        .market_publisher
                        .publish_market(MarketDataEvent::FundingUpdate(update))
                        .await
                    {
                        eprintln!("Extended funding publish error: {err}");
                    }
                }
                Err(err) => {
                    eprintln!("Extended funding polling error: {err}");
                }
            }
        }
    }

    async fn public_ws_once(
        &self,
        last_snapshot_warn: &mut Option<Instant>,
        stream_preference: ExtendedStreamPreference,
    ) -> Result<(), ExtendedPublicWsExit> {
        let mut first_decoded_top_logged = false;
        let mut decode_miss_count = 0usize;
        let mut first_ws_message_logged = false;
        let mut first_book_update_logged = false;
        let mut ws_snapshot_seq: u64 = 0;
        self.freshness.reset_for_new_connection();
        let rest_snapshot_attempt = self.fetch_snapshot().await;
        let mut snapshot_state: Option<ExtendedDepthSnapshot> = None;
        let rest_snapshot_latency_ms = Some(rest_snapshot_attempt.latency_ms);
        let rest_snapshot_bid_levels = Some(rest_snapshot_attempt.bid_levels as u64);
        let rest_snapshot_ask_levels = Some(rest_snapshot_attempt.ask_levels as u64);
        if let (Some(snapshot_raw), Some(snapshot)) = (
            rest_snapshot_attempt.raw.as_deref(),
            rest_snapshot_attempt.snapshot.clone(),
        ) {
            if let Some(recorder) = self.recorder.as_ref() {
                let mut guard = recorder.lock().await;
                if let Err(err) = guard.record_snapshot(snapshot_raw) {
                    eprintln!("WARN: Extended snapshot record failed: {err}");
                }
            }
            if let Ok(value) = serde_json::from_str::<Value>(snapshot_raw) {
                if let Some(top) =
                    TopOfBook::from_levels(&snapshot.bids, &snapshot.asks, Some(now_ms()))
                {
                    eprintln!(
                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                        top.best_bid_px, top.best_bid_sz, top.best_ask_px, top.best_ask_sz
                    );
                    first_decoded_top_logged = true;
                } else if decode_miss_count < 3 {
                    decode_miss_count += 1;
                    log_decode_miss(
                        "Extended",
                        &value,
                        snapshot_raw,
                        decode_miss_count,
                        self.cfg.ws_url.as_str(),
                    );
                }
            }
            snapshot_state = Some(snapshot);
        } else if last_snapshot_warn
            .map(|last| last.elapsed() >= Duration::from_secs(30))
            .unwrap_or(true)
        {
            *last_snapshot_warn = Some(Instant::now());
            eprintln!(
                "WARN: Extended REST snapshot skipped; relying on WS orderbook stream depth_levels={}",
                extended_ws_depth_levels()
            );
        }
        let mut seq_state = ExtendedSeqState::new(
            snapshot_state
                .as_ref()
                .and_then(|snapshot| snapshot.last_update_id),
            self.cfg.venue_index,
        );
        let rest_snapshot_seeded = snapshot_state.is_some();
        let rest_snapshot_seq = snapshot_state
            .as_ref()
            .and_then(|snapshot| snapshot.last_update_id);
        let venue_state_stale_ms = extended_runtime_state_stale_ms();
        let transport_stale_ms = extended_transport_stale_ms();
        let mut rest_seed_bridge_active = false;
        let mut rest_seed_bridge_anchor_ns: Option<u64> = None;
        if let Some(snapshot) = snapshot_state {
            let snapshot_event = MarketDataEvent::L2Snapshot(super::super::types::L2Snapshot {
                venue_index: self.cfg.venue_index,
                venue_id: self.cfg.market.clone(),
                seq: snapshot.last_update_id.unwrap_or(0),
                timestamp_ms: now_ms(),
                bids: snapshot.bids,
                asks: snapshot.asks,
            });
            if self.publish_market(snapshot_event).await.is_ok() {
                let seed_anchor_ns = mono_now_ns();
                self.freshness.activate_rest_seed_bridge(seed_anchor_ns);
                rest_seed_bridge_active = true;
                rest_seed_bridge_anchor_ns = Some(seed_anchor_ns);
                emit_extended_bootstrap_seed_bridge_audit(
                    "activated",
                    true,
                    0,
                    venue_state_stale_ms,
                    extended_connect_first_frame_timeout(),
                    None,
                );
            }
        }

        let primary_stream_kind = if extended_ws_depth_levels() <= 1 {
            stream_preference.preferred_stream_kind()
        } else {
            ExtendedBootstrapStreamKind::FullOrderbook
        };
        let fallback_stream_kind = ExtendedBootstrapStreamKind::FullOrderbook;
        let mut active_stream_kind = primary_stream_kind;
        let ws_url = self
            .cfg
            .orderbook_ws_url_for_stream_kind(primary_stream_kind);
        let fallback_ws_url = self
            .cfg
            .orderbook_ws_url_for_stream_kind(fallback_stream_kind);
        let mut active_ws_url = ws_url.clone();
        let read_timeout = extended_ws_read_timeout();
        let connect_first_frame_timeout = extended_connect_first_frame_timeout();
        let control_frame_only_timeout = extended_connect_control_frame_only_timeout();
        let control_frame_only_hedge_start_after =
            extended_connect_control_frame_only_hedge_start_after();
        let connect_book_timeout = extended_connect_book_timeout();
        let post_publish_fallback_after = extended_post_publish_fallback_after();
        let post_publish_fallback_deadline = extended_post_publish_fallback_deadline();
        let ws_stream = connect_extended_public_ws_stream(
            &ws_url,
            ExtendedBootstrapSocketRole::Primary,
            primary_stream_kind,
        )
        .await?;
        if extended_ws_audit_enabled() {
            eprintln!(
                "WS_AUDIT venue=extended extended_read_timeout_ms={} extended_connect_first_frame_timeout_ms={} extended_control_frame_only_timeout_ms={} extended_connect_book_timeout_ms={} extended_control_frame_only_hedge_start_after_ms={} extended_post_publish_fallback_after_ms={} extended_post_publish_fallback_deadline_ms={} extended_state_stale_ms={} extended_transport_stale_ms={} extended_primary_stream_kind={} extended_fallback_stream_kind={}",
                read_timeout.as_millis(),
                connect_first_frame_timeout.as_millis(),
                control_frame_only_timeout.as_millis(),
                connect_book_timeout.as_millis(),
                control_frame_only_hedge_start_after.as_millis(),
                post_publish_fallback_after.as_millis(),
                post_publish_fallback_deadline.as_millis(),
                venue_state_stale_ms,
                transport_stale_ms,
                primary_stream_kind.as_str(),
                fallback_stream_kind.as_str()
            );
        }
        let (mut write, mut read) = ws_stream.split();
        let mut hedge_write: Option<ExtendedWsWrite> = None;
        let mut hedge_read: Option<ExtendedWsRead> = None;
        let mut hedge_connect_task: Option<
            JoinHandle<Result<ExtendedWsStream, ExtendedPublicWsExit>>,
        > = None;
        let mut hedge_start_armed = false;
        let mut hedge_session_started = false;
        let mut hedge_started_at_ms: Option<u64> = None;
        let mut hedge_cleanup_winner: Option<ExtendedBootstrapSocketRole> = None;
        let mut hedge_mode: Option<ExtendedHedgeMode> = None;
        let mut hedge_seq_state: Option<ExtendedSeqState> = None;
        let mut hedge_ws_snapshot_seq: u64 = 0;
        let mut post_publish_fallback_attempted = false;
        let mut post_publish_fallback_attempt_count: u64 = 0;
        let mut post_publish_fallback_active_attempt_index: Option<u64> = None;

        const MAX_PARSE_ERRORS: usize = 25;
        let mut consecutive_parse_errors = 0usize;
        let mut first_message_logged = false;
        let ws_start = Instant::now();
        let mut first_control_frame_latency_ms: Option<u64> = None;
        let mut first_control_frame_kind: &'static str = "none";
        let mut first_message_latency_ms: Option<u64> = None;
        let mut first_book_latency_ms: Option<u64> = None;
        let mut first_publish_latency_ms: Option<u64> = None;
        let mut no_book_warned = false;
        let mut first_ws_keys: Option<String> = None;
        let mut first_ws_snippet: Option<String> = None;
        let connect_start_ns = mono_now_ns();
        let mut first_publish_observed = false;
        let first_publish_observed_watchdog = Arc::new(AtomicBool::new(false));
        let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
        let fixture_mode = std::env::var_os("EXTENDED_FIXTURE_DIR").is_some()
            || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some()
            || std::env::var("EXTENDED_FIXTURE_MODE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let stale_ms = extended_stale_ms();
        let bootstrap_first_frame_timeout = tokio::time::sleep(connect_first_frame_timeout);
        tokio::pin!(bootstrap_first_frame_timeout);
        let bootstrap_control_frame_only_timeout = tokio::time::sleep(Duration::from_secs(86_400));
        tokio::pin!(bootstrap_control_frame_only_timeout);
        let bootstrap_session_hedge_start = tokio::time::sleep(Duration::from_secs(86_400));
        tokio::pin!(bootstrap_session_hedge_start);
        let bootstrap_post_first_frame_timeout = tokio::time::sleep(connect_book_timeout);
        tokio::pin!(bootstrap_post_first_frame_timeout);
        let post_publish_fallback_timeout = tokio::time::sleep(Duration::from_secs(86_400));
        tokio::pin!(post_publish_fallback_timeout);
        let bootstrap_started_at = tokio::time::Instant::now();
        let control_frame_only_deadline = bootstrap_started_at + control_frame_only_timeout;
        let mut bootstrap_first_frame_timeout_consumed = false;
        let mut bootstrap_post_first_frame_timeout_armed = false;
        // WS-level ping timer to prevent idle connection drops.
        let ping_interval_ms: u64 = std::env::var("PARAPHINA_EXTENDED_PING_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let mut ping_timer = tokio::time::interval(Duration::from_millis(ping_interval_ms));
        ping_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ping_timer.tick().await; // skip first immediate tick
        let mut post_publish_monitor =
            tokio::time::interval(Duration::from_millis(EXTENDED_WATCHDOG_TICK_MS));
        post_publish_monitor.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        post_publish_monitor.tick().await;
        let watchdog_armed = Arc::new(AtomicBool::new(false));
        let degraded_stream_watchdog_armed = Arc::new(AtomicBool::new(false));
        let mut degraded_stream_watchdog_last_armed = false;
        let (degraded_stream_watchdog_tx, degraded_stream_watchdog_rx) =
            tokio::sync::oneshot::channel::<()>();
        let mut degraded_stream_watchdog_rx = degraded_stream_watchdog_rx;
        if fixture_mode {
            eprintln!("INFO: Extended fixture mode detected; freshness watchdog disabled");
        } else {
            let watchdog_stale_ms = stale_ms;
            let watchdog_transport_stale_ms = transport_stale_ms;
            let watchdog_freshness = self.freshness.clone();
            let watchdog_armed_task = watchdog_armed.clone();
            let watchdog_first_publish_observed = first_publish_observed_watchdog.clone();
            tokio::spawn(async move {
                let mut iv =
                    tokio::time::interval(Duration::from_millis(EXTENDED_WATCHDOG_TICK_MS));
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    iv.tick().await;
                    if !watchdog_armed_task.load(Ordering::Relaxed) {
                        continue;
                    }
                    let now = mono_now_ns();
                    if extended_watchdog_should_fire(
                        &watchdog_freshness,
                        connect_start_ns,
                        watchdog_first_publish_observed.load(Ordering::Relaxed),
                        watchdog_stale_ms,
                        watchdog_transport_stale_ms,
                        now,
                    ) {
                        let _ = stale_tx.send(());
                        break;
                    }
                }
            });
            let degraded_watchdog_freshness = self.freshness.clone();
            let degraded_watchdog_armed_task = degraded_stream_watchdog_armed.clone();
            let degraded_watchdog_after = post_publish_fallback_after;
            tokio::spawn(async move {
                let mut iv =
                    tokio::time::interval(Duration::from_millis(EXTENDED_WATCHDOG_TICK_MS));
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    iv.tick().await;
                    if !degraded_watchdog_armed_task.load(Ordering::Relaxed) {
                        continue;
                    }
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&degraded_watchdog_freshness);
                    if !extended_should_fire_degraded_stream_rebootstrap_watchdog(
                        age_data_rx_ms,
                        age_book_event_ms,
                        age_published_ms,
                        degraded_watchdog_after.as_millis() as u64,
                    ) {
                        continue;
                    }
                    degraded_watchdog_armed_task.store(false, Ordering::Relaxed);
                    emit_extended_degraded_stream_watchdog_audit(
                        "fired",
                        degraded_watchdog_after,
                        Some(age_ws_rx_ms),
                        Some(age_data_rx_ms),
                        Some(age_book_event_ms),
                        Some(age_published_ms),
                    );
                    let _ = degraded_stream_watchdog_tx.send(());
                    break;
                }
            });
        }
        macro_rules! sync_degraded_stream_watchdog {
            () => {{
                let should_arm = extended_should_arm_degraded_stream_rebootstrap_watchdog(
                    active_stream_kind,
                    first_publish_observed,
                    self.prefer_full_orderbook_on_reconnect
                        .load(Ordering::Relaxed),
                    hedge_session_started,
                    post_publish_fallback_attempted,
                );
                if should_arm != degraded_stream_watchdog_last_armed {
                    degraded_stream_watchdog_armed.store(should_arm, Ordering::Relaxed);
                    degraded_stream_watchdog_last_armed = should_arm;
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&self.freshness);
                    emit_extended_degraded_stream_watchdog_audit(
                        if should_arm { "armed" } else { "cleared" },
                        post_publish_fallback_after,
                        Some(age_ws_rx_ms),
                        Some(age_data_rx_ms),
                        Some(age_book_event_ms),
                        Some(age_published_ms),
                    );
                }
            }};
        }
        let ws_msg_audit_enabled = extended_ws_audit_enabled();
        let mut frames_text: u64 = 0;
        let mut frames_binary: u64 = 0;
        let mut frames_ping: u64 = 0;
        let mut frames_pong: u64 = 0;
        let mut frames_close: u64 = 0;
        let mut frames_other: u64 = 0;
        let mut cleaned_payload: u64 = 0;
        let mut parse_update_ok: u64 = 0;
        let mut parse_update_err: u64 = 0;
        let mut ws_snapshot_parsed: u64 = 0;
        let mut ws_delta_outcome_some: u64 = 0;
        let mut ws_delta_outcome_none: u64 = 0;
        let mut publish_ok: u64 = 0;
        let mut publish_err: u64 = 0;
        let mut last_frame_kind: &'static str = "none";
        let mut last_data_kind: &'static str = "none";
        let mut last_seq: u64 = 0;
        let mut last_snapshot_seq: u64 = 0;
        let mut last_book_seq: u64 = 0;
        let mut last_publish_seq: u64 = 0;
        let mut last_rx_mono_ns: u64 = 0;
        let mut max_gap_ms: u64 = 0;
        let mut last_audit_instant = Instant::now();
        macro_rules! arm_post_first_frame_timeout {
            () => {
                if !bootstrap_post_first_frame_timeout_armed {
                    bootstrap_post_first_frame_timeout
                        .as_mut()
                        .reset(tokio::time::Instant::now() + connect_book_timeout);
                    bootstrap_post_first_frame_timeout_armed = true;
                }
            };
        }
        macro_rules! start_session_hedge {
            () => {
                if !hedge_session_started {
                    let seed_age_ms = rest_seed_bridge_anchor_ns
                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                        .unwrap_or(0);
                    hedge_started_at_ms = Some(ws_start.elapsed().as_millis() as u64);
                    emit_extended_backend_attach_fallback_audit(
                        "started",
                        None,
                        None,
                        hedge_started_at_ms,
                        connect_first_frame_timeout,
                        control_frame_only_timeout,
                        seed_age_ms,
                        rest_seed_bridge_active,
                    );
                    let hedge_ws_url = fallback_ws_url.clone();
                    hedge_connect_task = Some(tokio::spawn(async move {
                        connect_extended_public_ws_stream(
                            &hedge_ws_url,
                            ExtendedBootstrapSocketRole::Hedge,
                            fallback_stream_kind,
                        )
                        .await
                    }));
                    hedge_mode = Some(ExtendedHedgeMode::BackendAttach);
                    hedge_seq_state = Some(ExtendedSeqState::new(None, self.cfg.venue_index));
                    hedge_ws_snapshot_seq = 0;
                    bootstrap_control_frame_only_timeout
                        .as_mut()
                        .reset(control_frame_only_deadline);
                    bootstrap_session_hedge_start
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_secs(86_400));
                    hedge_start_armed = false;
                    hedge_session_started = true;
                }
            };
        }
        macro_rules! start_post_publish_fallback {
            ($age_ws_rx_ms:expr, $age_data_rx_ms:expr, $age_book_event_ms:expr, $age_published_ms:expr) => {
                if !hedge_session_started && !post_publish_fallback_attempted {
                    let attempt_index = post_publish_fallback_attempt_count.saturating_add(1);
                    post_publish_fallback_attempt_count = attempt_index;
                    post_publish_fallback_active_attempt_index = Some(attempt_index);
                    let started_at_ms = Some(ws_start.elapsed().as_millis() as u64);
                    emit_extended_post_publish_stream_fallback_audit(
                        "armed",
                        active_stream_kind,
                        None,
                        Some(attempt_index),
                        started_at_ms,
                        post_publish_fallback_after,
                        post_publish_fallback_deadline,
                        Some($age_ws_rx_ms),
                        Some($age_data_rx_ms),
                        Some($age_book_event_ms),
                        Some($age_published_ms),
                        last_frame_kind,
                        last_data_kind,
                        stream_preference,
                    );
                    emit_extended_post_publish_stream_fallback_audit(
                        "started",
                        active_stream_kind,
                        None,
                        Some(attempt_index),
                        started_at_ms,
                        post_publish_fallback_after,
                        post_publish_fallback_deadline,
                        Some($age_ws_rx_ms),
                        Some($age_data_rx_ms),
                        Some($age_book_event_ms),
                        Some($age_published_ms),
                        last_frame_kind,
                        last_data_kind,
                        stream_preference,
                    );
                    let hedge_ws_url = fallback_ws_url.clone();
                    hedge_started_at_ms = started_at_ms;
                    hedge_connect_task = Some(tokio::spawn(async move {
                        connect_extended_public_ws_stream(
                            &hedge_ws_url,
                            ExtendedBootstrapSocketRole::Hedge,
                            fallback_stream_kind,
                        )
                        .await
                    }));
                    hedge_mode = Some(ExtendedHedgeMode::PostPublishStreamFallback);
                    hedge_seq_state = Some(ExtendedSeqState::new(None, self.cfg.venue_index));
                    hedge_ws_snapshot_seq = 0;
                    post_publish_fallback_attempted = true;
                    self.session_post_publish_fallback_used
                        .store(true, Ordering::Relaxed);
                    let progress_anchor_ns = self
                        .freshness
                        .last_data_rx_ns
                        .load(Ordering::Relaxed)
                        .max(self.freshness.last_book_event_ns.load(Ordering::Relaxed))
                        .max(self.freshness.last_published_ns.load(Ordering::Relaxed));
                    let progress_age_ms = age_ms(mono_now_ns(), progress_anchor_ns);
                    let remaining_ms = (post_publish_fallback_deadline.as_millis() as u64)
                        .saturating_sub(progress_age_ms);
                    post_publish_fallback_timeout
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_millis(remaining_ms));
                    hedge_session_started = true;
                }
            };
        }
        macro_rules! note_first_message {
            ($socket_role:expr, $stream_kind:expr) => {
                if !first_message_logged {
                    eprintln!("INFO: Extended public WS first message received");
                    first_message_logged = true;
                    first_message_latency_ms = Some(ws_start.elapsed().as_millis() as u64);
                    emit_extended_session_progress_audit(
                        "first_message",
                        $socket_role,
                        $stream_kind,
                        first_control_frame_latency_ms,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                    );
                    if !fixture_mode {
                        arm_post_first_frame_timeout!();
                    }
                }
            };
        }
        macro_rules! note_first_control_frame {
            ($kind:expr, $socket_role:expr, $stream_kind:expr) => {
                if first_control_frame_latency_ms.is_none() {
                    first_control_frame_kind = $kind;
                    first_control_frame_latency_ms = Some(ws_start.elapsed().as_millis() as u64);
                    emit_extended_session_progress_audit(
                        "first_control_frame",
                        $socket_role,
                        $stream_kind,
                        first_control_frame_latency_ms,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                    );
                    if !fixture_mode
                        && first_message_latency_ms.is_none()
                        && extended_should_start_control_frame_only_session_hedge(
                            primary_stream_kind,
                            true,
                            false,
                            rest_seed_bridge_active,
                            hedge_session_started,
                        )
                    {
                        bootstrap_session_hedge_start
                            .as_mut()
                            .reset(bootstrap_started_at + control_frame_only_hedge_start_after);
                        hedge_start_armed = true;
                    }
                }
            };
        }
        macro_rules! note_first_publish {
            ($seq:expr, $socket_role:expr, $stream_kind:expr) => {
                if first_publish_latency_ms.is_none() {
                    first_publish_latency_ms = Some(ws_start.elapsed().as_millis() as u64);
                    emit_extended_session_progress_audit(
                        "first_publish",
                        $socket_role,
                        $stream_kind,
                        first_control_frame_latency_ms,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                    );
                }
                if hedge_session_started {
                    let seed_age_ms = rest_seed_bridge_anchor_ns
                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                        .unwrap_or(0);
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&self.freshness);
                    let rearm_post_publish_fallback =
                        extended_should_rearm_post_publish_stream_fallback(hedge_mode);
                    match hedge_mode {
                        Some(ExtendedHedgeMode::BackendAttach) => {
                            emit_extended_backend_attach_fallback_audit(
                                if $socket_role == ExtendedBootstrapSocketRole::Primary {
                                    "primary_won"
                                } else {
                                    "fallback_won"
                                },
                                Some($socket_role),
                                Some($stream_kind),
                                hedge_started_at_ms,
                                connect_first_frame_timeout,
                                control_frame_only_timeout,
                                seed_age_ms,
                                rest_seed_bridge_active,
                            );
                        }
                        Some(ExtendedHedgeMode::PostPublishStreamFallback) => {
                            let action = if $socket_role == ExtendedBootstrapSocketRole::Primary {
                                "primary_recovered"
                            } else {
                                "fallback_won"
                            };
                            emit_extended_post_publish_stream_fallback_audit(
                                action,
                                active_stream_kind,
                                Some($stream_kind),
                                post_publish_fallback_active_attempt_index,
                                hedge_started_at_ms,
                                post_publish_fallback_after,
                                post_publish_fallback_deadline,
                                Some(age_ws_rx_ms),
                                Some(age_data_rx_ms),
                                Some(age_book_event_ms),
                                Some(age_published_ms),
                                last_frame_kind,
                                last_data_kind,
                                stream_preference,
                            );
                            if $socket_role == ExtendedBootstrapSocketRole::Hedge
                                && !self
                                    .prefer_full_orderbook_on_reconnect
                                    .load(Ordering::Relaxed)
                            {
                                self.prefer_full_orderbook_on_reconnect
                                    .store(true, Ordering::Relaxed);
                                emit_extended_post_publish_stream_fallback_audit(
                                    "preference_set",
                                    active_stream_kind,
                                    Some($stream_kind),
                                    post_publish_fallback_active_attempt_index,
                                    hedge_started_at_ms,
                                    post_publish_fallback_after,
                                    post_publish_fallback_deadline,
                                    Some(age_ws_rx_ms),
                                    Some(age_data_rx_ms),
                                    Some(age_book_event_ms),
                                    Some(age_published_ms),
                                    last_frame_kind,
                                    last_data_kind,
                                    ExtendedStreamPreference::FullOrderbookDegraded,
                                );
                            }
                        }
                        None => {}
                    }
                    hedge_cleanup_winner = Some($socket_role);
                    hedge_mode = None;
                    hedge_session_started = false;
                    bootstrap_control_frame_only_timeout
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_secs(86_400));
                    bootstrap_session_hedge_start
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_secs(86_400));
                    post_publish_fallback_timeout
                        .as_mut()
                        .reset(tokio::time::Instant::now() + Duration::from_secs(86_400));
                    hedge_start_armed = false;
                    if rearm_post_publish_fallback {
                        post_publish_fallback_attempted = false;
                        post_publish_fallback_active_attempt_index = None;
                    }
                }
                if !first_publish_observed {
                    first_publish_observed = true;
                    first_publish_observed_watchdog.store(true, Ordering::Relaxed);
                    if rest_seed_bridge_active {
                        let seed_age_ms = rest_seed_bridge_anchor_ns
                            .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                            .unwrap_or(0);
                        emit_extended_bootstrap_seed_bridge_audit(
                            "cleared",
                            false,
                            seed_age_ms,
                            venue_state_stale_ms,
                            connect_first_frame_timeout,
                            Some("first_publish"),
                        );
                        rest_seed_bridge_active = false;
                        rest_seed_bridge_anchor_ns = None;
                    }
                    if !fixture_mode {
                        watchdog_armed.store(true, Ordering::Relaxed);
                        emit_extended_watchdog_bootstrap_transition_audit(
                            first_publish_latency_ms,
                            true,
                        );
                    }
                }
                last_publish_seq = $seq;
            };
        }
        macro_rules! clear_rest_seed_bridge {
            ($reason:expr) => {
                if rest_seed_bridge_active {
                    let seed_age_ms = rest_seed_bridge_anchor_ns
                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                        .unwrap_or(0);
                    emit_extended_bootstrap_seed_bridge_audit(
                        "cleared",
                        false,
                        seed_age_ms,
                        venue_state_stale_ms,
                        connect_first_frame_timeout,
                        Some($reason),
                    );
                }
            };
        }
        macro_rules! emit_ws_msg_audit {
            ($reason:expr) => {{
                if ws_msg_audit_enabled {
                    let now_ns = mono_now_ns();
                    let age_ws_rx_ms =
                        age_ms(now_ns, self.freshness.last_ws_rx_ns.load(Ordering::Relaxed));
                    let age_data_rx_ms = age_ms(
                        now_ns,
                        self.freshness.last_data_rx_ns.load(Ordering::Relaxed),
                    );
                    let age_parsed_ms = age_ms(
                        now_ns,
                        self.freshness.last_parsed_ns.load(Ordering::Relaxed),
                    );
                    let age_book_event_ms = age_ms(
                        now_ns,
                        self.freshness.last_book_event_ns.load(Ordering::Relaxed),
                    );
                    let age_published_ms = age_ms(
                        now_ns,
                        self.freshness.last_published_ns.load(Ordering::Relaxed),
                    );
                    let reason = $reason.unwrap_or("periodic");
                    eprintln!(
                        concat!(
                            "WS_AUDIT venue=extended component=ws_msg reason={} interval_ms=1000 ",
                            "frames_text={} frames_bin={} ping={} pong={} close={} other={} ",
                            "cleaned={} parse_ok={} parse_err={} snap_evt={} delta_evt={} ",
                            "delta_none={} snapshot_evt_count={} delta_evt_count={} delta_none_count={} ",
                            "publish_ok={} publish_err={} max_gap_ms={} stale_ms={} ",
                            "last_frame_kind={} last_data_kind={} last_seq={} last_snapshot_seq={} ",
                            "last_book_seq={} last_publish_seq={} ",
                            "age_ws_rx_ms={} age_data_rx_ms={} age_parsed_ms={} ",
                            "age_book_event_ms={} age_published_ms={}",
                        ),
                        reason,
                        frames_text,
                        frames_binary,
                        frames_ping,
                        frames_pong,
                        frames_close,
                        frames_other,
                        cleaned_payload,
                        parse_update_ok,
                        parse_update_err,
                        ws_snapshot_parsed,
                        ws_delta_outcome_some,
                        ws_delta_outcome_none,
                        ws_snapshot_parsed,
                        ws_delta_outcome_some,
                        ws_delta_outcome_none,
                        publish_ok,
                        publish_err,
                        max_gap_ms,
                        stale_ms,
                        last_frame_kind,
                        last_data_kind,
                        last_seq,
                        last_snapshot_seq,
                        last_book_seq,
                        last_publish_seq,
                        age_ws_rx_ms,
                        age_data_rx_ms,
                        age_parsed_ms,
                        age_book_event_ms,
                        age_published_ms
                    );
                }
            }};
        }
        type ExtendedWsPollResult = Result<
            Option<Result<Message, tokio_tungstenite::tungstenite::Error>>,
            tokio::time::error::Elapsed,
        >;
        enum ExtendedSocketNext {
            Primary(ExtendedWsPollResult),
            Hedge(ExtendedWsPollResult),
        }
        loop {
            if !fixture_mode {
                sync_degraded_stream_watchdog!();
            }
            if ws_msg_audit_enabled && last_audit_instant.elapsed() >= Duration::from_millis(1000) {
                emit_ws_msg_audit!(None::<&str>);
                last_audit_instant = Instant::now();
            }
            let next = tokio::select! {
                biased;
                _ = &mut degraded_stream_watchdog_rx, if !fixture_mode => {
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&self.freshness);
                    emit_ws_msg_audit!(Some(ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap.as_str()));
                    emit_extended_post_publish_stream_fallback_audit(
                        "degraded_rebootstrap_started",
                        active_stream_kind,
                        None,
                        None,
                        Some(ws_start.elapsed().as_millis() as u64),
                        post_publish_fallback_after,
                        post_publish_fallback_deadline,
                        Some(age_ws_rx_ms),
                        Some(age_data_rx_ms),
                        Some(age_book_event_ms),
                        Some(age_published_ms),
                        last_frame_kind,
                        last_data_kind,
                        if self.prefer_full_orderbook_on_reconnect.load(Ordering::Relaxed) {
                            ExtendedStreamPreference::FullOrderbookDegraded
                        } else {
                            stream_preference
                        },
                    );
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                        format!(
                            "Extended degraded full-orderbook stream stalled past {}ms url={}",
                            post_publish_fallback_after.as_millis(),
                            active_ws_url
                        ),
                    ));
                }
                _ = &mut stale_rx => {
                    emit_ws_msg_audit!(Some("stale_watchdog"));
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::StaleWatchdog,
                        format!("Extended public WS stale: freshness exceeded {stale_ms}ms"),
                    ));
                }
                _ = &mut bootstrap_session_hedge_start, if !fixture_mode && hedge_start_armed && first_message_latency_ms.is_none() && !hedge_session_started => {
                    start_session_hedge!();
                    continue;
                }
                _ = post_publish_monitor.tick(), if !fixture_mode && first_publish_observed && !hedge_session_started && !post_publish_fallback_attempted => {
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&self.freshness);
                    let fallback_after_ms = post_publish_fallback_after.as_millis() as u64;
                    if extended_should_start_post_publish_stream_fallback(
                        active_stream_kind,
                        first_publish_observed,
                        hedge_session_started || post_publish_fallback_attempted,
                        age_ws_rx_ms,
                        age_data_rx_ms,
                        age_book_event_ms,
                        age_published_ms,
                        fallback_after_ms,
                    ) {
                        start_post_publish_fallback!(
                            age_ws_rx_ms,
                            age_data_rx_ms,
                            age_book_event_ms,
                            age_published_ms
                        );
                    } else if extended_should_start_degraded_stream_rebootstrap(
                        active_stream_kind,
                        first_publish_observed,
                        hedge_session_started || post_publish_fallback_attempted,
                        age_data_rx_ms,
                        age_book_event_ms,
                        age_published_ms,
                        fallback_after_ms,
                    ) {
                        emit_ws_msg_audit!(Some(ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap.as_str()));
                        emit_extended_post_publish_stream_fallback_audit(
                            "degraded_rebootstrap_started",
                            active_stream_kind,
                            None,
                            None,
                            Some(ws_start.elapsed().as_millis() as u64),
                            post_publish_fallback_after,
                            post_publish_fallback_deadline,
                            Some(age_ws_rx_ms),
                            Some(age_data_rx_ms),
                            Some(age_book_event_ms),
                            Some(age_published_ms),
                            last_frame_kind,
                            last_data_kind,
                            if self.prefer_full_orderbook_on_reconnect.load(Ordering::Relaxed) {
                                ExtendedStreamPreference::FullOrderbookDegraded
                            } else {
                                stream_preference
                            },
                        );
                        clear_rest_seed_bridge!("reconnect_exit");
                        return Err(ExtendedPublicWsExit::new(
                            ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                            format!(
                                "Extended degraded full-orderbook stream stalled past {}ms url={}",
                                fallback_after_ms,
                                active_ws_url
                            ),
                        ));
                    }
                    continue;
                }
                _ = &mut bootstrap_first_frame_timeout, if !fixture_mode && !bootstrap_first_frame_timeout_consumed && first_message_latency_ms.is_none() => {
                    bootstrap_first_frame_timeout_consumed = true;
                    if hedge_session_started {
                        continue;
                    }
                    if extended_should_start_control_frame_only_session_hedge(
                        active_stream_kind,
                        first_control_frame_latency_ms.is_some(),
                        first_message_latency_ms.is_some(),
                        rest_seed_bridge_active,
                        hedge_session_started,
                    ) {
                        start_session_hedge!();
                        continue;
                    }
                    let bootstrap_reason = ExtendedPublicReconnectReason::BootstrapNoFirstFrame;
                    emit_ws_msg_audit!(Some(bootstrap_reason.as_str()));
                    emit_extended_bootstrap_timeout_audit(
                        bootstrap_reason,
                        extended_bootstrap_timeout_stage(false),
                        connect_first_frame_timeout,
                        connect_book_timeout,
                        Some(control_frame_only_timeout),
                        rest_snapshot_seeded,
                        rest_seed_bridge_active,
                        rest_snapshot_seq,
                        rest_snapshot_latency_ms,
                        rest_snapshot_bid_levels,
                        rest_snapshot_ask_levels,
                        rest_seed_bridge_anchor_ns.map(|then_ns| age_ms(mono_now_ns(), then_ns)),
                        first_control_frame_latency_ms,
                        first_control_frame_kind,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                        last_frame_kind,
                        last_data_kind,
                        last_seq,
                        last_snapshot_seq,
                        last_book_seq,
                        last_publish_seq,
                        false,
                        true,
                    );
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        bootstrap_reason,
                        format!(
                            "Extended public WS bootstrap no first data frame within {}ms url={}",
                            connect_first_frame_timeout.as_millis(),
                            active_ws_url
                        ),
                    ));
                }
                _ = &mut bootstrap_control_frame_only_timeout, if !fixture_mode && hedge_session_started && first_message_latency_ms.is_none() => {
                    let bootstrap_reason = ExtendedPublicReconnectReason::BootstrapControlFrameOnlyBackendAttach;
                    let seed_age_ms = rest_seed_bridge_anchor_ns
                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                        .unwrap_or(0);
                    emit_ws_msg_audit!(Some(bootstrap_reason.as_str()));
                    emit_extended_backend_attach_fallback_audit(
                        "expired",
                        None,
                        None,
                        hedge_started_at_ms,
                        connect_first_frame_timeout,
                        control_frame_only_timeout,
                        seed_age_ms,
                        rest_seed_bridge_active,
                    );
                    emit_extended_bootstrap_timeout_audit(
                        bootstrap_reason,
                        "backend_attach_fallback",
                        connect_first_frame_timeout,
                        connect_book_timeout,
                        Some(control_frame_only_timeout),
                        rest_snapshot_seeded,
                        rest_seed_bridge_active,
                        rest_snapshot_seq,
                        rest_snapshot_latency_ms,
                        rest_snapshot_bid_levels,
                        rest_snapshot_ask_levels,
                        Some(seed_age_ms),
                        first_control_frame_latency_ms,
                        first_control_frame_kind,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                        last_frame_kind,
                        last_data_kind,
                        last_seq,
                        last_snapshot_seq,
                        last_book_seq,
                        last_publish_seq,
                        false,
                        true,
                    );
                    rest_seed_bridge_active = false;
                    rest_seed_bridge_anchor_ns = None;
                    hedge_session_started = false;
                    hedge_start_armed = false;
                    if let Some(task) = hedge_connect_task.take() {
                        task.abort();
                    }
                    hedge_read = None;
                    hedge_write = None;
                    return Err(ExtendedPublicWsExit::new(
                        bootstrap_reason,
                        format!(
                            "Extended public WS backend-attach fallback exceeded {}ms url={}",
                            control_frame_only_timeout.as_millis(),
                            active_ws_url
                        ),
                    ));
                }
                _ = &mut post_publish_fallback_timeout, if !fixture_mode && hedge_session_started && hedge_mode == Some(ExtendedHedgeMode::PostPublishStreamFallback) => {
                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                        freshness_ages_ms(&self.freshness);
                    emit_ws_msg_audit!(Some(ExtendedPublicReconnectReason::PostPublishTransportGap.as_str()));
                    emit_extended_post_publish_stream_fallback_audit(
                        "expired",
                        active_stream_kind,
                        None,
                        post_publish_fallback_active_attempt_index,
                        hedge_started_at_ms,
                        post_publish_fallback_after,
                        post_publish_fallback_deadline,
                        Some(age_ws_rx_ms),
                        Some(age_data_rx_ms),
                        Some(age_book_event_ms),
                        Some(age_published_ms),
                        last_frame_kind,
                        last_data_kind,
                        if self.prefer_full_orderbook_on_reconnect.load(Ordering::Relaxed) {
                            ExtendedStreamPreference::FullOrderbookDegraded
                        } else {
                            stream_preference
                        },
                    );
                    hedge_session_started = false;
                    hedge_mode = None;
                    hedge_start_armed = false;
                    if let Some(task) = hedge_connect_task.take() {
                        task.abort();
                    }
                    hedge_read = None;
                    hedge_write = None;
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::PostPublishTransportGap,
                        format!(
                            "Extended public WS post-publish transport gap exceeded {}ms url={}",
                            post_publish_fallback_deadline.as_millis(),
                            active_ws_url
                        ),
                    ));
                }
                hedge_connect = async {
                    match hedge_connect_task.as_mut() {
                        Some(task) => Some(task.await),
                        None => None,
                    }
                }, if hedge_connect_task.is_some() && hedge_read.is_none() => {
                    match hedge_connect {
                        Some(Ok(Ok(stream))) => {
                            let (write_half, read_half) = stream.split();
                            hedge_write = Some(write_half);
                            hedge_read = Some(read_half);
                        }
                        Some(Ok(Err(_))) | Some(Err(_)) => {
                            match hedge_mode {
                                Some(ExtendedHedgeMode::BackendAttach) => {
                                    let seed_age_ms = rest_seed_bridge_anchor_ns
                                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                                        .unwrap_or(0);
                                    emit_extended_backend_attach_fallback_audit(
                                        "cancelled",
                                        None,
                                        None,
                                        hedge_started_at_ms,
                                        connect_first_frame_timeout,
                                        control_frame_only_timeout,
                                        seed_age_ms,
                                        rest_seed_bridge_active,
                                    );
                                }
                                Some(ExtendedHedgeMode::PostPublishStreamFallback) => {
                                    let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                                        freshness_ages_ms(&self.freshness);
                                    emit_extended_post_publish_stream_fallback_audit(
                                        "cancelled",
                                        active_stream_kind,
                                        None,
                                        post_publish_fallback_active_attempt_index,
                                        hedge_started_at_ms,
                                        post_publish_fallback_after,
                                        post_publish_fallback_deadline,
                                        Some(age_ws_rx_ms),
                                        Some(age_data_rx_ms),
                                        Some(age_book_event_ms),
                                        Some(age_published_ms),
                                        last_frame_kind,
                                        last_data_kind,
                                        if self.prefer_full_orderbook_on_reconnect.load(Ordering::Relaxed) {
                                            ExtendedStreamPreference::FullOrderbookDegraded
                                        } else {
                                            stream_preference
                                        },
                                    );
                                }
                                None => {}
                            }
                        }
                        None => {}
                    }
                    hedge_connect_task = None;
                    continue;
                }
                _ = &mut bootstrap_post_first_frame_timeout, if !fixture_mode && bootstrap_post_first_frame_timeout_armed && !first_publish_observed => {
                    let bootstrap_reason = extended_bootstrap_timeout_reason(
                        first_message_latency_ms.is_some(),
                        first_book_latency_ms.is_some(),
                        first_publish_latency_ms.is_some(),
                    );
                    emit_ws_msg_audit!(Some(bootstrap_reason.as_str()));
                    emit_extended_bootstrap_timeout_audit(
                        bootstrap_reason,
                        extended_bootstrap_timeout_stage(true),
                        connect_first_frame_timeout,
                        connect_book_timeout,
                        Some(control_frame_only_timeout),
                        rest_snapshot_seeded,
                        rest_seed_bridge_active,
                        rest_snapshot_seq,
                        rest_snapshot_latency_ms,
                        rest_snapshot_bid_levels,
                        rest_snapshot_ask_levels,
                        rest_seed_bridge_anchor_ns.map(|then_ns| age_ms(mono_now_ns(), then_ns)),
                        first_control_frame_latency_ms,
                        first_control_frame_kind,
                        first_message_latency_ms,
                        first_book_latency_ms,
                        first_publish_latency_ms,
                        last_frame_kind,
                        last_data_kind,
                        last_seq,
                        last_snapshot_seq,
                        last_book_seq,
                        last_publish_seq,
                        false,
                        true,
                    );
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        bootstrap_reason,
                        match bootstrap_reason {
                            ExtendedPublicReconnectReason::BootstrapNoFirstFrame => format!(
                                "Extended public WS bootstrap no first data frame within {}ms url={}",
                                connect_book_timeout.as_millis(),
                                active_ws_url
                            ),
                            ExtendedPublicReconnectReason::BootstrapFrameNoBook => format!(
                                "Extended public WS bootstrap frame/no-book within {}ms url={}",
                                connect_book_timeout.as_millis(),
                                active_ws_url
                            ),
                            ExtendedPublicReconnectReason::BootstrapBookNoPublish => format!(
                                "Extended public WS bootstrap book/no-publish within {}ms url={}",
                                connect_book_timeout.as_millis(),
                                active_ws_url
                            ),
                            _ => format!(
                                "Extended public WS bootstrap timeout within {}ms url={}",
                                connect_book_timeout.as_millis(),
                                active_ws_url
                            ),
                        },
                    ));
                }
                _ = ping_timer.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        emit_ws_msg_audit!(Some("ping_send_fail"));
                        eprintln!("WARN: Extended public WS ping send failed: {e} — reconnecting");
                        clear_rest_seed_bridge!("reconnect_exit");
                        return Err(ExtendedPublicWsExit::new(
                            ExtendedPublicReconnectReason::PingSendFail,
                            format!("Extended public WS ping send failed: {e}"),
                        ));
                    }
                    continue;
                }
                next = tokio::time::timeout(read_timeout, read.next()) => ExtendedSocketNext::Primary(next),
                next = async {
                    match hedge_read.as_mut() {
                        Some(read_half) => Some(tokio::time::timeout(read_timeout, read_half.next()).await),
                        None => None,
                    }
                }, if hedge_read.is_some() => {
                    match next {
                        Some(result) => ExtendedSocketNext::Hedge(result),
                        None => unreachable!("hedge read polled without hedge stream"),
                    }
                },
            };
            let (msg_socket_role, msg_stream_kind, msg_url, msg) = match next {
                ExtendedSocketNext::Primary(Ok(Some(Ok(msg)))) => (
                    ExtendedBootstrapSocketRole::Primary,
                    active_stream_kind,
                    active_ws_url.clone(),
                    msg,
                ),
                ExtendedSocketNext::Primary(Ok(Some(Err(err)))) => {
                    emit_ws_msg_audit!(Some("stream_closed"));
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::StreamClosed,
                        format!("Extended public WS stream read error: {err}"),
                    ));
                }
                ExtendedSocketNext::Primary(Ok(None)) => {
                    emit_ws_msg_audit!(Some("stream_closed"));
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::StreamClosed,
                        format!("Extended public WS stream ended url={active_ws_url}"),
                    ));
                }
                ExtendedSocketNext::Primary(Err(_)) => {
                    let message = if !first_message_logged {
                        format!(
                            "Extended WS received no messages after {:?} url={}",
                            read_timeout, active_ws_url
                        )
                    } else {
                        format!(
                            "Extended public WS read timeout after {}ms url={}",
                            read_timeout.as_millis(),
                            active_ws_url
                        )
                    };
                    emit_ws_msg_audit!(Some("read_timeout"));
                    clear_rest_seed_bridge!("reconnect_exit");
                    return Err(ExtendedPublicWsExit::new(
                        ExtendedPublicReconnectReason::ReadTimeout,
                        message,
                    ));
                }
                ExtendedSocketNext::Hedge(Ok(Some(Ok(msg)))) => (
                    ExtendedBootstrapSocketRole::Hedge,
                    fallback_stream_kind,
                    fallback_ws_url.clone(),
                    msg,
                ),
                ExtendedSocketNext::Hedge(Ok(Some(Err(_))))
                | ExtendedSocketNext::Hedge(Ok(None))
                | ExtendedSocketNext::Hedge(Err(_)) => {
                    let seed_age_ms = rest_seed_bridge_anchor_ns
                        .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                        .unwrap_or(0);
                    emit_extended_backend_attach_fallback_audit(
                        "cancelled",
                        None,
                        None,
                        hedge_started_at_ms,
                        connect_first_frame_timeout,
                        control_frame_only_timeout,
                        seed_age_ms,
                        rest_seed_bridge_active,
                    );
                    hedge_read = None;
                    hedge_write = None;
                    hedge_connect_task = None;
                    continue;
                }
            };
            let now_ns = mono_now_ns();
            if last_rx_mono_ns != 0 {
                let gap_ms = age_ms(now_ns, last_rx_mono_ns);
                if gap_ms > max_gap_ms {
                    max_gap_ms = gap_ms;
                }
            }
            last_rx_mono_ns = now_ns;
            self.freshness
                .last_ws_rx_ns
                .store(now_ns, Ordering::Relaxed);
            match msg {
                Message::Text(text) => {
                    last_frame_kind = "text";
                    frames_text += 1;
                    note_first_message!(msg_socket_role, msg_stream_kind);
                    let Some(cleaned) = clean_ws_payload(&text) else {
                        continue;
                    };
                    cleaned_payload += 1;
                    self.freshness
                        .last_data_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_ws_message_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let keys = value
                                .as_object()
                                .map(|obj| {
                                    let mut keys: Vec<&str> =
                                        obj.keys().map(|k| k.as_str()).collect();
                                    keys.sort();
                                    format!("[{}]", keys.join(","))
                                })
                                .unwrap_or_else(|| "[non-object]".to_string());
                            let snippet: String = cleaned.chars().take(160).collect();
                            eprintln!(
                                "INFO: Extended public WS first message keys={} snippet={}",
                                keys, snippet
                            );
                            first_ws_keys = Some(keys);
                            first_ws_snippet = Some(snippet);
                            first_ws_message_logged = true;
                        }
                    }
                    if let Some(recorder) = self.recorder.as_ref() {
                        let mut guard = recorder.lock().await;
                        let _ = guard.record_ws_frame(cleaned);
                    }
                    let update = match parse_depth_update(cleaned) {
                        Ok(update) => {
                            parse_update_ok += 1;
                            update
                        }
                        Err(err) => {
                            last_data_kind = "parse_error";
                            parse_update_err += 1;
                            consecutive_parse_errors += 1;
                            if consecutive_parse_errors == 1
                                || consecutive_parse_errors.is_multiple_of(10)
                            {
                                let snippet: String = cleaned.chars().take(160).collect();
                                eprintln!(
                                    "WARN: Extended public WS parse error: {err} url={} snippet={}",
                                    msg_url, snippet
                                );
                            }
                            if consecutive_parse_errors > MAX_PARSE_ERRORS {
                                emit_ws_msg_audit!(Some("parse_error"));
                                eprintln!(
                                    "Extended public WS too many parse errors; reconnecting url={}",
                                    msg_url
                                );
                                clear_rest_seed_bridge!("reconnect_exit");
                                return Err(ExtendedPublicWsExit::new(
                                    ExtendedPublicReconnectReason::ParseError,
                                    format!(
                                        "Extended public WS too many parse errors; reconnecting url={}",
                                        msg_url
                                    ),
                                ));
                            }
                            continue;
                        }
                    };
                    let Some(update) = update else {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let (seq_state_ref, ws_snapshot_seq_ref) = match msg_socket_role {
                                ExtendedBootstrapSocketRole::Primary => {
                                    (&mut seq_state, &mut ws_snapshot_seq)
                                }
                                ExtendedBootstrapSocketRole::Hedge => (
                                    hedge_seq_state.get_or_insert_with(|| {
                                        ExtendedSeqState::new(None, self.cfg.venue_index)
                                    }),
                                    &mut hedge_ws_snapshot_seq,
                                ),
                            };
                            let mut parsed_snapshot = false;
                            if let Some(event) = parse_depth_snapshot_from_ws(
                                &value,
                                &self.cfg.market,
                                self.cfg.venue_index,
                                ws_snapshot_seq_ref,
                            ) {
                                parsed_snapshot = true;
                                ws_snapshot_parsed += 1;
                                let seq = match &event {
                                    MarketDataEvent::L2Snapshot(snapshot) => snapshot.seq,
                                    MarketDataEvent::L2Delta(delta) => delta.seq,
                                    _ => 0,
                                };
                                last_data_kind = "snapshot";
                                last_seq = seq;
                                last_snapshot_seq = seq;
                                last_book_seq = seq;
                                if seq > 0 {
                                    match seq_state_ref.observe_seq(seq) {
                                        Ok(should_apply) => {
                                            if !should_apply {
                                                continue;
                                            }
                                        }
                                        Err(err) => {
                                            let msg = err.to_string();
                                            let reason = extended_seq_error_reason(&msg);
                                            emit_ws_msg_audit!(Some(reason.as_str()));
                                            clear_rest_seed_bridge!("reconnect_exit");
                                            return Err(ExtendedPublicWsExit::new(reason, msg));
                                        }
                                    }
                                }
                                if !first_book_update_logged {
                                    eprintln!("INFO: Extended public WS first book update");
                                    first_book_update_logged = true;
                                    first_book_latency_ms =
                                        Some(ws_start.elapsed().as_millis() as u64);
                                    emit_extended_session_progress_audit(
                                        "first_book",
                                        msg_socket_role,
                                        msg_stream_kind,
                                        first_control_frame_latency_ms,
                                        first_message_latency_ms,
                                        first_book_latency_ms,
                                        first_publish_latency_ms,
                                    );
                                }
                                let now_ns = mono_now_ns();
                                self.freshness
                                    .last_parsed_ns
                                    .store(now_ns, Ordering::Relaxed);
                                self.freshness
                                    .last_book_event_ns
                                    .store(now_ns, Ordering::Relaxed);
                                match self.publish_market(event).await {
                                    Ok(()) => {
                                        publish_ok += 1;
                                        note_first_publish!(seq, msg_socket_role, msg_stream_kind);
                                    }
                                    Err(err) => {
                                        publish_err += 1;
                                        eprintln!("Extended public WS market send failed: {err}");
                                    }
                                }
                            }
                            if !parsed_snapshot {
                                last_data_kind = "non_book";
                            }
                            if !first_decoded_top_logged {
                                if let Some(top) = decode_top_from_value(&value) {
                                    eprintln!(
                                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                        top.best_bid_px,
                                        top.best_bid_sz,
                                        top.best_ask_px,
                                        top.best_ask_sz
                                    );
                                    first_decoded_top_logged = true;
                                }
                            }
                        }
                        continue;
                    };
                    self.freshness
                        .last_parsed_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    last_seq = update.seq;
                    if !first_decoded_top_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(top) = decode_top_from_value(&value) {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px,
                                    top.best_bid_sz,
                                    top.best_ask_px,
                                    top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged {
                            if let Some(top) = decode_top_from_update(&update, update.event_time) {
                                eprintln!(
                                "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                top.best_bid_px,
                                top.best_bid_sz,
                                top.best_ask_px,
                                top.best_ask_sz
                            );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged && decode_miss_count < 3 {
                            decode_miss_count += 1;
                            if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                                log_decode_miss(
                                    "Extended",
                                    &value,
                                    cleaned,
                                    decode_miss_count,
                                    msg_url.as_str(),
                                );
                            }
                        }
                    }
                    if !symbol_matches(&update.symbol, &self.cfg.market) {
                        continue;
                    }
                    let seq_state_ref = match msg_socket_role {
                        ExtendedBootstrapSocketRole::Primary => &mut seq_state,
                        ExtendedBootstrapSocketRole::Hedge => {
                            hedge_seq_state.get_or_insert_with(|| {
                                ExtendedSeqState::new(None, self.cfg.venue_index)
                            })
                        }
                    };
                    let outcome = match seq_state_ref.apply_update(&update) {
                        Ok(outcome) => outcome,
                        Err(err) => {
                            let msg = err.to_string();
                            let reason = extended_seq_error_reason(&msg);
                            emit_ws_msg_audit!(Some(reason.as_str()));
                            clear_rest_seed_bridge!("reconnect_exit");
                            return Err(ExtendedPublicWsExit::new(reason, msg));
                        }
                    };
                    if let Some(event) = outcome {
                        ws_delta_outcome_some += 1;
                        last_data_kind = "delta";
                        consecutive_parse_errors = 0;
                        self.freshness
                            .last_book_event_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        last_book_seq = update.seq;
                        if !first_book_update_logged {
                            eprintln!("INFO: Extended public WS first book update");
                            first_book_update_logged = true;
                            first_book_latency_ms = Some(ws_start.elapsed().as_millis() as u64);
                            emit_extended_session_progress_audit(
                                "first_book",
                                msg_socket_role,
                                msg_stream_kind,
                                first_control_frame_latency_ms,
                                first_message_latency_ms,
                                first_book_latency_ms,
                                first_publish_latency_ms,
                            );
                        }
                        match self.publish_market(event).await {
                            Ok(()) => {
                                publish_ok += 1;
                                note_first_publish!(update.seq, msg_socket_role, msg_stream_kind);
                            }
                            Err(err) => {
                                publish_err += 1;
                                eprintln!("Extended public WS market send failed: {err}");
                            }
                        }
                    } else {
                        ws_delta_outcome_none += 1;
                        last_data_kind = "delta_none";
                    }
                }
                Message::Binary(bytes) => {
                    last_frame_kind = "binary";
                    frames_binary += 1;
                    note_first_message!(msg_socket_role, msg_stream_kind);
                    let text = String::from_utf8_lossy(&bytes);
                    let Some(cleaned) = clean_ws_payload(&text) else {
                        continue;
                    };
                    cleaned_payload += 1;
                    self.freshness
                        .last_data_rx_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    if !first_ws_message_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let keys = value
                                .as_object()
                                .map(|obj| {
                                    let mut keys: Vec<&str> =
                                        obj.keys().map(|k| k.as_str()).collect();
                                    keys.sort();
                                    format!("[{}]", keys.join(","))
                                })
                                .unwrap_or_else(|| "[non-object]".to_string());
                            let snippet: String = cleaned.chars().take(160).collect();
                            eprintln!(
                                "INFO: Extended public WS first message keys={} snippet={}",
                                keys, snippet
                            );
                            first_ws_keys = Some(keys);
                            first_ws_snippet = Some(snippet);
                            first_ws_message_logged = true;
                        }
                    }
                    if let Some(recorder) = self.recorder.as_ref() {
                        let mut guard = recorder.lock().await;
                        let _ = guard.record_ws_frame(cleaned);
                    }
                    let update = match parse_depth_update(cleaned) {
                        Ok(update) => {
                            parse_update_ok += 1;
                            update
                        }
                        Err(err) => {
                            last_data_kind = "parse_error";
                            parse_update_err += 1;
                            consecutive_parse_errors += 1;
                            if consecutive_parse_errors == 1
                                || consecutive_parse_errors.is_multiple_of(10)
                            {
                                let snippet: String = cleaned.chars().take(160).collect();
                                eprintln!(
                                    "WARN: Extended public WS parse error: {err} url={} snippet={}",
                                    msg_url, snippet
                                );
                            }
                            if consecutive_parse_errors > MAX_PARSE_ERRORS {
                                emit_ws_msg_audit!(Some("parse_error"));
                                eprintln!(
                                    "Extended public WS too many parse errors; reconnecting url={}",
                                    msg_url
                                );
                                clear_rest_seed_bridge!("reconnect_exit");
                                return Err(ExtendedPublicWsExit::new(
                                    ExtendedPublicReconnectReason::ParseError,
                                    format!(
                                        "Extended public WS too many parse errors; reconnecting url={}",
                                        msg_url
                                    ),
                                ));
                            }
                            continue;
                        }
                    };
                    let Some(update) = update else {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            let (seq_state_ref, ws_snapshot_seq_ref) = match msg_socket_role {
                                ExtendedBootstrapSocketRole::Primary => {
                                    (&mut seq_state, &mut ws_snapshot_seq)
                                }
                                ExtendedBootstrapSocketRole::Hedge => (
                                    hedge_seq_state.get_or_insert_with(|| {
                                        ExtendedSeqState::new(None, self.cfg.venue_index)
                                    }),
                                    &mut hedge_ws_snapshot_seq,
                                ),
                            };
                            let mut parsed_snapshot = false;
                            if let Some(event) = parse_depth_snapshot_from_ws(
                                &value,
                                &self.cfg.market,
                                self.cfg.venue_index,
                                ws_snapshot_seq_ref,
                            ) {
                                parsed_snapshot = true;
                                ws_snapshot_parsed += 1;
                                let seq = match &event {
                                    MarketDataEvent::L2Snapshot(snapshot) => snapshot.seq,
                                    MarketDataEvent::L2Delta(delta) => delta.seq,
                                    _ => 0,
                                };
                                last_data_kind = "snapshot";
                                last_seq = seq;
                                last_snapshot_seq = seq;
                                last_book_seq = seq;
                                if seq > 0 {
                                    match seq_state_ref.observe_seq(seq) {
                                        Ok(should_apply) => {
                                            if !should_apply {
                                                continue;
                                            }
                                        }
                                        Err(err) => {
                                            let msg = err.to_string();
                                            let reason = extended_seq_error_reason(&msg);
                                            emit_ws_msg_audit!(Some(reason.as_str()));
                                            clear_rest_seed_bridge!("reconnect_exit");
                                            return Err(ExtendedPublicWsExit::new(reason, msg));
                                        }
                                    }
                                }
                                if !first_book_update_logged {
                                    eprintln!("INFO: Extended public WS first book update");
                                    first_book_update_logged = true;
                                    first_book_latency_ms =
                                        Some(ws_start.elapsed().as_millis() as u64);
                                    emit_extended_session_progress_audit(
                                        "first_book",
                                        msg_socket_role,
                                        msg_stream_kind,
                                        first_control_frame_latency_ms,
                                        first_message_latency_ms,
                                        first_book_latency_ms,
                                        first_publish_latency_ms,
                                    );
                                }
                                let now_ns = mono_now_ns();
                                self.freshness
                                    .last_parsed_ns
                                    .store(now_ns, Ordering::Relaxed);
                                self.freshness
                                    .last_book_event_ns
                                    .store(now_ns, Ordering::Relaxed);
                                match self.publish_market(event).await {
                                    Ok(()) => {
                                        publish_ok += 1;
                                        note_first_publish!(seq, msg_socket_role, msg_stream_kind);
                                    }
                                    Err(err) => {
                                        publish_err += 1;
                                        eprintln!("Extended public WS market send failed: {err}");
                                    }
                                }
                            }
                            if !parsed_snapshot {
                                last_data_kind = "non_book";
                            }
                            if !first_decoded_top_logged {
                                if let Some(top) = decode_top_from_value(&value) {
                                    eprintln!(
                                        "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                        top.best_bid_px,
                                        top.best_bid_sz,
                                        top.best_ask_px,
                                        top.best_ask_sz
                                    );
                                    first_decoded_top_logged = true;
                                }
                            }
                        }
                        continue;
                    };
                    self.freshness
                        .last_parsed_ns
                        .store(mono_now_ns(), Ordering::Relaxed);
                    last_seq = update.seq;
                    if !first_decoded_top_logged {
                        if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                            if let Some(top) = decode_top_from_value(&value) {
                                eprintln!(
                                    "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                    top.best_bid_px,
                                    top.best_bid_sz,
                                    top.best_ask_px,
                                    top.best_ask_sz
                                );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged {
                            if let Some(top) = decode_top_from_update(&update, update.event_time) {
                                eprintln!(
                                "FIRST_DECODED_TOP venue=extended bid_px={} bid_sz={} ask_px={} ask_sz={}",
                                top.best_bid_px,
                                top.best_bid_sz,
                                top.best_ask_px,
                                top.best_ask_sz
                            );
                                first_decoded_top_logged = true;
                            }
                        }
                        if !first_decoded_top_logged && decode_miss_count < 3 {
                            decode_miss_count += 1;
                            if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
                                log_decode_miss(
                                    "Extended",
                                    &value,
                                    cleaned,
                                    decode_miss_count,
                                    msg_url.as_str(),
                                );
                            }
                        }
                    }
                    if !symbol_matches(&update.symbol, &self.cfg.market) {
                        continue;
                    }
                    let seq_state_ref = match msg_socket_role {
                        ExtendedBootstrapSocketRole::Primary => &mut seq_state,
                        ExtendedBootstrapSocketRole::Hedge => {
                            hedge_seq_state.get_or_insert_with(|| {
                                ExtendedSeqState::new(None, self.cfg.venue_index)
                            })
                        }
                    };
                    let outcome = match seq_state_ref.apply_update(&update) {
                        Ok(outcome) => outcome,
                        Err(err) => {
                            let msg = err.to_string();
                            let reason = extended_seq_error_reason(&msg);
                            emit_ws_msg_audit!(Some(reason.as_str()));
                            clear_rest_seed_bridge!("reconnect_exit");
                            return Err(ExtendedPublicWsExit::new(reason, msg));
                        }
                    };
                    if let Some(event) = outcome {
                        ws_delta_outcome_some += 1;
                        last_data_kind = "delta";
                        consecutive_parse_errors = 0;
                        self.freshness
                            .last_book_event_ns
                            .store(mono_now_ns(), Ordering::Relaxed);
                        last_book_seq = update.seq;
                        if !first_book_update_logged {
                            eprintln!("INFO: Extended public WS first book update");
                            first_book_update_logged = true;
                            first_book_latency_ms = Some(ws_start.elapsed().as_millis() as u64);
                            emit_extended_session_progress_audit(
                                "first_book",
                                msg_socket_role,
                                msg_stream_kind,
                                first_control_frame_latency_ms,
                                first_message_latency_ms,
                                first_book_latency_ms,
                                first_publish_latency_ms,
                            );
                        }
                        match self.publish_market(event).await {
                            Ok(()) => {
                                publish_ok += 1;
                                note_first_publish!(update.seq, msg_socket_role, msg_stream_kind);
                            }
                            Err(err) => {
                                publish_err += 1;
                                eprintln!("Extended public WS market send failed: {err}");
                            }
                        }
                    } else {
                        ws_delta_outcome_none += 1;
                        last_data_kind = "delta_none";
                    }
                }
                Message::Ping(payload) => {
                    last_frame_kind = "ping";
                    frames_ping += 1;
                    note_first_control_frame!("ping", msg_socket_role, msg_stream_kind);
                    let pong_result = match msg_socket_role {
                        ExtendedBootstrapSocketRole::Primary => {
                            write.send(Message::Pong(payload)).await
                        }
                        ExtendedBootstrapSocketRole::Hedge => match hedge_write.as_mut() {
                            Some(hedge_writer) => hedge_writer.send(Message::Pong(payload)).await,
                            None => Ok(()),
                        },
                    };
                    if let Err(err) = pong_result {
                        emit_ws_msg_audit!(Some("ping_send_fail"));
                        if msg_socket_role == ExtendedBootstrapSocketRole::Primary {
                            clear_rest_seed_bridge!("reconnect_exit");
                            return Err(ExtendedPublicWsExit::new(
                                ExtendedPublicReconnectReason::PingSendFail,
                                format!("Extended public WS pong send failed: {err}"),
                            ));
                        }
                        match hedge_mode {
                            Some(ExtendedHedgeMode::BackendAttach) => {
                                let seed_age_ms = rest_seed_bridge_anchor_ns
                                    .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                                    .unwrap_or(0);
                                emit_extended_backend_attach_fallback_audit(
                                    "cancelled",
                                    None,
                                    None,
                                    hedge_started_at_ms,
                                    connect_first_frame_timeout,
                                    control_frame_only_timeout,
                                    seed_age_ms,
                                    rest_seed_bridge_active,
                                );
                            }
                            Some(ExtendedHedgeMode::PostPublishStreamFallback) => {
                                let (
                                    age_ws_rx_ms,
                                    age_data_rx_ms,
                                    age_book_event_ms,
                                    age_published_ms,
                                ) = freshness_ages_ms(&self.freshness);
                                emit_extended_post_publish_stream_fallback_audit(
                                    "cancelled",
                                    active_stream_kind,
                                    None,
                                    post_publish_fallback_active_attempt_index,
                                    hedge_started_at_ms,
                                    post_publish_fallback_after,
                                    post_publish_fallback_deadline,
                                    Some(age_ws_rx_ms),
                                    Some(age_data_rx_ms),
                                    Some(age_book_event_ms),
                                    Some(age_published_ms),
                                    last_frame_kind,
                                    last_data_kind,
                                    if self
                                        .prefer_full_orderbook_on_reconnect
                                        .load(Ordering::Relaxed)
                                    {
                                        ExtendedStreamPreference::FullOrderbookDegraded
                                    } else {
                                        stream_preference
                                    },
                                );
                            }
                            None => {}
                        }
                        hedge_read = None;
                        hedge_write = None;
                        hedge_connect_task = None;
                        hedge_mode = None;
                        hedge_session_started = false;
                    }
                }
                Message::Pong(_) => {
                    last_frame_kind = "pong";
                    frames_pong += 1;
                    note_first_control_frame!("pong", msg_socket_role, msg_stream_kind);
                }
                Message::Close(_) => {
                    last_frame_kind = "close";
                    frames_close += 1;
                    note_first_control_frame!("close", msg_socket_role, msg_stream_kind);
                    if msg_socket_role == ExtendedBootstrapSocketRole::Primary {
                        emit_ws_msg_audit!(Some("stream_closed"));
                        clear_rest_seed_bridge!("reconnect_exit");
                        return Err(ExtendedPublicWsExit::new(
                            ExtendedPublicReconnectReason::StreamClosed,
                            format!("Extended WS closed; reconnecting url={}", active_ws_url),
                        ));
                    }
                    match hedge_mode {
                        Some(ExtendedHedgeMode::BackendAttach) => {
                            let seed_age_ms = rest_seed_bridge_anchor_ns
                                .map(|then_ns| age_ms(mono_now_ns(), then_ns))
                                .unwrap_or(0);
                            emit_extended_backend_attach_fallback_audit(
                                "cancelled",
                                None,
                                None,
                                hedge_started_at_ms,
                                connect_first_frame_timeout,
                                control_frame_only_timeout,
                                seed_age_ms,
                                rest_seed_bridge_active,
                            );
                        }
                        Some(ExtendedHedgeMode::PostPublishStreamFallback) => {
                            let (age_ws_rx_ms, age_data_rx_ms, age_book_event_ms, age_published_ms) =
                                freshness_ages_ms(&self.freshness);
                            emit_extended_post_publish_stream_fallback_audit(
                                "cancelled",
                                active_stream_kind,
                                None,
                                post_publish_fallback_active_attempt_index,
                                hedge_started_at_ms,
                                post_publish_fallback_after,
                                post_publish_fallback_deadline,
                                Some(age_ws_rx_ms),
                                Some(age_data_rx_ms),
                                Some(age_book_event_ms),
                                Some(age_published_ms),
                                last_frame_kind,
                                last_data_kind,
                                if self
                                    .prefer_full_orderbook_on_reconnect
                                    .load(Ordering::Relaxed)
                                {
                                    ExtendedStreamPreference::FullOrderbookDegraded
                                } else {
                                    stream_preference
                                },
                            );
                        }
                        None => {}
                    }
                    hedge_read = None;
                    hedge_write = None;
                    hedge_connect_task = None;
                    hedge_mode = None;
                    hedge_session_started = false;
                }
                _ => {
                    last_frame_kind = "other";
                    frames_other += 1;
                    note_first_control_frame!("other", msg_socket_role, msg_stream_kind);
                }
            }
            if !first_decoded_top_logged
                && !no_book_warned
                && ws_start.elapsed() >= Duration::from_secs(10)
            {
                let keys = first_ws_keys.as_deref().unwrap_or("unknown");
                let snippet = first_ws_snippet.as_deref().unwrap_or("unknown");
                eprintln!(
                    "WARN: Extended WS no book decoded after 10s url={} keys={} snippet={}",
                    active_ws_url, keys, snippet
                );
                no_book_warned = true;
            }
            if let Some(winner) = hedge_cleanup_winner.take() {
                if let Some(task) = hedge_connect_task.take() {
                    task.abort();
                }
                match winner {
                    ExtendedBootstrapSocketRole::Primary => {
                        if let Some(mut loser_write) = hedge_write.take() {
                            let _ = loser_write.send(Message::Close(None)).await;
                        }
                        hedge_read = None;
                    }
                    ExtendedBootstrapSocketRole::Hedge => {
                        if let Some(new_write) = hedge_write.take() {
                            let mut old_write = std::mem::replace(&mut write, new_write);
                            let _ = old_write.send(Message::Close(None)).await;
                        }
                        if let Some(new_read) = hedge_read.take() {
                            let _old_read = std::mem::replace(&mut read, new_read);
                        }
                        active_ws_url = fallback_ws_url.clone();
                        active_stream_kind = fallback_stream_kind;
                        if let Some(new_seq_state) = hedge_seq_state.take() {
                            seq_state = new_seq_state;
                            ws_snapshot_seq = hedge_ws_snapshot_seq;
                        }
                    }
                }
                if winner == ExtendedBootstrapSocketRole::Primary {
                    hedge_seq_state = None;
                    hedge_ws_snapshot_seq = 0;
                }
                hedge_started_at_ms = None;
            }
        }
        unreachable!("Extended public WS loop should always exit via reconnect outcome")
    }

    async fn fetch_snapshot(&self) -> ExtendedRestSnapshotSeedAttempt {
        let url = format!(
            "{}/api/v1/info/markets/{}/orderbook",
            self.cfg.rest_url.trim_end_matches('/'),
            self.cfg.market
        );
        let started_at = Instant::now();
        let response = match self.http.get(&url).send().await {
            Ok(response) => response,
            Err(_) => {
                let attempt = ExtendedRestSnapshotSeedAttempt {
                    raw: None,
                    snapshot: None,
                    status: "http_error",
                    http_status: None,
                    latency_ms: started_at.elapsed().as_millis() as u64,
                    bid_levels: 0,
                    ask_levels: 0,
                };
                emit_extended_rest_snapshot_seed_audit(
                    attempt.status,
                    attempt.http_status,
                    attempt.latency_ms,
                    false,
                    attempt.bid_levels,
                    attempt.ask_levels,
                    &self.cfg.market,
                );
                return attempt;
            }
        };
        let response_status = response.status();
        let http_status = Some(response_status.as_u16());
        let response_is_success = response_status.is_success();
        let raw = match response.text().await {
            Ok(raw) => raw,
            Err(_) => {
                let attempt = ExtendedRestSnapshotSeedAttempt {
                    raw: None,
                    snapshot: None,
                    status: "http_error",
                    http_status,
                    latency_ms: started_at.elapsed().as_millis() as u64,
                    bid_levels: 0,
                    ask_levels: 0,
                };
                emit_extended_rest_snapshot_seed_audit(
                    attempt.status,
                    attempt.http_status,
                    attempt.latency_ms,
                    false,
                    attempt.bid_levels,
                    attempt.ask_levels,
                    &self.cfg.market,
                );
                return attempt;
            }
        };
        let latency_ms = started_at.elapsed().as_millis() as u64;
        if !response_is_success {
            let attempt = ExtendedRestSnapshotSeedAttempt {
                raw: Some(raw),
                snapshot: None,
                status: "http_error",
                http_status,
                latency_ms,
                bid_levels: 0,
                ask_levels: 0,
            };
            emit_extended_rest_snapshot_seed_audit(
                attempt.status,
                attempt.http_status,
                attempt.latency_ms,
                false,
                attempt.bid_levels,
                attempt.ask_levels,
                &self.cfg.market,
            );
            return attempt;
        }
        let cleaned = match clean_ws_payload(&raw) {
            Some(cleaned) => cleaned,
            None => {
                let attempt = ExtendedRestSnapshotSeedAttempt {
                    raw: Some(raw),
                    snapshot: None,
                    status: "empty",
                    http_status,
                    latency_ms,
                    bid_levels: 0,
                    ask_levels: 0,
                };
                emit_extended_rest_snapshot_seed_audit(
                    attempt.status,
                    attempt.http_status,
                    attempt.latency_ms,
                    false,
                    attempt.bid_levels,
                    attempt.ask_levels,
                    &self.cfg.market,
                );
                return attempt;
            }
        };
        let value: Value = match serde_json::from_str(cleaned) {
            Ok(value) => value,
            Err(_) => {
                let attempt = ExtendedRestSnapshotSeedAttempt {
                    raw: Some(raw),
                    snapshot: None,
                    status: "parse_error",
                    http_status,
                    latency_ms,
                    bid_levels: 0,
                    ask_levels: 0,
                };
                emit_extended_rest_snapshot_seed_audit(
                    attempt.status,
                    attempt.http_status,
                    attempt.latency_ms,
                    false,
                    attempt.bid_levels,
                    attempt.ask_levels,
                    &self.cfg.market,
                );
                return attempt;
            }
        };
        let Some(snapshot) = parse_depth_snapshot(&value) else {
            let attempt = ExtendedRestSnapshotSeedAttempt {
                raw: Some(raw),
                snapshot: None,
                status: "parse_error",
                http_status,
                latency_ms,
                bid_levels: 0,
                ask_levels: 0,
            };
            emit_extended_rest_snapshot_seed_audit(
                attempt.status,
                attempt.http_status,
                attempt.latency_ms,
                false,
                attempt.bid_levels,
                attempt.ask_levels,
                &self.cfg.market,
            );
            return attempt;
        };
        let bid_levels = snapshot.bids.len();
        let ask_levels = snapshot.asks.len();
        let seeded = bid_levels > 0 && ask_levels > 0;
        let status = if seeded { "ok" } else { "empty" };
        let attempt = ExtendedRestSnapshotSeedAttempt {
            raw: Some(raw),
            snapshot: if seeded { Some(snapshot) } else { None },
            status,
            http_status,
            latency_ms,
            bid_levels,
            ask_levels,
        };
        emit_extended_rest_snapshot_seed_audit(
            attempt.status,
            attempt.http_status,
            attempt.latency_ms,
            seeded,
            attempt.bid_levels,
            attempt.ask_levels,
            &self.cfg.market,
        );
        attempt
    }
}

#[derive(Clone)]
pub struct ExtendedRestClient {
    cfg: ExtendedConfig,
    http: Client,
    price_tick_size: Arc<Mutex<Option<f64>>>,
    poll_seq: Arc<AtomicU64>,
}

impl ExtendedRestClient {
    pub fn new(cfg: ExtendedConfig) -> Self {
        Self {
            cfg,
            http: Client::builder()
                .user_agent("paraphina")
                .timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(5)
                .build()
                .expect("extended rest http client build"),
            price_tick_size: Arc::new(Mutex::new(None)),
            poll_seq: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn has_account_auth(&self) -> bool {
        self.cfg.has_account_auth()
    }

    pub fn has_private_read_auth(&self) -> bool {
        self.cfg.has_private_read_auth()
    }

    pub fn has_execution_auth(&self) -> bool {
        self.cfg.has_execution_auth()
    }

    async fn resolve_price_tick_size(&self) -> LiveResult<f64> {
        if let Some(tick_size) = std::env::var("PARAPHINA_EXTENDED_PRICE_TICK_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<f64>().ok())
            .filter(|tick_size| tick_size.is_finite() && *tick_size > 0.0)
        {
            return Ok(tick_size);
        }

        {
            let guard = self.price_tick_size.lock().await;
            if let Some(tick_size) = *guard {
                return Ok(tick_size);
            }
        }

        let url = format!(
            "{}/api/v1/info/markets",
            self.cfg.rest_url.trim_end_matches('/')
        );
        let resp = self
            .http
            .get(&url)
            .query(&[("market", self.cfg.market.as_str())])
            .send()
            .await
            .map_err(|err| {
                LiveGatewayError::retryable(format!("extended market info error: {err}"))
            })?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_rest_error(status.as_u16(), &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LiveGatewayError::fatal(format!("extended market info parse error: {err}"))
        })?;
        let tick_size =
            parse_market_info_price_tick_size(&value, &self.cfg.market).ok_or_else(|| {
                LiveGatewayError::fatal(format!(
                    "extended market info missing minPriceChange for {}",
                    self.cfg.market
                ))
            })?;
        let mut guard = self.price_tick_size.lock().await;
        Ok(*guard.get_or_insert(tick_size))
    }

    async fn run_bridge_command(&self, op: &str, payload: Value) -> LiveResult<String> {
        let trader_cmd = self
            .cfg
            .trader_cmd
            .clone()
            .ok_or_else(|| LiveGatewayError::fatal("extended trader cmd missing"))?;
        let payload_raw = serde_json::to_string(&payload).map_err(|err| {
            LiveGatewayError::fatal(format!("extended bridge payload encode error: {err}"))
        })?;
        let op = op.to_string();
        let op_for_child = op.clone();
        let output = tokio::task::spawn_blocking(move || {
            StdCommand::new("bash")
                .arg("-lc")
                .arg(trader_cmd)
                .env("PARAPHINA_EXTENDED_BRIDGE_OP", op_for_child)
                .env("PARAPHINA_EXTENDED_BRIDGE_PAYLOAD", payload_raw)
                .output()
        })
        .await
        .map_err(|err| LiveGatewayError::retryable(format!("extended bridge join error: {err}")))?
        .map_err(|err| {
            LiveGatewayError::retryable(format!("extended bridge spawn error: {err}"))
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !output.status.success() {
            let message = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("extended bridge failed op={} status={}", op, output.status)
            };
            return Err(map_rest_error(400, &message));
        }
        if stdout.is_empty() {
            return Err(LiveGatewayError::fatal(format!(
                "extended bridge returned empty output for op={op}"
            )));
        }
        Ok(stdout)
    }

    async fn run_bridge_json<T: DeserializeOwned>(
        &self,
        op: &str,
        payload: Value,
    ) -> LiveResult<T> {
        let raw = self.run_bridge_command(op, payload).await?;
        serde_json::from_str::<T>(&raw).map_err(|err| {
            LiveGatewayError::fatal(format!(
                "extended bridge {} parse error: {} body={}",
                op, err, raw
            ))
        })
    }

    pub async fn fetch_account_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<AccountSnapshot> {
        let snapshot: ExtendedBridgeSnapshot = self
            .run_bridge_json(
                "snapshot",
                json!({
                    "market": self.cfg.market,
                }),
            )
            .await?;
        Ok(snapshot.into_account_snapshot(venue_id, venue_index))
    }

    async fn fetch_open_order_snapshot(
        &self,
        venue_id: &str,
        venue_index: usize,
    ) -> LiveResult<OrderSnapshot> {
        let open_orders: Vec<ExtendedBridgeOpenOrder> = self
            .run_bridge_json(
                "open_orders",
                json!({
                    "market": self.cfg.market,
                }),
            )
            .await?;
        Ok(OrderSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: self.poll_seq.fetch_add(1, Ordering::Relaxed),
            timestamp_ms: now_ms(),
            open_orders: open_orders
                .into_iter()
                .filter(|order| order.size > 0.0)
                .map(|order| OpenOrderSnapshot {
                    order_id: order.order_id,
                    client_order_id: order.client_order_id,
                    exchange_order_id: None,
                    side: order.side,
                    price: order.price,
                    size: order.size,
                    purpose: None,
                })
                .collect(),
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
        loop {
            interval.tick().await;
            match self.fetch_account_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    let _ = account_tx.send(AccountEvent::Snapshot(snapshot)).await;
                }
                Err(err) => {
                    eprintln!("Extended account snapshot error: {}", err.message);
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
        loop {
            interval.tick().await;
            match self.fetch_open_order_snapshot(&venue_id, venue_index).await {
                Ok(snapshot) => {
                    let _ = exec_tx.send(ExecutionEvent::OrderSnapshot(snapshot)).await;
                }
                Err(err) => {
                    eprintln!("Extended open order snapshot error: {}", err.message);
                }
            }
        }
    }

    pub async fn run_private_ws(
        self: Arc<Self>,
        account_tx: mpsc::Sender<AccountEvent>,
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
                .private_ws_once(account_tx.clone(), exec_tx.clone(), &venue_id, venue_index)
                .await
            {
                eprintln!("Extended private WS error: {err}");
            }

            if session_start.elapsed() >= healthy_threshold {
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    async fn private_ws_once(
        &self,
        account_tx: mpsc::Sender<AccountEvent>,
        exec_tx: mpsc::Sender<ExecutionEvent>,
        venue_id: &str,
        venue_index: usize,
    ) -> anyhow::Result<()> {
        let mut account_state = ExtendedPrivateAccountState::new(venue_id, venue_index);
        if self.has_execution_auth() {
            match self.fetch_account_snapshot(venue_id, venue_index).await {
                Ok(snapshot) => {
                    account_state = ExtendedPrivateAccountState::from_snapshot(snapshot.clone());
                    account_tx
                        .send(AccountEvent::Snapshot(snapshot))
                        .await
                        .map_err(|_| {
                            anyhow::anyhow!("extended account_tx closed during bootstrap")
                        })?;
                }
                Err(err) => {
                    eprintln!(
                        "WARN: Extended bootstrap account snapshot skipped: {}",
                        err.message
                    );
                }
            }
        }

        let mut order_state =
            ExtendedPrivateOrderState::new(&self.cfg.market, venue_id, venue_index);
        if self.has_execution_auth() {
            match self.fetch_open_order_snapshot(venue_id, venue_index).await {
                Ok(snapshot) => {
                    order_state = ExtendedPrivateOrderState::from_snapshot(
                        &self.cfg.market,
                        snapshot.clone(),
                    );
                    exec_tx
                        .send(ExecutionEvent::OrderSnapshot(snapshot))
                        .await
                        .map_err(|_| anyhow::anyhow!("extended exec_tx closed during bootstrap"))?;
                }
                Err(err) => {
                    eprintln!(
                        "WARN: Extended bootstrap order snapshot skipped: {}",
                        err.message
                    );
                }
            }
        }

        let ws_url = self.cfg.private_ws_url.clone();
        let read_timeout = extended_private_ws_read_timeout();
        eprintln!("INFO: Extended private WS connecting url={}", ws_url);
        let mut request = ws_url.as_str().into_client_request()?;
        request
            .headers_mut()
            .insert(USER_AGENT, HeaderValue::from_static("paraphina"));
        let api_key = self
            .cfg
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("extended api key missing for private ws"))?;
        request.headers_mut().insert(
            "X-Api-Key",
            HeaderValue::from_str(api_key)
                .map_err(|err| anyhow::anyhow!("extended api key header invalid: {err}"))?,
        );
        let (ws_stream, _) = tokio::time::timeout(Duration::from_secs(15), connect_async(request))
            .await
            .map_err(|_| anyhow::anyhow!("Extended private WS connect timeout (15s)"))?
            .map_err(|err| anyhow::anyhow!("Extended private WS connect error: {err}"))?;
        eprintln!("INFO: Extended private WS connected url={}", ws_url);

        let (mut write, mut read) = ws_stream.split();
        let mut seq_state = ExtendedPrivateSeqState::default();

        loop {
            let maybe = tokio::time::timeout(read_timeout, read.next())
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Extended private WS read timeout after {}ms",
                        read_timeout.as_millis()
                    )
                })?;
            let Some(message) = maybe else {
                break;
            };
            match message? {
                Message::Text(text) => {
                    self.handle_private_ws_message(
                        text.as_ref(),
                        &account_tx,
                        &exec_tx,
                        &mut account_state,
                        &mut order_state,
                        &mut seq_state,
                    )
                    .await?;
                }
                Message::Binary(bytes) => {
                    let text = String::from_utf8(bytes.to_vec()).map_err(|_| {
                        anyhow::anyhow!("Extended private WS non-utf8 binary frame")
                    })?;
                    self.handle_private_ws_message(
                        &text,
                        &account_tx,
                        &exec_tx,
                        &mut account_state,
                        &mut order_state,
                        &mut seq_state,
                    )
                    .await?;
                }
                Message::Ping(payload) => {
                    write.send(Message::Pong(payload)).await?;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        Ok(())
    }

    async fn handle_private_ws_message(
        &self,
        text: &str,
        account_tx: &mpsc::Sender<AccountEvent>,
        exec_tx: &mpsc::Sender<ExecutionEvent>,
        account_state: &mut ExtendedPrivateAccountState,
        order_state: &mut ExtendedPrivateOrderState,
        seq_state: &mut ExtendedPrivateSeqState,
    ) -> anyhow::Result<()> {
        let Some(cleaned) = clean_ws_payload(text) else {
            return Ok(());
        };
        let value: Value = serde_json::from_str(cleaned)
            .map_err(|err| anyhow::anyhow!("Extended private WS parse error: {err}"))?;
        let message_type = value.get("type").and_then(|raw| raw.as_str()).unwrap_or("");
        if message_type.is_empty() {
            return Ok(());
        }
        let seq = value
            .get("seq")
            .and_then(parse_i64_value)
            .and_then(|raw| (raw >= 0).then_some(raw as u64))
            .ok_or_else(|| anyhow::anyhow!("Extended private WS missing seq for {message_type}"))?;
        seq_state.observe(seq)?;
        let timestamp_ms = value
            .get("ts")
            .and_then(parse_i64_value)
            .unwrap_or_else(now_ms);

        match message_type {
            "BALANCE" => {
                if let Some(snapshot) =
                    account_state.apply_balance_message(&value, seq, timestamp_ms)
                {
                    account_tx
                        .send(AccountEvent::Snapshot(snapshot))
                        .await
                        .map_err(|_| anyhow::anyhow!("extended account_tx closed"))?;
                }
            }
            "POSITION" => {
                if let Some(snapshot) =
                    account_state.apply_position_message(&value, seq, timestamp_ms)
                {
                    account_tx
                        .send(AccountEvent::Snapshot(snapshot))
                        .await
                        .map_err(|_| anyhow::anyhow!("extended account_tx closed"))?;
                }
            }
            "ORDER" => {
                if let Some(snapshot) = order_state.apply_order_message(&value, seq, timestamp_ms) {
                    exec_tx
                        .send(ExecutionEvent::OrderSnapshot(snapshot))
                        .await
                        .map_err(|_| anyhow::anyhow!("extended exec_tx closed"))?;
                }
            }
            "TRADE" => {
                for source_owner_fill in phase51_extended_source_owner_fills_from_trade_message(
                    &value,
                    order_state.snapshot.venue_index,
                    &order_state.snapshot.venue_id,
                    seq,
                    timestamp_ms,
                    &self.cfg.market,
                ) {
                    exec_tx
                        .send(ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(
                            source_owner_fill,
                        ))
                        .await
                        .map_err(|_| anyhow::anyhow!("extended exec_tx closed"))?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    // Note: funding polling lives on ExtendedConnector (market publisher).
}

impl LiveRestClient for ExtendedRestClient {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let price_tick_size = self.resolve_price_tick_size().await?;
            let price = snap_price_to_tick(req.price, price_tick_size, req.side, req.post_only);
            let response: ExtendedBridgePlaceResponse = self
                .run_bridge_json(
                    "place",
                    json!({
                        "market": self.cfg.market,
                        "side": map_side(req.side),
                        "price": format_f64(price),
                        "size": format_f64(req.size),
                        "post_only": req.post_only,
                        "reduce_only": req.reduce_only,
                        "time_in_force": map_time_in_force(req.time_in_force),
                        "client_order_id": req.client_order_id,
                    }),
                )
                .await?;
            let order_id = response
                .order_id
                .clone()
                .or_else(|| response.client_order_id.clone())
                .or_else(|| Some(req.client_order_id.clone()));
            Ok(LiveRestResponse {
                order_id,
                client_order_id: response.client_order_id.or(Some(req.client_order_id)),
            })
        })
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if !is_extended_external_replace_identity(&req.order_id) {
                return Err(LiveGatewayError::fatal(
                    "extended_native_replace_requires_external_id",
                ));
            }
            let price_tick_size = self.resolve_price_tick_size().await?;
            let price = snap_price_to_tick(req.price, price_tick_size, req.side, req.post_only);
            let response: ExtendedBridgePlaceResponse = self
                .run_bridge_json(
                    "replace",
                    json!({
                        "market": self.cfg.market,
                        "side": map_side(req.side),
                        "price": format_f64(price),
                        "size": format_f64(req.size),
                        "post_only": req.post_only,
                        "reduce_only": req.reduce_only,
                        "time_in_force": map_time_in_force(req.time_in_force),
                        "order_id": req.order_id,
                        "client_order_id": req.client_order_id,
                    }),
                )
                .await?;
            let order_id = response
                .order_id
                .clone()
                .or_else(|| response.client_order_id.clone());
            Ok(LiveRestResponse {
                order_id,
                client_order_id: response.client_order_id,
            })
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let response: ExtendedBridgeCancelResponse = self
                .run_bridge_json(
                    "cancel",
                    json!({
                        "order_id": req.order_id,
                    }),
                )
                .await?;
            Ok(LiveRestResponse {
                order_id: response.order_id,
                client_order_id: None,
            })
        })
    }

    fn cancel_all(
        &self,
        _req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let _: ExtendedBridgeCancelAllResponse = self
                .run_bridge_json(
                    "cancel_all",
                    json!({
                        "market": self.cfg.market,
                    }),
                )
                .await?;
            Ok(LiveRestResponse {
                order_id: None,
                client_order_id: None,
            })
        })
    }
}

fn map_side(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

fn map_time_in_force(time_in_force: TimeInForce) -> &'static str {
    match time_in_force {
        TimeInForce::Ioc => "IOC",
        TimeInForce::Gtc => "GTT",
    }
}

fn format_f64(value: f64) -> String {
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

fn snap_price_to_tick(price: f64, tick_size: f64, side: Side, post_only: bool) -> f64 {
    if !price.is_finite() || !tick_size.is_finite() || tick_size <= 0.0 {
        return price;
    }
    let ticks = price / tick_size;
    let epsilon = 1e-9;
    let snapped_ticks = if post_only {
        match side {
            Side::Buy => (ticks + epsilon).floor(),
            Side::Sell => (ticks - epsilon).ceil(),
        }
    } else {
        ticks.round()
    };
    snapped_ticks * tick_size
}

fn is_extended_external_replace_identity(order_id: &str) -> bool {
    let order_id = order_id.trim();
    !order_id.is_empty() && !order_id.bytes().all(|byte| byte.is_ascii_digit())
}

fn parse_market_info_price_tick_size(value: &Value, market: &str) -> Option<f64> {
    let markets = value
        .get("data")
        .and_then(|raw| raw.as_array())
        .or_else(|| value.as_array())?;
    markets
        .iter()
        .find(|entry| {
            entry
                .get("name")
                .and_then(|raw| raw.as_str())
                .map(|name| symbol_matches(name, market))
                .unwrap_or(false)
                || entry
                    .get("uiName")
                    .and_then(|raw| raw.as_str())
                    .map(|name| symbol_matches(name, market))
                    .unwrap_or(false)
        })
        .and_then(|entry| entry.get("tradingConfig"))
        .and_then(|config| config.get("minPriceChange"))
        .and_then(parse_f64)
        .filter(|tick_size| tick_size.is_finite() && *tick_size > 0.0)
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgePlaceResponse {
    order_id: Option<String>,
    client_order_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgeCancelResponse {
    order_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgeCancelAllResponse {
    #[serde(default)]
    count: usize,
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgeOpenOrder {
    order_id: String,
    #[serde(default)]
    client_order_id: Option<String>,
    #[serde(deserialize_with = "deserialize_extended_bridge_side")]
    side: Side,
    price: f64,
    size: f64,
}

fn deserialize_extended_bridge_side<'de, D>(deserializer: D) -> Result<Side, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    match raw.trim() {
        side if side.eq_ignore_ascii_case("BUY") => Ok(Side::Buy),
        side if side.eq_ignore_ascii_case("SELL") => Ok(Side::Sell),
        other => Err(serde::de::Error::custom(format!(
            "unknown side '{other}', expected BUY or SELL"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgePosition {
    market: String,
    size: f64,
    entry_price: f64,
    liquidation_price: Option<f64>,
    updated_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ExtendedBridgeSnapshot {
    timestamp_ms: Option<i64>,
    collateral_asset: Option<String>,
    balance_usd: f64,
    used_usd: f64,
    available_usd: f64,
    #[serde(default)]
    positions: Vec<ExtendedBridgePosition>,
}

impl ExtendedBridgeSnapshot {
    fn into_account_snapshot(self, venue_id: &str, venue_index: usize) -> AccountSnapshot {
        let observed_ms = now_ms();
        let position_ts = self
            .positions
            .iter()
            .filter_map(|position| position.updated_at)
            .max();
        let exchange_update_ms = self.timestamp_ms.or(position_ts);
        // The bridge call itself is direct account truth. Extended position
        // updated_at can remain old while an unchanged residual is still live, so
        // cache freshness must track observation time rather than last position
        // mutation time.
        let timestamp_ms = exchange_update_ms
            .map(|ts| ts.max(observed_ms))
            .unwrap_or(observed_ms);
        let asset = self.collateral_asset.unwrap_or_else(|| "USD".to_string());
        let liquidation = LiquidationSnapshot {
            price_liq: self
                .positions
                .iter()
                .find_map(|position| position.liquidation_price),
            dist_liq_sigma: None,
        };
        let positions = self
            .positions
            .into_iter()
            .map(|position| PositionSnapshot {
                symbol: position.market,
                size: position.size,
                entry_price: position.entry_price,
            })
            .collect::<Vec<_>>();
        let balances = vec![BalanceSnapshot {
            asset,
            total: self.balance_usd,
            available: self.available_usd,
        }];
        AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: timestamp_ms.max(0) as u64,
            timestamp_ms,
            positions,
            balances,
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd: self.balance_usd,
                used_usd: self.used_usd,
                available_usd: self.available_usd,
            },
            liquidation,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ExtendedPrivateSeqState {
    last_seq: Option<u64>,
}

impl ExtendedPrivateSeqState {
    fn observe(&mut self, seq: u64) -> anyhow::Result<()> {
        if let Some(last_seq) = self.last_seq {
            if seq <= last_seq {
                anyhow::bail!(
                    "extended private seq regression last_seq={} next_seq={}",
                    last_seq,
                    seq
                );
            }
            if seq > last_seq + 1 {
                anyhow::bail!(
                    "extended private seq gap last_seq={} next_seq={}",
                    last_seq,
                    seq
                );
            }
        }
        self.last_seq = Some(seq);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ExtendedPrivateAccountState {
    snapshot: AccountSnapshot,
    positions: BTreeMap<String, PositionSnapshot>,
    balances: BTreeMap<String, BalanceSnapshot>,
    liquidation_price: Option<f64>,
}

impl ExtendedPrivateAccountState {
    fn new(venue_id: &str, venue_index: usize) -> Self {
        Self {
            snapshot: AccountSnapshot {
                venue_index,
                venue_id: venue_id.to_string(),
                seq: 0,
                timestamp_ms: now_ms(),
                positions: Vec::new(),
                balances: Vec::new(),
                funding_8h: None,
                margin: MarginSnapshot {
                    balance_usd: 0.0,
                    used_usd: 0.0,
                    available_usd: 0.0,
                },
                liquidation: LiquidationSnapshot {
                    price_liq: None,
                    dist_liq_sigma: None,
                },
            },
            positions: BTreeMap::new(),
            balances: BTreeMap::new(),
            liquidation_price: None,
        }
    }

    fn from_snapshot(snapshot: AccountSnapshot) -> Self {
        let positions = snapshot
            .positions
            .iter()
            .cloned()
            .map(|position| (position.symbol.clone(), position))
            .collect();
        let balances = snapshot
            .balances
            .iter()
            .cloned()
            .map(|balance| (balance.asset.clone(), balance))
            .collect();
        Self {
            liquidation_price: snapshot.liquidation.price_liq,
            snapshot,
            positions,
            balances,
        }
    }

    fn apply_balance_message(
        &mut self,
        value: &Value,
        seq: u64,
        timestamp_ms: TimestampMs,
    ) -> Option<AccountSnapshot> {
        let data = value.get("data").unwrap_or(value);
        let balance = data.get("balance").or_else(|| data.get("balances"))?;
        for update in iter_value_items(balance) {
            let asset = update
                .get("collateralName")
                .or_else(|| update.get("asset"))
                .and_then(|raw| raw.as_str())
                .unwrap_or("USD");
            let total = update
                .get("balance")
                .or_else(|| update.get("equity"))
                .and_then(parse_f64)
                .unwrap_or_else(|| {
                    self.balances
                        .get(asset)
                        .map(|entry| entry.total)
                        .unwrap_or(0.0)
                });
            let available = update
                .get("availableForTrade")
                .or_else(|| update.get("availableForWithdrawal"))
                .and_then(parse_f64)
                .unwrap_or_else(|| {
                    self.balances
                        .get(asset)
                        .map(|entry| entry.available)
                        .unwrap_or(total)
                });
            self.balances.insert(
                asset.to_string(),
                BalanceSnapshot {
                    asset: asset.to_string(),
                    total,
                    available,
                },
            );
            self.snapshot.margin.balance_usd = update
                .get("equity")
                .or_else(|| update.get("balance"))
                .and_then(parse_f64)
                .unwrap_or(total);
            self.snapshot.margin.used_usd = update
                .get("initialMargin")
                .or_else(|| update.get("usedMargin"))
                .and_then(parse_f64)
                .unwrap_or_else(|| {
                    (self.snapshot.margin.balance_usd - available.max(0.0)).max(0.0)
                });
            self.snapshot.margin.available_usd = available;
        }
        Some(self.snapshot_with_state(seq, timestamp_ms))
    }

    fn apply_position_message(
        &mut self,
        value: &Value,
        seq: u64,
        timestamp_ms: TimestampMs,
    ) -> Option<AccountSnapshot> {
        let data = value.get("data").unwrap_or(value);
        let positions = data.get("positions").or_else(|| data.get("position"))?;
        let mut liquidation_price = self.liquidation_price;
        for update in iter_value_items(positions) {
            let symbol = update
                .get("market")
                .or_else(|| update.get("symbol"))
                .and_then(|raw| raw.as_str())?;
            let size = update.get("size").and_then(parse_f64).unwrap_or(0.0);
            let signed_size =
                apply_position_side(size, update.get("side").and_then(|raw| raw.as_str()));
            let entry_price = update
                .get("openPrice")
                .or_else(|| update.get("entryPrice"))
                .and_then(parse_f64)
                .unwrap_or_else(|| {
                    self.positions
                        .get(symbol)
                        .map(|existing| existing.entry_price)
                        .unwrap_or(0.0)
                });
            let liq = update
                .get("liquidationPrice")
                .and_then(parse_f64)
                .filter(|price| price.is_finite() && *price > 0.0);
            if liq.is_some() {
                liquidation_price = liq;
            }
            if signed_size.abs() < 1e-12 {
                self.positions.remove(symbol);
            } else {
                self.positions.insert(
                    symbol.to_string(),
                    PositionSnapshot {
                        symbol: symbol.to_string(),
                        size: signed_size,
                        entry_price,
                    },
                );
            }
        }
        self.liquidation_price = liquidation_price;
        Some(self.snapshot_with_state(seq, timestamp_ms))
    }

    fn snapshot_with_state(&mut self, seq: u64, timestamp_ms: TimestampMs) -> AccountSnapshot {
        self.snapshot.seq = seq;
        self.snapshot.timestamp_ms = timestamp_ms;
        self.snapshot.positions = self.positions.values().cloned().collect();
        self.snapshot.balances = self.balances.values().cloned().collect();
        if self.snapshot.margin.balance_usd <= 0.0 {
            self.snapshot.margin.balance_usd = self
                .snapshot
                .balances
                .iter()
                .map(|balance| balance.total)
                .sum::<f64>();
        }
        if self.snapshot.margin.available_usd <= 0.0 && !self.snapshot.balances.is_empty() {
            self.snapshot.margin.available_usd = self
                .snapshot
                .balances
                .iter()
                .map(|balance| balance.available)
                .sum::<f64>();
        }
        if self.snapshot.margin.used_usd <= 0.0 {
            self.snapshot.margin.used_usd =
                (self.snapshot.margin.balance_usd - self.snapshot.margin.available_usd).max(0.0);
        }
        self.snapshot.liquidation.price_liq = self.liquidation_price;
        self.snapshot.clone()
    }
}

#[derive(Debug, Clone)]
struct ExtendedPrivateOrderState {
    market: String,
    snapshot: OrderSnapshot,
    open_orders: BTreeMap<String, OpenOrderSnapshot>,
}

impl ExtendedPrivateOrderState {
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

    fn apply_order_message(
        &mut self,
        value: &Value,
        seq: u64,
        timestamp_ms: TimestampMs,
    ) -> Option<OrderSnapshot> {
        let data = value.get("data").unwrap_or(value);
        let orders = data.get("orders").or_else(|| data.get("order"))?;
        for order in iter_value_items(orders) {
            let symbol = order
                .get("market")
                .or_else(|| order.get("symbol"))
                .and_then(|raw| raw.as_str())
                .unwrap_or("");
            if !symbol_matches(symbol, &self.market) {
                continue;
            }
            let order_id = order
                .get("id")
                .or_else(|| order.get("orderId"))
                .and_then(|raw| {
                    raw.as_str()
                        .map(|id| id.to_string())
                        .or_else(|| raw.as_i64().map(|id| id.to_string()))
                })?;
            let status = order
                .get("status")
                .and_then(|raw| raw.as_str())
                .unwrap_or("");
            let side = match order.get("side").and_then(|raw| raw.as_str()) {
                Some(side) if side.eq_ignore_ascii_case("BUY") => Side::Buy,
                Some(side) if side.eq_ignore_ascii_case("SELL") => Side::Sell,
                _ => continue,
            };
            let price = order.get("price").and_then(parse_f64).unwrap_or_else(|| {
                self.open_orders
                    .get(&order_id)
                    .map(|existing| existing.price)
                    .unwrap_or(0.0)
            });
            let qty = order
                .get("qty")
                .or_else(|| order.get("size"))
                .and_then(parse_f64)
                .unwrap_or(0.0);
            let filled_qty = order
                .get("filledQty")
                .or_else(|| order.get("filled"))
                .and_then(parse_f64)
                .unwrap_or(0.0);
            let remaining_size = (qty - filled_qty).max(0.0);
            let client_order_id = order
                .get("externalId")
                .or_else(|| order.get("clientOrderId"))
                .and_then(|raw| raw.as_str())
                .map(|id| id.to_string());
            if extended_order_is_open(status) && remaining_size > 0.0 {
                self.open_orders.insert(
                    order_id.clone(),
                    OpenOrderSnapshot {
                        order_id: order_id.clone(),
                        client_order_id,
                        exchange_order_id: None,
                        side,
                        price,
                        size: remaining_size,
                        purpose: None,
                    },
                );
            } else {
                self.open_orders.remove(&order_id);
            }
        }
        self.snapshot.seq = seq;
        self.snapshot.timestamp_ms = timestamp_ms;
        self.snapshot.open_orders = self.open_orders.values().cloned().collect();
        Some(self.snapshot.clone())
    }
}

fn phase51_extended_source_owner_fills_from_trade_message(
    value: &Value,
    venue_index: usize,
    venue_id: &str,
    seq: u64,
    timestamp_ms: TimestampMs,
    market: &str,
) -> Vec<Phase51ForwardRefreshSourceOwnerFill> {
    if value.get("type").and_then(|raw| raw.as_str()) != Some("TRADE") {
        return Vec::new();
    }
    let data = value.get("data").unwrap_or(value);
    let trades = data
        .get("trades")
        .or_else(|| data.get("trade"))
        .unwrap_or(data);
    iter_value_items(trades)
        .into_iter()
        .enumerate()
        .filter_map(|(trade_index, trade)| {
            let symbol = trade
                .get("market")
                .or_else(|| trade.get("symbol"))
                .and_then(|raw| raw.as_str())
                .unwrap_or("");
            if !symbol_matches(symbol, market) {
                return None;
            }
            let is_taker = trade.get("isTaker")?.as_bool()?;
            let order_id =
                parse_extended_stringish(trade.get("orderId").or_else(|| trade.get("order_id")));
            let client_order_id = parse_extended_stringish(
                trade
                    .get("externalId")
                    .or_else(|| trade.get("externalOrderId"))
                    .or_else(|| trade.get("clientOrderId"))
                    .or_else(|| trade.get("client_order_id")),
            );
            if order_id.is_none() && client_order_id.is_none() {
                return None;
            }

            Some(Phase51ForwardRefreshSourceOwnerFill::new(
                venue_index,
                venue_id,
                phase51_extended_source_owner_seq(seq, trade_index),
                timestamp_ms,
                order_id,
                client_order_id,
                Some(Phase51ForwardRefreshNativeRole::Extended { is_taker }),
            ))
        })
        .collect()
}

fn phase51_extended_source_owner_seq(seq: u64, trade_index: usize) -> u64 {
    seq.saturating_mul(1_000_000)
        .saturating_add(trade_index as u64)
}

fn parse_extended_stringish(value: Option<&Value>) -> Option<String> {
    value
        .and_then(|raw| {
            raw.as_str()
                .map(|id| id.to_string())
                .or_else(|| raw.as_i64().map(|id| id.to_string()))
                .or_else(|| raw.as_u64().map(|id| id.to_string()))
        })
        .filter(|id| !id.trim().is_empty())
}

async fn fetch_public_funding(
    client: &Client,
    cfg: &ExtendedConfig,
) -> anyhow::Result<FundingUpdate> {
    // Extended uses /api/v1/info/markets/{market}/stats for funding data.
    // The market is part of the path (e.g., ETH-USD), not a query parameter.
    let path = std::env::var("EXTENDED_FUNDING_PATH")
        .unwrap_or_else(|_| format!("/api/v1/info/markets/{}/stats", cfg.market));
    let url = format!("{}{}", cfg.rest_url.trim_end_matches('/'), path);
    let resp = client.get(&url).send().await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        anyhow::bail!(
            "Extended funding fetch failed: HTTP {} url={} body={}",
            status,
            url,
            body.chars().take(160).collect::<String>()
        );
    }

    let value: Value = serde_json::from_str(&body).map_err(|e| {
        anyhow::anyhow!(
            "Extended funding JSON parse error: {} body={}",
            e,
            body.chars().take(160).collect::<String>()
        )
    })?;

    parse_public_funding(&value, cfg)
        .ok_or_else(|| anyhow::anyhow!("invalid public funding response"))
}

fn parse_public_funding(value: &Value, cfg: &ExtendedConfig) -> Option<FundingUpdate> {
    let data = value
        .get("data")
        .or_else(|| value.get("result"))
        .unwrap_or(value);

    let rate_native = data
        .get("fundingRate")
        .or_else(|| data.get("funding_rate"))
        .or_else(|| data.get("lastFundingRate"))
        .and_then(parse_f64);

    // Extended API doesn't explicitly provide interval, but fundingRate is hourly.
    // Default to 3600s (1 hour) when rate is present.
    let interval_sec = data
        .get("fundingIntervalSec")
        .or_else(|| data.get("funding_interval_sec"))
        .or_else(|| data.get("fundingInterval"))
        .and_then(parse_i64_value)
        .and_then(|v| if v > 0 { Some(v as u64) } else { None })
        .or_else(|| {
            // Extended fundingRate is hourly; assume 3600s if rate is present
            if rate_native.is_some() {
                Some(3600)
            } else {
                None
            }
        });

    // Extended API uses "nextFundingRate" but it's actually the next funding TIME in ms
    let next_funding_ms = data
        .get("nextFundingRate") // Extended's field name (actually a timestamp, not a rate)
        .or_else(|| data.get("nextFundingTime"))
        .or_else(|| data.get("next_funding_time"))
        .or_else(|| data.get("nextFundingTimestamp"))
        .and_then(parse_i64_value);

    let as_of_ms = data
        .get("time")
        .or_else(|| data.get("timestamp"))
        .or_else(|| data.get("ts"))
        .and_then(parse_i64_value)
        .unwrap_or_else(now_ms);

    // Convert hourly rate to 8h: rate_8h = rate_native * (8h / interval_sec)
    let rate_8h = match (rate_native, interval_sec) {
        (Some(rate), Some(sec)) if sec > 0 => Some(rate * (8.0 * 60.0 * 60.0 / sec as f64)),
        (Some(rate), None) => Some(rate), // Assume already 8h if no interval
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
        settlement_price_kind: Some(SettlementPriceKind::Mark), // Extended uses mark price
        source: FundingSource::MarketDataRest,
    })
}

fn map_rest_error(status: u16, body: &str) -> LiveGatewayError {
    let lower = body.to_lowercase();
    if status == 429 || status == 418 || lower.contains("rate") && lower.contains("limit") {
        return LiveGatewayError::rate_limited(body);
    }
    if lower.contains("post") && lower.contains("only") {
        return LiveGatewayError::post_only_reject(body);
    }
    if lower.contains("reduce") && lower.contains("only") {
        return LiveGatewayError::reduce_only_violation(body);
    }
    if status >= 500 || lower.contains("timeout") || lower.contains("tempor") {
        return LiveGatewayError::retryable(body);
    }
    LiveGatewayError {
        kind: LiveGatewayErrorKind::Fatal,
        message: body.to_string(),
    }
}

#[derive(Debug, Clone)]
struct ExtendedDepthSnapshot {
    last_update_id: Option<u64>,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

#[derive(Debug, Clone)]
struct ExtendedDepthUpdate {
    symbol: String,
    event_time: Option<TimestampMs>,
    seq: u64,
    bids: Vec<BookLevelDelta>,
    asks: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone, Copy)]
struct ExtendedSeqState {
    last_seq: Option<u64>,
    venue_index: usize,
}

impl ExtendedSeqState {
    fn new(last_seq: Option<u64>, venue_index: usize) -> Self {
        Self {
            last_seq,
            venue_index,
        }
    }

    fn observe_seq(&mut self, seq: u64) -> anyhow::Result<bool> {
        if let Some(last_seq) = self.last_seq {
            if seq <= last_seq {
                return Ok(false);
            }
            if seq > last_seq + 1 {
                anyhow::bail!("extended seq gap last={} next={}", last_seq, seq);
            }
        }
        self.last_seq = Some(seq);
        Ok(true)
    }

    fn apply_update(
        &mut self,
        update: &ExtendedDepthUpdate,
    ) -> anyhow::Result<Option<MarketDataEvent>> {
        if !self.observe_seq(update.seq)? {
            return Ok(None);
        }
        let mut changes = Vec::with_capacity(update.bids.len() + update.asks.len());
        changes.extend(update.bids.iter().cloned());
        changes.extend(update.asks.iter().cloned());
        let event = MarketDataEvent::L2Delta(super::super::types::L2Delta {
            venue_index: self.venue_index,
            venue_id: update.symbol.clone(),
            seq: update.seq,
            timestamp_ms: update.event_time.unwrap_or_else(now_ms),
            changes,
        });
        Ok(Some(event))
    }
}

#[derive(Debug)]
struct ExtendedRecorder {
    dir: PathBuf,
}

impl ExtendedRecorder {
    fn new(dir: &PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self { dir: dir.clone() })
    }

    fn record_snapshot(&mut self, raw: &str) -> std::io::Result<()> {
        let path = self.dir.join("rest_snapshot.json");
        std::fs::write(path, raw)
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

fn parse_depth_snapshot(value: &Value) -> Option<ExtendedDepthSnapshot> {
    if let Some(last_update_id) = value.get("lastUpdateId").and_then(|raw| raw.as_u64()) {
        let bids = parse_levels_from_value(value.get("bids")?)?;
        let asks = parse_levels_from_value(value.get("asks")?)?;
        return Some(ExtendedDepthSnapshot {
            last_update_id: Some(last_update_id),
            bids,
            asks,
        });
    }

    if value
        .get("status")
        .and_then(|raw| raw.as_str())
        .is_some_and(|status| !status.eq_ignore_ascii_case("ok"))
    {
        return None;
    }
    let payload = value.get("data").unwrap_or(value);
    let bids = parse_levels_from_value(payload.get("bid")?)?;
    let asks = parse_levels_from_value(payload.get("ask")?)?;
    Some(ExtendedDepthSnapshot {
        last_update_id: None,
        bids,
        asks,
    })
}

fn parse_depth_update(text: &str) -> Result<Option<ExtendedDepthUpdate>, serde_json::Error> {
    let value: Value = serde_json::from_str(text)?;
    let payload = value.get("data").unwrap_or(&value);
    let event = value
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("t").and_then(|v| v.as_str()))
        .unwrap_or("");
    if !event.eq_ignore_ascii_case("DELTA") {
        return Ok(None);
    }
    let symbol = payload
        .get("m")
        .or_else(|| payload.get("symbol"))
        .or_else(|| payload.get("s"))
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let seq = value
        .get("seq")
        .or_else(|| payload.get("seq"))
        .and_then(|v| v.as_u64());
    let (symbol, seq) = match (symbol, seq) {
        (Some(symbol), Some(seq)) => (symbol, seq),
        _ => return Ok(None),
    };
    let event_time = payload
        .get("ts")
        .or_else(|| value.get("ts"))
        .or_else(|| payload.get("E"))
        .and_then(|v| v.as_i64())
        .map(|v| v as TimestampMs);
    let bids = match payload
        .get("b")
        .and_then(|v| parse_deltas_from_value(v, BookSide::Bid))
    {
        Some(bids) => bids,
        None => return Ok(None),
    };
    let asks = match payload
        .get("a")
        .and_then(|v| parse_deltas_from_value(v, BookSide::Ask))
    {
        Some(asks) => asks,
        None => return Ok(None),
    };
    Ok(Some(ExtendedDepthUpdate {
        symbol,
        event_time,
        seq,
        bids,
        asks,
    }))
}

fn clean_ws_payload(text: &str) -> Option<&str> {
    let cleaned = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn decode_top_from_update(
    update: &ExtendedDepthUpdate,
    timestamp_ms: Option<TimestampMs>,
) -> Option<TopOfBook> {
    let bid = update
        .bids
        .iter()
        .filter(|lvl| lvl.size > 0.0)
        .max_by(|a, b| {
            a.price
                .partial_cmp(&b.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let ask = update
        .asks
        .iter()
        .filter(|lvl| lvl.size > 0.0)
        .min_by(|a, b| {
            a.price
                .partial_cmp(&b.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    TopOfBook::from_levels(
        &[BookLevel {
            price: bid.price,
            size: bid.size,
        }],
        &[BookLevel {
            price: ask.price,
            size: ask.size,
        }],
        timestamp_ms,
    )
}

fn decode_top_from_value(value: &Value) -> Option<TopOfBook> {
    let payload = value
        .get("data")
        .or_else(|| value.get("order_book"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let bids = payload.get("b").or_else(|| payload.get("bids"))?;
    let asks = payload.get("a").or_else(|| payload.get("asks"))?;
    let bid = best_level_from_value(bids, true)?;
    let ask = best_level_from_value(asks, false)?;
    let timestamp_ms = payload
        .get("E")
        .or_else(|| payload.get("ts"))
        .and_then(|v| v.as_i64());
    TopOfBook::from_levels(
        &[BookLevel {
            price: bid.price,
            size: bid.size,
        }],
        &[BookLevel {
            price: ask.price,
            size: ask.size,
        }],
        timestamp_ms,
    )
}

fn parse_depth_snapshot_from_ws(
    value: &Value,
    market: &str,
    venue_index: usize,
    fallback_seq: &mut u64,
) -> Option<MarketDataEvent> {
    let payload = value
        .get("data")
        .or_else(|| value.get("order_book"))
        .or_else(|| value.get("result"))
        .unwrap_or(value);
    let event = value
        .get("type")
        .and_then(|raw| raw.as_str())
        .or_else(|| payload.get("t").and_then(|raw| raw.as_str()))
        .unwrap_or("");
    if !event.is_empty() && !event.eq_ignore_ascii_case("SNAPSHOT") {
        return None;
    }
    let bids_value = payload.get("bids").or_else(|| payload.get("b"))?;
    let asks_value = payload.get("asks").or_else(|| payload.get("a"))?;
    let bids = parse_levels_from_value(bids_value)?;
    let asks = parse_levels_from_value(asks_value)?;
    let seq = payload
        .get("lastUpdateId")
        .or_else(|| payload.get("u"))
        .or_else(|| value.get("seq"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| {
            *fallback_seq = fallback_seq.wrapping_add(1);
            *fallback_seq
        });
    let timestamp_ms = payload
        .get("E")
        .or_else(|| payload.get("ts"))
        .or_else(|| value.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(now_ms);
    let venue_id = payload
        .get("m")
        .or_else(|| payload.get("symbol"))
        .and_then(|v| v.as_str())
        .unwrap_or(market)
        .to_string();
    let has_bids = has_effective_levels(&bids);
    let has_asks = has_effective_levels(&asks);

    if has_bids && has_asks {
        return Some(MarketDataEvent::L2Snapshot(
            super::super::types::L2Snapshot {
                venue_index,
                venue_id,
                seq,
                timestamp_ms,
                bids,
                asks,
            },
        ));
    }

    // Guard against one-sided/empty WS "snapshot" frames: applying them as a
    // full snapshot would clear the opposite side and collapse depth to zero.
    let mut changes = Vec::new();
    if has_bids {
        changes.extend(levels_to_positive_deltas(&bids, BookSide::Bid));
    }
    if has_asks {
        changes.extend(levels_to_positive_deltas(&asks, BookSide::Ask));
    }
    if changes.is_empty() {
        return None;
    }
    Some(MarketDataEvent::L2Delta(super::super::types::L2Delta {
        venue_index,
        venue_id,
        seq,
        timestamp_ms,
        changes,
    }))
}

fn has_effective_levels(levels: &[BookLevel]) -> bool {
    levels
        .iter()
        .any(|lvl| lvl.price.is_finite() && lvl.size.is_finite() && lvl.size > 0.0)
}

fn levels_to_positive_deltas(levels: &[BookLevel], side: BookSide) -> Vec<BookLevelDelta> {
    levels
        .iter()
        .filter(|lvl| lvl.price.is_finite() && lvl.size.is_finite() && lvl.size > 0.0)
        .map(|lvl| BookLevelDelta {
            side,
            price: lvl.price,
            size: lvl.size,
        })
        .collect()
}

fn best_level_from_value(value: &Value, is_bid: bool) -> Option<BookLevel> {
    let entries = value.as_array()?;
    let mut best: Option<BookLevel> = None;
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        if size <= 0.0 {
            continue;
        }
        let replace = match best {
            None => true,
            Some(prev) => {
                if is_bid {
                    price > prev.price
                } else {
                    price < prev.price
                }
            }
        };
        if replace {
            best = Some(BookLevel { price, size });
        }
    }
    best
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

fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;
        out.push(BookLevel { price, size });
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
    if let Some(items) = value.as_array() {
        if items.len() < 2 {
            return None;
        }
        let price = parse_f64(&items[0])?;
        let size = parse_f64(&items[1])?;
        return Some((price, size));
    }
    if let Some(obj) = value.as_object() {
        let price = obj
            .get("price")
            .or_else(|| obj.get("px"))
            .or_else(|| obj.get("p"))
            .and_then(parse_f64)?;
        let size = obj
            .get("size")
            .or_else(|| obj.get("sz"))
            .or_else(|| obj.get("qty"))
            .or_else(|| obj.get("c"))
            .or_else(|| obj.get("q"))
            .and_then(parse_f64)?;
        return Some((price, size));
    }
    None
}

fn normalize_extended_market(raw: &str) -> String {
    let mut upper = raw.trim().to_uppercase();
    if let Some(stripped) = upper.strip_suffix("-USD-PERP") {
        upper = stripped.to_string();
    } else if let Some(stripped) = upper.strip_suffix("-PERP") {
        upper = stripped.to_string();
    }
    if upper.contains("-USD") {
        let base = upper.split("-USD").next().unwrap_or(&upper);
        return format!("{base}-USD");
    }
    if let Some(stripped) = upper.strip_suffix("USDT") {
        return format!("{stripped}-USD");
    }
    if let Some(stripped) = upper.strip_suffix("USD") {
        return format!("{stripped}-USD");
    }
    format!("{upper}-USD")
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

fn iter_value_items(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(items) => items.iter().collect(),
        Value::Null => Vec::new(),
        other => vec![other],
    }
}

fn apply_position_side(size: f64, side: Option<&str>) -> f64 {
    let Some(side) = side else {
        return size;
    };
    if side.eq_ignore_ascii_case("SELL") || side.eq_ignore_ascii_case("SHORT") {
        -size.abs()
    } else if side.eq_ignore_ascii_case("BUY") || side.eq_ignore_ascii_case("LONG") {
        size.abs()
    } else {
        size
    }
}

fn extended_order_is_open(status: &str) -> bool {
    status.eq_ignore_ascii_case("OPEN")
        || status.eq_ignore_ascii_case("NEW")
        || status.eq_ignore_ascii_case("PARTIALLY_FILLED")
        || status.eq_ignore_ascii_case("PARTIALLYFILLED")
        || status.eq_ignore_ascii_case("PLACED")
        || status.eq_ignore_ascii_case("ACCEPTED")
}

fn symbol_matches(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
pub struct ExtendedFixtureFeed {
    snapshot: FixtureSnapshot,
    deltas: Vec<FixtureDelta>,
    account: FixtureAccountSnapshot,
}

impl ExtendedFixtureFeed {
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
    use crate::config::Config;
    use crate::sim_eval::env_override::with_env_overrides;
    use crate::state::GlobalState;
    use crate::toxicity::update_toxicity_and_health;
    use crate::types::VenueStatus;
    use httpmock::Method::GET;
    use httpmock::MockServer;
    use std::collections::BTreeMap;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn ms_to_ns(ms: u64) -> u64 {
        ms * 1_000_000
    }

    struct EnvVarGuard {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.value.as_deref() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    fn test_ws_cfg() -> ExtendedConfig {
        ExtendedConfig {
            ws_url: "wss://api.extended.test/stream/".to_string(),
            private_ws_url: "wss://api.extended.test/stream/account".to_string(),
            rest_url: "https://api.extended.test".to_string(),
            market: "BTC-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: None,
            trader_cmd: None,
            record_dir: None,
        }
    }

    fn write_bridge_script(tmp: &TempDir, body: &str) -> String {
        let path = tmp.path().join("extended_bridge.sh");
        fs::write(&path, body).expect("write bridge script");
        let mut perms = fs::metadata(&path).expect("bridge metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("chmod bridge script");
        path.display().to_string()
    }

    #[test]
    fn orderbook_ws_url_defaults_to_depth_two_when_env_unset() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard = EnvVarGuard::new("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS");
        std::env::remove_var("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS");

        let cfg = test_ws_cfg();
        assert_eq!(
            cfg.orderbook_ws_url(),
            "wss://api.extended.test/stream/orderbooks/BTC-USD"
        );
    }

    #[test]
    fn orderbook_ws_url_uses_depth_one_endpoint_when_env_is_one() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard = EnvVarGuard::new("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS");
        std::env::set_var("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS", "1");

        let cfg = test_ws_cfg();
        assert_eq!(
            cfg.orderbook_ws_url(),
            "wss://api.extended.test/stream/orderbooks/BTC-USD?depth=1"
        );
    }

    #[test]
    fn orderbook_ws_url_uses_full_book_endpoint_when_env_is_two_or_higher() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard = EnvVarGuard::new("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS");

        let cfg = test_ws_cfg();
        std::env::set_var("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS", "2");
        assert_eq!(
            cfg.orderbook_ws_url(),
            "wss://api.extended.test/stream/orderbooks/BTC-USD"
        );

        std::env::set_var("PARAPHINA_EXTENDED_WS_DEPTH_LEVELS", "7");
        assert_eq!(
            cfg.orderbook_ws_url(),
            "wss://api.extended.test/stream/orderbooks/BTC-USD"
        );
    }

    #[test]
    fn private_ws_url_defaults_to_account_endpoint_and_read_auth_is_api_key_only() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let ws_guard = EnvVarGuard::new("EXTENDED_WS_URL");
        let private_ws_guard = EnvVarGuard::new("EXTENDED_PRIVATE_WS_URL");
        let api_key_guard = EnvVarGuard::new("EXTENDED_API_KEY");
        let trader_cmd_guard = EnvVarGuard::new("EXTENDED_TRADER_CMD");

        std::env::set_var("EXTENDED_WS_URL", "wss://api.extended.test/stream");
        std::env::remove_var("EXTENDED_PRIVATE_WS_URL");
        std::env::set_var("EXTENDED_API_KEY", "test-key");
        std::env::remove_var("EXTENDED_TRADER_CMD");

        let cfg = ExtendedConfig::from_env();
        assert_eq!(cfg.private_ws_url, "wss://api.extended.test/stream/account");
        assert!(cfg.has_private_read_auth());
        assert!(!cfg.has_execution_auth());

        drop(ws_guard);
        drop(private_ws_guard);
        drop(api_key_guard);
        drop(trader_cmd_guard);
    }

    #[test]
    fn private_seq_state_rejects_regressions_and_gaps() {
        let mut seq = ExtendedPrivateSeqState::default();
        seq.observe(7).expect("first seq");
        seq.observe(8).expect("contiguous seq");
        assert!(seq.observe(8).is_err(), "duplicate seq must fail");
        assert!(seq.observe(10).is_err(), "gap seq must fail");
    }

    #[test]
    fn private_balance_and_position_messages_emit_full_account_snapshot() {
        let mut state = ExtendedPrivateAccountState::new("extended", 4);

        let balance_msg = serde_json::json!({
            "type": "BALANCE",
            "seq": 11_u64,
            "ts": 1_700_000_000_111_i64,
            "data": {
                "balance": {
                    "collateralName": "USDC",
                    "balance": "80.25",
                    "equity": "81.5",
                    "availableForTrade": "77.0",
                    "initialMargin": "4.5"
                }
            }
        });
        let balance_snapshot = state
            .apply_balance_message(&balance_msg, 11, 1_700_000_000_111_i64)
            .expect("balance snapshot");
        assert_eq!(balance_snapshot.balances.len(), 1);
        assert_eq!(balance_snapshot.balances[0].asset, "USDC");
        assert_eq!(balance_snapshot.balances[0].total, 80.25);
        assert_eq!(balance_snapshot.margin.balance_usd, 81.5);
        assert_eq!(balance_snapshot.margin.available_usd, 77.0);
        assert_eq!(balance_snapshot.margin.used_usd, 4.5);

        let position_msg = serde_json::json!({
            "type": "POSITION",
            "seq": 12_u64,
            "ts": 1_700_000_000_222_i64,
            "data": {
                "positions": [{
                    "market": "ETH-USD",
                    "side": "SELL",
                    "size": "0.03",
                    "openPrice": "3450.5",
                    "liquidationPrice": "4100.0"
                }]
            }
        });
        let position_snapshot = state
            .apply_position_message(&position_msg, 12, 1_700_000_000_222_i64)
            .expect("position snapshot");
        assert_eq!(position_snapshot.positions.len(), 1);
        assert_eq!(position_snapshot.positions[0].symbol, "ETH-USD");
        assert_eq!(position_snapshot.positions[0].size, -0.03);
        assert_eq!(position_snapshot.positions[0].entry_price, 3450.5);
        assert_eq!(position_snapshot.liquidation.price_liq, Some(4100.0));
    }

    #[test]
    fn private_order_messages_emit_full_order_snapshot() {
        let mut state = ExtendedPrivateOrderState::new("ETH-USD", "extended", 2);
        let open_msg = serde_json::json!({
            "type": "ORDER",
            "seq": 21_u64,
            "ts": 1_700_000_000_333_i64,
            "data": {
                "orders": [{
                    "id": "oid_1",
                    "externalId": "co_1",
                    "market": "ETH-USD",
                    "side": "BUY",
                    "status": "OPEN",
                    "price": "3500.5",
                    "qty": "0.04",
                    "filledQty": "0.01"
                }]
            }
        });
        let snapshot = state
            .apply_order_message(&open_msg, 21, 1_700_000_000_333_i64)
            .expect("order snapshot");
        assert_eq!(snapshot.open_orders.len(), 1);
        assert_eq!(snapshot.open_orders[0].order_id, "oid_1");
        assert_eq!(
            snapshot.open_orders[0].client_order_id.as_deref(),
            Some("co_1")
        );
        assert_eq!(snapshot.open_orders[0].side, Side::Buy);
        assert_eq!(snapshot.open_orders[0].size, 0.03);

        let closed_msg = serde_json::json!({
            "type": "ORDER",
            "seq": 22_u64,
            "ts": 1_700_000_000_444_i64,
            "data": {
                "orders": [{
                    "id": "oid_1",
                    "market": "ETH-USD",
                    "side": "BUY",
                    "status": "FILLED",
                    "price": "3500.5",
                    "qty": "0.04",
                    "filledQty": "0.04"
                }]
            }
        });
        let snapshot = state
            .apply_order_message(&closed_msg, 22, 1_700_000_000_444_i64)
            .expect("closed order snapshot");
        assert!(snapshot.open_orders.is_empty());
    }

    #[test]
    fn trade_message_exact_is_taker_fields_create_source_owner_fills() {
        let trade_msg = serde_json::json!({
            "type": "TRADE",
            "seq": 31_u64,
            "ts": 1_700_000_000_555_i64,
            "data": {
                "trades": [
                    {
                        "market": "ETH-USD",
                        "orderId": "oid_ext_1",
                        "clientOrderId": "co_ext_1",
                        "isTaker": true
                    },
                    {
                        "market": "ETH-USD",
                        "orderId": "oid_ext_2",
                        "externalOrderId": "co_ext_2",
                        "isTaker": false
                    }
                ]
            }
        });

        let fills = phase51_extended_source_owner_fills_from_trade_message(
            &trade_msg,
            2,
            "extended",
            31,
            1_700_000_000_555_i64,
            "ETH-USD",
        );

        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].venue_id, "extended");
        assert_eq!(fills[0].order_id(), Some("oid_ext_1"));
        assert_eq!(fills[0].client_order_id(), Some("co_ext_1"));
        assert_eq!(fills[0].seq, 31_000_000);
        assert_eq!(fills[0].phase51_lighter_native_limit, None);
        assert_eq!(
            fills[0].phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: true })
        );
        assert_eq!(fills[1].order_id(), Some("oid_ext_2"));
        assert_eq!(fills[1].client_order_id(), Some("co_ext_2"));
        assert_eq!(fills[1].seq, 31_000_001);
        assert_eq!(
            fills[1].phase51_native_role,
            Some(Phase51ForwardRefreshNativeRole::Extended { is_taker: false })
        );
    }

    #[test]
    fn trade_message_missing_or_non_bool_is_taker_creates_no_source_owner_fill() {
        for trade in [
            serde_json::json!({"market": "ETH-USD", "orderId": "oid_ext_1"}),
            serde_json::json!({"market": "ETH-USD", "orderId": "oid_ext_1", "isTaker": "true"}),
            serde_json::json!({"market": "ETH-USD", "orderId": "oid_ext_1", "isTaker": 1}),
            serde_json::json!({"market": "ETH-USD", "orderId": "oid_ext_1", "isTaker": null}),
        ] {
            let trade_msg = serde_json::json!({
                "type": "TRADE",
                "seq": 32_u64,
                "data": { "trades": [trade] }
            });

            assert!(phase51_extended_source_owner_fills_from_trade_message(
                &trade_msg,
                2,
                "extended",
                32,
                1_700_000_000_666_i64,
                "ETH-USD",
            )
            .is_empty());
        }
    }

    #[test]
    fn trade_message_other_market_or_without_handle_creates_no_source_owner_fill() {
        for trade in [
            serde_json::json!({
                "market": "BTC-USD",
                "orderId": "oid_ext_1",
                "isTaker": true
            }),
            serde_json::json!({
                "market": "ETH-USD",
                "isTaker": true
            }),
            serde_json::json!({
                "market": "ETH-USD",
                "id": "trade_id_not_order_id",
                "isTaker": true
            }),
        ] {
            let trade_msg = serde_json::json!({
                "type": "TRADE",
                "seq": 33_u64,
                "data": { "trades": [trade] }
            });

            assert!(phase51_extended_source_owner_fills_from_trade_message(
                &trade_msg,
                2,
                "extended",
                33,
                1_700_000_000_777_i64,
                "ETH-USD",
            )
            .is_empty());
        }
    }

    fn apply_market_event_to_test_state(
        state: &mut GlobalState,
        cfg: &Config,
        event: &MarketDataEvent,
    ) {
        let venue = state.venues.get_mut(0).expect("extended venue");
        let max_levels = cfg.book.depth_levels.max(1) as usize;
        match event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                venue
                    .apply_l2_snapshot(
                        &snapshot.bids,
                        &snapshot.asks,
                        snapshot.seq,
                        snapshot.timestamp_ms,
                        max_levels,
                        cfg.volatility.fv_vol_alpha_short,
                        cfg.volatility.fv_vol_alpha_long,
                    )
                    .expect("apply l2 snapshot");
            }
            MarketDataEvent::L2Delta(delta) => {
                venue
                    .apply_l2_delta(
                        &delta.changes,
                        delta.seq,
                        delta.timestamp_ms,
                        max_levels,
                        cfg.volatility.fv_vol_alpha_short,
                        cfg.volatility.fv_vol_alpha_long,
                    )
                    .expect("apply l2 delta");
            }
            MarketDataEvent::Trade(_) | MarketDataEvent::FundingUpdate(_) => {
                panic!("unexpected market event variant in test");
            }
        }
    }

    #[test]
    fn fixture_snapshot_parses() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        assert!(!feed.snapshot.bids.is_empty());
        assert!(!feed.snapshot.asks.is_empty());
    }

    #[test]
    fn delta_applies_to_snapshot_levels() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
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
            seq: 2,
            timestamp_ms: 1_000,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let next = FixtureDelta {
            seq: 4,
            timestamp_ms: 1_010,
            side: "bid".to_string(),
            price: 100.0,
            size: 1.0,
        };
        let mut last_seq = gap.seq;
        let gap_detected = next.seq > last_seq + 1;
        last_seq = next.seq;
        assert!(gap_detected);
        assert_eq!(last_seq, 4);
    }

    #[test]
    fn deterministic_serialization_roundtrip() {
        let fixture_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/roadmap_b/extended");
        let feed = ExtendedFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
        let raw = serde_json::to_string(&feed.snapshot).expect("serialize");
        let reparsed: FixtureSnapshot = serde_json::from_str(&raw).expect("reparse");
        assert_eq!(feed.snapshot.seq, reparsed.seq);
        assert_eq!(feed.snapshot.bids.len(), reparsed.bids.len());
    }

    #[test]
    fn live_snapshot_fixture_parses() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/extended_live_recording");
        let raw =
            std::fs::read_to_string(fixture_dir.join("rest_snapshot.json")).expect("snapshot raw");
        let value: Value = serde_json::from_str(&raw).expect("snapshot json");
        let snapshot = parse_depth_snapshot(&value).expect("parse snapshot");
        assert!(snapshot.last_update_id.unwrap_or(0) > 0);
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[test]
    fn parse_official_rest_snapshot_shape_without_sequence() {
        let payload = serde_json::json!({
            "status": "OK",
            "data": {
                "market": "ETH-USD",
                "bid": [
                    {"price": "1780.5", "qty": "0.25"}
                ],
                "ask": [
                    {"price": "1781.0", "qty": "0.15"}
                ]
            }
        });
        let snapshot = parse_depth_snapshot(&payload).expect("official snapshot");
        assert_eq!(snapshot.last_update_id, None);
        assert_eq!(snapshot.bids.len(), 1);
        assert_eq!(snapshot.asks.len(), 1);
        assert_eq!(snapshot.bids[0].price, 1780.5);
        assert_eq!(snapshot.asks[0].size, 0.15);
    }

    #[test]
    fn live_ws_replay_is_deterministic_and_monotonic() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/roadmap_b/extended_live_recording");
        let snapshot_raw =
            std::fs::read_to_string(fixture_dir.join("rest_snapshot.json")).expect("snapshot raw");
        let snapshot_value: Value = serde_json::from_str(&snapshot_raw).expect("snapshot json");
        let snapshot = parse_depth_snapshot(&snapshot_value).expect("parse snapshot");
        let frames =
            std::fs::read_to_string(fixture_dir.join("ws_frames.jsonl")).expect("ws frames");

        let collect_events = |snapshot_id: Option<u64>| -> Vec<MarketDataEvent> {
            let mut state = ExtendedSeqState::new(snapshot_id, 0);
            let mut events = Vec::new();
            for line in frames.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(Some(update)) = parse_depth_update(trimmed) {
                    let outcome = state.apply_update(&update).expect("seq ok");
                    if let Some(event) = outcome {
                        events.push(event);
                    }
                }
            }
            events
        };

        let events_a = collect_events(snapshot.last_update_id);
        let events_b = collect_events(snapshot.last_update_id);
        assert_eq!(events_a, events_b);

        let mut last_ts: Option<TimestampMs> = None;
        for event in events_a {
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

    #[tokio::test]
    async fn fetch_snapshot_uses_official_orderbook_endpoint() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/api/v1/info/markets/ETH-USD/orderbook");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(
                        r#"{"status":"OK","data":{"market":"ETH-USD","bid":[{"price":"1780.5","qty":"0.25"}],"ask":[{"price":"1781.0","qty":"0.15"}]}}"#,
                    );
            })
            .await;
        let (market_tx, _market_rx) = mpsc::channel(4);
        let connector = ExtendedConnector::new(
            ExtendedConfig {
                ws_url: "wss://api.extended.test/stream".to_string(),
                private_ws_url: "wss://api.extended.test/stream/account".to_string(),
                rest_url: server.base_url(),
                market: "ETH-USD".to_string(),
                depth_limit: 100,
                venue_index: 0,
                api_key: None,
                trader_cmd: None,
                record_dir: None,
            },
            market_tx,
        );

        let attempt = connector.fetch_snapshot().await;
        mock.assert_async().await;
        assert_eq!(attempt.status, "ok");
        assert_eq!(attempt.http_status, Some(200));
        assert_eq!(attempt.bid_levels, 1);
        assert_eq!(attempt.ask_levels, 1);
        assert!(attempt
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.last_update_id)
            .is_none());
    }

    #[tokio::test]
    async fn rest_place_order_uses_bridge_command() {
        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARAPHINA_EXTENDED_BRIDGE_OP}" != "place" ]]; then
  echo "unexpected op=${PARAPHINA_EXTENDED_BRIDGE_OP}" >&2
  exit 1
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"client_order_id":"co_post_only"'* ]]; then
  echo "payload missing client_order_id" >&2
  exit 2
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"post_only":true'* ]]; then
  echo "payload missing post_only=true" >&2
  exit 3
fi
printf '%s\n' '{"order_id":"12345","client_order_id":"co_post_only"}'
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "BTC-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);

        let resp = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
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
        assert_eq!(resp.order_id.as_deref(), Some("12345"));
    }

    #[test]
    fn parse_market_info_price_tick_size_reads_min_price_change() {
        let payload = serde_json::json!({
            "status": "OK",
            "data": [{
                "name": "ETH-USD",
                "tradingConfig": {
                    "minPriceChange": "0.1"
                }
            }]
        });
        assert_eq!(
            parse_market_info_price_tick_size(&payload, "ETH-USD"),
            Some(0.1)
        );
    }

    #[tokio::test]
    async fn rest_place_order_snaps_post_only_price_to_market_tick() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard = EnvVarGuard::new("PARAPHINA_EXTENDED_PRICE_TICK_SIZE");
        std::env::set_var("PARAPHINA_EXTENDED_PRICE_TICK_SIZE", "0.1");

        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"price":"100"'* ]]; then
  echo "payload missing snapped price" >&2
  echo "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" >&2
  exit 4
fi
printf '%s\n' '{"order_id":"12345","client_order_id":"co_snap"}'
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "ETH-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);

        let resp = client
            .place_order(LiveRestPlaceRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
                side: Side::Buy,
                price: 100.07,
                size: 0.01,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "co_snap".to_string(),
            })
            .await
            .expect("place order");
        assert_eq!(resp.order_id.as_deref(), Some("12345"));
    }

    #[tokio::test]
    async fn rest_replace_order_uses_bridge_command_with_external_cancel_id() {
        let _env_lock = ENV_MUTEX.lock().expect("env mutex");
        let _guard = EnvVarGuard::new("PARAPHINA_EXTENDED_PRICE_TICK_SIZE");
        std::env::set_var("PARAPHINA_EXTENDED_PRICE_TICK_SIZE", "0.1");

        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARAPHINA_EXTENDED_BRIDGE_OP}" != "replace" ]]; then
  echo "unexpected op=${PARAPHINA_EXTENDED_BRIDGE_OP}" >&2
  exit 1
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"order_id":"d0_mm_v4_buy"'* ]]; then
  echo "payload missing prior external id" >&2
  echo "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" >&2
  exit 2
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"client_order_id":"d1_mm_v4_buy"'* ]]; then
  echo "payload missing new external id" >&2
  echo "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" >&2
  exit 3
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"price":"100"'* ]]; then
  echo "payload missing snapped price" >&2
  echo "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" >&2
  exit 4
fi
printf '%s\n' '{"order_id":"1784963886257016833","client_order_id":"d1_mm_v4_buy"}'
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "ETH-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);

        let resp = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
                order_id: "d0_mm_v4_buy".to_string(),
                side: Side::Buy,
                price: 100.07,
                size: 0.01,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "d1_mm_v4_buy".to_string(),
            })
            .await
            .expect("replace order");
        assert_eq!(resp.order_id.as_deref(), Some("1784963886257016833"));
        assert_eq!(resp.client_order_id.as_deref(), Some("d1_mm_v4_buy"));
    }

    #[tokio::test]
    async fn rest_replace_order_rejects_numeric_identity_before_bridge() {
        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
echo "bridge should not be called" >&2
exit 9
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "ETH-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);

        let err = client
            .replace_order(LiveRestReplaceRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
                order_id: "1784963886257016832".to_string(),
                side: Side::Sell,
                price: 100.0,
                size: 0.01,
                purpose: crate::types::OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: "d1_mm_v4_sell".to_string(),
            })
            .await
            .expect_err("numeric Extended replace identity should fail");
        assert!(err
            .message
            .contains("extended_native_replace_requires_external_id"));
    }

    #[tokio::test]
    async fn rest_cancel_all_uses_bridge_command() {
        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARAPHINA_EXTENDED_BRIDGE_OP}" != "cancel_all" ]]; then
  echo "unexpected op=${PARAPHINA_EXTENDED_BRIDGE_OP}" >&2
  exit 1
fi
if [[ "${PARAPHINA_EXTENDED_BRIDGE_PAYLOAD}" != *'"market":"BTC-USD"'* ]]; then
  echo "payload missing market" >&2
  exit 2
fi
printf '%s\n' '{"count":2}'
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "BTC-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);

        let _ = client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
            })
            .await
            .expect("cancel_all");
    }

    #[tokio::test]
    async fn account_snapshot_uses_bridge_command() {
        let tmp = TempDir::new().expect("temp dir");
        let trader_cmd = write_bridge_script(
            &tmp,
            r#"#!/usr/bin/env bash
set -euo pipefail
if [[ "${PARAPHINA_EXTENDED_BRIDGE_OP}" != "snapshot" ]]; then
  echo "unexpected op=${PARAPHINA_EXTENDED_BRIDGE_OP}" >&2
  exit 1
fi
printf '%s\n' '{"timestamp_ms":1700000000000,"collateral_asset":"USDC","balance_usd":100.0,"used_usd":12.5,"available_usd":87.5,"positions":[{"market":"BTC-USD","size":0.01,"entry_price":2500.0,"liquidation_price":2000.0,"updated_at":1700000000100}]}'
"#,
        );
        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "BTC-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: Some("test-key".to_string()),
            trader_cmd: Some(trader_cmd),
            record_dir: None,
        };
        let client = ExtendedRestClient::new(cfg);
        let before = now_ms();
        let snapshot = client
            .fetch_account_snapshot("extended", 3)
            .await
            .expect("snapshot");
        assert_eq!(snapshot.venue_index, 3);
        assert_eq!(snapshot.venue_id, "extended");
        assert!(snapshot.timestamp_ms >= before);
        assert!(snapshot.seq >= before as u64);
        assert_eq!(snapshot.margin.balance_usd, 100.0);
        assert_eq!(snapshot.margin.used_usd, 12.5);
        assert_eq!(snapshot.margin.available_usd, 87.5);
        assert_eq!(snapshot.balances[0].asset, "USDC");
        assert_eq!(snapshot.positions.len(), 1);
        assert_eq!(snapshot.positions[0].symbol, "BTC-USD");
        assert_eq!(snapshot.positions[0].size, 0.01);
        assert_eq!(snapshot.liquidation.price_liq, Some(2000.0));
    }

    #[test]
    fn empty_position_snapshot_uses_fresh_timestamp() {
        let stale_balance_ts = 1_700_000_000_000_i64;
        let before = now_ms();
        let snapshot = ExtendedBridgeSnapshot {
            timestamp_ms: Some(stale_balance_ts),
            collateral_asset: Some("USDC".to_string()),
            balance_usd: 100.0,
            used_usd: 0.0,
            available_usd: 100.0,
            positions: Vec::new(),
        }
        .into_account_snapshot("extended", 0);

        assert!(snapshot.timestamp_ms >= before);
        assert!(snapshot.seq >= before as u64);
        assert!(snapshot.positions.is_empty());
        assert_eq!(snapshot.margin.available_usd, 100.0);
    }

    #[test]
    fn ws_frame_whitespace_is_ignored_and_json_parses() {
        assert!(clean_ws_payload("").is_none());
        assert!(clean_ws_payload("   \n\t ").is_none());
        assert!(clean_ws_payload("\0\0").is_none());
        let raw = r#"{"type":"DELTA","seq":2,"ts":1700000000001,"data":{"m":"BTC-USD","t":"DELTA","b":[{"p":"100","q":"-1.5","c":"1.0"}],"a":[{"p":"101","q":"0.5","c":"2.0"}]}}"#;
        let cleaned = clean_ws_payload(raw).expect("cleaned");
        let update = parse_depth_update(cleaned).expect("parse").expect("update");
        assert_eq!(update.symbol, "BTC-USD");
        assert_eq!(update.seq, 2);
        assert_eq!(update.bids[0].size, 1.0, "delta must use absolute size c");
        assert_eq!(update.asks[0].size, 2.0, "delta must use absolute size c");
        assert_eq!(update.bids.len(), 1);
        assert_eq!(update.asks.len(), 1);
    }

    #[test]
    fn ws_frame_empty_json_is_non_fatal() {
        let cleaned = clean_ws_payload("{}").expect("cleaned");
        let parsed = parse_depth_update(cleaned).expect("parse");
        assert!(parsed.is_none());
    }

    #[test]
    fn ws_value_decode_handles_object_levels() {
        let value = serde_json::json!({
            "data": {
                "bids": [{"price":"100","size":"2"}],
                "asks": [{"price":"101","size":"3"}],
                "ts": 1700000000000i64
            }
        });
        let top = decode_top_from_value(&value).expect("top");
        assert_eq!(top.best_bid_px, 100.0);
        assert_eq!(top.best_bid_sz, 2.0);
        assert_eq!(top.best_ask_px, 101.0);
        assert_eq!(top.best_ask_sz, 3.0);
    }

    #[test]
    fn ws_delta_messages_are_not_misclassified_as_snapshots() {
        let mut fallback_seq = 0_u64;
        let delta = serde_json::json!({
            "type": "DELTA",
            "seq": 101_u64,
            "ts": 1_700_000_000_200_i64,
            "data": {
                "m": "ETH-USD",
                "t": "DELTA",
                "b": [{"p":"2000.0","q":"-1.0","c":"1.5"}],
                "a": [{"p":"2000.1","q":"-3.0","c":"0"}]
            }
        });
        assert!(
            parse_depth_snapshot_from_ws(&delta, "ETH-USD", 0, &mut fallback_seq).is_none(),
            "Extended DELTA frames must not be treated as snapshots"
        );
    }

    #[test]
    fn ws_one_sided_snapshot_keeps_opposite_side_and_depth() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let mut fallback_seq = 0_u64;

        let full_snapshot = serde_json::json!({
            "type": "SNAPSHOT",
            "seq": 100_u64,
            "ts": 1_700_000_000_000_i64,
            "data": {
                "m": "ETH-USD",
                "b": [{"p":"2000.0","q":"2.5"}],
                "a": [{"p":"2000.1","q":"3.0"}]
            }
        });
        let initial = parse_depth_snapshot_from_ws(&full_snapshot, "ETH-USD", 0, &mut fallback_seq)
            .expect("initial ws snapshot event");
        apply_market_event_to_test_state(&mut state, &cfg, &initial);
        assert!(state.venues[0].orderbook_l2.best_bid().is_some());
        assert!(state.venues[0].orderbook_l2.best_ask().is_some());
        assert!(
            state.venues[0].depth_near_mid > 0.0,
            "initial snapshot should set positive depth"
        );

        let one_sided_snapshot = serde_json::json!({
            "type": "SNAPSHOT",
            "seq": 101_u64,
            "ts": 1_700_000_000_200_i64,
            "data": {
                "m": "ETH-USD",
                "b": [{"p":"2000.0","q":"1.5"}],
                "a": [{"p":"2000.1","q":"0"}]
            }
        });
        let guarded =
            parse_depth_snapshot_from_ws(&one_sided_snapshot, "ETH-USD", 0, &mut fallback_seq)
                .expect("guarded ws event");
        match &guarded {
            MarketDataEvent::L2Delta(delta) => {
                assert!(
                    !delta.changes.is_empty(),
                    "one-sided snapshot should still produce a side update"
                );
                assert!(
                    delta
                        .changes
                        .iter()
                        .all(|change| change.side == BookSide::Bid),
                    "empty ask side must not clear the existing ask book"
                );
            }
            other => panic!("expected L2Delta for guarded one-sided snapshot, got {other:?}"),
        }
        apply_market_event_to_test_state(&mut state, &cfg, &guarded);

        let venue = &state.venues[0];
        assert!(
            venue.orderbook_l2.best_bid().is_some(),
            "bid side must remain present"
        );
        assert!(
            venue.orderbook_l2.best_ask().is_some(),
            "ask side must remain present"
        );
        assert!(
            venue.depth_near_mid > 0.0,
            "depth must stay positive after one-sided snapshot frame"
        );

        update_toxicity_and_health(&mut state, &cfg, 1_700_000_002_000_i64);
        assert_ne!(
            state.venues[0].status,
            VenueStatus::Disabled,
            "one-sided snapshot frame must not disable venue solely via depth collapse"
        );
    }

    #[test]
    fn normalize_extended_market_variants() {
        assert_eq!(normalize_extended_market("BTCUSDT"), "BTC-USD");
        assert_eq!(normalize_extended_market("BTCUSD"), "BTC-USD");
        assert_eq!(normalize_extended_market("BTC-USD"), "BTC-USD");
        assert_eq!(normalize_extended_market("btc-usd-perp"), "BTC-USD");
    }

    #[test]
    fn bridge_open_order_side_accepts_uppercase_variants() {
        let order: ExtendedBridgeOpenOrder = serde_json::from_value(serde_json::json!({
            "order_id": "oid_1",
            "client_order_id": "co_ext_1",
            "side": "SELL",
            "price": 2365.1,
            "size": 0.02
        }))
        .expect("parse uppercase side");
        assert_eq!(order.side, Side::Sell);

        let order: ExtendedBridgeOpenOrder = serde_json::from_value(serde_json::json!({
            "order_id": "oid_2",
            "client_order_id": "co_ext_2",
            "side": "buy",
            "price": 2364.9,
            "size": 0.01
        }))
        .expect("parse lowercase side");
        assert_eq!(order.side, Side::Buy);
    }

    #[test]
    fn extended_stale_ms_falls_back_to_state_override_with_guardband() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, extended_stale_ms);
        assert_eq!(stale_ms, 1000);
    }

    #[test]
    fn extended_stale_ms_prefers_explicit_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STALE_MS".to_string(),
            "2200".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let stale_ms = with_env_overrides(&overrides, extended_stale_ms);
        assert_eq!(stale_ms, 2200);
    }

    #[test]
    fn extended_connect_book_timeout_defaults_to_750ms() {
        let overrides = BTreeMap::new();
        let timeout = with_env_overrides(&overrides, extended_connect_book_timeout);
        assert_eq!(timeout, Duration::from_millis(750));
    }

    #[test]
    fn extended_connect_first_frame_timeout_defaults_to_1500ms() {
        let overrides = BTreeMap::new();
        let timeout = with_env_overrides(&overrides, extended_connect_first_frame_timeout);
        assert_eq!(timeout, Duration::from_millis(1_500));
    }

    #[test]
    fn extended_connect_first_frame_timeout_aligns_to_state_stale_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let timeout = with_env_overrides(&overrides, extended_connect_first_frame_timeout);
        assert_eq!(timeout, Duration::from_millis(1_100));
    }

    #[test]
    fn extended_connect_first_frame_timeout_prefers_env_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_CONNECT_FIRST_FRAME_TIMEOUT_MS".to_string(),
            "1600".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let timeout = with_env_overrides(&overrides, extended_connect_first_frame_timeout);
        assert_eq!(timeout, Duration::from_millis(1_600));
    }

    #[test]
    fn extended_connect_control_frame_only_timeout_defaults_to_1500ms() {
        let overrides = BTreeMap::new();
        let timeout = with_env_overrides(&overrides, extended_connect_control_frame_only_timeout);
        assert_eq!(timeout, Duration::from_millis(1_500));
    }

    #[test]
    fn extended_connect_control_frame_only_timeout_aligns_to_state_stale_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let timeout = with_env_overrides(&overrides, extended_connect_control_frame_only_timeout);
        assert_eq!(timeout, Duration::from_millis(1_450));
    }

    #[test]
    fn extended_connect_control_frame_only_timeout_prefers_env_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_CONNECT_CONTROL_FRAME_ONLY_TIMEOUT_MS".to_string(),
            "1700".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let timeout = with_env_overrides(&overrides, extended_connect_control_frame_only_timeout);
        assert_eq!(timeout, Duration::from_millis(1_700));
    }

    #[test]
    fn extended_connect_control_frame_only_hedge_start_after_tracks_first_data_timeout() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "1500".to_string(),
        );
        let timeout = with_env_overrides(
            &overrides,
            extended_connect_control_frame_only_hedge_start_after,
        );
        assert_eq!(timeout, Duration::from_millis(1_000));
    }

    #[test]
    fn extended_connect_control_frame_only_hedge_start_after_has_floor() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_CONNECT_FIRST_FRAME_TIMEOUT_MS".to_string(),
            "800".to_string(),
        );
        let timeout = with_env_overrides(
            &overrides,
            extended_connect_control_frame_only_hedge_start_after,
        );
        assert_eq!(timeout, Duration::from_millis(750));
    }

    #[test]
    fn extended_post_publish_fallback_defaults_to_state_stale_guardbands() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        let after = with_env_overrides(&overrides, extended_post_publish_fallback_after);
        let deadline = with_env_overrides(&overrides, extended_post_publish_fallback_deadline);
        assert_eq!(after, Duration::from_millis(2_700));
        assert_eq!(deadline, Duration::from_millis(2_950));
    }

    #[test]
    fn extended_post_publish_fallback_prefers_safe_env_overrides() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_AFTER_MS".to_string(),
            "1800".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_MS".to_string(),
            "2300".to_string(),
        );
        let after = with_env_overrides(&overrides, extended_post_publish_fallback_after);
        let deadline = with_env_overrides(&overrides, extended_post_publish_fallback_deadline);
        assert_eq!(after, Duration::from_millis(1_800));
        assert_eq!(deadline, Duration::from_millis(2_300));
    }

    #[test]
    fn extended_post_publish_fallback_env_overrides_clamp_before_state_stale() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE".to_string(),
            "3000".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_AFTER_MS".to_string(),
            "5000".to_string(),
        );
        overrides.insert(
            "PARAPHINA_EXTENDED_POST_PUBLISH_FALLBACK_DEADLINE_MS".to_string(),
            "6000".to_string(),
        );
        let after = with_env_overrides(&overrides, extended_post_publish_fallback_after);
        let deadline = with_env_overrides(&overrides, extended_post_publish_fallback_deadline);
        assert_eq!(after, Duration::from_millis(2_950));
        assert_eq!(deadline, Duration::from_millis(2_950));
    }

    #[test]
    fn extended_connect_book_timeout_prefers_env_override() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_CONNECT_BOOK_TIMEOUT_MS".to_string(),
            "1200".to_string(),
        );
        let timeout = with_env_overrides(&overrides, extended_connect_book_timeout);
        assert_eq!(timeout, Duration::from_millis(1200));
    }

    #[test]
    fn extended_stale_churn_window_and_limit_have_expected_defaults() {
        let overrides = BTreeMap::new();
        let window = with_env_overrides(&overrides, extended_stale_churn_window);
        let limit = with_env_overrides(&overrides, extended_stale_churn_limit);
        let healthy_reset = with_env_overrides(&overrides, extended_stale_churn_healthy_reset);
        assert_eq!(window, Duration::from_millis(120_000));
        assert_eq!(limit, 2);
        assert_eq!(healthy_reset, Duration::from_millis(30_000));
    }

    #[test]
    fn extended_bootstrap_churn_window_and_limit_have_expected_defaults() {
        let overrides = BTreeMap::new();
        let window = with_env_overrides(&overrides, extended_bootstrap_churn_window);
        let limit = with_env_overrides(&overrides, extended_bootstrap_churn_limit);
        let healthy_reset = with_env_overrides(&overrides, extended_bootstrap_churn_healthy_reset);
        assert_eq!(window, Duration::from_millis(120_000));
        assert_eq!(limit, 2);
        assert_eq!(healthy_reset, Duration::from_millis(30_000));
    }

    #[test]
    fn extended_transport_gap_reconnect_policy_is_fast_within_churn_budget() {
        let backoff = Duration::from_secs(4);
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::StaleWatchdog,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::StaleWatchdog,
                    2,
                    2,
                    0,
                    2,
                ),
                backoff,
            ),
            Duration::from_millis(100)
        );
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::BootstrapNoFirstFrame,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::BootstrapNoFirstFrame,
                    0,
                    2,
                    2,
                    2,
                ),
                backoff
            ),
            Duration::from_millis(100)
        );
        assert!(extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::StaleWatchdog,
            2,
            2,
            0,
            2,
        ));
        assert!(extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::BootstrapNoFirstFrame,
            0,
            2,
            2,
            2,
        ));
    }

    #[test]
    fn extended_transport_gap_reconnect_policy_escalates_after_churn_budget() {
        let backoff = Duration::from_secs(4);
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::StaleWatchdog,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::StaleWatchdog,
                    3,
                    2,
                    0,
                    2,
                ),
                backoff,
            ),
            backoff
        );
        assert!(!extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::StaleWatchdog,
            3,
            2,
            0,
            2,
        ));
        assert!(!extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::BootstrapNoFirstFrame,
            0,
            2,
            3,
            2,
        ));
    }

    #[test]
    fn extended_failure_reconnect_policy_preserves_backoff() {
        let backoff = Duration::from_secs(8);
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::ReadTimeout,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::ReadTimeout,
                    0,
                    2,
                    0,
                    2,
                ),
                backoff,
            ),
            backoff
        );
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::SeqGap,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::SeqGap,
                    0,
                    2,
                    0,
                    2,
                ),
                backoff,
            ),
            backoff
        );
        assert!(!extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::ReadTimeout,
            0,
            2,
            0,
            2,
        ));
        assert!(!extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::SeqGap,
            0,
            2,
            0,
            2,
        ));
    }

    #[test]
    fn extended_degraded_rebootstrap_reconnect_sleep_is_env_capped_after_churn_budget() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "PARAPHINA_EXTENDED_DEGRADED_REBOOTSTRAP_MAX_SLEEP_MS".to_string(),
            "1000".to_string(),
        );
        let sleep = with_env_overrides(&overrides, || {
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                    8,
                    7,
                    0,
                    2,
                ),
                Duration::from_secs(8),
            )
        });
        assert_eq!(sleep, Duration::from_millis(1_000));
        assert!(!extended_failure_escalation_suppressed(
            ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
            8,
            7,
            0,
            2,
        ));
    }

    #[test]
    fn extended_degraded_rebootstrap_reconnect_sleep_keeps_backoff_without_env_cap() {
        let backoff = Duration::from_secs(8);
        assert_eq!(
            extended_public_reconnect_sleep(
                ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                extended_failure_escalation_suppressed(
                    ExtendedPublicReconnectReason::DegradedStreamRebootstrapGap,
                    8,
                    7,
                    0,
                    2,
                ),
                backoff,
            ),
            backoff
        );
    }

    #[test]
    fn extended_stale_watchdog_churn_resets_after_healthy_session() {
        let mut churn = ExtendedReconnectChurnState::default();
        let now = Instant::now();
        let window = Duration::from_millis(120_000);
        assert_eq!(churn.observe(now, window), 1);
        assert_eq!(churn.observe(now + Duration::from_secs(1), window), 2);
        assert_eq!(
            churn.reset_after_healthy_session(
                Duration::from_millis(30_000),
                Duration::from_millis(30_000),
            ),
            Some(2)
        );
        assert!(churn.reconnects.is_empty());
    }

    #[test]
    fn extended_bootstrap_timeout_reason_classifies_first_progress_stage() {
        assert_eq!(
            extended_bootstrap_timeout_reason(false, false, false),
            ExtendedPublicReconnectReason::BootstrapNoFirstFrame
        );
        assert_eq!(
            extended_bootstrap_timeout_reason(true, false, false),
            ExtendedPublicReconnectReason::BootstrapFrameNoBook
        );
        assert_eq!(
            extended_bootstrap_timeout_reason(true, true, false),
            ExtendedPublicReconnectReason::BootstrapBookNoPublish
        );
    }

    #[test]
    fn extended_bootstrap_timeout_stage_separates_first_frame_from_post_first_frame() {
        assert_eq!(extended_bootstrap_timeout_stage(false), "first_frame");
        assert_eq!(extended_bootstrap_timeout_stage(true), "post_first_frame");
    }

    #[test]
    fn extended_control_frame_only_session_hedge_requires_control_frame_and_seed_bridge() {
        assert!(extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            false,
            true,
            false
        ));
        assert!(!extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::Depth1,
            false,
            false,
            true,
            false
        ));
        assert!(!extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            true,
            true,
            false
        ));
        assert!(!extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            false,
            false,
            false
        ));
        assert!(!extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            false,
            true,
            true
        ));
        assert!(!extended_should_start_control_frame_only_session_hedge(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            false,
            true,
            false
        ));
    }

    #[test]
    fn extended_post_publish_fallback_rearms_only_after_successful_post_publish_recovery() {
        assert!(extended_should_rearm_post_publish_stream_fallback(Some(
            ExtendedHedgeMode::PostPublishStreamFallback
        )));
        assert!(!extended_should_rearm_post_publish_stream_fallback(Some(
            ExtendedHedgeMode::BackendAttach
        )));
        assert!(!extended_should_rearm_post_publish_stream_fallback(None));
    }

    #[test]
    fn extended_degraded_stream_rebootstrap_requires_degraded_stream_and_gap() {
        assert!(extended_should_start_degraded_stream_rebootstrap(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            false,
            1_250,
            1_250,
            1_250,
            1_200,
        ));
        assert!(!extended_should_start_degraded_stream_rebootstrap(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            false,
            1_250,
            1_250,
            1_250,
            1_200,
        ));
        assert!(!extended_should_start_degraded_stream_rebootstrap(
            ExtendedBootstrapStreamKind::FullOrderbook,
            false,
            false,
            1_250,
            1_250,
            1_250,
            1_200,
        ));
        assert!(!extended_should_start_degraded_stream_rebootstrap(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            true,
            1_250,
            1_250,
            1_250,
            1_200,
        ));
        assert!(!extended_should_start_degraded_stream_rebootstrap(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            false,
            1_100,
            1_250,
            1_250,
            1_200,
        ));
    }

    #[test]
    fn extended_degraded_stream_watchdog_arming_requires_degraded_preference_and_clear_session() {
        assert!(extended_should_arm_degraded_stream_rebootstrap_watchdog(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            true,
            false,
            false,
        ));
        assert!(!extended_should_arm_degraded_stream_rebootstrap_watchdog(
            ExtendedBootstrapStreamKind::Depth1,
            true,
            true,
            false,
            false,
        ));
        assert!(!extended_should_arm_degraded_stream_rebootstrap_watchdog(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            false,
            false,
            false,
        ));
        assert!(!extended_should_arm_degraded_stream_rebootstrap_watchdog(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            true,
            true,
            false,
        ));
        assert!(!extended_should_arm_degraded_stream_rebootstrap_watchdog(
            ExtendedBootstrapStreamKind::FullOrderbook,
            true,
            true,
            false,
            true,
        ));
    }

    #[test]
    fn extended_degraded_stream_watchdog_fire_prioritizes_connector_freshness() {
        assert!(extended_should_fire_degraded_stream_rebootstrap_watchdog(
            1_250, 1_250, 1_250, 1_200,
        ));
        assert!(extended_should_fire_degraded_stream_rebootstrap_watchdog(
            1_250, 1_250, 1_100, 1_200,
        ));
        assert!(!extended_should_fire_degraded_stream_rebootstrap_watchdog(
            1_100, 1_250, 1_250, 1_200,
        ));
        assert!(!extended_should_fire_degraded_stream_rebootstrap_watchdog(
            1_250, 1_100, 1_250, 1_200,
        ));
        assert!(!extended_should_fire_degraded_stream_rebootstrap_watchdog(
            1_250, 1_250, 1_400, 1_500,
        ));
    }

    #[test]
    fn extended_seq_error_reason_maps_gap_and_mismatch() {
        assert_eq!(
            extended_seq_error_reason("extended seq gap last_seq=10 next_seq=12"),
            ExtendedPublicReconnectReason::SeqGap
        );
        assert_eq!(
            extended_seq_error_reason("extended seq mismatch snapshot_seq=10 ws_seq=9"),
            ExtendedPublicReconnectReason::SeqMismatch
        );
        assert_eq!(
            extended_seq_error_reason("extended parse failed"),
            ExtendedPublicReconnectReason::ParseError
        );
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

        // After reset with no book events, anchor falls back to connect_start_ns
        let connect_start_ns = 1_000;
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, connect_start_ns);

        // Non-book parsed events must NOT advance the watchdog anchor
        freshness.last_parsed_ns.store(2_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(
            anchor, connect_start_ns,
            "non-book parsed events must not advance watchdog anchor"
        );

        // Book events advance the anchor
        freshness.last_book_event_ns.store(3_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 3_000);

        // last_published_ns also advances the anchor
        freshness.last_published_ns.store(4_000, Ordering::Relaxed);
        let anchor = freshness.anchor_with_connect_start(connect_start_ns);
        assert_eq!(anchor, 4_000);
    }

    #[test]
    fn freshness_rest_seed_bridge_advances_anchor_before_first_data() {
        let freshness = Freshness::default();
        freshness.reset_for_new_connection();
        freshness.activate_rest_seed_bridge(2_500);
        assert_eq!(freshness.last_parsed_ns.load(Ordering::Relaxed), 2_500);
        assert_eq!(freshness.last_book_event_ns.load(Ordering::Relaxed), 2_500);
        assert_eq!(freshness.last_published_ns.load(Ordering::Relaxed), 2_500);
        let anchor = freshness.anchor_with_connect_start(1_000);
        assert_eq!(anchor, 2_500);
    }

    #[test]
    fn extended_watchdog_uses_transport_age_after_first_publish() {
        let freshness = Freshness::default();
        freshness
            .last_ws_rx_ns
            .store(ms_to_ns(9_000), Ordering::Relaxed);
        freshness
            .last_book_event_ns
            .store(ms_to_ns(1_000), Ordering::Relaxed);
        freshness
            .last_published_ns
            .store(ms_to_ns(1_000), Ordering::Relaxed);

        assert!(
            !extended_watchdog_should_fire(&freshness, 0, true, 1_800, 25_000, ms_to_ns(20_000)),
            "fresh ws transport should suppress reconnect even when book-progress age is old"
        );
        assert!(
            extended_watchdog_should_fire(&freshness, 0, true, 1_800, 25_000, ms_to_ns(40_000)),
            "transport watchdog should eventually fire on extended ws silence"
        );
    }

    #[test]
    fn extended_watchdog_keeps_book_progress_age_before_first_publish() {
        let freshness = Freshness::default();
        freshness
            .last_book_event_ns
            .store(ms_to_ns(2_000), Ordering::Relaxed);
        freshness
            .last_published_ns
            .store(ms_to_ns(2_000), Ordering::Relaxed);
        freshness
            .last_ws_rx_ns
            .store(ms_to_ns(20_000), Ordering::Relaxed);

        assert!(
            !extended_watchdog_should_fire(
                &freshness,
                ms_to_ns(1_000),
                false,
                1_800,
                25_000,
                ms_to_ns(3_500),
            ),
            "bootstrap watchdog should stay quiet inside the original stale budget"
        );
        assert!(
            extended_watchdog_should_fire(
                &freshness,
                ms_to_ns(1_000),
                false,
                1_800,
                25_000,
                ms_to_ns(4_000),
            ),
            "bootstrap watchdog should still key off book-progress age before first publish"
        );
    }

    #[test]
    fn parse_public_funding_market_stats_fixture() {
        // Test parsing Extended's /api/v1/info/markets/{market}/stats response format
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/extended/public_market_stats.json");
        let raw = std::fs::read_to_string(&fixture_path).expect("read fixture");
        let value: Value = serde_json::from_str(&raw).expect("parse fixture JSON");

        let cfg = ExtendedConfig {
            ws_url: "wss://example.invalid".to_string(),
            private_ws_url: "wss://example.invalid/account".to_string(),
            rest_url: "https://api.starknet.extended.exchange".to_string(),
            market: "ETH-USD".to_string(),
            depth_limit: 10,
            venue_index: 0,
            api_key: None,
            trader_cmd: None,
            record_dir: None,
        };

        let funding =
            parse_public_funding(&value, &cfg).expect("parse_public_funding should succeed");

        // Verify rate parsing: fixture has fundingRate "0.000013" (hourly)
        // rate_8h should be 0.000013 * 8 = 0.000104
        assert!(
            funding.funding_rate_native.is_some(),
            "native rate should be present"
        );
        let native = funding.funding_rate_native.unwrap();
        assert!(
            (native - 0.000013).abs() < 1e-10,
            "native rate mismatch: {}",
            native
        );

        assert!(
            funding.funding_rate_8h.is_some(),
            "8h rate should be present"
        );
        let rate_8h = funding.funding_rate_8h.unwrap();
        assert!(
            (rate_8h - 0.000104).abs() < 1e-10,
            "8h rate mismatch: {}",
            rate_8h
        );

        // Verify interval is detected as hourly (3600s)
        assert_eq!(
            funding.interval_sec,
            Some(3600),
            "interval_sec should be 3600"
        );

        // Verify next_funding_ms is extracted from "nextFundingRate" field
        // Fixture has: "nextFundingRate": 1770314400000
        assert_eq!(
            funding.next_funding_ms,
            Some(1770314400000),
            "next_funding_ms mismatch"
        );

        // Verify source and settlement
        assert!(matches!(funding.source, FundingSource::MarketDataRest));
        assert_eq!(
            funding.settlement_price_kind,
            Some(SettlementPriceKind::Mark)
        );

        // Verify venue info
        assert_eq!(funding.venue_index, 0);
        assert_eq!(funding.venue_id, "ETH-USD");
    }
}
