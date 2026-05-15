//! Live REST execution gateway (feature-gated).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;

use crate::config::Config;
use crate::io::{GatewayPolicy, RateLimiter};
use crate::live::instrument::InstrumentSpec;
use crate::live::ops::{config_hash, LiveMetrics};
use crate::live::trade_mode::TradeMode;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

pub type LiveResult<T> = Result<T, LiveGatewayError>;
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportHint {
    #[default]
    Default,
    HyperliquidSyncControl,
}

impl TransportHint {
    pub fn is_hyperliquid_sync_control(self) -> bool {
        matches!(self, Self::HyperliquidSyncControl)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiveGatewayErrorKind {
    Retryable,
    Fatal,
    PostOnlyReject,
    ReduceOnlyViolation,
    RateLimited,
}

#[derive(Debug, Clone)]
pub struct LiveGatewayError {
    pub kind: LiveGatewayErrorKind,
    pub message: String,
}

impl LiveGatewayError {
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            kind: LiveGatewayErrorKind::Retryable,
            message: message.into(),
        }
    }

    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            kind: LiveGatewayErrorKind::Fatal,
            message: message.into(),
        }
    }

    pub fn post_only_reject(message: impl Into<String>) -> Self {
        Self {
            kind: LiveGatewayErrorKind::PostOnlyReject,
            message: message.into(),
        }
    }

    pub fn reduce_only_violation(message: impl Into<String>) -> Self {
        Self {
            kind: LiveGatewayErrorKind::ReduceOnlyViolation,
            message: message.into(),
        }
    }

    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self {
            kind: LiveGatewayErrorKind::RateLimited,
            message: message.into(),
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind,
            LiveGatewayErrorKind::Retryable | LiveGatewayErrorKind::RateLimited
        )
    }

    pub fn reason_label(&self) -> &'static str {
        match self.kind {
            LiveGatewayErrorKind::Retryable => "retryable",
            LiveGatewayErrorKind::Fatal => "fatal",
            LiveGatewayErrorKind::PostOnlyReject => "post_only_reject",
            LiveGatewayErrorKind::ReduceOnlyViolation => "reduce_only_violation",
            LiveGatewayErrorKind::RateLimited => "rate_limited",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiveRestResponse {
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LiveRestPlaceRequest {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
    pub client_order_id: String,
}

#[derive(Debug, Clone)]
pub struct LiveRestCancelRequest {
    pub venue_index: usize,
    pub venue_id: String,
    pub order_id: String,
}

#[derive(Debug, Clone)]
pub struct LiveRestCancelAllRequest {
    pub venue_index: usize,
    pub venue_id: String,
}

#[derive(Debug, Clone)]
pub struct LiveRestReplaceRequest {
    pub venue_index: usize,
    pub venue_id: String,
    pub order_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
    pub client_order_id: String,
}

pub trait LiveRestClient: Send + Sync {
    fn place_order(&self, req: LiveRestPlaceRequest)
        -> BoxFuture<'_, LiveResult<LiveRestResponse>>;
    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>>;
    fn cancel_all(
        &self,
        req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>>;

    fn replace_order(
        &self,
        _req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async { Err(LiveGatewayError::fatal("native_replace_unsupported")) })
    }

    fn place_order_with_hint(
        &self,
        req: LiveRestPlaceRequest,
        _hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.place_order(req)
    }

    fn cancel_order_with_hint(
        &self,
        req: LiveRestCancelRequest,
        _hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.cancel_order(req)
    }

    fn cancel_all_with_hint(
        &self,
        req: LiveRestCancelAllRequest,
        _hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.cancel_all(req)
    }

    fn replace_order_with_hint(
        &self,
        req: LiveRestReplaceRequest,
        _hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.replace_order(req)
    }

    /// Batch place: default falls back to serial. Venues with native batch
    /// APIs (e.g. Hyperliquid `orders` array, Lighter `sendTxBatch`) should override.
    fn place_batch(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(reqs.len());
            for req in reqs {
                results.push(self.place_order(req).await);
            }
            results
        })
    }

    fn place_batch_with_hint(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
        _hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.place_batch(reqs)
    }

    /// Batch cancel: default falls back to serial.
    fn cancel_batch(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(reqs.len());
            for req in reqs {
                results.push(self.cancel_order(req).await);
            }
            results
        })
    }

    fn cancel_batch_with_hint(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
        _hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.cancel_batch(reqs)
    }

    fn replace_batch(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(reqs.len());
            for req in reqs {
                results.push(self.replace_order(req).await);
            }
            results
        })
    }

    fn replace_batch_with_hint(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
        _hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.replace_batch(reqs)
    }
}

fn terminal_sub_lot_reduce_only_passthrough_enabled(venue_id: &str) -> bool {
    let key = if venue_id.eq_ignore_ascii_case("aster") {
        "PARAPHINA_ASTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED"
    } else if venue_id.eq_ignore_ascii_case("hyperliquid") {
        "PARAPHINA_HYPERLIQUID_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED"
    } else if venue_id.eq_ignore_ascii_case("extended") {
        "PARAPHINA_EXTENDED_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED"
    } else if venue_id.eq_ignore_ascii_case("lighter") {
        "PARAPHINA_LIGHTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED"
    } else {
        return false;
    };
    std::env::var(key)
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

fn should_preserve_terminal_sub_lot_reduce_only_size(
    place: &crate::types::PlaceOrderIntent,
    spec: &InstrumentSpec,
) -> bool {
    terminal_sub_lot_reduce_only_passthrough_enabled(place.venue_id.as_ref())
        && place.reduce_only
        && place.time_in_force == TimeInForce::Ioc
        && matches!(place.purpose, OrderPurpose::Hedge | OrderPurpose::Exit)
        && place.size.is_finite()
        && place.size > 0.0
        && place.size < spec.lot_size_tao.max(1e-9)
}

#[derive(Clone)]
pub struct GatewayMux {
    clients: BTreeMap<String, Arc<dyn LiveRestClient>>,
}

impl GatewayMux {
    pub fn new(clients: BTreeMap<String, Arc<dyn LiveRestClient>>) -> Self {
        Self { clients }
    }

    fn client_for(&self, venue_id: &str) -> LiveResult<Arc<dyn LiveRestClient>> {
        self.clients
            .get(venue_id)
            .cloned()
            .ok_or_else(|| LiveGatewayError::fatal(format!("unknown venue_id={}", venue_id)))
    }
}

impl LiveRestClient for GatewayMux {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.place_order(req).await
        })
    }

    fn place_order_with_hint(
        &self,
        req: LiveRestPlaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.place_order_with_hint(req, hint).await
        })
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.cancel_order(req).await
        })
    }

    fn cancel_order_with_hint(
        &self,
        req: LiveRestCancelRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.cancel_order_with_hint(req, hint).await
        })
    }

    fn cancel_all(
        &self,
        req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if req.venue_id == "all" || req.venue_id.is_empty() {
                if self.clients.is_empty() {
                    return Err(LiveGatewayError::fatal("cancel_all: no clients registered"));
                }
                for (venue_id, client) in &self.clients {
                    let mut req = req.clone();
                    req.venue_id = venue_id.clone();
                    if let Err(err) = client.cancel_all(req).await {
                        return Err(err);
                    }
                }
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            let client = self.client_for(&req.venue_id)?;
            client.cancel_all(req).await
        })
    }

    fn cancel_all_with_hint(
        &self,
        req: LiveRestCancelAllRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            if req.venue_id == "all" || req.venue_id.is_empty() {
                if self.clients.is_empty() {
                    return Err(LiveGatewayError::fatal("cancel_all: no clients registered"));
                }
                for (venue_id, client) in &self.clients {
                    let mut req = req.clone();
                    req.venue_id = venue_id.clone();
                    if let Err(err) = client.cancel_all_with_hint(req, hint).await {
                        return Err(err);
                    }
                }
                return Ok(LiveRestResponse {
                    order_id: None,
                    client_order_id: None,
                });
            }
            let client = self.client_for(&req.venue_id)?;
            client.cancel_all_with_hint(req, hint).await
        })
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.replace_order(req).await
        })
    }

    fn replace_order_with_hint(
        &self,
        req: LiveRestReplaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.replace_order_with_hint(req, hint).await
        })
    }

    fn place_batch(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_place_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.place_batch(reqs).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }

    fn place_batch_with_hint(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_place_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.place_batch_with_hint(reqs, hint).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }

    fn cancel_batch(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_cancel_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.cancel_batch(reqs).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }

    fn cancel_batch_with_hint(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_cancel_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.cancel_batch_with_hint(reqs, hint).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }

    fn replace_batch(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_replace_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.replace_batch(reqs).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }

    fn replace_batch_with_hint(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        Box::pin(async move {
            if reqs.is_empty() {
                return Vec::new();
            }
            let venue_id = reqs[0].venue_id.clone();
            if reqs.iter().any(|req| req.venue_id != venue_id) {
                let err = LiveGatewayError::fatal("mixed_venue_replace_batch");
                return vec![Err(err); reqs.len()];
            }
            match self.client_for(&venue_id) {
                Ok(client) => client.replace_batch_with_hint(reqs, hint).await,
                Err(err) => vec![Err(err); reqs.len()],
            }
        })
    }
}

impl LiveRestClient for Arc<dyn LiveRestClient> {
    fn place_order(
        &self,
        req: LiveRestPlaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().place_order(req)
    }

    fn place_order_with_hint(
        &self,
        req: LiveRestPlaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().place_order_with_hint(req, hint)
    }

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_order(req)
    }

    fn cancel_order_with_hint(
        &self,
        req: LiveRestCancelRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_order_with_hint(req, hint)
    }

    fn cancel_all(
        &self,
        req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_all(req)
    }

    fn replace_order(
        &self,
        req: LiveRestReplaceRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().replace_order(req)
    }

    fn cancel_all_with_hint(
        &self,
        req: LiveRestCancelAllRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_all_with_hint(req, hint)
    }

    fn replace_order_with_hint(
        &self,
        req: LiveRestReplaceRequest,
        hint: TransportHint,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().replace_order_with_hint(req, hint)
    }

    fn place_batch(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().place_batch(reqs)
    }

    fn place_batch_with_hint(
        &self,
        reqs: Vec<LiveRestPlaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().place_batch_with_hint(reqs, hint)
    }

    fn cancel_batch(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().cancel_batch(reqs)
    }

    fn cancel_batch_with_hint(
        &self,
        reqs: Vec<LiveRestCancelRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().cancel_batch_with_hint(reqs, hint)
    }

    fn replace_batch(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().replace_batch(reqs)
    }

    fn replace_batch_with_hint(
        &self,
        reqs: Vec<LiveRestReplaceRequest>,
        hint: TransportHint,
    ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
        self.as_ref().replace_batch_with_hint(reqs, hint)
    }
}

type SleepFn = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

pub struct LiveGateway<C> {
    client: C,
    policy: GatewayPolicy,
    rate_limiter: RateLimiter,
    metrics: Option<LiveMetrics>,
    cfg_hash: u64,
    nonce: u64,
    sleep_fn: SleepFn,
    specs: Vec<InstrumentSpec>,
}

impl<C> LiveGateway<C>
where
    C: LiveRestClient,
{
    pub fn new(
        cfg: &Config,
        client: C,
        policy: GatewayPolicy,
        metrics: Option<LiveMetrics>,
        trade_mode: TradeMode,
    ) -> LiveResult<Self> {
        if trade_mode == TradeMode::Shadow {
            return Err(LiveGatewayError::fatal(
                "trade_mode=shadow: live REST gateway disabled",
            ));
        }
        let rate_limiter = RateLimiter::new(policy.rate_limit.clone());
        let cfg_hash = config_hash(cfg);
        let specs = InstrumentSpec::from_config(cfg);
        Ok(Self {
            client,
            policy,
            rate_limiter,
            metrics,
            cfg_hash,
            nonce: 0,
            sleep_fn: Arc::new(|duration| Box::pin(sleep(duration))),
            specs,
        })
    }

    pub fn with_sleep_fn(mut self, sleep_fn: SleepFn) -> Self {
        self.sleep_fn = sleep_fn;
        self
    }

    pub fn next_client_order_id(
        &mut self,
        venue_id: &str,
        tick: u64,
        side: Side,
        purpose: OrderPurpose,
    ) -> String {
        let nonce = self.nonce;
        self.nonce = self.nonce.wrapping_add(1);
        format!(
            "co_{:016x}_{}_{}_{}_{}_{}",
            self.cfg_hash,
            venue_id,
            tick,
            format!("{:?}", side).to_lowercase(),
            format!("{:?}", purpose).to_lowercase(),
            nonce
        )
    }

    pub async fn submit_intent(
        &mut self,
        intent: &OrderIntent,
        tick: u64,
        now_ms: TimestampMs,
        hint: TransportHint,
    ) -> LiveResult<LiveRestResponse> {
        match intent {
            OrderIntent::Place(place) => {
                let spec = self.specs.get(place.venue_index);
                let mut price = place.price;
                let mut size = place.size;
                if let Some(spec) = spec {
                    price = spec.round_price(price);
                    if !should_preserve_terminal_sub_lot_reduce_only_size(place, spec) {
                        size = spec.round_size(size);
                    }
                    if !spec.meets_min_notional(size, price) {
                        let err = LiveGatewayError::fatal("min_notional_usd");
                        self.record_submit_metrics(&Err(err.clone()));
                        return Err(err);
                    }
                }
                let client_order_id = place.client_order_id.clone().unwrap_or_else(|| {
                    self.next_client_order_id(
                        place.venue_id.as_ref(),
                        tick,
                        place.side,
                        place.purpose,
                    )
                });
                let req = LiveRestPlaceRequest {
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    side: place.side,
                    price,
                    size,
                    purpose: place.purpose,
                    time_in_force: place.time_in_force,
                    post_only: place.post_only,
                    reduce_only: place.reduce_only,
                    client_order_id,
                };
                let result = self
                    .execute_with_policy(now_ms, |client| {
                        let req = req.clone();
                        client.place_order_with_hint(req, hint)
                    })
                    .await;
                self.record_submit_metrics(&result);
                result
            }
            OrderIntent::Cancel(cancel) => {
                let req = LiveRestCancelRequest {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.to_string(),
                    order_id: cancel.order_id.clone(),
                };
                let result = self
                    .execute_with_policy(now_ms, |client| {
                        let req = req.clone();
                        client.cancel_order_with_hint(req, hint)
                    })
                    .await;
                self.record_cancel_metrics(&result);
                result
            }
            OrderIntent::CancelAll(cancel_all) => {
                let venue_index = cancel_all.venue_index.unwrap_or(0);
                let venue_id = cancel_all
                    .venue_id
                    .clone()
                    .unwrap_or_else(|| "unknown".into());
                let req = LiveRestCancelAllRequest {
                    venue_index,
                    venue_id: venue_id.to_string(),
                };
                let result = self
                    .execute_with_policy(now_ms, |client| {
                        let req = req.clone();
                        client.cancel_all_with_hint(req, hint)
                    })
                    .await;
                self.record_cancel_metrics(&result);
                result
            }
            OrderIntent::Replace(replace) => {
                let req = match self.build_replace_request(replace, tick) {
                    Ok(req) => req,
                    Err(err) => {
                        self.record_submit_metrics(&Err(err.clone()));
                        return Err(err);
                    }
                };
                let result = self
                    .execute_with_policy(now_ms, |client| {
                        let req = req.clone();
                        client.replace_order_with_hint(req, hint)
                    })
                    .await;
                self.record_submit_metrics(&result);
                result
            }
        }
    }

    pub async fn submit_batch(
        &mut self,
        intents: &[OrderIntent],
        tick: u64,
        now_ms: TimestampMs,
        hint: TransportHint,
    ) -> Vec<LiveResult<LiveRestResponse>> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum BatchKind {
            HyperliquidCancelOid,
            HyperliquidCancelByCloid,
            HyperliquidReplaceAlo,
            HyperliquidPlaceAlo,
            ParadexCancel,
        }

        fn is_hyperliquid_cloid(order_id: &str) -> bool {
            order_id.len() == 34
                && order_id.starts_with("0x")
                && order_id[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
        }

        fn batch_kind(intent: &OrderIntent) -> Option<BatchKind> {
            match intent {
                OrderIntent::Cancel(cancel)
                    if cancel.venue_id.eq_ignore_ascii_case("hyperliquid") =>
                {
                    if is_hyperliquid_cloid(&cancel.order_id) {
                        Some(BatchKind::HyperliquidCancelByCloid)
                    } else {
                        Some(BatchKind::HyperliquidCancelOid)
                    }
                }
                OrderIntent::Cancel(cancel) if cancel.venue_id.eq_ignore_ascii_case("paradex") => {
                    Some(BatchKind::ParadexCancel)
                }
                OrderIntent::Place(place)
                    if place.venue_id.eq_ignore_ascii_case("hyperliquid") && place.post_only =>
                {
                    Some(BatchKind::HyperliquidPlaceAlo)
                }
                OrderIntent::Replace(replace)
                    if replace.venue_id.eq_ignore_ascii_case("hyperliquid")
                        && replace.post_only
                        && !replace.reduce_only =>
                {
                    Some(BatchKind::HyperliquidReplaceAlo)
                }
                _ => None,
            }
        }

        let mut results = Vec::with_capacity(intents.len());
        let mut index = 0usize;
        while index < intents.len() {
            let Some(kind) = batch_kind(&intents[index]) else {
                results.push(
                    self.submit_intent(&intents[index], tick, now_ms, hint)
                        .await,
                );
                index += 1;
                continue;
            };

            let venue_id = match &intents[index] {
                OrderIntent::Place(place) => place.venue_id.as_ref(),
                OrderIntent::Cancel(cancel) => cancel.venue_id.as_ref(),
                OrderIntent::Replace(replace) => replace.venue_id.as_ref(),
                _ => "",
            };

            let mut end = index;
            match kind {
                BatchKind::HyperliquidCancelOid
                | BatchKind::HyperliquidCancelByCloid
                | BatchKind::ParadexCancel => {
                    let mut reqs = Vec::new();
                    while end < intents.len() {
                        let OrderIntent::Cancel(cancel) = &intents[end] else {
                            break;
                        };
                        if !cancel.venue_id.eq_ignore_ascii_case(venue_id)
                            || batch_kind(&intents[end]) != Some(kind)
                        {
                            break;
                        }
                        reqs.push(LiveRestCancelRequest {
                            venue_index: cancel.venue_index,
                            venue_id: cancel.venue_id.to_string(),
                            order_id: cancel.order_id.clone(),
                        });
                        end += 1;
                    }
                    if kind == BatchKind::ParadexCancel && reqs.len() < 2 {
                        results.push(
                            self.submit_intent(&intents[index], tick, now_ms, hint)
                                .await,
                        );
                        index += 1;
                        continue;
                    }
                    let kind_label = match kind {
                        BatchKind::HyperliquidCancelOid => "cancel_oid",
                        BatchKind::HyperliquidCancelByCloid => "cancel_by_cloid",
                        BatchKind::ParadexCancel => "cancel",
                        BatchKind::HyperliquidReplaceAlo => {
                            unreachable!("modify_alo handled separately")
                        }
                        BatchKind::HyperliquidPlaceAlo => {
                            unreachable!("place_alo handled separately")
                        }
                    };
                    eprintln!(
                        "BATCH_SUBMIT venue={} kind={} size={} tick={}",
                        venue_id,
                        kind_label,
                        reqs.len(),
                        tick
                    );
                    results.extend(
                        self.execute_cancel_batch_with_policy(reqs, now_ms, hint)
                            .await,
                    );
                }
                BatchKind::HyperliquidPlaceAlo => {
                    let mut reqs = Vec::new();
                    while end < intents.len() {
                        let OrderIntent::Place(place) = &intents[end] else {
                            break;
                        };
                        if !place.venue_id.eq_ignore_ascii_case(venue_id) || !place.post_only {
                            break;
                        }
                        match self.build_place_request(place, tick) {
                            Ok(req) => {
                                reqs.push(req);
                                end += 1;
                            }
                            Err(err) => {
                                if reqs.is_empty() {
                                    results.push(Err(err));
                                    end += 1;
                                }
                                break;
                            }
                        }
                    }
                    eprintln!(
                        "BATCH_SUBMIT venue={} kind=place_alo size={} tick={}",
                        venue_id,
                        reqs.len(),
                        tick
                    );
                    results.extend(
                        self.execute_place_batch_with_policy(reqs, now_ms, hint)
                            .await,
                    );
                }
                BatchKind::HyperliquidReplaceAlo => {
                    let mut reqs = Vec::new();
                    while end < intents.len() {
                        let OrderIntent::Replace(replace) = &intents[end] else {
                            break;
                        };
                        if !replace.venue_id.eq_ignore_ascii_case(venue_id)
                            || !replace.post_only
                            || replace.reduce_only
                        {
                            break;
                        }
                        match self.build_replace_request(replace, tick) {
                            Ok(req) => {
                                reqs.push(req);
                                end += 1;
                            }
                            Err(err) => {
                                if reqs.is_empty() {
                                    results.push(Err(err));
                                    end += 1;
                                }
                                break;
                            }
                        }
                    }
                    eprintln!(
                        "BATCH_SUBMIT venue={} kind=modify_alo size={} tick={}",
                        venue_id,
                        reqs.len(),
                        tick
                    );
                    results.extend(
                        self.execute_replace_batch_with_policy(reqs, now_ms, hint)
                            .await,
                    );
                }
            }
            index = end;
        }
        results
    }

    fn build_place_request(
        &mut self,
        place: &crate::types::PlaceOrderIntent,
        tick: u64,
    ) -> LiveResult<LiveRestPlaceRequest> {
        let spec = self.specs.get(place.venue_index);
        let mut price = place.price;
        let mut size = place.size;
        if let Some(spec) = spec {
            price = spec.round_price(price);
            if !should_preserve_terminal_sub_lot_reduce_only_size(place, spec) {
                size = spec.round_size(size);
            }
            if !spec.meets_min_notional(size, price) {
                return Err(LiveGatewayError::fatal("min_notional_usd"));
            }
        }
        let client_order_id = place.client_order_id.clone().unwrap_or_else(|| {
            self.next_client_order_id(place.venue_id.as_ref(), tick, place.side, place.purpose)
        });
        Ok(LiveRestPlaceRequest {
            venue_index: place.venue_index,
            venue_id: place.venue_id.to_string(),
            side: place.side,
            price,
            size,
            purpose: place.purpose,
            time_in_force: place.time_in_force,
            post_only: place.post_only,
            reduce_only: place.reduce_only,
            client_order_id,
        })
    }

    fn build_replace_request(
        &mut self,
        replace: &crate::types::ReplaceOrderIntent,
        tick: u64,
    ) -> LiveResult<LiveRestReplaceRequest> {
        let spec = self.specs.get(replace.venue_index);
        let mut price = replace.price;
        let mut size = replace.size;
        if let Some(spec) = spec {
            price = spec.round_price(price);
            size = spec.round_size(size);
            if !spec.meets_min_notional(size, price) {
                return Err(LiveGatewayError::fatal("min_notional_usd"));
            }
        }
        let client_order_id = replace.client_order_id.clone().unwrap_or_else(|| {
            self.next_client_order_id(
                replace.venue_id.as_ref(),
                tick,
                replace.side,
                replace.purpose,
            )
        });
        Ok(LiveRestReplaceRequest {
            venue_index: replace.venue_index,
            venue_id: replace.venue_id.to_string(),
            order_id: replace.order_id.clone(),
            side: replace.side,
            price,
            size,
            purpose: replace.purpose,
            time_in_force: replace.time_in_force,
            post_only: replace.post_only,
            reduce_only: replace.reduce_only,
            client_order_id,
        })
    }

    async fn execute_place_batch_with_policy(
        &mut self,
        reqs: Vec<LiveRestPlaceRequest>,
        mut now_ms: TimestampMs,
        hint: TransportHint,
    ) -> Vec<LiveResult<LiveRestResponse>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        let retry_cfg = self.policy.retry.clone();
        let mut attempt: u32 = 0;
        let mut backoff = retry_cfg.initial_backoff;
        loop {
            now_ms = self.apply_rate_limit(now_ms).await;
            let mut results = self.client.place_batch_with_hint(reqs.clone(), hint).await;
            if results.len() != reqs.len() {
                let err = LiveGatewayError::fatal("place_batch_len_mismatch");
                results = vec![Err(err); reqs.len()];
            }
            let retryable = retry_cfg.enabled
                && !results.is_empty()
                && results.iter().all(|result| match result {
                    Ok(_) => false,
                    Err(err) => err.is_retryable(),
                });
            if retryable && attempt < retry_cfg.max_retries {
                attempt += 1;
                if let Some(metrics) = self.metrics.as_ref() {
                    metrics.inc_retry();
                }
                (self.sleep_fn)(backoff).await;
                backoff = Duration::from_millis(
                    ((backoff.as_millis() as f64) * retry_cfg.backoff_multiplier)
                        .min(retry_cfg.max_backoff.as_millis() as f64) as u64,
                );
                continue;
            }
            for result in &results {
                self.record_submit_metrics(result);
            }
            return results;
        }
    }

    async fn execute_cancel_batch_with_policy(
        &mut self,
        reqs: Vec<LiveRestCancelRequest>,
        now_ms: TimestampMs,
        hint: TransportHint,
    ) -> Vec<LiveResult<LiveRestResponse>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        let retry_cfg = self.policy.retry.clone();
        let mut attempt: u32 = 0;
        let mut backoff = retry_cfg.initial_backoff;
        let mut now_ms = now_ms;
        loop {
            now_ms = self.apply_rate_limit(now_ms).await;
            let mut results = self.client.cancel_batch_with_hint(reqs.clone(), hint).await;
            if results.len() != reqs.len() {
                let err = LiveGatewayError::fatal("cancel_batch_len_mismatch");
                results = vec![Err(err); reqs.len()];
            }
            let retryable = retry_cfg.enabled
                && !results.is_empty()
                && results.iter().all(|result| match result {
                    Ok(_) => false,
                    Err(err) => err.is_retryable(),
                });
            if retryable && attempt < retry_cfg.max_retries {
                attempt += 1;
                if let Some(metrics) = self.metrics.as_ref() {
                    metrics.inc_retry();
                }
                (self.sleep_fn)(backoff).await;
                backoff = Duration::from_millis(
                    ((backoff.as_millis() as f64) * retry_cfg.backoff_multiplier)
                        .min(retry_cfg.max_backoff.as_millis() as f64) as u64,
                );
                continue;
            }
            for result in &results {
                self.record_cancel_metrics(result);
            }
            return results;
        }
    }

    async fn execute_replace_batch_with_policy(
        &mut self,
        reqs: Vec<LiveRestReplaceRequest>,
        now_ms: TimestampMs,
        hint: TransportHint,
    ) -> Vec<LiveResult<LiveRestResponse>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        let retry_cfg = self.policy.retry.clone();
        let mut attempt: u32 = 0;
        let mut backoff = retry_cfg.initial_backoff;
        let mut now_ms = now_ms;
        loop {
            now_ms = self.apply_rate_limit(now_ms).await;
            let mut results = self
                .client
                .replace_batch_with_hint(reqs.clone(), hint)
                .await;
            if results.len() != reqs.len() {
                let err = LiveGatewayError::fatal("replace_batch_len_mismatch");
                results = vec![Err(err); reqs.len()];
            }
            let retryable = retry_cfg.enabled
                && !results.is_empty()
                && results.iter().all(|result| match result {
                    Ok(_) => false,
                    Err(err) => err.is_retryable(),
                });
            if retryable && attempt < retry_cfg.max_retries {
                attempt += 1;
                if let Some(metrics) = self.metrics.as_ref() {
                    metrics.inc_retry();
                }
                (self.sleep_fn)(backoff).await;
                backoff = Duration::from_millis(
                    ((backoff.as_millis() as f64) * retry_cfg.backoff_multiplier)
                        .min(retry_cfg.max_backoff.as_millis() as f64) as u64,
                );
                continue;
            }
            for result in &results {
                self.record_submit_metrics(result);
            }
            return results;
        }
    }

    fn record_submit_metrics(&self, result: &LiveResult<LiveRestResponse>) {
        if let Some(metrics) = self.metrics.as_ref() {
            match result {
                Ok(_) => metrics.inc_order_submit_ok(),
                Err(err) => {
                    metrics.inc_order_submit_fail();
                    metrics.inc_reject_reason(err.reason_label());
                }
            }
        }
    }

    fn record_cancel_metrics(&self, result: &LiveResult<LiveRestResponse>) {
        if let Some(metrics) = self.metrics.as_ref() {
            match result {
                Ok(_) => metrics.inc_cancel_ok(),
                Err(err) => {
                    metrics.inc_cancel_fail();
                    metrics.inc_reject_reason(err.reason_label());
                }
            }
        }
    }

    async fn execute_with_policy<F>(
        &mut self,
        mut now_ms: TimestampMs,
        op: F,
    ) -> LiveResult<LiveRestResponse>
    where
        F: Fn(&C) -> BoxFuture<'_, LiveResult<LiveRestResponse>>,
    {
        let retry_cfg = self.policy.retry.clone();
        let mut attempt: u32 = 0;
        let mut backoff = retry_cfg.initial_backoff;
        loop {
            now_ms = self.apply_rate_limit(now_ms).await;
            let res = op(&self.client).await;
            match res {
                Ok(val) => return Ok(val),
                Err(err) => {
                    let retryable = err.is_retryable() && retry_cfg.enabled;
                    if retryable && attempt < retry_cfg.max_retries {
                        attempt += 1;
                        if let Some(metrics) = self.metrics.as_ref() {
                            metrics.inc_retry();
                        }
                        (self.sleep_fn)(backoff).await;
                        backoff = Duration::from_millis(
                            ((backoff.as_millis() as f64) * retry_cfg.backoff_multiplier)
                                .min(retry_cfg.max_backoff.as_millis() as f64)
                                as u64,
                        );
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    async fn apply_rate_limit(&mut self, now_ms: TimestampMs) -> TimestampMs {
        if !self.policy.rate_limit.enabled {
            return now_ms;
        }
        if self.rate_limiter.try_acquire(now_ms) {
            return now_ms;
        }
        let rps = self.policy.rate_limit.max_requests_per_second.max(0.1);
        let sleep_ms = (1000.0 / rps).ceil() as u64;
        if let Some(metrics) = self.metrics.as_ref() {
            metrics.add_rate_limit_sleep_ms(sleep_ms);
        }
        (self.sleep_fn)(Duration::from_millis(sleep_ms)).await;
        now_ms.saturating_add(sleep_ms as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::types::CancelOrderIntent;

    #[derive(Clone, Default)]
    struct RecordingClient {
        place_reqs: Arc<Mutex<Vec<LiveRestPlaceRequest>>>,
        single_cancel_ids: Arc<Mutex<Vec<String>>>,
        batch_cancel_ids: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl RecordingClient {
        fn place_reqs(&self) -> Vec<LiveRestPlaceRequest> {
            self.place_reqs.lock().unwrap().clone()
        }

        fn single_cancel_ids(&self) -> Vec<String> {
            self.single_cancel_ids.lock().unwrap().clone()
        }

        fn batch_cancel_ids(&self) -> Vec<Vec<String>> {
            self.batch_cancel_ids.lock().unwrap().clone()
        }
    }

    impl LiveRestClient for RecordingClient {
        fn place_order(
            &self,
            req: LiveRestPlaceRequest,
        ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
            let place_reqs = Arc::clone(&self.place_reqs);
            Box::pin(async move {
                place_reqs.lock().unwrap().push(req.clone());
                Ok(LiveRestResponse {
                    order_id: Some(req.client_order_id),
                    client_order_id: None,
                })
            })
        }

        fn cancel_order(
            &self,
            req: LiveRestCancelRequest,
        ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
            let single_cancel_ids = Arc::clone(&self.single_cancel_ids);
            Box::pin(async move {
                single_cancel_ids.lock().unwrap().push(req.order_id.clone());
                Ok(LiveRestResponse {
                    order_id: Some(req.order_id),
                    client_order_id: None,
                })
            })
        }

        fn cancel_all(
            &self,
            _req: LiveRestCancelAllRequest,
        ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
            Box::pin(async { Err(LiveGatewayError::fatal("unexpected_cancel_all")) })
        }

        fn replace_order(
            &self,
            _req: LiveRestReplaceRequest,
        ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
            Box::pin(async { Err(LiveGatewayError::fatal("unexpected_replace")) })
        }

        fn cancel_batch(
            &self,
            reqs: Vec<LiveRestCancelRequest>,
        ) -> BoxFuture<'_, Vec<LiveResult<LiveRestResponse>>> {
            let batch_cancel_ids = Arc::clone(&self.batch_cancel_ids);
            Box::pin(async move {
                let order_ids: Vec<String> = reqs.iter().map(|req| req.order_id.clone()).collect();
                batch_cancel_ids.lock().unwrap().push(order_ids);
                reqs.into_iter()
                    .map(|req| {
                        Ok(LiveRestResponse {
                            order_id: Some(req.order_id),
                            client_order_id: None,
                        })
                    })
                    .collect()
            })
        }
    }

    fn paradex_cancel(order_id: &str) -> OrderIntent {
        OrderIntent::Cancel(CancelOrderIntent {
            venue_index: 4,
            venue_id: Arc::<str>::from("paradex"),
            order_id: order_id.to_string(),
        })
    }

    fn extended_cancel(order_id: &str) -> OrderIntent {
        OrderIntent::Cancel(CancelOrderIntent {
            venue_index: 0,
            venue_id: Arc::<str>::from("extended"),
            order_id: order_id.to_string(),
        })
    }

    #[tokio::test]
    async fn submit_preserves_env_gated_aster_sub_lot_reduce_only_ioc_size() {
        std::env::set_var("PARAPHINA_ASTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED", "1");
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index: 2,
            venue_id: Arc::<str>::from("aster"),
            side: Side::Buy,
            price: 2_350.123,
            size: 0.008,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        })];

        let results = gateway
            .submit_batch(&intents, 7, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let places = inspector.place_reqs();
        assert_eq!(places.len(), 1);
        assert!((places[0].size - 0.008).abs() < 1e-12);
        assert_eq!(places[0].time_in_force, TimeInForce::Ioc);
        assert!(places[0].reduce_only);
        std::env::remove_var("PARAPHINA_ASTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED");
    }

    #[tokio::test]
    async fn submit_preserves_env_gated_hyperliquid_sub_lot_reduce_only_ioc_size() {
        std::env::set_var(
            "PARAPHINA_HYPERLIQUID_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED",
            "1",
        );
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index: 1,
            venue_id: Arc::<str>::from("hyperliquid"),
            side: Side::Buy,
            price: 2_350.123,
            size: 0.0044,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        })];

        let results = gateway
            .submit_batch(&intents, 7, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let places = inspector.place_reqs();
        assert_eq!(places.len(), 1);
        assert!((places[0].size - 0.0044).abs() < 1e-12);
        assert_eq!(places[0].time_in_force, TimeInForce::Ioc);
        assert!(places[0].reduce_only);
        std::env::remove_var("PARAPHINA_HYPERLIQUID_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED");
    }

    #[tokio::test]
    async fn submit_preserves_env_gated_extended_sub_lot_reduce_only_ioc_size() {
        std::env::set_var(
            "PARAPHINA_EXTENDED_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED",
            "1",
        );
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index: 0,
            venue_id: Arc::<str>::from("extended"),
            side: Side::Sell,
            price: 2_350.123,
            size: 0.006,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        })];

        let results = gateway
            .submit_batch(&intents, 7, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let places = inspector.place_reqs();
        assert_eq!(places.len(), 1);
        assert!((places[0].size - 0.006).abs() < 1e-12);
        assert_eq!(places[0].time_in_force, TimeInForce::Ioc);
        assert!(places[0].reduce_only);
        std::env::remove_var("PARAPHINA_EXTENDED_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED");
    }

    #[tokio::test]
    async fn submit_preserves_env_gated_lighter_sub_lot_reduce_only_ioc_size() {
        std::env::set_var(
            "PARAPHINA_LIGHTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED",
            "1",
        );
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![OrderIntent::Place(crate::types::PlaceOrderIntent {
            venue_index: 3,
            venue_id: Arc::<str>::from("lighter"),
            side: Side::Buy,
            price: 2_350.123,
            size: 0.006,
            purpose: OrderPurpose::Hedge,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: None,
            phase51_target_key: None,
        })];

        let results = gateway
            .submit_batch(&intents, 7, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let places = inspector.place_reqs();
        assert_eq!(places.len(), 1);
        assert!((places[0].size - 0.006).abs() < 1e-12);
        assert_eq!(places[0].time_in_force, TimeInForce::Ioc);
        assert!(places[0].reduce_only);
        std::env::remove_var("PARAPHINA_LIGHTER_TERMINAL_SUB_LOT_REDUCE_ONLY_ENABLED");
    }

    #[tokio::test]
    async fn submit_batch_groups_contiguous_paradex_cancels() {
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![paradex_cancel("co_1"), paradex_cancel("pdx_2")];

        let results = gateway
            .submit_batch(&intents, 42, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.is_ok()));
        assert_eq!(
            inspector.batch_cancel_ids(),
            vec![vec!["co_1".to_string(), "pdx_2".to_string()]]
        );
        assert!(inspector.single_cancel_ids().is_empty());
    }

    #[tokio::test]
    async fn submit_batch_keeps_single_paradex_cancel_on_serial_path_when_not_contiguous() {
        let cfg = Config::default();
        let client = RecordingClient::default();
        let inspector = client.clone();
        let mut gateway = LiveGateway::new(
            &cfg,
            client,
            GatewayPolicy::for_simulation(),
            None,
            TradeMode::Live,
        )
        .expect("gateway");
        let intents = vec![
            paradex_cancel("co_1"),
            extended_cancel("ext_2"),
            paradex_cancel("pdx_3"),
        ];

        let results = gateway
            .submit_batch(&intents, 99, 1_000, TransportHint::Default)
            .await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|result| result.is_ok()));
        assert!(inspector.batch_cancel_ids().is_empty());
        assert_eq!(
            inspector.single_cancel_ids(),
            vec!["co_1".to_string(), "ext_2".to_string(), "pdx_3".to_string()]
        );
    }
}
