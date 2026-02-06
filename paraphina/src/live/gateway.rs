//! Live REST execution gateway (feature-gated).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;

use crate::actions::ActionBatch;
use crate::config::Config;
use crate::io::{GatewayPolicy, RateLimiter};
use crate::live::instrument::InstrumentSpec;
use crate::live::ops::{config_hash, LiveMetrics};
use crate::live::trade_mode::TradeMode;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

pub type LiveResult<T> = Result<T, LiveGatewayError>;
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

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

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        Box::pin(async move {
            let client = self.client_for(&req.venue_id)?;
            client.cancel_order(req).await
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
                return Ok(LiveRestResponse { order_id: None });
            }
            let client = self.client_for(&req.venue_id)?;
            client.cancel_all(req).await
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

    fn cancel_order(
        &self,
        req: LiveRestCancelRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_order(req)
    }

    fn cancel_all(
        &self,
        req: LiveRestCancelAllRequest,
    ) -> BoxFuture<'_, LiveResult<LiveRestResponse>> {
        self.as_ref().cancel_all(req)
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
    ) -> LiveResult<LiveRestResponse> {
        match intent {
            OrderIntent::Place(place) => {
                let spec = self.specs.get(place.venue_index);
                let mut price = place.price;
                let mut size = place.size;
                if let Some(spec) = spec {
                    price = spec.round_price(price);
                    size = spec.round_size(size);
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
                        client.place_order(req)
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
                        client.cancel_order(req)
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
                        client.cancel_all(req)
                    })
                    .await;
                self.record_cancel_metrics(&result);
                result
            }
            OrderIntent::Replace(_) => LiveResult::Err(LiveGatewayError::fatal(
                "replace intents must be expanded before LiveGateway",
            )),
        }
    }

    pub async fn submit_batch(
        &mut self,
        batch: &ActionBatch,
        intents: &[OrderIntent],
        now_ms: TimestampMs,
    ) -> Vec<LiveResult<LiveRestResponse>> {
        let mut results = Vec::with_capacity(intents.len());
        for intent in intents {
            results.push(self.submit_intent(intent, batch.tick_index, now_ms).await);
        }
        results
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
