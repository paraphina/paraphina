use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::Mutex;

use super::types::MarketDataEvent;

static MARKET_PUBLISHER_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();
static MP_TRY_SEND_FULL_COUNT: AtomicU64 = AtomicU64::new(0);
static MP_PENDING_LATEST_REPLACED_COUNT: AtomicU64 = AtomicU64::new(0);
static MP_LOSSLESS_WAIT_COUNT: AtomicU64 = AtomicU64::new(0);

fn market_publisher_ws_audit_enabled() -> bool {
    *MARKET_PUBLISHER_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn market_publisher_audit_counter(name: &str, count: u64) {
    if !market_publisher_ws_audit_enabled() {
        return;
    }
    if count <= 3 || count % 1000 == 0 {
        eprintln!("WS_AUDIT component=market_publisher {}={}", name, count);
    }
}

pub(crate) struct MarketPublisher {
    market_pub_tx: mpsc::Sender<MarketDataEvent>,
    pending_latest: Arc<Mutex<Option<MarketDataEvent>>>,
    out_tx: mpsc::Sender<MarketDataEvent>,
    fixture_mode_now: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    is_lossless: Arc<dyn Fn(&MarketDataEvent) -> bool + Send + Sync>,
    on_published: Option<Arc<dyn Fn() + Send + Sync>>,
    err_out_tx_closed: &'static str,
    err_queue_closed: &'static str,
}

impl std::fmt::Debug for MarketPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarketPublisher")
            .field("has_fixture_mode_now", &self.fixture_mode_now.is_some())
            .field("has_on_published", &self.on_published.is_some())
            .finish()
    }
}

impl MarketPublisher {
    pub(crate) fn new(
        queue_cap: usize,
        drain_max: usize,
        out_tx: mpsc::Sender<MarketDataEvent>,
        fixture_mode_now: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
        is_lossless: Arc<dyn Fn(&MarketDataEvent) -> bool + Send + Sync>,
        on_published: Option<Arc<dyn Fn() + Send + Sync>>,
        err_out_tx_closed: &'static str,
        err_queue_closed: &'static str,
    ) -> Self {
        let (market_pub_tx, mut market_pub_rx) = mpsc::channel::<MarketDataEvent>(queue_cap);
        let pending_latest = Arc::new(Mutex::new(None));
        let forward_out_tx = out_tx.clone();
        let forward_pending = pending_latest.clone();
        let forward_on_published = on_published.clone();
        tokio::spawn(async move {
            while let Some(first) = market_pub_rx.recv().await {
                let mut batch = Vec::with_capacity(1 + drain_max);
                batch.push(first);
                for _ in 0..drain_max {
                    match market_pub_rx.try_recv() {
                        Ok(ev) => batch.push(ev),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
                for ev in batch {
                    if forward_out_tx.send(ev).await.is_err() {
                        return;
                    }
                    if let Some(cb) = &forward_on_published {
                        cb();
                    }
                }
                let overflow = {
                    let mut guard = forward_pending.lock().await;
                    guard.take()
                };
                if let Some(ev) = overflow {
                    if forward_out_tx.send(ev).await.is_err() {
                        return;
                    }
                    if let Some(cb) = &forward_on_published {
                        cb();
                    }
                }
            }
        });
        Self {
            market_pub_tx,
            pending_latest,
            out_tx,
            fixture_mode_now,
            is_lossless,
            on_published,
            err_out_tx_closed,
            err_queue_closed,
        }
    }

    pub(crate) async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        if self.fixture_mode_now.as_ref().is_some_and(|f| f()) {
            self.out_tx
                .send(event)
                .await
                .map_err(|_| anyhow::anyhow!("{}", self.err_out_tx_closed))?;
            if let Some(cb) = &self.on_published {
                cb();
            }
            return Ok(());
        }
        if (self.is_lossless)(&event) {
            if self.market_pub_tx.capacity() == 0 {
                let count = MP_LOSSLESS_WAIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                market_publisher_audit_counter("mp_lossless_wait_count", count);
            }
            self.market_pub_tx
                .send(event)
                .await
                .map_err(|_| anyhow::anyhow!("{}", self.err_queue_closed))?;
            return Ok(());
        }
        match self.market_pub_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                let full_count = MP_TRY_SEND_FULL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                market_publisher_audit_counter("mp_try_send_full_count", full_count);
                let mut pending = self.pending_latest.lock().await;
                if pending.is_some() {
                    let replaced_count =
                        MP_PENDING_LATEST_REPLACED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                    market_publisher_audit_counter(
                        "mp_pending_latest_replaced_count",
                        replaced_count,
                    );
                }
                *pending = Some(event);
                Ok(())
            }
            Err(TrySendError::Closed(_)) => {
                anyhow::bail!("{}", self.err_queue_closed)
            }
        }
    }
}
