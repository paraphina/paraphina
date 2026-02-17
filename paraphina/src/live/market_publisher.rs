use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::Mutex;

use super::types::MarketDataEvent;

const MARKET_PUBLISHER_AUDIT_INTERVAL_MS: u64 = 1_000;
static MARKET_PUBLISHER_WS_AUDIT_ENABLED: OnceLock<bool> = OnceLock::new();

fn market_publisher_ws_audit_enabled() -> bool {
    *MARKET_PUBLISHER_WS_AUDIT_ENABLED.get_or_init(|| {
        std::env::var("PARAPHINA_WS_AUDIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn update_max(cell: &AtomicU64, candidate: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    while candidate > current {
        match cell.compare_exchange_weak(
            current,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

struct MarketPublisherAudit {
    queue_cap: usize,
    drain_max: usize,
    venue: &'static str,
    last_emit_ms: AtomicU64,
    queued_hiwater: AtomicU64,
    pending_latest_present: AtomicU64,
    pending_overwrite: AtomicU64,
    lossless_send_count: AtomicU64,
    lossless_send_wait_ms_max: AtomicU64,
    try_send_ok: AtomicU64,
    try_send_full: AtomicU64,
    out_send_ok: AtomicU64,
    out_send_err: AtomicU64,
}

impl MarketPublisherAudit {
    fn new(queue_cap: usize, drain_max: usize, venue: &'static str) -> Self {
        Self {
            queue_cap,
            drain_max,
            venue,
            last_emit_ms: AtomicU64::new(now_ms()),
            queued_hiwater: AtomicU64::new(0),
            pending_latest_present: AtomicU64::new(0),
            pending_overwrite: AtomicU64::new(0),
            lossless_send_count: AtomicU64::new(0),
            lossless_send_wait_ms_max: AtomicU64::new(0),
            try_send_ok: AtomicU64::new(0),
            try_send_full: AtomicU64::new(0),
            out_send_ok: AtomicU64::new(0),
            out_send_err: AtomicU64::new(0),
        }
    }

    fn queue_len(&self, sender: &mpsc::Sender<MarketDataEvent>) -> u64 {
        self.queue_cap.saturating_sub(sender.capacity()) as u64
    }

    fn observe_queue_len(&self, queued_len: u64) {
        update_max(&self.queued_hiwater, queued_len);
    }

    fn set_pending_latest_present(&self, present: bool) {
        self.pending_latest_present
            .store(u64::from(present), Ordering::Relaxed);
    }

    fn record_pending_overwrite(&self) {
        self.pending_overwrite.fetch_add(1, Ordering::Relaxed);
    }

    fn record_lossless_send(&self, wait_ms: u64) {
        self.lossless_send_count.fetch_add(1, Ordering::Relaxed);
        update_max(&self.lossless_send_wait_ms_max, wait_ms);
    }

    fn record_try_send_ok(&self) {
        self.try_send_ok.fetch_add(1, Ordering::Relaxed);
    }

    fn record_try_send_full(&self) {
        self.try_send_full.fetch_add(1, Ordering::Relaxed);
    }

    fn record_out_send_ok(&self) {
        self.out_send_ok.fetch_add(1, Ordering::Relaxed);
    }

    fn record_out_send_err(&self) {
        self.out_send_err.fetch_add(1, Ordering::Relaxed);
    }

    fn maybe_emit(&self, sender: &mpsc::Sender<MarketDataEvent>) {
        let queued_len = self.queue_len(sender);
        self.observe_queue_len(queued_len);

        let now = now_ms();
        loop {
            let last_emit_ms = self.last_emit_ms.load(Ordering::Relaxed);
            let emit_since_ms = now.saturating_sub(last_emit_ms);
            if emit_since_ms < MARKET_PUBLISHER_AUDIT_INTERVAL_MS {
                return;
            }
            if self
                .last_emit_ms
                .compare_exchange_weak(
                    last_emit_ms,
                    now,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                let queued_hiwater = self
                    .queued_hiwater
                    .swap(queued_len, Ordering::Relaxed)
                    .max(queued_len);
                let pending_latest_present = self.pending_latest_present.load(Ordering::Relaxed);
                let pending_overwrite = self.pending_overwrite.swap(0, Ordering::Relaxed);
                let lossless_send_count = self.lossless_send_count.swap(0, Ordering::Relaxed);
                let lossless_send_wait_ms_max =
                    self.lossless_send_wait_ms_max.swap(0, Ordering::Relaxed);
                let try_send_ok = self.try_send_ok.swap(0, Ordering::Relaxed);
                let try_send_full = self.try_send_full.swap(0, Ordering::Relaxed);
                let out_send_ok = self.out_send_ok.swap(0, Ordering::Relaxed);
                let out_send_err = self.out_send_err.swap(0, Ordering::Relaxed);

                eprintln!(
                    "WS_AUDIT component=market_publisher reason=periodic interval_ms=1000 \
queue_cap={} drain_max={} queued_len={} queued_hiwater={} pending_latest_present={} \
pending_overwrite={} mp_pending_latest_replaced_count={} lossless_send_count={} \
lossless_send_wait_ms_max={} try_send_ok={} try_send_full={} mp_try_send_full_count={} \
out_send_ok={} out_send_err={} emit_since_ms={} venue={}",
                    self.queue_cap,
                    self.drain_max,
                    queued_len,
                    queued_hiwater,
                    pending_latest_present,
                    pending_overwrite,
                    pending_overwrite,
                    lossless_send_count,
                    lossless_send_wait_ms_max,
                    try_send_ok,
                    try_send_full,
                    try_send_full,
                    out_send_ok,
                    out_send_err,
                    emit_since_ms,
                    self.venue,
                );
                return;
            }
        }
    }
}

pub(crate) struct MarketPublisher {
    market_pub_tx: mpsc::Sender<MarketDataEvent>,
    pending_latest: Arc<Mutex<Option<MarketDataEvent>>>,
    out_tx: mpsc::Sender<MarketDataEvent>,
    fixture_mode_now: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    is_lossless: Arc<dyn Fn(&MarketDataEvent) -> bool + Send + Sync>,
    on_published: Option<Arc<dyn Fn() + Send + Sync>>,
    audit: Option<Arc<MarketPublisherAudit>>,
    err_out_tx_closed: &'static str,
    err_queue_closed: &'static str,
}

impl std::fmt::Debug for MarketPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MarketPublisher")
            .field("has_fixture_mode_now", &self.fixture_mode_now.is_some())
            .field("has_on_published", &self.on_published.is_some())
            .field("has_ws_audit", &self.audit.is_some())
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
        let audit = market_publisher_ws_audit_enabled()
            .then(|| Arc::new(MarketPublisherAudit::new(queue_cap, drain_max, "all")));
        let forward_out_tx = out_tx.clone();
        let forward_pending = pending_latest.clone();
        let forward_on_published = on_published.clone();
        let forward_audit = audit.clone();
        let forward_queue_tx = market_pub_tx.clone();
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
                    let send_result = forward_out_tx.send(ev).await;
                    if let Some(audit) = &forward_audit {
                        if send_result.is_ok() {
                            audit.record_out_send_ok();
                        } else {
                            audit.record_out_send_err();
                        }
                        audit.maybe_emit(&forward_queue_tx);
                    }
                    if send_result.is_err() {
                        return;
                    }
                    if let Some(cb) = &forward_on_published {
                        cb();
                    }
                }
                let overflow = {
                    let mut guard = forward_pending.lock().await;
                    let overflow = guard.take();
                    if overflow.is_some() {
                        if let Some(audit) = &forward_audit {
                            audit.set_pending_latest_present(false);
                        }
                    }
                    overflow
                };
                if let Some(ev) = overflow {
                    let send_result = forward_out_tx.send(ev).await;
                    if let Some(audit) = &forward_audit {
                        if send_result.is_ok() {
                            audit.record_out_send_ok();
                        } else {
                            audit.record_out_send_err();
                        }
                        audit.maybe_emit(&forward_queue_tx);
                    }
                    if send_result.is_err() {
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
            audit,
            err_out_tx_closed,
            err_queue_closed,
        }
    }

    pub(crate) async fn publish_market(&self, event: MarketDataEvent) -> anyhow::Result<()> {
        if self.fixture_mode_now.as_ref().is_some_and(|f| f()) {
            let send_result = self.out_tx.send(event).await;
            if let Some(audit) = &self.audit {
                if send_result.is_ok() {
                    audit.record_out_send_ok();
                } else {
                    audit.record_out_send_err();
                }
                audit.maybe_emit(&self.market_pub_tx);
            }
            send_result.map_err(|_| anyhow::anyhow!("{}", self.err_out_tx_closed))?;
            if let Some(cb) = &self.on_published {
                cb();
            }
            return Ok(());
        }
        if (self.is_lossless)(&event) {
            let started = self.audit.as_ref().map(|_| Instant::now());
            self.market_pub_tx
                .send(event)
                .await
                .map_err(|_| anyhow::anyhow!("{}", self.err_queue_closed))?;
            if let (Some(audit), Some(started)) = (&self.audit, started) {
                audit.record_lossless_send(started.elapsed().as_millis() as u64);
                audit.maybe_emit(&self.market_pub_tx);
            }
            return Ok(());
        }
        match self.market_pub_tx.try_send(event) {
            Ok(()) => {
                if let Some(audit) = &self.audit {
                    audit.record_try_send_ok();
                    audit.maybe_emit(&self.market_pub_tx);
                }
                Ok(())
            }
            Err(TrySendError::Full(event)) => {
                if let Some(audit) = &self.audit {
                    audit.record_try_send_full();
                }
                let mut pending = self.pending_latest.lock().await;
                if pending.is_some() {
                    if let Some(audit) = &self.audit {
                        audit.record_pending_overwrite();
                    }
                }
                *pending = Some(event);
                if let Some(audit) = &self.audit {
                    audit.set_pending_latest_present(true);
                    audit.maybe_emit(&self.market_pub_tx);
                }
                Ok(())
            }
            Err(TrySendError::Closed(_)) => {
                anyhow::bail!("{}", self.err_queue_closed)
            }
        }
    }
}
