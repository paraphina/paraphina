use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::Mutex;

use super::types::MarketDataEvent;

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
            self.market_pub_tx
                .send(event)
                .await
                .map_err(|_| anyhow::anyhow!("{}", self.err_queue_closed))?;
            return Ok(());
        }
        match self.market_pub_tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(event)) => {
                let mut pending = self.pending_latest.lock().await;
                *pending = Some(event);
                Ok(())
            }
            Err(TrySendError::Closed(_)) => {
                anyhow::bail!("{}", self.err_queue_closed)
            }
        }
    }
}
