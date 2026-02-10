//! Layer A — Application-level venue health enforcer.
//!
//! Monitors [`SharedVenueAges`] and force-restarts connector tasks that
//! have been stale beyond a configurable threshold by aborting their
//! `JoinHandle` and re-spawning via the stored closure.
//!
//! This is the ultimate safety net: even if a connector hangs in connect,
//! read, or any internal operation, it gets killed and restarted.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::task::JoinHandle;

use super::shared_venue_ages::SharedVenueAges;

/// One entry per connector task group (identified by venue name).
pub struct ConnectorSlot {
    /// Human-readable label, e.g. `"hyperliquid_public_ws"`.
    pub name: String,
    /// The venue index this slot monitors.
    pub venue_index: usize,
    /// Current JoinHandle for the supervised task.
    pub handle: JoinHandle<()>,
    /// Factory that creates a fresh supervised task, returning its JoinHandle.
    pub respawn: Box<dyn Fn() -> JoinHandle<()> + Send + Sync>,
    /// Monotonic instant of last abort (for cooldown).
    pub last_abort: Option<std::time::Instant>,
}

/// Configuration for the enforcer.
pub struct EnforcerConfig {
    /// Age threshold (ms) beyond which a venue's connectors are force-restarted.
    pub force_restart_ms: i64,
    /// Minimum time between consecutive aborts for the same slot.
    pub cooldown: Duration,
    /// How often the enforcer checks ages.
    pub poll_interval: Duration,
}

impl Default for EnforcerConfig {
    fn default() -> Self {
        Self {
            force_restart_ms: std::env::var("PARAPHINA_FORCE_RESTART_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(90_000),
            cooldown: Duration::from_secs(
                std::env::var("PARAPHINA_ENFORCER_COOLDOWN_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
            ),
            poll_interval: Duration::from_secs(5),
        }
    }
}

fn wall_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Run the enforcer loop.  Never returns (designed to be spawned supervised).
pub async fn run_venue_health_enforcer(
    ages: SharedVenueAges,
    mut slots: Vec<ConnectorSlot>,
    ecfg: EnforcerConfig,
) {
    let mut interval = tokio::time::interval(ecfg.poll_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Detect stuck runner: if the runner hasn't written for >30 s, log an error.
    let runner_idle_threshold_ms: i64 = 30_000;

    loop {
        interval.tick().await;
        let now_ms = wall_ms();

        // Check runner heartbeat.
        let idle = ages.runner_idle_ms(now_ms);
        if idle > runner_idle_threshold_ms {
            eprintln!(
                "ERROR: VenueHealthEnforcer: runner has not updated ages for {idle}ms \
                 (threshold={runner_idle_threshold_ms}ms) — runner may be stuck"
            );
        }

        for slot in slots.iter_mut() {
            let age = ages.age_ms(slot.venue_index);
            // i64::MAX means "unknown/uninitialized age"; do not restart until first real update.
            if age == i64::MAX {
                continue;
            }
            if age < ecfg.force_restart_ms {
                continue;
            }

            // Cooldown: don't re-abort too quickly.
            if let Some(last) = slot.last_abort {
                if last.elapsed() < ecfg.cooldown {
                    continue;
                }
            }

            eprintln!(
                "WARN: VenueHealthEnforcer force-restarting '{}' \
                 (venue_index={}, age_ms={}, threshold={})",
                slot.name, slot.venue_index, age, ecfg.force_restart_ms
            );

            // Abort the existing supervision wrapper (kills the entire task tree).
            slot.handle.abort();

            // Re-spawn via the stored factory closure.
            slot.handle = (slot.respawn)();
            slot.last_abort = Some(std::time::Instant::now());
        }
    }
}
