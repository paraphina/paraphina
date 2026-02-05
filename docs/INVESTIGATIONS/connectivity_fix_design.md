# Connectivity Infrastructure Fix Design

**Date**: 2026-02-05  
**Branch**: `research/funding-repo-audit`  
**Status**: Design Complete (Implementation Pending)

---

## Executive Summary

This document defines optimal fixes for each identified connectivity issue. Fixes are designed to:
1. Be minimal and non-breaking
2. Follow existing codebase patterns
3. Be testable in isolation
4. Be rollout-safe (can be feature-gated or env-controlled)

---

## Fix 1: Backoff Reset After Stable Connection

### Problem
Exponential backoff accumulates forever, causing 30s reconnect delays even after hours of stable operation.

### Design Principle
Reset backoff when the connection has been **healthy** for a defined period. "Healthy" means: the `public_ws_once()` function ran for a significant duration without error.

### Optimal Implementation

**Option A: Duration-Based Reset (Recommended)**

Reset backoff if `public_ws_once()` ran for more than `HEALTHY_CONNECTION_THRESHOLD_MS` (default: 60 seconds).

```rust
const HEALTHY_CONNECTION_THRESHOLD_MS: u64 = 60_000;

pub async fn run_public_ws(&self) {
    let mut backoff = Duration::from_secs(1);
    loop {
        let session_start = Instant::now();
        
        if let Err(err) = self.public_ws_once().await {
            eprintln!("... public WS error: {err}");
        }
        
        let session_duration = session_start.elapsed();
        
        // Reset backoff if connection was healthy for long enough
        let threshold = Duration::from_millis(
            std::env::var("PARAPHINA_WS_HEALTHY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(HEALTHY_CONNECTION_THRESHOLD_MS)
        );
        
        if session_duration >= threshold {
            backoff = Duration::from_secs(1);  // Reset to initial
        }
        
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}
```

**Why This Approach:**
- Simple: single duration check
- Observable: can log when backoff resets
- Configurable: env var allows tuning without code changes
- Conservative: 60s threshold ensures we don't reset on quick reconnects

**Option B: Data-Based Reset (Alternative)**

Reset backoff after N successful data events are published. More complex, requires threading freshness through.

**Recommendation: Option A** — simpler, meets requirements, consistent across all venues.

### Implementation Scope

| File | Changes |
|------|---------|
| `hyperliquid.rs` | Modify `run_public_ws()`, `run_private_ws()` |
| `paradex.rs` | Modify `run_public_ws()` |
| `lighter.rs` | Modify `run_public_ws()`, `run_private_ws()` |
| `extended.rs` | Modify `run_public_ws()` |
| `aster.rs` | Modify `run_public_ws()` |

### Testing Strategy

1. Unit test: verify backoff resets when elapsed >= threshold
2. Unit test: verify backoff increases when elapsed < threshold
3. Integration test: run WS for 70s, disconnect, verify next reconnect uses 1s backoff

---

## Fix 2: Add Freshness/Watchdog to Lighter

### Problem
Lighter has no mechanism to detect a stalled WebSocket connection. It will wait forever on `read.next()`.

### Design Principle
Port the exact same `Freshness` struct and watchdog pattern used by Hyperliquid, Paradex, Extended, and Aster.

### Optimal Implementation

**Step 1: Add Freshness struct (copy from other connectors)**

```rust
// At module level in lighter.rs
const LIGHTER_STALE_MS_DEFAULT: u64 = 10_000;
const LIGHTER_WATCHDOG_TICK_MS: u64 = 200;

static MONO_START: OnceLock<Instant> = OnceLock::new();

fn mono_now_ns() -> u64 {
    let start = MONO_START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn lighter_stale_ms() -> u64 {
    std::env::var("PARAPHINA_LIGHTER_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(LIGHTER_STALE_MS_DEFAULT)
}

fn age_ms(now_ns: u64, then_ns: u64) -> u64 {
    now_ns.saturating_sub(then_ns) / 1_000_000
}

#[derive(Debug, Default)]
struct Freshness {
    last_ws_rx_ns: AtomicU64,
    last_data_rx_ns: AtomicU64,
    last_parsed_ns: AtomicU64,
    last_published_ns: AtomicU64,
}

impl Freshness {
    fn reset_for_new_connection(&self) {
        self.last_ws_rx_ns.store(0, Ordering::Relaxed);
        self.last_data_rx_ns.store(0, Ordering::Relaxed);
        self.last_parsed_ns.store(0, Ordering::Relaxed);
        self.last_published_ns.store(0, Ordering::Relaxed);
    }

    fn anchor_with_connect_start(&self, connect_start_ns: u64) -> u64 {
        let last_pub = self.last_published_ns.load(Ordering::Relaxed);
        let last_parsed = self.last_parsed_ns.load(Ordering::Relaxed);
        let anchor = last_pub.max(last_parsed);
        if anchor == 0 { connect_start_ns } else { anchor }
    }
}
```

**Step 2: Add freshness field to LighterConnector**

```rust
pub struct LighterConnector {
    // existing fields...
    freshness: Arc<Freshness>,  // ADD THIS
}

impl LighterConnector {
    pub fn new(...) -> Self {
        Self {
            // existing fields...
            freshness: Arc::new(Freshness::default()),
        }
    }
}
```

**Step 3: Add watchdog to public_ws_once()**

```rust
async fn public_ws_once(&self) -> anyhow::Result<()> {
    let (market_symbol, market_id) = self.resolve_market_id_and_symbol().await?;
    
    // Setup watchdog
    let connect_start_ns = mono_now_ns();
    self.freshness.reset_for_new_connection();
    let (stale_tx, mut stale_rx) = tokio::sync::oneshot::channel::<()>();
    let stale_ms = lighter_stale_ms();
    
    let fixture_mode = std::env::var_os("LIGHTER_FIXTURE_DIR").is_some()
        || std::env::var_os("ROADMAP_B_FIXTURE_DIR").is_some();
    
    if !fixture_mode {
        let watchdog_freshness = self.freshness.clone();
        tokio::spawn(async move {
            let mut iv = tokio::time::interval(Duration::from_millis(LIGHTER_WATCHDOG_TICK_MS));
            iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                iv.tick().await;
                let now = mono_now_ns();
                let anchor = watchdog_freshness.anchor_with_connect_start(connect_start_ns);
                if anchor != 0 && age_ms(now, anchor) > stale_ms {
                    let _ = stale_tx.send(());
                    break;
                }
            }
        });
    }
    
    // Connect WebSocket
    let (ws_stream, _) = connect_async(self.cfg.ws_url.as_str()).await?;
    let (mut write, mut read) = ws_stream.split();
    // ... subscription code ...
    
    // Main loop with stale check
    loop {
        let msg = tokio::select! {
            biased;
            _ = &mut stale_rx => {
                anyhow::bail!("Lighter public WS stale: freshness exceeded {stale_ms}ms");
            }
            msg = read.next() => {
                match msg {
                    Some(Ok(msg)) => msg,
                    Some(Err(err)) => {
                        eprintln!("Lighter public WS read error: {err}");
                        break;
                    }
                    None => {
                        eprintln!("Lighter public WS stream ended");
                        break;
                    }
                }
            }
        };
        
        // Update freshness on each message
        self.freshness.last_ws_rx_ns.store(mono_now_ns(), Ordering::Relaxed);
        
        // ... process message, update last_parsed_ns on valid data ...
    }
    Ok(())
}
```

**Step 4: Update freshness on publish**

Either via `on_published` callback in MarketPublisher (like other connectors), or directly in the publish calls.

### Why This Approach:
- **Consistent**: Identical pattern to other 4 connectors
- **Proven**: Pattern works reliably in HL/Paradex/Extended/Aster
- **Configurable**: Env var `PARAPHINA_LIGHTER_STALE_MS` allows tuning
- **Copy-paste safe**: Same struct and logic, less chance of errors

### Testing Strategy

1. Unit test: Freshness struct methods
2. Mock test: Verify stale signal triggers after threshold
3. Integration test: Stall mock WS server, verify Lighter reconnects

---

## Fix 3: Lighter Funding Init Retry Loop

### Problem
`run_funding_polling()` calls `resolve_market_id_and_symbol()` once at startup. If it fails, the entire function exits forever with `return;`.

### Design Principle
Wrap initialization in a retry loop with exponential backoff. The main polling loop should be reached eventually.

### Optimal Implementation

```rust
pub async fn run_funding_polling(&self, interval_ms: u64) {
    let mut interval = tokio::time::interval(Duration::from_millis(interval_ms.max(500)));
    let mut seq: u64 = 0;
    
    // FIX: Retry initialization with backoff
    let mut init_backoff = Duration::from_secs(5);
    let (market_symbol, market_id) = loop {
        match self.resolve_market_id_and_symbol().await {
            Ok(val) => break val,
            Err(err) => {
                eprintln!(
                    "Lighter funding polling init error (retry in {:?}): {err}",
                    init_backoff
                );
                tokio::time::sleep(init_backoff).await;
                init_backoff = (init_backoff * 2).min(Duration::from_secs(60));
                // Continue loop to retry
            }
        }
    };
    
    eprintln!(
        "INFO: Lighter funding polling initialized symbol={} market_id={}",
        market_symbol, market_id
    );
    
    // Main polling loop (unchanged)
    loop {
        interval.tick().await;
        match fetch_public_funding(&self.http, &self.cfg, market_id, &market_symbol).await {
            Ok(mut update) => {
                seq = seq.wrapping_add(1);
                update.seq = seq;
                if let Err(err) = self
                    .market_publisher
                    .publish_market(MarketDataEvent::FundingUpdate(update))
                    .await
                {
                    eprintln!("Lighter funding publish error: {err}");
                }
            }
            Err(err) => {
                eprintln!("Lighter funding polling error: {err}");
            }
        }
    }
}
```

### Why This Approach:
- **Simple**: Just wrap existing code in `loop { ... break ... }`
- **Recoverable**: Transient API failures don't kill funding forever
- **Observable**: Logs each retry attempt with backoff duration
- **Bounded**: 60s max backoff prevents infinite fast retries

### Alternative Considered: Re-resolve on Each Poll

Could call `resolve_market_id_and_symbol()` on each poll iteration. Rejected because:
- Unnecessary API calls
- market_id is stable, doesn't need re-resolution
- More complex error handling

### Testing Strategy

1. Unit test: Mock `resolve_market_id_and_symbol` to fail N times then succeed
2. Verify: Initialization eventually completes
3. Verify: Log messages show retry attempts

---

## Fix 4: Task Supervision Framework

### Problem
All tasks are spawned fire-and-forget. Dead tasks are not detected or restarted.

### Design Principle
Implement lightweight supervision that:
1. Captures task handles
2. Detects task completion (normal or panic)
3. Restarts tasks with backoff
4. Logs task lifecycle events

### Optimal Implementation

**Option A: Simple Restart Wrapper (Recommended for MVP)**

Create a helper that wraps any async function in restart logic:

```rust
// In a new file: paraphina/src/live/supervision.rs

use std::future::Future;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Spawn a task that automatically restarts on completion or panic
pub fn spawn_supervised<N, F, Fut>(name: N, make_task: F) -> JoinHandle<()>
where
    N: Into<String> + Clone + Send + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let name = name.into();
    tokio::spawn(async move {
        let mut restart_count: u64 = 0;
        let mut backoff = Duration::from_secs(1);
        
        loop {
            let task_name = name.clone();
            let result = std::panic::AssertUnwindSafe(make_task())
                .catch_unwind()
                .await;
            
            restart_count += 1;
            
            match result {
                Ok(()) => {
                    // Task completed normally (should never happen for WS loops)
                    eprintln!(
                        "WARN: Supervised task '{}' exited normally (restart #{}), restarting in {:?}",
                        task_name, restart_count, backoff
                    );
                }
                Err(panic_info) => {
                    // Task panicked
                    eprintln!(
                        "ERROR: Supervised task '{}' panicked (restart #{}): {:?}, restarting in {:?}",
                        task_name, restart_count, panic_info, backoff
                    );
                }
            }
            
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
            
            // Reset backoff after many successful restarts? Or use different strategy?
            // For now, keep accumulating like WS reconnect (consistent behavior)
        }
    })
}
```

**Usage in paraphina_live.rs:**

```rust
// Before:
tokio::spawn(async move {
    hl_public.run_public_ws().await;
});

// After:
use paraphina::live::supervision::spawn_supervised;

spawn_supervised("hyperliquid_public_ws", move || {
    let hl = hl_public.clone();
    async move { hl.run_public_ws().await }
});
```

**Option B: JoinSet with Health Monitoring (More Complex)**

Use `tokio::task::JoinSet` to track all tasks and react to completions.

```rust
let mut tasks = JoinSet::new();

tasks.spawn(async move { hl_public.run_public_ws().await });
tasks.spawn(async move { hl_funding.run_funding_polling(poll_ms).await });
// ... etc

// In main loop, check for task completions:
tokio::select! {
    result = tasks.join_next() => {
        if let Some(result) = result {
            match result {
                Ok(()) => eprintln!("WARN: A connector task exited"),
                Err(e) => eprintln!("ERROR: A connector task panicked: {e}"),
            }
            // Could restart here, but need to know which task it was
        }
    }
    // ... other select branches
}
```

**Recommendation: Option A for MVP**

Option A is simpler, self-contained, and doesn't require major restructuring. Can upgrade to JoinSet later if needed.

### Why This Approach:
- **Minimal**: Single helper function, easy to adopt incrementally
- **Observable**: Every restart is logged with count and reason
- **Panic-safe**: Uses `catch_unwind` to survive panics
- **Backoff-aware**: Prevents restart storms

### Testing Strategy

1. Unit test: Verify restart occurs after task returns
2. Unit test: Verify panic is caught and restart occurs
3. Unit test: Verify backoff increases on consecutive restarts

---

## Fix 5: Log Channel Send Failures

### Problem
Some channel sends use `let _ = ...` which silently discards errors.

### Locations

| File | Line | Current Code |
|------|------|--------------|
| `hyperliquid.rs` | 540-543 | `let _ = self.market_tx.send(...)` |
| `lighter.rs` | 619-622 | `let _ = self.market_publisher.publish_market(...)` |

### Optimal Implementation

Replace silent ignores with explicit error handling:

```rust
// Before:
let _ = self.market_tx.send(MarketDataEvent::FundingUpdate(update)).await;

// After:
if let Err(err) = self.market_tx.send(MarketDataEvent::FundingUpdate(update)).await {
    eprintln!("Hyperliquid funding send failed: {err}");
}
```

For Lighter:
```rust
// Before:
let _ = self
    .market_publisher
    .publish_market(MarketDataEvent::FundingUpdate(update))
    .await;

// After:
if let Err(err) = self
    .market_publisher
    .publish_market(MarketDataEvent::FundingUpdate(update))
    .await
{
    eprintln!("Lighter funding publish error: {err}");
}
```

### Why This Approach:
- **Minimal**: 2-line change per location
- **Consistent**: Matches pattern already used by Extended/Aster/Paradex
- **Observable**: Failures become visible in logs

### Testing Strategy

1. Mock: Close channel receiver, verify error is logged
2. Integration: Monitor logs during extended runs

---

## Fix 6: Normalize Aster Stale Threshold

### Problem
Aster uses 1,800ms while all others use 10,000ms. This inconsistency may cause Aster to reconnect more frequently.

### Analysis

Looking at Aster's code, there are actually **two** stale thresholds:
1. `ASTER_STALE_MS = 1_800` — Used by the external watchdog task (line 308)
2. `STALE_MS = 2_000` — Used by internal resync logic (lines 287, 446, 694)

The internal resync has a cooldown (`COOLDOWN_MS = 7_000`) to prevent thrashing.

### Optimal Implementation

Make the watchdog threshold configurable and default to 10,000ms like other venues:

```rust
// Before:
const ASTER_STALE_MS: u64 = 1_800;

// After:
const ASTER_STALE_MS_DEFAULT: u64 = 10_000;  // Match other venues

fn aster_stale_ms() -> u64 {
    std::env::var("PARAPHINA_ASTER_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(ASTER_STALE_MS_DEFAULT)
}
```

The internal resync logic (2,000ms with 7,000ms cooldown) can stay as-is — it's a soft resync that just re-fetches the snapshot, not a full reconnect.

### Why This Approach:
- **Consistent**: All venues default to same threshold
- **Configurable**: Env var allows Aster-specific tuning if needed
- **Non-breaking**: Operators can set `PARAPHINA_ASTER_STALE_MS=1800` to preserve old behavior

### Testing Strategy

1. Verify: Default threshold is 10,000ms
2. Verify: Env var override works
3. Monitor: Reconnect frequency in shadow runs

---

## Implementation Priority and Order

| Priority | Fix | Risk | Effort | Impact |
|----------|-----|------|--------|--------|
| 1 | Fix 3: Lighter funding init retry | Low | Small | HIGH - Prevents permanent funding death |
| 2 | Fix 2: Lighter freshness/watchdog | Medium | Medium | HIGH - Prevents permanent WS stall |
| 3 | Fix 1: Backoff reset | Low | Small | HIGH - Improves recovery time |
| 4 | Fix 5: Log channel failures | Very Low | Tiny | MEDIUM - Improves observability |
| 5 | Fix 6: Aster threshold | Low | Tiny | LOW - Normalization only |
| 6 | Fix 4: Task supervision | Medium | Medium | MEDIUM - Defense in depth |

### Recommended Rollout

**Phase 1: Critical Fixes (Immediate)**
- Fix 3: Lighter funding init retry
- Fix 2: Lighter freshness/watchdog
- Fix 5: Log channel failures

**Phase 2: Reliability Improvements**
- Fix 1: Backoff reset (all venues)
- Fix 6: Aster threshold normalization

**Phase 3: Defense in Depth**
- Fix 4: Task supervision framework

---

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAPHINA_WS_HEALTHY_THRESHOLD_MS` | 60000 | Time before backoff resets |
| `PARAPHINA_LIGHTER_STALE_MS` | 10000 | Lighter watchdog threshold |
| `PARAPHINA_ASTER_STALE_MS` | 10000 | Aster watchdog threshold |
| `PARAPHINA_HL_STALE_MS` | 10000 | (existing) Hyperliquid threshold |
| `PARAPHINA_PARADEX_STALE_MS` | 10000 | (existing) Paradex threshold |
| `PARAPHINA_EXTENDED_STALE_MS` | 10000 | (existing) Extended threshold |

---

## Validation Plan

After implementing fixes:

1. **Unit Tests**: All new code paths have tests
2. **Shadow Run**: 24-hour shadow run with all 5 venues
3. **Metrics**: Monitor reconnect frequency, stale time, funding health
4. **Comparison**: Before/after metrics for same time period

### Success Criteria

- No venue stays Stale for > 2 minutes (excluding genuine exchange downtime)
- Lighter funding reaches Healthy within 60 seconds of startup
- Reconnects after long stable periods use 1s backoff
- All task panics are logged and recovered from

---

**End of Fix Design Document**
