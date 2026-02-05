# Connectivity Infrastructure Audit

**Date**: 2026-02-05  
**Branch**: `research/funding-repo-audit`  
**Status**: Investigation Complete (No Code Changes)

---

## Executive Summary

This audit systematically examines the connectivity infrastructure for all 5 exchange connectors (Hyperliquid, Paradex, Lighter, Extended, Aster). The investigation reveals **significant architectural inconsistencies** and **several critical failure modes** that explain the observed behavior of venues becoming stale and never recovering during extended shadow runs.

---

## 1. Architecture Overview

### 1.1 Common Components

Each connector follows a similar pattern with variations:

| Component | Description |
|-----------|-------------|
| **Connector Struct** | Holds config, HTTP client, channels, freshness tracking |
| **`run_public_ws()`** | Outer reconnection loop |
| **`public_ws_once()`** | Single WebSocket session |
| **`run_funding_polling()`** | REST polling loop for funding rates |
| **`MarketPublisher`** | Buffered channel publisher with overflow handling |
| **Freshness Tracking** | Atomic timestamps for staleness detection |
| **Watchdog Task** | Spawned task that monitors freshness and triggers reconnect |

### 1.2 Task Spawning Model

All tasks are spawned as fire-and-forget in `paraphina_live.rs`:

```rust
tokio::spawn(async move {
    connector.run_public_ws().await;
});
tokio::spawn(async move {
    connector.run_funding_polling(poll_ms).await;
});
```

**No JoinHandle captured. No supervision. No restart on panic.**

---

## 2. Per-Venue Analysis

### 2.1 Hyperliquid

**File**: `paraphina/src/live/connectors/hyperliquid.rs`

| Aspect | Implementation | Line(s) |
|--------|----------------|---------|
| WS Reconnect Loop | `run_public_ws()` with exponential backoff | 205-214 |
| Backoff Reset | **NEVER RESETS** | 212 |
| Backoff Cap | 30 seconds | 212 |
| Freshness Struct | `Freshness` with 5 atomic fields | 34-63 |
| Watchdog | Spawned task, checks every 200ms | 251-263 |
| Stale Threshold | 10,000ms (env: `PARAPHINA_HL_STALE_MS`) | 10, 23-28 |
| Stale Action | `bail!()` forces reconnect | 315 |
| Funding Polling | Infinite loop, errors logged only | 531-549 |
| Channel Send | `let _ = ...` (silent failure) | 540-543 |
| Internal Queue | 256 capacity with coalescing | 13, 265-298 |

**Unique Features**:
- Has internal publish queue (`tx_int`) with coalescing to handle bursts
- Supports snapshot refresh on REST if WS stalls
- Most mature freshness tracking

**Issues**:
1. Backoff never resets after successful connection
2. Watchdog task not supervised
3. Funding polling channel send silently fails

---

### 2.2 Paradex

**File**: `paraphina/src/live/connectors/paradex.rs`

| Aspect | Implementation | Line(s) |
|--------|----------------|---------|
| WS Reconnect Loop | `run_public_ws()` with exponential backoff | 207-216 |
| Backoff Reset | **NEVER RESETS** | 214 |
| Backoff Cap | 30 seconds | 214 |
| Freshness Struct | `Freshness` with 4 atomic fields | 36-62 |
| Watchdog | Spawned task, checks every 200ms | 277-289 |
| Stale Threshold | 10,000ms (env: `PARAPHINA_PARADEX_STALE_MS`) | 12, 24-29 |
| Stale Action | `bail!()` forces reconnect | 295 |
| Funding Polling | Infinite loop, errors logged | 218-237 |
| Channel Send | Via `publish_market()`, errors logged | 227-231 |
| Ping/Pong | Handled explicitly | 320-322 |

**Issues**:
1. Backoff never resets after successful connection
2. Watchdog task not supervised

---

### 2.3 Lighter

**File**: `paraphina/src/live/connectors/lighter.rs`

| Aspect | Implementation | Line(s) |
|--------|----------------|---------|
| WS Reconnect Loop | `run_public_ws()` with exponential backoff | 375-400 |
| Backoff Reset | **NEVER RESETS** | 398 |
| Backoff Cap | 30 seconds | 398 |
| Freshness Struct | **NONE** | - |
| Watchdog | **NONE** | - |
| Stale Threshold | **NONE** | - |
| Stale Action | N/A | - |
| Funding Polling | Has **FATAL EARLY RETURN** on init failure | 606-611 |
| Channel Send | `let _ = ...` (silent failure) | 619-622 |
| Subscribe Failure Tracking | Counts failures, logs warning | 377-395 |

**CRITICAL Issues**:
1. **No freshness tracking or watchdog** - connection can stall indefinitely without detection
2. **Funding polling exits permanently on init failure** - if `resolve_market_id_and_symbol()` fails once at startup, funding polling dies forever
3. Backoff never resets
4. Channel send silently fails

**Funding Polling Fatal Bug (lines 606-611)**:
```rust
let (market_symbol, market_id) = match self.resolve_market_id_and_symbol().await {
    Ok(val) => val,
    Err(err) => {
        eprintln!("Lighter funding polling init error: {err}");
        return;  // EXITS FOREVER - NO RETRY
    }
};
```

---

### 2.4 Extended

**File**: `paraphina/src/live/connectors/extended.rs`

| Aspect | Implementation | Line(s) |
|--------|----------------|---------|
| WS Reconnect Loop | `run_public_ws()` with exponential backoff | 223-232 |
| Backoff Reset | **NEVER RESETS** | 231 |
| Backoff Cap | 30 seconds | 231 |
| Freshness Struct | `Freshness` with 4 atomic fields | 37-62 |
| Watchdog | Spawned task, checks every 200ms | 351-363 |
| Stale Threshold | 10,000ms (env: `PARAPHINA_EXTENDED_STALE_MS`) | 12, 25-30 |
| Stale Action | `bail!()` forces reconnect | 370 |
| Funding Polling | Infinite loop, errors logged | 235-256 |
| Channel Send | Via `publish_market()`, errors logged | 244-250 |
| REST Snapshot | Fetched before WS connect, warns if fails | 266-298 |

**Issues**:
1. Backoff never resets after successful connection
2. Watchdog task not supervised

---

### 2.5 Aster

**File**: `paraphina/src/live/connectors/aster.rs`

| Aspect | Implementation | Line(s) |
|--------|----------------|---------|
| WS Reconnect Loop | `run_public_ws()` with exponential backoff | 222-230 |
| Backoff Reset | **NEVER RESETS** | 229 |
| Backoff Cap | 30 seconds | 229 |
| Freshness Struct | `Freshness` with 4 atomic fields | 74-99 |
| Watchdog | Spawned task, checks every 200ms | 300-313 |
| Stale Threshold | **1,800ms** (hardcoded `ASTER_STALE_MS`) | 44 |
| Stale Action | `bail!()` forces reconnect | 322, 544 |
| Funding Polling | Infinite loop, errors logged | 233-254 |
| Channel Send | Via `publish_market()`, errors logged | 242-247 |
| WS Resync | Re-fetches REST snapshot on stale | 446-453, 694-700 |

**Unique Features**:
- Most aggressive stale threshold (1.8s vs 10s for others)
- Has internal resync logic within WS session
- Two stale checks: watchdog + internal timer

**Issues**:
1. Backoff never resets after successful connection
2. Watchdog task not supervised
3. Very aggressive stale threshold may cause excessive reconnects

---

## 3. Comparative Analysis

### 3.1 Stale Detection Comparison

| Venue | Has Freshness | Has Watchdog | Stale Threshold | Env Override |
|-------|---------------|--------------|-----------------|--------------|
| Hyperliquid | ✅ | ✅ | 10,000ms | `PARAPHINA_HL_STALE_MS` |
| Paradex | ✅ | ✅ | 10,000ms | `PARAPHINA_PARADEX_STALE_MS` |
| **Lighter** | ❌ | ❌ | **NONE** | N/A |
| Extended | ✅ | ✅ | 10,000ms | `PARAPHINA_EXTENDED_STALE_MS` |
| Aster | ✅ | ✅ | 1,800ms | Hardcoded |

### 3.2 Funding Polling Comparison

| Venue | Init Retry | Loop Retry | Channel Error Handling |
|-------|------------|------------|------------------------|
| Hyperliquid | N/A (no init) | ✅ Continues | Silent (`let _ = ...`) |
| Paradex | N/A (no init) | ✅ Continues | Logged |
| **Lighter** | ❌ **EXITS** | ✅ Continues | Silent (`let _ = ...`) |
| Extended | N/A (no init) | ✅ Continues | Logged |
| Aster | N/A (no init) | ✅ Continues | Logged |

### 3.3 Backoff Behavior

**ALL venues have the same bug**: Backoff only increases, never resets.

```rust
// This pattern appears in ALL 5 connectors:
pub async fn run_public_ws(&self) {
    let mut backoff = Duration::from_secs(1);
    loop {
        if let Err(err) = self.public_ws_once().await {
            eprintln!("... public WS error: {err}");
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));  // NEVER RESETS
    }
}
```

**Impact Over Time**:
- After 5 reconnects: 30s wait between attempts (capped)
- Even if connection runs healthy for hours, next disconnect still waits 30s
- Accumulates across ALL disconnects in process lifetime

---

## 4. MarketPublisher Analysis

**File**: `paraphina/src/live/market_publisher.rs`

### 4.1 Architecture

```
Connector → publish_market() → market_pub_tx (bounded queue)
                                      ↓
                              [Spawned Forwarder Task]
                                      ↓
                              out_tx → Runner
```

### 4.2 Overflow Handling

When internal queue is full:
1. If event is "lossless" (e.g., snapshots, funding): blocks until send succeeds
2. If event is "lossy" (e.g., deltas): stores in `pending_latest`, overwrites previous

**Lines 108-117**:
```rust
match self.market_pub_tx.try_send(event) {
    Ok(()) => Ok(()),
    Err(TrySendError::Full(event)) => {
        let mut pending = self.pending_latest.lock().await;
        *pending = Some(event);  // Overwrite previous pending
        Ok(())
    }
    Err(TrySendError::Closed(_)) => {
        anyhow::bail!("{}", self.err_queue_closed)
    }
}
```

### 4.3 Issues

1. **Forwarder task not supervised** - if it panics, all publishing stops silently
2. **Lossy events can be dropped** - if multiple events overflow before drain, only last survives

---

## 5. Critical Issues Summary

### 5.1 Severity: CRITICAL

| Issue | Venues | Impact |
|-------|--------|--------|
| No freshness/watchdog | Lighter | Connection can stall forever undetected |
| Funding init exits forever | Lighter | Funding polling dies permanently on single failure |

### 5.2 Severity: HIGH

| Issue | Venues | Impact |
|-------|--------|--------|
| Backoff never resets | ALL | Recovery slows to 30s after multiple disconnects |
| No task supervision | ALL | Dead tasks not detected or restarted |
| Channel send silent failures | HL, Lighter | Data lost without logging |

### 5.3 Severity: MEDIUM

| Issue | Venues | Impact |
|-------|--------|--------|
| Aggressive stale threshold | Aster | 1.8s may cause excessive reconnects |
| Watchdog task unsupervised | All with watchdog | Watchdog can die silently |

---

## 6. Root Cause Analysis: "Venues Go Stale and Never Recover"

### 6.1 Scenario 1: Lighter Stalls

1. Lighter WS connects successfully
2. Exchange stops sending data (network issue, rate limit, etc.)
3. **No watchdog to detect staleness**
4. Runner marks venue Stale based on `venue_age_ms` exceeding threshold
5. No reconnect triggered - WS loop just waits for next message forever
6. Venue remains Stale indefinitely

### 6.2 Scenario 2: Backoff Accumulation (Any Venue)

1. Shadow runs for hours
2. Occasional disconnects occur (normal network variance)
3. Each disconnect increases backoff: 1s → 2s → 4s → 8s → 16s → 30s
4. After several disconnects, every reconnect waits 30s
5. During 30s wait, venue is Stale
6. Connection succeeds but backoff doesn't reset
7. Next disconnect still waits 30s
8. Cumulative effect: venues spend increasing time in Stale state

### 6.3 Scenario 3: Lighter Funding Dies at Startup

1. Shadow starts, spawns Lighter funding polling task
2. `resolve_market_id_and_symbol()` fails (network hiccup, API timeout)
3. `run_funding_polling()` prints error and **returns** (exits forever)
4. No retry mechanism - task is dead
5. Lighter funding remains Unknown/Stale for entire session

### 6.4 Scenario 4: Task Panic (Any Venue)

1. Unexpected panic in any spawned task (WS loop, funding poll, watchdog)
2. Tokio task terminates
3. No supervision to detect or restart
4. Affected functionality permanently disabled
5. No logging indicates the task died

---

## 7. Recommendations

### 7.1 Critical Fixes

1. **Add freshness/watchdog to Lighter** - consistent with other venues
2. **Wrap Lighter funding init in retry loop** - don't exit on transient failure
3. **Reset backoff after successful stable connection** - e.g., after 60s of healthy data

### 7.2 High Priority Fixes

4. **Add task supervision** - capture JoinHandles, restart on failure
5. **Log channel send failures** - don't use `let _ = ...` silently
6. **Add reconnect metrics** - track reconnect count, total stale time

### 7.3 Medium Priority Improvements

7. **Normalize stale thresholds** - Aster's 1.8s is much more aggressive than 10s for others
8. **Add circuit breaker** - if venue fails N times in M minutes, back off longer
9. **Add health endpoint** - expose per-venue connection state for monitoring

---

## 8. Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| HL run_public_ws | hyperliquid.rs | 205-214 |
| HL watchdog | hyperliquid.rs | 251-263 |
| HL funding poll | hyperliquid.rs | 531-549 |
| Paradex run_public_ws | paradex.rs | 207-216 |
| Paradex watchdog | paradex.rs | 277-289 |
| Lighter run_public_ws | lighter.rs | 375-400 |
| **Lighter funding fatal exit** | lighter.rs | **606-611** |
| Extended run_public_ws | extended.rs | 223-232 |
| Extended watchdog | extended.rs | 351-363 |
| Aster run_public_ws | aster.rs | 222-230 |
| Aster watchdog | aster.rs | 300-313 |
| MarketPublisher | market_publisher.rs | 1-120 |
| Task spawning | paraphina_live.rs | 1905-1915, 2061-2071, etc. |

---

## 9. Next Steps

1. Review and prioritize fixes with operator
2. Design task supervision framework
3. Implement backoff reset logic
4. Add Lighter freshness/watchdog
5. Fix Lighter funding init retry
6. Add comprehensive connectivity metrics
7. Create integration tests for recovery scenarios

---

**End of Investigation**
