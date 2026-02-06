# P0–P5 Deep Diagnosis Report

**Date**: 2026-02-06  
**Source Data**: 15,000-tick ETH/USD shadow run (`shadow_15k_eth_cb7d3f5_20260206_121552`)  
**Commit**: `cb7d3f5` (main, post-Lighter WS stability merge)  
**Duration**: ~4h 15m real time  
**Methodology**: Full source code audit + telemetry correlation + code path tracing

---

## Table of Contents

1. [P0 — Hyperliquid Reconnect Failure (CRITICAL)](#p0)
2. [P1 — Paradex Stale Flapping / stale_ms Tuning (MEDIUM)](#p1)
3. [P2 — Reconnect Telemetry / Observability Gap (MEDIUM)](#p2)
4. [P3 — Stale-Duration Escalation / Circuit Breaker (MEDIUM)](#p3)
5. [P4 — Paradex Depth/Spread Quality Gating (LOW-MEDIUM)](#p4)
6. [P5 — Lighter Decode-Miss Warning Cleanup (LOW)](#p5)

---

<a id="p0"></a>
## P0 — Hyperliquid Reconnect Failure (CRITICAL)

### 1. Observed Behavior

Hyperliquid entered an **irrecoverable stale state** at tick 10,528 and never recovered for
the remaining 4,472 ticks (~74.5 minutes). The venue's `mid` froze at $1,985.20 while the
consensus price drifted by up to ±$15.48.

**Key Telemetry Evidence:**

| Metric | Value |
|--------|-------|
| **Onset tick** | 10,528 (of 15,000) |
| **Onset `venue_age_ms`** | 2,609 ms |
| **Age at tick 14,528** | 4,002,609 ms (66.7 min) |
| **Age growth rate** | Exactly +1,000 ms/tick (1 tick ≈ 1s) |
| **Mid drift vs consensus** | +$9.65 at tick 10,600; +$15.48 at tick 11,000; −$10.29 at tick 14,000 |
| **Recovery** | **NONE** — remained stale until run termination |
| **Prior flapping** | 87 transitions total; 64 in ticks 8,051–8,683 (brief stale/healthy oscillation) |

### 2. Architecture Trace

The Hyperliquid reconnect path follows this call chain:

```
run_public_ws() [outer reconnect loop]
  └─ public_ws_once() [single connection session]
       ├─ connect_async() [wrapped in 15s timeout — FIX A1]
       ├─ subscribe l2Book
       ├─ freshness.reset_for_new_connection()
       ├─ spawn watchdog [checks every 200ms, fires after HL_STALE_MS_DEFAULT=10s]
       └─ tokio::select! { read loop with 30s read timeout — FIX A2 }
  └─ on error: backoff sleep (1s → 2s → 4s → ... → 30s max)
  └─ loop
```

**Two separate stale thresholds exist:**

| Threshold | Value | Purpose | Code Location |
|-----------|-------|---------|---------------|
| **Connector watchdog** (`HL_STALE_MS_DEFAULT`) | 10,000 ms | Triggers WS reconnection when no book events flow | `hyperliquid.rs:10` |
| **State-level** (`stale_ms_override`) | 2,000 ms | Determines `Healthy`/`Stale` in telemetry | `config.rs:646` |

### 3. Root Cause Analysis

The reconnect loop is functioning correctly at the connector level — each failed session
triggers reconnection per the backoff schedule. However, **no reconnection attempt succeeds
in producing publishable book data**. The mechanism:

1. **Connection establishment**: `connect_async` either succeeds (fast) or times out (15s).
2. **Subscription**: The `subscribe` message for `l2Book` is sent.
3. **Data phase**: The `tokio::select!` loop waits for either:
   - A WS frame (with 30s read timeout), or
   - The book-event watchdog (fires after 10s of no `last_book_event_ns` update)
4. **Failure**: If Hyperliquid responds with non-book messages (subscription ACK, pings,
   heartbeats) but **no `l2Book` data**, the watchdog fires after 10s.
5. **Bail**: `anyhow::bail!("...stale: freshness exceeded 10000ms")` propagates to the
   outer loop.
6. **Backoff**: Sleeps for up to 30s.
7. **Repeat**: Each cycle ≈ 10s + 30s = 40s. Over 74 minutes ≈ 111 failed cycles.

**The key insight**: After the initial successful period (~10,500 ticks, ~2.9 hours),
something changed on the Hyperliquid side (likely IP-level rate limiting, geographic routing
change, or server-side subscription throttling for the ETH coin). The existing reconnect
defenses (connect timeout, read timeout, book-event watchdog) all function correctly for
*detecting* the failure, but the **reconnect loop has no escalation strategy** for when
reconnection attempts repeatedly fail.

### 4. Specific Code Gaps Identified

**Gap 1: No reconnect attempt counter or telemetry**

```rust
// hyperliquid.rs:266-286 — run_public_ws()
loop {
    let session_start = std::time::Instant::now();
    if let Err(err) = self.public_ws_once().await {
        eprintln!("Hyperliquid public WS error: {err}");  // stderr only!
    }
    // No reconnect_count++, no reconnect event to telemetry channel
    tokio::time::sleep(backoff).await;
    backoff = (backoff * 2).min(Duration::from_secs(30));
}
```

**Gap 2: Backoff caps at 30s without further escalation**

The backoff doubles: 1s → 2s → 4s → 8s → 16s → 30s → 30s → 30s → (forever).
Once capped at 30s, each cycle burns 40-55s but there is:
- No exponential increase beyond 30s
- No circuit breaker (e.g., disable after N failures)
- No subscription variant (e.g., try different `nSigFigs` or `nLevels`)
- No endpoint failover

**Gap 3: Healthy threshold never resets backoff during stuck period**

```rust
let session_duration = session_start.elapsed();
if session_duration >= healthy_threshold {  // 60s
    backoff = Duration::from_secs(1);
}
```

When the watchdog fires after 10s, session duration is ~10s < 60s, so backoff keeps growing.
This is correct behavior, but it means the loop settles into a ~40s cadence permanently.

**Gap 4: No differentiation between "cannot connect" and "connected but no data"**

Both cases produce the same `eprintln!` error. An operator monitoring stderr cannot
distinguish:
- DNS/TCP/TLS failure (suggests network issue)
- Subscription rejection (suggests API issue)
- Connected but no l2Book flow (suggests rate limiting or coin-specific issue)

### 5. Prior Flapping Analysis

Before the terminal failure, Hyperliquid exhibited a **flapping burst** from tick 8,051 to
8,683 (64 transitions, each a 1-2 tick stale/healthy oscillation). This pattern reveals:

- `stale_ms_override = 2,000ms` is right at the edge of Hyperliquid's natural update cadence
  for ETH (observed `venue_age_ms` peaks of 2,000–2,900ms during the flapping window)
- The flapping itself is benign (each stale period is 1-2 ticks) but indicates the threshold
  is borderline for ETH specifically (it was tuned for TAO in the original fix)

### 6. Impact Assessment

| Impact Category | Severity | Detail |
|----------------|----------|--------|
| **Fair Value accuracy** | Moderate | FV computation correctly drops HL when stale; 4 venues remain |
| **Trading safety** | None (shadow mode) | Would be critical in live mode — stale HL price could poison hedging |
| **Observability** | High | Operator cannot see reconnect attempts in telemetry; only stderr |
| **Resource waste** | Low | Reconnect loop consumes CPU/network for 74 min with no recovery |

### 7. Recommended Fixes

1. **Reconnect attempt counter** emitted as a MarketDataEvent or telemetry field
2. **Escalation after N failures**: After 10 consecutive failed reconnect cycles (~7 min),
   emit a `VenueStatus::Disabled` event to the state engine (self-disable)
3. **Subscription variant rotation**: After 5 failures, try reduced `nSigFigs`/`nLevels`
4. **Backoff increase**: Cap at 60s instead of 30s after N failures
5. **HL `stale_ms_override` increase to 3,000ms**: Eliminates flapping for ETH while still
   catching genuine staleness. The connector watchdog (10s) handles actual disconnections.

---

<a id="p1"></a>
## P1 — Paradex Stale Flapping / stale_ms Tuning (MEDIUM)

### 1. Observed Behavior

Paradex exhibited **severe startup flapping** that self-resolved over time:

| Window (ticks) | Healthy % | Stale Ticks | Transitions | age_p95 |
|----------------|-----------|-------------|-------------|---------|
| 0–2,000 | 89.0% | 219 | 339 | 1,546 ms |
| 2,000–4,000 | 94.3% | 114 | 198 | 1,050 ms |
| 4,000–6,000 | 97.4% | 52 | 92 | 704 ms |
| 6,000–8,000 | 99.3% | 13 | 21 | 378 ms |
| 8,000–10,000 | 100.0% | 0 | 0 | 200 ms |
| 10,000–14,000 | 99.9% | 4 | 8 | ~250 ms |
| 14,000–15,000 | 100.0% | 0 | 0 | 206 ms |

### 2. Architecture Trace

Paradex uses a **BBO (Best Bid/Offer) feed** (`bbo.{market}`) rather than a full L2 orderbook
subscription. This feed publishes a single `{bid, bid_size, ask, ask_size}` message per BBO
update. The update cadence depends on market activity and Paradex's internal throttling.

**Stale determination path:**

```
telemetry.rs::build_venue_metrics()
  └─ compute_age_ms(effective_now, venue.last_mid_update_ms)
  └─ if age > venue_stale_ms → report "Stale"
       where venue_stale_ms = cfg.venues[4].effective_stale_ms(global_stale_ms)
       = stale_ms_override.unwrap_or(global_stale_ms)
       = None.unwrap_or(1_000) = 1_000 ms
```

Paradex's `stale_ms_override` is `None`, so it uses the global `book.stale_ms = 1,000 ms`.

### 3. Root Cause Analysis

**Root cause: The BBO feed cadence during startup exceeds the 1,000ms stale threshold.**

During Paradex WebSocket initialization, there is a natural delay between:
1. WS connection established
2. Subscription acknowledged
3. First BBO message received
4. Subsequent BBO messages at steady-state cadence

The BBO feed cadence during startup is 1,000–1,546ms (p95), which regularly exceeds the
1,000ms threshold. As the feed stabilizes, the cadence drops to ~200ms (p95 after tick 8,000),
well within the threshold.

**This is NOT a connectivity issue** — the connector-level watchdog (`PARADEX_STALE_MS_DEFAULT
= 10,000ms`) never fires, confirming the WS connection is healthy throughout. The flapping
occurs purely at the **state-level stale check** in `telemetry.rs`.

### 4. Specific Code Location

```rust
// config.rs:688-707 — Paradex VenueConfig
VenueConfig {
    id: "paradex".to_string(),
    // ...
    stale_ms_override: None,  // <-- uses global 1,000ms
}
```

Paradex's BBO feed does not have a `read.next()` timeout wrapper (unlike Extended which has
`tokio::time::timeout(Duration::from_secs(10), read.next())`). The only protection is the
connector-level freshness watchdog at 10,000ms.

### 5. Impact Assessment

| Impact Category | Severity | Detail |
|----------------|----------|--------|
| **Fair Value accuracy** | Low | Paradex is dropped from KF during stale ticks; 4 venues remain |
| **Trading safety** | None | Shadow mode; in live mode, stale flapping causes missed quoting opportunities |
| **Startup duration** | ~6 min | System reaches 99.3%+ health after ~8,000 ticks |
| **Operational noise** | Medium | 339 transitions in first 2,000 ticks generate stderr noise |

### 6. Recommended Fixes

1. **Set `stale_ms_override: Some(3_000)` for Paradex** in `config.rs`.
   - The BBO feed's worst-case startup gap is 1,546ms (p95).
   - 3,000ms provides 2x headroom over the worst startup case.
   - The connector watchdog (10,000ms) still catches genuine disconnections.
2. **Add environment variable `PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE`** (same pattern
   as Hyperliquid's existing env override).
3. **Add `tokio::time::timeout` to Paradex's WS read loop** (currently absent — the WS
   read has no explicit read timeout, unlike Extended which has 10s).

---

<a id="p2"></a>
## P2 — Reconnect Telemetry / Observability Gap (MEDIUM)

### 1. Observed Behavior

During the 74.5-minute Hyperliquid stale period, the **only observable signal** in the
telemetry was the monotonically increasing `venue_age_ms`. There was:

- No `reconnect_attempt` counter in telemetry
- No `reconnect_reason` field
- No `session_id` to correlate WS sessions
- No `last_reconnect_ms` timestamp
- No telemetry event distinguishing "connecting" from "connected but no data"

All reconnect information was emitted solely to stderr via `eprintln!`, which is:
- Not captured in the JSONL telemetry stream
- Not structured for automated analysis
- Not available to downstream monitoring tools

### 2. Architecture Trace

The reconnect information flow:

```
Connector Layer (run_public_ws)
  ├─ eprintln!("Hyperliquid public WS error: {err}")  ← stderr only
  ├─ eprintln!("INFO: Hyperliquid public WS connecting...")  ← stderr only
  ├─ eprintln!("INFO: Hyperliquid public WS connected...")   ← stderr only
  └─ NO EVENT → market_tx channel

Telemetry Layer (telemetry.rs)
  ├─ venue_age_ms: computed from state.venues[i].last_mid_update_ms
  ├─ venue_status: computed from age_ms vs stale threshold
  └─ NO reconnect fields
```

The connectors for ALL five venues (Extended, Hyperliquid, Aster, Lighter, Paradex) share
this same observability gap. The only difference between venues is whether they have
a read timeout on `read.next()`:

| Venue | Connect Timeout | Read Timeout | Book-Event Watchdog |
|-------|----------------|--------------|---------------------|
| Hyperliquid | 15s | 30s | 10s (book-specific) |
| Extended | 15s | 10s | 10s |
| Lighter | 15s | 30s | 10s |
| Aster | 15s | 10s | 10s |
| **Paradex** | 15s | **NONE** | 10s |

### 3. Specific Gaps

**Gap A: No reconnect counter in telemetry**

```rust
// hyperliquid.rs:266-286 — All connectors follow this pattern
loop {
    if let Err(err) = self.public_ws_once().await {
        eprintln!(...);  // ← Only observable via stderr
    }
    // Missing: reconnect_count.fetch_add(1, Ordering::Relaxed)
    // Missing: market_tx.send(MarketDataEvent::ReconnectAttempt { ... })
    tokio::time::sleep(backoff).await;
    backoff = (backoff * 2).min(Duration::from_secs(30));
}
```

**Gap B: No WS session identification**

Each call to `public_ws_once()` creates a new WS connection but there's no session ID or
connection number. When reading logs, it's impossible to tell if "public WS connected" refers
to the 1st or 111th connection attempt.

**Gap C: No differentiation of failure modes**

The current error handling collapses all failures into a single `anyhow::Error`:
- Connect timeout → "Hyperliquid public WS connect timed out after 15s"
- Read timeout → "Hyperliquid public WS read timeout after 30s"
- Watchdog → "Hyperliquid public WS stale: freshness exceeded 10000ms"

But these all produce the same `eprintln!` at the outer loop level with no structured categorization.

**Gap D: Paradex has no WS read timeout**

Looking at `paradex.rs:323-333`:
```rust
msg = read.next() => {
    let Some(msg) = msg else { break; };
    msg?
}
```

There is no `tokio::time::timeout` wrapping `read.next()`. If the WS connection enters a
half-open state (TCP alive, no WS frames), the read will block indefinitely until the
connector-level watchdog fires after 10s. Extended, Hyperliquid, and Lighter all have explicit
read timeouts.

### 4. Recommended Fixes

1. **Add an `AtomicU64` reconnect counter** per connector, exposed via a new telemetry field
   `venue_reconnect_count`
2. **Add `venue_last_connect_ms`** telemetry field showing when the last WS session started
3. **Add `venue_reconnect_reason`** enum field: `ConnectTimeout`, `ReadTimeout`, `Watchdog`,
   `StreamClosed`, `ParseError`
4. **Add `tokio::time::timeout(30s)` to Paradex's WS read loop** for consistency with
   other connectors

---

<a id="p3"></a>
## P3 — Stale-Duration Escalation / Circuit Breaker (MEDIUM)

### 1. Observed Behavior

When Hyperliquid entered the terminal stale state at tick 10,528, the system correctly:
- Reported `venue_status = "Stale"` in every subsequent tick
- Excluded Hyperliquid from `healthy_venues_used` (dropped from 5 to 4 venues)
- Continued computing fair value from the remaining 4 venues

But the system **failed to escalate** — Hyperliquid remained in `Stale` status for 74.5
minutes (4,492,609 ms) with no:
- Progression to `Disabled` status
- Alert/notification
- Self-healing action

### 2. Architecture Trace

The venue health state machine has three states:

```
Healthy ──→ Warning ──→ Disabled
   ↑          ↑            │
   └──────────┴────────────┘
```

Transitions are driven by `toxicity.rs::update_toxicity_and_health()`:

```rust
// toxicity.rs:247-254
venue.status = if venue.toxicity >= tox_high_threshold {
    VenueStatus::Disabled
} else if venue.toxicity >= tox_med_threshold {
    VenueStatus::Warning
} else {
    VenueStatus::Healthy
};
```

However, the `Stale` status reported in telemetry is **not a `VenueStatus` enum variant**.
It's a synthetic status computed in `telemetry.rs::build_venue_metrics()`:

```rust
// telemetry.rs:1494-1501
let effective_status = if matches!(venue.status, VenueStatus::Disabled) {
    "Disabled".to_string()
} else if !fixture_mode && (age < 0 || age > venue_stale_ms) {
    "Stale".to_string()  // ← synthetic, not in VenueStatus enum
} else {
    format!("{:?}", venue.status)
};
```

**This means "Stale" is a telemetry-only concept** — the internal engine still sees the venue
as `Healthy` (via `VenueStatus`). The venue is excluded from fair value computation via the
`compute_healthy_venues_used()` check, but it remains `Healthy` internally, which means:

- The toxicity module does not penalize it
- No `Disabled` transition occurs based on staleness duration
- The reconnect loop in the connector layer is completely independent of the state engine

### 3. Root Cause: Missing State-Level Staleness Escalation

The system has **no mechanism** to transition a venue from "Stale" (telemetry concept) to
"Disabled" (engine concept) based on prolonged staleness. The two systems are decoupled:

| Layer | Knows About | Acts On |
|-------|-------------|---------|
| Connector (hyperliquid.rs) | WS state, reconnect loop | Internal WS reconnection |
| Toxicity (toxicity.rs) | `mid`, `depth`, markouts | `VenueStatus` transitions |
| Telemetry (telemetry.rs) | `last_mid_update_ms` age | Synthetic "Stale" label |

There is no feedback path from "prolonged telemetry staleness" → "toxicity escalation" →
"VenueStatus::Disabled".

### 4. The Depth Fallback Path (Partial Coverage)

The toxicity module does have a depth-zero fallback:

```rust
// toxicity.rs:208-217
let depth_zero_prolonged = venue.depth_near_mid <= 0.0
    && venue.last_mid_update_ms
        .map(|last_ms| now_ms.saturating_sub(last_ms) > cfg.toxicity.depth_fallback_grace_ms)
        .unwrap_or(true);
if venue.mid.is_none() || depth_zero_prolonged {
    venue.toxicity = 1.0;  // → triggers Disabled
}
```

But in the Hyperliquid case, `venue.mid = Some(1985.20)` and `depth_near_mid > 0` because
the **last snapshot's data is retained** — the venue doesn't lose its mid/depth; the data
simply becomes stale. This is the correct architectural choice (preserving last-known data
for potential reuse), but it means the depth-zero fallback never fires for stale venues.

### 5. Recommended Fixes

1. **Add an age-based toxicity override in `update_toxicity_and_health()`**:
   ```
   if age_ms > catastrophic_stale_threshold (e.g., 120,000ms = 2min):
       venue.toxicity = 1.0  // → Disabled
   ```
2. **Alternatively, emit a `ReconnectExhausted` event** from the connector after N
   failed reconnect cycles, which the state engine can catch and use to set `Disabled`
3. **Add `venue_consecutive_stale_ticks` telemetry field** for monitoring dashboards

---

<a id="p4"></a>
## P4 — Paradex Depth/Spread Quality Gating (LOW-MEDIUM)

### 1. Observed Behavior

Paradex showed structurally thin depth and wide spreads compared to other venues:

**Depth comparison (p50 across 15,000 ticks):**

| Venue | p50 Depth | p95 Depth | Min Depth |
|-------|-----------|-----------|-----------|
| Extended | $452,390 | $631,901 | $1,288 |
| Hyperliquid | $763,079 | $1,949,342 | $649 |
| Aster | $148,086 | $332,300 | $561 |
| Lighter | $11,717 | $173,481 | $117 |
| **Paradex** | **$2,513** | **$7,221** | **$11** |

Paradex's p50 depth ($2,513) is **180x** smaller than Extended and **300x** smaller than
Hyperliquid.

**Spread analysis:**

| Metric | Value |
|--------|-------|
| Spread p50 | $0.55 |
| Spread p75 | $0.87 |
| Spread p95 | $1.74 |
| Spread max | $4.68 |
| Ticks with spread > $1 | 2,837 (18.9%) |
| Ticks with spread > $2 | 455 (3.0%) |
| Ticks with depth < $100 | 96 (0.6%) |

### 2. Root Cause Analysis

**Root cause: Paradex uses a BBO feed, which inherently provides only 1 level of depth.**

The Paradex connector subscribes to `bbo.{market}` (line 278):
```rust
let channel = format!("bbo.{}", self.cfg.market);
```

This returns a single best bid and best ask. The `depth_near_mid` is computed from this
single level: `depth_near_mid = best_bid_sz * best_bid_px + best_ask_sz * best_ask_px`.

For ETH at ~$1,975 with typical BBO sizes of 0.5–2.0 ETH per side, this yields depth of
$1,000–$4,000 — exactly matching the observed p50 of $2,513.

This is **not a data quality problem** — it's a structural limitation of the BBO feed.
Full L2 orderbook data would show significantly more depth, but Paradex's WS API for full
L2 uses a different subscription channel (`orderbook.{market}`).

### 3. Impact on Trading

In a live trading scenario, Paradex's thin depth has these implications:

1. **Kalman filter observation noise**: The KF uses depth to compute observation noise
   via `R = r_a + r_b / depth`. With depth ≈ $2,500, Paradex's observation noise is:
   R = 1e-6 + 100 / 2,500 = 0.04 — significantly higher than Extended (R ≈ 0.00022).
   The KF naturally downweights Paradex.

2. **Hedge venue selection**: The hedge engine requires `min_depth_usd = $500` (config.rs:823).
   96 ticks had depth < $100, meaning Paradex would be skipped for hedging in those moments.

3. **Exit venue selection**: Same `min_depth_usd = $500` gating applies.

4. **Spread-based quoting**: 18.9% of ticks have spread > $1, which would affect MM
   reservation price computation.

### 4. Recommended Fixes

1. **Consider subscribing to Paradex `orderbook.{market}` channel** for full L2 depth
   (requires protocol change and is a significant code change)
2. **Add a `min_depth_for_kf_usd` threshold** in `BookConfig` that excludes venues from
   KF observations when depth is too thin (e.g., < $1,000)
3. **Add depth-weighted venue contribution** to fair value: weight Paradex proportionally
   to its depth ratio vs other venues
4. **Short-term**: The existing KF `r_b` coefficient already provides implicit
   downweighting — this is the lowest-risk path

---

<a id="p5"></a>
## P5 — Lighter Decode-Miss Warning Cleanup (LOW)

### 1. Observed Behavior

Lighter showed periodic "WARN: Lighter WS decode miss" messages in stderr during the shadow
run, despite maintaining **100.00% health** across all 15,000 ticks with zero stale ticks and
zero depth-zero ticks.

### 2. Root Cause Analysis

The warnings originate from `lighter.rs:705-727`:

```rust
if let Some(top) = decode_order_book_top(&value) {
    // ... log first decoded top
} else if !initial_snapshot_applied
    && decode_miss_count < 3
    && has_lighter_book_fields(&value)
{
    decode_miss_count += 1;
    log_decode_miss("Lighter", &value, &payload, decode_miss_count, ...);
}
```

The guard `!initial_snapshot_applied` correctly limits these warnings to the **startup
phase only** (before the first L2 snapshot is processed). The `decode_miss_count < 3`
further caps the total warnings to 3 per connection.

**These warnings fire because:**

1. The first WS message received has `has_lighter_book_fields() = true` but
   `decode_order_book_top()` returns `None` (the full snapshot's format doesn't match
   the top-of-book extraction logic, which expects a different structure)
2. The full L2 decode path (`decode_order_book_channel_message`) then succeeds, setting
   `initial_snapshot_applied = true`
3. Subsequent messages are correctly processed as deltas with no warnings

**There is also a separate delta-decode failure path** at line 793-818 that logs
"Lighter order_book decode failed (mode=delta)". This fires for delta messages that
fail to parse (possibly one-sided deltas with empty arrays). These are rate-limited
to one warning per `LIGHTER_DECODE_WARN_INTERVAL_MS` (10s).

### 3. Functional Impact

**Zero functional impact.** The warnings are:
- Limited to 3 per connection (startup warnings)
- Rate-limited (delta warnings)
- Never affect venue health (100.00% healthy throughout)
- Never cause zero depth (min depth $117)

### 4. Verification

| Metric | Value |
|--------|-------|
| Health % | 100.00% |
| Stale ticks | 0 |
| Disabled ticks | 0 |
| Zero-depth ticks | 0 |
| Min depth | $117 |
| Max depth | $737,806 |

### 5. Recommended Fixes

1. **Downgrade startup warnings from WARN to DEBUG/INFO**: These are expected during the
   snapshot→delta transition and should not appear at WARN level
2. **Improve `decode_order_book_top()` to handle the full snapshot format**: This would
   eliminate the startup warnings entirely by extracting top-of-book from the snapshot
   structure
3. **For delta-mode decode failures**: Consider suppressing warnings for one-sided deltas
   that contain `asks: []` or `bids: []` (these are not failures — they're expected when
   only one side of the book changes)

---

## Summary Prioritization Matrix

| Issue | Priority | Severity | Effort | Fix Type |
|-------|----------|----------|--------|----------|
| **P0**: HL reconnect failure | **P0-CRITICAL** | Production-blocking | Medium | Code: escalation + circuit breaker |
| **P1**: Paradex stale flapping | **P1-MEDIUM** | Startup noise | Low | Config: `stale_ms_override: Some(3_000)` |
| **P2**: Reconnect observability | **P2-MEDIUM** | Operational risk | Medium | Code: telemetry fields + read timeout |
| **P3**: Stale-duration escalation | **P3-MEDIUM** | Architectural gap | Medium | Code: age-based toxicity override |
| **P4**: Paradex depth/spread | **P4-LOW** | KF auto-mitigates | High | Code: L2 subscription (if needed) |
| **P5**: Lighter decode warnings | **P5-LOW** | Cosmetic | Low | Code: log level change |

### Recommended Implementation Order

1. **P0 + P3** (together): Add reconnect counter, circuit breaker after N failures, and
   age-based Disabled escalation in toxicity module
2. **P1**: Set Paradex `stale_ms_override: Some(3_000)` + env var override
3. **P2**: Add reconnect telemetry fields + Paradex read timeout
4. **P5**: Downgrade Lighter startup warnings
5. **P4**: Evaluate full L2 subscription for Paradex (research phase)

---

*Report generated from 15,000-tick ETH/USD shadow observation on commit cb7d3f5.*
