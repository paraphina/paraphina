# ETH/USD 15,000-Tick Shadow Observation Report

**Date:** 2026-02-06
**Build:** `cb7d3f5` (post PR #134 merge)
**Pair:** ETH/USD across all 5 venues
**Duration:** 15,000 ticks (~4.2 hours)
**Telemetry:** `/tmp/paraphina_obs/shadow_15k_eth_cb7d3f5_20260206_121552/capture_15k/telemetry_window_15000.jsonl`

---

## Executive Summary

The 15,000-tick ETH/USD shadow observation revealed **one critical issue** and **three medium-priority optimizations**. The system operated with all 5 venues for the first ~2.5 hours, then Hyperliquid entered a prolonged stale state lasting 75+ minutes without recovery — despite the reconnect hardening deployed in PR #134. Meanwhile, Aster and Lighter were rock-solid (100% healthy), and Paradex's early flapping self-resolved.

### Headline Numbers

| Venue | Healthy % | Stale % | Transitions | Max Stale Streak | Max Age |
|-------|-----------|---------|-------------|-----------------|---------|
| Extended | 99.6% | 0.4% | 104 (25/hr) | 5 ticks | 5,482ms |
| Hyperliquid | **69.7%** | **30.3%** | 87 (21/hr) | **4,491 ticks (~75 min)** | **4,492,609ms** |
| Aster | 100.0% | 0.0% | 0 | 0 | 800ms |
| Lighter | 100.0% | 0.0% | 0 | 0 | 426ms |
| Paradex | 97.3% | 2.7% | 660 (158/hr) | 5 ticks | 5,545ms |

---

## 1. CRITICAL: Hyperliquid Reconnect Failure

### Observation

Hyperliquid operated at 100% healthy for the first ~10,000 ticks (~2.8 hours), then entered a prolonged stale state from which it **never recovered** for the remaining ~5,000 ticks (~1.4 hours).

**Time-segmented breakdown:**

| Window | Healthy % | Stale Ticks | Max Age (ms) |
|--------|-----------|-------------|-------------|
| 0-5,000 | 100.0% | 1 | 2,363 |
| 5,000-10,000 | 99.0% | 49 | 3,467 |
| 10,000-15,000 | **10.1%** | **4,493** | **4,492,609** |

**Evidence of frozen state:**
- Mid price froze at $1,985.20 while other venues moved to ~$2,010
- Age incremented by exactly 1,000ms per tick (= main loop interval), confirming zero freshness updates
- Mid deviation from consensus reached **+$29.55** at its worst

### Root Cause Hypothesis

The connect timeout (15s), read timeout (30s), and book-update watchdog (10s) implemented in PR #134 should trigger reconnection within ~30s of data loss. The 75-minute stale streak means **the reconnect loop itself is hanging or silently failing**. Possible causes:

1. **WS connect succeeds but subscribe silently fails** — `connect_async()` returns, but the Hyperliquid subscription message gets no response, and the code enters the read loop which then times out, reconnects, and repeats
2. **Rate limiting / IP throttling** — Hyperliquid may be throttling reconnect attempts, causing repeated connect timeouts with intervening backoff
3. **Reconnect loop has no outer timeout** — Individual operations time out, but the overall reconnect loop can retry indefinitely without ever updating freshness or escalating the failure

### Impact

- Fair value computation drops to 3-4 venues (min `healthy_venues_used=3` observed)
- Stale HL mid poisons any calculation that doesn't properly exclude stale venues
- Shadow mode prevents actual trading harm, but this **would be critical in live**

### Recommended Fix (HIGH LEVERAGE)

1. **Add reconnect-cycle timeout**: If `public_ws_once()` fails and the venue hasn't updated freshness within N seconds (e.g., 60s), log a CRITICAL warning and optionally disable the venue
2. **Add subscription ACK verification**: After connecting, verify the subscription response before entering the message loop
3. **Add stale-duration escalation**: If venue_age exceeds a catastrophic threshold (e.g., 5 minutes), force-kill the WS task and restart from scratch
4. **Log reconnect attempts to telemetry**: Currently invisible — we can't distinguish "stuck connecting" from "connected but no data"

---

## 2. MEDIUM: Paradex Stale Flapping (Startup Transient)

### Observation

Paradex flapped rapidly during startup (584 transitions in first 5,000 ticks) but **self-resolved** by the end of the run (8 transitions in last 5,000 ticks, 99.9% healthy).

| Window | Healthy % | Transitions |
|--------|-----------|-------------|
| 0-5,000 | 92.8% | 584 |
| 5,000-10,000 | 99.2% | 68 |
| 10,000-15,000 | 99.9% | 8 |

**Root cause:** Paradex BBO feed has natural update gaps. With the default 1,000ms stale threshold, any gap >1s triggers a Healthy→Stale flip. The early run had more gaps (possibly initial book building / subscription setup), which resolved as the feed stabilized.

### Impact

- 660 total transitions generate noise in monitoring
- Brief Stale periods (max 5 ticks) don't materially affect trading
- No toxicity triggered, no disablements

### Recommended Fix (MEDIUM LEVERAGE)

1. **Increase Paradex `stale_ms_override`** to 3,000ms or 5,000ms — Paradex's BBO feed is inherently less frequent than full L2 feeds. The current 1,000ms threshold is too aggressive for this venue's data cadence.
2. **Add startup grace period** — Suppress stale transitions for the first N seconds after venue connection, since initial book population naturally takes time.

---

## 3. MEDIUM: Paradex Structural Depth/Spread Quality

### Observation

Paradex has structurally thin depth and wide spreads compared to other venues:

| Metric | Extended | Hyperliquid | Aster | Lighter | Paradex |
|--------|----------|-------------|-------|---------|---------|
| Depth p50 | $452K | $763K | $148K | $11.7K | **$2.5K** |
| Depth min | $1,288 | $649 | $561 | $117 | **$11** |
| Ticks <$100 | 0 | 0 | 0 | 0 | **96** |
| Spread p50 | $0.10 | $0.10 | $0.10 | $0.08 | **$0.55** |
| Spread >$1 | 3 | 34 | 3 | 12 | **2,837** |

### Impact

- Paradex depth at $11 minimum is dangerously close to zero-depth disablement
- Wide spreads (19% of ticks >$1, 3% >$2) mean quotes placed on Paradex are structurally wider
- Fair value computation correctly uses Paradex mid, but the wide spread introduces noise

### Recommended Fix (MEDIUM LEVERAGE)

1. **Consider depth-weighted venue contribution to FV** — Venues with structurally thin depth could be downweighted
2. **Add minimum depth threshold** — Below a configurable depth floor (e.g., $50), the venue could be treated as "degraded" without full disablement
3. **Per-venue spread cap** — Refuse to quote on venues where observed spread exceeds a configurable maximum

---

## 4. LOW: Extended Mild Stale Flapping

### Observation

Extended showed 99.6% healthy with 25 transitions/hr. Max stale streak was 5 ticks. This is a mild, non-impactful issue. Trend improved over time (80 transitions in first 5K → 2 in last 5K).

### Recommended Fix (LOW LEVERAGE)

- Could benefit from a slightly higher `stale_ms_override` (e.g., 2,000ms), but current behavior is acceptable
- No action required if other priorities are addressed first

---

## 5. INFORMATIONAL: Lighter WS Decode Miss Warnings

### Observation

Throughout the run, "Lighter WS decode miss" warnings were logged at startup and periodically. Despite these warnings, Lighter maintained 100% healthy status with excellent metrics (p50 age=171ms, p95=178ms).

### Assessment

The warnings come from the `decode_order_book_top` path attempting to extract top-of-book from one-sided delta updates. When a delta only contains bid changes, the ask extraction fails and logs a warning. This is **cosmetically noisy but functionally correct** — the delta is still applied to the book properly.

### Recommended Fix (LOW LEVERAGE)

- Suppress the decode-miss warning for delta messages (only warn during initial snapshot phase)
- Or downgrade from WARN to DEBUG for messages after `initial_snapshot_applied=true`

---

## 6. INFORMATIONAL: Cross-Venue Mid Price Divergence

### Observation

| Venue | Mean Deviation | p5 | p95 | Max |
|-------|---------------|-----|-----|-----|
| Extended | -$0.86 | -$1.47 | $0.00 | $1.95 |
| Hyperliquid | +$0.43 | -$11.85 | +$13.45 | **$29.55** |
| Aster | +$0.32 | $0.00 | +$1.00 | $2.10 |
| Lighter | +$0.38 | $0.00 | +$1.27 | $2.65 |
| Paradex | -$1.11 | -$1.98 | $0.00 | $3.09 |

**Key observations:**
- Extended and Paradex consistently trade below the consensus mid
- Aster and Lighter consistently trade slightly above
- Hyperliquid's extreme deviation (+$29.55) is entirely from the stale period — when healthy, its deviation is comparable to others
- The systematic offsets (Extended -$0.86, Paradex -$1.11) represent structural basis and could inform cross-venue arbitrage signals

---

## Prioritized Optimization Roadmap

| Priority | Issue | Leverage | Effort | Risk |
|----------|-------|----------|--------|------|
| **P0** | Hyperliquid reconnect failure (75+ min stale) | CRITICAL | Medium | Low |
| **P1** | Paradex stale_ms_override tuning | HIGH | Trivial | None |
| **P2** | Reconnect telemetry/observability | HIGH | Low | None |
| **P3** | Stale-duration escalation (all venues) | HIGH | Low | Low |
| **P4** | Paradex depth/spread quality gating | MEDIUM | Medium | Low |
| **P5** | Lighter decode-miss warning cleanup | LOW | Trivial | None |
| **P6** | Extended stale_ms_override tuning | LOW | Trivial | None |

### P0: Hyperliquid Reconnect Failure — Proposed Implementation

```
1. Add `PARAPHINA_HL_RECONNECT_ESCALATION_MS` (default 120_000)
2. In the public_ws reconnect loop, track `reconnect_start_ns`
3. If (now - reconnect_start_ns) > escalation threshold:
   a. Log CRITICAL warning with reconnect attempt count
   b. Optionally: abort the task and let the parent spawn a fresh one
4. Add subscription ACK check after connect:
   a. After sending subscribe message, await first response within 10s
   b. If no response, bail immediately rather than entering the read loop
5. Emit reconnect events to telemetry:
   {"event": "ws_reconnect", "venue": "hyperliquid", "attempt": N, "reason": "..."}
```

### P1: Paradex stale_ms_override — Proposed Implementation

```
In config.rs, change paradex stale_ms_override from None to Some(3_000)
Or add env var PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE
```

---

## Appendix: Data Quality Notes

- **Tick rate:** 1.0 ticks/sec (consistent throughout)
- **Fair value:** Always available (15000/15000), range $98.08 ($1914.69 → $2012.78)
- **Kill switch:** Never triggered
- **Risk regime:** Normal throughout, ratio=0.25
- **Toxicity:** Zero across all venues for all 15,000 ticks
- **Funding:** All healthy, all rates non-null
- **Zero depth_near_mid ticks:** Zero (no book wipes) — PR #134 Lighter fix working perfectly
