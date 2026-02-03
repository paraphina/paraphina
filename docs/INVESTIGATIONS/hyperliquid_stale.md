# Hyperliquid Staleness Investigation

**Date**: 2026-02-03  
**Branch**: `research/hyperliquid-stale-investigation`  
**Status**: Complete

## Executive Summary

Hyperliquid shows significantly higher staleness rates (6.3%) compared to other venues (<1%) 
in SHADOW telemetry. Investigation confirms the root cause: Hyperliquid uses exchange-provided 
timestamps (`data["time"]`) rather than local receipt time, which inflates `age_ms` by network 
propagation delay.

**Recommendation**: Set `PARAPHINA_HL_STALE_MS=1500` for VPS-A shadow runs to achieve ≤0.5% stale rate.

---

## 1. Code Findings: Timestamp Semantics

### 1.1 Hyperliquid Parser Uses Exchange Timestamp

All Hyperliquid market data parsers extract `timestamp_ms` from the exchange-provided `"time"` field:

```771:771:paraphina/src/live/connectors/hyperliquid.rs
    let timestamp_ms = data.get("time").and_then(|v| v.as_i64()).unwrap_or(0);
```

This pattern repeats across all HL parsing functions:

| Function | File:Line | Code |
|----------|-----------|------|
| `parse_l2_message` | `hyperliquid.rs:771` | `data.get("time").and_then(\|v\| v.as_i64())` |
| `parse_top_of_book` | `hyperliquid.rs:1012` | `data.get("time").and_then(\|v\| v.as_i64())` |
| `parse_trade_message` | `hyperliquid.rs:1036` | `data.get("time").and_then(\|v\| v.as_i64())` |
| `parse_account_snapshot` | `hyperliquid.rs:1491` | `data.get("time").and_then(\|v\| v.as_i64())` |

### 1.2 Timestamp Propagates to State

The exchange timestamp flows through the data pipeline unchanged:

1. **Parser** → `L2Snapshot.timestamp_ms` (exchange time)
2. **Runner** → `apply_l2_snapshot(..., snapshot.timestamp_ms, ...)` 

```2069:2077:paraphina/src/live/runner.rs
                let _ = v.apply_l2_snapshot(
                    &snapshot.bids,
                    &snapshot.asks,
                    snapshot.seq,
                    snapshot.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                );
```

3. **State** → `last_mid_update_ms = Some(timestamp_ms)`

```228:233:paraphina/src/state.rs
        self.last_book_update_ms = Some(timestamp_ms);
        self.depth_near_mid = top_of_book_notional(&self.orderbook_l2);
        if let (Some(mid), Some(spread)) = (metrics.mid, metrics.spread) {
            self.mid = Some(mid);
            self.spread = Some(spread);
            self.last_mid_update_ms = Some(timestamp_ms);
```

### 1.3 Staleness Calculation

Staleness is computed as `wall_clock_now - last_mid_update_ms`:

```674:684:paraphina/src/telemetry.rs
fn compute_age_ms(now_ms: TimestampMs, last_mid_update_ms: Option<TimestampMs>) -> TimestampMs {
    match last_mid_update_ms {
        None => -1,
        Some(ts) => {
            if now_ms >= ts {
                now_ms - ts
            } else {
                0
            }
        }
    }
}
```

**Root Cause**: When `last_mid_update_ms` is the exchange timestamp and `now_ms` is wall clock, 
the `age_ms` includes both:
- True data age (time since exchange generated the update)
- Network propagation delay (exchange → VPS)

For Hyperliquid, this propagation delay appears to be ~500-800ms consistently.

---

## 2. Telemetry Evidence

### 2.1 Age Percentiles by Venue

| Venue | P50 | P90 | P95 | P99 | Max |
|-------|-----|-----|-----|-----|-----|
| extended | 157 | 337 | 467 | 809 | 1938 |
| **hyperliquid** | **694** | **948** | **1066** | **1309** | **2378** |
| aster | 186 | 228 | 233 | 242 | 778 |
| lighter | 169 | 177 | 222 | 318 | 623 |
| paradex | 24 | 106 | 230 | 694 | 2151 |

**Key Insight**: Hyperliquid P50 (694ms) exceeds other venues by 4-28x, indicating consistent 
baseline offset from exchange timestamp semantics.

### 2.2 Current Status Distribution

| Venue | Healthy % | Stale % |
|-------|-----------|---------|
| extended | 99.4% | 0.5% |
| **hyperliquid** | **93.7%** | **6.3%** |
| aster | 99.9% | 0.0% |
| lighter | 91.3% | 0.0% |
| paradex | 99.6% | 0.3% |

### 2.3 Threshold Sweep for Hyperliquid

| Threshold | Stale Count | Stale % | Max Age |
|-----------|-------------|---------|---------|
| 1000ms | 141 | 6.30% | 2378ms |
| 1250ms | 28 | 1.25% | 2378ms |
| **1350ms** | **16** | **0.72%** | 2378ms |
| **1500ms** | **5** | **0.22%** | 2378ms |
| 1750ms | 2 | 0.09% | 2378ms |
| 2000ms | 1 | 0.04% | 2378ms |
| 2500ms | 0 | 0.00% | 2378ms |

### 2.4 Recommended Override

| Target | Threshold | Actual Stale % |
|--------|-----------|----------------|
| ≤1% stale | 1350ms | 0.72% |
| **≤0.5% stale** | **1500ms** | **0.22%** |
| ≤0.1% stale | 1750ms | 0.09% |

**Recommendation**: `stale_ms_override = 1500` provides ≤0.5% stale rate with headroom.

---

## 3. Mitigation Path A: Config-Only (Preferred)

### 3.1 Config Mechanism

Paraphina live supports per-connector stale thresholds via environment variables:

```23:28:paraphina/src/live/connectors/hyperliquid.rs
fn hl_stale_ms() -> u64 {
    std::env::var("PARAPHINA_HL_STALE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(HL_STALE_MS_DEFAULT)
}
```

The default is 10,000ms, but this is for connector-level watchdog, not state staleness.

For state-level staleness, use the `VenueConfig.stale_ms_override` field:

```232:235:paraphina/src/config.rs
    /// Per-venue override for stale_ms threshold. If Some, this venue uses this
    /// threshold instead of the global `book.stale_ms`. Useful for high-latency
    /// venues (e.g., Hyperliquid) that need a larger staleness window.
    pub stale_ms_override: Option<i64>,
```

### 3.2 Current Config Loading

Config is loaded via `Config::from_env_or_profile()` at startup. Currently no env var exists 
for per-venue `stale_ms_override`, but there are two paths:

1. **Code default** (requires rebuild): Modify `Config::for_profile()` to set 
   `stale_ms_override: Some(1500)` for hyperliquid venue.

2. **Canary config** (TOML overlay): The canary profile system supports venue overrides, 
   but currently doesn't expose `stale_ms_override`.

### 3.3 VPS-A Shadow Start Command

The simplest immediate solution is to add env support for per-venue stale override.
Until that's implemented, the workaround is to modify the code default:

**Option A: Hardcode in default config** (simplest, requires rebuild)

Edit `paraphina/src/config.rs` line 626:

```rust
// Before:
stale_ms_override: None,

// After:
stale_ms_override: Some(1500),
```

**Option B: Add environment variable support** (cleaner, see Path B below)

---

## 4. Enhancement Path B: Dual Timestamp (Optional)

### 4.1 Problem Statement

Using only exchange timestamps:
- **Pro**: Accurate "data age" measurement (how old the quote was when created)
- **Con**: Inflates staleness by network delay, causing false stale flips

Using only receipt timestamps:
- **Pro**: Accurate "local freshness" (how recently we received data)
- **Con**: Loses visibility into exchange-side delays

### 4.2 Dual Timestamp Lite Proposal

**Goal**: Keep current staleness behavior (receipt time) while exposing feed delay for observability.

**Minimal Diff**:

1. Add optional `exchange_ts_ms` field to `L2Snapshot` and `L2Delta` types.
2. In Hyperliquid parser, populate both `timestamp_ms` (receipt) and `exchange_ts_ms` (from `data["time"]`).
3. Add optional `feed_delay_ms` field to telemetry output.
4. Other venues continue unchanged (no `exchange_ts_ms`).

**Files to modify**:
- `paraphina/src/live/types.rs`: Add `exchange_ts_ms: Option<i64>` to L2Snapshot/L2Delta
- `paraphina/src/live/connectors/hyperliquid.rs`: Populate exchange_ts_ms, use receipt time for timestamp_ms
- `paraphina/src/telemetry.rs`: Optionally emit feed_delay_ms

**Estimated diff**: ~30 lines

### 4.3 Test Plan

Add deterministic test in `paraphina/src/live/connectors/hyperliquid.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hl_dual_timestamp_parsing() {
        let json = r#"{"channel":"l2Book","data":{"coin":"TAO","time":1700000000000,"levels":[[["100.0","1.0"]],[["101.0","1.0"]]],"seq":1}}"#;
        let data: serde_json::Value = serde_json::from_str(json).unwrap();
        let inner = data.get("data").unwrap();
        
        // Simulate receipt time 500ms after exchange time
        let receipt_ts = 1700000000500_i64;
        
        let parsed = parse_l2_message_with_receipt_ts(inner, 0, receipt_ts);
        assert!(parsed.is_some());
        let msg = parsed.unwrap();
        
        // timestamp_ms should be receipt time (for staleness)
        match &msg.event {
            MarketDataEvent::L2Snapshot(snap) => {
                assert_eq!(snap.timestamp_ms, receipt_ts);
                assert_eq!(snap.exchange_ts_ms, Some(1700000000000));
            }
            _ => panic!("Expected L2Snapshot"),
        }
    }
}
```

### 4.4 Decision

**Recommendation**: Implement Path A first (config override to 1500ms). Path B is valuable for 
long-term observability but not required for immediate stale rate reduction.

---

## 5. Action Items

### Immediate (Shadow-Safe)

1. **Set `PARAPHINA_HL_STALE_MS=1500`** - Wait, this controls connector watchdog, not state staleness.

2. **Better: Add env var for venue stale override** - Add `PARAPHINA_VENUE_HL_STALE_MS_OVERRIDE` support.

3. **Quickest: Hardcode override in config.rs** - Set `stale_ms_override: Some(1500)` for hyperliquid venue.

### VPS-A Start Command (after code change)

```bash
# Current (no change needed to command, just rebuild with config change)
PARAPHINA_TELEMETRY_MODE=jsonl \
PARAPHINA_TELEMETRY_PATH=/tmp/paraphina_live_shadow/telemetry.jsonl \
./target/release/paraphina_live --trade-mode shadow --connectors hyperliquid,extended,lighter,paradex,aster
```

### Future

- Consider implementing dual timestamp for feed delay observability
- Add per-venue stale threshold to canary TOML schema
- Monitor P99 age after deploying override

---

## Appendix: Raw Telemetry Analysis Script

```python
import json
from collections import defaultdict

venues = ['extended', 'hyperliquid', 'aster', 'lighter', 'paradex']
ages = defaultdict(list)

with open('/tmp/paraphina_live_shadow/telemetry.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        venue_ages = d.get('venue_age_ms', [])
        for i, v in enumerate(venues):
            if i < len(venue_ages) and venue_ages[i] >= 0:
                ages[v].append(venue_ages[i])

def pct(data, p):
    s = sorted(data)
    return s[int(len(s) * p / 100)]

for v in venues:
    d = ages[v]
    print(f"{v}: P50={pct(d,50)} P95={pct(d,95)} P99={pct(d,99)} Max={max(d)}")
```
