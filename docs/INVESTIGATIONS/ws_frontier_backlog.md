# WS Frontier Backlog (Canonical)

## Current State (merged)
- PR A: `venue_age_ms` now represents APPLY-age; `venue_age_event_ms` added as event-time diagnostic.
- PR A: SharedVenueAges + Layer A enforcer + Layer B monitor decisions use APPLY-age semantics.
- PR B: Lighter outbound ping added; Extended read-timeout made env-configurable; reconnect and publisher-pressure audit counters added.
- PR C: Lighter is now wired into Layer B REST monitor using existing in-repo Lighter REST path (`/api/v1/orderBooks` / `/api/v1/orderbooks`) via `fetch_lighter_l2_snapshot`.

## Open Work (ranked by ROI)

### 1) Threshold tuning protocol (90m shadow soak)
- Goal: reduce venue apply-age p95/p99 and reconnect flaps without over-triggering fallback.
- Why it matters: static defaults are conservative; venue-specific cadence and jitter differ.
- File targets:
  - `paraphina/src/live/venue_health_enforcer.rs`
  - `paraphina/src/live/rest_health_monitor.rs`
  - `paraphina/src/live/connectors/{hyperliquid,lighter,extended,aster,paradex}.rs`
- Metrics that prove success:
  - per-venue `venue_age_ms` p50/p95/p99
  - per-venue reconnect reason counts (`WS_AUDIT ... reconnect_reason=...`)
  - rest monitor counters (`rest_attempt/success/fail/inject`)
- Risk notes:
  - overtight thresholds can create oscillation (restart/fallback churn);
  - keep unknown-age startup guard behavior unchanged.

### 2) Runner cap_hits pressure characterization, then mitigation
- Goal: identify whether runner input cap is causing stale plateaus under bursty venues.
- Why it matters: queue pressure can mask upstream connector improvements.
- File targets:
  - `paraphina/src/live/runner.rs`
  - `paraphina/src/live/market_publisher.rs`
- Metrics that prove success:
  - `PARAPHINA_MARKET_RX_STATS` cap_hits trend
  - `WS_AUDIT component=market_publisher` overflow/coalesce counters
  - `venue_age_ms` tail reduction with stable reconnect counts
- Risk notes:
  - mitigation before measurement risks trading one bottleneck for another;
  - preserve lossless behavior for snapshot classes.

### 3) Paradex feed semantic mismatch (BBO-only driver)
- Goal: explicitly account for Paradex freshness comparability limits vs deeper L2 venues.
- Why it matters: cross-venue age comparisons are biased when product semantics differ.
- File targets:
  - `paraphina/src/live/connectors/paradex.rs`
  - `tools/telemetry_analyzer.py`
  - docs in `docs/INVESTIGATIONS/`
- Metrics that prove success:
  - clear analyzer/report segmentation of Paradex vs full-depth venues
  - reduced false-positive stale interpretation in ops runbooks
- Risk notes:
  - avoid forcing artificial parity that hides genuine venue characteristics.

### 4) Backoff jitter alignment (optional, evidence-based)
- Goal: align reconnect backoff+jitter strategy across venues where currently absent.
- Why it matters: synchronized reconnect storms increase packet loss and stale windows.
- File targets:
  - `paraphina/src/live/connectors/{lighter,extended,aster,paradex}.rs`
- Metrics that prove success:
  - flatter reconnect burst histogram
  - lower repeated reconnect_reason spikes during venue incidents
- Risk notes:
  - changing retry cadence can delay recovery if jitter envelope is too broad.

### 5) Layer B fallback quality checks for Lighter snapshot parsing
- Goal: validate that Lighter REST fallback snapshot parsing handles real payload variants robustly.
- Why it matters: parsing errors during stale windows defeat Layer B recovery value.
- File targets:
  - `paraphina/src/live/rest_health_monitor.rs`
  - optional tests under `paraphina/tests/` or module tests
- Metrics that prove success:
  - rest_success/rest_inject ratios near 1.0 during induced stale tests
  - low parse-related rest_fail counts under `PARAPHINA_WS_AUDIT=1`
- Risk notes:
  - avoid endpoint/path invention; only use in-repo proven REST paths.

## 90m Shadow Soak Protocol (reference)
1. Run 90m shadow with all venues and audit enabled.
2. Capture telemetry JSONL + run log + market_rx_stats log.
3. Compute analyzer p50/p95/p99 for apply/event age.
4. Extract reconnect reason counts and rest monitor counters.
5. Compare before/after thresholds using the same venue set and runtime window.

Suggested command skeleton:
```bash
PARAPHINA_LIVE_MODE=1 \
PARAPHINA_LIVE_PREFLIGHT_OK=1 \
PARAPHINA_WS_AUDIT=1 \
PARAPHINA_MARKET_RX_STATS=1 \
PARAPHINA_TELEMETRY_MODE=jsonl \
PARAPHINA_TELEMETRY_PATH=/tmp/ws_soak/telemetry.jsonl \
timeout 90m cargo run --release -p paraphina --bin paraphina_live --features "live_hyperliquid live_lighter live_extended live_aster live_paradex" -- \
  --trade-mode shadow \
  --connectors hyperliquid,lighter,extended,aster,paradex \
  --out-dir /tmp/ws_soak
```

## Success Definition for next PRs
- Lower or stable per-venue apply-age p95/p99.
- No increase in reconnect thrash signatures.
- No growth in publisher overflow or runner cap_hits pressure.
- Layer A/B actions remain non-thrashy at startup and during transient feed stalls.
