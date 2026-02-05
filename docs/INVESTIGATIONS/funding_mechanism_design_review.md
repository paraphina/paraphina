# Funding Mechanism Design Review (Shadow-Only, No Code Changes)

This report is **design + compatibility review only**. I did **not** modify Rust code and will not in this task. The only artifact is this markdown report.

## TL;DR
- Funding is stored as `funding_8h: f64` in `VenueState` and defaults to `0.0`, so missing funding silently becomes zero. `GlobalState::new` sets this default. (`paraphina/src/state.rs:L629-L661`)
- Funding updates are currently applied only via account snapshots; `apply_account_snapshot_to_state` updates `funding_8h` only when `Some`. (`paraphina/src/live/runner.rs:L2313-L2339`)
- Shadow mode disables account snapshots for Hyperliquid and Lighter, so funding never updates in shadow. (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1951-L1984`)
- Market-data funding events exist, but `MarketDataEvent::FundingUpdate` is ignored in `apply_market_event`, so public funding feeds do not reach state. (`paraphina/src/live/types.rs:L73-L88`, `paraphina/src/live/runner.rs:L2060-L2094`)
- Aster and Extended account snapshots hardcode `funding_8h: None`, so those venues can never populate funding even in live. (`paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)
- Telemetry publishes `venue_funding_8h` from state with no null semantics, and per-intent telemetry uses `unwrap_or(0.0)` for `funding_8h`. (`paraphina/src/telemetry.rs:L1390-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`)
- MM/exit/hedge logic consumes `funding_8h` directly, so zero/unknown funding silently biases decisions. (`paraphina/src/mm.rs:L90-L123`, `paraphina/src/exit.rs:L110-L114`, `paraphina/src/hedge.rs:L366-L370`)
- Public, unauthenticated funding sources exist per venue (WS/REST), but need normalization and explicit null handling with staleness gates.
- Recommended: introduce a canonical `FundingState` (nullable + timestamps + source + status), gate funding usage by health, and ensure telemetry emits nulls not zero for unknown values.
- Shadow/live parity requires public market-data funding feeds to be wired to state (FundingUpdate) with fail-closed behavior.

## Current Repo Reality (authoritative, with evidence)

### Storage + Defaults
- `VenueState` stores `funding_8h: f64` and `GlobalState::new` sets `funding_8h: 0.0` by default. This is the primary silent-zero default. (`paraphina/src/state.rs:L629-L661`)

### Ingestion: Account Snapshots
- Account snapshots store `funding_8h: Option<f64>` in `VenueAccountCache` and it is set from account snapshots. (`paraphina/src/live/state_cache.rs:L164-L228`)
- Hyperliquid parses `funding8h` from account data. (`paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`)
- Lighter parses `funding_8h` from account data. (`paraphina/src/live/connectors/lighter.rs:L2192-L2219`)
- Paradex parses `funding_8h` from account data. (`paraphina/src/live/connectors/paradex.rs:L849-L860`)
- Aster and Extended set `funding_8h: None` in account snapshots. (`paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)

### Runner Application: Account → Global State
- Funding is applied to `GlobalState` only if `acct.funding_8h` is `Some`. Missing funding is ignored, leaving the default `0.0`. (`paraphina/src/live/runner.rs:L2313-L2339`)

### Market-Data Funding Event Path
- Market data has `FundingUpdate` in `MarketDataEvent`. (`paraphina/src/live/types.rs:L73-L88`)
- FundingUpdate events are propagated in connectors (timestamp/seq rewriting), but the live runner ignores them. (`paraphina/src/live/connectors/hyperliquid.rs:L930-L948`, `paraphina/src/live/connectors/lighter.rs:L804-L821`, `paraphina/src/live/runner.rs:L2060-L2094`)
- **Break**: `apply_market_event` explicitly drops `MarketDataEvent::FundingUpdate`. (`paraphina/src/live/runner.rs:L2060-L2094`)

### Telemetry + Watch
- `venue_funding_8h` is emitted directly from `GlobalState` arrays. (`paraphina/src/telemetry.rs:L1465-L1491`)
- Per-intent telemetry uses `venue.map(|v| v.funding_8h).unwrap_or(0.0)` (explicit zero default). (`paraphina/src/telemetry.rs:L1390-L1403`)
- Watch tool renders mid/spread/age/position/health but does not display funding fields. (`tools/paraphina_watch.py:L193-L285`)

### MM / Exit / Hedge Usage
- MM reservation price uses `funding_8h` directly. (`paraphina/src/mm.rs:L90-L123`)
- Exit edge and hedge cost use `funding_8h` directly. (`paraphina/src/exit.rs:L110-L114`, `paraphina/src/hedge.rs:L366-L370`)

### Shadow Breakpoints (why all-zero funding)
1) Shadow disables account snapshots for Hyperliquid and Lighter (no auth in shadow). (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1951-L1984`)
2) Aster and Extended account snapshots never populate funding (hardcoded `None`). (`paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)
3) Market-data funding updates are ignored by `apply_market_event`. (`paraphina/src/live/runner.rs:L2060-L2094`)
4) Defaults + telemetry emit `0.0` for missing funding (no null semantics). (`paraphina/src/state.rs:L629-L661`, `paraphina/src/telemetry.rs:L1390-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`)

### Current Funding Pipeline (as-is)
**Source (account snapshot)** → `AccountSnapshot.funding_8h` → `VenueAccountCache.funding_8h` → `apply_account_snapshot_to_state` → `GlobalState.venues[*].funding_8h` → telemetry `venue_funding_8h` → watch tool (not shown)

Evidence:
- Account snapshot parsing: Hyperliquid/Lighter/Paradex parse funding, Aster/Extended set `None`. (`paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`, `paraphina/src/live/connectors/lighter.rs:L2192-L2219`, `paraphina/src/live/connectors/paradex.rs:L849-L860`, `paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)
- Cache: `VenueAccountCache.funding_8h` and snapshot apply. (`paraphina/src/live/state_cache.rs:L164-L228`)
- Runner apply: `apply_account_snapshot_to_state` sets `funding_8h` when `Some`. (`paraphina/src/live/runner.rs:L2313-L2339`)
- Telemetry: `venue_funding_8h` array uses `funding_8h`. (`paraphina/src/telemetry.rs:L1465-L1491`)

## Exchange Semantics (official docs, accessed 2026-02-05)

### Hyperliquid
**Mechanics**
- Funding is paid every hour; formula is 8h rate paid at 1/8 per hour. Longs pay shorts when rate positive; shorts pay longs when negative. Funding cap 4%/hour. Payment uses **oracle price** (not mark).  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding (accessed 2026-02-05)
- Oracle price computed by validators; oracle used for funding computation.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/hypercore/oracle (accessed 2026-02-05)
- Mark price described separately; oracle price is distinct and updated ~every 3 seconds.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/robust-price-indices (accessed 2026-02-05)

**Public data availability (no auth)**
- REST `POST /info` with `type="metaAndAssetCtxs"` returns asset contexts including `funding`, `markPx`, `oraclePx`, and `premium`.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals (accessed 2026-02-05)
- REST `POST /info` with `type="predictedFundings"` returns funding rates and `nextFundingTime`.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals (accessed 2026-02-05)
- WS subscription `activeAssetCtx` uses `PerpsAssetCtx` with `funding` and `oraclePx`.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions (accessed 2026-02-05)

### Paradex
**Mechanics**
- Funding is **continuous accrual**; funding premium represents the **8h amount paid by longs to shorts** and is updated with mark price every 5 seconds; funding index accrues over time. Funding PnL is realized on position changes.  
Sources: https://docs.paradex.trade/docs/risk/funding-mechanism (accessed 2026-02-05)
- Funding premium uses Spot Oracle Price and USDC Oracle Price; accrued funding uses the funding index and USDC oracle price.  
Sources: https://docs.paradex.trade/docs/risk/funding-mechanism (accessed 2026-02-05)

**Public data availability (no auth)**
- REST `GET /v1/funding/data` returns `funding_rate`, `funding_rate_8h`, `funding_premium`, `funding_index`, `funding_period_hours`, and `created_at`.  
Sources: https://docs.paradex.trade/api/prod/markets/get-funding-data (accessed 2026-02-05)
- WS `SUB funding_data` provides `created_at`, `funding_index`, `funding_premium`, `funding_rate`, and `market`.  
Sources: https://docs.paradex.trade/ws/web-socket-channels/funding-data/funding-data (accessed 2026-02-05)

### Lighter
**Mechanics**
- Funding payments occur at each hour; positive rate means longs pay shorts, negative means shorts pay longs. Funding rate is based on mark vs index price.  
Sources: https://docs.lighter.xyz/trading/funding (accessed 2026-02-05)
- Hourly premium is TWAP of minute premiums; funding rate is `(premium/8) + interestRateComponent` and is clamped to [-0.5%, +0.5%].  
Sources: https://docs.lighter.xyz/trading/funding (accessed 2026-02-05)
- Funding period is 1 hour (per market config).  
Sources: https://docs.lighter.xyz/trading/contract-specifications (accessed 2026-02-05)

**Public data availability (no auth)**
- WS `market_stats/{MARKET_INDEX}` returns `current_funding_rate`, `funding_rate`, `funding_timestamp`, plus `index_price` and `mark_price`.  
Sources: https://apidocs.lighter.xyz/docs/websocket-reference (accessed 2026-02-05)

### Extended
**Mechanics**
- Funding payments applied every hour; formula: `Funding Payment = Position Size * Mark Price * (-Funding Rate)` so positive rate → longs pay shorts.  
Sources: https://docs.extended.exchange/extended-resources/trading/funding-payments (accessed 2026-02-05)
- Funding rate formula divides by 8 (8h realization), interest rate 0.01% per 8h, premium index sampled every 5 seconds, rate caps by market group.  
Sources: https://docs.extended.exchange/extended-resources/trading/funding-payments (accessed 2026-02-05)

**Public data availability (no auth)**
- Public REST `GET /api/v1/info/markets` returns `fundingRate` and `nextFundingRate` in `marketStats`.  
Sources: https://api.docs.extended.exchange/ (accessed 2026-02-05)
- Public REST `GET /api/v1/info/markets/{market}/stats` returns `fundingRate` and `nextFundingRate`; funding rate calculated every minute and applied hourly.  
Sources: https://api.docs.extended.exchange/ (accessed 2026-02-05)
- REST `GET /api/v1/info/{market}/funding` returns funding rate history.  
Sources: https://api.docs.extended.exchange/ (accessed 2026-02-05)
- Public WS streams include funding rates, mark price, and index price streams.  
Sources: https://api.docs.extended.exchange/ (accessed 2026-02-05)

### Aster
**Mechanics**
- Positive funding rate: longs pay shorts; negative: shorts pay longs. Premium index and interest rate used; premium index uses a price index (weighted spot).  
Sources: https://docs.asterdex.com/astherusex-orderbook-perp-guide/funding-rate (accessed 2026-02-05)

**Public data availability (no auth)**
- REST `GET /fapi/v1/premiumIndex` returns `markPrice`, `indexPrice`, `lastFundingRate`, `nextFundingTime`, `interestRate`.  
Sources: https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation (accessed 2026-02-05)
- REST `GET /fapi/v1/fundingRate` returns funding rate history.  
Sources: https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation (accessed 2026-02-05)
- WS `markPrice` stream includes funding rate (`r`) and next funding time (`T`).  
Sources: https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation (accessed 2026-02-05)

## Optimized Canonical Funding Model (semantics, units, sign, timestamps)

**Canonical sign convention (explicit):**  
`rate_8h > 0` means **longs pay shorts**; `rate_8h < 0` means **shorts pay longs**. This matches Hyperliquid, Lighter, Extended, and Aster docs; Paradex’s “funding premium represents 8h amount paid by longs to shorts” also matches this sign convention.

**Units**
- `rate_8h`: dimensionless fraction of notional over 8h (e.g., 0.0001 = 1 bp).  
- `rate_native`: exchange native rate (hourly, 8h, or continuous rate) with `interval_sec` describing period; for continuous venues, treat as **instantaneous 8h-equivalent** if provided (Paradex has `funding_rate_8h`).

**Timestamps**
- `as_of_ms_exchange`: timestamp from exchange payload (if provided).
- `received_ms`: time received by connector (always set).
- `next_funding_ms`: if exchange provides discrete schedule (Hyperliquid, Lighter, Extended, Aster).
- **Age** = `received_ms - as_of_ms_exchange` if present else use `received_ms - payload_timestamp_ms`.

**Settlement price basis**
- `settlement_price_kind` enumerates: `Oracle`, `Mark`, `Index`, `USDCOracleAdjusted`, `Unknown`.
- Hyperliquid funding payment uses **oracle price**.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding (accessed 2026-02-05)
- Extended funding payment uses **mark price**.  
Sources: https://docs.extended.exchange/extended-resources/trading/funding-payments (accessed 2026-02-05)
- Lighter funding uses mark and index in rate computation.  
Sources: https://docs.lighter.xyz/trading/funding (accessed 2026-02-05)
- Paradex funding premium uses Spot Oracle / USDC Oracle prices (USDC-adjusted).  
Sources: https://docs.paradex.trade/docs/risk/funding-mechanism (accessed 2026-02-05)
- Aster premium uses Price Index (spot-weighted) and funding rate; settlement basis not explicitly stated.  
Sources: https://docs.asterdex.com/astherusex-orderbook-perp-guide/funding-rate (accessed 2026-02-05)

**Status & staleness**
- `status = Healthy | Stale | Unknown` derived from age thresholds and presence of rate.
- `Unknown` if rate is missing or parse failed.
- `Stale` if `age_ms > funding_stale_ms` (per venue).

## Exchange Normalization Table

| Venue | Native rate + cadence | Canonical conversion | interval_sec | settlement_price_kind | Notes |
|---|---|---|---:|---|---|
| Hyperliquid | 8h rate paid hourly (1/8 per hour) | `rate_8h = native_8h` | 8h | Oracle | Funding payment uses oracle price; cap 4%/hour. Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding |
| Paradex | Continuous accrual; `funding_rate_8h` provided | `rate_8h = funding_rate_8h` | 8h | USDCOracleAdjusted | Funding premium uses Spot Oracle / USDC Oracle prices. Sources: https://docs.paradex.trade/docs/risk/funding-mechanism |
| Lighter | Hourly funding; premium/8 + interest, clamp | `rate_8h = hourly_rate * 8` (if hourly given) | 1h (per market) | Mark/Index | Funding rate based on mark vs index; payments hourly. Sources: https://docs.lighter.xyz/trading/funding , https://docs.lighter.xyz/trading/contract-specifications |
| Extended | Hourly funding; rate realization period 8h | `rate_8h = hourly_rate * 8` | 1h | Mark | Payment uses mark price; rate computed from premium. Sources: https://docs.extended.exchange/extended-resources/trading/funding-payments |
| Aster | Funding rate with premium + interest; nextFundingTime provided | `rate_8h = lastFundingRate * 8` (if hourly) or keep native with `interval_sec` from schedule | unknown / derive | Mark/Index (unspecified) | Docs do not state cadence explicitly; derive from `nextFundingTime`/history. Sources: https://docs.asterdex.com/astherusex-orderbook-perp-guide/funding-rate , https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation |

## Compatibility Map (design → repo)

### 1) Types layer
**Where to place canonical `FundingState`:**
- Extend `VenueState` (in `paraphina/src/state.rs`) with a nested `FundingState` (preferred) to preserve locality and avoid parallel vectors; keep `funding_8h` for backward compatibility until MM logic is upgraded.  
Evidence of current `funding_8h` usage and default: `paraphina/src/state.rs:L629-L661`.

**Avoid breaking MM logic:**
- Keep `funding_8h: f64` but set only when canonical is `Healthy`, otherwise set `NaN` or keep `0.0` behind a feature gate (proposed change only; no code change here).
- New canonical `FundingState` should be `Option` fields to represent unknown without converting to zero.

### 2) Connector layer (public funding ingestion)
**Add public funding ingestion per venue (no auth)**
Use existing market-data connectors to parse funding into `MarketDataEvent::FundingUpdate` (already defined) and **enhance** the event to carry `as_of_ms`, `interval_sec`, `next_funding_ms`, `settlement_price_kind`, `source`, `rate_native`, `rate_8h`.  
Evidence that `MarketDataEvent::FundingUpdate` exists but is ignored: `paraphina/src/live/types.rs:L73-L88`, `paraphina/src/live/runner.rs:L2060-L2094`.

**Per-venue source recommendations (public)**
- Hyperliquid: REST `metaAndAssetCtxs` (`funding`, `oraclePx`, `markPx`, `premium`); REST `predictedFundings` for `nextFundingTime`.  
Sources: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
- Paradex: WS `funding_data` and REST `funding/data` (funding_rate, funding_rate_8h, funding_premium).  
Sources: https://docs.paradex.trade/ws/web-socket-channels/funding-data/funding-data , https://docs.paradex.trade/api/prod/markets/get-funding-data
- Lighter: WS `market_stats/{MARKET_INDEX}` includes `current_funding_rate`, `funding_rate`, `funding_timestamp`, `index_price`, `mark_price`.  
Sources: https://apidocs.lighter.xyz/docs/websocket-reference
- Extended: REST `GET /api/v1/info/markets` or `/markets/{market}/stats` for `fundingRate` and `nextFundingRate`, WS funding rates stream.  
Sources: https://api.docs.extended.exchange/
- Aster: REST `GET /fapi/v1/premiumIndex` for `lastFundingRate`, `nextFundingTime`, `markPrice`, `indexPrice`; WS `markPrice` stream.  
Sources: https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation

### 3) Runner/state application
**Where to apply funding updates**
- Wire `MarketDataEvent::FundingUpdate` into `apply_market_event` to update canonical funding state (currently ignored).  
Evidence of ignore: `paraphina/src/live/runner.rs:L2060-L2094`.
- Keep `apply_account_snapshot_to_state` as a **secondary** source (source precedence: MarketDataWS/REST > AccountSnapshot).
Evidence of account snapshot application: `paraphina/src/live/runner.rs:L2313-L2339`.

**Shadow-safe**
- In shadow, funding must be driven by public market-data feed. Account snapshots are disabled in shadow.  
Evidence: `paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1951-L1984`.

**Avoid silent 0.0**
- Do not set `funding_8h` unless canonical funding is `Healthy`. Unknown must remain `None` in canonical state.  
Evidence of default zero and telemetry emitting zeros: `paraphina/src/state.rs:L629-L661`, `paraphina/src/telemetry.rs:L1390-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`.

### 4) Telemetry + operator UX
**Telemetry upgrades (proposed)**
Add nullable fields:
- `venue_funding_rate_8h`
- `venue_funding_age_ms`
- `venue_funding_interval_sec`
- `venue_next_funding_ms`
- `venue_funding_source`
- `venue_funding_status`
- `venue_funding_settlement_price_kind`

Where to change:
- `paraphina/src/telemetry.rs` for metric arrays (currently only `venue_funding_8h`). (`paraphina/src/telemetry.rs:L1465-L1491`)
- `tools/paraphina_watch.py` to display funding + age in venue table. (`tools/paraphina_watch.py:L193-L285`)

**Null semantics**
- Emit `null` for unknown funding (no coercion to `0.0`).
- Keep legacy `venue_funding_8h` only if needed for backward compatibility, but avoid reusing it for decisions.

### 5) Staleness/health gating
- Add per-venue `funding_stale_ms` with safe defaults. If `funding_status != Healthy`, funding must be excluded from MM/Exit/Hedge scoring.
- Add “funding boundary window” (e.g., skip funding usage within ±N minutes of next funding time).
- Make funding usage feature-gated (OFF by default) to avoid unexpected live impact.

## Telemetry + Watch Tool Plan (null semantics; schema updates)
- Telemetry should emit `null` for Unknown funding values and include age/status/source so operators can see why funding is ignored.
- Update telemetry schema docs to include the new fields and nullability rules.
- Update `tools/paraphina_watch.py` table to include columns: `funding_8h`, `funding_age_ms`, `next_funding_ms`, `funding_status` (when present).  
Evidence of current watch output: `tools/paraphina_watch.py:L193-L285`.

## Safety Gates (fail-closed)
- **Fail-closed rule:** If funding is missing or stale, it must **not** be treated as `0.0`.
- **Staleness gate:** `status = Stale` if `age_ms > funding_stale_ms`.
- **Boundary gate:** Disable funding usage within `avoid_window_ms` around `next_funding_ms` to avoid discontinuity.
- **Feature flags:** Funding usage in MM/Exit/Hedge off by default; enable per-venue once validation passes.
- **Shadow vs live parity:** In shadow, only public market-data funding should populate canonical state; account snapshot funding is optional in live and must not be required.

## MM/Exit/Hedge Integration Impacts (conceptual, off-by-default)
- Replace direct `funding_8h` reads with canonical `FundingState` (when `Healthy`).
- Use `rate_8h` * horizon fraction * price basis aligned to venue (oracle/mark/index).
- Keep the existing weight parameters, but apply only when funding is healthy and not near boundary window.

Evidence of current usage (for later refactor):
- MM: `funding_8h` drives reservation price adjustment. (`paraphina/src/mm.rs:L90-L123`)
- Exit: `funding_8h` affects ranking. (`paraphina/src/exit.rs:L110-L114`)
- Hedge: `funding_8h` affects cost. (`paraphina/src/hedge.rs:L366-L370`)

## Testing + Rollout Plan (design only)
1) **Fixtures**
   - Add deterministic funding payload fixtures for each venue (public REST/WS).
2) **Propagation tests**
   - Connector parse → FundingUpdate event → runner apply → canonical FundingState → telemetry.
3) **Staleness tests**
   - Verify `status` transitions to `Stale` and MM ignores funding.
4) **Telemetry schema tests**
   - Ensure new fields exist; unknown funding is `null`, not `0.0`.
5) **Feature gate tests**
   - Funding usage OFF by default; enabling should change behavior only when `Healthy`.
6) **Later phase**
   - Funding PnL attribution stream (out of scope; design only).

## Ranked Recommendations

**Critical**
- Wire `MarketDataEvent::FundingUpdate` into `apply_market_event`; otherwise public funding data will never reach state. (`paraphina/src/live/runner.rs:L2060-L2094`)
- Introduce explicit `Unknown` funding state; stop coercing missing funding to `0.0`. (`paraphina/src/state.rs:L629-L661`, `paraphina/src/telemetry.rs:L1390-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`)

**High**
- Add funding staleness + boundary gates; use only `Healthy` funding in MM/Exit/Hedge. (`paraphina/src/mm.rs:L90-L123`, `paraphina/src/exit.rs:L110-L114`, `paraphina/src/hedge.rs:L366-L370`)
- Add public funding ingestion for Aster/Extended (no account snapshot funding). (`paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)

**Medium**
- Add telemetry fields for funding age/status/source; update watch tool for operator visibility. (`paraphina/src/telemetry.rs:L1465-L1491`, `tools/paraphina_watch.py:L193-L285`)

**Low**
- Add funding PnL attribution stream after funding state is stable.

## Open Questions / Assumptions
- Aster funding **cadence** is not explicitly stated in docs; recommend deriving from `nextFundingTime`/history and treating as per-symbol schedule.  
Sources: https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation (accessed 2026-02-05)
- Paradex continuous funding: best choice is to use `funding_rate_8h` when available; confirm if any venue-specific clamps apply.  
Sources: https://docs.paradex.trade/api/prod/markets/get-funding-data (accessed 2026-02-05)
- Lighter market config can set funding period; treat `interval_sec` as per-market.  
Sources: https://docs.lighter.xyz/trading/contract-specifications (accessed 2026-02-05)

## Commands Run
```
pwd && git status -sb && git checkout -b research/funding-design-review
rg -n "funding|funding_8h|FundingUpdate|venue_funding|funding_rate|nextFunding|premium|index" paraphina/src tools docs tests
rg -n "MarketDataEvent::FundingUpdate" -S paraphina/src && rg -n "funding_weight|funding_horizon|funding" paraphina/src/mm.rs paraphina/src/exit.rs paraphina/src/hedge.rs
rg -n "account_snapshots_disabled" paraphina/src/bin/paraphina_live.rs
```

## Key rg excerpts
```
paraphina/src/state.rs:660:                funding_8h: 0.0,
paraphina/src/live/runner.rs:2337:        if let Some(funding) = acct.funding_8h {
paraphina/src/telemetry.rs:1490:        venue_funding_8h.push(venue.funding_8h);
```
```
paraphina/src/live/runner.rs:2093:        | super::types::MarketDataEvent::FundingUpdate(_) => {}
paraphina/src/live/connectors/hyperliquid.rs:945:        MarketDataEvent::FundingUpdate(mut funding) => {
paraphina/src/live/connectors/lighter.rs:818:        MarketDataEvent::FundingUpdate(mut funding) => {
```
```
paraphina/src/bin/paraphina_live.rs:1868:                            "paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=hyperliquid"
paraphina/src/bin/paraphina_live.rs:1981:                        eprintln!("paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=lighter");
```
