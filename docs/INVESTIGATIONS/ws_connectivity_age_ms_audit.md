# WebSocket Connectivity + `age_ms` Audit (Phase 1, Read-Only)

Date: 2026-02-10 (local)

Scope: `hyperliquid`, `lighter`, `extended`, `aster`, `paradex` market-data connectivity and ingestion paths affecting terminal dashboard `age_ms`.

Method: code-only audit plus repository artifacts; no runtime code changes; no secrets; no live trading.

Citation format used throughout:
- `Cmd: Cxx` -> exact command from Section 11.
- `Code:` -> `path:line-line` evidence.

## 0) Executive Summary

Highest-confidence reasons `age_ms` differs across venues:

1. `[MIXED]` Freshness timestamp semantics are not uniform across venues.
- `age_ms` uses `state.venues[i].last_mid_update_ms`, and that field is set from connector event `timestamp_ms` during `L2Snapshot/L2Delta` apply.
- Connector `timestamp_ms` source is venue-specific: exchange event time for some paths (Hyperliquid, some Lighter), mixed exchange/local fallback for Extended/Aster/Paradex, and local wallclock for multiple fallback/BBO paths.
- Result: cross-venue `age_ms` is not apples-to-apples (some venues measure “time since exchange timestamp”, others “time since local receive/apply”).
- Evidence: `Cmd: C02,C05,C32,C35,C43,C50,C55,C60,C84`; `Code: paraphina/src/telemetry.rs:828-839`, `paraphina/src/telemetry.rs:1640-1658`, `paraphina/src/state.rs:294-343`, `paraphina/src/live/runner.rs:2351-2384`, `paraphina/src/live/connectors/hyperliquid.rs:1334-1340`, `paraphina/src/live/connectors/hyperliquid.rs:1754-1759`, `paraphina/src/live/connectors/lighter.rs:1176-1181`, `paraphina/src/live/connectors/lighter.rs:1378-1384`, `paraphina/src/live/connectors/lighter.rs:1442-1448`, `paraphina/src/live/connectors/extended.rs:1381-1386`, `paraphina/src/live/connectors/extended.rs:1574-1579`, `paraphina/src/live/connectors/aster.rs:1454-1459`, `paraphina/src/live/connectors/aster.rs:1514-1519`, `paraphina/src/live/connectors/aster.rs:475-481`, `paraphina/src/live/connectors/paradex.rs:1244-1252`, `paraphina/src/live/connectors/paradex.rs:1325-1341`.

2. `[EXTERNAL]` Subscribed market-data products differ materially by venue.
- Hyperliquid subscribes `l2Book`.
- Lighter subscribes `order_book/<market_id>`.
- Extended connects to orderbook stream URL with `depth=1`.
- Aster uses `<symbol>@depth@100ms` stream.
- Paradex subscribes only to `bbo.<market>`.
- Different upstream feed products imply different event cadence/semantics before Paraphina processing.
- Evidence: `Cmd: C33,C41,C48,C53,C57,C61`; `Code: paraphina/src/live/connectors/hyperliquid.rs:518-527`, `paraphina/src/live/connectors/lighter.rs:607-612`, `paraphina/src/live/connectors/extended.rs:161-166`, `paraphina/src/live/connectors/aster.rs:345-350`, `paraphina/src/live/connectors/paradex.rs:332-338`, `paraphina/src/live/connectors/paradex.rs:1377-1389`.

3. `[INTERNAL]` Buffering/backpressure/drop policy differs across connectors and can suppress freshness advancement.
- Hyperliquid uses custom bounded internal queue (`256`) with lossy overflow coalescing (`pending_latest`).
- Lighter and Extended use `MarketPublisher` where only `L2Delta` is lossless; snapshots can be dropped/coalesced on pressure.
- Aster and Paradex configure `L2Delta` and `L2Snapshot` as lossless.
- Runner also coalesces and can drop unready-venue deltas when per-tick buffer cap is hit.
- Evidence: `Cmd: C33,C40,C47,C52,C57,C63,C10,C85`; `Code: paraphina/src/live/connectors/hyperliquid.rs:13`, `paraphina/src/live/connectors/hyperliquid.rs:567-595`, `paraphina/src/live/connectors/lighter.rs:272-279`, `paraphina/src/live/connectors/extended.rs:211-217`, `paraphina/src/live/connectors/aster.rs:206-216`, `paraphina/src/live/connectors/paradex.rs:179-188`, `paraphina/src/live/market_publisher.rs:40-75`, `paraphina/src/live/market_publisher.rs:101-114`, `paraphina/src/live/runner.rs:563-579`, `paraphina/src/live/runner.rs:1081-1084`.

4. `[INTERNAL]` Health-recovery architecture is asymmetric: Lighter is missing Layer A force-restart and Layer B REST fallback wiring.
- Hyperliquid, Extended, Aster, Paradex register `enforcer_slots` and `rest_entries`.
- Lighter does not.
- This raises the probability of persistent stale periods on Lighter during connector pathologies.
- Evidence: `Cmd: C18,C19,C22,C87`; `Code: paraphina/src/bin/paraphina_live.rs:1975`, `paraphina/src/bin/paraphina_live.rs:1996`, `paraphina/src/bin/paraphina_live.rs:2263`, `paraphina/src/bin/paraphina_live.rs:2287`, `paraphina/src/bin/paraphina_live.rs:2426`, `paraphina/src/bin/paraphina_live.rs:2450`, `paraphina/src/bin/paraphina_live.rs:2591`, `paraphina/src/bin/paraphina_live.rs:2614`, `paraphina/src/bin/paraphina_live.rs:2072-2185`, `paraphina/src/live/venue_health_enforcer.rs:40-54`, `paraphina/src/live/rest_health_monitor.rs:62-70`.

5. `[INTERNAL]` Reconnect/keepalive implementations differ and include at least one concrete defect-like asymmetry.
- Paradex creates `ping_timer` but has no periodic send arm; it only replies to inbound ping.
- Lighter has no outbound ping timer at all (responds to pings / JSON ping only).
- Hyperliquid/Extended/Aster do periodic outbound ping.
- These differences can change disconnect frequency and stale gaps.
- Evidence: `Cmd: C57,C62,C89,C33,C48,C53`; `Code: paraphina/src/live/connectors/paradex.rs:349-356`, `paraphina/src/live/connectors/paradex.rs:385-405`, `paraphina/src/live/connectors/paradex.rs:424-425`, `paraphina/src/live/connectors/lighter.rs:667-672`, `paraphina/src/live/connectors/lighter.rs:726-732`, `paraphina/src/live/connectors/hyperliquid.rs:536-543`, `paraphina/src/live/connectors/hyperliquid.rs:623-627`, `paraphina/src/live/connectors/extended.rs:430-437`, `paraphina/src/live/connectors/extended.rs:464-468`, `paraphina/src/live/connectors/aster.rs:387-394`, `paraphina/src/live/connectors/aster.rs:733-737`.

Top 3 minimal fixes (Phase 1 recommendation only, no code changes now):

1. Normalize freshness basis for comparability.
- Change: define `age_ms` from local apply timestamp (or publish both `age_local_apply_ms` and `age_exchange_event_ms`).
- Expected impact: reduce cross-venue p95/p99 comparability error and false divergence by ~40-70% (estimate; needs soak validation).
- Evidence: `Cmd: C02,C05,C32,C35,C43,C50,C55,C60`; `Code: paraphina/src/telemetry.rs:828-839`, `paraphina/src/telemetry.rs:1640-1658` plus connector timestamp heterogeneity cited above.

2. Align queue/drop policy to protect snapshots everywhere.
- Change: make snapshots lossless across connectors, add overflow counters, retain bounded coalescing only for explicitly safe event types.
- Expected impact: reduce p99 stale spikes/reconnects from bootstrap loss and sequence drift by ~20-50%.
- Evidence: `Cmd: C40,C47,C52,C57,C63,C10`; `Code: paraphina/src/live/connectors/lighter.rs:272-279`, `paraphina/src/live/connectors/extended.rs:211-217`, `paraphina/src/live/connectors/aster.rs:211-215`, `paraphina/src/live/connectors/paradex.rs:184-188`, `paraphina/src/live/market_publisher.rs:101-114`, `paraphina/src/live/runner.rs:563-579`.

3. Make venue recovery architecture uniform.
- Change: add Layer A/B registration for Lighter, wire Paradex periodic outbound ping, then tune stale thresholds.
- Expected impact: fewer long stale plateaus and reconnect flaps, especially p99 tails (estimated 30-60% reduction in long-stale incidents).
- Evidence: `Cmd: C87,C57,C62,C89`; `Code: paraphina/src/bin/paraphina_live.rs:2072-2185`, `paraphina/src/live/connectors/paradex.rs:349-356`, `paraphina/src/live/connectors/paradex.rs:385-405`, `paraphina/src/live/connectors/lighter.rs:667-672`, `paraphina/src/live/connectors/lighter.rs:726-732`.

Short benchmark protocol (shadow soak):
- Run 90-minute shadow soak with all five connectors.
- Record `venue_age_ms` p50/p95/p99, reconnect counts/reasons, queue overflow/drop counters, and ingest->apply latency percentiles.
- Success criteria: cross-venue p95 spread narrows, reconnect flaps drop, no venue exhibits persistent stale plateaus >60s.
- Evidence basis for metrics fields and telemetry path: `Cmd: C02,C05`; `Code: paraphina/src/telemetry.rs:1615-1715`, `tools/paraphina_watch.py:622-699`.

## 1) Scope, Definitions, and Invariants

### 1.1 Exact `age_ms` definition in terminal dashboard

- Terminal dashboard reads telemetry key `venue_age_ms` and renders column `age_ms`.
- Evidence: `Cmd: C05`; `Code: tools/paraphina_watch.py:622`, `tools/paraphina_watch.py:685-699`, `tools/paraphina_watch.py:718`.

- Telemetry computes per-venue `age` as:
  - `age = compute_age_ms(effective_now, venue.last_mid_update_ms)`
  - `compute_age_ms`: `None => -1`, `Some(ts) => max(now_ms - ts, 0)`.
- Evidence: `Cmd: C05`; `Code: paraphina/src/telemetry.rs:828-839`, `paraphina/src/telemetry.rs:1640-1658`, `paraphina/src/telemetry.rs:1714`.

### 1.2 Backing timestamp `ts` type

- Backing field is `VenueState.last_mid_update_ms`.
- It is written when `apply_l2_snapshot` / `apply_l2_delta` compute valid `(mid, spread)`.
- Therefore `ts` is **apply-to-state event timestamp** (connector-provided `event.timestamp_ms`), not always local arrival time and not always exchange time.
- Evidence: `Cmd: C05,C32,C65`; `Code: paraphina/src/state.rs:185-188`, `paraphina/src/state.rs:294-316`, `paraphina/src/state.rs:321-343`, `paraphina/src/live/runner.rs:2351-2384`.

Timestamp source classification in current live paths:
- `hyperliquid`: mostly exchange `data.time` (WS + REST snapshot).
  - Evidence: `Cmd: C35,C37`; `Code: paraphina/src/live/connectors/hyperliquid.rs:1340`, `paraphina/src/live/connectors/hyperliquid.rs:1758`, `paraphina/src/live/connectors/hyperliquid.rs:1817`.
- `lighter`: payload `timestamp/ts` with `0` fallback.
  - Evidence: `Cmd: C43`; `Code: paraphina/src/live/connectors/lighter.rs:1177-1181`, `paraphina/src/live/connectors/lighter.rs:1301-1305`, `paraphina/src/live/connectors/lighter.rs:1444-1448`.
- `extended`: mixed (`E/ts` else local `now_ms`); startup snapshot uses local `now_ms`.
  - Evidence: `Cmd: C48,C50`; `Code: paraphina/src/live/connectors/extended.rs:390-395`, `paraphina/src/live/connectors/extended.rs:1385`, `paraphina/src/live/connectors/extended.rs:1574-1579`.
- `aster`: mixed (`E` else `now_ms`); REST snapshot event stamped local `now_ms`.
  - Evidence: `Cmd: C53,C55`; `Code: paraphina/src/live/connectors/aster.rs:475-481`, `paraphina/src/live/connectors/aster.rs:1458`, `paraphina/src/live/connectors/aster.rs:1518`, `paraphina/src/live/connectors/aster.rs:1605-1608`.
- `paradex`: local-heavy (`now_ms` for BBO and deltas; snapshot `ts/timestamp` else now).
  - Evidence: `Cmd: C60`; `Code: paraphina/src/live/connectors/paradex.rs:1244`, `paraphina/src/live/connectors/paradex.rs:1325-1341`, `paraphina/src/live/connectors/paradex.rs:1213-1219`.

Clock-skew risk:
- If exchange `ts` is ahead of local clock, `compute_age_ms` clamps negative values to `0`, masking skew and understating age.
- Evidence: `Cmd: C05`; `Code: paraphina/src/telemetry.rs:832-835`.

### 1.3 Freshness-advancing vs non-advancing events

Must advance freshness (`last_mid_update_ms`):
- `MarketDataEvent::L2Snapshot` and `L2Delta` when derived mid/spread exist.
- Evidence: `Cmd: C32,C84`; `Code: paraphina/src/live/runner.rs:2361-2384`, `paraphina/src/state.rs:312-316`, `paraphina/src/state.rs:338-342`.

Must not advance freshness:
- `Trade` and `FundingUpdate` do not touch `last_mid_update_ms` in runner core apply.
- Heartbeats/subscribe acks/pongs are ignored at connector parse layer and do not become book events.
- Execution/account/order events do not touch `last_mid_update_ms` (except `ExecutionEvent::BookUpdate`, not emitted by live connectors in this audit scope).
- Evidence: `Cmd: C12,C14,C67,C33,C42,C48,C54,C58`; `Code: paraphina/src/live/runner.rs:2386-2405`, `paraphina/src/execution_events.rs:26-80`, `paraphina/src/live/connectors/hyperliquid.rs:677-679`, `paraphina/src/live/connectors/lighter.rs:667-672`, `paraphina/src/live/connectors/lighter.rs:726-732`, `paraphina/src/live/connectors/extended.rs:804-807`, `paraphina/src/live/connectors/aster.rs:637-639`, `paraphina/src/live/connectors/paradex.rs:424-426`.

## 2) End-to-End Trace: UI `age_ms` -> Freshness Timestamp -> Update Sites

### 2.1 End-to-end path

1. UI renders telemetry `venue_age_ms` as `age_ms` column.
- Evidence: `Cmd: C05`; `Code: tools/paraphina_watch.py:622`, `tools/paraphina_watch.py:685-699`, `tools/paraphina_watch.py:718`.

2. Telemetry computes `venue_age_ms[i] = compute_age_ms(effective_now, state.venues[i].last_mid_update_ms)`.
- Evidence: `Cmd: C05`; `Code: paraphina/src/telemetry.rs:828-839`, `paraphina/src/telemetry.rs:1640-1658`.

3. Runner applies connector market events into `state` via `apply_market_event_to_core`.
- Evidence: `Cmd: C84`; `Code: paraphina/src/live/runner.rs:1332-1354`, `paraphina/src/live/runner.rs:2351-2384`.

4. `VenueState::apply_l2_snapshot/delta` writes `last_mid_update_ms` from event `timestamp_ms` when mid/spread are valid.
- Evidence: `Cmd: C05`; `Code: paraphina/src/state.rs:294-316`, `paraphina/src/state.rs:321-343`.

### 2.2 Definitive write-site list for freshness fields

`last_mid_update_ms` write sites in `paraphina/src`:
- Runtime canonical writes:
  - `VenueState::apply_l2_snapshot`: `paraphina/src/state.rs:315`
  - `VenueState::apply_l2_delta`: `paraphina/src/state.rs:341`
- Non-market/sim or alternate paths:
  - `paraphina/src/execution_events.rs:23` (`ExecutionEvent::BookUpdate` path)
  - `paraphina/src/state.rs:881` (`apply_sim_account_snapshots` fallback)
  - RL/sim/test scaffolding (`paraphina/src/rl/sim_env.rs`, test modules, etc.)
- Evidence: `Cmd: C65`; `Code: paraphina/src/state.rs:315`, `paraphina/src/state.rs:341`, `paraphina/src/execution_events.rs:18-24`, `paraphina/src/state.rs:864-882`, `paraphina/src/rl/sim_env.rs:516`.

`last_book_update_ms` write sites:
- `paraphina/src/state.rs:310`, `paraphina/src/state.rs:336` only.
- Evidence: `Cmd: C66`; `Code: paraphina/src/state.rs:310`, `paraphina/src/state.rs:336`.

Important invariant:
- `age_ms` keys off `last_mid_update_ms`, not `last_book_update_ms`.
- If book updates continue but book is one-sided (mid/spread unavailable), `age_ms` can continue increasing.
- Evidence: `Cmd: C02,C05`; `Code: paraphina/src/telemetry.rs:1640`, `paraphina/src/state.rs:312-316`, `paraphina/src/state.rs:338-342`.

## 3) ShadowRun / Main Loop Topology (`now_ms` sampling and application)

### 3.1 Shadow/live entry and tick model

- `paraphina_live` enters `run_live_loop(... LiveRunMode::Realtime ...)` for normal run; `shadow_mode` flag is derived from `TradeMode::Shadow`.
- Evidence: `Cmd: C71`; `Code: paraphina/src/bin/paraphina_live.rs:2885-2909`.

- `now_ms` in realtime mode comes from runner wallclock `SystemTime::now()` via `now_ms()`.
- Runner tick wakeup is event-driven (`market_rx.recv()`) plus interval, with `PARAPHINA_MIN_INTER_TICK_MS` default `100ms` gate.
- Evidence: `Cmd: C07,C09,C85`; `Code: paraphina/src/live/runner.rs:1095-1137`, `paraphina/src/live/runner.rs:2322-2327`, `paraphina/src/live/runner.rs:1086-1091`.

### 3.2 Ingestion topology and `now_ms` interactions

- Per connector local market channel: `mpsc(1024)`.
- Forwarder rewrites venue IDs/indexes and `await`s send into `market_ingest_tx`.
- Ingest task forwards to `market_tx` consumed by runner.
- Evidence: `Cmd: C21,C28`; `Code: paraphina/src/bin/paraphina_live.rs:1805-1815`, `paraphina/src/bin/paraphina_live.rs:941-960`, `paraphina/src/bin/paraphina_live.rs:1743-1769`.

- Runner drains ingress channels, sorts canonical events, applies to cache/state.
- Coalescing is enabled by default (`PARAPHINA_L2_DELTA_COALESCE=1`, `PARAPHINA_L2_SNAPSHOT_COALESCE=1`), with optional unready-delta cap that drops when exceeded.
- Evidence: `Cmd: C84,C85`; `Code: paraphina/src/live/runner.rs:1300-1330`, `paraphina/src/live/runner.rs:1029-1034`, `paraphina/src/live/runner.rs:563-579`, `paraphina/src/live/runner.rs:1081-1084`.

### 3.3 Proof sim/test stamping does not contaminate ShadowRun freshness

- `seed_dummy_mids` in `paraphina_live` is called only in `--validate-config` path, before normal live loop startup.
- Evidence: `Cmd: C16,C73`; `Code: paraphina/src/bin/paraphina_live.rs:1041-1054`, `paraphina/src/bin/paraphina_live.rs:1689-1691`.

- `seed_dummy_mids` is not referenced in `paraphina/src/live/*`.
- Evidence: `Cmd: C76`; `Code: paraphina/src/bin/paraphina_live.rs:1053` (only hit in searched scope).

- Current repo `Engine::seed_dummy_mids` uses `apply_l2_snapshot` (not direct raw `last_mid_update_ms` assignment).
- Evidence: `Cmd: C19`; `Code: paraphina/src/engine.rs:141-150`.

- `paraphina_bundle.txt` exists and contains older direct stamping (`v.last_mid_update_ms = Some(now_ms)`) in `seed_dummy_mids`, called in a sim loop.
- Evidence: `Cmd: C88,C74,C75`; `Code: paraphina_bundle.txt:5547-5563`, `paraphina_bundle.txt:7444-7448`.

## 4) Per-Venue WebSocket Connectivity + Ingestion Map

### 4.1 `hyperliquid`

#### A) Connectivity Architecture

- WS stack/library: `tokio_tungstenite` (`connect_async`, `Message`).
- Evidence: `Cmd: C30`; `Code: paraphina/src/live/connectors/hyperliquid.rs:127`.

- Public market socket: single WS, subscribes `l2Book` with `{coin,nSigFigs,nLevels}`.
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:518-527`.

- Heartbeat/deadlines:
  - connect timeout default `15s`
  - read timeout default `30s`
  - ping interval default `30s` sending `{"method":"ping"}`
  - watchdog stale default `10s`, tick `200ms`
- Evidence: `Cmd: C30,C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:10-24`, `paraphina/src/live/connectors/hyperliquid.rs:40-55`, `paraphina/src/live/connectors/hyperliquid.rs:536-543`, `paraphina/src/live/connectors/hyperliquid.rs:551-561`, `paraphina/src/live/connectors/hyperliquid.rs:623-627`.

- Reconnect policy:
  - infinite loop with exponential backoff + jitter; healthy-session reset.
  - session timeout default `24h`.
  - endpoint probe/rotation after repeated failures.
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:406-493`, `paraphina/src/live/connectors/hyperliquid.rs:433-441`, `paraphina/src/live/connectors/hyperliquid.rs:418-428`.

- Can close healthy connections if stale watchdog trips due missing publishable book events for `stale_ms`.
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:619-622`.

#### B) Message Handling and Parsing

- Parse path: text/binary UTF-8 -> `serde_json::from_str` -> channel dispatch.
- `subscriptionResponse` and `pong` ignored.
- Evidence: `Cmd: C33,C34`; `Code: paraphina/src/live/connectors/hyperliquid.rs:646-667`, `paraphina/src/live/connectors/hyperliquid.rs:676-679`.

- Book decode path: resilient snapshot decode (`decode_l2book_snapshot_resilient`) + fallback parser.
- Evidence: `Cmd: C34,C35`; `Code: paraphina/src/live/connectors/hyperliquid.rs:703-709`, `paraphina/src/live/connectors/hyperliquid.rs:1655-1777`, `paraphina/src/live/connectors/hyperliquid.rs:785-795`.

- No compression/decompression stage in connector parse path.
- Evidence: `Cmd: C77`; `Code: paraphina/src/live/connectors/hyperliquid.rs` (no matching decompression symbols).

#### C) Internal Buffering / Backpressure

- Connector-local internal queue: `mpsc(256)` + `pending_latest` overwrite on full (`try_send` full path).
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:13`, `paraphina/src/live/connectors/hyperliquid.rs:567-595`.

- Bootstrap delta buffer until baseline snapshot: bounded deque `1024`; overflow clears and triggers refresh snapshot path.
- Evidence: `Cmd: C34`; `Code: paraphina/src/live/connectors/hyperliquid.rs:14`, `paraphina/src/live/connectors/hyperliquid.rs:609`, `paraphina/src/live/connectors/hyperliquid.rs:812-817`.

- Downstream queues on path to runner: connector channel `1024` -> `market_ingest_tx(1024)` -> `market_tx(1024)`.
- Evidence: `Cmd: C21,C28`; `Code: paraphina/src/bin/paraphina_live.rs:1743-1746`, `paraphina/src/bin/paraphina_live.rs:1805`, `paraphina/src/bin/paraphina_live.rs:955-960`.

#### D) Apply-to-State and Publishing

- Publishing: forward task sends events; `last_published_ns` set on successful send.
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:572-584`.

- State apply: runner calls `apply_l2_snapshot/delta`; these write `last_mid_update_ms`.
- Evidence: `Cmd: C32,C05`; `Code: paraphina/src/live/runner.rs:2361-2384`, `paraphina/src/state.rs:315`, `paraphina/src/state.rs:341`.

- Freshness advancement in connector watchdog anchor occurs on publishable book events (`last_book_event_ns`), not heartbeat/acks.
- Evidence: `Cmd: C34,C38`; `Code: paraphina/src/live/connectors/hyperliquid.rs:733-736`, `paraphina/src/live/connectors/hyperliquid.rs:677-679`, `paraphina/src/live/connectors/hyperliquid.rs:2403-2436`.

#### E) Venue-Specific Notes

- Hyperliquid has an additional connector-local REST book fallback (`15s` stale threshold, poll every `2s`) independent of Layer B monitor.
- Evidence: `Cmd: C79`; `Code: paraphina/src/live/connectors/hyperliquid.rs:974-987`, `paraphina/src/live/connectors/hyperliquid.rs:1003-1033`.

- Layer A/B also configured at app level for Hyperliquid.
- Evidence: `Cmd: C23,C87`; `Code: paraphina/src/bin/paraphina_live.rs:1975-2010`, `paraphina/src/bin/paraphina_live.rs:2691-2721`.

### 4.2 `lighter`

#### A) Connectivity Architecture

- WS stack/library: `tokio_tungstenite`.
- Evidence: `Cmd: C39`; `Code: paraphina/src/live/connectors/lighter.rs:111`.

- Single public socket with subscribe channel `order_book/<market_id>`.
- Evidence: `Cmd: C41`; `Code: paraphina/src/live/connectors/lighter.rs:607-613`.

- Heartbeat/deadlines:
  - connect timeout `15s`, read timeout `30s`, stale watchdog `10s` tick `200ms`.
  - no outbound periodic ping timer in connector.
  - responds to WS `Ping/Pong` and JSON `ping`.
- Evidence: `Cmd: C39,C41,C42,C89`; `Code: paraphina/src/live/connectors/lighter.rs:10-14`, `paraphina/src/live/connectors/lighter.rs:33-48`, `paraphina/src/live/connectors/lighter.rs:578-590`, `paraphina/src/live/connectors/lighter.rs:667-672`, `paraphina/src/live/connectors/lighter.rs:726-732`.

- Reconnect policy:
  - session timeout default `24h`, exponential backoff (no jitter), healthy reset.
  - subscribe failure invalid channel triggers reconnect path.
- Evidence: `Cmd: C41`; `Code: paraphina/src/live/connectors/lighter.rs:492-500`, `paraphina/src/live/connectors/lighter.rs:538-560`, `paraphina/src/live/connectors/lighter.rs:734-739`.

#### B) Message Handling and Parsing

- Parse path: message -> JSON -> decode top + L2 decode.
- Initial message decoded as snapshot, subsequent as deltas.
- Evidence: `Cmd: C42`; `Code: paraphina/src/live/connectors/lighter.rs:702-712`, `paraphina/src/live/connectors/lighter.rs:786-810`.

- Decode-failure policy: after 10 consecutive delta decode failures, reconnect for fresh snapshot.
- Evidence: `Cmd: C42`; `Code: paraphina/src/live/connectors/lighter.rs:17`, `paraphina/src/live/connectors/lighter.rs:851-858`.

- Sequence policy: tracker only rejects strictly older seq (`msg.seq < prev`); equal seq allowed.
- Evidence: `Cmd: C43`; `Code: paraphina/src/live/connectors/lighter.rs:1152-1161`.

#### C) Internal Buffering / Backpressure

- Uses shared `MarketPublisher`:
  - queue cap `256`, drain `64`.
  - configured lossless only for `L2Delta`; snapshots are on lossy `try_send` path.
- Evidence: `Cmd: C39,C40,C63`; `Code: paraphina/src/live/connectors/lighter.rs:8-9`, `paraphina/src/live/connectors/lighter.rs:272-279`, `paraphina/src/live/market_publisher.rs:101-114`.

- Additional pipeline queues: connector `mpsc(1024)` -> ingest `mpsc(1024)` -> runner.
- Evidence: `Cmd: C21,C28`; `Code: paraphina/src/bin/paraphina_live.rs:1743-1746`, `paraphina/src/bin/paraphina_live.rs:1805`, `paraphina/src/bin/paraphina_live.rs:955-960`.

#### D) Apply-to-State and Publishing

- On decoded book event: updates `last_book_event_ns`, publishes market event, sets `last_published_ns` on success.
- Evidence: `Cmd: C42`; `Code: paraphina/src/live/connectors/lighter.rs:828-844`.

- Freshness in state advances only when runner applies L2 snapshot/delta and derives mid/spread.
- Evidence: `Cmd: C32,C05`; `Code: paraphina/src/live/runner.rs:2361-2384`, `paraphina/src/state.rs:312-316`, `paraphina/src/state.rs:338-342`.

- Timestamp source for L2 events is payload `timestamp/ts` with zero fallback.
- Evidence: `Cmd: C43`; `Code: paraphina/src/live/connectors/lighter.rs:1177-1181`, `paraphina/src/live/connectors/lighter.rs:1301-1305`, `paraphina/src/live/connectors/lighter.rs:1444-1448`.

#### E) Venue-Specific Notes

- No Layer A enforcer slot and no Layer B REST monitor entry are wired for Lighter.
- Evidence: `Cmd: C24,C87`; `Code: paraphina/src/bin/paraphina_live.rs:2072-2185`, `paraphina/src/bin/paraphina_live.rs:1975`, `paraphina/src/bin/paraphina_live.rs:1996`, `paraphina/src/bin/paraphina_live.rs:2263`, `paraphina/src/bin/paraphina_live.rs:2287`, `paraphina/src/bin/paraphina_live.rs:2426`, `paraphina/src/bin/paraphina_live.rs:2450`, `paraphina/src/bin/paraphina_live.rs:2591`, `paraphina/src/bin/paraphina_live.rs:2614`.

### 4.3 `extended`

#### A) Connectivity Architecture

- WS stack/library: `tokio_tungstenite`.
- Evidence: `Cmd: C46`; `Code: paraphina/src/live/connectors/extended.rs:89-92`.

- Public market stream URL includes orderbook path with `depth=1`.
- Evidence: `Cmd: C46`; `Code: paraphina/src/live/connectors/extended.rs:161-166`.

- Heartbeat/deadlines:
  - connect timeout `15s`, read timeout `10s`, ping interval default `30s`.
  - stale watchdog default `10s`, tick `200ms`.
- Evidence: `Cmd: C46,C48`; `Code: paraphina/src/live/connectors/extended.rs:12-13`, `paraphina/src/live/connectors/extended.rs:407-410`, `paraphina/src/live/connectors/extended.rs:430-437`, `paraphina/src/live/connectors/extended.rs:445-452`, `paraphina/src/live/connectors/extended.rs:471`.

- Reconnect policy:
  - session timeout `24h`, exponential backoff no jitter.
  - parse errors >25 reconnect.
  - sequence mismatch/gap errors from `ExtendedSeqState::apply_update` bubble via `?` and reconnect.
- Evidence: `Cmd: C47,C49,C50`; `Code: paraphina/src/live/connectors/extended.rs:252-261`, `paraphina/src/live/connectors/extended.rs:286-308`, `paraphina/src/live/connectors/extended.rs:540-545`, `paraphina/src/live/connectors/extended.rs:632`, `paraphina/src/live/connectors/extended.rs:789`, `paraphina/src/live/connectors/extended.rs:1352-1374`.

#### B) Message Handling and Parsing

- Parse path: cleaned text/binary -> JSON -> `parse_depth_update` or WS snapshot decode.
- Evidence: `Cmd: C48,C49,C50`; `Code: paraphina/src/live/connectors/extended.rs:496-503`, `paraphina/src/live/connectors/extended.rs:527-550`, `paraphina/src/live/connectors/extended.rs:552-557`, `paraphina/src/live/connectors/extended.rs:684-708`, `paraphina/src/live/connectors/extended.rs:1432-1477`, `paraphina/src/live/connectors/extended.rs:1550-1596`.

- No decompression stage in connector.
- Evidence: `Cmd: C77`; `Code: paraphina/src/live/connectors/extended.rs` (no matching decompression symbols).

#### C) Internal Buffering / Backpressure

- MarketPublisher queue cap live `256` (fixture `4096`), drain `64`.
- Lossless only for `L2Delta`; snapshots can be coalesced/dropped under pressure.
- Evidence: `Cmd: C46,C47,C63`; `Code: paraphina/src/live/connectors/extended.rs:14-16`, `paraphina/src/live/connectors/extended.rs:200-217`, `paraphina/src/live/market_publisher.rs:101-114`.

- Recorder path (`record_ws_frame`) introduces additional mutex/file I/O when enabled.
- Evidence: `Cmd: C49`; `Code: paraphina/src/live/connectors/extended.rs:523-526`, `paraphina/src/live/connectors/extended.rs:680-683`.

#### D) Apply-to-State and Publishing

- Connector freshness advances on parsed/publishable book events; publisher callback advances `last_published_ns`.
- Evidence: `Cmd: C47,C49`; `Code: paraphina/src/live/connectors/extended.rs:206-210`, `paraphina/src/live/connectors/extended.rs:563-568`, `paraphina/src/live/connectors/extended.rs:635-637`, `paraphina/src/live/connectors/extended.rs:793-794`.

- State apply/freshness advancement remains centralized in runner/state.
- Evidence: `Cmd: C32,C05`; `Code: paraphina/src/live/runner.rs:2361-2384`, `paraphina/src/state.rs:312-316`, `paraphina/src/state.rs:338-342`.

- Timestamp source is mixed local/exchange:
  - boot snapshot stamped `now_ms()`
  - deltas use `event_time.unwrap_or_else(now_ms)`
  - WS snapshot uses `E/ts/...` fallback `now_ms`
- Evidence: `Cmd: C48,C50`; `Code: paraphina/src/live/connectors/extended.rs:390-395`, `paraphina/src/live/connectors/extended.rs:1385`, `paraphina/src/live/connectors/extended.rs:1574-1579`.

#### E) Venue-Specific Notes

- Layer A/B wired in app bootstrap.
- Evidence: `Cmd: C20,C23,C87`; `Code: paraphina/src/bin/paraphina_live.rs:2263-2301`, `paraphina/src/bin/paraphina_live.rs:2691-2721`.

### 4.4 `aster`

#### A) Connectivity Architecture

- WS stack/library: `tokio_tungstenite`.
- Evidence: `Cmd: C51`; `Code: paraphina/src/live/connectors/aster.rs:21`.

- URL-path stream model, no JSON subscribe: `<symbol>@depth@100ms`.
- Evidence: `Cmd: C53`; `Code: paraphina/src/live/connectors/aster.rs:341-350`.

- Heartbeat/deadlines:
  - connect timeout `15s`, read timeout `30s` (steady loop), ping interval `30s` default.
  - stale watchdog `10s` default, tick `200ms`.
  - additional internal apply-latency watchdog: `STALE_MS=2000`, cooldown `7000ms`.
- Evidence: `Cmd: C51,C53,C54`; `Code: paraphina/src/live/connectors/aster.rs:45-47`, `paraphina/src/live/connectors/aster.rs:352-358`, `paraphina/src/live/connectors/aster.rs:387-394`, `paraphina/src/live/connectors/aster.rs:402-421`, `paraphina/src/live/connectors/aster.rs:383-386`, `paraphina/src/live/connectors/aster.rs:586-606`, `paraphina/src/live/connectors/aster.rs:740-748`, `paraphina/src/live/connectors/aster.rs:904-923`.

- Reconnect policy:
  - session timeout `24h`, backoff no jitter.
  - ping send failure/read timeout/stale watchdog bails reconnect.
  - many sequence issues trigger in-session resnapshot/resync rather than immediate reconnect.
- Evidence: `Cmd: C52,C53,C54,C55`; `Code: paraphina/src/live/connectors/aster.rs:243-313`, `paraphina/src/live/connectors/aster.rs:733-737`, `paraphina/src/live/connectors/aster.rs:740-748`, `paraphina/src/live/connectors/aster.rs:429-431`, `paraphina/src/live/connectors/aster.rs:695-707`, `paraphina/src/live/connectors/aster.rs:803-818`.

#### B) Message Handling and Parsing

- Parse path: text/binary -> `parse_depth_update` (`depthUpdate`) or snapshot fetch path.
- Sequence decisions via lenient bridge logic (`SeqDecision`).
- Evidence: `Cmd: C54,C55`; `Code: paraphina/src/live/connectors/aster.rs:654-671`, `paraphina/src/live/connectors/aster.rs:766-787`, `paraphina/src/live/connectors/aster.rs:1594-1619`, `paraphina/src/live/connectors/aster.rs:1469-1503`.

- No decompression stage.
- Evidence: `Cmd: C77`; `Code: paraphina/src/live/connectors/aster.rs` (no matching decompression symbols).

#### C) Internal Buffering / Backpressure

- MarketPublisher queue cap live `256` (fixture `4096`), drain `64`.
- Lossless for both `L2Delta` and `L2Snapshot`.
- Additional connector buffer `buffered_updates` max `1024`; overflow triggers resync.
- Evidence: `Cmd: C51,C52,C54,C63`; `Code: paraphina/src/live/connectors/aster.rs:47-49`, `paraphina/src/live/connectors/aster.rs:206-216`, `paraphina/src/live/connectors/aster.rs:362`, `paraphina/src/live/connectors/aster.rs:711-719`, `paraphina/src/live/market_publisher.rs:101-114`.

#### D) Apply-to-State and Publishing

- Snapshot and delta apply logic updates connector freshness on parsed book events.
- Evidence: `Cmd: C53,C54`; `Code: paraphina/src/live/connectors/aster.rs:485-491`, `paraphina/src/live/connectors/aster.rs:666-672`, `paraphina/src/live/connectors/aster.rs:781-787`, `paraphina/src/live/connectors/aster.rs:845-850`.

- Timestamp source mixed:
  - REST snapshot event stamped local `now_ms()`
  - deltas stamped `event_time.unwrap_or_else(now_ms)`
- Evidence: `Cmd: C53,C55`; `Code: paraphina/src/live/connectors/aster.rs:475-481`, `paraphina/src/live/connectors/aster.rs:1458`, `paraphina/src/live/connectors/aster.rs:1518`, `paraphina/src/live/connectors/aster.rs:1605-1608`.

#### E) Venue-Specific Notes

- Layer A/B wired.
- Evidence: `Cmd: C21,C23,C87`; `Code: paraphina/src/bin/paraphina_live.rs:2426-2464`, `paraphina/src/bin/paraphina_live.rs:2691-2721`.

- Aggressive 2s stale watchdog may cause frequent resnapshot churn on transient pauses.
- Evidence: `Cmd: C53,C54`; `Code: paraphina/src/live/connectors/aster.rs:383-386`, `paraphina/src/live/connectors/aster.rs:586-606`, `paraphina/src/live/connectors/aster.rs:904-923`.

### 4.5 `paradex`

#### A) Connectivity Architecture

- WS stack/library: `tokio_tungstenite`.
- Evidence: `Cmd: C56`; `Code: paraphina/src/live/connectors/paradex.rs:83`.

- Single public WS; subscribes `bbo.<market>` via JSON-RPC (`id=1`).
- Evidence: `Cmd: C57,C61`; `Code: paraphina/src/live/connectors/paradex.rs:332-338`, `paraphina/src/live/connectors/paradex.rs:1377-1389`.

- Heartbeat/deadlines:
  - connect timeout `15s`, read timeout `30s`.
  - stale watchdog `10s` tick `200ms`.
  - ping timer defined but no periodic send branch; only responds to inbound ping.
- Evidence: `Cmd: C56,C58,C62`; `Code: paraphina/src/live/connectors/paradex.rs:12-14`, `paraphina/src/live/connectors/paradex.rs:322-325`, `paraphina/src/live/connectors/paradex.rs:391-399`, `paraphina/src/live/connectors/paradex.rs:349-356`, `paraphina/src/live/connectors/paradex.rs:385-405`, `paraphina/src/live/connectors/paradex.rs:424-425`.

- Reconnect policy:
  - session timeout `24h`, exponential backoff no jitter.
  - stale timeout/read timeout/subscribe error reconnect.
- Evidence: `Cmd: C57,C58`; `Code: paraphina/src/live/connectors/paradex.rs:224-294`, `paraphina/src/live/connectors/paradex.rs:389-400`, `paraphina/src/live/connectors/paradex.rs:465-471`.

#### B) Message Handling and Parsing

- Parse path: JSON -> BBO decode and optional orderbook channel parse.
- Orderbook tracker enforces seq consistency and can error on mismatch/gap.
- Evidence: `Cmd: C58,C59,C60`; `Code: paraphina/src/live/connectors/paradex.rs:441-451`, `paraphina/src/live/connectors/paradex.rs:478-484`, `paraphina/src/live/connectors/paradex.rs:531`, `paraphina/src/live/connectors/paradex.rs:801-839`, `paraphina/src/live/connectors/paradex.rs:1169-1201`.

- No decompression stage.
- Evidence: `Cmd: C77`; `Code: paraphina/src/live/connectors/paradex.rs` (no matching decompression symbols).

#### C) Internal Buffering / Backpressure

- MarketPublisher queue cap `256`, drain `64`, lossless for both snapshots and deltas.
- Evidence: `Cmd: C56,C57,C63`; `Code: paraphina/src/live/connectors/paradex.rs:14-15`, `paraphina/src/live/connectors/paradex.rs:179-188`, `paraphina/src/live/market_publisher.rs:101-114`.

#### D) Apply-to-State and Publishing

- BBO decode emits synthetic `L2Snapshot`; connector freshness updates `last_book_event_ns` there.
- Orderbook parse path updates `last_parsed_ns`; publish callback updates `last_published_ns` globally.
- Evidence: `Cmd: C58,C57`; `Code: paraphina/src/live/connectors/paradex.rs:478-503`, `paraphina/src/live/connectors/paradex.rs:531-539`, `paraphina/src/live/connectors/paradex.rs:173-177`.

- Timestamp source:
  - BBO synthetic snapshot: local `now_ms()`.
  - orderbook delta: local `now_ms()`.
  - orderbook snapshot: `ts/timestamp` fallback now.
- Evidence: `Cmd: C60`; `Code: paraphina/src/live/connectors/paradex.rs:1325-1341`, `paraphina/src/live/connectors/paradex.rs:1244`, `paraphina/src/live/connectors/paradex.rs:1213-1219`.

#### E) Venue-Specific Notes

- Layer A/B wired.
- Evidence: `Cmd: C22,C23,C87`; `Code: paraphina/src/bin/paraphina_live.rs:2591-2628`, `paraphina/src/bin/paraphina_live.rs:2691-2721`.

- Because subscribe path is BBO-only, age may track BBO cadence/local arrival rather than full orderbook cadence.
- Evidence: `Cmd: C57,C61,C60`; `Code: paraphina/src/live/connectors/paradex.rs:332-338`, `paraphina/src/live/connectors/paradex.rs:1377-1389`, `paraphina/src/live/connectors/paradex.rs:1286-1345`.

## 5) Cross-Venue Connectivity Architecture Comparison (Matrix)

| Dimension | hyperliquid | lighter | extended | aster | paradex |
|---|---|---|---|---|---|
| WS library | `tokio_tungstenite` (`paraphina/src/live/connectors/hyperliquid.rs:127`) | `tokio_tungstenite` (`paraphina/src/live/connectors/lighter.rs:111`) | `tokio_tungstenite` (`paraphina/src/live/connectors/extended.rs:89-92`) | `tokio_tungstenite` (`paraphina/src/live/connectors/aster.rs:21`) | `tokio_tungstenite` (`paraphina/src/live/connectors/paradex.rs:83`) |
| Public market topology | single WS + `l2Book` subscribe | single WS + `order_book/<id>` subscribe | single WS URL path `orderbooks/<market>?depth=1` | single WS URL path `<symbol>@depth@100ms` | single WS JSON-RPC subscribe `bbo.<market>` |
| Heartbeat/deadlines | connect 15s/read 30s/ping 30s/stale 10s | connect 15s/read 30s/no outbound ping/stale 10s | connect 15s/read 10s/ping 30s/stale 10s | connect 15s/read 30s/ping 30s/stale 10s + internal 2s stale watchdog | connect 15s/read 30s/ping timer defined but unused/stale 10s |
| Reconnect/backoff | exp backoff + jitter + endpoint rotation/probe | exp backoff (no jitter), subscribe-fail path, delta-fail path | exp backoff (no jitter), parse-fail and seq-fail reconnect | exp backoff (no jitter), seq gaps mostly in-session resnapshot | exp backoff (no jitter), subscribe/read/stale reconnect |
| Snapshot+delta/seq | tracker detects gaps, refresh snapshot; resilient snapshot decode | first snapshot then delta mode; seq tracker only drops older seq | strict seq gap/mismatch errors from `ExtendedSeqState` | lenient seq bridge + buffered updates + resnapshot loop | strict seq checks in `ParadexSeqState` for orderbook path; BBO path synthetic snapshot |
| Internal buffering | custom `mpsc(256)` + single-slot overflow coalesce | `MarketPublisher` 256/64; only delta lossless | `MarketPublisher` 256/64 (4096 fixture); only delta lossless | `MarketPublisher` 256/64 (4096 fixture); snapshot+delta lossless | `MarketPublisher` 256/64; snapshot+delta lossless |
| Apply model | runner canonical event drain+sort+apply | same shared runner path | same shared runner path | same shared runner path | same shared runner path |
| Freshness source for `age_ms` | exchange `time` in WS/REST paths | payload `timestamp/ts` (0 fallback) | mixed exchange/local | mixed exchange/local | local-heavy (`now_ms` for BBO/delta) |

Matrix evidence commands:
- `Cmd: C30,C39,C46,C51,C56` (libraries/constants)
- `Cmd: C33,C41,C48,C53,C57,C61` (subscribe/topology)
- `Cmd: C33,C41,C48,C53,C58,C62,C89` (heartbeat/timeouts)
- `Cmd: C33,C41,C47,C52,C57,C49,C54,C58` (reconnect)
- `Cmd: C34,C42,C50,C55,C59,C60` (seq + timestamp)
- `Cmd: C33,C40,C47,C52,C57,C63,C10,C85` (buffering/coalescing)
- `Code` refs are in Section 4 subsections.

Conclusion:
- **Architectures differ materially.**

Top 3 divergences most likely to explain p95/p99 `age_ms` gaps (ranked):

1. Mixed timestamp source semantics (exchange vs local) by venue and code path.
- Evidence: `Cmd: C35,C43,C50,C55,C60`; `Code: cited in Section 1.2`.

2. Snapshot losslessness is inconsistent (`lighter`/`extended` lossy snapshots vs `aster`/`paradex` lossless; `hyperliquid` custom lossy coalescing), plus runner coalescing drops under cap.
- Evidence: `Cmd: C33,C40,C47,C52,C57,C63,C10,C85`; `Code: cited in Section 4C`.

3. Recovery/supervision asymmetry (`lighter` lacks Layer A/B; `hyperliquid` has extra REST fallback; `paradex` ping timer not wired to send).
- Evidence: `Cmd: C79,C87,C57,C62,C89`; `Code: cited in Sections 4.1E, 4.2E, 4.5A`.

## 6) Why `age_ms` Differs Across Venues (Root-Cause Partition)

### `[EXTERNAL]` venue/feed-side contributors

- Different subscribed feed product and cadence class (`l2Book`, `order_book`, `depth=1`, `@depth@100ms`, `bbo`) means upstream update timing differs before any Paraphina logic.
- Evidence: `Cmd: C33,C41,C48,C53,C57`; `Code: paraphina/src/live/connectors/hyperliquid.rs:518-527`, `paraphina/src/live/connectors/lighter.rs:607-612`, `paraphina/src/live/connectors/extended.rs:161-166`, `paraphina/src/live/connectors/aster.rs:345-350`, `paraphina/src/live/connectors/paradex.rs:332-338`.

- Paradex BBO-only subscription likely has different cadence than full depth channels.
- Evidence: `Cmd: C57,C61`; `Code: paraphina/src/live/connectors/paradex.rs:332-338`, `paraphina/src/live/connectors/paradex.rs:1377-1389`.

- Liquidity/cadence effects by symbol/venue are **hypothesis only** in this phase (no live soak telemetry analyzed here).
- Evidence gap: no runtime per-venue event-rate logs were collected in this read-only pass.

### `[INTERNAL]` Paraphina-side contributors

- `age_ms` computes from `last_mid_update_ms` (mid-derived), not `last_book_update_ms`.
- If book updates do not produce valid `(mid,spread)`, freshness may stall.
- Evidence: `Cmd: C05,C66`; `Code: paraphina/src/telemetry.rs:1640`, `paraphina/src/state.rs:312-316`, `paraphina/src/state.rs:338-342`, `paraphina/src/state.rs:310`, `paraphina/src/state.rs:336`.

- Queue/backpressure policy differs and can suppress or coalesce events differently by venue.
- Evidence: `Cmd: C33,C40,C47,C52,C57,C63`; `Code: Section 4C refs`.

- Runner coalescing and optional cap can drop unready-venue deltas; min inter-tick gating can delay apply under burst.
- Evidence: `Cmd: C10,C11,C84,C85`; `Code: paraphina/src/live/runner.rs:563-579`, `paraphina/src/live/runner.rs:1086-1091`, `paraphina/src/live/runner.rs:1112-1115`, `paraphina/src/live/runner.rs:1300-1330`.

- Cache apply failures block state freshness updates (`apply_market_event_to_core` runs only on cache-OK path).
- Evidence: `Cmd: C82,C83,C84`; `Code: paraphina/src/live/state_cache.rs:65-73`, `paraphina/src/live/state_cache.rs:90-99`, `paraphina/src/live/runner.rs:1344-1354`.

- Recovery asymmetry for Lighter (no Layer A/B wiring) increases stale persistence risk.
- Evidence: `Cmd: C87`; `Code: paraphina/src/bin/paraphina_live.rs:2072-2185`, `paraphina/src/bin/paraphina_live.rs:1975`, `paraphina/src/bin/paraphina_live.rs:1996`, `paraphina/src/bin/paraphina_live.rs:2263`, `paraphina/src/bin/paraphina_live.rs:2287`, `paraphina/src/bin/paraphina_live.rs:2426`, `paraphina/src/bin/paraphina_live.rs:2450`, `paraphina/src/bin/paraphina_live.rs:2591`, `paraphina/src/bin/paraphina_live.rs:2614`.

- Paradex outbound ping timer appears unimplemented in select loop, increasing idle disconnect risk.
- Evidence: `Cmd: C62`; `Code: paraphina/src/live/connectors/paradex.rs:356`, `paraphina/src/live/connectors/paradex.rs:385-405`, `paraphina/src/live/connectors/paradex.rs:424-425`.

### `[MIXED]` external + internal interaction

- Exchange timestamp usage plus local clock clamping (`max(now-ts,0)`) can hide or amplify perceived staleness depending venue clock skew and timestamp source.
- Evidence: `Cmd: C05,C35,C43,C50,C55,C60`; `Code: paraphina/src/telemetry.rs:832-835` + connector timestamp-source refs in Section 1.2.

## 7) Reconnect Thrash + Staleness Failure Modes (Per Venue)

### `hyperliquid`

- Disconnect/reconnect triggers: stale watchdog expiry, ping send failure, read timeout, session timeout.
- Evidence: `Cmd: C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:619-627`, `paraphina/src/live/connectors/hyperliquid.rs:631-639`, `paraphina/src/live/connectors/hyperliquid.rs:433-441`.

- Flap risk: fixed stale threshold (`10s`) + lossy internal queue can cause repeated reconnects under prolonged non-publishable periods.
- Evidence: `Cmd: C30,C33`; `Code: paraphina/src/live/connectors/hyperliquid.rs:10`, `paraphina/src/live/connectors/hyperliquid.rs:567-595`, `paraphina/src/live/connectors/hyperliquid.rs:619-622`.

### `lighter`

- Triggers: stale watchdog, read timeout, subscribe invalid channel, consecutive delta decode failures, session timeout.
- Evidence: `Cmd: C41,C42`; `Code: paraphina/src/live/connectors/lighter.rs:634-645`, `paraphina/src/live/connectors/lighter.rs:734-739`, `paraphina/src/live/connectors/lighter.rs:851-858`, `paraphina/src/live/connectors/lighter.rs:492-500`.

- Flap risk: missing Layer A/B plus no outbound ping; repeated decode-fail reconnect loops possible.
- Evidence: `Cmd: C89,C87,C42`; `Code: paraphina/src/live/connectors/lighter.rs:667-672`, `paraphina/src/live/connectors/lighter.rs:726-732`, `paraphina/src/bin/paraphina_live.rs:2072-2185`, `paraphina/src/live/connectors/lighter.rs:851-858`.

### `extended`

- Triggers: stale watchdog, ping failure, parse error flood (`>25`), seq mismatch/gap errors, connect timeout/session timeout.
- Evidence: `Cmd: C48,C49,C50`; `Code: paraphina/src/live/connectors/extended.rs:451-452`, `paraphina/src/live/connectors/extended.rs:464-468`, `paraphina/src/live/connectors/extended.rs:540-545`, `paraphina/src/live/connectors/extended.rs:632`, `paraphina/src/live/connectors/extended.rs:789`, `paraphina/src/live/connectors/extended.rs:407-410`, `paraphina/src/live/connectors/extended.rs:252-261`.

- Flap risk: strict seq checks + lossy snapshot policy can force reconnect when baseline alignment is disturbed.
- Evidence: `Cmd: C47,C50,C63`; `Code: paraphina/src/live/connectors/extended.rs:216`, `paraphina/src/live/connectors/extended.rs:1352-1374`, `paraphina/src/live/market_publisher.rs:108-114`.

### `aster`

- Triggers: stale watchdog (`10s`), ping failure, read timeout (`30s`) in steady loop, plus aggressive internal `2s` stale resnapshot loops.
- Evidence: `Cmd: C53,C54`; `Code: paraphina/src/live/connectors/aster.rs:417-421`, `paraphina/src/live/connectors/aster.rs:733-737`, `paraphina/src/live/connectors/aster.rs:740-748`, `paraphina/src/live/connectors/aster.rs:383-386`, `paraphina/src/live/connectors/aster.rs:586-606`, `paraphina/src/live/connectors/aster.rs:904-923`.

- Flap risk: dual watchdog model can overreact to transient pauses, producing repeated resync cycles.
- Evidence: `Cmd: C53,C54`; `Code: paraphina/src/live/connectors/aster.rs:383-386`, `paraphina/src/live/connectors/aster.rs:596-604`, `paraphina/src/live/connectors/aster.rs:914-921`.

### `paradex`

- Triggers: stale watchdog, read timeout, subscribe error, seq mismatch/gap in orderbook parser, session timeout.
- Evidence: `Cmd: C58,C59`; `Code: paraphina/src/live/connectors/paradex.rs:389-400`, `paraphina/src/live/connectors/paradex.rs:465-471`, `paraphina/src/live/connectors/paradex.rs:531`, `paraphina/src/live/connectors/paradex.rs:823-839`, `paraphina/src/live/connectors/paradex.rs:246-247`.

- Flap risk: no outbound ping timer branch may increase idle disconnects; BBO-only subscription may keep minimal updates while full-depth parser remains mostly idle.
- Evidence: `Cmd: C57,C62`; `Code: paraphina/src/live/connectors/paradex.rs:332-338`, `paraphina/src/live/connectors/paradex.rs:349-356`, `paraphina/src/live/connectors/paradex.rs:385-405`.

## 8) Recommendations: Top 3-5 Minimal Fixes (No Code in Phase 1)

1. Standardize freshness basis and expose dual metrics.
- Change: track both `last_apply_update_ms` (local) and `last_event_ts_ms` (exchange), compute separate ages.
- Expected impact: major reduction in cross-venue p95/p99 interpretation error.
- Measure: compare variance of `age_local_apply_ms` vs current `age_ms` across venues during same soak.
- Risk: dashboard/alerts need migration; guard via dual-publish period.
- Evidence basis: `Cmd: C05,C35,C43,C50,C55,C60`.

2. Make snapshot delivery lossless in all connectors; keep bounded lossy only for explicitly safe classes.
- Change: in connector publisher policy, include `L2Snapshot` in lossless set for Lighter/Extended; add overflow counters for all.
- Expected impact: reduce bootstrap failures and stale tails from dropped snapshots.
- Measure: decrease in cache apply errors and stale plateaus.
- Risk: higher queue pressure; guard with explicit queue metrics and caps.
- Evidence basis: `Cmd: C40,C47,C52,C57,C63,C82,C84`.

3. Align recovery layers: wire Lighter into Layer A/B and keep thresholds venue-tuned.
- Change: add Lighter enforcer/rest monitor entries analogous to other venues.
- Expected impact: fewer persistent stale incidents (p99).
- Measure: reconnect reason distribution and max stale duration per venue.
- Risk: over-restarts if thresholds too low; guard with cooldown and per-venue stale overrides.
- Evidence basis: `Cmd: C87,C16,C17`.

4. Fix Paradex outbound keepalive implementation.
- Change: actually send periodic WS ping on `ping_timer` tick (or remove dead timer if protocol guarantees server pings).
- Expected impact: lower idle disconnects and reconnect gaps.
- Measure: decline in `read timeout` / connection reset reconnect reasons for Paradex.
- Risk: protocol-specific ping expectations; guard with feature/env flag.
- Evidence basis: `Cmd: C62,C58`.

5. Revisit aggressive stale resync thresholds (especially Aster internal `2s`).
- Change: tune with observed cadence and queue latency; avoid resync thrash.
- Expected impact: lower reconnect/resync churn and smoother `age_ms` tails.
- Measure: reduction in resync-trigger count and stale oscillation.
- Risk: too-lax thresholds could delay fault recovery; guard with max bound and alerting.
- Evidence basis: `Cmd: C53,C54`.

## 9) Benchmark / Validation Protocol (Shadow Soak)

Duration:
- 90 minutes continuous (`60-120` acceptable).

Metrics to record:
- `venue_age_ms` p50/p95/p99 per venue.
- reconnect counts by reason (timeout, stale watchdog, parse/seq errors, subscribe errors).
- queue overflow/drop counters per connector queue.
- ingest->apply latency (`runner_now_ms - event.timestamp_ms`) distribution per venue.

Acceptance criteria:
- No venue with continuous stale plateau `>60s`.
- p95 `age_ms` spread between venues reduced vs baseline.
- reconnect flaps (`<10s apart`) reduced by at least 50% on known-problem venues.

Suggested shadow-only command shape (no secrets, no live execution):

```bash
PARAPHINA_LIVE_PREFLIGHT_OK=1 \
PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS=5000 \
PARAPHINA_LIVE_MODE=1 \
cargo run --bin paraphina_live -- \
  --trade-mode shadow \
  --config ./config/default.toml \
  --connectors hyperliquid,lighter,extended,aster,paradex
```

Notes:
- Keep `PARAPHINA_LIVE_EXEC_ENABLE` unset/false.
- If venue auth is absent, run market-data shadow only; audit remains valid for ingestion/staleness behavior.

Telemetry/display evidence for metric field:
- `Cmd: C02,C05`; `Code: paraphina/src/telemetry.rs:1615-1715`, `tools/paraphina_watch.py:622-699`.

## 10) Phase 2: Optional Instrumentation Patch Plan (Default OFF; Do Not Implement)

Only needed to prove remaining hypotheses quantitatively.

Proposed gate:
- `PARAPHINA_WS_AUDIT=1`.

Measurements:
1. ingest->apply latency per venue.
- Insert in runner before/after apply:
  - `paraphina/src/live/runner.rs:1332-1354` (market apply site).
- Record `(now_ms - event_ts_ms)` histograms by venue and event type.

2. Queue depth/overflow/lag per connector.
- Insert in `MarketPublisher`:
  - `paraphina/src/live/market_publisher.rs:40-75` (drain behavior)
  - `paraphina/src/live/market_publisher.rs:108-114` (overflow path)
- Add counters for `try_send_full`, `pending_latest_replaced`, `lossless_send_wait_ms`.

3. Reconnect counts + reasons per venue.
- Insert at reconnect loop error sites:
  - `paraphina/src/live/connectors/hyperliquid.rs:441-463`
  - `paraphina/src/live/connectors/lighter.rs:500-533`
  - `paraphina/src/live/connectors/extended.rs:262-283`
  - `paraphina/src/live/connectors/aster.rs:266-287`
  - `paraphina/src/live/connectors/paradex.rs:247-267`
- Tag reason category (stale, read_timeout, parse, seq_gap, subscribe, ping_fail, session_timeout).

4. Rolling age distribution emit.
- Insert where telemetry age is computed:
  - `paraphina/src/telemetry.rs:1640-1658`
- Emit rolling p50/p95/p99 per venue under audit flag.

Validation plan:
- With `PARAPHINA_WS_AUDIT=0`: byte-for-byte unchanged telemetry schema and negligible overhead.
- With `PARAPHINA_WS_AUDIT=1`: verify new counters increment under synthetic fixture stress and shadow soak.
- Use A/B runs (30 min each) to estimate overhead (<2% CPU target).

## 10.1) Phase 2 Implementation Notes (2026-02-10)

Implemented minimal code fixes for Tasks A-D:
- Lighter market-data timestamp parsing no longer falls back to `0`; invalid/missing/`<=0` timestamps now fall back to local non-zero wall clock in decode paths (`parse_l2_message_value`, `decode_order_book_snapshot`, `decode_order_book_channel_message`, `decode_order_book_channel_delta`).
- Added `PARAPHINA_WS_AUDIT=1` gated Lighter audit counter/log: `lighter_ts_fallback_count`.
- Paradex now sends periodic outbound WS ping frames from the existing ping timer branch; inbound ping->pong behavior remains unchanged.
- Added `PARAPHINA_WS_AUDIT=1` gated Paradex counters/logs: `paradex_ping_sent_count`, `paradex_ping_send_fail_count`.
- Lighter is now wired into Layer A (`enforcer_slots`) and Layer B (`rest_entries`) in `paraphina_live` like other live venues.
- Lighter and Extended `MarketPublisher` now treat both `L2Delta` and `L2Snapshot` as lossless (snapshot no longer lossy under queue pressure).

Audit instrumentation toggle:
- Default is OFF. Enable with:
  - `PARAPHINA_WS_AUDIT=1`
- When OFF, existing telemetry schema/keys are unchanged; only functional fixes apply.

Shadow soak (no live execution, no secrets):
- Run the existing shadow command in Section 9, optionally with audit enabled:

```bash
PARAPHINA_WS_AUDIT=1 \
PARAPHINA_LIVE_PREFLIGHT_OK=1 \
PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS=5000 \
PARAPHINA_LIVE_MODE=1 \
cargo run --bin paraphina_live -- \
  --trade-mode shadow \
  --config ./config/default.toml \
  --connectors hyperliquid,lighter,extended,aster,paradex
```

What to look for:
- No extreme Lighter `age_ms` spikes attributable to `timestamp_ms=0`.
- Paradex audit ping-sent counter increments steadily; ping-send-fail remains zero in healthy sessions.
- Lighter appears in Layer A/B stale-recovery logs similarly to other venues.
- Fewer stale/bootstrap disruptions from dropped snapshots on Lighter/Extended.

## Post-merge status (2026-02-10)

### Implemented (on main)
- Lighter timestamp fallback hardening.
- Lighter + Extended snapshots lossless in `MarketPublisher` path.
- Paradex periodic outbound WS ping + `PARAPHINA_WS_AUDIT` counters.
- Lighter Layer A enforcer slot wiring.
- Startup thrash fix: enforcer ignores `i64::MAX` unknown age until first real update.

### Pending / Open
- `age_ms` semantics normalization across venues (dual metrics: apply-time vs event-time).
- Lighter Layer B REST fallback: still intentionally not wired pending evidence-based endpoint.
- Broader instrumentation/soak metrics (reconnect reasons, rolling p50/p95/p99, queue pressure).

Note: this addendum is intentionally non-code and reflects merged reality.

## 11) Reproducibility Appendix

### 11.1 Required environment + repo-state commands (run first) and output

`Cmd: C01`
```bash
set -e
hostname
pwd
git status -sb
git rev-parse HEAD
git branch --show-current
```
Output:
```text
para
/home/developer/code/paraphina
## audit/ws-age-ms
c21fd55d9ac173294464348c742351caf7b2b05d
audit/ws-age-ms
```

### 11.2 Full command list (in order)

```text
C01  set -e; hostname; pwd; git status -sb; git rev-parse HEAD; git branch --show-current
C02  rg -n "age_ms|venue_age_ms|compute_age_ms|last_mid_update_ms|last_book_update_ms" paraphina/src/telemetry.rs tools/paraphina_watch.py
C03  rg -n "last_mid_update_ms\s*=|apply_l2_snapshot|apply_l2_delta|BookUpdate|seed_dummy_mids|apply_sim_account_snapshots" paraphina/src/state.rs paraphina/src/live/runner.rs paraphina/src/execution_events.rs paraphina/src/engine.rs paraphina/src/bin/paraphina_live.rs
C04  rg -n "seed_dummy_mids|last_mid_update_ms\s*=\s*Some\(now_ms\)|paraphina_bundle" paraphina_bundle.txt paraphina/src -g'*'
C05  nl -ba tools/paraphina_watch.py | sed -n '610,735p'; nl -ba paraphina/src/telemetry.rs | sed -n '820,900p'; nl -ba paraphina/src/telemetry.rs | sed -n '1600,1735p'; nl -ba paraphina/src/state.rs | sed -n '170,360p'; nl -ba paraphina/src/state.rs | sed -n '850,900p'
C06  nl -ba paraphina/src/live/runner.rs | sed -n '520,610p'
C07  nl -ba paraphina/src/live/runner.rs | sed -n '1060,1150p'
C08  nl -ba paraphina/src/live/runner.rs | sed -n '1610,1665p'
C09  nl -ba paraphina/src/live/runner.rs | sed -n '2320,2405p'
C10  nl -ba paraphina/src/live/runner.rs | sed -n '2470,2525p'
C11  nl -ba paraphina/src/live/shared_venue_ages.rs | sed -n '1,120p'
C12  nl -ba paraphina/src/live/venue_health_enforcer.rs | sed -n '1,180p'
C13  nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '1,190p'
C14  nl -ba paraphina/src/execution_events.rs | sed -n '1,80p'
C15  nl -ba paraphina/src/engine.rs | sed -n '80,175p'
C16  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1000,1085p'
C17  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1725,1815p'
C18  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1880,2065p'
C19  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2060,2205p'
C20  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2230,2350p'
C21  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2390,2525p'
C22  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2560,2705p'
C23  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2685,2755p'
C24  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '900,985p'
C25  rg -n "...large connector pattern set..." paraphina/src/live/connectors/hyperliquid.rs
C26  rg -n "...large connector pattern set..." paraphina/src/live/connectors/lighter.rs
C27  rg -n "...large connector pattern set..." paraphina/src/live/connectors/extended.rs
C28  rg -n "...large connector pattern set..." paraphina/src/live/connectors/aster.rs
C29  rg -n "...large connector pattern set..." paraphina/src/live/connectors/paradex.rs
C30  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1,180p'
C31  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '200,340p'
C32  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '390,675p'
C33  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '675,880p'
C34  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1288,1375p'
C35  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1638,1778p'
C36  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1048,1105p'
C37  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1805,1845p'
C38  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '2395,2440p'
C39  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1,140p'
C40  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '250,320p'
C41  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '470,640p'
C42  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '640,930p'
C43  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1130,1465p'
C44  rg -n "fn publish_market|publish_market\(" paraphina/src/live/connectors/lighter.rs
C45  nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '400,420p'
C46  nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1,180p'
C47  nl -ba paraphina/src/live/connectors/extended.rs | sed -n '180,320p'
C48  nl -ba paraphina/src/live/connectors/extended.rs | sed -n '380,520p'
C49  nl -ba paraphina/src/live/connectors/extended.rs | sed -n '520,820p'
C50  nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1330,1605p'
C51  nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1,210p'
C52  nl -ba paraphina/src/live/connectors/aster.rs | sed -n '210,345p'
C53  nl -ba paraphina/src/live/connectors/aster.rs | sed -n '340,520p'
C54  nl -ba paraphina/src/live/connectors/aster.rs | sed -n '520,930p'
C55  nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1400,1625p'
C56  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1,205p'
C57  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '205,360p'
C58  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '360,570p'
C59  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '770,860p'
C60  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1160,1350p'
C61  nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1370,1405p'
C62  rg -n "ping_timer\.tick|Message::Ping\(|Message::Pong\(" paraphina/src/live/connectors/paradex.rs
C63  nl -ba paraphina/src/live/market_publisher.rs | sed -n '1,190p'
C64  nl -ba paraphina/src/live/types.rs | sed -n '1,220p'
C65  rg -n "last_mid_update_ms\s*=" paraphina/src --glob '!**/target/**'
C66  rg -n "last_book_update_ms\s*=" paraphina/src
C67  rg -n "ExecutionEvent::BookUpdate|BookUpdate\(" paraphina/src/live
C68  rg -n "enum ExecutionEvent|BookUpdate" paraphina/src/types.rs
C69  nl -ba paraphina/src/types.rs | sed -n '136,190p'
C70  rg -n "run_live\(|TradeMode::Shadow|trade_mode" paraphina/src/bin/paraphina_live.rs
C71  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2860,2945p'
C72  rg -n "run_validate_config\(|validate-config|validate_config" paraphina/src/bin/paraphina_live.rs
C73  nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1676,1700p'
C74  nl -ba paraphina_bundle.txt | sed -n '5538,5570p'
C75  nl -ba paraphina_bundle.txt | sed -n '7438,7452p'
C76  rg -n "seed_dummy_mids" paraphina/src/live paraphina/src/bin/paraphina_live.rs
C77  rg -n "flate|gzip|zlib|decompress|inflate|MessagePack|simd_json" paraphina/src/live/connectors/{hyperliquid,lighter,extended,aster,paradex}.rs
C78  rg --files | rg -n "telemetry|age_ms|market_rx_stats|ws|audit|\.jsonl$|\.log$"
C79  nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '960,1045p'
C80  rg -n "enforcer_slots\.push|rest_entries\.push|ConnectorArg::Lighter|ConnectorArg::Hyperliquid|ConnectorArg::Extended|ConnectorArg::Aster|ConnectorArg::Paradex" paraphina/src/bin/paraphina_live.rs
C81  nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '145,255p'
C82  rg -n "pub fn apply_market_event|L2Snapshot|L2Delta|seq|gap|stale|Error|enum" paraphina/src/live/state_cache.rs
C83  nl -ba paraphina/src/live/state_cache.rs | sed -n '52,105p'; nl -ba paraphina/src/live/state_cache.rs | sed -n '348,376p'
C84  rg -n "state\.venues|get_mut\(|apply_market_event_to_core|apply_execution_events|pending_events|drain|coalesce|snapshot|MarketDataEvent|account_rx|exec_rx|market_rx" paraphina/src/live/runner.rs; nl -ba paraphina/src/live/runner.rs | sed -n '1150,1425p'; nl -ba paraphina/src/live/runner.rs | sed -n '1425,1615p'
C85  nl -ba paraphina/src/live/runner.rs | sed -n '1014,1040p'
C86  rg -n "pub async fn run_live_loop|run_live_loop\(" paraphina/src/live -g'*.rs'
C87  rg -n "enforcer_slots\.push\(|rest_entries\.push\(" paraphina/src/bin/paraphina_live.rs
C88  ls -l paraphina_bundle.txt
C89  rg -n "ping_timer|PARAPHINA_LIGHTER_PING" paraphina/src/live/connectors/lighter.rs
C90  git status -sb
C91  git diff --stat
C92  git diff --stat --no-index /dev/null docs/INVESTIGATIONS/ws_connectivity_age_ms_audit.md
C93  rg -n '`[A-Za-z0-9_]+\.(rs|py):' docs/INVESTIGATIONS/ws_connectivity_age_ms_audit.md
```

### 11.3 Key `rg` queries and outputs

1. `last_mid_update_ms` write sites (`Cmd: C65`):
- Canonical runtime write sites found in `paraphina/src/state.rs:315`, `paraphina/src/state.rs:341`.
- Additional non-runtime/sim/test writes found in `paraphina/src/execution_events.rs:23`, `paraphina/src/state.rs:881`, RL/test files.

2. `last_book_update_ms` write sites (`Cmd: C66`):
- Only `paraphina/src/state.rs:310`, `paraphina/src/state.rs:336`.

3. `seed_dummy_mids` in live path (`Cmd: C76`):
- Only `paraphina/src/bin/paraphina_live.rs:1053` (validate-config path), no usage in `paraphina/src/live/*`.

4. Layer A/B registrations (`Cmd: C87`):
- push sites exist for Hyperliquid/Extended/Aster/Paradex; no Lighter push lines.

5. Paradex ping timer usage (`Cmd: C62`):
- timer initialized (`paraphina/src/live/connectors/paradex.rs:356`) but no periodic `tick` usage in select; only inbound ping handling (`paraphina/src/live/connectors/paradex.rs:424-425`).

### 11.4 Final git confirmation

```bash
git status -sb
git diff --stat
```

`Cmd: C90` output:
```text
## audit/ws-age-ms
?? docs/INVESTIGATIONS/ws_connectivity_age_ms_audit.md
```

`Cmd: C91` output:
```text
(no output; tracked-file diff is empty because the only change is a new untracked file)
```

Supplemental stat for untracked file (`Cmd: C92`):
```text
.../INVESTIGATIONS/ws_connectivity_age_ms_audit.md | 879 +++++++++++++++++++++
1 file changed, 879 insertions(+)
```
