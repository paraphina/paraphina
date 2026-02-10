# WebSocket Connectivity Pipeline Ultra Audit (Read-Only, Shadow-Only)

Date: 2026-02-10  
Host: `para`  
Repo: `/home/developer/code/paraphina`  
Scope venues: `hyperliquid`, `lighter`, `extended`, `aster`, `paradex`

Method:
- Pass 1: read-only repo/command investigation.
- Pass 2: write this report only.
- No source edits, no branch operations, no commits/pushes in this audit task.

Evidence format used throughout:
- `Cmd: Uxx` references exact command(s) in Section 10.
- `Code:` cites `path:line-line`.

---

## Table of Contents

1. [0) Executive Summary](#0-executive-summary)
2. [1) Phase 0 Baseline Snapshot](#1-phase-0-baseline-snapshot)
3. [2) Global Pipeline Map (Connector -> Runner -> State -> Telemetry -> UI)](#2-global-pipeline-map-connector---runner---state---telemetry---ui)
4. [3) Per-Venue Deep Dives](#3-per-venue-deep-dives)
5. [4) Cross-Venue Comparison Matrix](#4-cross-venue-comparison-matrix)
6. [5) Why `age_ms` Differs (Root-Cause Partition + Ranking)](#5-why-age_ms-differs-root-cause-partition--ranking)
7. [6) Frontier Bottlenecks / Thrash Sources Catalog](#6-frontier-bottlenecks--thrash-sources-catalog)
8. [7) Findings -> Hypotheses -> Measurements (Implementation-Ready, No Code Changes)](#7-findings---hypotheses---measurements-implementation-ready-no-code-changes)
9. [8) Measurement Protocol (Shadow-Only; Commands Only)](#8-measurement-protocol-shadow-only-commands-only)
10. [9) Notes on Merged Reality](#9-notes-on-merged-reality)
11. [10) Reproducibility Appendix (Commands Run, In Order)](#10-reproducibility-appendix-commands-run-in-order)

---

## 0) Executive Summary

Top 5 causes of cross-venue `age_ms` divergence (ranked):

1) `[MIXED]` Timestamp semantics are not uniform, but `age_ms` is computed as one scalar from `last_mid_update_ms`.
- `compute_age_ms(None)=-1`; future timestamps clamp to `0`.
- `last_mid_update_ms` is populated from connector event timestamps, which are venue-specific (exchange time vs local fallback vs local-heavy).
- This makes cross-venue `age_ms` partially non-comparable.  
Evidence: `Cmd: U11,U12,U17,U47,U58,U79,U92,U104,U105`;  
Code: `paraphina/src/telemetry.rs:828-839`, `paraphina/src/state.rs:294-343`, `paraphina/src/live/connectors/hyperliquid.rs:1340-1364`, `paraphina/src/live/connectors/lighter.rs:252-276`, `paraphina/src/live/connectors/extended.rs:1579-1584`, `paraphina/src/live/connectors/aster.rs:1454-1459`, `paraphina/src/live/connectors/paradex.rs:1251-1257`, `paraphina/src/live/connectors/paradex.rs:1282-1289`, `paraphina/src/live/connectors/paradex.rs:1363-1379`.

2) `[EXTERNAL]` Feed products/cadence are materially different by venue.
- Hyperliquid subscribes `l2Book`.
- Lighter subscribes `order_book/<market_id>`.
- Extended uses URL-path stream `orderbooks/<symbol>?depth=1`.
- Aster uses URL-path `<symbol>@depth@100ms`.
- Paradex subscribes `bbo.<market>` (not full orderbook channel at subscribe point).  
Evidence: `Cmd: U46,U67,U71,U88,U101,U123`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:519-527`, `paraphina/src/live/connectors/lighter.rs:652-656`, `paraphina/src/live/connectors/extended.rs:161-166`, `paraphina/src/live/connectors/aster.rs:345-350`, `paraphina/src/live/connectors/paradex.rs:343-349`.

3) `[INTERNAL]` Buffering/coalescing/drop behavior differs across pipeline stages.
- Hyperliquid has connector-local lossy overwrite path (`pending_latest`) on internal queue pressure.
- Lighter/Extended/Aster/Paradex use `MarketPublisher`; non-lossless events use `try_send` with overflow coalescing.
- Runner coalescing can drop unready venue deltas when per-venue cap is hit (`tick_delta_buffer_max`).  
Evidence: `Cmd: U21,U24,U26,U44,U58,U72,U86,U99`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:567-595`, `paraphina/src/live/market_publisher.rs:101-114`, `paraphina/src/live/runner.rs:562-579`, `paraphina/src/live/runner.rs:618-625`, `paraphina/src/live/connectors/lighter.rs:312-322`, `paraphina/src/live/connectors/extended.rs:211-220`, `paraphina/src/live/connectors/aster.rs:206-215`, `paraphina/src/live/connectors/paradex.rs:190-198`.

4) `[INTERNAL]` Recovery architecture is still asymmetric.
- Layer A enforcer wired for all five.
- Layer B REST monitor entries exist for Hyperliquid/Extended/Aster/Paradex.
- Lighter still has no `rest_entries.push(...)` despite captured `ltr_rest_url/ltr_market`.  
Evidence: `Cmd: U34,U35,U36,U37,U38,U40`;  
Code: `paraphina/src/bin/paraphina_live.rs:1975-2015`, `paraphina/src/bin/paraphina_live.rs:2183-2200`, `paraphina/src/bin/paraphina_live.rs:2305-2318`, `paraphina/src/bin/paraphina_live.rs:2470-2483`, `paraphina/src/bin/paraphina_live.rs:2637-2649`, `paraphina/src/bin/paraphina_live.rs:2161-2162`.

5) `[INTERNAL]` Keepalive/reconnect mechanics are not symmetric.
- Hyperliquid uses jittered backoff and JSON ping.
- Extended/Aster/Paradex use outbound WS Ping frame timers.
- Lighter has inbound ping/pong handling and JSON pong response, but no outbound periodic ping timer branch.
- These differences affect disconnect frequency and stale plateaus.  
Evidence: `Cmd: U45,U44,U60,U61,U68,U73,U74,U88,U102,U120,U118`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:487-491`, `paraphina/src/live/connectors/hyperliquid.rs:623-627`, `paraphina/src/live/connectors/lighter.rs:677-689`, `paraphina/src/live/connectors/lighter.rs:712-716`, `paraphina/src/live/connectors/lighter.rs:771-777`, `paraphina/src/live/connectors/extended.rs:435-442`, `paraphina/src/live/connectors/extended.rs:469-474`, `paraphina/src/live/connectors/aster.rs:388-394`, `paraphina/src/live/connectors/paradex.rs:361-367`, `paraphina/src/live/connectors/paradex.rs:402-427`, `paraphina/src/live/connectors/lighter.rs:712-716`.

Highest ROI non-refactor follow-ups:
1. Normalize freshness semantics by publishing dual metrics (`age_apply_ms`, `age_event_ms`) before changing alerting logic.
2. Close Layer B asymmetry for Lighter with evidence-backed endpoint choice.
3. Add standardized reconnect-reason counters + queue-pressure observability in shadow soak.
4. Recalibrate stale thresholds after (1) and (2).

---

## 1) Phase 0 Baseline Snapshot

- `hostname`: `para` (`Cmd: U01`)
- `date -Is`: `2026-02-10T21:27:25+00:00` (`Cmd: U02`)
- `git status -sb`: `## docs/ws-age-ms-postmerge-addendum...origin/docs/ws-age-ms-postmerge-addendum` (`Cmd: U03`)
- `git rev-parse HEAD`: `88e5cb1a981197ab68de8353de7bbc3cec9751a7` (`Cmd: U04`)
- current branch: `docs/ws-age-ms-postmerge-addendum` (`Cmd: U05`)
- toolchain: `ripgrep 14.1.1`, `cargo 1.91.1`, `rustc 1.91.1` (`Cmd: U06`)
- no active `paraphina_live` process (excluding the checking command itself) (`Cmd: U07,U08`)

---

## 2) Global Pipeline Map (Connector -> Runner -> State -> Telemetry -> UI)

### 2.1 UI mapping (`age_ms` column)
- Dashboard reads `record["venue_age_ms"]` and renders column header `age_ms`.  
Evidence: `Cmd: U09,U10`;  
Code: `tools/paraphina_watch.py:622-623`, `tools/paraphina_watch.py:655`, `tools/paraphina_watch.py:685-699`, `tools/paraphina_watch.py:714-719`.

### 2.2 Telemetry age computation
- Telemetry computes per-venue age via `compute_age_ms(effective_now, venue.last_mid_update_ms)`.
- Sentinel behavior: `None -> -1`; future timestamp clamps to `0`.  
Evidence: `Cmd: U11,U12,U13,U14`;  
Code: `paraphina/src/telemetry.rs:828-839`, `paraphina/src/telemetry.rs:1606-1608`, `paraphina/src/telemetry.rs:1640-1651`, `paraphina/src/telemetry.rs:1714`.

### 2.3 State write sites for freshness timestamp
- `last_mid_update_ms` initializes `None`.
- It is updated only when `apply_l2_snapshot/apply_l2_delta` produce valid `(mid, spread)`.  
Evidence: `Cmd: U15,U17,U18`;  
Code: `paraphina/src/state.rs:732`, `paraphina/src/state.rs:294-316`, `paraphina/src/state.rs:321-342`.

### 2.4 Runner ingestion/coalescing topology
- Runner receives `LiveChannels.market_rx/account_rx/exec_rx`.
- Ingress drain coalesces snapshots/deltas when enabled and may drop unready deltas on buffer cap hits.
- Coalescing readiness uses observed snapshots (`saw_l2_snapshot_mask...`).  
Evidence: `Cmd: U23,U24,U26,U27`;  
Code: `paraphina/src/live/runner.rs:75-84`, `paraphina/src/live/runner.rs:465-487`, `paraphina/src/live/runner.rs:562-579`, `paraphina/src/live/runner.rs:618-625`, `paraphina/src/live/runner.rs:777-783`, `paraphina/src/live/runner.rs:1282-1312`.

### 2.5 Shared age store + Layer A/B
- Runner writes `SharedVenueAges` each tick with `None -> i64::MAX`.
- Layer A enforcer reads the store, now explicitly skipping `i64::MAX` unknown ages.
- Layer B REST monitor also reads same age store and injects REST snapshots when stale.  
Evidence: `Cmd: U20,U28,U29,U30`;  
Code: `paraphina/src/live/runner.rs:1631-1640`, `paraphina/src/live/shared_venue_ages.rs:35-49`, `paraphina/src/live/venue_health_enforcer.rs:91-97`, `paraphina/src/live/rest_health_monitor.rs:73-90`, `paraphina/src/live/rest_health_monitor.rs:101-113`.

### 2.6 Live entrypoint wiring and queue capacities
- Global channels:
  - `market_ingest_tx/rx`: 1024
  - `market_tx/rx`: 1024
  - `account`: 256
  - `exec`: 512
  - `order_snapshot`: 128
  - `order_tx`: 256
- Per-connector forwarders await-send into shared channels (backpressure point).
- Runner receives `shared_venue_ages: Some(...)`.
- Layer A and Layer B tasks are spawned from same shared age source.  
Evidence: `Cmd: U32,U33,U39,U116,U119`;  
Code: `paraphina/src/bin/paraphina_live.rs:1746-1777`, `paraphina/src/bin/paraphina_live.rs:1808-1810`, `paraphina/src/bin/paraphina_live.rs:1832-1838`, `paraphina/src/bin/paraphina_live.rs:941-960`, `paraphina/src/bin/paraphina_live.rs:2723-2751`, `paraphina/src/bin/paraphina_live.rs:2897-2905`.

### 2.7 Global ASCII diagram

```text
WS Connector Task (per venue)
  -> connector-local decode/seq/watchdog/ping
  -> publish path:
     HL: internal queue cap=256, try_send overflow => pending_latest overwrite (lossy)
     Others: MarketPublisher queue cap=256 (live), drain_max=64
             lossless predicate = L2Snapshot/L2Delta
             non-lossless overflow => pending_latest overwrite
  -> per-connector channel (market 1024 / account 256 / exec 512)
  -> spawn_connector_forwarders (await send)
  -> market_ingest_tx cap=1024
      -> ingress bridge task:
         optional timestamp override (paper mode)
         optional paper_market_tx try_send (drop on full)
         await send market_tx cap=1024
  -> runner LiveChannels.market_rx
      -> drain_ordered_events()
         optional snapshot/delta coalescing
         optional per-venue delta cap (PARAPHINA_L2_TICK_DELTA_BUFFER_MAX) with drops
      -> apply_market_event_to_core()
      -> state.apply_l2_snapshot/delta()
         sets last_mid_update_ms only if mid/spread valid
      -> telemetry compute_age_ms(effective_now, last_mid_update_ms)
      -> venue_age_ms JSONL field
      -> tools/paraphina_watch.py renders `age_ms` column

Side channels:
  runner -> SharedVenueAges
    -> Layer A VenueHealthEnforcer (force-restart)
    -> Layer B REST health monitor (inject REST snapshots)
```

---

## 3) Per-Venue Deep Dives

## 3.1 Hyperliquid

### A) Connectivity architecture
- Endpoint rotation/probing exists (`ws_urls/rest_urls/info_urls`) with round-robin and health probe before reconnect after repeated failures.  
Evidence: `Cmd: U52,U53,U54,U45`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:150-157`, `paraphina/src/live/connectors/hyperliquid.rs:229-231`, `paraphina/src/live/connectors/hyperliquid.rs:340-349`, `paraphina/src/live/connectors/hyperliquid.rs:417-428`.
- Timeouts/watchdogs:
  - connect timeout default 15s
  - read timeout default 30s
  - stale threshold default 10s
  - watchdog tick 200ms
  - session timeout default 24h  
Evidence: `Cmd: U42,U45,U44`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:10-24`, `paraphina/src/live/connectors/hyperliquid.rs:40-55`, `paraphina/src/live/connectors/hyperliquid.rs:433-440`, `paraphina/src/live/connectors/hyperliquid.rs:554-560`.
- Keepalive/reconnect:
  - outbound JSON ping (`{"method":"ping"}`) every 30s default
  - exponential backoff with jitter (only venue with explicit jitter).  
Evidence: `Cmd: U46,U44,U45,U118`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:532-539`, `paraphina/src/live/connectors/hyperliquid.rs:623-627`, `paraphina/src/live/connectors/hyperliquid.rs:480-491`.

### B) Message model + parsing
- Subscribes `l2Book`; ignores `subscriptionResponse` and `pong`.  
Evidence: `Cmd: U46,U50`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:519-522`, `paraphina/src/live/connectors/hyperliquid.rs:677-680`.
- Snapshot/delta parsing from `levels`/`changes`; sequence tracking can trigger snapshot refresh on gaps.  
Evidence: `Cmd: U47,U55`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:1341-1368`, `paraphina/src/live/connectors/hyperliquid.rs:795-804`, `paraphina/src/live/connectors/hyperliquid.rs:812-824`.

### C) Timestamp semantics
- WS and REST snapshots primarily use exchange `time` field (`data.time` / `value.time`) with fallback `0`.  
Evidence: `Cmd: U47,U48,U49`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:1340`, `paraphina/src/live/connectors/hyperliquid.rs:1758`, `paraphina/src/live/connectors/hyperliquid.rs:1817`.
- Classification: mostly `EXCHANGE_TIME` (with zero fallback edge path).

### D) Buffering/backpressure/drop policy
- Connector-local internal queue: cap 256, `try_send` full -> overwrite `pending_latest` (lossy coalescing).  
Evidence: `Cmd: U43,U44`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:13`, `paraphina/src/live/connectors/hyperliquid.rs:567-595`.
- Bootstrap delta buffer cap 1024; overflow clears and forces snapshot refresh.  
Evidence: `Cmd: U42,U55`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:14`, `paraphina/src/live/connectors/hyperliquid.rs:812-816`.

### E) Recovery architecture (Layer A/B)
- Layer A enforcer registered.
- Layer B REST monitor entry registered.
- Additional internal connector REST fallback loop exists and is spawned.  
Evidence: `Cmd: U34,U122`;  
Code: `paraphina/src/bin/paraphina_live.rs:1975-1993`, `paraphina/src/bin/paraphina_live.rs:1994-2015`, `paraphina/src/bin/paraphina_live.rs:2026-2030`, `paraphina/src/live/connectors/hyperliquid.rs:974-983`.

### F) Venue-specific failure modes
- `connect timeout`, `read timeout`, `stale watchdog`, `ping send failed`, `session timeout` all force reconnect paths with explicit logs.  
Evidence: `Cmd: U45,U44`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:506-513`, `paraphina/src/live/connectors/hyperliquid.rs:631-639`, `paraphina/src/live/connectors/hyperliquid.rs:619-621`, `paraphina/src/live/connectors/hyperliquid.rs:625-627`, `paraphina/src/live/connectors/hyperliquid.rs:455-461`.

## 3.2 Lighter

### A) Connectivity architecture
- JSON subscribe to `order_book/<market_id>` after market-id resolution from REST endpoint fallbacks.  
Evidence: `Cmd: U61,U67,U123`;  
Code: `paraphina/src/live/connectors/lighter.rs:652-656`, `paraphina/src/live/connectors/lighter.rs:1517-1549`.
- Timeouts/watchdogs:
  - connect timeout default 15s
  - read timeout default 30s
  - stale default 10s
  - watchdog tick 200ms
  - session timeout default 24h  
Evidence: `Cmd: U57,U60,U61`;  
Code: `paraphina/src/live/connectors/lighter.rs:10-14`, `paraphina/src/live/connectors/lighter.rs:28-50`, `paraphina/src/live/connectors/lighter.rs:537-544`, `paraphina/src/live/connectors/lighter.rs:623-635`.
- Reconnect/backoff:
  - exponential backoff, deterministic (no jitter path found).  
Evidence: `Cmd: U60,U118`;  
Code: `paraphina/src/live/connectors/lighter.rs:597-606`.
- Keepalive:
  - no outbound periodic ping timer branch in public loop;
  - inbound WS Ping->Pong and JSON ping->pong reply only.  
Evidence: `Cmd: U61,U68,U120`;  
Code: `paraphina/src/live/connectors/lighter.rs:677-703`, `paraphina/src/live/connectors/lighter.rs:712-716`, `paraphina/src/live/connectors/lighter.rs:771-777`.

### B) Message model + parsing
- First book message decoded as snapshot, then delta mode.
- Consecutive delta decode failures >= 10 trigger reconnect for fresh snapshot.  
Evidence: `Cmd: U62,U57`;  
Code: `paraphina/src/live/connectors/lighter.rs:831-855`, `paraphina/src/live/connectors/lighter.rs:898-903`, `paraphina/src/live/connectors/lighter.rs:15-17`.

### C) Timestamp semantics
- `decode_market_timestamp_ms`: uses `timestamp`/`ts` if `>0`, else fallback to local non-zero wall-clock.
- Audit counter logs fallback usage (`PARAPHINA_WS_AUDIT`).  
Evidence: `Cmd: U58,U63,U64,U65`;  
Code: `paraphina/src/live/connectors/lighter.rs:252-276`, `paraphina/src/live/connectors/lighter.rs:1222`, `paraphina/src/live/connectors/lighter.rs:1342`, `paraphina/src/live/connectors/lighter.rs:1417`, `paraphina/src/live/connectors/lighter.rs:1477`.
- Classification: `MIXED` (exchange-provided when valid, local fallback otherwise).

### D) Buffering/backpressure/drop policy
- Uses `MarketPublisher` cap=256, drain_max=64.
- Lossless predicate includes both `L2Delta` and `L2Snapshot` (non-lossless events can coalesce on overflow).  
Evidence: `Cmd: U58,U59,U21`;  
Code: `paraphina/src/live/connectors/lighter.rs:8-9`, `paraphina/src/live/connectors/lighter.rs:312-322`, `paraphina/src/live/market_publisher.rs:101-114`.

### E) Recovery architecture (Layer A/B)
- Layer A enforcer slot exists.
- Layer B REST monitor entry for Lighter is still not wired (`ltr_rest_url/ltr_market` captured but no push).  
Evidence: `Cmd: U35,U40`;  
Code: `paraphina/src/bin/paraphina_live.rs:2183-2200`, `paraphina/src/bin/paraphina_live.rs:2161-2162`, `paraphina/src/bin/paraphina_live.rs:2001`, `paraphina/src/bin/paraphina_live.rs:2315`, `paraphina/src/bin/paraphina_live.rs:2480`, `paraphina/src/bin/paraphina_live.rs:2646`.

### F) Venue-specific failure modes
- Subscribe failures loop with dedicated warning escalation.
- Delta decode failure loop triggers reconnect.
- Read timeout and stale watchdog reconnect.
- Ping-idle disconnect risk is comparatively higher due lack of outbound timer.  
Evidence: `Cmd: U60,U61,U62,U120`;  
Code: `paraphina/src/live/connectors/lighter.rs:549-556`, `paraphina/src/live/connectors/lighter.rs:898-903`, `paraphina/src/live/connectors/lighter.rs:686-690`, `paraphina/src/live/connectors/lighter.rs:679-680`, `paraphina/src/live/connectors/lighter.rs:712-716`.

## 3.3 Extended

### A) Connectivity architecture
- Uses URL-path WS stream: `/orderbooks/<market>?depth=1` (no JSON subscribe loop).  
Evidence: `Cmd: U71,U123`;  
Code: `paraphina/src/live/connectors/extended.rs:161-166`.
- Timeouts/watchdogs:
  - connect timeout 15s (hardcoded in `timeout(...)`)
  - read timeout 10s (hardcoded in select)
  - ping interval default 30s
  - stale threshold default 10s
  - watchdog tick 200ms  
Evidence: `Cmd: U70,U74,U75`;  
Code: `paraphina/src/live/connectors/extended.rs:12-13`, `paraphina/src/live/connectors/extended.rs:412-415`, `paraphina/src/live/connectors/extended.rs:436-440`, `paraphina/src/live/connectors/extended.rs:450-457`, `paraphina/src/live/connectors/extended.rs:476-485`.
- Reconnect/backoff:
  - deterministic exponential backoff, no jitter code path.  
Evidence: `Cmd: U73,U82,U118`;  
Code: `paraphina/src/live/connectors/extended.rs:305-313`.

### B) Message model + parsing
- Parses `depthUpdate` deltas; non-depth events return `None`.
- Parse error threshold (`MAX_PARSE_ERRORS=25`) triggers reconnect.
- Startup path optionally fetches REST snapshot and publishes it before WS deltas.  
Evidence: `Cmd: U75,U81,U80`;  
Code: `paraphina/src/live/connectors/extended.rs:419`, `paraphina/src/live/connectors/extended.rs:532-550`, `paraphina/src/live/connectors/extended.rs:1437-1443`, `paraphina/src/live/connectors/extended.rs:348-356`, `paraphina/src/live/connectors/extended.rs:395-403`.

### C) Timestamp semantics
- Delta event timestamp uses `update.event_time.unwrap_or_else(now_ms)`.
- WS snapshot parser uses `E/ts/...` fallback `now_ms`.
- Bootstrap REST snapshot publish uses `timestamp_ms: now_ms()`.  
Evidence: `Cmd: U78,U79,U80,U81`;  
Code: `paraphina/src/live/connectors/extended.rs:1386-1391`, `paraphina/src/live/connectors/extended.rs:1579-1584`, `paraphina/src/live/connectors/extended.rs:399`, `paraphina/src/live/connectors/extended.rs:1455-1458`.
- Classification: `MIXED`.

### D) Buffering/backpressure/drop policy
- `MarketPublisher` cap 256 (live) / 4096 (fixture), drain 64.
- Lossless predicate includes snapshots + deltas.  
Evidence: `Cmd: U70,U72,U21`;  
Code: `paraphina/src/live/connectors/extended.rs:14-16`, `paraphina/src/live/connectors/extended.rs:211-220`, `paraphina/src/live/market_publisher.rs:101-114`.

### E) Recovery architecture (Layer A/B)
- Layer A enforcer slot wired.
- Layer B REST monitor entry wired.
- Internal connector snapshot bootstrap/resync path exists (`fetch_snapshot()` inside WS loop).  
Evidence: `Cmd: U36,U80,U74`;  
Code: `paraphina/src/bin/paraphina_live.rs:2286-2303`, `paraphina/src/bin/paraphina_live.rs:2305-2318`, `paraphina/src/live/connectors/extended.rs:348-349`, `paraphina/src/live/connectors/extended.rs:341-347`.

### F) Venue-specific failure modes
- Read timeout pre-first-message can reconnect loop.
- Parse-error streak reconnect.
- Stale watchdog reconnect.
- Ping send failure reconnect.  
Evidence: `Cmd: U74,U75`;  
Code: `paraphina/src/live/connectors/extended.rs:481-487`, `paraphina/src/live/connectors/extended.rs:545-550`, `paraphina/src/live/connectors/extended.rs:466-467`, `paraphina/src/live/connectors/extended.rs:470-473`.

## 3.4 Aster

### A) Connectivity architecture
- URL-path WS stream `<symbol>@depth@100ms` (no subscribe ack dependence).  
Evidence: `Cmd: U88,U123`;  
Code: `paraphina/src/live/connectors/aster.rs:341-350`.
- Timeouts/watchdogs:
  - connect timeout 15s
  - read timeout 30s
  - ping interval default 30s
  - stale default 10s via env
  - watchdog tick 200ms
  - additional local stale resync watchdog path (`STALE_MS=2000`, cooldown).  
Evidence: `Cmd: U84,U88,U89,U90,U94`;  
Code: `paraphina/src/live/connectors/aster.rs:45-63`, `paraphina/src/live/connectors/aster.rs:352-358`, `paraphina/src/live/connectors/aster.rs:388-394`, `paraphina/src/live/connectors/aster.rs:401-417`, `paraphina/src/live/connectors/aster.rs:385-386`, `paraphina/src/live/connectors/aster.rs:588-605`, `paraphina/src/live/connectors/aster.rs:740-747`.
- Reconnect/backoff:
  - deterministic exponential backoff (no jitter path found).  
Evidence: `Cmd: U87,U118`;  
Code: `paraphina/src/live/connectors/aster.rs:304-307`.

### B) Message model + parsing
- Hybrid model: REST snapshot + WS delta bridge; seq decisions (`Apply/Stale/Gap`), gap triggers resync.
- Buffered updates and snapshot bridging logic present.  
Evidence: `Cmd: U88,U91,U92,U94`;  
Code: `paraphina/src/live/connectors/aster.rs:362-370`, `paraphina/src/live/connectors/aster.rs:775-806`, `paraphina/src/live/connectors/aster.rs:1450-1502`, `paraphina/src/live/connectors/aster.rs:588-605`.

### C) Timestamp semantics
- Delta timestamps use `event_time` if available, else local `now_ms`.
- Snapshot events created from REST snapshot path use local `now_ms`.  
Evidence: `Cmd: U92,U93,U95`;  
Code: `paraphina/src/live/connectors/aster.rs:1458-1459`, `paraphina/src/live/connectors/aster.rs:1518`, `paraphina/src/live/connectors/aster.rs:1605-1608`, `paraphina/src/live/connectors/aster.rs:480`, `paraphina/src/live/connectors/aster.rs:475-482`.
- Classification: `MIXED`.

### D) Buffering/backpressure/drop policy
- `MarketPublisher` cap 256 (live) / 4096 (fixture), drain 64.
- Lossless predicate includes snapshots + deltas.
- Additional connector-local buffered update vector for sequencing (`MAX_BUFFERED_UPDATES=1024`).  
Evidence: `Cmd: U84,U86,U88,U21`;  
Code: `paraphina/src/live/connectors/aster.rs:47-49`, `paraphina/src/live/connectors/aster.rs:206-215`, `paraphina/src/live/connectors/aster.rs:362-364`, `paraphina/src/live/market_publisher.rs:101-114`.

### E) Recovery architecture (Layer A/B)
- Layer A enforcer slot wired.
- Layer B REST monitor entry wired.
- Internal REST snapshot fetch path is actively used for resync.  
Evidence: `Cmd: U37,U94,U95`;  
Code: `paraphina/src/bin/paraphina_live.rs:2451-2468`, `paraphina/src/bin/paraphina_live.rs:2470-2483`, `paraphina/src/live/connectors/aster.rs:426-433`, `paraphina/src/live/connectors/aster.rs:568-578`.

### F) Venue-specific failure modes
- WS stale -> local resync loop (not immediate full reconnect).
- Read timeout / ping send failure paths reconnect.
- Seq gap -> snapshot resync path.  
Evidence: `Cmd: U90,U94,U91`;  
Code: `paraphina/src/live/connectors/aster.rs:730-737`, `paraphina/src/live/connectors/aster.rs:740-747`, `paraphina/src/live/connectors/aster.rs:596-603`, `paraphina/src/live/connectors/aster.rs:803-806`.

## 3.5 Paradex

### A) Connectivity architecture
- JSON subscribe to `bbo.<market>`.
- Timeouts/watchdogs:
  - connect timeout 15s
  - read timeout 30s
  - ping interval default 30s
  - stale default 10s
  - watchdog tick 200ms
  - session timeout default 24h  
Evidence: `Cmd: U101,U102,U100,U97`;  
Code: `paraphina/src/live/connectors/paradex.rs:343-349`, `paraphina/src/live/connectors/paradex.rs:333-339`, `paraphina/src/live/connectors/paradex.rs:429-437`, `paraphina/src/live/connectors/paradex.rs:361-365`, `paraphina/src/live/connectors/paradex.rs:380-390`, `paraphina/src/live/connectors/paradex.rs:12-13`, `paraphina/src/live/connectors/paradex.rs:250-257`.
- Outbound ping implemented and audit-counted (`PARAPHINA_WS_AUDIT`).  
Evidence: `Cmd: U102`;  
Code: `paraphina/src/live/connectors/paradex.rs:402-422`.
- Reconnect/backoff:
  - deterministic exponential backoff (no jitter path found).  
Evidence: `Cmd: U100,U106,U118`;  
Code: `paraphina/src/live/connectors/paradex.rs:296-304`.

### B) Message model + parsing
- Active path subscribes BBO and synthesizes L2 snapshot from BBO bid/ask.
- `parse_orderbook_message_value` supports `snapshot/delta` orderbook channels with strict seq handling, but that is separate from BBO subscription product.  
Evidence: `Cmd: U101,U103,U105,U110,U107,U109`;  
Code: `paraphina/src/live/connectors/paradex.rs:343-349`, `paraphina/src/live/connectors/paradex.rs:1324-1341`, `paraphina/src/live/connectors/paradex.rs:1373-1382`, `paraphina/src/live/connectors/paradex.rs:1216-1237`, `paraphina/src/live/connectors/paradex.rs:824-877`, `paraphina/src/live/connectors/paradex.rs:569`.

### C) Timestamp semantics
- BBO snapshot path uses local `now_ms()`.
- Delta path uses local `now_ms()`.
- Snapshot parse path uses payload `ts/timestamp` else local `now_ms()`.  
Evidence: `Cmd: U105,U104`;  
Code: `paraphina/src/live/connectors/paradex.rs:1363-1379`, `paraphina/src/live/connectors/paradex.rs:1282-1289`, `paraphina/src/live/connectors/paradex.rs:1251-1257`.
- Classification: local-heavy `MIXED` (mostly local wall-clock).

### D) Buffering/backpressure/drop policy
- `MarketPublisher` cap 256, drain 64, snapshots+deltas lossless.
- Non-lossless types can still coalesce via generic publisher behavior.  
Evidence: `Cmd: U99,U21`;  
Code: `paraphina/src/live/connectors/paradex.rs:190-198`, `paraphina/src/live/connectors/paradex.rs:12-15`, `paraphina/src/live/market_publisher.rs:101-114`.

### E) Recovery architecture (Layer A/B)
- Layer A enforcer slot wired.
- Layer B REST monitor entry wired.
- No dedicated Paradex connector-internal REST market fallback loop analogous to Hyperliquid `run_rest_book_fallback` found in this file.  
Evidence: `Cmd: U38,U51,U96`;  
Code: `paraphina/src/bin/paraphina_live.rs:2618-2635`, `paraphina/src/bin/paraphina_live.rs:2637-2649`, `paraphina/src/live/connectors/hyperliquid.rs:974-983`.

### F) Venue-specific failure modes
- Subscribe error -> immediate reconnect.
- Ping send fail/read timeout/stale watchdog -> reconnect.
- Seq mismatch or seq gap in orderbook parser -> error propagation to reconnect path.  
Evidence: `Cmd: U103,U102,U107,U109,U110`;  
Code: `paraphina/src/live/connectors/paradex.rs:503-509`, `paraphina/src/live/connectors/paradex.rs:424-425`, `paraphina/src/live/connectors/paradex.rs:433-437`, `paraphina/src/live/connectors/paradex.rs:399-401`, `paraphina/src/live/connectors/paradex.rs:861-877`, `paraphina/src/live/connectors/paradex.rs:569`.

---

## 4) Cross-Venue Comparison Matrix

| Venue | Feed Product | Connect / Read Timeout | Keepalive | Watchdog | Reconnect Backoff | Connector Buffering | Timestamp Semantics | Layer A | Layer B | Internal Market REST Fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| Hyperliquid | `l2Book` subscribe | 15s / 30s | JSON ping every 30s default | 10s default, 200ms tick | Exponential + jitter | internal Q=256 + `pending_latest`; delta bootstrap buf=1024 | mostly exchange `time` | Yes | Yes | Yes (`run_rest_book_fallback`) |
| Lighter | `order_book/<id>` subscribe | 15s / 30s | inbound ping/pong + JSON pong only; no outbound timer | 10s default, 200ms tick | Exponential, deterministic | `MarketPublisher` Q=256 drain64; L2 lossless | `timestamp/ts` else local nonzero | Yes | No | No dedicated stale REST fallback loop |
| Extended | URL stream `orderbooks/<m>?depth=1` | 15s / 10s | WS Ping frame timer (30s default) | 10s default, 200ms tick | Exponential, deterministic | `MarketPublisher` Q=256 (live)/4096 fixture, L2 lossless | `E/ts` else local now | Yes | Yes | REST snapshot bootstrap/resync in connector |
| Aster | URL stream `<sym>@depth@100ms` | 15s / 30s | WS Ping frame timer (30s default) | 10s default, 200ms tick (+ local stale resync watchdog) | Exponential, deterministic | `MarketPublisher` Q=256 (live)/4096 fixture, L2 lossless | `E` else local now | Yes | Yes | REST snapshot bridge/resync in connector |
| Paradex | `bbo.<market>` subscribe | 15s / 30s | WS Ping frame timer (30s default) + WS_AUDIT ping counters | 10s default, 200ms tick | Exponential, deterministic | `MarketPublisher` Q=256 drain64, L2 lossless | local-heavy (`now_ms` for BBO/delta) | Yes | Yes | No dedicated loop found |

Evidence: `Cmd: U42,U44,U45,U46,U57,U60,U61,U67,U70,U71,U72,U74,U82,U84,U86,U88,U90,U97,U99,U101,U102,U106,U118,U123,U34,U35,U36,U37,U38,U40,U51`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:10-24`, `paraphina/src/live/connectors/hyperliquid.rs:519-527`, `paraphina/src/live/connectors/hyperliquid.rs:567-595`, `paraphina/src/live/connectors/lighter.rs:8-17`, `paraphina/src/live/connectors/lighter.rs:652-656`, `paraphina/src/live/connectors/lighter.rs:712-716`, `paraphina/src/live/connectors/extended.rs:12-16`, `paraphina/src/live/connectors/extended.rs:161-166`, `paraphina/src/live/connectors/aster.rs:45-49`, `paraphina/src/live/connectors/aster.rs:345-350`, `paraphina/src/live/connectors/paradex.rs:12-15`, `paraphina/src/live/connectors/paradex.rs:343-349`, `paraphina/src/bin/paraphina_live.rs:1975-2015`, `paraphina/src/bin/paraphina_live.rs:2183-2200`, `paraphina/src/bin/paraphina_live.rs:2305-2318`, `paraphina/src/bin/paraphina_live.rs:2470-2483`, `paraphina/src/bin/paraphina_live.rs:2637-2649`, `paraphina/src/bin/paraphina_live.rs:2161-2162`.

---

## 5) Why `age_ms` Differs (Root-Cause Partition + Ranking)

### 5.1 EXTERNAL
1. Feed product mismatch (full depth vs depth=1 vs BBO) changes update cadence before Paraphina sees events.
2. Aster explicit `@100ms` stream policy differs from venue-specific cadence elsewhere.  
Evidence: `Cmd: U123`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:521-522`, `paraphina/src/live/connectors/lighter.rs:1517-1518`, `paraphina/src/live/connectors/extended.rs:163-166`, `paraphina/src/live/connectors/aster.rs:345`, `paraphina/src/live/connectors/paradex.rs:343-349`.

### 5.2 INTERNAL
1. Queue pressure and coalescing can suppress freshness advancement under load.
2. Runner unready-delta buffering cap can drop events.
3. Recovery asymmetry (Lighter lacking Layer B entry) increases probability of prolonged stale plateaus on that venue.
4. Keepalive/reconnect differences change disconnect frequency and stale tails.  
Evidence: `Cmd: U21,U24,U26,U40,U34,U35,U74,U90,U102,U120`;  
Code: `paraphina/src/live/market_publisher.rs:108-114`, `paraphina/src/live/runner.rs:563-579`, `paraphina/src/live/runner.rs:618-625`, `paraphina/src/bin/paraphina_live.rs:2161-2162`, `paraphina/src/bin/paraphina_live.rs:2305-2318`, `paraphina/src/live/connectors/extended.rs:469-474`, `paraphina/src/live/connectors/aster.rs:734-737`, `paraphina/src/live/connectors/paradex.rs:402-427`, `paraphina/src/live/connectors/lighter.rs:712-716`.

### 5.3 MIXED
1. Age scalar combines heterogeneous connector timestamp semantics.
2. Telemetry clamps future-skew to `0`, masking clock skew/ordering anomalies.
3. Shared age mapping converts unknown/negative to `i64::MAX`; enforcer now skips unknown sentinel but monitors still depend on mapped values.  
Evidence: `Cmd: U12,U13,U14,U28,U29,U20`;  
Code: `paraphina/src/telemetry.rs:830-835`, `paraphina/src/live/shared_venue_ages.rs:46-47`, `paraphina/src/live/shared_venue_ages.rs:61-65`, `paraphina/src/live/venue_health_enforcer.rs:92-95`, `paraphina/src/live/runner.rs:1634-1638`.

### 5.4 Venue-specific likely drivers (hypothesis level)
- Hyperliquid: exchange-time semantics + lossy connector queue and jittered reconnect profile can make behavior differ from others.
- Lighter: timestamp fallback to local now + no outbound ping timer + missing Layer B likely contributes to tail staleness events.
- Extended/Aster: mixed timestamp fallback to local now and REST-assisted sequencing can alter apparent freshness distributions.
- Paradex: BBO-only subscribed feed and local timestamping materially decouple displayed age from exchange event-time age.  
Evidence basis: `Cmd: U44,U58,U61,U74,U88,U105,U123,U40`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:567-595`, `paraphina/src/live/connectors/lighter.rs:252-276`, `paraphina/src/live/connectors/lighter.rs:712-716`, `paraphina/src/live/connectors/extended.rs:1579-1584`, `paraphina/src/live/connectors/aster.rs:1458-1459`, `paraphina/src/live/connectors/paradex.rs:1363-1379`, `paraphina/src/live/connectors/paradex.rs:343-349`, `paraphina/src/bin/paraphina_live.rs:2161-2162`.

---

## 6) Frontier Bottlenecks / Thrash Sources Catalog

| Symptom | Trigger | Code Site | Observable Evidence (log signatures) |
|---|---|---|---|
| Early/recurring stale restarts | age unknown sentinel interpreted as stale (historical) | `paraphina/src/live/venue_health_enforcer.rs:91-97` | `"VenueHealthEnforcer force-restarting..."` |
| Unready venue loses delta continuity | runner per-venue cap reached | `paraphina/src/live/runner.rs:572-579`, `paraphina/src/live/runner.rs:618-625` | `market_rx_stats ... cap_hits=` |
| HL event loss under pressure | internal publish queue full -> overwrite latest | `paraphina/src/live/connectors/hyperliquid.rs:588-595` | stale/sequence refresh patterns with reduced output continuity |
| Lighter decode fail loop | repeated delta decode failure | `paraphina/src/live/connectors/lighter.rs:898-903` | `"consecutive delta decode failures"` |
| Extended parse-fail reconnect | parse errors > 25 | `paraphina/src/live/connectors/extended.rs:419`, `paraphina/src/live/connectors/extended.rs:545-550` | `"too many parse errors; reconnecting"` |
| Aster sequence gap resync churn | `SeqDecision::Gap` path | `paraphina/src/live/connectors/aster.rs:803-806`, `paraphina/src/live/connectors/aster.rs:596-603` | `"Aster WS seq gap; resyncing"` and `"Aster WS stale; resyncing"` |
| Paradex subscribe/product mismatch effects | BBO-only channel with local-time snapshots | `paraphina/src/live/connectors/paradex.rs:343-349`, `paraphina/src/live/connectors/paradex.rs:1363-1379` | `"Paradex subscribed channel=bbo..."` |
| Layer B asymmetry for Lighter | no REST monitor entry | `paraphina/src/bin/paraphina_live.rs:2161-2162`, `paraphina/src/bin/paraphina_live.rs:2001`, `paraphina/src/bin/paraphina_live.rs:2315` | no `REST health monitor: lighter ...` line expected |

Evidence: `Cmd: U29,U24,U44,U62,U75,U91,U102,U103,U40,U126`.

---

## 7) Findings -> Hypotheses -> Measurements (Implementation-Ready, No Code Changes)

| Finding | Hypothesis | Measurement (shadow-only) | Pass/Fail Signal |
|---|---|---|---|
| Mixed timestamp semantics | Cross-venue age spread inflated by semantics mismatch rather than transport quality | Collect `venue_age_ms` distributions + WS_AUDIT fallback counts + connector reconnect reasons | Large spread with low reconnects suggests semantic mismatch dominates |
| Lighter Layer B missing | Long stale tails on Lighter persist longer than peers | Compare stale episode durations by venue in same 10m/90m runs | Lighter p99 stale duration materially above peers |
| Runner cap-hits dropping deltas | Cap hits correlate with stale spikes and/or seq repair churn | Enable `PARAPHINA_MARKET_RX_STATS=1`, inspect `cap_hits` and stale windows | cap_hits bursts align with stale episodes |
| Paradex BBO local timestamping | Paradex age may look artificially better/worse vs exchange-time venues | Compare Paradex age behavior vs reconnect/read-timeout frequency | weak coupling between age and transport errors supports metric bias |
| Keepalive asymmetry | Lighter disconnects more under quiet periods due no outbound ping timer | Compare reconnect/read-timeout counts across venues during low activity periods | higher lighter timeout/reconnect frequency |

Evidence basis: `Cmd: U58,U102,U126,U40,U123,U120,U14`;  
Code: `paraphina/src/live/connectors/lighter.rs:252-276`, `paraphina/src/live/connectors/paradex.rs:402-422`, `paraphina/src/live/runner.rs:1245-1259`, `paraphina/src/bin/paraphina_live.rs:2161-2162`, `paraphina/src/live/connectors/paradex.rs:1363-1379`, `paraphina/src/live/connectors/lighter.rs:712-716`, `paraphina/src/telemetry.rs:1640-1658`.

---

## 8) Measurement Protocol (Shadow-Only; Commands Only)

Important:
- No API keys required for these commands.
- Do not run live mode.
- Long soak command is provided but not auto-run in this audit.

### 8.1 10-minute smoke (all five venues)

```bash
mkdir -p /tmp/ws_ultra_10m
PARAPHINA_LIVE_MODE=1 \
PARAPHINA_LIVE_PREFLIGHT_OK=1 \
PARAPHINA_TELEMETRY_MODE=jsonl \
PARAPHINA_TELEMETRY_PATH=/tmp/ws_ultra_10m/telemetry.jsonl \
PARAPHINA_WS_AUDIT=1 \
PARAPHINA_MARKET_RX_STATS=1 \
PARAPHINA_MARKET_RX_STATS_PATH=/tmp/ws_ultra_10m/market_rx_stats.log \
cargo run --bin paraphina_live -- \
  --trade-mode shadow \
  --connectors hyperliquid,lighter,extended,aster,paradex \
  --out-dir /tmp/ws_ultra_10m 2>&1 | tee /tmp/ws_ultra_10m/live.log
```

Why this command is grounded:
- CLI supports `--trade-mode`, `--connectors`, `--out-dir`.  
Evidence: `Cmd: U112,U133`;  
Code: `paraphina/src/bin/paraphina_live.rs:142-151`, `paraphina/src/bin/paraphina_live.rs:165-167`.
- Telemetry env/path behavior is defined in code.  
Evidence: `Cmd: U128,U129,U130`;  
Code: `paraphina/src/bin/paraphina_live.rs:501-510`, `paraphina/src/telemetry.rs:14-19`, `paraphina/src/telemetry.rs:141-170`.

### 8.2 90-minute soak (same command, longer runtime)
- Re-run the 10-minute command for 90 minutes to expose tail/reconnect behavior.

### 8.3 Post-run extraction commands

1) Venue age percentiles and trend report (existing tool):
```bash
python3 tools/telemetry_analyzer.py \
  --telemetry /tmp/ws_ultra_10m/telemetry.jsonl \
  > /tmp/ws_ultra_10m/telemetry_report.txt
```
Evidence: `Cmd: U135,U134`;  
Code: `tools/telemetry_analyzer.py:8-13`, `tools/telemetry_analyzer.py:147-149`, `tools/telemetry_analyzer.py:239-240`, `tools/telemetry_analyzer.py:876`.

2) Live dashboard replay:
```bash
python3 tools/paraphina_watch.py \
  --telemetry /tmp/ws_ultra_10m/telemetry.jsonl
```
Evidence: `Cmd: U133`;  
Code: `tools/paraphina_watch.py:79-85`.

3) Reconnect/failure reason counts from logs:
```bash
rg -n "read timeout|stale: freshness exceeded|ping send failed|subscribe failed|seq gap|session timeout|force-restarting" \
  /tmp/ws_ultra_10m/live.log
```

4) WS_AUDIT counters (already implemented):
```bash
rg -n "WS_AUDIT venue=lighter|WS_AUDIT venue=paradex" /tmp/ws_ultra_10m/live.log
```
Evidence: `Cmd: U58,U102`;  
Code: `paraphina/src/live/connectors/lighter.rs:269-272`, `paraphina/src/live/connectors/paradex.rs:409-421`.

5) Queue pressure / cap-hit signals:
```bash
rg -n "market_rx_stats .*cap_hits=" /tmp/ws_ultra_10m/market_rx_stats.log /tmp/ws_ultra_10m/live.log
```
Evidence: `Cmd: U126`;  
Code: `paraphina/src/live/runner.rs:1245-1259`, `paraphina/src/live/runner.rs:1263-1270`.

### 8.4 Unknowns / external verification needed
- True upstream venue feed cadence and exchange-side throttling behavior require external docs/runtime captures.
- No explicit connector-level decompression code found in these files; transport-level compression behavior in websocket stack is not asserted here.  
Evidence: `Cmd: U121`;  
Code: `paraphina/src/live/connectors/hyperliquid.rs:1-3`, `paraphina/src/live/connectors/lighter.rs:1`, `paraphina/src/live/connectors/extended.rs:1`, `paraphina/src/live/connectors/aster.rs:1`, `paraphina/src/live/connectors/paradex.rs:1`.

---

## 9) Notes on Merged Reality

- Startup restart thrash on unknown age is mitigated: enforcer now skips `i64::MAX` unknown age until first real update.  
Evidence: `Cmd: U29`;  
Code: `paraphina/src/live/venue_health_enforcer.rs:92-95`.

- This report is intentionally non-code and implementation-ready for a subsequent change plan.

---

## 10) Reproducibility Appendix (Commands Run, In Order)

`Cmd U01`
```bash
hostname
```

`Cmd U02`
```bash
date -Is
```

`Cmd U03`
```bash
git status -sb
```

`Cmd U04`
```bash
git rev-parse HEAD
```

`Cmd U05`
```bash
git branch --show-current
```

`Cmd U06`
```bash
rg --version ; cargo --version ; rustc --version
```

`Cmd U07`
```bash
pgrep -fa 'paraphina_live' || true
```

`Cmd U08`
```bash
pgrep -fa 'paraphina_live' | rg -v "pgrep -fa" || true
```

`Cmd U09`
```bash
rg -n "age_ms|venue_age_ms|HMS|age" tools/paraphina_watch.py
```

`Cmd U10`
```bash
nl -ba tools/paraphina_watch.py | sed -n '610,735p'
```

`Cmd U11`
```bash
rg -n "compute_age_ms|venue_age_ms|last_mid_update_ms|effective_now|venue_stale_ms" paraphina/src/telemetry.rs
```

`Cmd U12`
```bash
nl -ba paraphina/src/telemetry.rs | sed -n '820,845p'
```

`Cmd U13`
```bash
nl -ba paraphina/src/telemetry.rs | sed -n '1600,1672p'
```

`Cmd U14`
```bash
nl -ba paraphina/src/telemetry.rs | sed -n '1708,1720p'
```

`Cmd U15`
```bash
rg -n "last_mid_update_ms|apply_l2_snapshot|apply_l2_delta|mid|spread" paraphina/src/state.rs paraphina/src/live/runner.rs
```

`Cmd U16`
```bash
nl -ba paraphina/src/state.rs | sed -n '175,205p'
```

`Cmd U17`
```bash
nl -ba paraphina/src/state.rs | sed -n '292,346p'
```

`Cmd U18`
```bash
nl -ba paraphina/src/state.rs | sed -n '724,738p'
```

`Cmd U19`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '2348,2392p'
```

`Cmd U20`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '1628,1643p'
```

`Cmd U21`
```bash
nl -ba paraphina/src/live/market_publisher.rs | sed -n '1,200p'
```

`Cmd U22`
```bash
rg -n "market_rx|try_recv|tick_delta_buffer_max|coalesce|saw_l2_snapshot_mask|max_events_per_tick|buffer|drop|shared_venue_ages|LiveChannels|mpsc::channel" paraphina/src/live/runner.rs
```

`Cmd U23`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '64,92p'
```

`Cmd U24`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '458,706p'
```

`Cmd U25`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '760,808p'
```

`Cmd U26`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '1018,1090p'
```

`Cmd U27`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '1280,1320p'
```

`Cmd U28`
```bash
nl -ba paraphina/src/live/shared_venue_ages.rs | sed -n '1,90p'
```

`Cmd U29`
```bash
nl -ba paraphina/src/live/venue_health_enforcer.rs | sed -n '1,130p'
```

`Cmd U30`
```bash
nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '1,150p'
```

`Cmd U31`
```bash
rg -n "SharedVenueAges::new|run_venue_health_enforcer|run_rest_health_monitor|mpsc::channel\(|market_ingest_tx|market_tx|account_tx|exec_tx|order_tx|order_snapshot|Layer A|Layer B|enforcer_slots.push|rest_entries.push|PARAPHINA_LIVE_.*CAP|_CAP" paraphina/src/bin/paraphina_live.rs
```

`Cmd U32`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1736,1814p'
```

`Cmd U33`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1814,1850p'
```

`Cmd U34`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1970,2020p'
```

`Cmd U35`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2168,2210p'
```

`Cmd U36`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2280,2330p'
```

`Cmd U37`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2446,2493p'
```

`Cmd U38`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2612,2660p'
```

`Cmd U39`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2720,2755p'
```

`Cmd U40`
```bash
rg -n "rest_entries\.push|name: \"lighter\"|ltr_rest_url|ltr_market" paraphina/src/bin/paraphina_live.rs
```

`Cmd U41`
```bash
rg -n "const .*TIMEOUT|STALE|WATCHDOG|PING|BACKOFF|JITTER|run_public_ws|connect|subscribe|l2Book|Message::Ping|Message::Pong|timeout|read timeout|reconnect|session|MarketPublisher::new|lossless|timestamp_ms|now_ms|data\.time|fetch_l2_snapshot|seq" paraphina/src/live/connectors/hyperliquid.rs
```

`Cmd U42`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1,140p'
```

`Cmd U43`
```bash
rg -n "HL_INTERNAL_PUB_Q|pending_latest|delta_bootstrap|try_send|mpsc::channel\(|VecDeque|lossless|publish_market|market_tx" paraphina/src/live/connectors/hyperliquid.rs
```

`Cmd U44`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '540,640p'
```

`Cmd U45`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '396,490p'
```

`Cmd U46`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '492,540p'
```

`Cmd U47`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1318,1372p'
```

`Cmd U48`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1738,1776p'
```

`Cmd U49`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1792,1836p'
```

`Cmd U50`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '666,720p'
```

`Cmd U51`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '968,1038p'
```

`Cmd U52`
```bash
rg -n "struct HyperliquidConfig|ws_url|rest_url|info_url|rotate_endpoint|probe_info_endpoint|from_env|public_ws_url|private_ws_url" paraphina/src/live/connectors/hyperliquid.rs
```

`Cmd U53`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '148,260p'
```

`Cmd U54`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '262,375p'
```

`Cmd U55`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '780,832p'
```

`Cmd U56`
```bash
rg -n "const .*TIMEOUT|STALE|WATCHDOG|PING|BACKOFF|MAX_CONSECUTIVE|run_public_ws|connect_async|subscribe|order_book|Message::Ping|Message::Pong|read timeout|timeout|reconnect|session|MarketPublisher::new|is_lossless|timestamp|decode_market_timestamp_ms|now_timestamp_ms_nonzero|PARAPHINA_WS_AUDIT|fetch_lighter_orderbooks_with_fallbacks|REST|stale" paraphina/src/live/connectors/lighter.rs
```

`Cmd U57`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1,80p'
```

`Cmd U58`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '232,332p'
```

`Cmd U59`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '306,326p'
```

`Cmd U60`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '512,606p'
```

`Cmd U61`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '606,734p'
```

`Cmd U62`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '778,924p'
```

`Cmd U63`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1214,1230p'
```

`Cmd U64`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1316,1350p'
```

`Cmd U65`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1390,1484p'
```

`Cmd U66`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '372,470p'
```

`Cmd U67`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1517,1605p'
```

`Cmd U68`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '734,778p'
```

`Cmd U69`
```bash
rg -n "const .*TIMEOUT|STALE|WATCHDOG|PING|BACKOFF|JITTER|run_public_ws|connect_async|subscribe|depth=1|Message::Ping|Message::Pong|timeout|reconnect|session|MarketPublisher::new|timestamp_ms|E\)|now_ms\(|decode|l2|bookTicker|depth|rest|enforcer|stale" paraphina/src/live/connectors/extended.rs
```

`Cmd U70`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1,80p'
```

`Cmd U71`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '100,170p'
```

`Cmd U72`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '206,225p'
```

`Cmd U73`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '241,303p'
```

`Cmd U74`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '404,476p'
```

`Cmd U75`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '476,560p'
```

`Cmd U76`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '808,820p'
```

`Cmd U77`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1120,1142p'
```

`Cmd U78`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1382,1412p'
```

`Cmd U79`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1528,1602p'
```

`Cmd U80`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '332,404p'
```

`Cmd U81`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1426,1498p'
```

`Cmd U82`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '300,324p'
```

`Cmd U83`
```bash
rg -n "const .*TIMEOUT|STALE|WATCHDOG|PING|BACKOFF|JITTER|run_public_ws|connect_async|subscribe|depth@100ms|Message::Ping|Message::Pong|timeout|reconnect|session|MarketPublisher::new|timestamp_ms|E\)|now_ms\(|parse_depth|rest|fallback|decode|MAX_PARSE_ERRORS|stale" paraphina/src/live/connectors/aster.rs
```

`Cmd U84`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '40,90p'
```

`Cmd U85`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '110,158p'
```

`Cmd U86`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '202,220p'
```

`Cmd U87`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '243,307p'
```

`Cmd U88`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '338,406p'
```

`Cmd U89`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '406,448p'
```

`Cmd U90`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '724,752p'
```

`Cmd U91`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '756,806p'
```

`Cmd U92`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1450,1524p'
```

`Cmd U93`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1590,1666p'
```

`Cmd U94`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '560,606p'
```

`Cmd U95`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '468,486p'
```

`Cmd U96`
```bash
rg -n "const .*TIMEOUT|STALE|WATCHDOG|PING|BACKOFF|JITTER|run_public_ws|connect_async|subscribe|bbo|Message::Ping|Message::Pong|timeout|reconnect|session|MarketPublisher::new|timestamp_ms|now_ms\(|ts|last_updated_at|decode|snapshot|delta|rest|stale" paraphina/src/live/connectors/paradex.rs
```

`Cmd U97`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1,70p'
```

`Cmd U98`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '106,146p'
```

`Cmd U99`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '184,199p'
```

`Cmd U100`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '235,299p'
```

`Cmd U101`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '328,352p'
```

`Cmd U102`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '356,441p'
```

`Cmd U103`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '459,523p'
```

`Cmd U104`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1240,1292p'
```

`Cmd U105`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1324,1384p'
```

`Cmd U106`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '296,328p'
```

`Cmd U107`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '820,892p'
```

`Cmd U108`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '516,552p'
```

`Cmd U109`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '552,606p'
```

`Cmd U110`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1216,1240p'
```

`Cmd U111`
```bash
rg -n "struct Args|trade-mode|connectors|--config|PARAPHINA_LIVE_PREFLIGHT_OK|PARAPHINA_LIVE_MODE" paraphina/src/bin/paraphina_live.rs
```

`Cmd U112`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '132,170p'
```

`Cmd U113`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '170,236p'
```

`Cmd U114`
```bash
rg -n "load_config|default.toml|PARAPHINA_CONFIG|Config::load|from_file" paraphina/src/bin/paraphina_live.rs paraphina/src/config.rs
```

`Cmd U115`
```bash
rg -n "let cfg =|Config::|read_config|from_env|build_config|default" paraphina/src/bin/paraphina_live.rs | head -n 40
```

`Cmd U116`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '928,980p'
```

`Cmd U117`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '1348,1384p'
```

`Cmd U118`
```bash
rg -n "rand::|gen_range|jitter" paraphina/src/live/connectors/hyperliquid.rs paraphina/src/live/connectors/lighter.rs paraphina/src/live/connectors/extended.rs paraphina/src/live/connectors/aster.rs paraphina/src/live/connectors/paradex.rs
```

`Cmd U119`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2892,2910p'
```

`Cmd U120`
```bash
rg -n "ping_interval|PARAPHINA_.*PING|Message::Ping\(|Message::Pong\(" paraphina/src/live/connectors/lighter.rs
```

`Cmd U121`
```bash
rg -n "gzip|deflate|zlib|flate|decompress|Compression" paraphina/src/live/connectors/hyperliquid.rs paraphina/src/live/connectors/lighter.rs paraphina/src/live/connectors/extended.rs paraphina/src/live/connectors/aster.rs paraphina/src/live/connectors/paradex.rs
```

`Cmd U122`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2016,2032p'
```

`Cmd U123`
```bash
rg -n "l2Book|order_book/|orderbooks/.+depth=1|@depth@100ms|bbo\.|subscribe|subscription" paraphina/src/live/connectors/hyperliquid.rs paraphina/src/live/connectors/lighter.rs paraphina/src/live/connectors/extended.rs paraphina/src/live/connectors/aster.rs paraphina/src/live/connectors/paradex.rs
```

`Cmd U124`
```bash
rg -n "def format_ms|hms|days|hours|minutes" tools/paraphina_watch.py
```

`Cmd U125`
```bash
nl -ba tools/paraphina_watch.py | sed -n '128,168p'
```

`Cmd U126`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '1230,1270p'
```

`Cmd U127`
```bash
rg -n "TELEMETRY_PATH|PARAPHINA_TELEMETRY|resolve_telemetry_path|TelemetryConfig::from_env|TELEMETRY_MODE|append" paraphina/src/bin/paraphina_live.rs paraphina/src/telemetry.rs
```

`Cmd U128`
```bash
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '492,518p'
```

`Cmd U129`
```bash
nl -ba paraphina/src/telemetry.rs | sed -n '10,22p'
```

`Cmd U130`
```bash
nl -ba paraphina/src/telemetry.rs | sed -n '131,186p'
```

`Cmd U131`
```bash
rg -n "HMS|hms|venue_age_ms|age_ms" tools paraphina/src | head -n 80
```

`Cmd U132`
```bash
rg -n "argparse|--telemetry|--path|def main\(|ArgumentParser|out-dir|tail" tools/paraphina_watch.py | head -n 40
```

`Cmd U133`
```bash
nl -ba tools/paraphina_watch.py | sed -n '76,96p'
```

`Cmd U134`
```bash
rg -n "p50|p95|p99|percentile|quantile|venue_age" tools/telemetry_analyzer.py | head -n 40
```

`Cmd U135`
```bash
nl -ba tools/telemetry_analyzer.py | sed -n '1,120p'
```
