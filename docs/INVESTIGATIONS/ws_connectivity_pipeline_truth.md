# WS Exchange Connectivity Pipeline Truth Doc

## Phase 0 — Repo Snapshot

- Command:
```bash
cd /home/developer/code/paraphina
git rev-parse HEAD
```
Output:
```text
e25a2665bb51da50c0366f271289f6c703bbbf38
```
- Command:
```bash
cd /home/developer/code/paraphina
git status --porcelain
```
Output:
```text
?? docs/INVESTIGATIONS/ws_connectivity_pipeline_truth.md
```
- Command:
```bash
cd /home/developer/code/paraphina
rg -n "WebSocket|WS|MarketPublisher|VenueHealthEnforcer|RestHealthMonitor|SharedVenueAges" -S paraphina/src docs/INVESTIGATIONS .github/workflows tools | head
```
Output:
```text
.github/workflows/ws_shadow_soak.yml:21:        description: "PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS"
.github/workflows/ws_shadow_soak.yml:33:        description: "PARAPHINA_EXTENDED_WS_DEPTH_LEVELS"
.github/workflows/ws_shadow_soak.yml:80:          EXTENDED_WS_READ_TIMEOUT_MS: ${{ github.event.inputs.extended_ws_read_timeout_ms || '45000' }}
.github/workflows/ws_shadow_soak.yml:82:          EXTENDED_WS_DEPTH_LEVELS: ${{ github.event.inputs.extended_ws_depth_levels || '1' }}
.github/workflows/ws_shadow_soak.yml:89:          PARAPHINA_WS_AUDIT=1 \
.github/workflows/ws_shadow_soak.yml:94:          PARAPHINA_LIGHTER_WS_READONLY=1 \
.github/workflows/ws_shadow_soak.yml:96:          PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS=$EXTENDED_WS_READ_TIMEOUT_MS \
.github/workflows/ws_shadow_soak.yml:98:          PARAPHINA_EXTENDED_WS_DEPTH_LEVELS=$EXTENDED_WS_DEPTH_LEVELS \
tools/ws_soak_report.py:374:WS_AUDIT_RECONNECT_RE = re.compile(
tools/ws_soak_report.py:375:    r"WS_AUDIT\s+venue=(?P<venue>[a-zA-Z0-9_]+)\s+reconnect_reason=(?P<reason>[a-zA-Z0-9_]+)\s+count=(?P<count>\d+)"
```

---

## Overview

Pipeline (shadow-only path) is:

```text
WS/REST connector tasks
  -> per-connector publish path (MarketPublisher OR connector-local queue)
  -> spawn_connector_forwarders (venue_id/index rewrite)
  -> market_ingest_tx (global ingress)
  -> ingest bridge task (optional wallclock timestamp override + paper tap)
  -> market_tx
  -> runner drain/order/coalesce/future-partition
  -> cache.apply_market_event
  -> apply_market_event_to_core
  -> state.apply_l2_* + orderbook trim
  -> telemetry JSON fields + SharedVenueAges writes
  -> watch UI columns (age_ms, age_event_ms)

Parallel health layers:
  Layer A: VenueHealthEnforcer reads SharedVenueAges and force-restarts stale connector tasks.
  Layer B: RestHealthMonitor reads SharedVenueAges and injects REST L2 snapshots via market_ingest_tx.
```

**Evidence**
Files: `paraphina/src/bin/paraphina_live.rs:1746-1773`; `paraphina/src/bin/paraphina_live.rs:941-977`; `paraphina/src/live/runner.rs:1326-1450`; `paraphina/src/live/runner.rs:1766-1777`; `paraphina/src/telemetry.rs:1641-1720`; `tools/paraphina_watch.py:622-730`; `paraphina/src/bin/paraphina_live.rs:2748-2778`
Commands:
```bash
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '920,990p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1728,1848p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1296,1468p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1660,1798p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/telemetry.rs | sed -n '1600,1736p'
cd /home/developer/code/paraphina && nl -ba tools/paraphina_watch.py | sed -n '596,760p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2738,2798p'
```

---

## Task/Channel Topology

Global channels in `paraphina_live`:

| Channel | Capacity | Producer(s) | Consumer(s) | Behavior |
|---|---:|---|---|---|
| `market_ingest_tx/rx` | 1024 | connector forwarders; RestHealthMonitor | ingest bridge task | Awaited send from forwarders/REST monitor |
| `market_tx/rx` | 1024 | ingest bridge task | runner | Awaited send |
| `paper_market_tx/rx` | 1024 | ingest bridge task (paper mode) | paper feed | `try_send`; dropped if full |
| `_account_tx/account_rx` | 256 | connector forwarders | runner | Awaited send in forwarder |
| `exec_tx/exec_rx` | 512 | connector forwarders / live gateway path | runner | Awaited send for connector forwarder; `try_send` for fire-and-forget gateway path |
| `_order_snapshot_tx/order_snapshot_rx` | 128 | reconcile path (stubbed) | runner | Awaited send when used |
| `order_tx/order_rx` | 256 | runner | live order handler | `try_send` in runner request path; dropped if full |

Per-connector fan-in channels (created per connector): `market` 1024, `account` 256, `exec` 512.

`spawn_connector_forwarders` rewrites venue metadata before forwarding:
- market events: overwrites `venue_id` and `venue_index`, then awaited send to `market_ingest_tx`.
- account/execution events: same metadata rewrite, then awaited send to global account/exec channels.

Ingest bridge (`market_ingest_rx -> market_tx`) does two taps:
- optional wallclock timestamp override in paper mode (`PARAPHINA_PAPER_USE_WALLCLOCK_TS`).
- paper top-of-book tap via `paper_market_update_from_event`; this tap is non-blocking and can drop.

Runner construction passes `market_rx`, `account_rx`, optional `exec_rx`, optional `order_snapshot_rx`, `order_tx`, and `shared_venue_ages` into `LiveChannels`, then into `run_live_loop`.

**Evidence**
Files: `paraphina/src/bin/paraphina_live.rs:1746-1777`; `paraphina/src/bin/paraphina_live.rs:1808-1827`; `paraphina/src/bin/paraphina_live.rs:941-977`; `paraphina/src/bin/paraphina_live.rs:843-932`; `paraphina/src/bin/paraphina_live.rs:798-841`; `paraphina/src/bin/paraphina_live.rs:2908-2913`; `paraphina/src/bin/paraphina_live.rs:2922-2965`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "market_ingest_tx|market_tx|spawn_connector_forwarders|LiveChannels|shared_venue_ages|VenueHealthEnforcer|RestHealthMonitor" -S paraphina/src/bin/paraphina_live.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '836,932p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1728,1848p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2908,2955p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2950,3008p'
```

---

## Per-Venue Connectors

| Venue | Connect / read / stale defaults | Keepalive | Decode + sequence contract | Timestamp basis | Publish path | Depth subscription semantics |
|---|---|---|---|---|---|---|
| Hyperliquid | Connect timeout default 15s (`PARAPHINA_HL_WS_CONNECT_TIMEOUT_MS`), read timeout default 30s (`PARAPHINA_HL_WS_READ_TIMEOUT_MS`), stale default 10s (`PARAPHINA_HL_STALE_MS`) | Outbound JSON `{"method":"ping"}` every `PARAPHINA_HL_PING_INTERVAL_MS` (default 30s); inbound channel `pong` ignored | `l2Book` decode with resilient snapshot path; `L2SeqTracker` requests snapshot refresh on seq gap (`msg.seq > prev+1`) and drops `<= prev` | Uses exchange `data.time`; snapshot refresh and REST fallback preserve exchange fields when present | Connector-local internal queue `HL_INTERNAL_PUB_Q=256`, `try_send` with one-slot `pending_latest` overwrite | `l2Book` subscribe payload includes `nSigFigs` + `nLevels` (`HL_L2_SIGFIGS`, `HL_L2_LEVELS`) |
| Lighter | Connect timeout default 15s (`PARAPHINA_LIGHTER_WS_CONNECT_TIMEOUT_MS`), read timeout default 30s (`PARAPHINA_LIGHTER_WS_READ_TIMEOUT_MS`), stale default 10s (`PARAPHINA_LIGHTER_STALE_MS`) | Outbound WS Ping frame every `PARAPHINA_LIGHTER_PING_INTERVAL_MS`; inbound WS ping -> pong; also JSON ping -> JSON pong | First book message decoded as snapshot, subsequent book messages decoded as delta; reconnect after `LIGHTER_MAX_CONSECUTIVE_DELTA_FAILURES=10`; seq tracker drops only strictly older seq (`msg.seq < prev`) | `decode_market_timestamp_ms`: uses `timestamp`/`ts` if >0, else local `now` fallback | `MarketPublisher` cap 256 / drain 64; L2 events are lossless | Channel is fixed `order_book/{market_id}` (no depth knob) |
| Extended | Connect timeout fixed 15s, read timeout from `PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS` default 10s, stale default 10s (`PARAPHINA_EXTENDED_STALE_MS`) | Outbound WS Ping frame (`PARAPHINA_EXTENDED_PING_INTERVAL_MS`, default 30s); inbound ping -> pong; inbound pong counted | REST snapshot bootstraps seq state; WS `depthUpdate` parsed into deltas; `ExtendedSeqState` errors on seq mismatch/gap and reconnects | Delta event uses exchange `E` else local `now`; WS snapshot parse uses `E/ts` else local `now`; initial REST snapshot event timestamp set to local `now` | `MarketPublisher` cap 256 live / 4096 fixture; drain 64; L2 events lossless | `PARAPHINA_EXTENDED_WS_DEPTH_LEVELS`: `<=1` uses `.../orderbooks/{market}?depth=1`; `>1` uses `.../orderbooks/{market}` URL |
| Aster | Connect timeout fixed 15s, read timeout fixed 30s, stale default 10s (`PARAPHINA_ASTER_STALE_MS`) | Outbound WS Ping frame (`PARAPHINA_ASTER_PING_INTERVAL_MS`, default 30s); inbound ping -> pong | URL stream mode (`<symbol>@depth@100ms`), REST snapshot + bridge-delta lock-on, lenient seq decisions (`Apply/Stale/Gap`) with snapshot re-fetch on gaps | Delta event uses exchange `E` else local `now`; REST snapshot publish uses local `now` | `MarketPublisher` cap 256 live / 4096 fixture; drain 64; L2 events lossless | Fixed stream cadence `@depth@100ms`; REST snapshot depth from `ASTER_DEPTH_LIMIT` |
| Paradex | Connect timeout fixed 15s, read timeout fixed 30s, stale default 10s (`PARAPHINA_PARADEX_STALE_MS`) | Outbound WS Ping frame (`PARAPHINA_PARADEX_PING_INTERVAL_MS`, default 30s); inbound ping -> pong | Mode-select via `PARAPHINA_PARADEX_PUBLIC_FEED`: `orderbook` uses snapshot+delta seq/prev_seq checks; `bbo` synthesizes snapshots with local seq counter | `parse_snapshot` uses payload `ts/timestamp` else local `now`; `parse_delta` uses local `now`; BBO snapshot uses local `now` | `MarketPublisher` cap 256 / drain 64; L2 events lossless | `bbo` channel or `order_book.<market>.snapshot@15@100ms` |

**Evidence**
Files: `paraphina/src/live/connectors/hyperliquid.rs:10-57`; `paraphina/src/live/connectors/hyperliquid.rs:536-569`; `paraphina/src/live/connectors/hyperliquid.rs:574-679`; `paraphina/src/live/connectors/hyperliquid.rs:605-637`; `paraphina/src/live/connectors/hyperliquid.rs:1328-1364`; `paraphina/src/live/connectors/hyperliquid.rs:1368-1415`; `paraphina/src/live/connectors/lighter.rs:8-62`; `paraphina/src/live/connectors/lighter.rs:366-376`; `paraphina/src/live/connectors/lighter.rs:665-787`; `paraphina/src/live/connectors/lighter.rs:808-814`; `paraphina/src/live/connectors/lighter.rs:921-953`; `paraphina/src/live/connectors/lighter.rs:995-1001`; `paraphina/src/live/connectors/lighter.rs:1276-1303`; `paraphina/src/live/connectors/lighter.rs:1415-1442`; `paraphina/src/live/connectors/lighter.rs:1491-1540`; `paraphina/src/live/connectors/lighter.rs:1540-1586`; `paraphina/src/live/connectors/lighter.rs:1616-1618`; `paraphina/src/live/connectors/extended.rs:12-41`; `paraphina/src/live/connectors/extended.rs:201-218`; `paraphina/src/live/connectors/extended.rs:263-273`; `paraphina/src/live/connectors/extended.rs:394-401`; `paraphina/src/live/connectors/extended.rs:448-456`; `paraphina/src/live/connectors/extended.rs:460-474`; `paraphina/src/live/connectors/extended.rs:495-500`; `paraphina/src/live/connectors/extended.rs:605-635`; `paraphina/src/live/connectors/extended.rs:805-817`; `paraphina/src/live/connectors/extended.rs:1579-1629`; `paraphina/src/live/connectors/extended.rs:1673-1712`; `paraphina/src/live/connectors/extended.rs:1791-1820`; `paraphina/src/live/connectors/aster.rs:45-67`; `paraphina/src/live/connectors/aster.rs:160-169`; `paraphina/src/live/connectors/aster.rs:235-245`; `paraphina/src/live/connectors/aster.rs:370-383`; `paraphina/src/live/connectors/aster.rs:375-376`; `paraphina/src/live/connectors/aster.rs:412-426`; `paraphina/src/live/connectors/aster.rs:518-579`; `paraphina/src/live/connectors/aster.rs:686-735`; `paraphina/src/live/connectors/aster.rs:762-779`; `paraphina/src/live/connectors/aster.rs:927-929`; `paraphina/src/live/connectors/aster.rs:1461-1493`; `paraphina/src/live/connectors/aster.rs:1514-1536`; `paraphina/src/live/connectors/aster.rs:1539-1553`; `paraphina/src/live/connectors/aster.rs:963-979`; `paraphina/src/live/connectors/aster.rs:1628-1643`; `paraphina/src/live/connectors/paradex.rs:12-33`; `paraphina/src/live/connectors/paradex.rs:210-219`; `paraphina/src/live/connectors/paradex.rs:352-360`; `paraphina/src/live/connectors/paradex.rs:365-383`; `paraphina/src/live/connectors/paradex.rs:396-401`; `paraphina/src/live/connectors/paradex.rs:435-477`; `paraphina/src/live/connectors/paradex.rs:500-503`; `paraphina/src/live/connectors/paradex.rs:556-586`; `paraphina/src/live/connectors/paradex.rs:609-621`; `paraphina/src/live/connectors/paradex.rs:878-940`; `paraphina/src/live/connectors/paradex.rs:1391-1406`; `paraphina/src/live/connectors/paradex.rs:1418-1433`; `paraphina/src/live/connectors/paradex.rs:1474-1532`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "connect|connect_async|timeout|read_timeout|stale|watchdog|ping|pong|subscribe|snapshot|delta|sequence|seq|timestamp|event_time|MarketPublisher" -S paraphina/src/live/connectors/hyperliquid.rs
cd /home/developer/code/paraphina && rg -n "connect|connect_async|timeout|read_timeout|stale|watchdog|ping|pong|subscribe|snapshot|delta|sequence|seq|timestamp|event_time|MarketPublisher" -S paraphina/src/live/connectors/lighter.rs
cd /home/developer/code/paraphina && rg -n "connect|connect_async|timeout|read_timeout|stale|watchdog|ping|pong|subscribe|snapshot|delta|sequence|seq|timestamp|event_time|MarketPublisher" -S paraphina/src/live/connectors/extended.rs
cd /home/developer/code/paraphina && rg -n "connect|connect_async|timeout|read_timeout|stale|watchdog|ping|pong|subscribe|snapshot|delta|sequence|seq|timestamp|event_time|MarketPublisher" -S paraphina/src/live/connectors/aster.rs
cd /home/developer/code/paraphina && rg -n "connect|connect_async|timeout|read_timeout|stale|watchdog|ping|pong|subscribe|snapshot|delta|sequence|seq|timestamp|event_time|MarketPublisher" -S paraphina/src/live/connectors/paradex.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1,120p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '568,792p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/extended.rs | sed -n '394,522p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/aster.rs | sed -n '370,462p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '352,478p'
```

---

## Publishing & Backpressure

`MarketPublisher` behavior:
- Internal bounded queue with configurable `queue_cap`; each loop drains up to `drain_max` extra events with `try_recv` before forwarding.
- In fixture mode, it bypasses the queue and writes directly to `out_tx` (awaited send).
- Lossless events use awaited `.send()` into the queue (can block when full).
- Non-lossless events use `try_send`; on full, one `pending_latest` slot is overwritten (newest wins).

Lossless event classification in venue connectors:
- Lighter / Extended / Aster / Paradex mark only L2 events (`L2Delta`, `L2Snapshot`) as lossless.

Additional drop/overwrite points:
- Hyperliquid bypasses `MarketPublisher`, using its own internal `try_send` queue (`HL_INTERNAL_PUB_Q=256`) and one-slot overwrite (`pending_latest`).
- Paper tap in ingest bridge (`paper_market_tx.try_send`) drops when full.
- Runner order intent request path (`order_tx.try_send`) drops intents when full.
- Runner account reconcile request path (`account_reconcile_tx.try_send`) drops request when full.

Runner delta buffer cap behavior (`PARAPHINA_L2_TICK_DELTA_BUFFER_MAX`):
- Unready venue deltas: buffered until cap, then dropped for that venue in the tick.
- Ready venue deltas: buffered until cap; after cap they are emitted immediately (not dropped) unless snapshot-dominated.

**Evidence**
Files: `paraphina/src/live/market_publisher.rs:54-101`; `paraphina/src/live/market_publisher.rs:114-156`; `paraphina/src/live/connectors/lighter.rs:366-376`; `paraphina/src/live/connectors/extended.rs:263-273`; `paraphina/src/live/connectors/aster.rs:235-245`; `paraphina/src/live/connectors/paradex.rs:210-219`; `paraphina/src/live/connectors/hyperliquid.rs:13-15`; `paraphina/src/live/connectors/hyperliquid.rs:605-637`; `paraphina/src/bin/paraphina_live.rs:1763-1767`; `paraphina/src/live/runner.rs:374-403`; `paraphina/src/live/runner.rs:423-443`; `paraphina/src/live/runner.rs:558-585`; `paraphina/src/live/runner.rs:586-631`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "struct MarketPublisher|lossless|try_send|pending_latest|send\(" -S paraphina/src/live/market_publisher.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/market_publisher.rs | sed -n '1,220p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '536,692p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '360,426p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '440,860p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1748,1792p'
```

---

## Runner Ordering/Apply

Ordering and partition:
- Tick wake-up is event-driven: interval tick or incoming market event; early market wakeups push one event into `pending_events` and can reset interval.
- `drain_ordered_events` canonicalizes into `OrderedEvent` and sorts by `(venue_index, source_seq, event_ts_ms, type_order)`.
- Runner partitions events by time: `event_ts_ms <= now_ms` goes to ordered apply now; future timestamps are deferred in `pending_events`.

Apply pipeline:
- Market event path is `cache.apply_market_event(&event)` then `apply_market_event_to_core(...)` on cache success.
- Core apply routes to `state.venues[*].apply_l2_snapshot/delta(...)` with `max_levels = cfg.book.depth_levels.max(1)`.

Extended apply-age semantics:
- `PARAPHINA_EXTENDED_APPLY_AGE_ON_ANY_L2` is parsed in runner (default false).
- If true and venue index is `EXTENDED_IDX=0`, `last_mid_apply_ms` is set on any successful L2 apply; otherwise it is set only when both `mid` and `spread` are present.

Shared venue ages write semantics:
- After apply each tick, runner writes `age = now_ms - last_mid_apply_ms` into `SharedVenueAges`.
- Unknown `last_mid_apply_ms` writes `i64::MAX` sentinel.
- Runner also calls `ages.mark_write(now_ms)` heartbeat each tick.

**Evidence**
Files: `paraphina/src/live/runner.rs:1122-1158`; `paraphina/src/live/runner.rs:465-820`; `paraphina/src/live/runner.rs:1326-1353`; `paraphina/src/live/runner.rs:1361-1450`; `paraphina/src/live/runner.rs:1409-1439`; `paraphina/src/live/runner.rs:1038-1040`; `paraphina/src/live/runner.rs:1099-1102`; `paraphina/src/live/runner.rs:2568-2613`; `paraphina/src/live/runner.rs:1768-1777`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "drain_.*ordered|future|ordered|L2_TICK_DELTA_BUFFER_MAX|drop|apply_market_event|apply_market_event_to_core|cache\.apply|shared_venue_ages|EXTENDED_APPLY_AGE" -S paraphina/src/live/runner.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1122,1196p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '440,860p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1296,1468p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1008,1125p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '2560,2708p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '1660,1798p'
```

---

## State + Telemetry

State write semantics:
- `apply_l2_snapshot` and `apply_l2_delta` both call `orderbook_l2.trim_levels(max_levels)` after applying book updates.
- `last_mid_update_ms` is written only when computed `metrics.mid` and `metrics.spread` are both present.
- `last_mid_apply_ms` is not set in `state.rs`; runtime writes happen in runner core apply path.

Trim algorithm:
- `trim_levels(0)` clears both sides.
- Otherwise each side is independently truncated to `max_levels` (`bids.truncate(max_levels)`, `asks.truncate(max_levels)`).
- If a side already has fewer levels, it is unchanged.

Freshness fields in telemetry/watch:
- `compute_age_ms(now, None) == -1`; if timestamp is in the future, value clamps to `0`.
- `venue_age_ms` is computed from `last_mid_apply_ms` (apply-age).
- `venue_age_event_ms` is computed from `last_mid_update_ms` (event-age).
- `paraphina_watch.py` renders both columns (`age_ms`, optional `age_event_ms`) from telemetry fields `venue_age_ms` and `venue_age_event_ms`.

**Evidence**
Files: `paraphina/src/state.rs:296-347`; `paraphina/src/state.rs:185-188`; `paraphina/src/live/runner.rs:2580-2613`; `paraphina/src/orderbook_l2.rs:122-133`; `paraphina/src/telemetry.rs:828-837`; `paraphina/src/telemetry.rs:1641-1661`; `paraphina/src/telemetry.rs:1717-1720`; `tools/paraphina_watch.py:622-623`; `tools/paraphina_watch.py:688-692`; `tools/paraphina_watch.py:727-730`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "last_mid_update_ms|last_mid_apply_ms|apply_l2_snapshot|apply_l2_delta|trim_levels|mid|spread" -S paraphina/src/state.rs
cd /home/developer/code/paraphina && rg -n "fn trim_levels|trim_levels\(|truncate\(" -S paraphina/src/orderbook_l2.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/state.rs | sed -n '168,360p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/orderbook_l2.rs | sed -n '96,152p'
cd /home/developer/code/paraphina && rg -n "compute_age_ms|venue_age_ms|venue_age_event_ms|last_mid_apply_ms|last_mid_update_ms" -S paraphina/src/telemetry.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/telemetry.rs | sed -n '812,944p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/telemetry.rs | sed -n '1600,1736p'
cd /home/developer/code/paraphina && rg -n "age_ms|age_event_ms|venue_age_ms|venue_age_event_ms" -S tools/paraphina_watch.py
cd /home/developer/code/paraphina && nl -ba tools/paraphina_watch.py | sed -n '596,760p'
```

---

## Layer A/B Health

SharedVenueAges semantics:
- Initialized to `i64::MAX` per venue (unknown/uninitialized).
- Runner writes apply-age values per venue each tick; negative age maps to `i64::MAX`.
- `last_write_ms` heartbeat tracks runner liveness (`runner_idle_ms`).

Layer A (`VenueHealthEnforcer`):
- Polls every 5s.
- Defaults: `force_restart_ms=90_000` (`PARAPHINA_FORCE_RESTART_MS`), cooldown 30s (`PARAPHINA_ENFORCER_COOLDOWN_SECS`).
- Unknown age (`i64::MAX`) is skipped (no restart until first real value).
- If age exceeds threshold and cooldown passed, aborts connector task and respawns via stored closure.
- Also logs runner-stuck warning when `runner_idle_ms > 30_000`.

Layer B (`RestHealthMonitor`):
- Polls every 5s.
- Default threshold `20_000ms` (`PARAPHINA_REST_MONITOR_THRESHOLD_MS`).
- Unknown age (`i64::MAX`) is remapped to elapsed time since monitor start (so fallback can eventually activate even with no writes).
- Fetch timeout is 5s; successful fetch injects `MarketDataEvent` into provided market channel via awaited send.

Wiring and injection path:
- `paraphina_live` creates `SharedVenueAges` once and passes clone to runner, enforcer, and rest monitor.
- Rest monitor is intentionally wired to `market_ingest_tx` (not direct runner `market_tx`) so injected events traverse ingest bridge logic (timestamp override/paper tap).

Per-venue REST fetchers (Layer B):
- In `rest_health_monitor.rs`: `fetch_extended_l2_snapshot`, `fetch_lighter_l2_snapshot`, `fetch_aster_l2_snapshot`, `fetch_paradex_l2_snapshot`.
- Hyperliquid is wired through `paraphina_live` using `connectors::hyperliquid::fetch_l2_snapshot` in a `VenueRestEntry` closure.

Asymmetry note:
- Hyperliquid also runs connector-local REST fallback (`run_rest_book_fallback`) in parallel to central Layer B monitor.

**Evidence**
Files: `paraphina/src/live/shared_venue_ages.rs:32-77`; `paraphina/src/live/shared_venue_ages.rs:44-49`; `paraphina/src/live/runner.rs:1768-1777`; `paraphina/src/live/venue_health_enforcer.rs:40-55`; `paraphina/src/live/venue_health_enforcer.rs:74-119`; `paraphina/src/live/rest_health_monitor.rs:61-69`; `paraphina/src/live/rest_health_monitor.rs:147-155`; `paraphina/src/live/rest_health_monitor.rs:156-223`; `paraphina/src/bin/paraphina_live.rs:1832-1838`; `paraphina/src/bin/paraphina_live.rs:2748-2775`; `paraphina/src/bin/paraphina_live.rs:2765-2769`; `paraphina/src/live/rest_health_monitor.rs:237-376`; `paraphina/src/bin/paraphina_live.rs:1994-2015`; `paraphina/src/bin/paraphina_live.rs:2026-2029`; `paraphina/src/live/connectors/hyperliquid.rs:1013-1077`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "SharedVenueAges|i64::MAX|unknown|VenueHealthEnforcer|restart|threshold" -S paraphina/src/live/shared_venue_ages.rs paraphina/src/live/venue_health_enforcer.rs
cd /home/developer/code/paraphina && rg -n "RestHealthMonitor|market_ingest_tx|fetch_.*_l2_snapshot|unknown|i64::MAX" -S paraphina/src/live/rest_health_monitor.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/shared_venue_ages.rs | sed -n '1,170p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/venue_health_enforcer.rs | sed -n '1,220p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '1,240p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '232,380p'
cd /home/developer/code/paraphina && rg -n "enforcer_slots|rest_entries|fetch_.*_l2_snapshot|rest_fallback" -S paraphina/src/bin/paraphina_live.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1948,2044p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '2738,2798p'
```

---

## Config/Env/Workflow Knobs

Precedence model used by this binary:
1. `resolve_effective_profile(None, None)` resolves profile source (CLI/env/scenario/default rules live in config module).
2. `Config::from_env_or_profile(profile)` builds config defaults and applies config-level env overrides.
3. Connector `*Config::from_env()` reads connector env vars directly.
4. Workflow dispatch inputs can set env vars for CI runtime.

### Upstream subscription knobs

| Knob | Default | Parsed at | Effect | Stage | Workflow wiring |
|---|---|---|---|---|---|
| `HL_L2_SIGFIGS` | `5` | `hyperliquid.rs:261-264` | Hyperliquid WS/REST l2Book precision | Connector | None |
| `HL_L2_LEVELS` | `20` | `hyperliquid.rs:265-268` | Hyperliquid WS/REST l2Book depth levels | Connector | None |
| `PARAPHINA_EXTENDED_WS_DEPTH_LEVELS` | `1` | `extended.rs:201-205` | Extended WS URL mode (`?depth=1` vs full orderbooks stream) | Connector | `ws_shadow_soak.yml:32-33`, `:82`, `:98` |
| `PARAPHINA_PARADEX_PUBLIC_FEED` | `bbo` | `paradex.rs:365-366` | Paradex feed mode (`bbo` vs `orderbook`) | Connector | `ws_shadow_soak.yml:24-25`, `:81`, `:97` |
| `EXTENDED_DEPTH_LIMIT` | `100` | `extended.rs:168-171` | REST snapshot depth limit for Extended | Connector | None |
| `ASTER_DEPTH_LIMIT` | `100` | `aster.rs:166-169` | REST snapshot depth limit for Aster | Connector | None |

### Canonicalization knobs

| Knob | Default | Parsed at | Effect | Stage | Workflow wiring |
|---|---|---|---|---|---|
| `cfg.book.depth_levels` | `10` | `config.rs:742`; consumed `runner.rs:2576` | Internal canonical max per-side levels (trim) | Runner/State | No workflow input |
| `PARAPHINA_EXTENDED_APPLY_AGE_ON_ANY_L2` | `false` | `runner.rs:1038-1040` | Extended apply-age update on any successful L2 apply | Runner | `ws_shadow_soak.yml:36-37`, `:57` |
| `PARAPHINA_L2_DELTA_COALESCE` | `true` | `runner.rs:1047-1049` | Enables tick delta coalescing | Runner | None |
| `PARAPHINA_L2_SNAPSHOT_COALESCE` | `true` | `runner.rs:1050-1052` | Enables tick snapshot coalescing | Runner | None |
| `PARAPHINA_L2_TICK_DELTA_BUFFER_MAX` | unset (no cap) | `runner.rs:1099-1102` | Per-venue delta buffer cap per tick; affects drop/emit behavior | Runner | None |

### Freshness/health knobs

| Knob | Default | Parsed at | Effect | Stage | Workflow wiring |
|---|---|---|---|---|---|
| `cfg.book.stale_ms` | `1000` | `config.rs:743` | Global stale threshold baseline | Health/Telemetry | No workflow input |
| `VenueConfig.stale_ms_override` | varies by venue | `config.rs:642-735` | Per-venue stale threshold override | Health/Telemetry | No workflow input |
| `PARAPHINA_HL_STATE_STALE_MS_OVERRIDE` | unset | `config.rs:1345-1353` | Overrides Hyperliquid state stale threshold | Health/Telemetry | None |
| `PARAPHINA_EXTENDED_STATE_STALE_MS_OVERRIDE` | unset | `config.rs:1369-1377` | Overrides Extended state stale threshold | Health/Telemetry | None |
| `PARAPHINA_PARADEX_STATE_STALE_MS_OVERRIDE` | unset | `config.rs:1392-1400` | Overrides Paradex state stale threshold | Health/Telemetry | None |
| `PARAPHINA_CATASTROPHIC_STALE_MS` | `120000` | `config.rs:1433-1439`; default `config.rs:938` | Toxicity fail-closed catastrophic stale threshold | Health | None |
| `PARAPHINA_FORCE_RESTART_MS` | `90000` | `venue_health_enforcer.rs:43-46` | Layer A forced restart threshold | Health Layer A | None |
| `PARAPHINA_ENFORCER_COOLDOWN_SECS` | `30` | `venue_health_enforcer.rs:48-52` | Layer A restart cooldown | Health Layer A | None |
| `PARAPHINA_REST_MONITOR_THRESHOLD_MS` | `20000` | `rest_health_monitor.rs:64-67` | Layer B REST fallback activation threshold | Health Layer B | None |
| `PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS` | unset | `paraphina_live.rs:558-588`; `runner.rs:139-148` | Enables/intervals account reconcile path | Runner/Health | None |
| `PARAPHINA_LIVE_ACCOUNT_POLL_MS` | `5000` | `runner.rs:981-985`; connector spawn uses `paraphina_live.rs:1951`, `2168`, `2367`, `2532`, `2698` | Account snapshot max-age + account polling cadence | Runner/Connectors | None |
| `PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF` | unset | `paraphina_live.rs:1661-1667` | Paper-mode override for `cfg.book.min_healthy_for_kf` | Health gate | None |
| `PARAPHINA_PAPER_USE_WALLCLOCK_TS` | false | `paraphina_live.rs:1755` | Overrides market event timestamps at ingest bridge | Ingest freshness semantics | None |
| `PARAPHINA_HL_STALE_MS` | `10000` | `hyperliquid.rs:35-39` | Hyperliquid WS stale watchdog threshold | Connector | None |
| `PARAPHINA_LIGHTER_STALE_MS` | `10000` | `lighter.rs:32-36` | Lighter WS stale watchdog threshold | Connector | None |
| `PARAPHINA_EXTENDED_STALE_MS` | `10000` | `extended.rs:28-33` | Extended WS stale watchdog threshold | Connector | None |
| `PARAPHINA_ASTER_STALE_MS` | `10000` | `aster.rs:62-67` | Aster WS stale watchdog threshold | Connector | None |
| `PARAPHINA_PARADEX_STALE_MS` | `10000` | `paradex.rs:28-33` | Paradex WS stale watchdog threshold | Connector | None |

### Observability knobs

| Knob | Default | Parsed at | Effect | Stage | Workflow wiring |
|---|---|---|---|---|---|
| `PARAPHINA_WS_AUDIT` | false | connector files + `runner.rs:1035-1037` + `market_publisher.rs:16-21` + `rest_health_monitor.rs:83-87` | Enables WS_AUDIT counters/logs | Observability | `ws_shadow_soak.yml:89` sets `1` |
| `PARAPHINA_MARKET_RX_STATS` | false | `runner.rs:1032-1034` | Enables market RX stats collection | Observability | `ws_shadow_soak.yml:90` sets `1` |
| `PARAPHINA_MARKET_RX_STATS_EVERY_TICKS` | `1` | `runner.rs:1041-1045` | Stats emission cadence | Observability | None |
| `PARAPHINA_MARKET_RX_STATS_PATH` | unset | `runner.rs:1046` | Optional stats file path | Observability | `ws_shadow_soak.yml:91` sets path |

### Buffer/drop and session control knobs

| Knob | Default | Parsed at | Effect | Stage | Workflow wiring |
|---|---|---|---|---|---|
| `PARAPHINA_WS_MAX_SESSION_SECS` | `86400` | all connector run loops (e.g. `lighter.rs:593-596`, `extended.rs:311-314`, `aster.rs:289-292`, `paradex.rs:272-275`, `hyperliquid.rs:474-479`) | Hard max session duration before reconnect | Connector | None |
| `PARAPHINA_WS_HEALTHY_THRESHOLD_MS` | `60000` | all connector run loops (e.g. `lighter.rs:591-631`, `extended.rs:311-336`, `aster.rs:289-312`, `paradex.rs:272-295`, `hyperliquid.rs:474-499`) | Backoff reset threshold after healthy session | Connector | None |
| `PARAPHINA_HL_WS_CONNECT_TIMEOUT_MS` | `15000` | `hyperliquid.rs:42-49` | Hyperliquid connect timeout | Connector | None |
| `PARAPHINA_HL_WS_READ_TIMEOUT_MS` | `30000` | `hyperliquid.rs:51-57` | Hyperliquid read timeout | Connector | None |
| `PARAPHINA_LIGHTER_WS_CONNECT_TIMEOUT_MS` | `15000` | `lighter.rs:39-45` | Lighter connect timeout | Connector | None |
| `PARAPHINA_LIGHTER_WS_READ_TIMEOUT_MS` | `30000` | `lighter.rs:48-54` | Lighter read timeout | Connector | None |
| `PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS` | `10000` in code; workflow input default `45000` | `extended.rs:35-41`; `ws_shadow_soak.yml:20-23`, `:80`, `:96` | Extended read timeout | Connector | Wired |
| `PARAPHINA_LIGHTER_PING_INTERVAL_MS` | `30000` in code; workflow input default `10000` | `lighter.rs:57-62`; `ws_shadow_soak.yml:16-19`, `:79`, `:95` | Lighter ping cadence | Connector | Wired |
| `PARAPHINA_EXTENDED_PING_INTERVAL_MS` | `30000` | `extended.rs:496-500` | Extended ping cadence | Connector | None |
| `PARAPHINA_ASTER_PING_INTERVAL_MS` | `30000` | `aster.rs:412-416` | Aster ping cadence | Connector | None |
| `PARAPHINA_PARADEX_PING_INTERVAL_MS` | `30000` | `paradex.rs:397-401` | Paradex ping cadence | Connector | None |
| `PARAPHINA_LIGHTER_WS_READONLY` | false | `lighter.rs:70-75` | Adds `readonly=true` to Lighter public WS URL | Connector | `ws_shadow_soak.yml:94` sets `1` |

**Evidence**
Files: `paraphina/src/bin/paraphina_live.rs:1622-1624`; `paraphina/src/config.rs:86-103`; `paraphina/src/config.rs:137-155`; `paraphina/src/config.rs:1038-1047`; `paraphina/src/config.rs:1341-1400`; `paraphina/src/config.rs:742-743`; `paraphina/src/live/runner.rs:1032-1052`; `paraphina/src/live/runner.rs:1099-1102`; `paraphina/src/live/runner.rs:2576`; `paraphina/src/live/connectors/hyperliquid.rs:261-268`; `paraphina/src/live/connectors/extended.rs:201-218`; `paraphina/src/live/connectors/paradex.rs:365-369`; `.github/workflows/ws_shadow_soak.yml:6-40`; `.github/workflows/ws_shadow_soak.yml:57`; `.github/workflows/ws_shadow_soak.yml:79-99`
- Supplemental knob anchors: `paraphina/src/config.rs:642-735`; `paraphina/src/config.rs:938`; `paraphina/src/config.rs:1433-1439`; `paraphina/src/live/connectors/extended.rs:168-171`; `paraphina/src/live/connectors/lighter.rs:70-75`; `paraphina/src/live/market_publisher.rs:16-21`; `paraphina/src/bin/paraphina_live.rs:558-588`; `paraphina/src/bin/paraphina_live.rs:1661-1667`; `paraphina/src/bin/paraphina_live.rs:1951-1954`; `paraphina/src/live/rest_health_monitor.rs:83-87`; `paraphina/src/live/runner.rs:139-148`; `paraphina/src/live/runner.rs:981-985`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "PARAPHINA_[A-Z0-9_]+|env::var|depth_levels|book\.depth_levels|timeout|stale|buffer|cap|audit|WS_AUDIT" -S paraphina/src/config.rs paraphina/src/bin/paraphina_live.rs paraphina/src/live
cd /home/developer/code/paraphina && rg -n "workflow_dispatch|inputs:|paradex_public_feed|extended_ws_depth_levels|extended_apply_age_on_any_l2" -S .github/workflows/ws_shadow_soak.yml
cd /home/developer/code/paraphina && nl -ba .github/workflows/ws_shadow_soak.yml | sed -n '1,170p'
cd /home/developer/code/paraphina && rg -n "depth_levels" -S paraphina/src/config.rs paraphina/src/bin/paraphina_live.rs paraphina/src/live/runner.rs paraphina/src/state.rs
cd /home/developer/code/paraphina && nl -ba paraphina/src/config.rs | sed -n '620,760p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/config.rs | sed -n '1338,1458p'
```

---

## Known Non-Uniformities

1. Hyperliquid does not use shared `MarketPublisher`; it uses a connector-local queue with overwrite-on-full semantics. Other four venues use `MarketPublisher`.
2. Aster has two stale-time notions in `public_ws_once`: env-driven `PARAPHINA_ASTER_STALE_MS` watchdog and an additional local watchdog constant `STALE_MS=2000` branch.
3. Extended stamps the initial REST snapshot event with local `now_ms()` instead of exchange event time.
4. Paradex in `bbo` mode synthesizes seq and timestamp locally, unlike orderbook mode which uses exchange seq fields.
5. Lighter has in-file comment drift risk: one comment says “always snapshot” for channel decode helper, while live path explicitly switches to delta mode after first snapshot.
6. Layer B fetcher implementations are asymmetric: centralized monitor has explicit fetchers for lighter/extended/aster/paradex; Hyperliquid uses connector function wiring and also runs its own connector-local REST fallback.

**Evidence**
Files: `paraphina/src/live/connectors/hyperliquid.rs:605-637`; `paraphina/src/live/connectors/lighter.rs:366-376`; `paraphina/src/live/connectors/aster.rs:409-426`; `paraphina/src/live/connectors/aster.rs:612-621`; `paraphina/src/live/connectors/extended.rs:448-453`; `paraphina/src/live/connectors/paradex.rs:365-369`; `paraphina/src/live/connectors/paradex.rs:1513-1529`; `paraphina/src/live/connectors/lighter.rs:1491-1494`; `paraphina/src/live/connectors/lighter.rs:921-947`; `paraphina/src/live/rest_health_monitor.rs:237-376`; `paraphina/src/bin/paraphina_live.rs:1994-2015`; `paraphina/src/bin/paraphina_live.rs:2026-2029`
Commands:
```bash
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '536,692p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/aster.rs | sed -n '370,462p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/aster.rs | sed -n '518,634p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/extended.rs | sed -n '394,456p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '352,383p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1474,1534p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1491,1540p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '921,953p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '232,380p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1948,2044p'
```

---

## Failure Modes & Restart Triggers

Connector-level reconnect triggers (venue-specific loops):
- session timeout (`PARAPHINA_WS_MAX_SESSION_SECS`), connect timeout/read timeout, stale watchdog expiry, ping send failure.
- parsing/sequence triggers: Extended seq mismatch/gap and parse-error threshold; Aster seq gaps and stale watchdog; Paradex seq mismatch/gap; Lighter repeated delta decode failures.

Layer A trigger:
- Shared apply-age reaches/exceeds `PARAPHINA_FORCE_RESTART_MS` and is not unknown (`i64::MAX`), plus cooldown satisfied.

Layer B trigger:
- Shared apply-age reaches/exceeds `PARAPHINA_REST_MONITOR_THRESHOLD_MS` (with unknown ages mapped to monitor-elapsed time), then REST snapshot fetch/injection attempted.

Non-restart but data-loss/degradation triggers:
- `MarketPublisher` non-lossless full queue overwrites `pending_latest`.
- Runner unready delta cap can drop deltas for a venue within tick.
- `order_tx`/`account_reconcile_tx` full causes dropped requests.

**Evidence**
Files: `paraphina/src/live/connectors/hyperliquid.rs:474-499`; `paraphina/src/live/connectors/hyperliquid.rs:657-679`; `paraphina/src/live/connectors/lighter.rs:591-631`; `paraphina/src/live/connectors/lighter.rs:747-787`; `paraphina/src/live/connectors/lighter.rs:995-1001`; `paraphina/src/live/connectors/extended.rs:311-336`; `paraphina/src/live/connectors/extended.rs:605-635`; `paraphina/src/live/connectors/extended.rs:805-817`; `paraphina/src/live/connectors/aster.rs:289-312`; `paraphina/src/live/connectors/aster.rs:758-778`; `paraphina/src/live/connectors/aster.rs:835-849`; `paraphina/src/live/connectors/paradex.rs:272-295`; `paraphina/src/live/connectors/paradex.rs:435-477`; `paraphina/src/live/connectors/paradex.rs:613-620`; `paraphina/src/live/venue_health_enforcer.rs:90-119`; `paraphina/src/live/rest_health_monitor.rs:151-214`; `paraphina/src/live/market_publisher.rs:136-151`; `paraphina/src/live/runner.rs:563-580`; `paraphina/src/live/runner.rs:395-403`; `paraphina/src/live/runner.rs:440-444`
Commands:
```bash
cd /home/developer/code/paraphina && rg -n "session_timeout|read_timeout|stale_watchdog|ping_send_fail|seq_gap|seq_mismatch|decode_fail_loop" -S paraphina/src/live/connectors
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/venue_health_enforcer.rs | sed -n '74,121p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/rest_health_monitor.rs | sed -n '144,214p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/market_publisher.rs | sed -n '125,152p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '360,426p'
cd /home/developer/code/paraphina && nl -ba paraphina/src/live/runner.rs | sed -n '558,631p'
```

---

## Verification Playbook

1. Snapshot repo state.
```bash
cd /home/developer/code/paraphina
git rev-parse HEAD
git status --porcelain
rg -n "WebSocket|WS|MarketPublisher|VenueHealthEnforcer|RestHealthMonitor|SharedVenueAges" -S paraphina/src docs/INVESTIGATIONS .github/workflows tools | head
```

2. Validate channel topology and ingress bridge.
```bash
cd /home/developer/code/paraphina
rg -n "market_ingest_tx|market_tx|spawn_connector_forwarders|LiveChannels|shared_venue_ages|VenueHealthEnforcer|RestHealthMonitor" -S paraphina/src/bin/paraphina_live.rs
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '1728,1848p'
nl -ba paraphina/src/bin/paraphina_live.rs | sed -n '920,990p'
```

3. Validate publish/drop semantics.
```bash
cd /home/developer/code/paraphina
rg -n "struct MarketPublisher|lossless|try_send|pending_latest|send\(" -S paraphina/src/live/market_publisher.rs
nl -ba paraphina/src/live/market_publisher.rs | sed -n '54,157p'
```

4. Validate runner ordering + apply-age semantics.
```bash
cd /home/developer/code/paraphina
rg -n "future|ordered|L2_TICK_DELTA_BUFFER_MAX|apply_market_event_to_core|shared_venue_ages|PARAPHINA_EXTENDED_APPLY_AGE_ON_ANY_L2" -S paraphina/src/live/runner.rs
nl -ba paraphina/src/live/runner.rs | sed -n '1326,1450p'
nl -ba paraphina/src/live/runner.rs | sed -n '2568,2613p'
```

5. Validate telemetry/watch mapping.
```bash
cd /home/developer/code/paraphina
rg -n "compute_age_ms|venue_age_ms|venue_age_event_ms" -S paraphina/src/telemetry.rs
rg -n "age_ms|age_event_ms|venue_age_ms|venue_age_event_ms" -S tools/paraphina_watch.py
```

6. Validate Layer A/B and workflow wiring.
```bash
cd /home/developer/code/paraphina
rg -n "SharedVenueAges|i64::MAX|force_restart_ms|PARAPHINA_REST_MONITOR_THRESHOLD_MS|fetch_.*_l2_snapshot" -S paraphina/src/live/shared_venue_ages.rs paraphina/src/live/venue_health_enforcer.rs paraphina/src/live/rest_health_monitor.rs
nl -ba .github/workflows/ws_shadow_soak.yml | sed -n '1,110p'
```

---

## Open Questions / TODOs

1. UNKNOWN: runtime env values in production outside this repo (actual thresholds/timeouts/depth choices) cannot be proven from source alone.
- Missing evidence: deployment manifests / runtime environment dump.
- Suggested command (outside repo scope): `env | rg '^PARAPHINA_|^HL_|^LIGHTER_|^EXTENDED_|^ASTER_|^PARADEX_'` on target runtime.

2. UNKNOWN: external process supervision policy for `paraphina_live` (systemd/Kubernetes/job runner) is not present in this repo.
- Missing evidence: infra deployment configs.
- Suggested command: inspect deployment repo or host service configs (`systemctl cat ...` / k8s manifests).

3. UNKNOWN: behavior when live_* connector features are disabled at compile time for a given build artifact.
- Missing evidence: build command for that artifact and enabled feature list.
- Suggested command: inspect build invocation / CI pipeline for that artifact.
