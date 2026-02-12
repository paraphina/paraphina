# WS Feed Depth Current State (Main)

## Scope
- Goal: map current WebSocket feed depth behavior for `hyperliquid`, `lighter`, `extended`, `aster`, `paradex` on `main`, and separate three different depth concepts:
1. subscription depth (wire request/feed shape),
2. retention depth (`cfg.book.depth_levels` cap),
3. consumption depth (how retained book data is consumed downstream).

## Evidence Commands
- `Cmd C1`
```bash
rg -n 'subscribe|subscription|orderbook|order_book|l2Book|bbo\.|depth=|@depth@|snapshot|delta|trim_levels|depth_levels' paraphina/src -S
```
- `Cmd C2`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '180,280p'
```
- `Cmd C3`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '540,590p'
```
- `Cmd C4`
```bash
nl -ba paraphina/src/live/connectors/hyperliquid.rs | sed -n '1368,1425p'
```
- `Cmd C5`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '700,740p'
```
- `Cmd C6`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '920,980p'
```
- `Cmd C7`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1495,1530p'
```
- `Cmd C8`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1530,1590p'
```
- `Cmd C9`
```bash
nl -ba paraphina/src/live/connectors/lighter.rs | sed -n '1608,1625p'
```
- `Cmd C10`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '146,210p'
```
- `Cmd C11`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '380,452p'
```
- `Cmd C12`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1410,1475p'
```
- `Cmd C13`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1508,1565p'
```
- `Cmd C14`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '1630,1680p'
```
- `Cmd C15`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '910,928p'
```
- `Cmd C16`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '140,190p'
```
- `Cmd C17`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '194,208p'
```
- `Cmd C18`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '360,410p'
```
- `Cmd C19`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '492,548p'
```
- `Cmd C20`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '960,980p'
```
- `Cmd C21`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1534,1560p'
```
- `Cmd C22`
```bash
nl -ba paraphina/src/live/connectors/aster.rs | sed -n '1620,1660p'
```
- `Cmd C23`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '356,375p'
```
- `Cmd C24`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1438,1468p'
```
- `Cmd C25`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1244,1275p'
```
- `Cmd C26`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1360,1425p'
```
- `Cmd C27`
```bash
nl -ba tests/fixtures/roadmap_b/paradex_live_recording/ws_frames.jsonl | sed -n '1,60p'
```
- `Cmd C28`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '2348,2392p'
```
- `Cmd C29`
```bash
nl -ba paraphina/src/state.rs | sed -n '292,338p'
```
- `Cmd C30`
```bash
nl -ba paraphina/src/config.rs | sed -n '248,268p'
```
- `Cmd C31`
```bash
nl -ba paraphina/src/config.rs | sed -n '736,748p'
```
- `Cmd C32`
```bash
nl -ba paraphina/src/orderbook_l2.rs | sed -n '118,136p'
```
- `Cmd C33`
```bash
nl -ba paraphina/src/mm.rs | sed -n '260,274p'
```

## Depth Definitions (Three Different Depths)
- Subscription depth: whatever each connector asks for or receives on the wire (JSON subscribe payload or URL stream/path). This is venue-specific and can be explicit (`nLevels`, `depth=1`) or implicit (channel/stream without level count). Evidence: `paraphina/src/live/connectors/hyperliquid.rs:556-563`, `paraphina/src/live/connectors/extended.rs:200-205`, `paraphina/src/live/connectors/lighter.rs:714-718`, `paraphina/src/live/connectors/aster.rs:375-377`, `paraphina/src/live/connectors/paradex.rs:364-367` (`Cmd C3`, `Cmd C10`, `Cmd C5`, `Cmd C18`, `Cmd C23`).
- Retention depth: live core apply path sets `max_levels = cfg.book.depth_levels.max(1)`, then both snapshot and delta apply call `trim_levels(max_levels)` in `VenueState`; default `cfg.book.depth_levels` is `10`. Evidence: `paraphina/src/live/runner.rs:2352-2386`, `paraphina/src/state.rs:296-335`, `paraphina/src/config.rs:254-257`, `paraphina/src/config.rs:741-743`, `paraphina/src/orderbook_l2.rs:122-133` (`Cmd C28`, `Cmd C29`, `Cmd C30`, `Cmd C31`, `Cmd C32`).
- Consumption depth: downstream logic consumes derived outputs (for example `depth_near_mid`) from retained book state, and MM explicitly uses `depth_near_mid` as a liquidity proxy. Evidence: `paraphina/src/state.rs:308-314`, `paraphina/src/state.rs:334-339`, `paraphina/src/mm.rs:263-270` (`Cmd C29`, `Cmd C33`).

## Current Feed Depth Matrix
| Venue | Subscription channel/product | Wire depth param | Snapshot? | Delta? | Retention cap | Notes | Evidence (path:lines) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hyperliquid | JSON `method=subscribe`, `subscription.type=l2Book` | `nLevels` (from `HL_L2_LEVELS`, default `20`) | Yes (`data.levels` -> `L2Snapshot`) | Yes (`data.changes` -> `L2Delta`) | `cfg.book.depth_levels` (default `10`) | Wire depth is explicit and currently defaults above retention default. | `paraphina/src/live/connectors/hyperliquid.rs:556-563`, `paraphina/src/live/connectors/hyperliquid.rs:265-268`, `paraphina/src/live/connectors/hyperliquid.rs:1376-1415`, `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/state.rs:307-334`, `paraphina/src/config.rs:741-743` (`Cmd C3`, `Cmd C2`, `Cmd C4`, `Cmd C28`, `Cmd C29`, `Cmd C31`) |
| Lighter | JSON `type=subscribe`, `channel=order_book/{market_id}` | No explicit level-count field in subscribe frame | Yes (initial message path emits `L2Snapshot`) | Yes (post-initial path emits `L2Delta`) | `cfg.book.depth_levels` (default `10`) | Channel naming is explicit; parse path keys on incoming `order_book:` channels. | `paraphina/src/live/connectors/lighter.rs:714-718`, `paraphina/src/live/connectors/lighter.rs:1616-1617`, `paraphina/src/live/connectors/lighter.rs:929-953`, `paraphina/src/live/connectors/lighter.rs:1495-1528`, `paraphina/src/live/connectors/lighter.rs:1532-1586`, `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/config.rs:741-743` (`Cmd C5`, `Cmd C9`, `Cmd C6`, `Cmd C7`, `Cmd C8`, `Cmd C28`, `Cmd C31`) |
| Extended | URL-path subscription `.../orderbooks/{market}?depth=1` | `depth=1` in WS URL | Yes (startup REST snapshot publish; WS snapshot parse helper exists) | Yes (`depthUpdate` -> `L2Delta`) | `cfg.book.depth_levels` (default `10`) | WS path is hardcoded to one level; REST snapshot call independently uses `limit={depth_limit}`. | `paraphina/src/live/connectors/extended.rs:200-205`, `paraphina/src/live/connectors/extended.rs:388-443`, `paraphina/src/live/connectors/extended.rs:913-917`, `paraphina/src/live/connectors/extended.rs:1435-1471`, `paraphina/src/live/connectors/extended.rs:1515-1559`, `paraphina/src/live/connectors/extended.rs:1633-1677`, `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/config.rs:741-743` (`Cmd C10`, `Cmd C11`, `Cmd C15`, `Cmd C12`, `Cmd C13`, `Cmd C14`, `Cmd C28`, `Cmd C31`) |
| Aster | URL-path stream `.../{symbol}@depth@100ms` | No explicit WS level-count; stream encodes depth-update cadence (`100ms`) | Yes (REST `/fapi/v1/depth?...&limit={depth_limit}` then publish snapshot) | Yes (`depthUpdate` parse -> `L2Delta`) | `cfg.book.depth_levels` (default `10`) | WS level count is not explicit; REST snapshot limit is configurable (`ASTER_DEPTH_LIMIT`, default `100`). | `paraphina/src/live/connectors/aster.rs:375-377`, `paraphina/src/live/connectors/aster.rs:200-201`, `paraphina/src/live/connectors/aster.rs:963-967`, `paraphina/src/live/connectors/aster.rs:500-507`, `paraphina/src/live/connectors/aster.rs:542-547`, `paraphina/src/live/connectors/aster.rs:1628-1645`, `paraphina/src/live/connectors/aster.rs:1540-1554`, `paraphina/src/live/connectors/aster.rs:166-169`, `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/config.rs:741-743` (`Cmd C18`, `Cmd C17`, `Cmd C20`, `Cmd C19`, `Cmd C19`, `Cmd C22`, `Cmd C21`, `Cmd C16`, `Cmd C28`, `Cmd C31`) |
| Paradex | JSON-RPC subscribe to `channel=bbo.{market}` | No explicit level-count parameter; bbo decoder synthesizes one level per side | Yes (`bbo.*` decode emits synthetic `L2Snapshot`) | Current bbo path: no delta; separate orderbook parser supports snapshot+delta on `orderbook`/`order_book` channels | `cfg.book.depth_levels` (default `10`) | Current feed is bbo top-of-book; code can parse full orderbook channel shapes; fixtures include `orderbook` snapshot+delta frames. | `paraphina/src/live/connectors/paradex.rs:364-367`, `paraphina/src/live/connectors/paradex.rs:1459-1464`, `paraphina/src/live/connectors/paradex.rs:1363-1421`, `paraphina/src/live/connectors/paradex.rs:1246-1270`, `tests/fixtures/roadmap_b/paradex_live_recording/ws_frames.jsonl:1-2`, `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/config.rs:741-743` (`Cmd C23`, `Cmd C24`, `Cmd C26`, `Cmd C25`, `Cmd C27`, `Cmd C28`, `Cmd C31`) |

## Per-Venue Notes (Required)

### Extended
- WS orderbook URL is hardcoded with `depth=1`: `paraphina/src/live/connectors/extended.rs:200-205` (`Cmd C10`).
- Connector parses `depthUpdate` updates and emits `L2Delta`: `paraphina/src/live/connectors/extended.rs:1515-1559`, `paraphina/src/live/connectors/extended.rs:1435-1471` (`Cmd C13`, `Cmd C12`).

### Paradex
- Current WS subscription is `bbo.{market}` (`subscribe` params channel), not `orderbook`: `paraphina/src/live/connectors/paradex.rs:364-367`, `paraphina/src/live/connectors/paradex.rs:1459-1464` (`Cmd C23`, `Cmd C24`).
- `bbo.*` decode synthesizes 1-level bids/asks and emits `L2Snapshot`: `paraphina/src/live/connectors/paradex.rs:1379-1421` (`Cmd C26`).
- Code also supports `channel=orderbook` and `channel=order_book` with `snapshot`/`delta` message types: `paraphina/src/live/connectors/paradex.rs:1259-1270` (`Cmd C25`).
- Fixtures show `orderbook` snapshot and delta WS frames: `tests/fixtures/roadmap_b/paradex_live_recording/ws_frames.jsonl:1-2` (`Cmd C27`).

### Hyperliquid
- Public WS subscribe payload is `l2Book` and includes explicit `nLevels`: `paraphina/src/live/connectors/hyperliquid.rs:556-563` (`Cmd C3`).
- `nLevels` is sourced from config (`HL_L2_LEVELS`, default `20`): `paraphina/src/live/connectors/hyperliquid.rs:265-268` (`Cmd C2`).
- Decode path emits snapshots from `levels` and deltas from `changes`: `paraphina/src/live/connectors/hyperliquid.rs:1388-1415` (`Cmd C4`).

### Lighter
- Subscription channel is `order_book/{market_id}`: `paraphina/src/live/connectors/lighter.rs:714-718`, `paraphina/src/live/connectors/lighter.rs:1616-1617` (`Cmd C5`, `Cmd C9`).
- Runtime logic uses snapshot first, then delta mode: `paraphina/src/live/connectors/lighter.rs:929-953`, `paraphina/src/live/connectors/lighter.rs:963-968` (`Cmd C6`).
- Parsing paths are explicit (`order_book:` snapshot parser and delta parser): `paraphina/src/live/connectors/lighter.rs:1495-1503`, `paraphina/src/live/connectors/lighter.rs:1532-1586` (`Cmd C7`, `Cmd C8`).

### Aster
- WS stream naming is URL-path based: `"{symbol}@depth@100ms"`: `paraphina/src/live/connectors/aster.rs:375-377`, symbol lowercased via `stream_symbol()`: `paraphina/src/live/connectors/aster.rs:200-201` (`Cmd C18`, `Cmd C17`).
- No explicit WS level count is set in that stream string; REST snapshot uses `limit={depth_limit}` (`ASTER_DEPTH_LIMIT`, default `100`): `paraphina/src/live/connectors/aster.rs:963-967`, `paraphina/src/live/connectors/aster.rs:166-169` (`Cmd C20`, `Cmd C16`).
- Snapshot publish and delta publish paths are explicit: `paraphina/src/live/connectors/aster.rs:500-507`, `paraphina/src/live/connectors/aster.rs:542-547`, `paraphina/src/live/connectors/aster.rs:1540-1554`, `paraphina/src/live/connectors/aster.rs:1628-1645` (`Cmd C19`, `Cmd C21`, `Cmd C22`).
