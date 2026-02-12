# WS Uniform Depth Experiment Plan (Shadow-Only, Defaults Unchanged)

## Objective
- Standardize or deepen **wire subscription depth** where feasible, while keeping current runtime defaults unchanged.
- Primary target: drive wire depth to be `>= cfg.book.depth_levels` (default `10`) because live core apply already trims retained book levels to `cfg.book.depth_levels.max(1)` on both snapshots and deltas.
  - Evidence: `paraphina/src/live/runner.rs:2358-2385`, `paraphina/src/state.rs:296-335`, `paraphina/src/config.rs:741-743`, `paraphina/src/orderbook_l2.rs:122-133` (`Cmd P8`, `Cmd P9`, `Cmd P10`, `Cmd P11`).

## Evidence Commands
- `Cmd P1`
```bash
rg -n 'ws_shadow_soak|workflow_dispatch|duration_minutes|connectors|--gate|expected-connectors' .github tools -S
```
- `Cmd P2`
```bash
nl -ba .github/workflows/ws_shadow_soak.yml | sed -n '1,140p'
```
- `Cmd P3`
```bash
rg -n 'venue_age_ms|venue_age_event_ms|cap_hits|publisher|reconnect|plateau|coverage|gate|expected_connectors' tools/ws_soak_report.py -S
```
- `Cmd P4`
```bash
nl -ba tools/ws_soak_report.py | sed -n '1,220p'
```
- `Cmd P5`
```bash
nl -ba tools/ws_soak_report.py | sed -n '320,372p'
```
- `Cmd P6`
```bash
nl -ba tools/ws_soak_report.py | sed -n '420,446p'
```
- `Cmd P7`
```bash
nl -ba tools/ws_soak_report.py | sed -n '452,500p'
```
- `Cmd P8`
```bash
nl -ba paraphina/src/live/runner.rs | sed -n '2348,2392p'
```
- `Cmd P9`
```bash
nl -ba paraphina/src/state.rs | sed -n '292,338p'
```
- `Cmd P10`
```bash
nl -ba paraphina/src/config.rs | sed -n '736,748p'
```
- `Cmd P11`
```bash
nl -ba paraphina/src/orderbook_l2.rs | sed -n '118,136p'
```
- `Cmd P12`
```bash
nl -ba paraphina/src/live/connectors/extended.rs | sed -n '146,210p'
```
- `Cmd P13`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '356,375p'
```
- `Cmd P14`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1244,1275p'
```
- `Cmd P15`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1280,1330p'
```
- `Cmd P16`
```bash
nl -ba paraphina/src/live/connectors/paradex.rs | sed -n '1360,1425p'
```
- `Cmd P17`
```bash
nl -ba tests/fixtures/roadmap_b/paradex_live_recording/ws_frames.jsonl | sed -n '1,60p'
```
- `Cmd P18`
```bash
nl -ba tools/ws_soak_report.py | sed -n '515,598p'
```
- `Cmd P19`
```bash
nl -ba tools/ws_soak_report.py | sed -n '771,842p'
```

## Proven Shallow Venues (Current Code)
- Extended: WS orderbook URL is hardcoded as `.../orderbooks/{market}?depth=1`, so current wire depth is explicitly 1 level.
  - Evidence: `paraphina/src/live/connectors/extended.rs:200-205` (`Cmd P12`).
- Paradex: current public subscription uses `channel=bbo.{market}`; bbo decoder creates snapshot bids/asks as single-element vectors.
  - Evidence: `paraphina/src/live/connectors/paradex.rs:364-367`, `paraphina/src/live/connectors/paradex.rs:1404-1411`, `paraphina/src/live/connectors/paradex.rs:1413-1421` (`Cmd P13`, `Cmd P16`).

## Proposed Feature Flags (PROPOSED, Defaults Unchanged)

### 1) Paradex feed selector
- `PROPOSED`: `PARAPHINA_PARADEX_PUBLIC_FEED=orderbook|bbo` (default `bbo`).
- Behavior:
1. `bbo` keeps current behavior (no default change): subscribe `bbo.{market}` and synthesize single-level snapshots.
2. `orderbook` mode subscribes full book channel.
- Subscription params for proposed `orderbook` mode should use:
  - `channel="orderbook"`
  - `market="<symbol>"`
- Evidence for shape compatibility:
  - parser accepts `orderbook`/`order_book` channels and snapshot/delta types: `paraphina/src/live/connectors/paradex.rs:1259-1270` (`Cmd P14`);
  - snapshot/delta parsing requires `market`/`symbol`: `paraphina/src/live/connectors/paradex.rs:1281-1312` (`Cmd P15`);
  - fixture frames use `params.channel="orderbook"` and `params.market="BTC-USD-PERP"` for snapshot+delta: `tests/fixtures/roadmap_b/paradex_live_recording/ws_frames.jsonl:1-2` (`Cmd P17`).
- Inference: proposed subscribe params include `market` because parser+fixtures indicate it is part of valid orderbook frame shape.

### 2) Extended WS depth selector
- `PROPOSED`: `PARAPHINA_EXTENDED_WS_DEPTH_LEVELS=<int>` (default `1`).
- Wire mapping:
1. build WS URL as `.../orderbooks/{market}?depth=<PARAPHINA_EXTENDED_WS_DEPTH_LEVELS>`;
2. default remains `1`, preserving current behavior.
- Evidence: current URL is hardcoded to `depth=1`: `paraphina/src/live/connectors/extended.rs:200-205` (`Cmd P12`).

## CI Comparison Protocol (Using Existing Soak Workflow + Gate)
- Baseline (current defaults): 10 minutes.
  - Evidence for 10-minute option in workflow_dispatch: `.github/workflows/ws_shadow_soak.yml:8-15` (`Cmd P2`).
- Experimental (flags ON): 10 minutes.
- If experimental 10-minute run is clean, run experimental 90 minutes.
  - Evidence for 90-minute option in workflow_dispatch: `.github/workflows/ws_shadow_soak.yml:13-15` (`Cmd P2`).
- Use same connector set and gate invocation path:
  - runtime receives `--connectors` from workflow input: `.github/workflows/ws_shadow_soak.yml:24-27`, `.github/workflows/ws_shadow_soak.yml:62`, `.github/workflows/ws_shadow_soak.yml:78` (`Cmd P2`);
  - gate uses `tools/ws_soak_report.py --gate --expected-connectors`: `.github/workflows/ws_shadow_soak.yml:92-95` (`Cmd P2`).

## Metrics + Gates (Must Pass)
- `venue_age_ms` tails: apply-age p95 and p99 thresholds per venue.
  - Threshold constants: `APPLY_P95_MAX_MS=10000`, `APPLY_P99_MAX_MS=30000`: `tools/ws_soak_report.py:30-31` (`Cmd P4`).
  - Gate checks: `tools/ws_soak_report.py:533-542` (`Cmd P18`).
- `venue_age_event_ms` (diagnostic + gated by default): event-age p95 and p99 thresholds per venue.
  - telemetry parsing of event-age field: `tools/ws_soak_report.py:333-336`, `tools/ws_soak_report.py:364-366` (`Cmd P5`);
  - default gate requires event-age (`--require-event-age` default true): `tools/ws_soak_report.py:778-781` (`Cmd P19`);
  - event-age gate checks: `tools/ws_soak_report.py:567-581` (`Cmd P18`).
- `plateau@30s == 0s`:
  - plateau gate threshold is 30000ms: `tools/ws_soak_report.py:34` (`Cmd P4`);
  - any non-zero 30s plateau duration fails gate: `tools/ws_soak_report.py:544-551` (`Cmd P18`).
- Reconnect reasons within threshold:
  - reasons gated: `stale_watchdog`, `read_timeout`, `ping_send_fail`, `session_timeout`: `tools/ws_soak_report.py:36-41` (`Cmd P4`);
  - maximum allowed combined count per reason/venue is `3`: `tools/ws_soak_report.py:35`, `tools/ws_soak_report.py:554-563` (`Cmd P4`, `Cmd P18`);
  - reconnect evidence extracted from WS_AUDIT and signature inference: `tools/ws_soak_report.py:420-434`, `tools/ws_soak_report.py:444-446` (`Cmd P6`).
- Market publisher pressure counters must be zero:
  - gated counters: `mp_try_send_full_count`, `mp_pending_latest_replaced_count`: `tools/ws_soak_report.py:42-45`, `tools/ws_soak_report.py:583-586` (`Cmd P4`, `Cmd P18`);
  - parsed from `component=market_publisher` log lines: `tools/ws_soak_report.py:436-442` (`Cmd P6`).
- `cap_hits == 0`:
  - parser reads `cap_hits` from `market_rx_stats.log`: `tools/ws_soak_report.py:457-470` (`Cmd P7`);
  - gate requires both `total_cap_hits_est == 0` and `max_burst == 0`: `tools/ws_soak_report.py:591-596` (`Cmd P18`).
- Coverage intact:
  - apply coverage required for every expected connector: `tools/ws_soak_report.py:528-531` (`Cmd P18`);
  - event coverage required (default gate mode): `tools/ws_soak_report.py:567-570` (`Cmd P18`);
  - expected connector list comes from `--expected-connectors`: `tools/ws_soak_report.py:773-775`, `tools/ws_soak_report.py:823-835` (`Cmd P19`).

## Follow-up PR Needed (Not Implemented Here)
- Current `ws_shadow_soak` workflow inputs only include:
  - `duration_minutes`, `lighter_ping_interval_ms`, `extended_ws_read_timeout_ms`, `connectors`.
  - Evidence: `.github/workflows/ws_shadow_soak.yml:7-27` (`Cmd P2`).
- Current run-step env wiring does not pass any Paradex public-feed selector or Extended WS depth-level env.
  - Evidence: `.github/workflows/ws_shadow_soak.yml:59-63`, `.github/workflows/ws_shadow_soak.yml:66-76` (`Cmd P2`).
- Therefore, to execute flag-on experiments in CI, a follow-up PR is needed to add workflow_dispatch inputs and env passthrough for:
1. `PARAPHINA_PARADEX_PUBLIC_FEED`
2. `PARAPHINA_EXTENDED_WS_DEPTH_LEVELS`

