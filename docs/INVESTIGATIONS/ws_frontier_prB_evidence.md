# WS Frontier PR B Evidence (Connectivity Uniformity + Observability)

## Metadata
- Branch: `connectivity/ws-frontier-uniformity-observability`
- Base HEAD during implementation: `b2e29e2`
- Mode: SHADOW-only, no secrets, no live trading enablement.

## Scope
This PR applies minimal, local diffs for Frontier Step 2:
1. Lighter outbound periodic ping (asymmetry fix).
2. Extended read-timeout env wiring (hardcoded 10s -> configurable, default preserved).
3. Uniform reconnect-reason audit counters across all 5 venues.
4. MarketPublisher pressure counters (audit-gated).

All new observability is gated by `PARAPHINA_WS_AUDIT=1` and default-off.

## Findings -> Hypotheses -> Implementation

### H1: Lighter had no outbound periodic ping, causing keepalive asymmetry
- Hypothesis: adding outbound WS Ping at fixed interval will reduce idle disconnect/reconnect flaps relative to venues already pinging.
- Implementation:
  - New env + default: `PARAPHINA_LIGHTER_PING_INTERVAL_MS` (default `30000`, `0` disables) in `paraphina/src/live/connectors/lighter.rs:15-62`.
  - Ping branch in public loop with audit counters:
    - `lighter_ping_sent_count`, `lighter_ping_send_fail_count`
    - `WS_AUDIT venue=lighter ...`
    - `paraphina/src/live/connectors/lighter.rs:706-754`
  - Reconnect-reason helper and callsites (`subscribe_error`, `session_timeout`, `stale_watchdog`, `read_timeout`, `ping_send_fail`, `decode_fail_loop`):
    - helper: `paraphina/src/live/connectors/lighter.rs:256-277`
    - callsites: `paraphina/src/live/connectors/lighter.rs:576-603`, `paraphina/src/live/connectors/lighter.rs:715-717`, `paraphina/src/live/connectors/lighter.rs:733-734`, `paraphina/src/live/connectors/lighter.rs:750-751`, `paraphina/src/live/connectors/lighter.rs:821-823`, `paraphina/src/live/connectors/lighter.rs:965-966`.

Evidence commands: `Cmd U08`, `Cmd U13`.

### H2: Extended 10s read-timeout is too rigid and may flap under transient stalls
- Hypothesis: keeping default at 10s but making it env-configurable enables evidence-driven tuning without code churn.
- Implementation:
  - New env + default preserved: `PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS` default `10000`:
    - `paraphina/src/live/connectors/extended.rs:14`, `paraphina/src/live/connectors/extended.rs:35-42`
  - Runtime wiring + one-time audit emission of effective timeout:
    - `paraphina/src/live/connectors/extended.rs:443-460`, `paraphina/src/live/connectors/extended.rs:522-533`
  - Reconnect-reason helper + callsites (`session_timeout`, `stale_watchdog`, `ping_send_fail`, `read_timeout`, `parse_error`, `seq_gap`, `seq_mismatch`):
    - helper: `paraphina/src/live/connectors/extended.rs:44-65`
    - callsites: `paraphina/src/live/connectors/extended.rs:317-319`, `paraphina/src/live/connectors/extended.rs:510-517`, `paraphina/src/live/connectors/extended.rs:528-529`, `paraphina/src/live/connectors/extended.rs:593-595`, `paraphina/src/live/connectors/extended.rs:685-693`, `paraphina/src/live/connectors/extended.rs:764-766`, `paraphina/src/live/connectors/extended.rs:856-863`.

Evidence commands: `Cmd U09`, `Cmd U13`.

### H3: Reconnect reason taxonomy was non-uniform across venues
- Hypothesis: a per-connector audit counter with standard reason labels enables apples-to-apples reconnect diagnostics.
- Implementation (audit-gated helpers in each connector):
  - Hyperliquid: helper + callsites
    - `paraphina/src/live/connectors/hyperliquid.rs:27-81`, `paraphina/src/live/connectors/hyperliquid.rs:480-483`, `paraphina/src/live/connectors/hyperliquid.rs:645-647`, `paraphina/src/live/connectors/hyperliquid.rs:652-654`, `paraphina/src/live/connectors/hyperliquid.rs:663-664`
  - Lighter: see H1 citations.
  - Extended: see H2 citations.
  - Aster: helper + callsites
    - `paraphina/src/live/connectors/aster.rs:53-107`, `paraphina/src/live/connectors/aster.rs:306-308`, `paraphina/src/live/connectors/aster.rs:456-459`, `paraphina/src/live/connectors/aster.rs:559-560`, `paraphina/src/live/connectors/aster.rs:624-625`, `paraphina/src/live/connectors/aster.rs:726-727`, `paraphina/src/live/connectors/aster.rs:761-763`, `paraphina/src/live/connectors/aster.rs:767-768`, `paraphina/src/live/connectors/aster.rs:777-778`, `paraphina/src/live/connectors/aster.rs:838-839`, `paraphina/src/live/connectors/aster.rs:902-903`
  - Paradex: helper + callsites
    - `paraphina/src/live/connectors/paradex.rs:18-56`, `paraphina/src/live/connectors/paradex.rs:289-291`, `paraphina/src/live/connectors/paradex.rs:417-419`, `paraphina/src/live/connectors/paradex.rs:435-436`, `paraphina/src/live/connectors/paradex.rs:453-454`, `paraphina/src/live/connectors/paradex.rs:525-527`, `paraphina/src/live/connectors/paradex.rs:591-601`

Evidence commands: `Cmd U10`, `Cmd U13`.

### H4: MarketPublisher overflow/coalesce pressure lacked uniform counters
- Hypothesis: low-frequency counters for full queue / replacement / lossless wait will identify pressure plateaus and coalescing hotspots.
- Implementation:
  - Audit-gated counters + logger:
    - `paraphina/src/live/market_publisher.rs:11-31`
  - Counter increments:
    - `mp_lossless_wait_count`: `paraphina/src/live/market_publisher.rs:125-129`
    - `mp_try_send_full_count`: `paraphina/src/live/market_publisher.rs:139-140`
    - `mp_pending_latest_replaced_count`: `paraphina/src/live/market_publisher.rs:142-148`

Evidence commands: `Cmd U11`, `Cmd U13`.

## Runtime Evidence (Before/After Smoke)

### Before (pre-change) attempt
- OUT_DIR: `/tmp/ws_prB_before_20260211T001446Z`
- Result: build blocked before runtime.
- Failure:
  - `error: failed to write ... libparaphina-...rmeta: Invalid cross-device link (os error 18)`
- No telemetry produced; reconnect/pressure runtime counters unavailable.

Evidence commands: `Cmd U04`, `Cmd U14`.

### After (post-change) attempt
- OUT_DIR: `/tmp/ws_prB_after_20260211T001928Z`
- Mitigation attempted:
  - `TMPDIR=$PWD/target/.tmp_rustc`
  - `RUSTC_TMPDIR=$PWD/target/.tmp_rustc`
- Result: same EXDEV block before runtime.
- Failure:
  - `error: failed to write ... libparaphina-...rmeta: Invalid cross-device link (os error 18)`
- No telemetry produced; reconnect/pressure runtime counters unavailable.

Evidence commands: `Cmd U07`, `Cmd U14`.

## Validation Evidence
- `cargo check -q` passed after edits.
- Python tool syntax checks passed:
  - `python3 -m py_compile tools/paraphina_watch.py`
  - `python3 -m py_compile tools/telemetry_analyzer.py`
- `cargo test -q` blocked by EXDEV on this host (`os error 18`).

Evidence commands: `Cmd U06`, `Cmd U12`, `Cmd U15`.

## Interpretation
- Code-level asymmetries requested for PR B are implemented with minimal local changes.
- Runtime before/after p50/p95/p99 deltas are **blocked on this VPS** due host filesystem EXDEV during compilation.
- CI should be treated as source of truth for build/test execution and subsequent runtime verification.

## Shadow Runbook (for when host build issue is resolved)
Use these commands to collect the promised metrics without code changes:
1. 10-minute smoke (`PARAPHINA_WS_AUDIT=1`, `PARAPHINA_MARKET_RX_STATS=1`, JSONL telemetry path).
2. Extract age metrics:
   - `python3 tools/telemetry_analyzer.py --telemetry <telemetry.jsonl>`
3. Reconnect counts from logs:
   - `rg -n 'WS_AUDIT .*reconnect_reason=' <run.log>`
4. MarketPublisher pressure counts:
   - `rg -n 'WS_AUDIT component=market_publisher' <run.log>`
5. Runner cap hits:
   - `rg -n 'market_rx_stats .*cap_hits=' <market_rx_stats.log> <run.log>`

## Explicit Blockers
- Lighter Layer B REST fallback endpoint remains blocked (no evidence-backed endpoint added in this PR).

## Command Index (Repro)
- `Cmd U01`: `git switch main && git pull --ff-only origin main && git status -sb && git rev-parse --short HEAD`
- `Cmd U02`: `pgrep -fa '[p]araphina_live.*--trade-mode shadow' || echo '(none)'`
- `Cmd U03`: `git switch -c connectivity/ws-frontier-uniformity-observability && git status -sb`
- `Cmd U04`: baseline 10m smoke command (`/tmp/ws_prB_before_20260211T001446Z`) with WS audit + market_rx stats enabled
- `Cmd U05`: `python3 - <<'PY' ... errno 18 ... PY` (EXDEV meaning)
- `Cmd U06`: `python3 -m py_compile tools/paraphina_watch.py && python3 -m py_compile tools/telemetry_analyzer.py`
- `Cmd U07`: post-change 10m smoke command (`/tmp/ws_prB_after_20260211T001928Z`) with `TMPDIR` and `RUSTC_TMPDIR` mitigation
- `Cmd U08`: `rg -n 'ping|Ping|Pong|ping_interval|read timeout|read_timeout|timeout|stale' paraphina/src/live/connectors/lighter.rs`
- `Cmd U09`: `rg -n 'read timeout|read_timeout|timeout\(' paraphina/src/live/connectors/extended.rs`
- `Cmd U10`: `rg -n 'reconnect|backoff|session timeout|stale' paraphina/src/live/connectors/{hyperliquid,lighter,extended,aster,paradex}.rs`
- `Cmd U11`: `rg -n 'MarketPublisher|try_send|pending_latest|overflow|lossless' paraphina/src/live/market_publisher.rs`
- `Cmd U12`: `cargo check -q`
- `Cmd U13`: `nl -ba` evidence excerpts for all changed files
- `Cmd U14`: artifact tails/counts for before/after runs under `/tmp/ws_prB_before_...` and `/tmp/ws_prB_after_...`
- `Cmd U15`: `cargo test -q 2>&1 | tee /tmp/ws_prB_cargo_test.log || true; tail -n 120 /tmp/ws_prB_cargo_test.log`
