# WS Frontier PR C Evidence — Lighter Layer‑B REST Fallback

## Metadata
- Branch: `connectivity/lighter-layerb-rest-fallback`
- Base commit on branch start: `29c19fd`
- Scope: wire Lighter into Layer‑B REST monitor using in-repo proven REST path; add audit-gated rest-monitor counters.
- Result: **Wired (not blocked)**.

## Decision Gate: Endpoint Proof
Requirement: no invented endpoint.

Proof from existing repo artifacts:
- Existing Lighter REST path usage already in connector code:
  - `paraphina/src/live/connectors/lighter.rs:1591-1649`
  - includes explicit endpoints `"/api/v1/orderBooks"` and `"/api/v1/orderbooks"` at `paraphina/src/live/connectors/lighter.rs:1614`.
- Lighter config values already captured in live entrypoint for potential monitor wiring:
  - `paraphina/src/bin/paraphina_live.rs:2161-2162` (`ltr_rest_url`, `ltr_market`).

Conclusion: Layer‑B wiring is permitted by constraints via in-repo proven Lighter URL/path.

## Existing Layer‑B Wiring Pattern (baseline)
`rest_entries.push(...)` already exists for four venues in `paraphina_live.rs`:
- Hyperliquid: `paraphina/src/bin/paraphina_live.rs:1994-2015`
- Extended: `paraphina/src/bin/paraphina_live.rs:2315-2329`
- Aster: `paraphina/src/bin/paraphina_live.rs:2480-2494`
- Paradex: `paraphina/src/bin/paraphina_live.rs:2646-2660`

Associated REST snapshot builders and timestamp semantics:
- Extended uses `fetch_extended_l2_snapshot` with `timestamp_ms = wall_ms()`:
  - `paraphina/src/live/rest_health_monitor.rs:234-262`
- Aster uses `fetch_aster_l2_snapshot` with `timestamp_ms = wall_ms()`:
  - `paraphina/src/live/rest_health_monitor.rs:317-345`
- Paradex uses `fetch_paradex_l2_snapshot` with exchange timestamp fallback to wall clock:
  - `paraphina/src/live/rest_health_monitor.rs:347-376`.

## Changes Implemented

### 1) Lighter Layer‑B rest entry wiring
- New Lighter `rest_entries.push(...)` added in live startup flow:
  - `paraphina/src/bin/paraphina_live.rs:2202-2225`
- Uses existing captured values:
  - `paraphina/src/bin/paraphina_live.rs:2161-2162`
- Calls new monitor fetch helper:
  - `paraphina/src/bin/paraphina_live.rs:2219-2221`

### 2) Lighter REST snapshot fetch helper (Layer‑B)
- New helper added:
  - `fetch_lighter_l2_snapshot` at `paraphina/src/live/rest_health_monitor.rs:264-315`
- Endpoint usage (proven path only):
  - `paraphina/src/live/rest_health_monitor.rs:274`
- Snapshot parsing helpers:
  - `parse_lighter_snapshot_response`: `paraphina/src/live/rest_health_monitor.rs:393-454`
  - `parse_lighter_levels`: `paraphina/src/live/rest_health_monitor.rs:470-499`
  - `parse_lighter_str_or_number`: `paraphina/src/live/rest_health_monitor.rs:501-515`

### 3) Rest monitor audit-gated counters/logging
- Counter state struct:
  - `paraphina/src/live/rest_health_monitor.rs:73-81`
- Audit flag + compact logger:
  - `paraphina/src/live/rest_health_monitor.rs:83-116`
- Loop integration and per-venue counts:
  - check count increment: `paraphina/src/live/rest_health_monitor.rs:145-147`
  - startup unknown-age guard (preexisting, retained): `paraphina/src/live/rest_health_monitor.rs:148-153`
  - attempt/success/fail/inject increments: `paraphina/src/live/rest_health_monitor.rs:185-222`
- Log format:
  - `WS_AUDIT subsystem=rest_monitor venue=... rest_check_count=... rest_attempt_count=... rest_success_count=... rest_fail_count=... rest_inject_count=...`

## Validation

### Commands + outcomes
- `python3 -m py_compile tools/paraphina_watch.py || true` → pass
- `python3 -m py_compile tools/telemetry_analyzer.py || true` → pass
- `cargo check -q` → pass
- `cargo test -q` → **fails on host environment** with `Invalid cross-device link (os error 18)` (EXDEV)
  - tail captured from `/tmp/ws_prC_cargo_test.log`
- Optional 3-minute shadow smoke attempted:
  - OUT_DIR: `/tmp/ws_prC_smoke_20260211T010645Z`
  - blocked before runtime by same EXDEV compilation error

### EXDEV proof
- errno mapping command output:
  - `errno 18: EXDEV Invalid cross-device link`

## Notes on startup-thrash safety
- Rest monitor unknown/uninitialized age guard remains in place and unchanged:
  - `paraphina/src/live/rest_health_monitor.rs:98-103` (comment)
  - implemented branch in loop: `paraphina/src/live/rest_health_monitor.rs:148-153`
- Therefore adding Lighter rest entry does not introduce immediate startup injection at `i64::MAX` age.

## Files touched
- `paraphina/src/bin/paraphina_live.rs`
- `paraphina/src/live/rest_health_monitor.rs`
- `docs/INVESTIGATIONS/ws_frontier_backlog.md`
- `docs/INVESTIGATIONS/ws_frontier_prC_evidence.md`

## Command Index (this PR)
- `Cmd C01`: `git switch main && git pull --ff-only origin main && git status -sb && git rev-parse --short HEAD`
- `Cmd C02`: `pgrep -fa '[p]araphina_live.*--trade-mode shadow' || echo '(none)'`
- `Cmd C03`: `git switch -c connectivity/lighter-layerb-rest-fallback && git status -sb`
- `Cmd C04`: locate Layer‑B wiring in `paraphina_live.rs` via ripgrep
- `Cmd C05`: locate Lighter REST/orderBooks evidence via ripgrep
- `Cmd C06`: `nl -ba` inspection around Lighter connector and existing rest_entries push sites
- `Cmd C07`: inspect rest monitor fetch helpers and startup guard in `rest_health_monitor.rs`
- `Cmd C08`: `cargo check -q` (post-edit)
- `Cmd C09`: `python3 -m py_compile tools/paraphina_watch.py || true`
- `Cmd C10`: `python3 -m py_compile tools/telemetry_analyzer.py || true`
- `Cmd C11`: `cargo test -q 2>&1 | tee /tmp/ws_prC_cargo_test.log || true; tail -n 120 /tmp/ws_prC_cargo_test.log`
- `Cmd C12`: optional shadow smoke attempt (3m) with `PARAPHINA_WS_AUDIT=1`, logs in `/tmp/ws_prC_smoke_20260211T010645Z`
- `Cmd C13`: `python3` errno mapping for `EXDEV`
