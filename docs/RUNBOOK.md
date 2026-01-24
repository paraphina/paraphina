# Live Trading Runbook

This runbook covers operational controls for live trading. It is designed for
offline CI safety and manual use in production-like environments.

See `docs/DEPLOYMENT_CHAIN.md` for the end-to-end release → VPS → canary chain.

## Trade Modes

- **shadow**: run live ingestion + strategy, but do not place orders (no order placement).
- **paper**: run full pipeline, emit intent logs, no exchange orders.
- **testnet**: run full pipeline against testnet venues.
- **live**: place real orders (requires keys).

Set `PARAPHINA_TRADE_MODE=shadow|paper|testnet|live` or pass `--trade-mode`.
Default is `shadow`.

## Roadmap-B Operator Progression

Progression (all venues unless noted):

1) Shadow (all venues) with fixtures and/or public feeds.
2) Paper (all venues) with fixture-driven cancel-all and telemetry contract checks.
3) Canary live (single venue) once preflight gates are green and readiness report shows no feature gaps.
4) Scale live (add venues one at a time) after sustained canary stability.

Move to the new VPS after Shadow + Paper runs are green and before Canary live.
See "Migration to new VPS" for the concrete checklist.

## All-5 Connected Definition

- **All-5 Connected (Market)**: all five venues emit non-stale market data and
  telemetry shows `healthy_venues_used_count=5`.
- **All-5 Connected (Execution)**: live execution is enabled for all venues that
  support it (currently Hyperliquid + Lighter), with market connectivity for
  the remaining venues.

## Venue Health States

- **Healthy**: market data is fresh and the venue passes health gates, so it is
  eligible for price discovery and quoting.
- **Disabled**: market data is stale or connector errors exceed thresholds, so the
  venue is excluded from pricing/quoting and will be cancel-all'd if needed, but
  the process continues running for other venues.

## All-5 Connected Smoke (Manual-Only)

Manual-only smoke flow using fixtures (no network, safe for local or VPS use).
Do not run in CI:

```
tools/all5_smoke.sh /tmp/paraphina_all5_smoke
```

Live public-feed smoke (manual-only, requires network):

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex \
PARAPHINA_LIVE_OUT_DIR=./live_runs/all5_shadow_live \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run -p paraphina --bin paraphina_live --features live,live_hyperliquid,live_lighter,live_extended,live_aster,live_paradex
```

## All-5 Live Shadow Market Data (Live Feeds)

Left pane (market data + telemetry):

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex \
PARAPHINA_LIVE_OUT_DIR=./live_runs/all5_shadow_live \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run -p paraphina --bin paraphina_live --features live,live_hyperliquid,live_lighter,live_extended,live_aster,live_paradex
```

Right pane (terminal watch, 1000ms refresh):

```
python3 tools/paraphina_watch.py --telemetry ./live_runs/all5_shadow_live/telemetry.jsonl --refresh-ms 1000
```

## All-5 PaperExec (Offline)

Offline deterministic PaperExec run (fixtures only, safe for CI/local):

```
PARAPHINA_TRADE_MODE=paper \
PARAPHINA_LIVE_CONNECTORS=hyperliquid_fixture,lighter,extended,aster,paradex \
EXTENDED_FIXTURE_MODE=1 ASTER_FIXTURE_MODE=1 PARADEX_FIXTURE_MODE=1 \
HL_FIXTURE_DIR=./tests/fixtures/hyperliquid \
LIGHTER_FIXTURE_DIR=./tests/fixtures/lighter \
ROADMAP_B_FIXTURE_DIR=./tests/fixtures/roadmap_b \
PARAPHINA_PAPER_FILL_MODE=mid \
PARAPHINA_PAPER_SMOKE_INTENTS=1 \
PARAPHINA_LIVE_OUT_DIR=./live_runs/all5_paper \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run -p paraphina --bin paraphina_live --features live,live_hyperliquid,live_lighter,live_extended,live_aster,live_paradex
```

Validate telemetry contract:

```
python3 tools/check_telemetry_contract.py ./live_runs/all5_paper/telemetry.jsonl
```

## All-5 CanaryLive (Online)

Canary live requires credentials and explicit live execution gates. Run manually only:

```
PARAPHINA_TRADE_MODE=live \
PARAPHINA_LIVE_CONNECTORS=hyperliquid,lighter,extended,aster,paradex \
PARAPHINA_LIVE_EXEC_ENABLE=1 PARAPHINA_LIVE_EXECUTION_CONFIRM=YES \
PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS=5000 \
--enable-live-execution \
cargo run -p paraphina --bin paraphina_live --features live,live_hyperliquid,live_lighter,live_extended,live_aster,live_paradex
```

## Live Execution Guardrails

Shadow mode is always safe: it never submits live orders.

Live mode requires all of the following to start:

- `--trade-mode live`
- `--enable-live-execution`
- `PARAPHINA_LIVE_EXEC_ENABLE=1` (or `true`)
- `PARAPHINA_LIVE_EXECUTION_CONFIRM=YES`
- Required venue credentials are present
 - `PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS=<ms>` must be a positive integer (set to `0`, `-1`, or `false` to disable)

If any requirement is missing, the binary refuses to start and suggests running
in shadow mode instead.

Paper/testnet execution remains gated behind `PARAPHINA_LIVE_EXEC_ENABLE=1`
and required venue credentials; otherwise it falls back to shadow execution.

## Paper Execution Model

Paper mode uses a deterministic internal execution adapter by default and never
calls real execution endpoints unless explicitly enabled.

Environment toggles:

- `PARAPHINA_PAPER_FILL_MODE=none|marketable|mid|always` (default `none`)
- `PARAPHINA_PAPER_SLIPPAGE_BPS=<float>` (default `0.0`)
- `PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF=<u32>` to relax FV gating in paper mode
- `PARAPHINA_PAPER_DISABLE_FV_GATE=1` to bypass FV gating for quote generation
- `PARAPHINA_PAPER_USE_WALLCLOCK_TS=1` to treat market timestamps as fresh wallclock
- `PARAPHINA_PAPER_DISABLE_HEALTH_GATES=1` to bypass venue health disables in paper mode
- `PARAPHINA_PAPER_SMOKE_INTENTS=1` to force a deterministic paper intent per tick
- `PARAPHINA_PAPER_ROUTE_SANDBOX=1` to route paper execution to sandbox/testnet
  via the live gateway (still treated as Paper, but requires credentials).

## Preflight Checklist

Run preflight to validate connector config, out-dir, telemetry path, key presence
(without printing secrets), and required feature flags:

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTOR=hyperliquid \
PARAPHINA_LIVE_OUT_DIR=./live_runs/preflight_001 \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run --bin paraphina_live --features live,live_hyperliquid -- --preflight
```

## Shadow Session

Shadow run (real public feeds, no keys):

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTOR=hyperliquid \
PARAPHINA_LIVE_OUT_DIR=./live_runs/shadow_001 \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run --bin paraphina_live --features live,live_hyperliquid
```

Offline fixture run (no network):

```
PARAPHINA_TRADE_MODE=shadow \
PARAPHINA_LIVE_CONNECTOR=hyperliquid_fixture \
HL_FIXTURE_DIR=./tests/fixtures/hyperliquid \
PARAPHINA_LIVE_OUT_DIR=./live_runs/fixture_001 \
PARAPHINA_TELEMETRY_MODE=jsonl \
cargo run --bin paraphina_live --features live,live_hyperliquid
```

Validate telemetry contract:

```
python3 tools/check_telemetry_contract.py ./live_runs/fixture_001/telemetry.jsonl
```

## Health + Metrics

- `GET /health` → 200 if process is healthy.
- `GET /ready` → 200 after first valid market snapshot for all venues.
- `GET /metrics` → Prometheus text format.
- `GET /config` → JSON bundle of `config_resolved.json` + `build_info.json`.

Default bind: `127.0.0.1:9898` (override with `PARAPHINA_LIVE_METRICS_ADDR`).

## Startup Artifacts

At startup, Paraphina writes:

- `config_resolved.json`
- `build_info.json`

Directory is `./live_audit` by default (override with `PARAPHINA_LIVE_AUDIT_DIR`).
If `PARAPHINA_LIVE_OUT_DIR` is set, all artifacts (including `telemetry.jsonl`
and `summary.json`) are written there.

## Kill Procedures

1) **Primary kill switch**: set `PARAPHINA_LIVE_KILL_FLATTEN=false` (default).
2) Trigger kill switch (risk engine or manual).
3) Live runner sends cancel‑all per venue and halts.

Optional:
`PARAPHINA_LIVE_KILL_FLATTEN=true` triggers best‑effort reduce‑only IOC flatten
after cancel‑all.

## Reconciliation

- On WS gaps, request REST snapshot and reconcile open orders.
- Periodic reconciliation cadence via `PARAPHINA_LIVE_RECONCILE_MS`.

## Reconciliation Required (Live)

Live mode requires reconciliation to prevent silent drift between internal state
and venue state. Configure:

- `PARAPHINA_LIVE_RECONCILE_MS=<ms>` to schedule account reconciliation requests.
- `PARAPHINA_RECONCILE_POS_TAO_TOL` and `PARAPHINA_RECONCILE_BALANCE_USD_TOL` for tolerances.
- `PARAPHINA_RECONCILE_ORDER_COUNT_TOL` for open-order drift tolerance.

## Canary Live Checklist

- [ ] Run preflight and export `PARAPHINA_LIVE_PREFLIGHT_OK=1`.
- [ ] Select canary profile (`--canary-profile prod_canary` or `PARAPHINA_LIVE_CANARY_PROFILE`).
- [ ] Enable reconciliation (`PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS`).
- [ ] Verify rate limit vars set (from canary profile).
- [ ] Confirm telemetry path + out-dir are writable.
- [ ] Start in `shadow`, then switch to `live` with enable flags.

## Rollback / Kill Drill

1. Set `PARAPHINA_TRADE_MODE=shadow` and restart the service.
2. Verify cancel‑all acknowledgements and drift events in `reconcile_drift.jsonl`.
3. Clear outstanding orders on venue UI/API.
4. Only re‑enable live after drift clears and `/ready` is healthy.

## Scaling Plan

1. Canary live with `prod_canary.toml` for a fixed window.
2. Review burn‑in report + reconcile drift logs.
3. Gradually increase limits (order size, max position, open orders).
4. Expand venues one at a time with the same canary process.

## Incident Checklist

- Confirm health endpoints responding.
- Validate `build_info.json` and `config_resolved.json` match deployment.
- Check cancel‑all acknowledgements after kill switch.
- Validate open‑order snapshot matches internal order state.
- For Hyperliquid: confirm correct network (`HL_NETWORK`) and paper mode.

## Incident Playbooks

### Order Reject Storm

Symptoms:
- Rapid rise in `paraphina_live_reject_by_reason`.
- `paraphina_live_order_submit_fail` spikes.

Actions:
1. Inspect reject reasons (post‑only vs reduce‑only vs fatal).
2. Pause live mode (`PARAPHINA_TRADE_MODE=paper`) to stop submissions.
3. Reconcile open orders and compare with exchange snapshot.
4. If post‑only rejects dominate, widen quotes or check book staleness.
5. If reduce‑only violations dominate, check position drift vs account snapshots.

### Feed Gap

Symptoms:
- Missing market updates, stale venue snapshots, or seq gaps.

Actions:
1. Trigger REST snapshot refresh.
2. Validate staleness clears and `ready_market_count > 0`.
3. If repeated gaps, restart connector with backoff.

### Reconcile Mismatch

Symptoms:
- `paraphina_live_reconcile_mismatch_count` increments.
- New diffs in `live_audit/account_reconcile.jsonl`.

Actions:
1. Inspect latest reconcile diff record.
2. Confirm account snapshot seq monotonicity.
3. If mismatch persists, halt trading and re‑sync account snapshot.
4. Verify funding/margin/liquidation fields are populated.

### Stuck Cancel‑All

Symptoms:
- Cancel‑all requests acknowledged late or not at all.

Actions:
1. Stop new submissions (set `PARAPHINA_TRADE_MODE=paper`).
2. Re‑issue cancel‑all per venue.
3. Fetch open‑order snapshot and reconcile.
4. If still stuck, trigger manual exchange cancel in UI/API and restart runner.

## Dual Venue Paper Run

Example (Hyperliquid + Lighter, paper mode):

1. Enable live features at build time:
   - `cargo build -p paraphina --features live,live_hyperliquid,live_lighter`
2. Configure env:
   - `PARAPHINA_TRADE_MODE=paper`
   - `HL_NETWORK=testnet`
   - `HL_PAPER_MODE=true`
   - `LIGHTER_PAPER_MODE=true`
   - `PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS=5000`
3. Start:
   - `paraphina_live`

Verify:
- `/ready` transitions to 200 once both venues publish snapshots.
- `/metrics` shows per‑venue order submit/cancel counters.

## Watchdog Recommendation

Use an external watchdog to:

- restart on process exit,
- alert on `/health` failures,
- capture stdout/stderr for audit.

In‑process guardrails:

- cancel‑all on kill,
- reduce‑only flatten (optional),
- reject post‑only orders that would cross (mock/exchange enforced).

## Migration to new VPS

Checklist:

- [ ] Run `deploy/bootstrap.sh` (packages, users, limits).
- [ ] Copy binary + configs to `/opt/paraphina`.
- [ ] Install `/etc/paraphina/paraphina_live.env` from `deploy/env.example` (fill secrets via secure channel).
- [ ] Install systemd unit `deploy/systemd/paraphina_live.service`.
- [ ] Install logrotate config `deploy/logrotate/paraphina_live`.
- [ ] Enable + start systemd service.

## Smoke test on new VPS

Run Step‑B baseline, one shadow session, and a telemetry contract check:

```
python3 -m pytest -q
PARAPHINA_LIVE_OUT_DIR=/tmp/paraphina_vps_smoke \
PARAPHINA_TELEMETRY_MODE=jsonl \
tools/vps_smoke_check.sh /tmp/paraphina_vps_smoke
```

## WP100 Completion Gate (CI)

This gate prevents regressions in whitepaper completion before paper trading.
It fails if any requirement becomes Partial/Missing or if required validators
fail.

### How to run WP100 locally

Run the Step‑B baseline and audit regeneration:

```
cargo test --all
python3 -m pytest -q
python3 tools/check_docs_truth_drift.py
python3 tools/check_docs_integrity.py
python3 tools/generate_wp100_audit.py
python3 tools/check_wp100_completion.py docs/WHITEPAPER_COMPLETION_AUDIT.json
```

Full matrix (nightly/scheduled):

```
cargo test --features event_log
cargo test -p paraphina --features live,live_hyperliquid
cargo test -p paraphina --features live,live_lighter
PARAPHINA_TELEMETRY_MODE=jsonl PARAPHINA_TELEMETRY_PATH=/tmp/telemetry.jsonl \
cargo run --bin paraphina -- --ticks 300 --seed 42
python3 tools/check_telemetry_contract.py /tmp/telemetry.jsonl
```

### What the CI gate means

- PRs must keep `docs/WHITEPAPER_COMPLETION_AUDIT.json` at 100% completion.
- Required doc validators and tests must pass.
- Nightly runs the live feature matrix + telemetry contract checks.

### Updating the canonical hash

Never edit `docs/CANONICAL_SPEC_V1_SHA256.txt` by hand. Use:

```
python3 tools/check_docs_integrity.py
```

That tool is the only approved way to update the canonical hash after
intentional changes to `docs/WHITEPAPER.md`.
