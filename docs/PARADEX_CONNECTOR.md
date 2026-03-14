# Paradex Connector

Status: **market + account + execution + cancel-all** (paper/testnet/live gated).

## Public API Summary

Paradex exposes a public WebSocket API for market data:

- WS base: `wss://ws.api.prod.paradex.trade/v1` (mainnet)
- Public channel: `orderbook` (snapshot + delta updates)
- Expected fields: `market`, `type` (`snapshot`/`delta`), `seq`, `prev_seq`, `bids`, `asks`, `ts`

## Offline Fixture Mode

- Enable feature: `live_paradex`
- Connector name: `paradex`
- Fixture directory:
  - `PARADEX_FIXTURE_DIR=/path/to/paradex`
  - or `ROADMAP_B_FIXTURE_DIR=/path/to/roadmap_b` (uses `/paradex`)
- Force fixture mode:
  - `--paradex-fixture` or `PARADEX_FIXTURE_MODE=1`

See `docs/PARADEX_FIXTURES.md` for the fixture schema.

## Live Market Data (Public WS)

- Enable feature: `live_paradex`
- Connector name: `paradex`
- Defaults:
  - `PARADEX_WS_URL=wss://ws.api.prod.paradex.trade/v1`
  - `PARADEX_MARKET=BTC-USD-PERP`
- REST/auth defaults (see official docs: https://docs.paradex.trade):
  - `PARADEX_REST_URL=https://api.prod.paradex.trade/v1`
  - `PARADEX_AUTH_URL=https://api.prod.paradex.trade/v1/auth/token`
  - `PARADEX_ACCOUNT_PATH=/account`
  - `PARADEX_ORDER_PATH=/orders`
  - `PARADEX_JWT_CMD="/opt/paraphina/.venv_paradex/bin/python3 /opt/paraphina/tools/paradex_jwt.py"` (preferred for long-lived live sessions; command prints a fresh JWT)
  - `PARADEX_JWT=...` (static fallback)
  - `PARADEX_AUTH_PAYLOAD_JSON=...` (legacy fallback, Starknet auth payload)
- Recording (manual only, no CI):
  - `--record-fixtures` or `PARADEX_RECORD_FIXTURES=1`
  - Optional `PARADEX_RECORD_DIR=/path/to/tests/fixtures/roadmap_b/paradex_live_recording`
  - Records `ws_frames.jsonl`

## Execution + Account (REST)

- Paradex uses bearer token auth (JWT) with optional Starknet-signed payloads
  (see https://docs.paradex.trade).
- For long-lived live sessions, prefer `PARADEX_JWT_CMD` over a static
  `PARADEX_JWT`. The connector refreshes command-based tokens automatically.
- Install the official SDK into the dedicated helper venv:
  - `python3 -m venv /opt/paraphina/.venv_paradex`
  - `/opt/paraphina/.venv_paradex/bin/pip install paradex_py`
- Execution endpoints (from docs):
  - `POST /orders`
  - `DELETE /orders/{order_id}`
  - `POST /orders/cancel_all`
- Account snapshot:
  - `GET /account`

### Mode gating

- Shadow: `ShadowAckAdapter` only (no REST execution).
- Paper: paper adapter only (REST execution disabled; account snapshots require auth if enabled).
- Testnet/Live: REST execution requires `PARAPHINA_LIVE_EXEC_ENABLE=1` + preflight + `--enable-live-execution`.
