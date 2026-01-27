# Extended Connector

Status: **market + account + execution + cancel-all** (paper/testnet/live gated).

## Public API Summary

Extended exposes a Binance-compatible futures market-data API:

- WS base: `wss://stream.extended.exchange/ws`
- WS channel: `<symbol>@depth@100ms` (orderbook depth deltas)
- WS payload fields: `e=depthUpdate`, `E` (event time), `s` (symbol), `U/u/pu` (sequence ids), `b/a` (bid/ask deltas)
- REST snapshot: `https://api.extended.exchange/fapi/v1/depth?symbol=...&limit=...`

## Offline Fixture Mode

- Enable feature: `live_extended`
- Connector name: `extended`
- Fixture directory:
  - `EXTENDED_FIXTURE_DIR=/path/to/extended`
  - or `ROADMAP_B_FIXTURE_DIR=/path/to/roadmap_b` (uses `/extended`)
- Force fixture mode:
  - `--extended-fixture` or `EXTENDED_FIXTURE_MODE=1`

See `docs/EXTENDED_FIXTURES.md` for the fixture schema.

## Live Market Data (Public WS)

- Enable feature: `live_extended`
- Connector name: `extended`
- Defaults:
  - `EXTENDED_WS_URL=wss://stream.extended.exchange/ws`
  - `EXTENDED_REST_URL=https://api.extended.exchange`
  - `EXTENDED_MARKET=BTCUSDT`
- Auth (signed REST, per official docs: https://docs.extended.exchange):
  - `EXTENDED_API_KEY=...`
  - `EXTENDED_API_SECRET=...`
  - `EXTENDED_RECV_WINDOW=5000` (optional)
- Overrides:
  - `EXTENDED_DEPTH_LIMIT` (REST snapshot depth, default 100)
- Recording (manual only, no CI):
  - `--record-fixtures` or `EXTENDED_RECORD_FIXTURES=1`
  - Optional `EXTENDED_RECORD_DIR=/path/to/tests/fixtures/roadmap_b/extended_live_recording`
  - Records `rest_snapshot.json` + `ws_frames.jsonl`

## Execution + Account (REST)

- Signed endpoints use HMAC SHA256 over the canonical query string, with `X-MBX-APIKEY`
  + `timestamp` + `signature` parameters (see https://docs.extended.exchange).
- Execution endpoints:
  - `POST /fapi/v1/order` (limit orders, post-only uses `timeInForce=GTX`)
  - `DELETE /fapi/v1/order`
  - `DELETE /fapi/v1/allOpenOrders`
- Account snapshot:
  - `GET /fapi/v2/account`

### Mode gating

- Shadow: `ShadowAckAdapter` only (no REST execution).
- Paper: paper adapter only (REST execution disabled; account snapshots require API keys if enabled).
- Testnet/Live: REST execution requires `PARAPHINA_LIVE_EXEC_ENABLE=1` + preflight + `--enable-live-execution`.
