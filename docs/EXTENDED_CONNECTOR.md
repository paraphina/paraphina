# Extended Connector

Status: **market + account + execution + cancel-all** (paper/testnet/live gated, SDK bridge for account/execution).

## Public API Summary

Extended exposes a Starknet orderbook/feed and REST API:

- WS base: `wss://api.starknet.extended.exchange/stream.extended.exchange/v1`
- WS channel: `/orderbooks/<market>` or `/orderbooks/<market>?depth=1`
- REST snapshot: `https://api.starknet.extended.exchange/api/v1/orderbooks/<market>`

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
  - `EXTENDED_WS_URL=wss://api.starknet.extended.exchange/stream.extended.exchange/v1`
  - `EXTENDED_REST_URL=https://api.starknet.extended.exchange`
  - `EXTENDED_MARKET=BTC-USD`
- Auth / execution bridge:
  - `EXTENDED_API_KEY=...`
- `EXTENDED_TRADER_CMD="/opt/paraphina/.venv_extended/bin/python3 /opt/paraphina/tools/extended_trade.py"`
  - `EXTENDED_STARK_PRIVATE_KEY=...`
  - `EXTENDED_STARK_PUBLIC_KEY=...`
  - `EXTENDED_L2_VAULT=...`
  - `EXTENDED_ACCOUNT_ID=...` (optional metadata from Extended tooling)
  - `EXTENDED_SDK_ENV=mainnet` (optional; use `testnet` for sepolia)
- Overrides:
  - `EXTENDED_DEPTH_LIMIT` (REST snapshot depth, default 100)
- Recording (manual only, no CI):
  - `--record-fixtures` or `EXTENDED_RECORD_FIXTURES=1`
  - Optional `EXTENDED_RECORD_DIR=/path/to/tests/fixtures/roadmap_b/extended_live_recording`
  - Records `rest_snapshot.json` + `ws_frames.jsonl`

## Execution + Account (SDK bridge)

- Official Extended auth is Starknet-based. For day-1 live, Paraphina uses the
  official Python SDK through [extended_trade.py](/home/ubuntu/paraphina/tools/extended_trade.py)
  instead of handwritten Rust signing.
- The helper normalizes these SDK calls back into Paraphina’s account/execution
  contract:
  - `account.get_balance()`
  - `account.get_positions(market_names=[...])`
  - `place_order(... post_only, reduce_only, external_id ...)`
  - `orders.cancel_order(order_id=...)`
  - `orders.mass_cancel(order_ids=[...])`
- Install the official SDK into the dedicated helper venv:
  - `python3 -m venv /opt/paraphina/.venv_extended`
  - `/opt/paraphina/.venv_extended/bin/pip install x10-python-trading-starknet`

### Mode gating

- Shadow: `ShadowAckAdapter` only (no REST execution).
- Paper: paper adapter only (REST execution disabled; account snapshots require the bridge env if enabled).
- Testnet/Live: REST execution requires `PARAPHINA_LIVE_EXEC_ENABLE=1` + preflight + `--enable-live-execution`.
