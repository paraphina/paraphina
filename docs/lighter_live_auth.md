# Lighter Live Auth

## A) Current live path
- Account snapshot: `GET /api/v1/account`
- Execution submit: `POST /api/v1/sendTx`
- Signing: local bridge service backed by the official `lighter-sdk`

The Rust connector expects `LIGHTER_SIGNER_URL` to point at the signer bridge
base URL, for example `http://127.0.0.1:9001`. The connector appends `/sign`
itself and also tolerates the legacy `.../sign` env value.

## B) Authoritative endpoints
- REST base: `https://mainnet.zklighter.elliot.ai`
- Account: `GET /api/v1/account` (by account index or L1 address)
- Transactions: `POST /api/v1/sendTx` and `POST /api/v1/sendTxBatch`
- Nonce: `GET /api/v1/nextNonce` (bootstrap only; local nonce is preferred)

Hard requirement: **must sign tx body before sending**.

## C) Required env vars
- `LIGHTER_API_KEY_INDEX` (u64)
- `LIGHTER_ACCOUNT_INDEX` (u64)
- `LIGHTER_API_PRIVATE_KEY_HEX` (hex string)
- `LIGHTER_AUTH_TOKEN` (optional; only if required by some endpoints)
- `LIGHTER_NONCE_PATH` (optional persistence)
- `LIGHTER_SIGNER_URL` (signer bridge base URL, e.g. `http://127.0.0.1:9001`)
- `LIGHTER_MARKET` or `LIGHTER_MARKET_ID` (used by the signer bridge to resolve market id)
- `LIGHTER_NETWORK` (`mainnet` or `testnet`)

The shipped bridge entrypoint is:

```bash
/opt/paraphina/.venv_lighter/bin/python3 /opt/paraphina/tools/lighter_signer_service.py --host 127.0.0.1 --port 9001
```

There is also a systemd template at
`deploy/systemd/lighter_signer.service.template`.

## D) Nonce strategy
- Nonce must be strictly increasing.
- Primary: local monotonic nonce manager (AtomicU64), `next = max(now_ms, last + 1)`.
- Optional persistence: `LIGHTER_NONCE_PATH` to survive restarts.
- Bootstrap: call `GET /api/v1/nextNonce` only if local nonce is uninitialized.

## E) Execution mapping from Paraphina types
- Map `LiveRestPlaceRequest` / `LiveRestCancelRequest` / `LiveRestCancelAllRequest`
  to Lighter transaction intents submitted via `sendTx` / `sendTxBatch`.
- Acceptance criteria for `order_id`:
  - Parse if present in response.
  - If absent, define a safe fallback strategy (e.g., return `None` and rely on later
    account snapshot reconciliation or an explicit mapping from client_order_id).

## F) Execution mapping
- `create_order`: market id comes from `LIGHTER_MARKET_ID` or `LIGHTER_MARKET`
  lookup; `post_only` is preserved and mapped to Lighter's post-only TIF.
- `cancel_order`: Paraphina now includes `market_index` in signer requests so the
  bridge can call the official signer path without guessing.
- `cancel_all`: uses the bridge `cancel_all` request and Rust still submits the
  signed payload to `sendTx`.

## G) Test plan (mock-only)
- Mock endpoints:
  - `GET /api/v1/account`
  - `POST /api/v1/sendTx`
  - `POST /api/v1/sendTxBatch`
  - `GET /api/v1/nextNonce`
- Verify auth fields included and nonce monotonicity.
- Verify error handling on 401/403 and non-2xx.
