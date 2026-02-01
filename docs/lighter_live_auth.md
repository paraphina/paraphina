# Lighter Live Auth (PR2 Spec)

## A) Why PR2 exists (repo-proven current state)
- `lighter.rs` currently uses unsigned placeholder REST endpoints:
  - `GET {rest_url}/account`
  - `POST {rest_url}/orders`, `POST {rest_url}/cancel`, `POST {rest_url}/cancel_all`
- Market data via WS is implemented and works; authenticated account snapshots and execution are not real.

## B) Authoritative endpoints we will implement (names only)
- REST base: `https://mainnet.zklighter.elliot.ai`
- Account: `GET /api/v1/account` (by account index or L1 address)
- Transactions: `POST /api/v1/sendTx` and `POST /api/v1/sendTxBatch`
- Nonce: `GET /api/v1/nextNonce` (bootstrap only; local nonce is preferred)

Hard requirement: **must sign tx body before sending**.

## C) Required env vars (for PR2)
- `LIGHTER_API_KEY_INDEX` (u64)
- `LIGHTER_ACCOUNT_INDEX` (u64)
- `LIGHTER_API_PRIVATE_KEY_HEX` (hex string)
- `LIGHTER_AUTH_TOKEN` (optional; only if required by some endpoints)
- `LIGHTER_NONCE_PATH` (optional persistence)
- `LIGHTER_SIGNER_URL` (optional fallback if signer bridge is used)

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

## F) Implementation plan (two options)
### Option 1: Native Rust signer (preferred long-term)
**Pros**
- No external dependency, simpler deployment.
- Lower latency and fewer failure modes.

**Cons**
- Requires correct crypto implementation and careful key handling.

**Required tests (mock server)**
- Mock `sendTx` / `sendTxBatch` accept signed payloads.
- Verify signature presence and per-request nonce monotonicity.

### Option 2: Signer bridge (fastest)
Call an external signer service/binary to obtain a signed tx blob, then submit to `sendTx`.

**Pros**
- Faster to ship; signing logic isolated from Paraphina.
- Easier to update signer independently.

**Cons**
- External dependency and additional operational surface.
- Requires signer availability and secure transport.

**Required tests (mock server)**
- Mock signer responses for valid/invalid signatures.
- Mock `sendTx` / `sendTxBatch` submission with signed blob.

## G) Test plan (mock-only)
- Mock endpoints:
  - `GET /api/v1/account`
  - `POST /api/v1/sendTx`
  - `POST /api/v1/sendTxBatch`
  - `GET /api/v1/nextNonce`
- Verify auth fields included and nonce monotonicity.
- Verify error handling on 401/403 and non-2xx.
