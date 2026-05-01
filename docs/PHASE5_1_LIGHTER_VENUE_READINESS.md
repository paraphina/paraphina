# Phase 5.1 Lighter Venue Readiness Evidence Pack

Date: 2026-05-01

Scope: Phase 5.1 non-live venue-readiness evidence only. This document does
not authorize live orders, canary promotion, capital escalation, risk-limit
relaxation, or runtime service changes.

## Executive Verdict

M5 status: `complete_for_nonlive_evidence_pack`

Venue verdict: `PASS` for continuing Phase 5.1 with Lighter as the first
venue-local non-live shadow/replay target.

Promotion verdict: `HOLD` for any live, canary, capital, or production-readiness
claim.

The reason is narrow and operational: Lighter has documented post-only order
semantics, account-mode-specific fees/latencies, public/private data channels,
signed transaction paths, account/order endpoints, and local connector coverage.
However, account tier, venue-native limits, replace post-only preservation, and
fill/calibration evidence remain insufficient for any economic or live-readiness
claim.

## Official Source Snapshot

Official Lighter sources reviewed on 2026-05-01:

| Source | URL | Evidence used |
|---|---|---|
| Trading fees | https://docs.lighter.xyz/trading/trading-fees | Standard accounts currently have 0 maker and 0 taker fees; Premium accounts have explicit maker/taker fees and latency differences. |
| Account types | https://apidocs.lighter.xyz/docs/account-types | Premium is opt-in and suitable for HFT; Standard is default; account switches require no open positions, no open orders, and a 24-hour cooldown. |
| API keys | https://apidocs.lighter.xyz/docs/api-keys | API keys are per internal account with separate nonces; Premium accounts may mark maker-only keys for post-only, modify-on-ALO, cancel, and cancel-all paths. |
| Rate limits | https://apidocs.lighter.xyz/docs/rate-limits | REST, WebSocket, sendTx/sendTxBatch, transaction-type, pending-order, and active-order limits are account/IP/profile-sensitive. |
| Volume quota | https://apidocs.lighter.xyz/docs/volume-quota-program | Volume quota is currently Premium-only, applies to create/cancel-all/modify/grouped order transactions, and is shared across sub-accounts under the same L1 address. |
| Order types and matching | https://docs.lighter.xyz/trading/order-types-and-matching | Post-only limit orders are canceled if they would cross; IOC orders are not maker-book orders; matching uses price-time priority. |
| sendTx | https://apidocs.lighter.xyz/reference/sendtx | Signed transactions are submitted through `POST /api/v1/sendTx` with `tx_type` and `tx_info`. |
| orderBooks | https://apidocs.lighter.xyz/reference/orderbooks | Market metadata includes maker/taker fee percentages, min amounts, and supported price/size decimals. |
| account | https://apidocs.lighter.xyz/reference/account-1 | Account data can be requested by account index or L1 address. |
| accountLimits | https://apidocs.lighter.xyz/reference/accountlimits | Account limit data is available through an authenticated endpoint. |
| accountActiveOrders | https://apidocs.lighter.xyz/reference/accountactiveorders | Active orders are available through an authenticated endpoint. |
| trades | https://apidocs.lighter.xyz/reference/trades | Trades endpoint supports maker/taker role filtering and account-index queries. |
| WebSocket reference | https://apidocs.lighter.xyz/docs/websocket-reference | Order/trade/account payloads expose order status, maker/taker attribution fields, fees, timestamps, positions, and order counts. |

## Account/Profile Decision

Phase 5.1 must not assume Premium, maker-only API keys, zero latency, fee
discounts, staking benefits, or volume quota unless the account state proves
them.

Default research stance:

| Item | Decision |
|---|---|
| First experiment account assumption | Standard unless authenticated account evidence says otherwise. |
| Fee model | Load from account/market metadata where possible; otherwise mark fee/profile assumptions as `HOLD`. |
| Premium mode | Not assumed. Treat as a future board decision because it changes fee, latency, and quota economics. |
| Maker-only API keys | Not assumed. Treat as a future execution-hardening option after account-tier review. |
| Sub-accounts | Not a quota bypass. Volume quota is shared across sub-accounts under the same L1 address according to official docs. |

## Local Connector Evidence

The local implementation has substantial Lighter coverage, but this evidence is
connector-level and test-level only. It is not a live venue-readiness proof.

| Area | Local evidence | Verdict |
|---|---|---|
| Shadow/read-only WebSocket mode | `paraphina/src/live/connectors/lighter.rs:114` enables read-only public WS behavior in shadow mode. | PASS |
| Public order book handling | `paraphina/src/live/connectors/lighter.rs:1200` treats the first message as snapshot and later messages as deltas with continuity checks. | PASS |
| Account polling backoff | `paraphina/src/live/connectors/lighter.rs:1379` applies 429-aware account polling backoff. | PASS |
| Private execution stream | `paraphina/src/live/connectors/lighter.rs:1498` runs private WS with reconnect/backoff and translates private execution events. | PASS |
| Create order signer payload | `paraphina/src/live/connectors/lighter.rs:5131` builds signed `create_order` payloads with market, client index, price, size, TIF, post-only, and reduce-only fields. | PASS |
| Signer field contract | `paraphina/src/live/connectors/lighter_signer.rs:66` defines signed create/cancel/cancel-all/modify payload structs. | PASS |
| Cancel path | `paraphina/src/live/connectors/lighter.rs:5207` signs `cancel_order` and supports client-order-index or exchange-order-index identity. | PASS |
| Native replace gate | `paraphina/src/live/connectors/lighter.rs:5272` only allows native replace for MM/GTC/post-only/non-reduce-only requests. | PASS locally, FLAG externally |
| Replace payload | `paraphina/src/live/connectors/lighter.rs:5337` signs `modify_order`, but the modify payload does not carry post-only/TIF/reduce-only fields. | FLAG |
| Cancel-all path | `paraphina/src/live/connectors/lighter.rs:5384` signs `cancel_all` with nil cancel-all sentinels. | PASS |
| Error classification | `paraphina/src/live/connectors/lighter.rs:5437` maps post-only, reduce-only, and rate-limit errors into typed gateway errors. | PASS |

## Existing Test Evidence

Existing tests already freeze the major signer and lifecycle payload paths:

| Test evidence | File reference | Verdict |
|---|---|---|
| MM GTC post-only create order carries `post_only:1` into signer payload and submits `sendTx`. | `paraphina/src/live/connectors/lighter.rs:2860` | PASS |
| Reduce-only IOC uses IOC TIF and nil `order_expiry`. | `paraphina/src/live/connectors/lighter.rs:2943` | PASS |
| Emergency IOC timeout only applies to reduce-only exit/hedge, not MM. | `paraphina/src/live/connectors/lighter.rs:3023` | PASS |
| Numeric client-order IDs above u48 are rejected. | `paraphina/src/live/connectors/lighter.rs:3097` | PASS |
| Private fill translation preserves venue identity, IDs, side, price, size, purpose, and fee bps. | `paraphina/src/live/connectors/lighter.rs:3150` | PASS |
| Cancel order signs and submits `cancel_order`. | `paraphina/src/live/connectors/lighter.rs:3230` | PASS |
| Cancel order uses exchange order index for large IDs. | `paraphina/src/live/connectors/lighter.rs:3300` | PASS |
| Replace order signs and submits `modify_order`. | `paraphina/src/live/connectors/lighter.rs:3371` | PASS locally, FLAG externally |
| Replace rejects non-MM or reduce-only paths. | `paraphina/src/live/connectors/lighter.rs:3530` | PASS |
| Cancel-all signs and submits `cancel_all`. | `paraphina/src/live/connectors/lighter.rs:3572` | PASS |

No additional readiness contract test is required for M5 because the existing
tests already freeze the create, IOC, cancel, replace, and cancel-all local
payload contracts. The unresolved items are not missing local unit coverage;
they are account-state and authoritative venue-behavior evidence gaps.

## Readiness Matrix

| Requirement | Status | Rationale |
|---|---|---|
| Non-live-only Phase 5.1 scope | PASS | M5 changes are documentation only and do not touch runtime behavior. |
| Post-only maker protection documented | PASS | Official docs state post-only limit orders are canceled if they would cross and otherwise rest as maker orders. |
| Post-only flag wired locally | PASS | Local create-order path carries `post_only` into signer payload and has unit coverage. |
| IOC not treated as maker path | PASS | Official docs say IOC orders are not placed into the book as maker orders; local emergency IOC is limited to reduce-only exit/hedge paths. |
| Account tier known | FLAG | The repo must not infer Standard/Premium state without authenticated account evidence. |
| Fee tier known | FLAG | Official docs and market metadata expose fees, but Phase 5.1 evidence does not yet bind actual account/market fee assumptions into `V2_BALANCE_SNAPSHOT` or `V2_EV_EVALUATED`. |
| Latency model known | FLAG | Standard/Premium have materially different latency. No calibrated local latency evidence exists for Phase 5.1 EV yet. |
| sendTx/sendTxBatch limits modeled | FLAG | Official docs provide limits, but Phase 5.1 telemetry does not yet emit venue-native headroom for Lighter. |
| Volume quota modeled | FLAG | Premium-only quota exists and is shared under the L1 address. It is not yet represented in the V2 churn/rate-limit EV terms. |
| Open-order limits modeled | FLAG | Official docs expose pending/active order limits; current V2 shadow evidence does not yet log native order-limit headroom. |
| Account limits endpoint integrated into readiness telemetry | FLAG | Official endpoint exists, but Phase 5.1 evidence does not yet ingest it into V2 events. |
| Active orders endpoint integrated into readiness telemetry | FLAG | Official endpoint exists, but Phase 5.1 evidence does not yet ingest it into V2 events. |
| Private fills support maker/taker and fee attribution | FLAG | Official payloads include maker/taker/fee fields; local generic fill translation retains fee bps, but Phase 5.1 v2 fill attribution is not yet complete. |
| Replace post-only preservation proven externally | FLAG | Local replace is gated to MM/GTC/post-only/non-reduce-only, but signed modify payload does not explicitly carry post-only/TIF/reduce-only. Venue-side modify semantics require official or paper/testnet proof before promotion. |
| Live/canary readiness | HOLD | No live/canary authorization, no balance-authoritative evidence, no fill calibration, and no account-state evidence. |

## Risks and Controls

| Risk | Control |
|---|---|
| Treating Lighter as universally superior because it is first | Do not generalize. Lighter is first only because M5 can evidence its connector/doc contract fastest under non-live scope. |
| Assuming zero fees incorrectly | Bind fee assumptions to account type and market metadata before any EV or PnL claim. |
| Assuming Premium/HFT latency without account proof | Default to Standard unless authenticated account evidence proves Premium. |
| Treating sub-accounts as a quota bypass | Do not do this. Official volume quota is shared across sub-accounts under the same L1 address. |
| Replace semantics create taker leakage | Keep live/canary blocked until modify-on-ALO behavior is proven by official docs, testnet/paper evidence, or account-level maker-only API key evidence. |
| Rate-limit/churn omitted from EV | Keep all Lighter EV candidates in `HOLD` until venue-native rate/quota/open-order headroom is captured in V2 telemetry. |

## M5 Decision

M5 is complete as a venue-readiness evidence pack with `HOLD` promotion status.

Allowed next work:

| Next item | Description |
|---|---|
| M6 risk/system invariants | Add no-live enforcement, metadata propagation, residual-state placeholders, and double-action prevention precondition tests. |
| Phase 5.1b Lighter account-state telemetry | Use `tools/phase51b_lighter_account_limits.py` with `configs/phase51b_lighter_account_native_limits.json` to capture account type, account limits, active-order counts/headroom, fee/market metadata, and maker/taker trade-role samples into V2 readiness events. This is read-only and remains `HOLD` for live/canary/economics. |
| Lighter fill/markout calibration | Keep blocked until enough non-live/paper/testnet/observed labels exist for P-fill, maker/taker attribution, adverse selection, queue reset, churn, and tail-risk terms. |

Rejected next work:

| Rejected item | Reason |
|---|---|
| Live Lighter orders | Outside Phase 5.1 scope. |
| Canary promotion | Outside Phase 5.1 scope and unsupported by evidence. |
| Premium switch | Requires separate board/account decision and would change fee/latency/quota economics. |
| Multi-venue rollout | Phase 5.1 first experiment is venue-local by design. |
