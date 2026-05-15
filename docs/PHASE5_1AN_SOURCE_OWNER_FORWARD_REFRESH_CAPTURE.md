# Phase 5.1an Source-Owner Forward-Refresh Capture

Phase 5.1an is a HOLD-only source-owner inbox scaffold for the Phase 5.1
forward-refresh route. It does not fetch venue data, touch credentials, place
orders, cancel orders, call `sendTx` or `sendTxBatch`, infer source links,
infer maker/taker role, infer Lighter pressure, authorize live/canary/capital
changes, train models, admit EV candidates, or support financial claims.

The source owner supplies exactly one operator-authored evidence file:

```text
/home/ubuntu/source_owner_inbox/phase51/forward_refresh.jsonl
```

The file must contain sanitized JSONL rows only. Raw order IDs, raw client IDs,
trade IDs, transaction hashes, credentials, signed payloads, `.env` content,
and unsafe true flags are rejected before Phase 5.1al materialization.

## Tool

```text
tools/phase51an_source_owner_forward_refresh_capture.py
tests/test_phase51an_source_owner_forward_refresh_capture.py
```

Default command:

```bash
python3 tools/phase51an_source_owner_forward_refresh_capture.py \
  --inbox /home/ubuntu/source_owner_inbox/phase51 \
  --update-intake-manifest
```

When `forward_refresh.jsonl` is empty, the tool writes:

```text
/home/ubuntu/source_owner_inbox/phase51/SOURCE_OWNER_CAPTURE_CONTRACT.md
/home/ubuntu/source_owner_inbox/phase51/intake.generated.json
```

With `--update-intake-manifest`, it also rewrites:

```text
/home/ubuntu/source_owner_inbox/phase51/intake.json
```

to a safe waiting manifest with all artifact lists empty and all live/capital
permissions false.

When `forward_refresh.jsonl` has valid rows, Phase 5.1an calls Phase 5.1al,
then writes an intake manifest whose `phase51al_summaries` field points to the
generated `phase51al_forward_refresh_capture_summary.json`. The raw
`forward_refresh.jsonl` is not routed directly into Phase 5.1aj or Phase 5.1ab.

## Accepted Row Families

Native role row:

```json
{"target_type":"native_role","venue_id":"extended","canonical_group_id":"target-group","order_key":"target-order","isTaker":false,"no_live_flag":true,"approved_for_live":false,"live_orders_allowed":false}
```

Accepted native-role fields:

- Aster: `m` or `maker_side`, plus positive `l` or `lastFilledQty`.
- Extended: `isTaker` or `is_taker`.
- Hyperliquid: `crossed`.
- Lighter: `account_index`, `is_maker_ask`, `ask_account_id`,
  `bid_account_id`.
- Paradex: `liquidity` as `MAKER` or `TAKER`.

Lighter pressure row:

```json
{"target_type":"lighter_native_limit","venue_id":"lighter","canonical_group_id":"target-group","order_key":"target-order","active_order_headroom_account":10,"active_order_headroom_market":10,"sendtx_per_minute_limit":100,"sendtx_per_minute_remaining":80,"rest_requests_per_minute_limit":1000,"rest_requests_per_minute_remaining":900,"native_limit_event_time_status":"EVENT_TIME_ALIGNED","no_live_flag":true,"approved_for_live":false,"live_orders_allowed":false}
```

Lighter pressure requires:

- `active_order_headroom_account`
- `active_order_headroom_market`
- `sendtx_per_minute_limit`
- `sendtx_per_minute_remaining`
- REST limit/remaining or weighted request limit/remaining
- `native_limit_event_time_status=EVENT_TIME_ALIGNED`

## Downstream Flow

After Phase 5.1an updates the intake manifest, Phase 5.1am is the next
orchestrator:

```bash
python3 tools/phase51am_nonlive_executive_orchestrator.py \
  --source-owner-intake-manifest /home/ubuntu/source_owner_inbox/phase51/intake.json
```

If a real Phase 5.1al pack was materialized, Phase 5.1am selects the
`forward_refresh` route and emits the Phase 5.1ak validation command. Phase
5.1ak remains HOLD-only and does not itself authorize promotion.
