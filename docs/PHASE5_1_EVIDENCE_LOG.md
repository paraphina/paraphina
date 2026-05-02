# Phase 5.1 Evidence Log

This log records compact evidence pointers for Phase 5.1 non-live runs. Raw
run artifacts under `runs/` are ignored by Git because they contain large
telemetry snapshots; this file preserves the reproducible evidence boundary in
the repository.

## LTR-EV-SHADOW-001 Phase 5 Tail M4

- Run id: `LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m4`
- Local run directory: `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m4`
- Source snapshot: `/tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl`
- Input records scanned: `1000`
- Input bytes: `35781912`
- Input SHA256: `c2b50d00912b22f877e6e79be0ae16e2342d5ea3eaad22b7be3049f059312b64`
- Output telemetry records: `4001`
- Candidates evaluated: `2000`
- Replay labels emitted: `2000`
- Gate status: `HOLD`
- Calibration status: `SPARSE`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`
- Replay timestamp: `1709159174120450357`
- Replay timestamp UTC: `2024-02-28T22:26:14.120450+00:00`
- Timestamp semantics: deterministic replay timestamp, not wall-clock artifact creation time.

Command:

```bash
python3 tools/phase51_ev_shadow.py \
  --input-telemetry /tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl \
  --run-id LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m4 \
  --output-root runs/phase51_lighter_only_ev_shadow
```

Validation:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m4/telemetry.jsonl
```

Result:

```text
OK: 4001 record(s) validated against schema v2
```

Artifact hashes:

```text
35e35982d0fdf154313f9fde514f46932124e4e34da316e75654ef7c80e2975d  telemetry.jsonl
20977f31533f91980d1ecd6f28d08880d8f5067d2874e702bd5c806dabb5401c  manifest.json
21858114e9e391e2b7c68e72e766f7d5c7c409df806a3e934f7fbacc67bcb89d  evidence_pack/artifact_index.json
```

HOLD reason counts:

```text
missing_pfill_calibration: 2000
missing_markout_calibration: 2000
missing_hedge_success_calibration: 2000
missing_queue_reset_calibration: 2000
missing_churn_calibration: 2000
missing_tail_risk_calibration: 2000
sparse_calibration_bucket: 2000
counterfactual_only_nonfinancial: 2000
```

## Phase 5.1 Board Decision M8

- Decision document: `docs/PHASE5_1_BOARD_DECISION.md`
- Decision date: `2026-05-01`
- Decision: `PROMOTE_FOR_NEXT_NONLIVE_STEP`
- Scope: non-live scaffold only
- EV admission status: `HOLD`
- Live/canary status: `HOLD`
- Economic/profitability status: `HOLD`
- Next non-live work: read-only Lighter account-state/native-limit capture and
  calibration-label ingestion

## Phase 5.1b Lighter Account/Native-Limit Gate

- Gate id: `PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS`
- Gate spec: `configs/phase51b_lighter_account_native_limits.json`
- Collector: `tools/phase51b_lighter_account_limits.py`
- Scope: Lighter-only read-only account-state/native-limit evidence
- Gate status: `implemented_for_readonly_capture`
- Live/canary/capital/risk authorization: all false
- Required accepted evidence before calibration-label ingestion:
  `V2_LIGHTER_ACCOUNT_PROFILE`, `V2_LIGHTER_ACCOUNT_LIMITS`,
  `V2_LIGHTER_ACTIVE_ORDERS`, optional
  `V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE`, sanitized source snapshots, manifest,
  artifact index, and schema v2 validation output.
- Current limitation: real account/native-limit capture is accepted for
  non-live evidence review only; calibration-label ingestion remains blocked
  until external schema validation and secret audit pass.

### Read-Only Capture 2026-05-02

- Run id: `PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z`
- Local run directory:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z`
- Created UTC: `2026-05-02T00:25:50.978342+00:00`
- Collector commit: `319089d`
- Gate status: `HOLD`
- `phase51b_capture_complete`: `true`
- `approved_for_nonlive_evidence_review`: `true`
- `approved_for_calibration_label_ingestion`: `false`
- Calibration hold reason:
  `requires_external_schema_validation_and_secret_audit`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- Secret scan: `sensitive_env_values_checked=5`,
  `sensitive_value_leak_found=false`

Command:

```bash
python3 tools/phase51b_lighter_account_limits.py \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --fetch-readonly \
  --include-trades \
  --allow-sdk-auth \
  --lighter-sdk-path /tmp/lighter_sdk \
  --run-id PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z
```

Validation:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/telemetry.jsonl
```

Result:

```text
OK: 5 record(s) validated against schema v2
```

Extracted non-secret readiness facts:

```text
event_types:
  V2_RUN_CONTEXT
  V2_LIGHTER_ACCOUNT_PROFILE
  V2_LIGHTER_ACCOUNT_LIMITS
  V2_LIGHTER_ACTIVE_ORDERS
  V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE
lighter_account_type: ACCOUNT_TYPE_0
lighter_account_profile_status: OBSERVED
market_id: 0
market_symbol: ETH
market_metadata_status: OBSERVED
maker_fee_bps: 0.0
taker_fee_bps: 0.0
price_decimals: 2
size_decimals: 4
lighter_user_tier: premium
lighter_user_tier_name: premium
current_maker_fee_tick: 40
current_taker_fee_tick: 280
active_orders_count_total: 0
active_orders_count_market: 0
open_order_limit_status: UNKNOWN
trade_sample_count: 100
maker_trade_count: 56
taker_trade_count: 44
unknown_role_trade_count: 0
maker_taker_attribution_status: OBSERVED
```

Artifact hashes:

```text
feb1f074fac83cc9331dfbb8aa46d51d30318bd1bee1ba5f1fc2fa59c18cbbb2  telemetry.jsonl
8095d5a35c2e21bd25deb8377210d41344d839573b4247a8f130367ee2f99ad5  lighter_account_native_limits_summary.json
fbbee0193abf774c2dc75c57fd6f4b36aa925465410b4d6d6d7f00b88607355c  gate_result.json
6bca31f446a3504719a680213e634aab13df7c014a59f5b40d57a08af52853d2  evidence_pack/artifact_index.json
951551534a89570a18ae3a2d02bce8be9d7e71521752c4402289803658c1ac93  command_log.json
96907e31dcbbfb4d171724d5a6becc56b31e7f7cb8d84a544735367c07a2c93e  source_snapshots/account.sanitized.json
f564e0b56517e70dad5e7c19f5e3a005a1b9a1049201a41a55c0be5dcaa07d8f  source_snapshots/account_limits.sanitized.json
38fc4fdf6355aeb29cbccc4716e1bd9612ba3c0c21f10506b19651700fdf8a64  source_snapshots/active_orders.sanitized.json
e2b930d2f0cfc0e1bacdcaaca6ac907746cd743cbfd934689b0022779bb83df7  source_snapshots/order_books.sanitized.json
249c441525520716a4c139dbb03d386e825d7102d7ea96d480fb511c737c1d0c  source_snapshots/trades.sanitized.json
06537f4610394509a790bd18c5ee0185ef09c4888394e7a3f87c42658f14312b  spec_resolved.json
```

## LTR-LIGHTER-VENUE-READINESS M5

- Evidence id: `LTR-LIGHTER-VENUE-READINESS-M5`
- Evidence document: `docs/PHASE5_1_LIGHTER_VENUE_READINESS.md`
- Evidence date: `2026-05-01`
- Scope: Lighter-only venue-readiness documentation and connector evidence
- Gate status: `HOLD`
- M5 status: `complete_for_nonlive_evidence_pack`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Official sources reviewed:

```text
https://docs.lighter.xyz/trading/trading-fees
https://apidocs.lighter.xyz/docs/account-types
https://apidocs.lighter.xyz/docs/api-keys
https://apidocs.lighter.xyz/docs/rate-limits
https://apidocs.lighter.xyz/docs/volume-quota-program
https://docs.lighter.xyz/trading/order-types-and-matching
https://apidocs.lighter.xyz/reference/sendtx
https://apidocs.lighter.xyz/reference/orderbooks
https://apidocs.lighter.xyz/reference/account-1
https://apidocs.lighter.xyz/reference/accountlimits
https://apidocs.lighter.xyz/reference/accountactiveorders
https://apidocs.lighter.xyz/reference/trades
https://apidocs.lighter.xyz/docs/websocket-reference
```

Evidence summary:

```text
PASS: Lighter remains the first Phase 5.1 venue-local non-live target.
PASS: Local connector has create, IOC, cancel, replace, cancel-all, public WS,
      private WS, and account polling coverage.
FLAG: Account tier, fee tier, native venue limits, replace post-only
      preservation, and fill/calibration evidence remain unresolved.
HOLD: No live/canary/economic promotion is supported by M5.
```

## LTR-EV-SHADOW-001 Phase 5 Tail M6

- Run id: `LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6`
- Local run directory: `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6`
- Source snapshot: `/tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl`
- Input records scanned: `1000`
- Input SHA256: `c2b50d00912b22f877e6e79be0ae16e2342d5ea3eaad22b7be3049f059312b64`
- Output telemetry records: `4001`
- Candidates evaluated: `2000`
- Replay labels emitted: `2000`
- Gate status: `HOLD`
- Calibration status: `SPARSE`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`
- `admissible_for_financial_claim`: `false`
- Replay timestamp: `1704529507787138797`
- Replay timestamp UTC: `2024-01-06T08:25:07.787139+00:00`
- Timestamp semantics: deterministic replay timestamp, not wall-clock artifact creation time.

M6 invariant additions:

```text
pair_conditioned_flag: false
fast_hedge_allowed: false
fast_hedge_serialization_state: NOT_APPLICABLE_NONLIVE_SHADOW
residual_state_required: false
residual_state_status: NO_FILL_NO_RESIDUAL
action_owner: NO_ACTION_NONLIVE_SHADOW
double_action_prevention_state: NO_EXECUTION_EVENTS_EMITTED
```

Command:

```bash
python3 tools/phase51_ev_shadow.py \
  --input-telemetry /tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl \
  --run-id LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6 \
  --output-root runs/phase51_lighter_only_ev_shadow
```

Validation:

```bash
python3 -m py_compile tools/check_telemetry_contract.py tools/phase51_ev_shadow.py
python3 -m unittest tests.test_telemetry_contract_gate
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6/telemetry.jsonl
```

Result:

```text
Ran 71 tests
OK
OK: 4001 record(s) validated against schema v2
```

Artifact hashes:

```text
6ddbf774f6fb05f11508866d604d6ac7da89a54ef22e83a9bca6c32eb59090dc  telemetry.jsonl
884e4cd3459ff99d92db8ab25479253fd7f7b75a2f0296da205c4f3b8ac27b54  manifest.json
08ffe38dc4e4b8a0f8e8f5918ddd963815583098f0f994987ab65be6dcd003e8  evidence_pack/artifact_index.json
```

HOLD reason counts:

```text
missing_pfill_calibration: 2000
missing_markout_calibration: 2000
missing_hedge_success_calibration: 2000
missing_queue_reset_calibration: 2000
missing_churn_calibration: 2000
missing_tail_risk_calibration: 2000
sparse_calibration_bucket: 2000
counterfactual_only_nonfinancial: 2000
```
