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

## LTR-EV-SHADOW-001 Phase 5 Tail 20k M6 Reference Replay

- Run id: `LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6`
- Local run directory: `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6`
- Source snapshot: `/tmp/phase51_inputs/phase5_tail_20000_20260501T214411Z.telemetry.jsonl`
- Input artifact mode: `reference`
- Input records scanned: `20000`
- Input SHA256: `a05927b7e02d3e1a8987d9b385d09fe1e7908273e6a23d40585ae851cb6c48c9`
- Candidates evaluated: `40000`
- Replay labels emitted: `40000`
- Gate status: `HOLD`
- Calibration status: `SPARSE`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`
- `admissible_for_financial_claim`: `false`
- Replay timestamp: `1701007808057214863`
- Replay timestamp UTC: `2023-11-26T14:10:08.057215+00:00`
- Timestamp semantics: deterministic replay timestamp, not wall-clock artifact creation time.

The source artifact was referenced rather than copied because root disk was at `100%`
utilization and the source telemetry hash was already recorded in the manifest.
This preserves evidence integrity without duplicating a large input file.

Command:

```bash
python3 tools/phase51_ev_shadow.py \
  --input-telemetry /tmp/phase51_inputs/phase5_tail_20000_20260501T214411Z.telemetry.jsonl \
  --run-id LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6 \
  --output-root runs/phase51_lighter_only_ev_shadow \
  --input-artifact-mode reference
```

Validation:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6/telemetry.jsonl
```

Result:

```text
OK: 80001 record(s) validated against schema v2
```

Artifact hashes:

```text
bcb62978eb22f5105dba92a420e336dc41cd9d28853a2683dcad3e1b3bcf7420  telemetry.jsonl
558e1633b7557563d9f49f82363e1e83a081c6ab2c68ae186bfcdb439827b9c6  ev_shadow_summary.json
0030db862079c8e05603b392e5730b23d06160162d468e3491d701064c31a9f5  evidence_pack/artifact_index.json
```

HOLD reason counts:

```text
missing_pfill_calibration: 40000
missing_markout_calibration: 40000
missing_hedge_success_calibration: 40000
missing_queue_reset_calibration: 40000
missing_churn_calibration: 40000
missing_tail_risk_calibration: 40000
sparse_calibration_bucket: 40000
counterfactual_only_nonfinancial: 40000
```

## PHASE51C-LABEL-LAKE-20K

- Run id: `PHASE51C-LABEL-LAKE-20K-20260502T004902Z`
- Local run directory: `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-20K-20260502T004902Z`
- Source snapshot: `/tmp/phase51_inputs/phase5_tail_20000_20260501T214411Z.telemetry.jsonl`
- EV shadow telemetry: `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6/telemetry.jsonl`
- Phase 5.1b acceptance artifact: `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json`
- Gate status: `HOLD`
- Gate reason: `label_lake_scaffold_missing_fill_markout_balance_coverage`
- Record count: `66814`
- Quote decision labels: `40000`
- Order lifecycle labels: `26813`
- Fill labels: `0`
- Markout labels: `0`
- Balance reconciliation labels: `0`
- Balance reconciliation status: `MISSING`
- Native limit pressure status: `UNKNOWN`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_label_lake.py \
  --source-telemetry /tmp/phase51_inputs/phase5_tail_20000_20260501T214411Z.telemetry.jsonl \
  --ev-shadow-telemetry runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_20k_m6/telemetry.jsonl \
  --phase51b-acceptance runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json \
  --run-id PHASE51C-LABEL-LAKE-20K-20260502T004902Z
```

Order action counts:

```text
cancel: 10128
place: 10130
replace: 6555
```

Quote decision counts:

```text
HOLD: 40000
```

Artifact hashes:

```text
d1f168393a5e36e3413c82931db6096c202b763992c7351a7d9e828e521bb0d7  labels.jsonl
94ce7b8e7f21c5f9610a12c96b7e99b78a19cb5a968fb339a1afced4d1869e48  label_lake_summary.json
fdc37d39d4c7ec7f91440dd44d47ff8fb4d1eca5eb04c0307405ecb1b02a016f  evidence_pack/artifact_index.json
```

## PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S

- Run id: `PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T073231Z`
- Local run directory: `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T073231Z`
- Source telemetry:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/telemetry_bounded.jsonl`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Balance pre snapshot:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_pre_snapshot.json`
- Balance post snapshot:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_post_snapshot.json`
- Balance comparison:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_snapshot_comparison.json`
- Gate status: `HOLD`
- Gate reason: `observed_label_pack_missing_maker_taker_attribution`
- Fill labels: `356`
- Markout labels: `1424`
- Balance reconciliation labels: `1`
- Record count: `1781`
- Fill label status: `OBSERVED`
- Markout label status: `OBSERVED`
- Markout horizons: `100ms`, `500ms`, `1000ms`, `5000ms`
- Balance reconciliation status: `OBSERVED`
- Maker/taker role counts: `UNKNOWN=356`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_observed_labels.py \
  --source-telemetry /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/telemetry_bounded.jsonl \
  --balance-pre /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_pre_snapshot.json \
  --balance-post /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_post_snapshot.json \
  --balance-comparison /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/balance_snapshot_comparison.json \
  --run-id PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T073231Z
```

Per-venue fill counts:

```text
aster: 141
extended: 10
hyperliquid: 4
lighter: 189
paradex: 12
```

Artifact hashes:

```text
6e9a35bc43ef19eb1cb34c9cd78de8cc28217276e8e1a436ac3dca2640335ac5  labels.jsonl
23e53b090394ffaade8b3a3f09f6014df545ea2d549fcde44b2c51ee5d7c2acd  observed_label_summary.json
3f3dd64278c1276e12052d44784546e01fbb8d8080a2dcbb9515f4001bd90fa7  evidence_pack/artifact_index.json
```

## PHASE51C-LABEL-LAKE-20260502T004621Z

- Run id: `PHASE51C-LABEL-LAKE-20260502T004621Z`
- Local run directory:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-20260502T004621Z`
- Created UTC: `2026-05-02T00:46:33.634536+00:00`
- Tool commit: `5857163`
- Gate status: `HOLD`
- Gate reason: `label_lake_scaffold_missing_fill_markout_balance_coverage`
- Source telemetry:
  `/tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl`
- Source telemetry SHA256:
  `c2b50d00912b22f877e6e79be0ae16e2342d5ea3eaad22b7be3049f059312b64`
- EV shadow telemetry:
  `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6/telemetry.jsonl`
- EV shadow telemetry SHA256:
  `6ddbf774f6fb05f11508866d604d6ac7da89a54ef22e83a9bca6c32eb59090dc`
- Phase 5.1b acceptance artifact:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json`
- Phase 5.1b acceptance SHA256:
  `de68ee111a4dcf7eababb9ddc1d0886add484912916a82dabed63f72ab2b8c84`
- Record count: `3319`
- Quote-decision labels: `2000`
- Order-lifecycle labels: `1318`
- Fill labels: `0`
- Markout labels: `0`
- Balance-reconciliation labels: `0`
- Native limit pressure status: `UNKNOWN`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_label_lake.py \
  --source-telemetry /tmp/phase51_inputs/phase5_tail_1000_20260501T214411Z.telemetry.jsonl \
  --ev-shadow-telemetry runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_phase5_tail_20260501T214411Z_m6/telemetry.jsonl \
  --phase51b-acceptance runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json \
  --run-id PHASE51C-LABEL-LAKE-20260502T004621Z
```

Result:

```text
phase51c_label_lake: status HOLD (label scaffold only)
manifest_hashes_valid=true
```

Label summary:

```text
quote_decision_counts: HOLD=2000
order_action_counts: cancel=505, place=501, replace=312
fill_label_status: MISSING
markout_label_status: MISSING
balance_reconciliation_status: MISSING
native_limit_pressure_status: UNKNOWN
```

Artifact hashes:

```text
35db92cb71234a88c02e98593cea24f0d6b0555c4a9b9ab618a9c530d4d5164f  labels.jsonl
1372b298a6ce8e09b24c454714eda0df1655a22147f602362047e8682856c30e  label_lake_summary.json
f463e1f62d8302849013579cf14854e7ded6a8669c73e9ad7e3ac4edd98068f3  evidence_pack/artifact_index.json
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
- Acceptance artifact:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json`
- Acceptance status: `PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION`
- Acceptance verdict: accepted only for beginning Gate 5.1c
  calibration-label ingestion; live, canary, capital escalation,
  risk-limit relaxation, and financial-authority use remain blocked.
- Acceptance limitations:
  `lighter_sendtx_limit_not_exposed_by_account_limits_payload`,
  `lighter_rest_request_limit_not_exposed_by_account_limits_payload`,
  `lighter_open_order_limit_headroom_unknown`.
- Ingestion rule: native order-limit/headroom fields must be ingested as
  explicit `UNKNOWN`/hold labels, not as usable numeric limit pressure.

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

Acceptance:

```bash
python3 tools/phase51b_accept_evidence.py \
  runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z \
  --sensitive-env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env
```

Result:

```text
phase51b_accept_evidence: status PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION
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
de68ee111a4dcf7eababb9ddc1d0886add484912916a82dabed63f72ab2b8c84  phase51b_acceptance.json
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
