# Phase 5.1 Evidence Log

This log records compact evidence pointers for Phase 5.1 non-live runs. Raw
run artifacts under `runs/` are ignored by Git because they contain large
telemetry snapshots; this file preserves the reproducible evidence boundary in
the repository.

## Phase 5.1v Source-Link Readiness Patch

Date: 2026-05-04

Purpose: make Phase 5.1v apply validated source-link sidecars during local
bundle-readiness validation, matching the Phase 5.1s/5.1r source-link contract.
This is a non-live systems/evidence-gate patch only.

Changed repo-owned behavior:

```text
accepted: local source row with required native fields + source-link hash to a Phase 5.1u target
rejected: duplicate source-link hashes
held: source-link-only manifests with no native source rows
not allowed: source truth inference, maker/taker inference, Lighter limit inference, EV/PnL claims
```

Validation:

```bash
python3 -m py_compile tools/phase51v_forward_capture_bundle_readiness.py tests/test_telemetry_contract_gate.py

python3 -m unittest \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_accepts_local_bundle \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_applies_source_link_sidecar \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_source_link_only_holds \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_rejects_duplicate_source_link_hash \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_holds_template_placeholders \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_rejects_unsafe_manifest

python3 -m unittest tests.test_telemetry_contract_gate

python3 tools/check_docs_integrity.py

git diff --check
```

Result:

```text
py_compile: PASS
focused Phase 5.1v unittest: PASS, 6 tests
tests.test_telemetry_contract_gate: PASS, 127 tests
docs integrity: PASS
git diff --check: PASS
gate_status remains HOLD
clears_phase51_blockers remains false
```

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

## Phase 5.1x Hyperliquid Native-Role Adapter

- Run id: `PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z`
- Local run directory: `runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z`
- Source type: read-only Hyperliquid `info` / `userFills` snapshot captured to `/tmp`
- Official source: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint`
- Raw source commit status: not repo-owned and not committed
- Redacted output: `hyperliquid_forward_native_role_snapshot.jsonl`
- Gate status: `HOLD`
- Hyperliquid native-role targets recovered: `6 / 6`
- All-five native-role targets recovered after Phase 5.1v: `6 / 287`
- Lighter native-limit targets recovered after Phase 5.1v: `0 / 3132`
- Source rows scanned: `2000`
- Source rows with boolean `crossed`: `2000`
- Redacted source rows emitted: `7`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`
- `clears_phase51_blockers`: `false`
- `raw_identifier_redaction_status`: `PASS`

Commands:

```bash
python3 tools/phase51x_hyperliquid_native_role_adapter.py \
  --observed-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --source-json /tmp/phase51x_hyperliquid_userfills_1777896919.json \
  --output-root runs/phase51x_hyperliquid_native_role_adapter \
  --run-id PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z \
  --timestamp-ns 1777896919000000000

python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest /tmp/phase51x_hyperliquid_phase51v_candidate_manifest.json \
  --output-root runs/phase51v_forward_capture_bundle_readiness \
  --run-id PHASE51V-HYPERLIQUID-PARTIAL-SOURCE-HOLD-20260504T000000Z \
  --timestamp-ns 1777896920000000000
```

Phase 5.1v partial-source result:

```text
gate_status: HOLD
gate_reason: phase51v_forward_capture_bundle_incomplete_nonlive_hold
native_role_capture_target_ready_count: 6 / 287
native_role_capture_target_missing_count: 281
lighter_native_limit_capture_target_ready_count: 0 / 3132
source_file_status_counts:
- LOCAL_FILE_READY: 1
generated_phase51s_manifest_ready: false
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

Validation:

```bash
python3 -m py_compile tools/phase51x_hyperliquid_native_role_adapter.py tests/test_telemetry_contract_gate.py
python3 -m unittest \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51x_hyperliquid_native_role_adapter_emits_phase51v_ready_rows \
  tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51x_hyperliquid_native_role_adapter_rejects_network_sources
```

Result:

```text
Ran 2 tests
OK
```

Board verdict: `PROMOTE` only for Hyperliquid subset source-readiness evidence.
The system remains `HOLD` for Phase 5.1s, model training, EV admission, canary,
live orders, capital escalation, risk-limit relaxation, financial claims, and
24/7 production readiness until the remaining venue-native role sources and
Lighter event-time native-limit pressure are captured and Phase 5.1v emits a
source-complete generated Phase 5.1s manifest.

## Phase 5.1u Forward Capture Target Manifest

Run id: `PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z`

Local run directory:
`runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z`

Input:
`runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`

Result:

```text
gate_status: HOLD
gate_reason: phase51u_forward_capture_targets_emitted_nonlive_hold
observed_pfill_label_count: 6140
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

Command:

```bash
python3 tools/phase51u_forward_capture_target_manifest.py \
  --observed-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --run-id PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --timestamp-ns 1777766400000000000
```

Validation:

```bash
python3 -m py_compile tools/phase51u_forward_capture_target_manifest.py tools/phase51r_forward_native_source_acquisition.py tests/test_telemetry_contract_gate.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51u_forward_capture_target_manifest_emits_exact_targets tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51u_forward_capture_target_manifest_rejects_unsafe_labels tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_acquisition_aggregates_multi_fill_native_roles
```

Result:

```text
Ran 3 tests
OK
```

Verdict:

```text
HOLD for live, canary, model training, EV admission, capital escalation,
risk-limit relaxation, financial claims, and 24/7 readiness.

PROMOTE only for a fresh read-only forward source-capture pilot against the
Phase 5.1u target manifest, followed by Phase 5.1s -> 5.1r -> 5.1q -> 5.1n ->
5.1h -> 5.1i.
```

## Phase 5.1v Forward Capture Bundle Readiness

Run id: `PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z`

Local run directory:
`runs/phase51v_forward_capture_bundle_readiness/PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z`

Input target run:
`runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z`

Candidate manifest:
`runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z/capture_bundle_manifest_template.json`

Result:

```text
gate_status: HOLD
gate_reason: phase51v_forward_capture_bundle_incomplete_nonlive_hold
native_role_capture_target_ready_count: 0 / 287
lighter_native_limit_capture_target_ready_count: 0 / 3132
source_file_count: 6
source_file_status_counts:
- PLACEHOLDER_PATH: 6
source_link_file_count: 1
source_link_file_status_counts:
- PLACEHOLDER_PATH: 1
generated_phase51s_manifest_ready: false
generated_phase51s_source_count: 0
generated_phase51s_source_link_count: 0
downstream_chain_ready: false
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

Command:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z/capture_bundle_manifest_template.json \
  --output-root runs/phase51v_forward_capture_bundle_readiness \
  --run-id PHASE51V-FORWARD-CAPTURE-BUNDLE-READINESS-TEMPLATE-20260503T000000Z \
  --timestamp-ns 1777766400000000000
```

Validation:

```bash
python3 -m py_compile tools/phase51v_forward_capture_bundle_readiness.py tests/test_telemetry_contract_gate.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_accepts_local_bundle tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_holds_template_placeholders tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51v_forward_capture_bundle_readiness_rejects_unsafe_manifest
```

Result:

```text
Ran 3 tests
OK
```

Artifact hashes:

```text
53ffe4d3d0d943596fd6757f73529e9753a08614a599e2061963cf6bb942ae58  phase51v_forward_capture_bundle_readiness_summary.json
b7f5be276dcc7820a7a4e81404b47a311034e29b3db833707df977767f8ab1fc  phase51v_manifest.json
70db5f85fe73eb7193fca273732df6c43be34e0d9f0d0180a6d3f39faa4c44b1  phase51s_manifest.generated.json
```

Verdict:

```text
HOLD for live, canary, model training, EV admission, capital escalation,
risk-limit relaxation, financial claims, and 24/7 readiness.

PROMOTE only for acquiring or producing a sanitized local read-only all-five
forward capture bundle and rerunning Phase 5.1v before Phase 5.1s.
```

## Phase 5.1w Forward Capture Request Pack

Run id: `PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z`

Local run directory:
`runs/phase51w_forward_capture_request_pack/PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z`

Input target run:
`runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z`

Result:

```text
gate_status: HOLD
gate_reason: phase51w_forward_capture_request_pack_emitted_nonlive_hold
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
required_local_source_file_count: 6
clears_phase51_blockers: false
raw_identifier_redaction_status: PASS
```

Command:

```bash
python3 tools/phase51w_forward_capture_request_pack.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --output-root runs/phase51w_forward_capture_request_pack \
  --run-id PHASE51W-FORWARD-CAPTURE-REQUEST-PACK-CANONICAL-20260504T000000Z \
  --timestamp-ns 1777852800000000000
```

Validation:

```bash
python3 -m py_compile tools/phase51w_forward_capture_request_pack.py tests/test_telemetry_contract_gate.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51w_forward_capture_request_pack_emits_operator_pack tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51w_forward_capture_request_pack_rejects_unsafe_targets
```

Result:

```text
Ran 2 tests
OK
```

Artifact hashes:

```text
3b0b2dcbdec00dbfe8ec2d3f156952213a499876ab1448f92fdfb0f1775844f8  phase51w_forward_capture_request_pack_summary.json
b3f37c0cc0772e805d099c411de33b949a800c365b1e12fa8e493bb97c8cdfbe  forward_capture_request_pack.json
93899bec8921a3539152b95ca119de2a0bcca61cb262e06b89acc04ff11c2aec  forward_capture_request_pack.md
bcec3fad3a78e21ca40c35d52425fe72da3393e93605da0b98cfc48d1e8c2886  capture_bundle_manifest.skeleton.json
```

Verdict:

```text
HOLD for live, canary, model training, EV admission, capital escalation,
risk-limit relaxation, financial claims, and 24/7 readiness.

PROMOTE only for using the generated request pack to provide six sanitized
local read-only source files, then rerunning Phase 5.1v.
```

## Phase 5.1w Staged Local Source Bundle

Run id:

`PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z`

Local run directory:

`runs/phase51w_forward_capture_request_pack/PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z`

Input target run:

`runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z`

Result:

```text
gate_status: HOLD
gate_reason: phase51w_forward_capture_request_pack_emitted_nonlive_hold
native_role_capture_target_count: 287
native_role_capture_target_counts_by_venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- lighter: 125
- paradex: 15
lighter_native_limit_capture_target_count: 3132
required_local_source_file_count: 6
local_source_staging_enabled: true
source_file_count: 6
clears_phase51_blockers: false
no_live_flag: true
approved_for_live: false
live_orders_allowed: false
```

Staged files:

```text
aster_forward_native_role_snapshot.jsonl 0
extended_forward_native_role_snapshot.jsonl 0
hyperliquid_forward_native_role_snapshot.jsonl 0
lighter_forward_native_limit_pressure_snapshot.jsonl 0
lighter_forward_native_role_snapshot.jsonl 0
paradex_forward_native_role_snapshot.jsonl 0
```

Command:

```bash
python3 tools/phase51w_forward_capture_request_pack.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --output-root runs/phase51w_forward_capture_request_pack \
  --run-id PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z \
  --timestamp-ns 1777894485526000000 \
  --stage-local-source-dir local_source_staging
```

Phase 5.1v validation:

```text
run_id: PHASE51V-EMPTY-STAGED-SOURCE-BUNDLE-HOLD-20260504T000000Z
local_run_directory: runs/phase51v_forward_capture_bundle_readiness/PHASE51V-EMPTY-STAGED-SOURCE-BUNDLE-HOLD-20260504T000000Z
gate_status: HOLD
gate_reason: phase51v_forward_capture_bundle_incomplete_nonlive_hold
source_file_status_counts: {"LOCAL_FILE_READY": 6}
native_role_capture_target_ready_count: 0
native_role_capture_target_missing_count: 287
lighter_native_limit_capture_target_ready_count: 0
lighter_native_limit_capture_target_missing_count: 3132
generated_phase51s_manifest_ready: false
clears_phase51_blockers: false
```

Phase 5.1v command:

```bash
python3 tools/phase51v_forward_capture_bundle_readiness.py \
  --target-run runs/phase51u_forward_capture_target_manifest/PHASE51U-FORWARD-CAPTURE-TARGET-MANIFEST-CANONICAL-PFILL-20260503T000000Z \
  --candidate-manifest runs/phase51w_forward_capture_request_pack/PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z/local_capture_bundle_manifest.json \
  --output-root runs/phase51v_forward_capture_bundle_readiness \
  --run-id PHASE51V-EMPTY-STAGED-SOURCE-BUNDLE-HOLD-20260504T000000Z \
  --timestamp-ns 1777894485526000000
```

Verdict:

```text
HOLD. The local staging contract is now repo-owned and Phase 5.1v-readable.
This does not clear all-five Phase 5.1w because the staged files are empty
templates. Populate them with sanitized native rows before rerunning 5.1v.
```

## Phase 5.1w Read-Only Lighter Private Source Capture Attempt

Run ids:

- `PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z`
- `PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z`

Local run directories:

- `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z`
- `runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z`

Scope:

```text
Authorized read-only private source capture attempt using existing local
credentials only. No live orders, no canary/live mode, no capital escalation,
no risk-limit relaxation, and no secret values printed. The run used
repo-owned Lighter GET-only collectors and emitted sanitized HOLD artifacts.
```

Board audit summary:

```text
Safe existing private collectors are Lighter-only.
Aster, Extended, Hyperliquid, and Paradex still require safe repo-owned
read-only native-role collectors/parsers before private source capture.
The Lighter capture does not clear all-five Phase 5.1w.
```

Result:

```text
phase51b_capture_complete: true
phase51b_event_count: 6
phase51b_source_names:
- account
- account_limits
- active_orders
- official_limits
- order_books
- trades
phase51c_source_mode: readonly_lighter_api
phase51c_pages_fetched: 3
phase51c_trade_count: 300
phase51c_role_counts_for_account:
- maker: 189
- taker: 111
- unknown: 0
gate_status: HOLD
approved_for_live: false
approved_for_canary: false
approved_for_capital_escalation: false
admissible_for_financial_claim: false
clears_all_five_phase51w_blocker: false
```

Commands:

```bash
python3 tools/phase51b_lighter_account_limits.py \
  --fetch-readonly \
  --include-trades \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --allow-sdk-auth \
  --timeout-s 20

/opt/paraphina/.venv_lighter/bin/python3 tools/phase51b_lighter_account_limits.py \
  --fetch-readonly \
  --include-trades \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --allow-sdk-auth \
  --timeout-s 20

/opt/paraphina/.venv_lighter/bin/python3 tools/phase51c_lighter_trade_backfill.py \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --allow-sdk-auth \
  --market-id 0 \
  --market-type perp \
  --pages 3 \
  --limit 100 \
  --sleep-s 1.6
```

Validation:

```bash
python3 -m py_compile tools/phase51b_lighter_account_limits.py tools/phase51c_lighter_trade_backfill.py
python3 tools/check_telemetry_contract.py runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z/telemetry.jsonl
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51b_lighter_account_limits_collector_emits_valid_hold_artifact tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51c_lighter_trade_backfill_ingests_offline_pages_without_promotion
rg -c --pcre2 "0x[0-9a-fA-F]{64}|(?i)bearer\\s+[A-Za-z0-9._-]{20,}|(?i)api[_-]?secret\\s*[=:]|(?i)private[_-]?key\\s*[=:]" runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z
```

Validation output:

```text
Initial python3 collector attempt: ERROR lighter-sdk is required for --allow-sdk-auth
Initial /opt/paraphina/.venv_lighter collector attempt: ERROR Unsupported platform/architecture: Linux/aarch64
py_compile: PASS
check_telemetry_contract: OK: 6 record(s) validated against schema v2
unittest: Ran 2 tests; OK
secret-value-pattern scan: no matches
```

Artifact hashes:

```text
cbc1b8dc4157185ece2841786da1ccc575a0232af421d920bc610bc0547cc124  runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z/lighter_account_native_limits_summary.json
0802e153fceea31cfbba7b0423fcbccf8f3834b559d0d783dca921b1b08dea88  runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z/telemetry.jsonl
3a8afee51b56c71a692cab4cf3b55ab43c4bfa4bf29aa158a9f97988d39bc901  runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS_20260504T110402Z/manifest.json
ef0ddede64a08231edf46d8b64d52706752ac58fdd8eda54fb60dee6c781cc2d  runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z/lighter_trade_backfill_summary.json
1ec3285f3cf8e4c21e7dfdd96e2cd98eac7c0aafa57e1fbcce96c97d3ef090b4  runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z/source_snapshots/trades_backfill.sanitized.json
15629f8730ea12e603d423276fafda678d9b44640e85361f985b278d0292dc64  runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-20260504T110416Z/manifest.json
```

Verdict:

```text
HOLD for all-five Phase 5.1w completion, live, canary, model training,
EV admission, capital escalation, risk-limit relaxation, financial claims,
and 24/7 readiness.

PROMOTE only for Lighter-only sanitized source availability. The next single
move is to implement safe repo-owned read-only native-role collectors/parsers
for Aster, Extended, Hyperliquid, and Paradex, and complete Lighter
event-time native-limit pressure linkage before rerunning Phase 5.1v.
```

## Phase 5.1t - Source-Link Sidecar Builder

- Run id:
  `PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z`
- Local run directory:
  `runs/phase51t_source_link_sidecar_builder/PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z`
- Scope: existing local Lighter read-only source snapshots and trade backfills.
- Gate status: `HOLD`
- Gate reason: `phase51t_source_link_sidecar_builder_complete_nonlive_hold`
- Source files scanned: `31`
- Source rows scanned: `1522`
- Redacted source-link rows emitted: `363`
- Ambiguous identity-hash target count: `48`
- `clears_phase51_blockers`: `false`
- Safety flags: all live, canary, capital, risk-relaxation, EV-admission,
  model-training, and financial-claim authorization fields are `false`.

Source-link status counts:

```text
SOURCE_LINK_EMITTED: 363
DUPLICATE_SOURCE_HASH_ALREADY_EMITTED: 546
NO_OBSERVED_IDENTITY_MATCH: 570
NO_ORDER_IDENTITY_HASH: 17
AMBIGUOUS_OBSERVED_IDENTITY_MATCH: 26
```

Downstream rerun chain:

```text
5.1s run:
runs/phase51s_local_native_source_acquisition/PHASE51T-TO-S-EXISTING-LIGHTER-SOURCES-20260503T000000Z
staged_source_row_count: 1522
staged_source_link_row_count: 363

5.1r run:
runs/phase51r_forward_native_source_acquisition/PHASE51T-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z
source_link_applied_count: 909
native_role_source_record_count: 296
native_role_target_recovered_count: 0 / 287
lighter_native_limit_target_recovered_count: 0 / 3132

5.1q run:
runs/phase51q_forward_native_evidence/PHASE51T-FORWARD-NATIVE-EVIDENCE-EXISTING-LIGHTER-SOURCES-20260503T000000Z
recovered_forward_native_role_count: 0

5.1n run:
runs/phase51n_maker_taker_attribution_recovery/PHASE51T-MAKER-TAKER-ATTRIBUTION-RECOVERY-EXISTING-LIGHTER-SOURCES-20260503T000000Z
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287

5.1h run:
runs/phase51h_observed_pfill_feature_audit/PHASE51T-OBSERVED-PFILL-FEATURE-AUDIT-EXISTING-LIGHTER-SOURCES-20260503T000000Z
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65

5.1i run:
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51T-PFILL-FEATURE-MATRIX-ADMISSIBILITY-EXISTING-LIGHTER-SOURCES-20260503T000000Z
gate_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_count: 4
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Commands:

```bash
python3 tools/phase51t_source_link_sidecar_builder.py \
  --observed-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --source-root runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T001126Z/source_snapshots \
  --source-root runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002006Z/source_snapshots \
  --source-root runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/source_snapshots \
  --source-root runs/phase51b_lighter_account_native_limits/PHASE51M-LIGHTER-OFFICIAL-LIMIT-ENRICHMENT-20260503T000000Z/source_snapshots \
  --source-root runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-COMBINED-TERMINAL-STALE-7200S-20260429T073231Z/source_snapshots \
  --source-root runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S-20260429T073231Z/source_snapshots \
  --source-root runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-OFFLINE-EXISTING-20260502T1135Z/source_snapshots \
  --source-root runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-7200S-20260429T025435Z/source_snapshots \
  --source-root runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-7200S-20260429T073231Z/source_snapshots \
  --output-root runs/phase51t_source_link_sidecar_builder \
  --run-id PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z \
  --timestamp-ns 1777852800000000000

python3 tools/phase51s_local_native_source_acquisition.py \
  --manifest runs/phase51t_source_link_sidecar_builder/PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-EXISTING-LIGHTER-SOURCES-20260503T000000Z/phase51s_existing_lighter_sources_manifest.json \
  --output-root runs/phase51s_local_native_source_acquisition \
  --run-id PHASE51T-TO-S-EXISTING-LIGHTER-SOURCES-20260503T000000Z \
  --timestamp-ns 1777852800000000000

python3 tools/phase51r_forward_native_source_acquisition.py \
  --observed-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --source-json runs/phase51s_local_native_source_acquisition/PHASE51T-TO-S-EXISTING-LIGHTER-SOURCES-20260503T000000Z/local_native_source.jsonl \
  --source-link-jsonl runs/phase51s_local_native_source_acquisition/PHASE51T-TO-S-EXISTING-LIGHTER-SOURCES-20260503T000000Z/local_source_link_sidecar.jsonl \
  --output-root runs/phase51r_forward_native_source_acquisition \
  --run-id PHASE51T-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z \
  --timestamp-ns 1777852800000000000

python3 tools/phase51q_forward_native_evidence_capture.py \
  --observed-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --native-role-jsonl runs/phase51r_forward_native_source_acquisition/PHASE51T-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z/native_role_source.jsonl \
  --native-limit-jsonl runs/phase51r_forward_native_source_acquisition/PHASE51T-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z/native_limit_source.jsonl \
  --output-root runs/phase51q_forward_native_evidence \
  --run-id PHASE51T-FORWARD-NATIVE-EVIDENCE-EXISTING-LIGHTER-SOURCES-20260503T000000Z \
  --timestamp-ns 1777852800000000000
```

Artifact hashes:

```text
c7046d5a763eddd06faded62bb9f461ea9918f619492b84da20e4ca85047bec8  phase51t_source_link_sidecar_builder_summary.json
018f2fa0671de3496853788c037438336eaed9288d09b25fcf16d1a7802e7125  source_links.sanitized.jsonl
83da939f05757b19ab81658d08632a1af37ff994f05ca98c7bbc52bea16584ec  phase51s_local_native_source_acquisition_summary.json
5cd5f04896bd74d3afb05ea75928a3d35ecd797cac30016b29aa63bd5d720e7f  phase51r_forward_native_source_acquisition_summary.json
6521fa74d37903981c322dd60f28345dfe732314bed8dd762e9e5305d18a54f2  phase51q_forward_native_evidence_summary.json
56de16b4384a2753e022658e84c1510c59f994ddd911599691b9155918260bdc  maker_taker_attribution_recovery_summary.json
c73f3ff5fc428578541cd4369b26e6b8db923ca2ef32eb958fabf13288902c65  pfill_feature_audit_summary.json
a445a340aed5b695216dabcadd8cedd0dccec7842ec6d896e0c7886064cbedbd  pfill_feature_matrix_admissibility_summary.json
```

Interpretation: Phase 5.1t is a useful and safe source-link plumbing gate, but
existing local artifacts do not clear the current blockers. The next evidence
move remains fresh forward read-only native source capture with canonical
linkage or 5.1t-compatible sidecars across all five venues, plus complete
Lighter event-time active-order/sendTx/REST-or-weighted-request pressure.

## Phase 5.1m - Lighter Official-Doc Native-Limit Enrichment

Date: 2026-05-03

Purpose: add official Lighter active-order cap evidence and verify whether it
can clear the Lighter native-limit feature-completeness blocker without using
live orders, canary authority, model training, EV admission, or financial
claims.

Evidence packs:

```text
runs/phase51b_lighter_account_native_limits/PHASE51M-LIGHTER-OFFICIAL-LIMIT-ENRICHMENT-20260503T000000Z
runs/phase51c_queue_churn/PHASE51M-QUEUE-CHURN-NATIVE-DOC-CAP-TERMINAL-STALE-7200S-20260429T025435Z
runs/phase51c_queue_churn/PHASE51M-QUEUE-CHURN-NATIVE-DOC-CAP-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
runs/phase51h_observed_pfill_feature_audit/PHASE51M-OBSERVED-PFILL-FEATURE-AUDIT-NATIVE-DOC-CAP-TWO-LANE-20260503T000000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51M-PFILL-FEATURE-MATRIX-ADMISSIBILITY-NATIVE-DOC-CAP-TWO-LANE-20260503T000000Z
```

Official Lighter sources used:

```text
https://apidocs.lighter.xyz/docs/rate-limits
https://apidocs.lighter.xyz/docs/account-types
https://apidocs.lighter.xyz/reference/accountlimits
https://apidocs.lighter.xyz/reference/accountactiveorders
https://apidocs.lighter.xyz/reference/trades
```

Result:

```text
matrix_admissibility_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_count: 4
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved

label_count: 4527
filled_count: 461
native_limit_observed_count: 0
native_limit_partial_count: 2288
native_limit_unknown_count: 0
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
observed_horizon_missing_count: 0
filled_horizon_source_key_unrecovered_count: 0
raw_identifier_redaction_status: PASS
missing_feature_total: 2575
```

Safety boundary:

```text
approved_for_live: false
approved_for_canary: false
approved_for_capital_escalation: false
approved_for_financial_claim: false
admissible_for_ev_admission: false
official_doc_cap_not_event_time_usage: true
limitations:
- lighter_sendtx_remaining_not_observed
- lighter_rest_request_limit_not_exposed_by_account_limits_payload
```

Interpretation: official Lighter active-order caps plus observed current
active-order counts are useful capacity evidence, but they are not sufficient
to label historical Phase 5 rows as event-time native-limit pressure observed.
The next non-live gate is event-time native-limit pressure and venue-native
maker/taker completion across all filled venues.

## Phase 5.1n - Event-Time Native-Limit and Maker/Taker Evidence Gate

Date: 2026-05-03

Purpose: align Lighter historical account snapshot logs to Phase 5 label event
times, add a repo-owned all-venue maker/taker attribution recovery gate, and
rerun 5.1h/5.1i without inferring native-limit pressure or maker/taker role.

Evidence packs:

```text
runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-20260429T025435Z
runs/phase51n_lighter_native_limit_time_alignment/PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51N-MAKER-TAKER-ATTRIBUTION-RECOVERY-OBSERVED-ONLY-TWO-LANE-20260503T000000Z
runs/phase51c_queue_churn/PHASE51N-QUEUE-CHURN-EVENT-TIME-NATIVE-LIMIT-TERMINAL-STALE-7200S-20260429T025435Z
runs/phase51c_queue_churn/PHASE51N-QUEUE-CHURN-EVENT-TIME-NATIVE-LIMIT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
runs/phase51h_observed_pfill_feature_audit/PHASE51N-OBSERVED-PFILL-FEATURE-AUDIT-EVENT-TIME-NATIVE-LIMIT-TWO-LANE-20260503T000000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51N-PFILL-FEATURE-MATRIX-ADMISSIBILITY-EVENT-TIME-NATIVE-LIMIT-TWO-LANE-20260503T000000Z
```

Result:

```text
matrix_admissibility_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_count: 4
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved

label_count: 4527
filled_count: 461
native_limit_observed_count: 0
native_limit_partial_count: 2288
native_limit_unknown_count: 0
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Phase 5.1n native-limit detail:

```text
025435 lane:
- lighter labels: 2194
- event-time aligned active-order snapshots: 1728
- stale Lighter snapshots: 466
- all pressure dimensions observed: 0

073231 lane:
- lighter labels: 3954
- event-time aligned active-order snapshots: 3700
- stale Lighter snapshots: 254
- all pressure dimensions observed: 0

Known limitation:
- lighter_sendtx_remaining_not_observed
- lighter_rest_request_remaining_not_observed
```

Phase 5.1n maker/taker detail:

```text
filled rows: 461
input role counts already complete: 174
missing venue-native role source: 287
native role evidence supplied for recovery: 0
```

Interpretation: Phase 5.1n improves the forensic boundary but does not clear
the matrix. Historical Lighter active-order pressure is now partially
event-time aligned, but sendTx/REST pressure was not captured historically.
The all-venue maker/taker gate confirms that the remaining `287` filled rows
need venue-native fill/trade role evidence; quote intent, post-only behavior,
strategy purpose, and fee expectations remain inadmissible as role inference.

## Phase 5.1o - Native Role Source Inventory and Recovered Matrix Rerun

Date: 2026-05-03

Purpose: inventory existing all-venue venue-native maker/taker source material,
emit only exact canonical native-role evidence for 5.1n recovery, and rerun
5.1h/5.1i without inferring roles from intent, post-only behavior, strategy
purpose, price position, or fee schedule.

Evidence packs:

```text
runs/phase51o_native_role_source_inventory/PHASE51O-NATIVE-ROLE-SOURCE-INVENTORY-ALL-VENUE-20260503T120000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51O-MAKER-TAKER-ATTRIBUTION-RECOVERY-ALL-VENUE-20260503T120000Z
runs/phase51h_observed_pfill_feature_audit/PHASE51O-OBSERVED-PFILL-FEATURE-AUDIT-NATIVE-ROLE-RECOVERY-20260503T120000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51O-PFILL-FEATURE-MATRIX-ADMISSIBILITY-NATIVE-ROLE-RECOVERY-20260503T120000Z
```

Native-role source inventory result:

```text
gate_status: HOLD
gate_reason: phase51o_native_role_sources_incomplete
label_count: 4527
filled_count: 461
input_observed_preserved_count: 174
recovered_native_role_count: 0
native_role_evidence_record_count: 0
missing_native_role_source_count: 162
source_available_no_canonical_join_count: 125
raw_identifier_redaction_status: PASS

source_available_no_canonical_join by venue:
- lighter: 125

missing_native_role_source by venue:
- aster: 113
- extended: 28
- hyperliquid: 6
- paradex: 15
```

Recovered matrix result:

```text
matrix_admissibility_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_count: 4
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved

label_count: 4527
filled_count: 461
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Artifact hashes:

```text
2e785de0b686b1d1f7858a923ec9fb88fd33c9ceb5cfe17c966c4c926726ecb4  native_role_source_inventory_summary.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  native_role_evidence.jsonl
9c513f5565687d407d693660eeeee72c9171d80241f04f68a6be8b47d3393dce  maker_taker_attribution_recovery_summary.json
be19e44b7c14304d4ac5696de97534a54fdad23e18b87adadbff67fc98f5a54c  pfill_feature_audit_summary.json
5f488d288b6019d0c9a83d2e8122e1835c2f7d5e019eb962ee55bb8bb4744cdf  pfill_feature_matrix_admissibility_summary.json
```

Interpretation: Phase 5.1o proves that current retained artifacts are
insufficient for exact canonical maker/taker role recovery. Lighter has `125`
filled rows with source material available but not canonically joined to the
observed P_fill labels; Aster, Extended, Hyperliquid, and Paradex have `162`
filled rows with no retained venue-native role source in current artifacts. The
correct next move is forward canonical capture plus a narrowly scoped Lighter
backfill where raw identifier handling can remain quarantined; the matrix must
not be cleared by inference.

## Phase 5.1p - Lighter Native Role Canonical Join Attempt

Date: 2026-05-03

Purpose: run the narrow quarantined Lighter-only native trade join authorized by
Phase 5.1o. The tool may read raw Lighter side IDs inside the local process,
but board-facing outputs contain only hashes, canonical group IDs, counts, and
source classifications.

Evidence packs:

```text
runs/phase51p_lighter_native_role_canonical_join/PHASE51P-LIGHTER-NATIVE-ROLE-CANONICAL-JOIN-ALL-BACKFILLS-20260503T140000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51P-MAKER-TAKER-ATTRIBUTION-RECOVERY-LIGHTER-NATIVE-20260503T140000Z
runs/phase51h_observed_pfill_feature_audit/PHASE51P-OBSERVED-PFILL-FEATURE-AUDIT-LIGHTER-NATIVE-20260503T140000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51P-PFILL-FEATURE-MATRIX-ADMISSIBILITY-LIGHTER-NATIVE-20260503T140000Z
```

Lighter canonical join result:

```text
gate_status: HOLD
gate_reason: phase51p_lighter_native_role_join_incomplete
label_count: 4527
filled_count: 461
lighter_source_available_target_count: 125
recovered_lighter_native_role_count: 0
unrecovered_lighter_native_role_count: 125
native_role_evidence_record_count: 0
raw_identifier_redaction_status: PASS

lighter_native_role_join_status_counts:
- NATIVE_ID_HASH_NO_MATCH: 125
- NOT_TARGETED: 4402
```

Native Lighter trade backfill coverage used by the join:

```text
unique native trades indexed with role: 531
native side identity hash index entries: 2124
native role counts across unique indexed rows:
- MAKER: 345
- TAKER: 186
- UNKNOWN: 0
```

Recovered matrix result:

```text
matrix_admissibility_status: HOLD
gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
matrix_blocker_count: 4
matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved

label_count: 4527
filled_count: 461
native_limit_observed_count: 0
native_limit_partial_count: 2288
maker_taker_observed_count: 174
maker_taker_partial_or_unknown_count: 222
maker_taker_missing_count: 65
raw_identifier_redaction_status: PASS
```

Artifact hashes:

```text
17c4a3405967a4c6cf9c9005e9448f0a09acdc0e50a7bdea57ae187a6f8dede9  lighter_native_role_canonical_join_summary.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  lighter_native_role_evidence.jsonl
14ea7d5b58ed4b860ed208645a7057780f7fb4832ca616b3905af33c78860718  maker_taker_attribution_recovery_summary.json
a0fe91ccf13d2fd4101b4cc3df581b4e90ce339dd6215bd8f6d0c37d0b494ea5  pfill_feature_audit_summary.json
a65bab824b031b148af527b0765293e0611ab7247a564b4a7f48b76d77703710  pfill_feature_matrix_admissibility_summary.json
```

Interpretation: Phase 5.1p closes the safe historical Lighter backfill attempt
without clearing the maker/taker blocker. The retained Lighter native trades do
carry explicit maker/taker truth, but their native side IDs do not exactly match
the canonical source order/client hashes for the `125` source-available filled
rows. The next maker/taker evidence path is therefore forward capture of native
role fields/IDs at order/fill time across all five venues, not inference from
post-only intent or historical fee/account assumptions.

## Phase 5.1q - Forward Native Evidence Gate Baseline

Date: 2026-05-03

Purpose: formalize the forward native-evidence gate and run it against the
latest real canonical P_fill artifact with no forward native source rows
provided. This establishes the replayable HOLD boundary and proves the gate
does not infer maker/taker role or native-limit pressure.

Evidence packs:

```text
runs/phase51q_forward_native_evidence/PHASE51Q-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51Q-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z
```

Phase 5.1q result:

```text
gate_status: HOLD
gate_reason: phase51q_forward_native_evidence_incomplete
observed_pfill_label_count: 6140
native_role_evidence_record_count: 0
recovered_forward_native_role_count: 0
native_role_capture_status_counts:
- OBSERVED_PRESERVED: 174
- MISSING_FORWARD_NATIVE_ROLE_SOURCE: 287
- NO_FILL_NOT_APPLICABLE: 5679
native_limit_pressure_status_counts:
- MISSING_NATIVE_LIMIT_PRESSURE_SOURCE: 3132
- NOT_APPLICABLE_NON_LIGHTER: 3008
raw_identifier_redaction_status: PASS
```

Maker/taker recovery rerun from Phase 5.1q output:

```text
gate_status: HOLD
gate_reason: phase51n_maker_taker_attribution_incomplete
filled_count: 461
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
native_role_inputs: 0 records from Phase 5.1q
raw_identifier_redaction_status: PASS
```

Interpretation: Phase 5.1q is now repo-owned and runnable, but no blockers are
cleared until real forward-captured native source rows are provided. The next
evidence move is to capture explicit venue-native maker/taker role fields for
all five venues and Lighter event-time active-order, sendTx, and REST/weighted
request pressure, then rerun Phase 5.1q -> 5.1n -> 5.1h -> 5.1i.

## Phase 5.1r - Forward Native Source Acquisition Baseline

Date: 2026-05-03

Purpose: add a redaction-safe acquisition layer that can ingest local,
read-only venue-native snapshots and emit the sanitized Phase 5.1q source rows
without leaking raw venue identifiers or creating any live/training authority.

Evidence packs:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51R-FORWARD-NATIVE-SOURCE-ACQUISITION-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51q_forward_native_evidence/PHASE51R-FORWARD-NATIVE-EVIDENCE-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51n_maker_taker_attribution_recovery/PHASE51R-MAKER-TAKER-ATTRIBUTION-RECOVERY-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51h_observed_pfill_feature_audit/PHASE51R-OBSERVED-PFILL-FEATURE-AUDIT-BASELINE-NO-SOURCES-20260503T000000Z
runs/phase51i_pfill_feature_matrix_admissibility/PHASE51R-PFILL-FEATURE-MATRIX-ADMISSIBILITY-BASELINE-NO-SOURCES-20260503T000000Z
```

Phase 5.1r result:

```text
gate_status: HOLD
gate_reason: phase51r_forward_native_source_acquisition_incomplete
observed_pfill_label_count: 6140
source_file_count: 0
source_row_count: 0
native_role_target_count: 287
native_role_source_record_count: 0
native_role_target_recovered_count: 0
lighter_native_limit_target_count: 3132
native_limit_source_record_count: 0
native_limit_complete_source_record_count: 0
lighter_native_limit_target_recovered_count: 0
raw_identifier_redaction_status: PASS
```

Downstream Phase 5.1q result:

```text
gate_status: HOLD
gate_reason: phase51q_forward_native_evidence_incomplete
native_role_evidence_record_count: 0
recovered_forward_native_role_count: 0
native_role_capture_status_counts:
- OBSERVED_PRESERVED: 174
- MISSING_FORWARD_NATIVE_ROLE_SOURCE: 287
- NO_FILL_NOT_APPLICABLE: 5679
native_limit_pressure_status_counts:
- MISSING_NATIVE_LIMIT_PRESSURE_SOURCE: 3132
- NOT_APPLICABLE_NON_LIGHTER: 3008
raw_identifier_redaction_status: PASS
```

Downstream Phase 5.1n result:

```text
gate_status: HOLD
gate_reason: phase51n_maker_taker_attribution_incomplete
filled_count: 461
maker_taker_observed_or_recovered_count: 174
maker_taker_partial_or_missing_count: 287
native_role_inputs: 0 records from Phase 5.1q
raw_identifier_redaction_status: PASS
```

Recovered matrix result:

```text
5.1h gate_reason: phase51h_lighter_native_limit_pressure_not_fully_observed
5.1i gate_reason: phase51i_lighter_native_limit_pressure_not_fully_observed
5.1i matrix_blocker_ids:
- lighter_native_limit_pressure_not_fully_observed
- maker_taker_not_fully_observed_for_filled_orders
- sparse_pfill_feature_buckets
- observed_only_selection_bias_not_resolved
```

Interpretation: Phase 5.1r is now repo-owned and runnable. The no-source
baseline intentionally clears no blocker; it proves the adapter preserves the
HOLD boundary until real read-only native snapshots are supplied.

## Phase 5.1s - Local Native Source Acquisition

Date: 2026-05-03

Purpose: add a manifest-driven local source staging gate in front of Phase 5.1r
so read-only native snapshots are acquired through an auditable, redaction-safe
path instead of ad hoc source-root scans.

Repo-owned gate:

```text
tool: tools/phase51s_local_native_source_acquisition.py
spec: docs/PHASE5_1S_LOCAL_NATIVE_SOURCE_ACQUISITION.md
example manifest: configs/phase51s_local_native_source_manifest.example.json
status: HOLD
```

First local-source staging evidence:

```text
runs/phase51s_local_native_source_acquisition/PHASE51S-LOCAL-NATIVE-SOURCE-EXISTING-LIGHTER-SOURCES-20260503T000000Z
```

Result:

```text
gate_status: HOLD
gate_reason: phase51s_local_native_source_acquisition_complete_nonlive_hold
source_file_count: 7
source_row_count: 405
staged_source_row_count: 405
join_key_source_row_count: 0
source_row_without_join_key_count: 405
complete_lighter_native_limit_source_row_count: 0
raw_identifier_fields_stripped_count: 3500
raw_identifier_redaction_status: PASS
clears_phase51_blockers: false
```

Downstream Phase 5.1r rerun from the Phase 5.1s staged source:

```text
runs/phase51r_forward_native_source_acquisition/PHASE51S-TO-R-EXISTING-LIGHTER-SOURCES-20260503T000000Z
```

Downstream result:

```text
gate_status: HOLD
gate_reason: phase51r_forward_native_source_acquisition_incomplete
source_row_count: 405
native_source_acquisition_status_counts:
- UNJOINED_NO_CANONICAL_GROUP: 405
native_role_target_count: 287
native_role_source_record_count: 0
native_role_target_recovered_count: 0
lighter_native_limit_target_count: 3132
native_limit_source_record_count: 0
native_limit_complete_source_record_count: 0
lighter_native_limit_target_recovered_count: 0
raw_identifier_redaction_status: PASS
```

Interpretation: Phase 5.1s successfully stages existing local Lighter source
snapshots and strips raw identifiers, but those historical snapshots do not
contain canonical join keys or complete event-time limit-pressure rows.
Therefore no Phase 5.1 blocker is cleared. The next evidence move remains
capturing forward native source rows with canonical group or order-key linkage
at decision/fill time, plus complete Lighter event-time active-order, sendTx,
and REST/weighted-request pressure context.

## Phase 5.1r - Source-Link Sidecar Gate

Date: 2026-05-03

Purpose: add a validated non-live source-link sidecar path so Phase 5.1r can
join future redacted source rows by deterministic source hash when direct
`canonical_group_id` or `order_key` fields are not present in the staged source
row.

Repo-owned gate:

```text
tool: tools/phase51r_forward_native_source_acquisition.py
input: repeated --source-link-jsonl
status: HOLD
```

Validation evidence:

```text
python3 -m py_compile tools/phase51r_forward_native_source_acquisition.py tests/test_telemetry_contract_gate.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_link_sidecar_recovers_joinable_staged_rows tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_link_sidecar_rejects_ambiguous_or_raw_links
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_acquisition_feeds_phase51r_without_raw_ids tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_acquisition_does_not_false_clear_partial_sources tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_acquisition_feeds_phase51q_without_raw_ids tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_acquisition_does_not_false_clear_partial_or_inferred_sources tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51q_forward_native_evidence_feeds_all_five_venues_without_raw_ids
```

Expected sidecar behavior:

```text
accepted: source hash -> observed canonical_group_id/order_key
rejected: duplicate source hashes
rejected: conflicting group/order mappings
rejected: raw order/client/fill/trade identifiers
rejected: unsafe true authorization flags
not allowed: maker/taker inference, Lighter limit-pressure inference, EV/PnL claims
```

Interpretation: the sidecar gate improves source-acquisition plumbing and
resume safety, but no live or economic blocker is cleared until real forward
source snapshots and source-link rows produce observed venue-native role and
Lighter native-limit evidence.

## Phase 5.1s - Source-Link Sidecar Staging

Date: 2026-05-03

Purpose: make Phase 5.1s the mandatory local preflight for both redacted native
source rows and redacted source-link sidecars before Phase 5.1r consumes them.
This keeps forward captures manifest-bound, local-only, raw-ID checked, and
resume-safe.

Repo-owned gate:

```text
tool: tools/phase51s_local_native_source_acquisition.py
manifest input: optional source_links list
output: local_source_link_sidecar.jsonl
status: HOLD
```

Validation evidence:

```text
python3 -m py_compile tools/phase51s_local_native_source_acquisition.py tests/test_telemetry_contract_gate.py
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_link_sidecar_feeds_phase51r_deterministically tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_source_link_sidecar_rejects_unsafe_rows
python3 -m unittest tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_acquisition_feeds_phase51r_without_raw_ids tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_acquisition_rejects_secrets_and_network_sources tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51s_local_source_acquisition_does_not_false_clear_partial_sources tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_link_sidecar_recovers_joinable_staged_rows tests.test_telemetry_contract_gate.TestValidatorSubprocess.test_phase51r_source_link_sidecar_rejects_ambiguous_or_raw_links
```

Expected staging behavior:

```text
accepted: source hash -> observed canonical_group_id/order_key
rejected: network paths, .env files, symlinks
rejected: secret-shaped fields
rejected: raw order/client/fill/trade identifiers
rejected: duplicate source hashes
rejected: non-string source-hash or join fields
rejected: unsupported sidecar fields
rejected: unsafe true authorization flags
sidecar-only run: incomplete_source_links_only
not allowed: maker/taker inference, Lighter limit-pressure inference, EV/PnL claims
```

Interpretation: Phase 5.1s now stages both source rows and optional source-link
sidecars, but it still clears no blocker by itself. Blocker reduction remains
downstream-only through Phase 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i with real
forward venue-native evidence.

## Phase 5.1j - Observed-Horizon Recovery and Recovered Matrix

- Recovery run id:
  `PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovery run directory:
  `runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered 5.1h run:
  `runs/phase51j_recovered_observed_pfill_feature_audit/PHASE51J-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered 5.1i run:
  `runs/phase51j_pfill_feature_matrix_admissibility/PHASE51J-PFILL-FEATURE-MATRIX-ADMISSIBILITY-RECOVERED-TWO-LANE-20260502T000000Z`
- Baseline commit: `18dd09512288a85e440d3977e32432c3aabc1190`
- Gate status: `HOLD`
- Recovery gate reason:
  `phase51j_observed_horizon_recovery_partial_horizon_missing`
- Matrix gate reason: `phase51i_missing_observed_horizon_features`
- Raw identifier redaction status: `PASS`
- Matrix labels: `4527`
- Input observed horizon available / missing: `18` / `4509`
- Recovered terminal not-filled horizons: `4048`
- Preserved existing horizons: `18`
- Remaining fill-time/source-time horizons missing: `461`
- Recovered observed horizon available / missing: `4066` / `461`
- Recovered 5.1h horizon recovery applied count: `4048`
- Lighter native-limit observed / partial remains: `0` / `2288`
- Filled-order maker/taker observed / partial-unknown / missing remains:
  `174` / `222` / `65`
- Excluded quarantine/review groups remain: `1613`
- Safety flags: all live, canary, capital, risk-relaxation, EV-admission,
  model-training, and financial-claim authorization fields are `false`.

Commands:

```bash
python3 tools/phase51j_observed_horizon_recovery.py \
  --feature-audit-run runs/phase51i_redacted_observed_pfill_feature_audit/PHASE51I-REDACTED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --lifecycle-truth-run runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51j_observed_horizon_recovery \
  --run-id PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000

python3 tools/phase51h_observed_pfill_feature_audit.py \
  --observed-pfill-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --quarantine-review-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z \
  --markout-readiness-run runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z \
  --horizon-recovery-run runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51j_recovered_observed_pfill_feature_audit \
  --run-id PHASE51J-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000

python3 tools/phase51i_pfill_feature_matrix_admissibility.py \
  --feature-audit-run runs/phase51j_recovered_observed_pfill_feature_audit/PHASE51J-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51j_pfill_feature_matrix_admissibility \
  --run-id PHASE51J-PFILL-FEATURE-MATRIX-ADMISSIBILITY-RECOVERED-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
2ab20ec4d916e9f61b758ab5fc54d6cc70cb23c75975db56a5b2d4d69f2922bf  observed_horizon_recovery_summary.json
9ba357b41753dbc7cc9f7592497abf81dc5a6474b2b15963fa78627fe6413a2e  observed_horizon_recovery_buckets.jsonl
767da905e574d5c58d73ee96a89da88295e75d1d102bc011b861e172a07f9cec  observed_horizon_recovery_labels.jsonl
0c3c1dd74588b96339b498437e3652398de61ff005711b6add5ddcbbac328a3c  observed_horizon_recovery/manifest.json
904e3d97eccbc33adee9a9c97043254881e79cda1fc71b12bcb4aa884387497f  recovered pfill_feature_audit_summary.json
aa909ba43ccd34fe055eb9c62f0d98a85371fea2d0e0038fe6164c6712ceea07  recovered pfill_feature_bucket_readiness.jsonl
b2aaf83faa0caeea81f365fe04b7502b91006622809489399433c283ca6425c3  recovered pfill_feature_coverage_labels.jsonl
2dcce34fa99e3257a8ee06d43a6a3dbc2da23b12723c51ea33f3eed7fc182309  recovered pfill_feature_matrix_admissibility_summary.json
775abf60df16e12913995a3834bdc69abac0ce5951e1b4acf5a7ef01d9804377  recovered pfill_feature_matrix_buckets.jsonl
0c94a2b37c7f2e89835c1eb414555e5f521c53dc500658b1ae89c9e5aaec9194  recovered pfill_feature_matrix_blockers.jsonl
a875f33de1cc47b4df6e896fa17bd12bdfdbf6ac736f1f2fd689acc4efb0611a  recovered matrix manifest.json
```

Recovered matrix blockers:

```text
missing_observed_horizon_features
lighter_native_limit_pressure_not_fully_observed
maker_taker_not_fully_observed_for_filled_orders
sparse_pfill_feature_buckets
observed_only_selection_bias_not_resolved
```

## Phase 5.1k - Filled-Horizon Timebase Recovery and Recovered Matrix

- Recovery run id:
  `PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovery run directory:
  `runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered 5.1h run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51K-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered 5.1i run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51K-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z`
- Baseline commit: `18dd09512288a85e440d3977e32432c3aabc1190`
- Gate status: `HOLD`
- Recovery gate reason: `phase51k_filled_horizon_timebase_partial`
- Matrix gate reason: `phase51i_filled_horizon_source_tick_still_missing`
- Raw identifier redaction status: `PASS`
- Matrix labels: `4527`
- Input filled horizons missing after 5.1j: `461`
- Filled source-tick horizons recovered in 5.1k: `396`
- Filled horizons still missing after 5.1k: `65`
- Exchange-ms-only horizons: `0`
- Recovered observed horizon available / missing: `4462` / `65`
- Lighter native-limit observed / partial remains: `0` / `2288`
- Filled-order maker/taker observed / partial-unknown / missing remains:
  `174` / `222` / `65`
- Excluded quarantine/review groups remain: `1613`
- Safety flags: all live, canary, capital, risk-relaxation, EV-admission,
  model-training, and financial-claim authorization fields are `false`.

Commands:

```bash
python3 tools/phase51k_filled_horizon_timebase_recovery.py \
  --feature-audit-run runs/phase51j_recovered_observed_pfill_feature_audit/PHASE51J-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --lifecycle-truth-run runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z \
  --run-id PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z

python3 tools/phase51h_observed_pfill_feature_audit.py \
  --observed-pfill-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --quarantine-review-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z \
  --markout-readiness-run runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z \
  --horizon-recovery-run runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z \
  --filled-horizon-recovery-run runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z \
  --run-id PHASE51K-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z

python3 tools/phase51i_pfill_feature_matrix_admissibility.py \
  --feature-audit-run runs/phase51h_observed_pfill_feature_audit/PHASE51K-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --run-id PHASE51K-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z
```

Artifact hashes:

```text
fb445f9b7c8dc7655f5796c6378c64be5df00349ec4ef93b8ca7ee35ae603ecc  filled_horizon_timebase_recovery_summary.json
cfbb7c1dd86d731448b13b2f1a3f0a632705795c2b256bf69ffff7ddbb85ea06  filled_horizon_timebase_recovery_buckets.jsonl
a1cfb1844604dd643decf68e9c654969a87bba7ef5d551f8143366193802fb51  filled_horizon_timebase_recovery_labels.jsonl
dd79db7f37573a1e6886db36a4fc889704ea406696fbfdd4c122868eae79fb5b  recovered pfill_feature_audit_summary.json
87495895b3aa66ad5acd86bf29d25cda68699677452ef390fa3d7b918df57e5c  recovered pfill_feature_bucket_readiness.jsonl
206c9d1e2a5340a9b9caf56e367733d858df8ee77750daceabdc5a72e313e6b2  recovered pfill_feature_coverage_labels.jsonl
ea85005b6ce7e19f6db889486b5e91e5f8c207b2e2443f08415eef9e44944909  recovered pfill_feature_matrix_admissibility_summary.json
62927129e74983467a75f47cca517d27869dacb899ce29626cec54195bf22dde  recovered pfill_feature_matrix_buckets.jsonl
c6971841da4ca84291aa8a320214dde1d0a945ddf5a76a2b1ad1a04da792fa6f  recovered pfill_feature_matrix_blockers.jsonl
```

Recovered matrix blockers:

```text
filled_horizon_source_tick_still_missing
missing_observed_horizon_features
lighter_native_limit_pressure_not_fully_observed
maker_taker_not_fully_observed_for_filled_orders
sparse_pfill_feature_buckets
observed_only_selection_bias_not_resolved
```

## Phase 5.1l - Filled-Horizon Source-Key Recovery and Recovered Matrix

- Recovery run id:
  `PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovery run directory:
  `runs/phase51l_filled_horizon_source_key_recovery/PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z`
- Recovered 5.1h run:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51L-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Recovered 5.1i run:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51L-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z`
- Baseline commit: `18dd09512288a85e440d3977e32432c3aabc1190`
- Gate status: `HOLD`
- Recovery gate reason:
  `phase51l_filled_horizon_source_key_complete_nonlive_hold`
- Matrix gate reason:
  `phase51i_lighter_native_limit_pressure_not_fully_observed`
- Raw identifier redaction status: `PASS`
- Matrix labels: `4527`
- Phase 5.1k target missing joins: `65`
- Filled horizons recovered in 5.1l: `65`
- Source P-fill horizon recoveries: `43`
- Observed-fill hash fallback recoveries: `22`
- Filled horizons still missing after 5.1l: `0`
- Recovered observed horizon available / missing: `4527` / `0`
- Lighter native-limit observed / partial remains: `0` / `2288`
- Filled-order maker/taker observed / partial-unknown / missing remains:
  `174` / `222` / `65`
- Excluded quarantine/review groups remain: `1613`
- Safety flags: all live, canary, capital, risk-relaxation, EV-admission,
  model-training, and financial-claim authorization fields are `false`.

Commands:

```bash
python3 tools/phase51l_filled_horizon_source_key_recovery.py \
  --phase51k-recovery-run runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --source-pfill-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --source-pfill-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --observed-label-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z \
  --observed-label-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z \
  --run-id PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z

python3 tools/phase51h_observed_pfill_feature_audit.py \
  --observed-pfill-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --quarantine-review-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z \
  --markout-readiness-run runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z \
  --horizon-recovery-run runs/phase51j_observed_horizon_recovery/PHASE51J-OBSERVED-HORIZON-RECOVERY-TWO-LANE-20260502T000000Z \
  --filled-horizon-recovery-run runs/phase51k_filled_horizon_timebase_recovery/PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-TWO-LANE-20260502T000000Z \
  --filled-horizon-source-key-recovery-run runs/phase51l_filled_horizon_source_key_recovery/PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-TWO-LANE-20260502T000000Z \
  --run-id PHASE51L-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z

python3 tools/phase51i_pfill_feature_matrix_admissibility.py \
  --feature-audit-run runs/phase51h_observed_pfill_feature_audit/PHASE51L-RECOVERED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --run-id PHASE51L-PFILL-FEATURE-MATRIX-ADMISSIBILITY-TWO-LANE-20260502T000000Z
```

Artifact hashes:

```text
d355a58043cb285454a2005725274f6c34ea9fdb19d1a906220b34f41b862790  filled_horizon_source_key_recovery_summary.json
ea0eb037d67a8ce2e36667e3e835331c48a778f0bd200af9826ae2dd2a0106b7  filled_horizon_source_key_recovery_buckets.jsonl
c2befefc86a808bee7674ccd021b5a608d4529dd7489cd3b3878e47899385607  filled_horizon_source_key_recovery_labels.jsonl
e68990f39d1c0d08a4601d2a344ed782b43ae492b24c04dabad76e82a3a014e2  recovered pfill_feature_audit_summary.json
ffb3574e7effe35e1550f1273f3e99cb636b1928393e5d579901e9b43bebc8a2  recovered pfill_feature_bucket_readiness.jsonl
a95e0d7e9e8f6c815821c790f72fe3b56fd6384bbd888bad695b71cfc7a1f993  recovered pfill_feature_coverage_labels.jsonl
4b087d1e44d4af778d5be7fb21c0dbdbbf9cb1f63c9f97b0d4168aa1241794b4  recovered pfill_feature_matrix_admissibility_summary.json
95ca05d6e4f160c9564847c1da427fc761298b2099429e5a445dc4ed0bd7091c  recovered pfill_feature_matrix_buckets.jsonl
f9a7feeacffbe66498f03086e3036e73b146cbfc9f62038e638df50825dc97df  recovered pfill_feature_matrix_blockers.jsonl
```

Recovered matrix blockers:

```text
lighter_native_limit_pressure_not_fully_observed
maker_taker_not_fully_observed_for_filled_orders
sparse_pfill_feature_buckets
observed_only_selection_bias_not_resolved
```

## Phase 5.1i - Redacted P-Fill Feature-Matrix Admissibility

- Run id: `PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-REDACTED-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51i_pfill_feature_matrix_admissibility/PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-REDACTED-TWO-LANE-20260502T000000Z`
- Source redacted feature-audit run:
  `runs/phase51i_redacted_observed_pfill_feature_audit/PHASE51I-REDACTED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Baseline commit: `18dd09512288a85e440d3977e32432c3aabc1190`
- Gate status: `HOLD`
- Gate reason: `phase51i_missing_observed_horizon_features`
- Raw identifier redaction status: `PASS`
- Raw identifier input present: `0`
- Matrix labels: `4527`
- Filled / terminal not-filled: `461` / `4066`
- Train / holdout: `3625` / `902`
- Queue/churn joined all source keys: `4527`
- Markout source available: `4527`
- Observed horizon available / missing: `18` / `4509`
- Lighter native-limit observed / partial: `0` / `2288`
- Filled-order maker/taker observed / partial-unknown / missing:
  `174` / `222` / `65`
- Excluded quarantine/review groups: `1613`
- Safety flags: all live, canary, capital, risk-relaxation, EV-admission,
  model-training, and financial-claim authorization fields are `false`.

Commands:

```bash
python3 tools/phase51f_canonical_pfill_outcome_rebuild.py \
  --lifecycle-truth-run runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --output-root runs/phase51i_redacted_canonical_pfill_outcome \
  --run-id PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000

python3 tools/phase51g_pfill_quarantine_review.py \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51i_redacted_pfill_quarantine_review \
  --run-id PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000

python3 tools/phase51h_observed_pfill_feature_audit.py \
  --observed-pfill-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --quarantine-review-run runs/phase51i_redacted_pfill_quarantine_review/PHASE51I-REDACTED-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51i_redacted_canonical_pfill_outcome/PHASE51I-REDACTED-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z \
  --markout-readiness-run runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51i_redacted_observed_pfill_feature_audit \
  --run-id PHASE51I-REDACTED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000

python3 tools/phase51i_pfill_feature_matrix_admissibility.py \
  --feature-audit-run runs/phase51i_redacted_observed_pfill_feature_audit/PHASE51I-REDACTED-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51i_pfill_feature_matrix_admissibility \
  --run-id PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-REDACTED-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
e1c191bb50128b4b42ed716007975bb4b5724a16a55d2cdb506c5faaaf268ae7  redacted canonical_pfill_order_labels.jsonl
50fd363193ee24bbc6f4a62d4e3f25fdba7c47ee13ba8b0f6707a628bbda9fd9  redacted canonical_pfill_outcome_summary.json
1ed04a832441456606f4a2e0074af1d11d703dacc9cffa439d01f20aa3b5b855  redacted observed_only_pfill_outcome/pfill_order_labels.jsonl
29e7703a8b85945a0725419fb11b3eaf75d67ffed1095daf353a7b0a97989eca  redacted pfill_feature_audit_summary.json
428753544c42c7bcaeeb95786cdfa5423b7a619164cbff2c7cf4805b9fbd2962  pfill_feature_matrix_admissibility_summary.json
b9135cdc6e7cdf09ef39d33fdc1bb708ffd67231ef1dd8f4bb822bdb850946ae  pfill_feature_matrix_buckets.jsonl
4d21e552ddc2efc93fa3fa1021de9a067cee0c0d4262b494ab6f8f696ecc847d  pfill_feature_matrix_blockers.jsonl
8e8759afc3f61a6cd709e38c8e5ae7c14fba9c7fa0b455d3561f73b7300aa6ec  manifest.json
```

Matrix blockers:

```text
missing_observed_horizon_features
lighter_native_limit_pressure_not_fully_observed
maker_taker_not_fully_observed_for_filled_orders
sparse_pfill_feature_buckets
observed_only_selection_bias_not_resolved
```

## Phase 5.1e Lifecycle/Native Truth Audit

- Run id: `PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z`
- Local run directory: `runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z`
- Gate status: `HOLD`
- Gate reason: `phase51e_canonical_lifecycle_reviewable_movements_found`
- Source telemetry SHA256 list:
  - `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
  - `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- P-fill rows audited: `11935`
- Current state entering audit: `489` filled, `2835` terminal not-filled,
  `8611` censored.
- Canonical lifecycle groups: `6999`
- Lifecycle events inspected: `46487`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Canonical lifecycle status counts:

```text
STAYS_FILLED: 489
STAYS_NOT_FILLED: 2835
CENSORED_TO_CANONICAL_FILLED_REVIEW: 464
CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW: 5206
CENSORED_TO_REPLACE_CHAIN_REVIEW: 288
DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW: 2270
CANCEL_ALL_SCOPE_REVIEW: 375
REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW: 8
```

Lighter native truth:

```text
lighter_native_gap_label_count: 189
ATTRIBUTED_NATIVE_ROLE: 96
NO_NATIVE_TRADE_MATCH: 93
lighter_raw_native_truth_label_count: 189
MATCHED_NATIVE_ID: 96
NATIVE_WINDOW_COVERED_NO_MATCH: 93
raw_native_roles: MAKER=64, TAKER=32, UNKNOWN=93
```

Interpretation:

```text
Phase 5.1d's 8611 censored P-fill labels are not primarily true
no-terminal/no-fill observations. Phase 5.1e shows that only 8 remain true
NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW after lifecycle canonicalization.
The dominant issue is label canonicalization around place intent/ack aliases,
order/client-id aliases, replace chains, and cancel-all scope.

The next non-live move is a separate canonical P-fill outcome rebuild/review
gate. Phase 5.1e does not authorize model training, EV admission, live orders,
canary, capital escalation, risk-limit relaxation, or financial claims.
```

Command:

```bash
python3 tools/phase51e_lifecycle_truth_audit.py \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --lighter-attribution-gap-run runs/phase51c_lighter_attribution_gap_audit/PHASE51D-LIGHTER-ATTRIBUTION-GAP-AUDIT-20260502T000000Z \
  --run-id PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
861fd63459957d1b6508c19ccdda020c4a2b4b40a60465457d7ad89d131a8f66  lifecycle_truth_audit_summary.json
82fad77b173825ccca0650368a9d68426c5e6d80c08ef5d05d0203ffbea189ce  order_lifecycle_truth_labels.jsonl
fe099f177ae981c81a41a75a6c757d1960a2331221bdf439bab5cecbdc0903ff  lighter_native_identity_gap_labels.jsonl
3892c024a8e6d60d766eece7d0c7effb9125801c9c18a2bac1b599735c487d11  lighter_raw_native_truth_labels.jsonl
6db64ea422d445cc63d32505577dc71b0e72003b533374ae35a25c1dd70ced50  manifest.json
afcf0fbce4268b96f4ee7a2a848e8950671e570fa7bc74a44f87e065a1ce684c  evidence_pack/artifact_index.json
```

## Phase 5.1f Canonical P-Fill Outcome Rebuild

- Run id:
  `PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51f_canonical_pfill_outcome_rebuild.py`
- Input lifecycle truth run:
  `runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z`
- Input P-fill outcome run:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input P-fill outcome run:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Gate status: `HOLD`
- Gate reason: `phase51f_canonical_pfill_contains_quarantined_review_groups`
- Source P-fill rows accounted for: `11935`
- Canonical P-fill groups emitted: `6140`
- Lifecycle-graph canonical groups from 5.1e: `6999`
- P-fill group diff vs lifecycle graph: `-859`
- Filled groups: `461`
- Terminal not-filled groups: `4066`
- Quarantined/censored review groups: `1613`
- Split conflicts in old per-order splits: `1673`
- Positive observed fill rate: `0.10183344378175392`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51f_canonical_pfill_outcome_rebuild.py \
  --lifecycle-truth-run runs/phase51e_lifecycle_truth_audit/PHASE51E-LIFECYCLE-TRUTH-AUDIT-TWO-LANE-20260502T000000Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --output-root runs/phase51f_canonical_pfill_outcome \
  --run-id PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
5df9223e2c617569eb3d8d3c5359558517907856341472f1e225a1046cb49f23  canonical_pfill_order_labels.jsonl
89e1835e6b3a090a99481d46b6287b5e22f7e5b466cff7b7b8eea8e75d1db187  canonical_pfill_outcome_summary.json
cee129458060843872a363cd54a32018781771736bcdc8129f99bce7bd181ac2  source_to_canonical_order_manifest.jsonl
3a73e53e3f0418b16ccaa712f4181c75a1c64d0870b36102be5a165ef2f4772a  split_conflict_manifest.jsonl
69c3881505b812238d7132fda4f7ab09a21600bb42b3d6bdbaae5facf57a5a36  quarantined_review_labels.jsonl
fe67edf1cd28fbce2291d193ce86e9a6c397fe99eadeae472b2127a5f331e137  evidence_pack/artifact_index.json
```

## Phase 5.1f P-Fill Readiness From Canonical Labels

- Run id:
  `PHASE51F-PFILL-CALIBRATION-READINESS-FROM-CANONICAL-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51f_pfill_calibration_readiness_from_canonical/PHASE51F-PFILL-CALIBRATION-READINESS-FROM-CANONICAL-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51c_pfill_calibration_readiness.py`
- Input P-fill outcome run:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Gate status: `HOLD`
- Gate reason: `pfill_calibration_contains_censored_orders`
- Order labels: `6140`
- Observed outcomes: `4527`
- Filled: `461`
- Terminal not-filled: `4066`
- Censored/quarantined: `1613`
- Censored rate: `0.26270358306188923`
- Holdout observed outcomes: `902`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_pfill_calibration_readiness.py \
  --pfill-outcome-run runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51f_pfill_calibration_readiness_from_canonical \
  --run-id PHASE51F-PFILL-CALIBRATION-READINESS-FROM-CANONICAL-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
a3db6620d5b93b07387b6a053b3fddea23c898bb09a817a3fb40a186fea68410  pfill_calibration_readiness_summary.json
4b724f5031fb9a817b2e84c1478c16a5de0de9596aa31a53b8eaeb05b13bf7ee  pfill_calibration_buckets.jsonl
345f486d9c5a639e5883a911574decf39a3a45536c35137ddbe5685bf382e186  pfill_order_split_manifest.jsonl
e1645ec99c27156a8153c8e6f830b5d5335703b2bf0e74f6a51b4a4a51bb9ba7  evidence_pack/artifact_index.json
```

## Phase 5.1g Canonical P-Fill Quarantine Review

- Run id:
  `PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51g_pfill_quarantine_review.py`
- Input canonical P-fill run:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Gate status: `HOLD`
- Gate reason: `phase51g_quarantine_review_observed_only_diagnostic_pack`
- Canonical P-fill groups reviewed: `6140`
- Observed terminal groups: `4527`
- Filled groups: `461`
- Terminal not-filled groups: `4066`
- Excluded quarantine/review groups: `1613`
- Train groups: `4911`
- Holdout groups: `1229`
- Exclusion reason counts:
  `EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL=1135`,
  `EXCLUDED_CANCEL_ALL_SCOPE_REVIEW=375`,
  `EXCLUDED_REPLACE_CHAIN_REVIEW=95`,
  `RIGHT_CENSORED_NO_TERMINAL=8`
- Venue quarantine counts:
  `lighter=844`, `hyperliquid=339`, `extended=236`, `aster=170`,
  `paradex=24`
- Old split conflicts preserved from source: `1673`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51g_pfill_quarantine_review.py \
  --canonical-pfill-run runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51g_pfill_quarantine_review \
  --run-id PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
de3cf674a26ae01a9189d86f15b52617a41e9889215552161a0050217602e8ec  binary_observed_pfill_order_labels.jsonl
de3cf674a26ae01a9189d86f15b52617a41e9889215552161a0050217602e8ec  observed_only_pfill_outcome/pfill_order_labels.jsonl
2121788d7b50ee593dc57a4a92daae668514bf45e52cda214dfa700a96e794fe  observed_only_pfill_outcome/pfill_outcome_summary.json
be33fab71e636c9801e6c083f9b0aee432ba05cf2c59170ccd829aaf09564f8b  quarantine_review_labels.jsonl
bf07596603805c5706e5bbb11c9aebb807d0b71ccb026893f826df177c878694  quarantine_review_summary.json
4a27d7b5772e3d49a58debd269dd1a426b9fb6958a776fc5f6dd4cd6461a2067  source_reconciliation_manifest.jsonl
```

## Phase 5.1g Observed-Only P-Fill Readiness Rerun

- Run id:
  `PHASE51G-PFILL-CALIBRATION-READINESS-OBSERVED-ONLY-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51g_pfill_calibration_readiness_observed_only/PHASE51G-PFILL-CALIBRATION-READINESS-OBSERVED-ONLY-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51c_pfill_calibration_readiness.py`
- Input P-fill outcome run:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome`
- Gate status: `HOLD`
- Gate reason: `pfill_calibration_sparse_buckets`
- Order labels: `4527`
- Observed outcomes: `4527`
- Filled: `461`
- Terminal not-filled: `4066`
- Censored/quarantined: `0`
- Train observed outcomes: `3625`
- Holdout observed outcomes: `902`
- Buckets: `12`
- Minimum observed per bucket: `200`
- Minimum holdout observed per bucket: `50`
- Missing observed horizon count: `4509`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_pfill_calibration_readiness.py \
  --pfill-outcome-run runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --output-root runs/phase51g_pfill_calibration_readiness_observed_only \
  --run-id PHASE51G-PFILL-CALIBRATION-READINESS-OBSERVED-ONLY-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
19330c451fe84a476ef2a42584e49854bb8e9534540c10b11ed2d34a42eaecfd  pfill_calibration_buckets.jsonl
219f35fee50f764a11aad97d5ce37b5c82d3f7b8d6d2675f77ef71b745cb7774  pfill_calibration_readiness_summary.json
ffecbfa3254c0b940d37a14651f97a0ae711678689fa6bcaf4aebeb7e75d2d5d  pfill_order_split_manifest.jsonl
```

## Phase 5.1h Observed-Only P-Fill Feature Audit

- Run id:
  `PHASE51H-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Local run directory:
  `runs/phase51h_observed_pfill_feature_audit/PHASE51H-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51h_observed_pfill_feature_audit.py`
- Input observed P-fill run:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome`
- Input quarantine review run:
  `runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z`
- Input canonical P-fill run:
  `runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z`
- Input queue/churn run:
  `runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input queue/churn run:
  `runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z`
- Input markout readiness run:
  `runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z`
- Gate status: `HOLD`
- Gate reason: `phase51h_raw_identifier_present_in_input_not_emitted`
- Observed labels audited: `4527`
- Filled labels: `461`
- Terminal not-filled labels: `4066`
- Train labels: `3625`
- Holdout labels: `902`
- Queue/churn joined all source keys: `4527`
- Queue/churn missing: `0`
- Queue reset proxy present: `694`
- Markout source context available: `4527`
- Observed horizon available: `18`
- Observed horizon missing: `4509`
- Lighter native limit status partial: `2288`
- Native limit not applicable for non-Lighter labels: `2239`
- Filled-order maker/taker observed: `174`
- Filled-order maker/taker partial/unknown: `222`
- Filled-order maker/taker missing: `65`
- Inherited raw decision ID present in input: `4389`; Phase 5.1h emits
  boolean presence only and does not emit raw IDs.
- Bucket records: `20`
- Missing feature total across labels: `11473`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_ev_admission`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51h_observed_pfill_feature_audit.py \
  --observed-pfill-run runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z/observed_only_pfill_outcome \
  --quarantine-review-run runs/phase51g_pfill_quarantine_review/PHASE51G-PFILL-QUARANTINE-REVIEW-TWO-LANE-20260502T000000Z \
  --canonical-pfill-run runs/phase51f_canonical_pfill_outcome/PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-TWO-LANE-20260502T000000Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --queue-churn-run runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z \
  --markout-readiness-run runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z \
  --output-root runs/phase51h_observed_pfill_feature_audit \
  --run-id PHASE51H-OBSERVED-PFILL-FEATURE-AUDIT-TWO-LANE-20260502T000000Z \
  --timestamp-ns 1777680000000000000
```

Artifact hashes:

```text
74a38fd69b1fea6254c2fc5ad748319ca8cda13e8a0c4a38c681bd462b180404  pfill_feature_audit_summary.json
fad30084482bf284f0c8bc2c759edcee013ff45a170c039aaf2120290268c80e  pfill_feature_bucket_readiness.jsonl
acc8830aff969df032f714bf8d5ee7709b7a17437034d85cc086ea39d0c3c87f  pfill_feature_coverage_labels.jsonl
```

## Phase 5.1c Markout Calibration Readiness Two-Lane Pack

- Run id: `PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z`
- Local run directory: `runs/phase51c_markout_calibration_readiness/PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51c_markout_calibration_readiness.py`
- Input observed run: `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z`
- Input join/holdout run: `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-20260429T025435Z`
- Input observed run: `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input join/holdout run: `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Source telemetry SHA256 list:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`,
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Gate status: `HOLD`
- Gate reason: `markout_readiness_sparse_buckets`
- Markout rows: `2196`
- Unique fills: `549`
- Train fills: `448`
- Holdout fills: `101`
- Markout horizons ms: `100`, `500`, `1000`, `5000`
- Buckets: `141`
- Adverse rows: `780`
- Adverse rate: `0.3551912568306011`
- Mean signed markout PnL: `-0.040162968947815805`
- Maker/taker counts by fill: `MAKER=118`, `TAKER=58`, `UNKNOWN=373`
- Candidate join counts by fill: `JOINED=306`, `MISSING=243`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`

Command:

```bash
python3 tools/phase51c_markout_calibration_readiness.py \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z \
  --join-holdout-run runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-20260429T025435Z \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --join-holdout-run runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --run-id PHASE51C-MARKOUT-CALIBRATION-READINESS-TWO-LANE-20260502T000000Z
```

Artifact hashes:

```text
21e8c977a2beb0019aba2b3e6dd2a5037a05450566911c68eb82dc95ad416fd1  evidence_pack/artifact_index.json
daf4ee36e77d2da2cdca0d678737a19ace72dbc7251a83d03e96260cb194f6cf  markout_calibration_buckets.jsonl
618eff73f99f91562143bd1eb8fb1ce50147c752f2c457371aa1f783e6d00022  markout_calibration_readiness_summary.json
1e995930404db07b097c113545aede55314bdb6cdd78ea0009810e46c416ab92  markout_fill_split_manifest.jsonl
```

Blockers:

```text
Sparse venue/side/horizon/split buckets remain below calibration thresholds.
373/549 fills still have UNKNOWN maker/taker role.
243/549 fills still have missing candidate joins.
Future reference prices are fair-value labels, not independent tape.
This is descriptive observed-fill markout evidence only, not unconditional EV
or balance-authoritative financial proof.
```

## Phase 5.1d P-fill Censoring Audit Two-Lane Pack

- Run id: `PHASE51D-PFILL-CENSORING-AUDIT-TWO-LANE-20260502T000000Z`
- Local run directory: `runs/phase51c_pfill_censoring_audit/PHASE51D-PFILL-CENSORING-AUDIT-TWO-LANE-20260502T000000Z`
- Tool: `tools/phase51c_pfill_censoring_audit.py`
- Input P-fill outcome run: `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input P-fill outcome run: `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Source telemetry SHA256 list:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`,
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Gate status: `HOLD`
- Gate reason: `pfill_censoring_audit_censored_orders_classified`
- Order labels: `11935`
- Observed outcomes: `3324`
- Filled: `489`
- Terminal not-filled: `2835`
- Censored: `8611`
- Censored rate: `0.7214914118139925`
- Buckets: `141`
- Reason counts: `OBSERVED_FILLED=489`,
  `OBSERVED_NOT_FILLED_TO_TERMINAL=2835`,
  `NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW=8611`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `approved_for_model_training`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`

Command:

```bash
python3 tools/phase51c_pfill_censoring_audit.py \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --run-id PHASE51D-PFILL-CENSORING-AUDIT-TWO-LANE-20260502T000000Z
```

Artifact hashes:

```text
76674e04dbbea9759b62c4e32690a3a410a26f58eefe747a6e1d7588783eb6b6  evidence_pack/artifact_index.json
f50459b5e60dc354f9101e2a345685aad59076a723dbe02093127f7258cabf69  pfill_censoring_audit_summary.json
1ea8f1e32a1ab63d567550c2897f1bf76f841265055767c6c83b3cb5d88dfbe7  pfill_censoring_buckets.jsonl
0b5e2abc4f0c9645006dd8d87a0d6fa6ba871db027239d5772a0bba64abb748a  pfill_censoring_labels.jsonl
```

Interpretation:

```text
The P-fill blocker is not explained by end-of-window truncation in this audit.
All 8611 censored labels have sufficient source-window coverage but no matched
fill or terminal lifecycle event. These remain censored and must not be treated
as terminal not-filled observations.
```

## Phase 5.1d Lighter Attribution Gap Audit

- Run id: `PHASE51D-LIGHTER-ATTRIBUTION-GAP-AUDIT-20260502T000000Z`
- Local run directory: `runs/phase51c_lighter_attribution_gap_audit/PHASE51D-LIGHTER-ATTRIBUTION-GAP-AUDIT-20260502T000000Z`
- Tool: `tools/phase51c_lighter_attribution_gap_audit.py`
- Input observed run: `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input join/holdout run: `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input Lighter trade backfill: `runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-COMBINED-TERMINAL-STALE-7200S-20260429T073231Z`
- Input Phase 5.1b native evidence: `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z`
- Source telemetry SHA256: `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Gate status: `HOLD`
- Gate reason: `lighter_attribution_gap_unknowns_unresolved`
- Lighter fills: `189`
- Native trade count: `300`
- Observed role counts: `MAKER=64`, `TAKER=32`, `UNKNOWN=93`
- Gap reason counts: `ATTRIBUTED_NATIVE_ROLE=96`, `NO_NATIVE_TRADE_MATCH=93`
- Stale unknowns upgradable from native identity: `0`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `approved_for_model_training`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`

Command:

```bash
python3 tools/phase51c_lighter_attribution_gap_audit.py \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --join-holdout-run runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --lighter-trade-backfill-run runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-COMBINED-TERMINAL-STALE-7200S-20260429T073231Z \
  --phase51b-native-run runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z \
  --run-id PHASE51D-LIGHTER-ATTRIBUTION-GAP-AUDIT-20260502T000000Z
```

Artifact hashes:

```text
e119855ff007e5961d10e4b4c04b9f4153199a6d976fc6705f4ec15947fef4ca  evidence_pack/artifact_index.json
4523a0c82015e45b632d363095161572a4464e0a35b91ea60ddde5d200732446  lighter_attribution_gap_labels.jsonl
6240c04f46bb332eb88ff1ecd5e21dbaee8d142955abd37842017eedf177be52  lighter_attribution_gap_summary.json
```

Interpretation:

```text
The current native backfill explains the 96 already-attributed Lighter fills.
The remaining 93 UNKNOWN Lighter fills have no native trade match in the
current 300-trade backfill and are not upgradable from existing native identity
evidence. Do not infer maker/taker role from intent.
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
- Gate reason: `observed_label_pack_partial_maker_taker_attribution`
- Fill labels: `356`
- Markout labels: `1424`
- Balance reconciliation labels: `1`
- Record count: `1781`
- Fill label status: `OBSERVED`
- Markout label status: `OBSERVED`
- Markout horizons: `100ms`, `500ms`, `1000ms`, `5000ms`
- Balance reconciliation status: `OBSERVED`
- Maker/taker role counts: `MAKER=7`, `TAKER=5`, `UNKNOWN=344`
- Lighter trade role index size: `368`
- Lighter trades JSON:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/source_snapshots/trades.sanitized.json`
- Lighter trades JSON SHA256:
  `249c441525520716a4c139dbb03d386e825d7102d7ea96d480fb511c737c1d0c`
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
  --lighter-trades-json runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/source_snapshots/trades.sanitized.json \
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
83a1d89c167bcd42f15d491da24efa4048662574d2d00f13a3e73585cc82691c  labels.jsonl
7c4fcad6ee154fc0c2876a369738bc8cc89d57ccc411ecf95ad6b19fa87f8efc  observed_label_summary.json
a9ae3b72f02dfea2d4c2795b2f38fa3ac8510cbc7528fd77dcb73a1e53dcdf5d  evidence_pack/artifact_index.json
```

## Source-Aligned Terminal-Stale EV Shadow M7

- Run id: `LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T073231Z_m7`
- Local run directory:
  `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T073231Z_m7`
- Source telemetry:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/telemetry_bounded.jsonl`
- Input artifact mode: `reference`
- Input records scanned: `27792`
- Input SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Candidates evaluated: `55584`
- Replay labels emitted: `55584`
- Gate status: `HOLD`
- Gate reason: `nonlive_shadow_requires_calibration_and_board_review`
- Calibration status: `SPARSE`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `live_orders_allowed`: `false`
- `capital_change_allowed`: `false`
- `risk_limit_relaxation_allowed`: `false`
- `admissible_for_financial_claim`: `false`

This replay uses the same source telemetry hash as the observed-label pack, so
it is suitable for deterministic quote/order/fill/markout join evidence. The
Phase 5.1 parsers were hardened to support concatenated JSON objects observed
in this historical source artifact without mutating the raw artifact.

Command:

```bash
python3 tools/phase51_ev_shadow.py \
  --input-telemetry /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/telemetry_bounded.jsonl \
  --run-id LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T073231Z_m7 \
  --output-root runs/phase51_lighter_only_ev_shadow \
  --input-artifact-mode reference
```

Validation:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T073231Z_m7/telemetry.jsonl
```

Result:

```text
OK: 111169 record(s) validated against schema v2
```

Artifact hashes:

```text
70d4d937b9804ce83f0307d3a3fb1853d58b56ee8bf467bfa35ebac7fec1d8c5  telemetry.jsonl
13d6e4c4eea8ed033c26f0316bfe60aedf03a2c853f0903bd8d6e9724ccc5688  ev_shadow_summary.json
1c8e17bb2675cf5c6ee7a8981b2b2624260fa2ce9805fd47e4a1e103047a6018  evidence_pack/artifact_index.json
```

## PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S

- Run id: `PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z`
- Local run directory:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- EV shadow telemetry SHA256:
  `70d4d937b9804ce83f0307d3a3fb1853d58b56ee8bf467bfa35ebac7fec1d8c5`
- Phase 5.1b acceptance artifact:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json`
- Gate status: `HOLD`
- Gate reason: `label_lake_scaffold_missing_fill_markout_balance_coverage`
- Record count: `77213`
- Quote decision labels: `55584`
- Order lifecycle labels: `21628`
- Fill labels: `0`
- Markout labels: `0`
- Balance reconciliation labels: `0`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_label_lake.py \
  --source-telemetry /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T073231Z/live_canary/telemetry_bounded.jsonl \
  --ev-shadow-telemetry runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T073231Z_m7/telemetry.jsonl \
  --phase51b-acceptance runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json \
  --run-id PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z
```

Artifact hashes:

```text
c6a83e0400aeda5e858b35e0b45c14715fa369f2d647ea2826529f94a0e54714  labels.jsonl
a6e46679d8c88b9b41a6d6985346c036317d8d530e9aa7ec8c1712ce72e8f3cf  label_lake_summary.json
df2e15bf5390d5b99b6c80ade9ee33dc4f8ae53cdcd2e50bfb39a8a98bc24256  evidence_pack/artifact_index.json
```

## PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S

- Run id:
  `PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-20260429T073231Z`
- Local run directory:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-20260429T073231Z`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- EV shadow telemetry SHA256:
  `70d4d937b9804ce83f0307d3a3fb1853d58b56ee8bf467bfa35ebac7fec1d8c5`
- Gate status: `HOLD`
- Gate reason: `deterministic_join_partial_maker_taker_attribution`
- Quote decision labels: `55584`
- Order lifecycle labels: `21628`
- Fill labels: `356`
- Order joins: `356`
- Candidate joins: `189`
- Complete quote/order/fill/markout joins: `189`
- Markout joins: `356`
- Balance reconciliation labels: `1`
- Deterministic train split: `295`
- Deterministic holdout split: `61`
- Maker/taker role counts: `MAKER=7`, `TAKER=5`, `UNKNOWN=344`
- Join reason counts:
  `complete_join=12`, `maker_taker_unknown=344`, `missing_candidate_join=167`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_join_holdout.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T073231Z \
  --run-id PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-20260429T073231Z
```

Artifact hashes:

```text
8928f5871d844f418af5bcbf8cb0edb2f74faacffdb839712c68aa9e32af3322  joined_labels.jsonl
9af2447ecbc088d5f4496f8cd5e8820ff58bcab117d2206701ed344805e0fffb  join_holdout_summary.json
cbb3d95e9a6a778b8a0fbae9552d4fd86f5e96136a156806d206bb64b2246a64  evidence_pack/artifact_index.json
```

Current blocker:

```text
5.1c remains HOLD because native maker/taker role attribution is incomplete
for 344/356 observed fills. No model training, live, canary, capital
escalation, risk-limit relaxation, or financial claim is authorized from this
pack.
```

## PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S

- Run id: `PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S-20260429T073231Z`
- Local run directory:
  `runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S-20260429T073231Z`
- Purpose: read-only native Lighter trade-history pagination for maker/taker
  attribution.
- Official docs: `https://apidocs.lighter.xyz/reference/trades`
- Source mode: `readonly_lighter_api`
- Query mode: documented `from` timestamp parameter plus cursor pagination.
- `from_timestamp_ms`: `1777454484345`
- `stop_at_or_before_ms`: `1777448154939`
- Pages fetched: `3`
- Trade count: `300`
- Timestamp range: `1777437192113` to `1777454483547`
- Complete to requested stop: `true`
- Role counts for account: `maker=196`, `taker=104`, `unknown=0`
- Gate status: `HOLD`
- Gate reason: `native_trade_backfill_readonly_attribution_input_only`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_lighter_trade_backfill.py \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --allow-sdk-auth \
  --lighter-sdk-path /tmp/lighter_sdk \
  --run-id PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S-20260429T073231Z \
  --market-id 0 \
  --market-type perp \
  --pages 10 \
  --limit 100 \
  --from-timestamp-ms 1777454484345 \
  --stop-at-or-before-ms 1777448154939 \
  --sleep-s 1.6
```

Artifact hashes:

```text
f421acc8d6db9e7704e99c3c857fd30ba4def8ef9b9f16ec531722c5336c04d6  source_snapshots/trades_backfill.sanitized.json
648311a7e4a0cd3985b0e32cb65ba2a1c21745c7e5588b6c0f79f81b925e4c06  lighter_trade_backfill_summary.json
a47dbfac1704c63412af4d919b2c33fe775aceba06cff9a25e146314bed679e2  evidence_pack/artifact_index.json
```

## PHASE51C-OBSERVED-LABELS-FROM-BACKFILL-M2

- Run id: `PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z`
- Local run directory:
  `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Lighter trades JSON SHA256:
  `f421acc8d6db9e7704e99c3c857fd30ba4def8ef9b9f16ec531722c5336c04d6`
- Gate status: `HOLD`
- Gate reason: `observed_label_pack_partial_maker_taker_attribution`
- Fill labels: `356`
- Markout labels: `1424`
- Balance reconciliation labels: `1`
- Maker/taker role counts: `MAKER=64`, `TAKER=32`, `UNKNOWN=260`
- Per-venue role counts:
  `lighter=64/32/93`, `aster=0/0/141`, `extended=0/0/10`,
  `hyperliquid=0/0/4`, `paradex=0/0/12` as `MAKER/TAKER/UNKNOWN`.
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
  --lighter-trades-json runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-FROM-TERMINAL-STALE-7200S-20260429T073231Z/source_snapshots/trades_backfill.sanitized.json \
  --run-id PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z
```

Artifact hashes:

```text
686cfd2a587d4f382a2771c225265b12dbd016f36e8cbb2a9783539d78eefadc  labels.jsonl
01ef7715362bb28c0406042797ad6012affe316a535b734bd4e7e95914df0731  observed_label_summary.json
14384e0d8b588ebf2b41a2f8b3f643712fb8939ad78af87d00f144a43dcf44d9  evidence_pack/artifact_index.json
```

## PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-FROM-BACKFILL-M3

- Run id:
  `PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M3-20260429T073231Z`
- Local run directory:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M3-20260429T073231Z`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- EV shadow telemetry SHA256:
  `70d4d937b9804ce83f0307d3a3fb1853d58b56ee8bf467bfa35ebac7fec1d8c5`
- Gate status: `HOLD`
- Gate reason: `deterministic_join_partial_maker_taker_attribution`
- Quote decision labels: `55584`
- Order lifecycle labels: `21628`
- Fill labels: `356`
- Order joins: `356`
- Candidate joins: `189`
- Complete quote/order/fill/markout joins: `189`
- Markout joins: `356`
- Balance reconciliation labels: `1`
- Deterministic train split: `295`
- Deterministic holdout split: `61`
- Maker/taker role counts: `MAKER=64`, `TAKER=32`, `UNKNOWN=260`
- Per-venue role counts:
  `lighter=64/32/93`, `aster=0/0/141`, `extended=0/0/10`,
  `hyperliquid=0/0/4`, `paradex=0/0/12` as `MAKER/TAKER/UNKNOWN`.
- Join reason counts:
  `complete_join=96`, `maker_taker_unknown=260`,
  `missing_candidate_join=167`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_join_holdout.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z \
  --run-id PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M3-20260429T073231Z
```

Artifact hashes:

```text
0615eae87dc1e2bb00a17dd77be444300985d900415f2d031738c3738b5d100a  joined_labels.jsonl
07742c38233f91b906ece7c6601a8b0dc8f0a9dfd236abfffdef256c5c2571bf  join_holdout_summary.json
5b249dfa45b6b969fd37d1effffe826a2a4acacc1b9d80b6c0fd24219fac1441  evidence_pack/artifact_index.json
```

Current blocker:

```text
5.1c remains HOLD. The Lighter-only lane has 96 native-role-attributed fills
and 93 Lighter fills whose telemetry IDs are absent from the run-window native
trade history. The remaining 167 unknown-role fills are non-Lighter fills and
do not imply a Lighter quote-candidate join failure.
```

## PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-FROM-BACKFILL-M4

- Run id:
  `PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M4-20260429T073231Z`
- Local run directory:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M4-20260429T073231Z`
- Purpose: refresh the first terminal-stale join pack with order identity fields
  required for order-level P_fill outcome labels.
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Gate status: `HOLD`
- Gate reason: `deterministic_join_partial_maker_taker_attribution`
- Fill labels: `356`
- Order joins: `356`
- Candidate joins: `189`
- Complete quote/order/fill/markout joins: `189`
- Maker/taker role counts: `MAKER=64`, `TAKER=32`, `UNKNOWN=260`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_join_holdout.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-FROM-BACKFILL-M2-20260429T073231Z \
  --run-id PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M4-20260429T073231Z
```

Artifact hashes:

```text
af3b274aff4881d810cf05547bf5514cd073a24ec02adbc157f752f61cbcc2c2  joined_labels.jsonl
639d11cff87c555e3433b9c79bc854ae0278963302ee4070174fb351c786fa6e  join_holdout_summary.json
204ddcd04660b3367efdf0266ca31ccfe49783cc54afc501141f36873bb2dc9e  evidence_pack/artifact_index.json
```

## PHASE51C-PFILL-OUTCOME-FROM-BACKFILL

- Run id:
  `PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Local run directory:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Input label lake:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z`
- Input join holdout:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M4-20260429T073231Z`
- Gate status: `HOLD`
- Gate reason: `pfill_outcome_contains_censored_orders`
- Place-order outcome labels: `6815`
- Filled orders: `310`
- Terminal not-filled orders: `1929`
- Censored/unobserved orders: `4576`
- Deterministic train split: `5478`
- Deterministic holdout split: `1337`
- Observed positive fill rate on non-censored orders: `0.13845466726217062`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_pfill_outcome_labels.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z \
  --join-holdout-run runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-FROM-BACKFILL-M4-20260429T073231Z \
  --run-id PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
```

Artifact hashes:

```text
8ca0b4d691a5f6a2096b958ab271ed9f56db0be57cc7dec98ce714d0952b0002  pfill_order_labels.jsonl
68363ab1a28e2b59703c1b3a06062be2ed69819d45fe5cc1dc1b2c702f7b356b  pfill_outcome_summary.json
e4f07e07ea4e990f15cf14d4561689800eecdf3fb5b4b3609090c4f86315a1f7  evidence_pack/artifact_index.json
```

## Source-Aligned Terminal-Stale 025435 EV Shadow

- Run id: `LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T025435Z`
- Local run directory:
  `runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T025435Z`
- Source telemetry:
  `/home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T025435Z/live_canary/telemetry_bounded.jsonl`
- Input artifact mode: `reference`
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- Candidates evaluated: `57016`
- Replay labels emitted: `57016`
- Gate status: `HOLD`
- Calibration status: `SPARSE`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51_ev_shadow.py \
  --input-telemetry /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T025435Z/live_canary/telemetry_bounded.jsonl \
  --run-id LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T025435Z \
  --output-root runs/phase51_lighter_only_ev_shadow \
  --input-artifact-mode reference
```

Validation:

```bash
python3 tools/check_telemetry_contract.py \
  runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T025435Z/telemetry.jsonl
```

Result:

```text
OK: 114033 record(s) validated against schema v2
```

Artifact hashes:

```text
50145959c159857992bf19c0ac2b4173b58bc828437a1f5b386fe6db9657b518  telemetry.jsonl
5c95b25328fdc3e3974570193aad71c19ad3494ddf66f013b43014c82ca37731  ev_shadow_summary.json
bcb2e8ae49bc4e2a754cd47c2b4187fae36d7dcec64c837d3d5fa6e80fe812b3  evidence_pack/artifact_index.json
```

## PHASE51C-LABEL-LAKE-TERMINAL-STALE-025435

- Run id: `PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z`
- Local run directory:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z`
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- EV shadow telemetry SHA256:
  `50145959c159857992bf19c0ac2b4173b58bc828437a1f5b386fe6db9657b518`
- Gate status: `HOLD`
- Gate reason: `label_lake_scaffold_missing_fill_markout_balance_coverage`
- Quote decision labels: `57016`
- Order lifecycle labels: `24859`
- Fill labels: `0`
- Markout labels: `0`
- Balance reconciliation labels: `0`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_label_lake.py \
  --source-telemetry /home/ubuntu/promotion_runs/phase5_reopened_terminal_stale_order_residual_requal_7200s_20260429T025435Z/live_canary/telemetry_bounded.jsonl \
  --ev-shadow-telemetry runs/phase51_lighter_only_ev_shadow/LTR-EV-SHADOW-001_terminal_stale_7200s_20260429T025435Z/telemetry.jsonl \
  --phase51b-acceptance runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z/phase51b_acceptance.json \
  --run-id PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z
```

Artifact hashes:

```text
11ae96f58963a1b8513794714e48164d2179e5b1a79ffa4c49385b289c403e89  labels.jsonl
697dfc919cd0654e29d8cd492c6b6c5fde12ecb3097d671cd61472370ed2dd52  label_lake_summary.json
c64a834dfc7e28688e74007262f1fc7439f05d8976dc7d745c36bb0f7cdbaec5  evidence_pack/artifact_index.json
```

## PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-025435

- Run id: `PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-7200S-20260429T025435Z`
- Local run directory:
  `runs/phase51c_lighter_trade_backfill/PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-7200S-20260429T025435Z`
- Purpose: read-only native Lighter trade-history pagination for maker/taker
  attribution.
- Source mode: `readonly_lighter_api`
- `from_timestamp_ms`: `1777438240934`
- `stop_at_or_before_ms`: `1777431463614`
- Pages fetched: `2`
- Trade count: `200`
- Timestamp range: `1777421078053` to `1777438206596`
- Complete to requested stop: `true`
- Role counts for account: `maker=136`, `taker=64`, `unknown=0`
- Gate status: `HOLD`
- Gate reason: `native_trade_backfill_readonly_attribution_input_only`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_lighter_trade_backfill.py \
  --env-file /home/ubuntu/paraphina/deploy/env/all5_recover_20260314.env \
  --allow-sdk-auth \
  --lighter-sdk-path /tmp/lighter_sdk \
  --run-id PHASE51C-LIGHTER-TRADE-BACKFILL-TERMINAL-STALE-7200S-20260429T025435Z \
  --market-id 0 \
  --market-type perp \
  --pages 10 \
  --limit 100 \
  --from-timestamp-ms 1777438240934 \
  --stop-at-or-before-ms 1777431463614 \
  --sleep-s 1.6
```

Artifact hashes:

```text
8b99bd01978be9ed2afd68710cd1e46c632ff45444173f6244a3ca9a736e2fbb  source_snapshots/trades_backfill.sanitized.json
8564016622e8e21e052f1e72ef5ab8a25a0a047682142113c9012949d1901dc9  lighter_trade_backfill_summary.json
474cf4227fef8bd13652e1c3fe62c5bbdc5d0ed3237eb3e3ed1ff7237c889f41  evidence_pack/artifact_index.json
```

## PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-025435

- Run id: `PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z`
- Local run directory:
  `runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z`
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- Lighter trades JSON SHA256:
  `8b99bd01978be9ed2afd68710cd1e46c632ff45444173f6244a3ca9a736e2fbb`
- Gate status: `HOLD`
- Gate reason: `observed_label_pack_partial_maker_taker_attribution`
- Fill labels: `193`
- Markout labels: `772`
- Balance reconciliation labels: `1`
- Maker/taker role counts: `MAKER=54`, `TAKER=26`, `UNKNOWN=113`
- Per-venue role counts:
  `lighter=54/26/37`, `aster=0/0/51`, `extended=0/0/19`,
  `hyperliquid=0/0/3`, `paradex=0/0/3` as `MAKER/TAKER/UNKNOWN`.
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Artifact hashes:

```text
3aaeb363ceb4700ddaf8a1e4d30ee2315270f85c100881fd5e3da78528c8dafe  labels.jsonl
507db3157ab54ce38c23d209359d135a13a9c1967b7008b8bf56363b04b188da  observed_label_summary.json
f9ed9d6c1f9272078bd318c2e775306a6986ef23edd0a81799352503779a57ac  evidence_pack/artifact_index.json
```

## PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-025435

- Run id:
  `PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-M2-20260429T025435Z`
- Local run directory:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-M2-20260429T025435Z`
- Purpose: refresh the second terminal-stale join pack with order identity
  fields required for order-level P_fill outcome labels.
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- EV shadow telemetry SHA256:
  `50145959c159857992bf19c0ac2b4173b58bc828437a1f5b386fe6db9657b518`
- Gate status: `HOLD`
- Gate reason: `deterministic_join_partial_maker_taker_attribution`
- Quote decision labels: `57016`
- Order lifecycle labels: `24859`
- Fill labels: `193`
- Order joins: `193`
- Candidate joins: `117`
- Complete quote/order/fill/markout joins: `117`
- Markout joins: `193`
- Balance reconciliation labels: `1`
- Deterministic train split: `153`
- Deterministic holdout split: `40`
- Maker/taker role counts: `MAKER=54`, `TAKER=26`, `UNKNOWN=113`
- Per-venue role counts:
  `lighter=54/26/37`, `aster=0/0/51`, `extended=0/0/19`,
  `hyperliquid=0/0/3`, `paradex=0/0/3` as `MAKER/TAKER/UNKNOWN`.
- Join reason counts:
  `complete_join=80`, `maker_taker_unknown=113`,
  `missing_candidate_join=76`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_join_holdout.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z \
  --observed-run runs/phase51c_observed_labels/PHASE51C-OBSERVED-LABELS-TERMINAL-STALE-7200S-20260429T025435Z \
  --run-id PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-M2-20260429T025435Z
```

Artifact hashes:

```text
01b6f3243b267f46c194f2596f7211e66af836c49ab7c3bb3248602829b6e22d  joined_labels.jsonl
5f405cec3671aefe40db04faa09aa36d6c75861566b5c320567c4921f5d8fe23  join_holdout_summary.json
5c27ef102805cffae4f8e2bad5528e6c9198e24b6fbf630adc49787c338ffb16  evidence_pack/artifact_index.json
```

## PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-025435

- Run id:
  `PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Local run directory:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- Input label lake:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z`
- Input join holdout:
  `runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-M2-20260429T025435Z`
- Gate status: `HOLD`
- Gate reason: `pfill_outcome_sparse_observed_fills`
- Place-order outcome labels: `5120`
- Filled orders: `179`
- Terminal not-filled orders: `906`
- Censored/unobserved orders: `4035`
- Deterministic train split: `4110`
- Deterministic holdout split: `1010`
- Observed positive fill rate on non-censored orders: `0.16497695852534563`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`

Command:

```bash
python3 tools/phase51c_pfill_outcome_labels.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z \
  --join-holdout-run runs/phase51c_join_holdout/PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-TERMINAL-STALE-7200S-M2-20260429T025435Z \
  --run-id PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z
```

Artifact hashes:

```text
e0ad36ed92f94f320a548fd48961ccde0c6c5efb07a1e6a5d4f72508504c4d4d  pfill_order_labels.jsonl
97f6c94f0780010d382cc9e2b5691c165b0158207f20c7e2caaa2b2f49583fbf  pfill_outcome_summary.json
34fcf5d99c1212421de97b2831a166a362a419213dc762edc7ebd0f9300f6792  evidence_pack/artifact_index.json
```

## PHASE51C-PFILL-CALIBRATION-READINESS-TWO-LANE

- Run id:
  `PHASE51C-PFILL-CALIBRATION-READINESS-TWO-LANE-20260502`
- Local run directory:
  `runs/phase51c_pfill_calibration_readiness/PHASE51C-PFILL-CALIBRATION-READINESS-TWO-LANE-20260502`
- Purpose: aggregate the two balance-backed P_fill outcome lanes into a
  HOLD-only calibration-readiness pack that preserves immutable order
  train/holdout splits, audits censoring, and computes Wilson 95% fill-rate
  intervals without training a model.
- Source telemetry SHA256 inputs:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`,
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- Input P_fill outcome runs:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`,
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Gate status: `HOLD`
- Gate reason: `pfill_calibration_contains_censored_orders`
- Bucket fields: `venue_id`, `side`
- Bucket count: `16`
- Order labels: `11935`
- Observed non-censored outcomes: `3324`
- Filled orders: `489`
- Terminal not-filled orders: `2835`
- Censored/unobserved orders: `8611`
- Censored rate: `0.7214914118139925`
- Train observed outcomes: `2655`
- Train filled outcomes: `382`
- Train censored labels: `6933`
- Holdout observed outcomes: `669`
- Holdout filled outcomes: `107`
- Holdout censored labels: `1678`
- Global observed fill rate: `0.14711191335740073`
- Global observed fill-rate Wilson 95% CI:
  `[0.13547763363313772, 0.15956089839135207]`
- Missing observed horizon fields: `9029`
- Terminal action counts: `NONE=9029`, `cancel=2551`, `replace=355`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`

Command:

```bash
python3 tools/phase51c_pfill_calibration_readiness.py \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --run-id PHASE51C-PFILL-CALIBRATION-READINESS-TWO-LANE-20260502
```

Artifact hashes:

```text
678cda5831e0a007dfa8947d7a3d285845e5897cead9729de16103f97c60df97  pfill_calibration_buckets.jsonl
5383610e1387a59882d6cdb494be30624dcf19a5cacc0f3c5e5fd6b95642c5ab  pfill_order_split_manifest.jsonl
ad236c4b40f08e1929edea2ce2d874ace6ebaba2f911664c6c2e85582a085b43  pfill_calibration_readiness_summary.json
91b86a13311a5f5b9496d31cc3f99f805e6d4bcdcbf766061deff86f88c82ff5  evidence_pack/artifact_index.json
```

## PHASE51C-QUEUE-CHURN-FROM-BACKFILL

- Run id:
  `PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Local run directory:
  `runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Purpose: emit HOLD-only order-level queue/churn proxy labels from the first
  balance-backed terminal-stale lane, joined to the accepted Phase 5.1b
  Lighter native-limit context where venue is Lighter.
- Source telemetry SHA256:
  `f89b92af3ff52bf953cdcc8f7736051a8833de776cb4612a3717e1d049f6ecd4`
- Input label lake:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z`
- Input P_fill outcome run:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z`
- Input Lighter native-limit context:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z`
- Gate status: `HOLD`
- Gate reason: `queue_churn_native_limit_pressure_unknown`
- Queue/churn labels: `6815`
- Lifecycle joins: `6815`
- Lifecycle misses: `0`
- Filled orders: `310`
- Terminal not-filled orders: `1929`
- Censored orders: `4576`
- Orders with churn: `3087`
- Orders with replace/queue-reset proxy: `1337`
- Orders with cancel: `2810`
- Orders with terminal horizon: `1990`
- Native-limit pressure partial labels: `3954`
- Native-limit pressure unknown labels: `2861`
- Native-limit pressure observed labels: `0`
- Lighter native-limit limitations:
  `lighter_sendtx_limit_not_exposed_by_account_limits_payload`,
  `lighter_rest_request_limit_not_exposed_by_account_limits_payload`,
  `lighter_open_order_limit_headroom_unknown`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`

Command:

```bash
python3 tools/phase51c_queue_churn_labels.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T073231Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z \
  --lighter-native-limits-run runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z \
  --run-id PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-FROM-BACKFILL-20260429T073231Z
```

Artifact hashes:

```text
3741491556a35084d364edc95573de3f7b36e3d8dfb9763cc2949f01e7c08147  queue_churn_labels.jsonl
4a658678d9153210b15fe8e330caf878370dbce409489ee77fbfa31359a11d43  queue_churn_summary.json
635e32980864b6cf97776a048afe8d35e6deda3c3aa505f2ca62184564ae3c18  evidence_pack/artifact_index.json
```

## PHASE51C-QUEUE-CHURN-TERMINAL-STALE-025435

- Run id:
  `PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z`
- Local run directory:
  `runs/phase51c_queue_churn/PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z`
- Purpose: emit HOLD-only order-level queue/churn proxy labels from the second
  balance-backed terminal-stale lane, joined to the accepted Phase 5.1b
  Lighter native-limit context where venue is Lighter.
- Source telemetry SHA256:
  `c1b0184628f04cf9e7db2671a8cbcc2d97473e5e5777625cd4855362bf543b89`
- Input label lake:
  `runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z`
- Input P_fill outcome run:
  `runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z`
- Input Lighter native-limit context:
  `runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z`
- Gate status: `HOLD`
- Gate reason: `queue_churn_native_limit_pressure_unknown`
- Queue/churn labels: `5120`
- Lifecycle joins: `5120`
- Lifecycle misses: `0`
- Filled orders: `179`
- Terminal not-filled orders: `906`
- Censored orders: `4035`
- Orders with churn: `1292`
- Orders with replace/queue-reset proxy: `440`
- Orders with cancel: `1187`
- Orders with terminal horizon: `916`
- Native-limit pressure partial labels: `2194`
- Native-limit pressure unknown labels: `2926`
- Native-limit pressure observed labels: `0`
- Lighter native-limit limitations:
  `lighter_sendtx_limit_not_exposed_by_account_limits_payload`,
  `lighter_rest_request_limit_not_exposed_by_account_limits_payload`,
  `lighter_open_order_limit_headroom_unknown`
- `approved_for_model_training`: `false`
- `approved_for_live`: `false`
- `approved_for_canary`: `false`
- `approved_for_capital_escalation`: `false`
- `admissible_for_financial_claim`: `false`
- `admissible_for_ev_admission`: `false`

Command:

```bash
python3 tools/phase51c_queue_churn_labels.py \
  --label-lake-run runs/phase51c_label_lake/PHASE51C-LABEL-LAKE-TERMINAL-STALE-7200S-20260429T025435Z \
  --pfill-outcome-run runs/phase51c_pfill_outcome/PHASE51C-PFILL-OUTCOME-TERMINAL-STALE-7200S-20260429T025435Z \
  --lighter-native-limits-run runs/phase51b_lighter_account_native_limits/PHASE51B-LIGHTER-ACCOUNT-NATIVE-LIMITS-20260502T002535Z \
  --run-id PHASE51C-QUEUE-CHURN-NATIVE-CONTEXT-TERMINAL-STALE-7200S-20260429T025435Z
```

Artifact hashes:

```text
bed50d7fb93eacf69e07ed9b31dde229432d60d5512f8ca5958ba04375b58b0e  queue_churn_labels.jsonl
c265e20fe54e8a908b827fe07ad71789b4bda032459ff0401cf41216171228ba  queue_churn_summary.json
1cc6cafc322f112cb724312432935a4d79c567ef08342a1f64670645caa66e71  evidence_pack/artifact_index.json
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
