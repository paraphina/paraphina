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
