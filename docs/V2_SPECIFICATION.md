# Paraphina V2 Specification

Status: active target specification for Phase 5.1 non-live research and
implementation review.

This document is the repo-owned V2 strategy specification. It supersedes the
external proposal language wherever that proposal conflicts with the Phase 5.1
board addenda. It does not authorize live orders, canary promotion, capital
escalation, risk-limit relaxation, model training, EV admission, or financial
claims.

Primary source addenda:

- `/home/ubuntu/paraphina/docs/Paraphina_V2_Whitepaper_Specification_Proposal`
- `/home/ubuntu/paraphina/docs/PARAPHINA PHASE 5.1 V2 SPECIFICATION REVISION ADDENDA`

Current repo evidence boundary:

- Phase 5 closeout baseline: `18dd09512288a85e440d3977e32432c3aabc1190`
- Current Phase 5.1 evidence boundary: Phase 5.1u emits the active forward
  capture target manifest, Phase 5.1w emits the operator request pack for the
  required sanitized local files, Phase 5.1x provides a Hyperliquid-only
  offline native-role adapter for local `userFills` snapshots, Phase 5.1y
  provides an all-venue offline native-role normalizer for already-local
  source rows, and Phase 5.1v is the active offline readiness gate that
  verifies a sanitized local capture bundle before Phase 5.1s; Phase 5.1s
  stages local source bundles, Phase 5.1t builds optional source-link sidecars,
  and Phase 5.1r validates redacted source-hash joins before Phase 5.1q
- Current V2 verdict: `HOLD` for model training, EV admission, canary, live
  orders, capital escalation, risk-limit relaxation, and financial claims

## 1. Scope And Non-Negotiables

V2 is a fill-aware, hedge-aware, arbitrage-informed market-making upgrade. It
keeps passive post-only market making as the default alpha capture mechanism
and makes executable cross-venue economics a first-class input to quote
admission, sizing, layering, and evidence.

V2 is not:

- a pure taker arbitrage bot;
- a license to place live orders from Phase 5.1 evidence;
- a capital escalation plan;
- a risk-limit relaxation;
- a multi-venue launch authorization;
- proof that Phase 5 already demonstrated V2 profitability, improved fills,
  lower directional exposure, successful fill-to-hedge, paired-inventory
  safety, or all-five economic participation.

Balance deltas across exchange accounts remain the live financial PnL
authority. Telemetry PnL, model EV, fill-level PnL, replay PnL, and simulated
PnL are diagnostic until reconciled to exchange balances with pre-registered
tolerances.

## 2. Rejected Formulations

The following formulations are explicitly rejected as canonical V2 logic:

```text
true_edge = max(local_edge, pair_edge)
Q_raw = true_edge / eta
```

`local_edge` and executable `pair_edge` may remain raw explanatory features.
Neither may directly determine quote admission or quote size without fill
probability, hedge success/failure, costs, confidence bounds, and failure-state
EV.

Positive apparent pair edge is not sufficient. A quote with positive apparent
pair edge must still be rejected if fill probability is too low, hedge success
is too uncertain, adverse selection dominates, rounded size is uneconomic, rate
or order-management churn consumes the edge, residual inventory is unsafe, or
required telemetry is missing.

## 3. Order Candidate Contract

Every evaluated quote must be represented as an `OrderCandidate` before
admission. The candidate must be serializable, replayable, and reconstructable
from raw market/inventory events, config hash, model version, and baseline
commit.

Required candidate fields:

- `candidate_id`: deterministic ID derived from run, event sequence, venue,
  side, layer, price, size, model version, and config hash.
- `run_id`, `baseline_commit`, `config_hash`, `model_version`.
- `instrument_id`, `entry_venue_id`, `side`, `layer`.
- `passive_price`: final tick-snapped post-only price after passivity checks.
- `candidate_size_Q`: final candidate size in base units and notional USD.
- `intended_lifetime_ms`: planned rest time before cancel/replace evaluation.
- `local_edge_feature`: fair-value-local maker edge after fees, ticks,
  passivity, volatility buffer, and venue quote-cost buffer.
- `pair_edge_feature`: executable pair-edge feature at candidate price and
  size for eligible hedge venues.
- `primary_hedge_venue_id`, `backup_hedge_venue_id`, or explicit
  `NO_BACKUP_AVAILABLE`.
- `quote_context`: fair value, book age, spread, depth, volatility, toxicity,
  risk regime, inventory, margin, rate-limit state, and venue health.
- `model_features_hash`: deterministic hash of the feature vector.

Acceptance criterion: every admitted or rejected quote decision must be
replayable from candidate records, raw events, baseline commit, config hash,
and model version.

## 4. Canonical EV Objective

The canonical V2 objective for a candidate order is:

```text
EV(order, Q, p, T) =
  P_fill(order, Q, p, T) * [
      P_hedge_success(order, Q, p, T) * E_locked_edge(order, Q, p)
    + P_hedge_partial(order, Q, p, T) * E_partial_hedge_state(order, Q, p)
    + P_hedge_fail(order, Q, p, T) * E_residual_inventory_state(order, Q, p)
  ]
  - E_adverse_selection(order, Q, p, T)
  - E_queue_reset(order, Q, p, T)
  - E_churn(order, Q, p, T)
  - E_capital_funding(order, Q, p, T)
  - E_tail_risk(order, Q, p, T)
```

For pair-conditioned candidates:

```text
P_hedge_success + P_hedge_partial + P_hedge_fail = 1
```

For local-edge-driven candidates with no intended immediate hedge:

```text
P_hedge_success = 0
P_hedge_partial = 0
P_hedge_fail = 1
```

The residual inventory state must capture expected value after a passive maker
fill without immediate hedge. It must be non-positive by default unless
specific reproducible evidence supports a positive local inventory state.

### Required EV Components

`P_fill(order, Q, p, T)` estimates the probability that a passive order gets a
fill or economically material partial fill during the intended lifetime.
Calibration must condition on venue, side, layer, distance to touch, size
relative to depth, queue age if available, cancel/replace history, spread,
volatility, time of day, toxicity, venue health, rate state, recent fill
intensity, post-only outcomes, and order lifetime.

`P_hedge_success`, `P_hedge_partial`, and `P_hedge_fail` estimate hedge outcome
probabilities conditional on passive fill. They must condition on entry/hedge
venue pair, side, size bucket, hedge book age, usable depth, guard price,
latency, rate headroom, margin, liquidation distance, connector health, error
rate, recent hedge attempts, and edge decay half-life.

`E_locked_edge` is the expected realized edge when passive fill and intended
hedge both succeed, after maker/taker fees, slippage, latency decay, connector
or settlement cost, and funding/carry over the expected holding window.

`E_partial_hedge_state` is the expected value of a partial hedge, including
executed hedge fraction distribution, residual delta, repair cost, liquidation
and margin risk, basis exposure, and pair-budget impact.

`E_residual_inventory_state` is the expected value when hedge fails, is skipped,
is rejected, becomes uneconomic, or is blocked by risk. It must include markout
while unpaired, periodic hedge or target-relative exit cost, forced risk
reduction probability, extra capital/margin use, basis/fragmentation penalties,
liquidation-distance penalty, and kill-switch contribution.

`E_adverse_selection` is the expected toxic-fill cost measured from conditional
markout after maker fill over pre-registered horizons: `100ms`, `500ms`, `1s`,
`5s`, intended hedge completion horizon, and next fair-value update horizon.

`E_queue_reset` is the expected opportunity cost of cancel/replace behavior that
loses queue priority or causes post-only rejection/re-entry.

`E_churn` is the operational and opportunity cost of order lifecycle actions:
rate-limit use, connector load, message failure probability, and blocked future
risk-reducing actions.

`E_capital_funding` is expected cost or benefit from margin, funding, borrow,
collateral, account fee profile, and opportunity cost of capital.

`E_tail_risk` is a conservative penalty for low-frequency high-loss states:
venue outage after fill, stale hedge book, volatility shock, partial hedge with
widening spread, connector reject while unpaired, rate exhaustion preventing
cancel/hedge, close-only mode, liquidation deterioration, and correlated basis
collapse.

## 5. Admission, Sizing, And Layering

A candidate may be admitted only when all hard gates pass:

- kill switch inactive;
- risk regime permits new quote risk;
- entry venue enabled;
- entry book fresh and valid;
- fair value available where required;
- toxicity below hard-block threshold;
- margin sufficient for entry fill and residual states;
- liquidation-safe size positive;
- order price valid under tick, passivity, and venue constraints;
- rate-limit headroom sufficient for entry management and risk-reducing action;
- telemetry and replay metadata complete.

The canonical admission rule is:

```text
admit(order, Q, p, T) only if LCB_alpha(EV(order, Q, p, T)) > 0
```

The lower confidence bound must remain positive after final tick snapping,
passivity checks, lot rounding, size-step rounding, and min-notional checks.

Pair-conditioned candidates require additional hedgeability gates:

- primary hedge venue valid;
- backup venue recorded or explicitly unavailable;
- hedge book fresh;
- hedge depth supports `Q` under guard price;
- hedge venue margin supports the hedge action;
- hedge-confidence model available and fresh;
- immediate hedge serialization enabled in non-live simulation or explicitly
  disabled with no pair-conditioned admission.

Sizing is a constrained discrete optimization:

```text
Q* = argmax_Q in F(order) LCB_alpha(EV(order, Q, p, T))
```

The feasible set `F(order)` is the venue size grid constrained by min lot, min
notional, hedge depth, entry and hedge margin, pair budget, unpaired delta
budget, liquidation-safe size, order limits, rate-limit proxy, and configured
non-live experiment cap.

The optimizer must evaluate size-dependent slippage, hedge-success probability,
adverse selection, tail penalty, residual inventory cost, and binding
constraints. A discrete grid search is acceptable for Phase 5.1 non-live work.

V2 may evaluate multiple layers per venue-side:

- `TOUCH`: small, strict-confidence candidates near top of book.
- `WORKING`: less aggressive passive quotes with lower queue/churn pressure.
- `INVENTORY_REDUCING`: candidates that reduce existing unpaired risk.
- `BASELINE_COMPAT`: candidates used to compare against the clean baseline.

If venue/order limits allow only one order per venue-side, select the candidate
with the highest positive lower-bound EV after risk-priority rules.

## 6. Pair Edge, Fair Value, And Local Edge

Global fair value remains the anchor for risk, mark-to-market, local edge,
basis, and inventory accounting. Pair-edge computation must not corrupt fair
value construction.

Executable pair edge is a feature generator, not the objective. It must be
computed by a deterministic market-state module that consumes fresh books,
fees, account profile, venue health, margin state, depth, latency estimates,
and rate-limit state. Pair edge is invalid if any required input is stale,
missing, disabled, toxic, margin-constrained, rate-constrained, or liquidation
unsafe.

Local-edge-driven quoting remains valid when no hedge venue has fresh depth,
hedge confidence is below minimum, pair-edge evidence is unavailable, the quote
reduces unpaired inventory, baseline-compatible MM is being evaluated, pair
budget is exhausted, apparent pair edge is within uncertainty, or local-edge EV
has superior confidence-bounded value.

## 7. Inventory, Residual State, And Risk

Inventory must be decomposed as:

```text
q_total_v = q_paired_v + q_unpaired_v
```

Paired inventory must carry a `pair_id`, long venue, short venue, instrument
mapping, delta ratio, hedge venue where applicable, pair budget consumed, pair
creation timestamp, validity state, and invalidation reason.

Unpaired directional exposure must be separately observable and tighter than
ordinary global delta. Paired basis inventory is not free risk; it consumes
pair budget, margin budget, and basis risk budget.

A pair is degraded or invalidated when either venue is stale, disabled, toxic,
margin-constrained, near liquidation, connector-degraded, fee/account profile
assumptions change, instrument mapping diverges, funding/basis moves outside
model support, required telemetry is missing, or kill switch is active.

For each fill on a pair-conditioned candidate, the system must create exactly
one fast-hedge decision before periodic hedge or target-relative exit can act
on that quantity. The mutually exclusive outcomes are `SEND`, `SKIP`, `DEFER`,
and `BLOCKED`. Failed, skipped, blocked, or partial hedges immediately create
auditable residual state.

Each quantity of inventory must have exactly one active action owner:
`FAST_HEDGE`, `TARGET_RELATIVE_EXIT`, `PERIODIC_HEDGE`,
`RISK_REDUCE_ONLY`, or `NO_ACTION_INSIDE_BUDGET`.

Hard risk limits always override pair preservation.

## 8. Telemetry And Evidence

V2 evidence must separate observed facts, model estimates, counterfactual
decisions, counterfactual outcomes, simulated outcomes, paper/testnet outcomes,
and balance-authoritative live economics.

Required event families for V2 implementation review:

- `V2_RUN_CONTEXT`
- `V2_MARKET_SNAPSHOT`
- `V2_PAIR_EDGE_SNAPSHOT`
- `V2_EV_EVALUATED`
- `V2_ORDER_INTENT`
- `V2_ORDER_LIFECYCLE`
- `V2_FILL_OBSERVED`
- `V2_FAST_HEDGE_DECISION`
- `V2_HEDGE_LIFECYCLE`
- `V2_INVENTORY_SNAPSHOT`
- `V2_BALANCE_SNAPSHOT`
- `V2_REPLAY_LABEL`
- `V2_GUARDRAIL_EVENT`

Every event path must preserve non-live authorization fields, config hash,
baseline commit, model version, schema version, and enough identifiers to
replay without exposing secrets or raw private identifiers.

Replay must:

- verify the clean baseline commit or record the exact diff hash;
- reproduce baseline decisions;
- run V2 in decision-shadow mode on the same event stream;
- emit EV and replay-label records for every candidate;
- store manifest, artifact index, hashes, commands, configs, model versions,
  dataset hashes, and seeds;
- rerun byte-stable where determinism is expected;
- block economic claims if attribution does not reconcile to balance facts.

## 9. Calibration And Statistical Gates

V2 model calibration requires out-of-sample or regime-separated holdout before
promotion beyond non-live evidence.

Primary metrics:

- guardrail non-regression;
- deterministic reproducibility;
- EV calibration;
- economic superiority only in authorized evidence lanes.

Secondary metrics:

- candidate count by venue/side/layer;
- admission/rejection reason distribution;
- P_fill estimate quality;
- markout distribution;
- hedge success estimate quality;
- queue reset and churn cost quality;
- capital/funding attribution;
- paired vs unpaired inventory residence time.

Provisional minimum sample rules:

- at least `1000` quote candidate observations per venue/side/layer/regime
  bucket;
- at least `200` observed fills per fill-model bucket, or hierarchical pooling
  with explicit uncertainty inflation;
- at least `100` observed or paper/testnet hedge attempts per hedge-success
  bucket before venue-pair-specific estimates;
- sparse buckets force `HOLD` unless the board approves pooling.

Confidence intervals must account for serial correlation; IID intervals are
not acceptable for order-book time series unless justified.

## 10. Systems Delta Contract

The clean Phase 5 baseline is insufficient for V2 behavior until these
components are implemented, tested, and evidenced:

- pair-edge data model;
- EV evaluation service or module;
- hedge-confidence model;
- layer-aware OMS;
- order metadata propagation from candidate to fill and hedge;
- immediate fill-to-hedge serialization state machine;
- double-action prevention;
- paired/unpaired inventory accounting;
- pair invalidation and residual-state handling;
- target-relative exit semantics;
- pair-aware periodic hedge semantics;
- V2 telemetry validation;
- deterministic replay and unit-test invariants.

Minimum invariant tests:

- stale entry or hedge venue invalidates pair edge;
- disabled or toxic venue invalidates pair edge;
- positive pair-edge feature with negative lower-bound EV is rejected;
- post-rounding negative EV is rejected;
- size optimizer selects discrete EV argmax, not `true_edge / eta`;
- one fast-hedge decision per pair-conditioned fill;
- partial or rejected hedge creates residual state;
- periodic hedge and target-relative exit cannot double-act on fast-hedge-owned
  quantity;
- paired inventory cannot also count as unpaired;
- kill switch blocks new quote risk and overrides pair preservation;
- missing metadata blocks economic claims;
- no-live flag prevents order sending in non-live experiments.

## 11. Venue Readiness And First Experiment

V2 must not treat venue execution as homogeneous. Each venue needs evidence for
account mode, fee profile, maker/taker economics, funding, post-only semantics,
cancel/replace behavior, rate limits, active-order limits, lot/tick/min-notional
constraints, telemetry visibility, connector readiness, and experiment
blockers.

The first executable V2 experiment remains venue-local and non-live:

```text
Experiment ID: LTR-EV-SHADOW-001
Venue: Lighter only
Mode: replay/shadow plus connector dry-run
Authorization: no live orders, no capital changes, no risk-limit changes
```

Lighter-first does not mean other venues are strategically inferior. It means
the first V2 evidence lane must prove one venue-local telemetry and lifecycle
contract before generalizing.

V2 strategic scope remains all five current venues: Aster, Extended,
Hyperliquid, Paradex, and Lighter. Phase 5.1 remains a Lighter-first,
venue-local non-live evidence lane selected to prove the telemetry,
lifecycle, and evidence contract before generalizing. It does not narrow V2 to
Lighter, does not authorize multi-venue launch, and does not prove non-Lighter
venue contracts are ready.

Aster remains blocked from first-experiment promotion unless taker-fee leakage
is explicitly bounded and observable. Extended remains excluded from the first
experiment due to bridge opacity and asynchronous acceptance. Paradex and
Hyperliquid require profile/account/fee/queue evidence before promotion.

## 12. Promotion Matrix

`PROMOTE_FOR_NEXT_NONLIVE_STEP` only if:

- baseline replay reproduces;
- V2 decisions replay deterministically;
- EV telemetry is complete;
- hard gates are not bypassed;
- no-live/no-capital/no-risk authorization fields remain safe;
- source artifacts, commands, configs, hashes, and manifests are reproducible.

`HOLD` if:

- guardrails pass but calibration is sparse or inconclusive;
- fill or hedge labels are insufficient;
- venue/account/fee assumptions are unresolved;
- balance reconciliation is unavailable for any economic claim;
- telemetry required for EV components is missing.

`REJECT` if:

- objective cannot be computed from telemetry;
- V2 bypasses hard gates;
- residual state is unauditable;
- model produces positive EV for known negative holdout outcomes;
- replay cannot reproduce the baseline;
- any non-live evidence path changes live execution.

Canary and live deployment are outside this specification and require a later
board decision, separate evidence, unchanged hard risk limits, operations
runbook approval, balance-authoritative reconciliation, and rollback proof.

## 13. Current Open Board Decisions

These decisions remain open and must be resolved before model training or EV
admission:

- confidence standard for `LCB_alpha`, with `alpha = 0.05` one-sided as the
  provisional default;
- exact uncertainty inflation for sparse hierarchical pooling;
- exact tighter unpaired directional exposure budget;
- non-live pair-budget accounting limits;
- Warning/Critical/Kill behavior for paired inventory invalidation;
- whether immediate fill-to-hedge may advance beyond simulation/paper;
- whether pure one-sided arbitrage remains research-only or moves to appendix;
- whether the term `true_edge` is removed entirely or retained only as a
  deprecated shorthand.

## 14. Current Phase 5.1 Blockers

As of Phase 5.1s, the V2 evidence path remains blocked by:

- Lighter native-limit context still partial for `2288` labels because
  historical active-order snapshots do not include full sendTx/REST pressure;
- maker/taker status incomplete for `287` fills because existing source
  artifacts do not provide exact canonical venue-native role joins;
- sparse venue/side/layer/regime buckets;
- observed-only selection bias from `1613` excluded quarantine/review groups.

Phase 5.1l resolved the previous filled-horizon blocker by recovering the
remaining `65 / 65` source-tick horizons: `43` via source P-fill horizon
evidence and `22` via hashed observed-fill fallback. The recovered matrix now
reports `4527 / 4527` observed horizons available and `0` missing horizons.

Phase 5.1m added official Lighter active-order cap evidence and records current
active-order headroom without treating it as historical event-time pressure.
The recovered matrix correctly remains at `0 / 2288` native-limit rows observed
and `2288 / 2288` partial, while preserving `0` missing horizons, `0`
unrecovered filled-horizon source-key rows, and raw-identifier redaction `PASS`.

Phase 5.1n added historical Lighter account snapshot alignment. It aligned
`1728 / 2194` Lighter rows in the 025435 lane and `3700 / 3954` rows in the
073231 lane to event-time active-order snapshots, but it kept
`native_limit_observed_count = 0` because sendTx/REST pressure was not captured
historically. Phase 5.1n also added all-venue maker/taker recovery and
classified the remaining `287` filled rows as missing venue-native role source.

Phase 5.1o added a venue-native role source inventory. It found `174` filled
rows remain input-observed, `0` roles were recovered by exact canonical
evidence, `125` Lighter filled rows have source material available but no exact
canonical join, and `162` filled rows across Aster, Extended, Hyperliquid, and
Paradex have no retained native-role source in current artifacts. The rerun
matrix therefore remains `HOLD` with the same four blockers.

Phase 5.1p added a quarantined Lighter native-role canonical join attempt over
all retained Lighter native trade backfills. It hashed native side order/client
identifiers internally, emitted no raw order/client/trade identifiers, indexed
`531` unique native Lighter trades with explicit maker/taker roles, and found
`0 / 125` exact canonical matches against the source-available Lighter filled
rows. The rerun matrix remains `HOLD` with `174` maker/taker-observed filled
rows and `287` filled rows still lacking complete venue-native role evidence.

Phase 5.1q adds the forward native-evidence capture gate. It consumes sanitized
all-five venue-native role source rows and Lighter event-time native-limit
pressure rows, emits `native_role_evidence.jsonl` for the Phase 5.1n recovery
gate, and rejects raw order/client/fill/trade identifiers. This is the active
repo-owned path for future exact canonical maker/taker completeness and
sendTx/REST native-limit pressure evidence.

Phase 5.1r adds the source-acquisition layer in front of Phase 5.1q. It may
ingest local quarantined raw venue-native snapshots, but it emits only redacted
canonical-group/count/hash evidence: `native_role_source.jsonl`,
`native_limit_source.jsonl`, and source-acquisition labels. Its baseline
no-source run is redaction-safe and clears no blockers: `0 / 287` native-role
targets recovered and `0 / 3132` Lighter native-limit targets recovered.
Phase 5.1r also accepts optional validated source-link sidecars that map
redacted source-record hashes to observed `canonical_group_id` or `order_key`
when the staged source row cannot carry direct join fields. The sidecar is join
evidence only; it does not infer maker/taker role, Lighter native-limit
pressure, EV, PnL, or economic performance.

Phase 5.1s adds the local manifest-driven source-staging layer in front of
Phase 5.1r. It rejects network paths, `.env` files, symlinks, secret-shaped
fields, and unsafe true authorization flags, strips raw venue identifiers, and
emits `local_native_source.jsonl` for Phase 5.1r. It also stages optional
manifest `source_links` as `local_source_link_sidecar.jsonl` when direct
canonical group/order-key fields cannot be embedded in redacted source rows.
Its first run over existing Lighter local snapshots staged `405` rows and
stripped `3500` raw identifier fields, but found `0` join-key rows; the
downstream Phase 5.1r rerun therefore kept `0 / 287` native-role targets
recovered and `0 / 3132` Lighter native-limit targets recovered.

Phase 5.1t adds a HOLD-only source-link sidecar builder in front of Phase 5.1s.
It reads quarantined local source snapshots, matches only by existing redacted
order/client identifier hashes in observed P-fill labels, and emits a Phase
5.1s-compatible `source_links.sanitized.jsonl` sidecar. It does not infer
maker/taker role, native-limit pressure, EV, PnL, or economic performance. The
first 5.1t run over existing local Lighter snapshots processed `1522` source
rows, emitted `363` redacted source-link rows, and let Phase 5.1r apply `909`
source-link joins. Those joins were not sufficient to clear the blocker:
5.1q/5.1n still recovered `0 / 287` missing native-role targets and `0 / 3132`
Lighter native-limit targets. This proves the source-link path is runnable and
redaction-safe, but existing local artifacts still lack the exact all-five
venue-native role and complete Lighter native-limit evidence required for
calibrated EV review.

Phase 5.1u adds a HOLD-only forward capture target manifest in front of the
fresh source-capture pilot. It consumes the canonical P-fill labels and emits
the exact missing target set: `287` all-five venue-native role targets
(`aster=113`, `extended=28`, `hyperliquid=6`, `lighter=125`, `paradex=15`) and
`3132` Lighter event-time native-limit pressure targets. It also emits a local
capture bundle manifest template for Phase 5.1s. Phase 5.1u does not capture
source truth, does not clear blockers by itself, and does not infer maker/taker
role, native-limit pressure, EV, PnL, or economic performance.

Phase 5.1v adds a HOLD-only forward capture bundle-readiness gate in front of
Phase 5.1s. It consumes the Phase 5.1u target run and a local candidate
capture-bundle manifest, rejects unsafe source surfaces, checks whether local
redacted source rows cover the required all-five native role targets and
Lighter native-limit targets, applies validated source-link sidecars as join
aids when source rows cannot carry direct `canonical_group_id` or `order_key`
fields, and emits a Phase 5.1s-ready generated manifest only when all targets
are structurally ready. Source-link sidecars alone are not evidence; linked
source rows must also be present and carry the required native role or
native-limit fields. Its baseline run against the
Phase 5.1u placeholder template intentionally remains `HOLD`: `0 / 287`
native-role targets ready, `0 / 3132` Lighter native-limit targets ready, and
`PLACEHOLDER_PATH` source status. Phase 5.1v does not capture source truth,
does not clear blockers by itself, and does not infer maker/taker role,
native-limit pressure, EV, PnL, or economic performance.

Phase 5.1w adds a HOLD-only request-pack gate in front of Phase 5.1v. It
consumes the Phase 5.1u target run and emits an operator Markdown pack, JSON
pack, and capture-bundle manifest skeleton for the exact six sanitized local
files required: five venue-native role snapshots and one Lighter event-time
native-limit pressure snapshot. Phase 5.1w may also emit an optional local
staging bundle: six empty `.jsonl` source files, a ready-to-edit
`local_capture_bundle_manifest.json`, and a field guide. The staged files are
templates only; Phase 5.1v must still validate supplied source rows before any
Phase 5.1s manifest is generated. Phase 5.1w does not capture source truth,
validate source rows, call venue APIs, read secrets, clear blockers, infer
maker/taker role, infer native-limit pressure, or authorize economics.

Phase 5.1x adds a HOLD-only Hyperliquid native-role adapter. It consumes an
already-local Hyperliquid `userFills` or `userFillsByTime` JSON/JSONL snapshot,
matches raw `oid`/`cloid` identity only against canonical redacted order hashes,
and emits `hyperliquid_forward_native_role_snapshot.jsonl` rows containing only
`venue_id`, `canonical_group_id`, `order_key`, boolean `crossed`, a source
record hash, and safety flags. The adapter performs no network calls, reads no
secrets or `.env` files, rejects URI paths, rejects secret-shaped fields,
strips raw identifiers from output, and never infers maker/taker role. Official
Hyperliquid API documentation for the `info` endpoint lists `userFills` and
`userFillsByTime` as user-address queries and includes `crossed` on fill
records; the docs also require querying the actual account or subaccount
address rather than an API-agent wallet for account data.

The first Phase 5.1x evidence run used a read-only public Hyperliquid account
address from local configuration to fetch `2000` `userFills` records to `/tmp`,
all with boolean `crossed`, then adapted that local file into redacted repo
evidence at
`runs/phase51x_hyperliquid_native_role_adapter/PHASE51X-HYPERLIQUID-USERFILLS-NATIVE-ROLE-20260504T000000Z`.
Phase 5.1x recovered `6 / 6` current Hyperliquid native-role target groups and
emitted `7` redacted `crossed` source rows. A downstream Phase 5.1v
Hyperliquid-only candidate run at
`runs/phase51v_forward_capture_bundle_readiness/PHASE51V-HYPERLIQUID-PARTIAL-SOURCE-HOLD-20260504T000000Z`
recognizes `6 / 287` all-five native-role targets ready, keeps
`281 / 287` native-role targets missing, keeps `0 / 3132` Lighter native-limit
targets ready, and remains `HOLD`. This is real blocker reduction for the
Hyperliquid subset only; it is not a source-complete all-five bundle and does
not authorize Phase 5.1s promotion, model training, EV admission, canary, live
orders, capital escalation, risk-limit relaxation, or financial claims.

Phase 5.1y adds a HOLD-only all-venue native-role adapter for already-local
JSON/JSONL source rows. It normalizes only explicit venue-native role fields:
Aster `ORDER_TRADE_UPDATE` / `o.m` with positive fill quantity, Extended
`isTaker` / `is_taker`, Hyperliquid `crossed`, Lighter `account_index` plus
`is_maker_ask` and side account IDs, and Paradex `liquidity`. Phase 5.1y
requires direct canonical group/order-key linkage in the source rows; rows
without target linkage are labeled HOLD and do not emit source truth. It rejects
network paths, `.env` files, symlink paths, unsafe flags, and secret-shaped
fields, strips raw order/trade identifiers from output, and never infers role
from strategy intent, post-only status, fees, or economics. The first staged
empty-source evidence run correctly recovered `0 / 287` native-role targets,
and a Hyperliquid-reuse evidence run recovered the existing `6 / 287`
Hyperliquid subset only. Phase 5.1y is an intake/normalization gate; it does
not capture venue truth by itself and does not authorize Phase 5.1s promotion,
model training, EV admission, canary, live orders, capital escalation,
risk-limit relaxation, or financial claims.

Phase 5.1z adds a HOLD-only read-only private-source capture/sanitizer for the
same venue-native role contract. It may fetch only read-only private source
surfaces using existing local credentials, or consume already-local JSON/JSONL
source rows. It maps raw/private venue rows to existing Phase 5.1u targets by
redacted hashes, emits only sanitized target-linked rows for Phase 5.1y/5.1v,
records credential presence as booleans only, rejects network paths for local
source inputs, rejects `.env` source files, rejects unsafe true flags and
secret-shaped fields, and does not persist raw order IDs, raw client IDs, raw
trade IDs, tokens, signatures, or private keys. It now emits redaction-safe
per-venue diagnostics: target count, source row count, native-field-ready
count, matched row count, duplicate matched row count, no-target-match count,
hash-candidate coverage, and target time windows. Its current read-only
evidence recovered `67 / 287` all-five native-role target groups (`aster=39`,
`extended=21`, `paradex=7`) and the combined Phase 5.1v run with existing
Hyperliquid source recognizes `73 / 287` native-role targets ready. The
diagnostic retry showed that target linkage, not native-field parsing, is the
main source blocker for retained rows. This is blocker reduction, not
promotion: `214 / 287` native-role targets remain missing and Lighter
event-time native-limit pressure remains `0 / 3132`.

Phase 5.1z may optionally emit sanitized unlinked native-role source rows behind
the explicit `--emit-unlinked-native-role-source-rows` flag. These rows contain
only venue ID, redacted `source_record_sha256`, and native role fields; they do
not contain canonical target keys and cannot mark Phase 5.1v targets ready by
themselves. The 2026-05-05 Lighter retained-backfill replay emitted `531`
sanitized unlinked Lighter source rows, `0` target-linked Lighter source rows,
and retained `HOLD` in Phase 5.1v without a validated redacted source-link
sidecar. This path preserves evidence that can later be joined safely, but it
does not authorize model training, EV admission, canary, live orders, capital
escalation, risk-limit relaxation, or financial claims.

Phase 5.1z also includes a HOLD-only source-link request-pack step for those
unlinked rows. The request pack emits the `531` redacted source hashes, the
`125` current Lighter native-role targets, an empty proposed sidecar placeholder,
and a Phase 5.1v candidate manifest for validation. The empty-sidecar validation
must remain `HOLD`; only a separately validated redacted sidecar that maps
`source_record_sha256` to `canonical_group_id` or `order_key` may reduce Lighter
native-role target misses. The request pack is not native truth by itself and is
not an authorization for model training, EV admission, canary, live orders,
capital escalation, risk-limit relaxation, or financial claims.

The source-link request-pack step is now venue-neutral when explicitly run with
`--venue-id all`. The stronger current-target wide read-only run preserved
`2819` sanitized unlinked source hashes (`aster=784`, `extended=1579`,
`lighter=300`, `paradex=156`) and packaged them against the `281` current Phase
5.1u native-role targets for venues with unlinked request sources (`aster=113`,
`extended=28`, `lighter=125`, `paradex=15`). Empty-sidecar validation remains
`HOLD` with `67 / 287` native-role targets ready and `0 / 3132` Lighter
native-limit targets ready. This supersedes the narrower `2130`-hash request
pack and converts the all-venue native-role linkage gap into a deterministic
redacted sidecar request, not a readiness claim.

Phase 5.1ac adds a HOLD-only source-link reuse audit. It scans existing local
`source_links.sanitized.jsonl` sidecars and compares their redacted source hashes
against the Phase 5.1z all-venue request pack. It performs no network access,
does not use raw identifiers, does not infer links, and cannot clear blockers by
itself. The current-target wide audit found `0 / 2819` reusable links across
`785` existing sidecar rows, leaving all request sources missing (`aster=784`,
`extended=1579`, `lighter=300`, `paradex=156`). This proves that the all-venue
sidecar cannot be deterministically derived from existing repo-owned sidecars;
a new validated redacted sidecar or directly target-linkable source remains
required.

Phase 5.1ad adds a HOLD-only source-link sidecar materializer. It consumes a
Phase 5.1z request pack plus an externally validated redacted mapping containing
only `source_record_sha256` and `canonical_group_id` or `order_key`, validates
the mapping against request-pack sources and Phase 5.1u targets, rejects raw
identifier fields, secrets, unsafe true flags, duplicate source hashes, unknown
source hashes, unknown target keys, and cross-venue joins, and emits a
Phase 5.1v-compatible `source_links.sanitized.jsonl` plus candidate manifest.
It does not infer missing links and does not itself clear blockers; it makes the
next redacted sidecar submission deterministic and repo-owned once a validated
mapping exists.

Phase 5.1ae adds a HOLD-only candidate-manifest composer. It combines
already-local Phase 5.1v candidate manifests plus explicit local source or
source-link artifacts into a single `candidate_manifest.composed.json`, with
fail-closed checks for network paths, env files, symlinks, secrets, raw
identifier fields, unsafe true flags, missing files, baseline mismatches, and
conflicting duplicate paths. Its purpose is to remove manual manifest stitching
after Phase 5.1ad materialization, while preserving the no-live/no-claim
boundary. The first real Phase 5.1ae evidence run combines the current-target
wide Aster/Extended/Lighter/Paradex request manifest with the existing
Phase 5.1x Hyperliquid source; Phase 5.1v remains `HOLD` with `73 / 287`
native-role targets ready and `0 / 3132` Lighter native-limit targets ready.

Phase 5.1af adds a HOLD-only local source retrieval audit. It scans the
current-target wide request pack, expected-hash bounded Phase 5 telemetry
artifacts, and explicit runtime log files to determine whether existing local
files can provide either a complete redacted Phase 5.1ad source-link mapping or
complete sanitized Phase 5.1ab Lighter event-time pressure rows without
inference. The reference run
`PHASE51AF-LOCAL-SOURCE-RETRIEVAL-AUDIT-HOLD-20260505T000000Z` found
`source_link_retrieval_status=MISSING_REQUIRED_LINKAGE`,
`lighter_pressure_retrieval_status=MISSING_REQUIRED_PRESSURE_FIELDS`, and
`runtime_log_pattern_status=NO_USABLE_PRESSURE_PATTERN`. It preserves `HOLD`:
existing local files do not supply the missing mapping or pressure rows, and
runtime `rate_limit` pattern counts are not evidence of event-time sendTx or
REST-or-weighted pressure.

The 2026-05-05 bounded GET-only Lighter target-window diagnostic was attempted
after hardening the Lighter trade-backfill redaction path. The regenerated
HOLD-only run captured `400` read-only trades, passed fail-closed raw
identifier/cursor key validation, and Phase 5.1z emitted `400` sanitized
unlinked Lighter native-role rows. It recovered `0 / 125` current Lighter
native-role targets, and Phase 5.1v remained `HOLD` with `0 / 287` native-role
targets ready and `0 / 3132` Lighter native-limit targets ready. This confirms
the GET-only diagnostic path is redaction-safe but not target-linkable for the
current Phase 5.1u Lighter targets. Do not repeat the same diagnostic unless
the target window or source surface changes.

The 2026-05-05 Phase 5.1aa Lighter WebSocket account-source diagnostic then
tested a different source surface: official `account_all` and
`account_all_trades` account snapshot channels. The collector is repo-owned,
HOLD-only, read-only, writes message metadata instead of private account
payloads, source-redacts trade rows, and fails closed on raw identifier-like
output keys. The authorized local-credential captures reached both channels,
but each snapshot contained `0` trade rows. The downstream Phase 5.1z/5.1v run
therefore remained `HOLD` with `0 / 287` native-role targets ready and
`0 / 3132` Lighter native-limit targets ready. This confirms the WebSocket
account-source surface is safe and runnable, but it does not reduce the current
Lighter blocker unless future account activity or source semantics produce
target-linkable trade rows.

Phase 5.1ab adds a HOLD-only repo-owned preflight gate for externally supplied
sanitized Lighter native-limit pressure rows. The gate accepts only local
`.jsonl` inputs, performs no network access, does not call `sendTx`,
`sendTxBatch`, `nextNonce`, or any venue write path, rejects raw identifiers,
secret-shaped fields, unsafe true flags, network/env/symlink inputs, and emits a
Phase 5.1v candidate manifest plus `lighter_forward_native_limit_pressure_snapshot.jsonl`.
It requires explicit event-time active-order headroom, sendTx limit/remaining,
REST-or-weighted limit/remaining, and native-limit event-time status. Phase 5.1ab
does not observe pressure by itself, does not infer pressure from docs/account
tiers/GET caps/empty headers, and does not clear Phase 5.1 blockers. Its purpose
is to make any future non-live-authorized sanitized pressure rows immediately
replayable and reviewable by Phase 5.1v without changing the no-live boundary.

The 2026-05-04 authorized read-only private source attempt produced Lighter-only
sanitized account/native-limit and trade-backfill artifacts; it does not change
this spec gate because Phase 5.1w requires all-five native role files and
event-time Lighter native-limit pressure. Phase 5.1n now emits a downstream
5.1v manifest and `lighter_forward_native_limit_pressure_snapshot.jsonl` only
for Lighter rows where event-time active-order headroom plus sendTx and
REST-or-weighted limit/remaining pressure are all observed. The retest
`PHASE51N-LIGHTER-NATIVE-LIMIT-FORWARD-SOURCE-RETEST-20260504T000000Z`
emitted `0` complete forward rows, so it preserves `HOLD` and does not clear
any of the `3132` Lighter native-limit targets. The canonical staged-source evidence
run `PHASE51W-LOCAL-STAGED-SOURCE-BUNDLE-CANONICAL-20260504T000000Z` proves the
local bundle contract without clearing the blocker: Phase 5.1v sees six local
files but remains `HOLD` with `0 / 287` native-role targets and `0 / 3132`
Lighter native-limit targets ready. The next evidence move is a deterministic
redacted Lighter source-link sidecar, a different target-linkable Lighter
source, and a separate event-time Lighter request-pressure source. For all
venues, populate sanitized local read-only all-five forward source-capture
files with real native rows, run the bundle through Phase 5.1v, and only if
5.1v emits `generated_phase51s_manifest_ready=true`, stage the generated
manifest through Phase 5.1s. The bundle must contain native snapshots with
canonical group/order-key linkage, or validated source-link sidecars that bind
redacted source hashes to observed labels. Phase 5.1v now applies those
sidecars during bundle-readiness validation, but only source rows with required
native fields can mark targets ready. After Phase 5.1s, run Phase 5.1r, feed
the sanitized outputs into Phase 5.1q, and rerun Phase 5.1n/5.1h/5.1i. No
model training, EV admission, canary, live orders, capital escalation,
risk-limit relaxation, or financial claim is authorized from Phase 5.1s, Phase
5.1r, Phase 5.1q, Phase 5.1t, Phase 5.1u, Phase 5.1v, Phase 5.1w, Phase 5.1x,
Phase 5.1y, Phase 5.1z, Phase 5.1ad, or Phase 5.1ae.
