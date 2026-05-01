# World-Class Runtime Goals

Scope: runtime objective framework.

This document defines quality bars and promotion gates. It is not the current
topology source of truth. For current Phase 5 state, read `phase5/status.md`
and `phase5/queue.yaml` as read-only context.

## Purpose

This document replaces vague runtime phase language with explicit goals,
qualification gates, and exit criteria for the live multi-venue system.

The standard is not "good enough to keep running."
The standard is:

- connector behavior is trustworthy,
- telemetry is trustworthy,
- venue roles are earned by evidence,
- promotions are reproducible,
- economics are judged only on clean surfaces.

## Core Principle

Do not ask:

- "Are we on Phase 4 or Phase 5?"

Ask:

- "Which exact runtime objective is still incomplete?"
- "What is the measurable blocker?"
- "What binary gate must pass before we spend more live risk?"

## World-Class Definition

Paraphina is world-class only when all of the following are true:

1. Platform fidelity is high.
2. Venue attribution is correct.
3. Every promoted venue has qualified for its role.
4. The promoted live surface is operationally clean over long soaks.
5. Economics are optimized on a clean surface, not on a contaminated one.
6. Rollback and recovery are automatic, predictable, and clean.
7. Promotion decisions are evidence-backed and reproducible.

## Non-Negotiables

1. No live-affecting mixed hypotheses.
2. No long live soak unless the short gate already proved the intended mechanism.
3. No economic conclusions from known-bad telemetry.
4. No venue promoted because "it exists"; each venue earns a role.
5. No connector excluded permanently until we separate exchange limitations from our own implementation defects.

## Runtime Objective Stack

### Objective 1: Platform Fidelity

The system must represent venue state consistently and correctly before any
economic conclusion is trusted.

This objective is complete only when all venues share the same quality bar for:

- market-data freshness semantics
- snapshot/delta correctness
- queue/drop policy
- ping/heartbeat behavior
- reconnect behavior
- stale detection
- private account/order truth
- telemetry venue mapping

#### Qualification gates

1. Freshness semantics are normalized.
   - `age_ms` and related fields are comparable across venues.
   - Local receive/apply timestamps and exchange timestamps are clearly separated when needed.

2. Feed semantics are explicit per venue.
   - We know whether each venue is using BBO, shallow L2, or full L2.
   - We do not confuse weak feed products with exchange weakness.

3. Telemetry attribution is correct.
   - Active venue set in telemetry must match the actual connector set.
   - No noncanonical venue-order mislabeling.

4. Recovery architecture is uniform enough to compare venues fairly.
   - stale handling
   - reconnect behavior
   - health enforcement
   - backstop polling where required

#### Failure examples

- A venue looks stale because its freshness timestamp source differs from other venues.
- A venue looks economically weak because we only subscribed to depth `1`.
- A report shows `extended` in a `4v` run because the analyzer assumed a canonical `5`-venue order.

### Objective 2: Venue Qualification

Each venue must qualify for one or more roles:

- `fill`
- `anchor`
- `connected_non_fv`
- `excluded_pending_rescue`

No venue is assumed to be a fill venue. No venue is assumed to deserve FV weight.

#### Venue role definitions

`fill`
- contributes real fills with acceptable reject/churn profile
- can be trusted to participate in execution

`anchor`
- improves price discovery / stability
- does not materially degrade primary fill venues
- may contribute to FV without being a major execution venue

`connected_non_fv`
- connected for optional utility or future rescue
- not allowed to move global fair value

`excluded_pending_rescue`
- currently not safe or useful enough for the promoted surface
- must be worked in an isolated rescue branch

#### Qualification ladder per venue

1. Local / unit correctness
2. Shadow attribution
3. `5m` live correctness
4. `20m` live soak
5. `60m+` qualification soak

#### Venue qualification gates

Operational:

- no kill event
- no restart loop
- no unresolved reconcile drift
- no dirty rollback residue

Connector correctness:

- low reject rate from connector/path bugs
- stable account/order truth
- no stale/feed pathologies dominating the run

Economic usefulness:

- either contributes fills
- or measurably improves the promoted surface without hurting fill venues

### Objective 3: Promoted Surface Qualification

The promoted live surface must be safe as a system, not just as a set of
individually tolerable venues.

This objective is complete only when the promoted surface passes:

1. `5m` correctness canary
2. `20m` soak
3. `60m` exit soak
4. clean rollback on first pass

#### Surface gates

Operational:

- `healthy=true`
- `ready=true`
- `NRestarts=0`
- `kill_events_present=false`
- `reconcile_mismatch_count=0`

Recovery:

- clean post-run venue audit
- no stranded inventory/open orders after restore

Interpretability:

- telemetry/reporting must be trustworthy for the active connector set

### Objective 4: Economics Optimization

Only after Objectives 1-3 are sufficiently complete do we optimize economics.

Economics work includes:

- FV venue participation
- FV weighting
- stale/outlier/quality gating
- fill conversion
- churn reduction
- queue competitiveness
- longer soak markout / PnL quality

This objective is explicitly downstream of platform fidelity and surface qualification.

#### Economics gates

1. A candidate must not degrade the current promoted baseline on a short canary
   unless the upside hypothesis requires a longer read and is operationally safe.
2. Fair-value changes are judged against:
   - quote survival
   - would-send activity
   - fills
   - markout / PnL
3. No economics promotion on contaminated attribution.

### Objective 5: Autonomous Operational Excellence

The system should not require ad hoc heroics.

This objective is complete only when:

- canary assembly is standardized
- artifact extraction is reliable
- reports are reproducible
- rollback is automatic
- promotion decisions are evidence-packed

## What "All 5 Venues" Actually Means

The right target is not:

- all `5` venues used identically for every aspect of market making

The right target is:

- all `5` venues rehabilitated if possible
- each venue assigned the role it actually earns

Homogeneous all-`5` topology is allowed only if evidence supports it.
It is not the default success condition.

That means:

- if a venue is excluded because of our connector design, that is our problem to fix
- if a venue still does not qualify after a fair rescue effort, forcing it into the production surface is not world-class

## Current Position

### Current-state source

This document no longer embeds the promoted venue surface. Earlier revisions
described a four-venue surface with Lighter FV-disabled and Extended excluded;
that language is historical and can conflict with Phase 5 state.

Use read-only `phase5/status.md` and `phase5/queue.yaml` for the current
topology, then apply the objective gates in this document to decide what is
still incomplete.

### What is not yet complete

Objective 1 is not fully complete across all venues.

Why:

- venue WS/feed semantics are still asymmetric
- feed depth is not yet normalized
- freshness/recovery quality is still connector-specific
- `extended` remains the clearest unfinished venue rescue branch

Objective 4 has begun but should remain subordinate to Objective 1 cleanup where
connector asymmetry materially contaminates conclusions.

## Required KPI Set

Every runtime branch should score at least these:

### Platform KPIs

- venue age p50 / p95 / p99
- stale-run count and max stale-run length
- reconnect count and reason
- queue overflow / dropped event counters
- feed depth / product class per venue

### Execution KPIs

- place / replace / cancel intents
- place / cancel acks
- rejects by reason
- fills by venue
- fill base size by venue
- would-send nonzero ticks
- active quote rows

### Safety KPIs

- kill events
- reconcile mismatches
- dirty rollback residue
- restart count

### Economics KPIs

- final PnL
- markout / toxicity
- fill conversion
- FV-minus-mid displacement
- churn / replace burden

## Operating Model

### Allowed parallelism

Parallel:

- research
- local connector fixes
- tooling
- shadow-only diagnostics

Serialized:

- live-affecting runtime promotions

### Branch discipline

Every branch must define:

1. exact hypothesis
2. exact changed axis
3. exact baseline
4. exact short gate
5. exact longer gate
6. exact promote/hold/rollback rule

## Immediate Priorities

The next priorities should be:

1. Finish Objective 1 platform-fidelity cleanup across venue connectors.
   - especially WS/feed/freshness/recovery parity
   - especially the `extended` rescue branch

2. Continue Objective 4 on the promoted `4v` surface only when attribution is clean.
   - current near-term branch: Aster vs Paradex contribution attribution
   - Lighter role remains a controlled economics branch, not a promotion assumption

3. Do not declare victory based on vague phase naming.
   - declare completion only at objective-gate level

## Final Standard

A world-class result is:

- clean and comparable connector behavior across venues
- explicit and evidence-backed venue roles
- a promoted surface that survives long soaks without operational residue
- economics optimized on top of that clean foundation
- all venues rehabilitated where possible, and excluded only when evidence says they remain net harmful after a fair rescue attempt
